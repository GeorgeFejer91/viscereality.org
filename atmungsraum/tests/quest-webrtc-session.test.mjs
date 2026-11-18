import assert from 'node:assert/strict';
import { createHmac } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import test from 'node:test';
import { joinQuest } from '../quest-gateway.js';
import { createQuestRtcTransport } from '../quest-webrtc-session.js';
import { commissioningBroker, recoverCommissionedEnrollment } from '../quest-commissioning.js';
import { openQuestWebRtc } from '../quest-webrtc.js';
import { VdoNinjaTransport } from '../vendor/quest-webrtc/vdo-ninja-transport.js';

const owner = { device: '2'.repeat(32), epoch: '42', native_epoch: '7', serial: '', identity_kind: 'app-installation' };
const enrollment = { controller: '1'.repeat(32), key: '3'.repeat(64) };
const room = 'a'.repeat(32);
const build = 'e'.repeat(64);

test('WebRTC keeps negotiating a selected peer and falls back quickly while still discovering', async t => {
  const deadlines = new Map();
  let carrier;
  let stopped = 0;
  t.mock.method(AbortSignal, 'timeout', duration => {
    const clock = new AbortController();
    deadlines.set(duration, clock);
    return clock.signal;
  });
  t.mock.method(VdoNinjaTransport.prototype, 'start', async function () { carrier = this; });
  t.mock.method(VdoNinjaTransport.prototype, 'stop', async () => { stopped++; });
  const discovery = { owner, rtc: { r: room, p: 'b'.repeat(64) }, endpoints: ['http://192.168.1.2:8766'] };
  const signal = new AbortController().signal;
  const timeout = new DOMException('deadline reached', 'TimeoutError');
  const abortWithoutPeer = assert.rejects(openQuestWebRtc(discovery, enrollment, signal), timeout);
  deadlines.get(5000).abort(timeout);
  await abortWithoutPeer;

  const pendingPeer = openQuestWebRtc(discovery, enrollment, signal);
  carrier.dispatchEvent(new CustomEvent('status', { detail: { phase: 'connecting-peer' } }));
  deadlines.get(5000).abort(timeout);
  carrier.dispatchEvent(new CustomEvent('peeropen', { detail: { peerKey: 'test-peer' } }));
  const { transport } = await pendingPeer;
  assert.deepEqual(await (await transport.request(undefined, signal)).json(),
    { ...owner, v: 2, enabled: true, remote_available: true });
  transport.close();

  deadlines.clear();
  const abortWithoutEnrollment = assert.rejects(openQuestWebRtc(discovery, null, signal), timeout);
  assert.equal(deadlines.has(5000), false);
  deadlines.get(15000).abort(timeout);
  await abortWithoutEnrollment;
  assert.equal(stopped, 3);
});

test('the WebRTC transport exposes discovery identity and passes opaque, room-scoped frames', async () => {
  const requests = [];
  let closed = 0;
  const signal = new AbortController();
  const channel = {
    request(frame) {
      requests.push(frame);
      return Promise.resolve(frame);
    },
    close() { closed++; },
  };
  const bridge = await createQuestRtcTransport({ owner, enrollment, room, channel });
  const identity = await (await bridge.request(undefined, signal.signal)).json();
  assert.deepEqual(identity, { ...owner, v: 2, enabled: true, remote_available: true });
  const body = { v: 1, nonce: '4'.repeat(32), controller: enrollment.controller, action: 'receipt',
    payload: '{"id":"0123456789abcdef"}', mac: '9'.repeat(64), epoch: owner.epoch };
  await assert.rejects(bridge.request(body, signal.signal), 'a reflected request cannot be decoded as a reply');
  assert.deepEqual(Object.keys(requests[0]).sort(), ['data','id','iv','tag','v']);
  assert.equal(requests[0].id, body.nonce);
  assert.ok(!JSON.stringify(requests[0]).includes(enrollment.controller));
  assert.ok(!JSON.stringify(requests[0]).includes('0123456789abcdef'));
  const another = await createQuestRtcTransport({ owner, enrollment, room: 'b'.repeat(32), channel });
  await assert.rejects(another.request(body, signal.signal));
  assert.notEqual(requests[0].tag, requests[1].tag, 'public membership tag changes across rooms');
  await assert.rejects(bridge.request({ ...body, payload: 'x'.repeat(8192) }, signal.signal));
  assert.equal(requests.length, 2, 'oversized plaintext stays out of data channel');
  bridge.close();
  bridge.close();
  assert.equal(closed, 1);
  await assert.rejects(bridge.request(body, signal.signal), { name: 'AbortError' });
  assert.equal(requests.length, 2);
});

test('WebCrypto transports exact native receipts and large recording replies', {
  skip: !process.env.RQ_RTC_JAVA_CLASSPATH && 'Set the classpath of the compiled native gateway test',
}, async t => {
  const java = spawn(process.env.RQ_PAIRING_JAVA || 'java', ['-cp', process.env.RQ_RTC_JAVA_CLASSPATH,
    'io.github.mesmerprism.rustyquest.native_renderer.ExperimentSessionGatewayProtocolTest', '--serve-webrtc', room], {
    stdio: ['pipe', 'pipe', 'inherit'], windowsHide: true,
  });
  const output = createInterface({ input: java.stdout });
  const pending = [];
  output.on('line', line => pending.shift()?.resolve(JSON.parse(line)));
  java.on('error', error => pending.splice(0).forEach(reply => reply.reject(error)));
  java.on('exit', code => pending.splice(0).forEach(reply => reply.reject(Error(`Java peer quit (${code})`))));
  t.after(() => { java.stdin.end(); java.kill(); output.close(); });
  const request = message => new Promise((resolve, reject) => {
    pending.push({ resolve, reject });
    java.stdin.write(JSON.stringify(message) + '\n');
  });
  const frames = [];
  let receivedBytes = 0;
  const channel = {
    async request(frame) {
      frames.push(structuredClone(frame));
      const response = await request({ op: 'frame', frame });
      assert.equal(response.ok, true);
      receivedBytes = response.result.data.length;
      return response.result;
    },
    close() {},
  };
  const signal = new AbortController().signal;
  const bootstrap = createHmac('sha256', enrollment.key)
    .update(`ATMB1|credential|${owner.device}|${enrollment.controller}`).digest('hex');
  const mac = text => createHmac('sha256', bootstrap).update(text).digest('hex');
  const binding = `ATMB1|RQG1-RTC1|${owner.device}|${owner.epoch}|${owner.native_epoch}|${room}`;
  const transcript = frame => `${binding}|request|${frame.id}|${frame.tag}|${frame.public_key}`;
  const broker = {
    async authorize({ id, public_key }) {
      const frame = { v: 2, id, public_key, tag: mac(binding + '|member') };
      return { ...frame, proof: mac(transcript(frame)) };
    },
    async verify(frame, reply) {
      return reply.proof === mac(`${transcript(frame)}|reply|${reply.public_key}|${reply.iv}|${reply.data}`);
    },
  };
  const recovered = await recoverCommissionedEnrollment({ discovery: { owner, rtc: { r: room } }, channel, broker, signal });
  assert.deepEqual(recovered, enrollment, 'a blank browser recovered the registered member without displaying a SAS');
  assert.equal((await request({ op: 'stats' })).result, 0, 'commissioning dispatched no experiment action');
  assert.equal((await request({ op: 'frame', frame: frames[0] })).ok, false, 'commissioning replay was accepted');
  const forged = { ...broker, verify: async () => false };
  await assert.rejects(recoverCommissionedEnrollment({ discovery: { owner, rtc: { r: room } }, channel, broker: forged, signal }));

  const transport = await createQuestRtcTransport({ owner, enrollment: recovered, room, channel });
  const quest = await joinQuest({ ...owner, enrollment, transport, build });
  assert.equal(quest.initial.p, 'IDLE');
  const id = '0123456789abcdef';
  assert.deepEqual(await quest.receipt(id), { id, state: 'confirmed' });
  assert.equal((await request({ op: 'frame', frame: frames[3] })).ok, false, 'replay reached native owner');
  const exportReply = await quest.recordings({ id }, new AbortController().signal);
  assert.equal(exportReply.synthetic, 'abc'.repeat(8000));
  assert.ok(receivedBytes > 32768, 'covers data-channel fragmentation and bounded reply decoding');
  assert.equal((await request({ op: 'stats' })).result, 3, 'status, receipt, recording each dispatched once');
  await request({ op: 'retire' });
  await assert.rejects(quest.receipt(id));
  assert.equal((await request({ op: 'stats' })).result, 3, 'retired owner accepted no additional dispatch');
  quest.close();
});

const commissioningDiscovery = { owner, rtc: { r: room, p: 'b'.repeat(64) }, endpoints: [] };
const commissioningOffer = { device: owner.device, epoch: owner.epoch, native_epoch: owner.native_epoch,
  room, id: '4'.repeat(32), public_key: 'A'.repeat(122) };
const readyDiscovery = (rtc = commissioningDiscovery.rtc) => Response.json({
  v: 1, state: 'ready', metadata: { ...owner, v: 2, mode: 'remote', pairing: 1, gateways: [], rtc },
});

test('commissioning refreshes a rejected discovery snapshot and reuses the exact offer', async t => {
  const requests = [];
  const authorized = { v: 2, proof: 'fixture-proof' };
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push({ path: new URL(url).pathname, body: options.body && JSON.parse(options.body) });
    switch (requests.length) {
      case 1: return Response.json({}, { status: 403 });
      case 2: return Response.json({ v: 1, state: 'searching' });
      case 3: return readyDiscovery();
      case 4: return Response.json(authorized);
      default: assert.fail('unexpected commissioning request');
    }
  });
  const result = await commissioningBroker.authorize(commissioningOffer, AbortSignal.timeout(2000), commissioningDiscovery);
  assert.deepEqual(result, authorized);
  const proofPath = '/v1/commissioning/request';
  const discoveryPath = '/v1/quest/' + owner.device;
  assert.deepEqual(requests, [
    { path: proofPath, body: commissioningOffer },
    { path: discoveryPath, body: undefined },
    { path: discoveryPath, body: undefined },
    { path: proofPath, body: commissioningOffer },
  ]);
});

test('a changed rendezvous leaves the commissioning request denied', async t => {
  const requests = [];
  t.mock.method(globalThis, 'fetch', async url => {
    requests.push(new URL(url).pathname);
    if (requests.length === 1) return Response.json({}, { status: 403 });
    if (requests.length === 2) return readyDiscovery({ ...commissioningDiscovery.rtc, p: 'c'.repeat(64) });
    assert.fail('the changed rendezvous received another proof request');
  });
  await assert.rejects(commissioningBroker.authorize(commissioningOffer, AbortSignal.timeout(2000), commissioningDiscovery),
    { status: 403 });
  assert.deepEqual(requests, ['/v1/commissioning/request', '/v1/quest/' + owner.device]);
});

test('commissioning stops polling when the shared deadline aborts', async t => {
  const controller = new AbortController();
  const reason = new DOMException('fixture canceled', 'AbortError');
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    if (requests === 1) return Response.json({}, { status: 403 });
    controller.abort(reason);
    throw reason;
  });
  await assert.rejects(commissioningBroker.authorize(commissioningOffer, controller.signal, commissioningDiscovery), reason);
  assert.equal(requests, 2);
});
