import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import test from 'node:test';
import { joinQuest } from '../quest-gateway.js';
import { createQuestRtcTransport } from '../quest-webrtc-session.js';

const owner = { device: '2'.repeat(32), epoch: '42', native_epoch: '7', serial: '', identity_kind: 'app-installation' };
const enrollment = { controller: '1'.repeat(32), key: '3'.repeat(64) };
const room = 'a'.repeat(32);
const build = 'e'.repeat(64);

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
  const transport = await createQuestRtcTransport({ owner, enrollment, room, channel });
  const quest = await joinQuest({ ...owner, enrollment, transport, build });
  assert.equal(quest.initial.p, 'IDLE');
  const id = '0123456789abcdef';
  assert.deepEqual(await quest.receipt(id), { id, state: 'confirmed' });
  assert.equal((await request({ op: 'frame', frame: frames[1] })).ok, false, 'replay reached native owner');
  const exportReply = await quest.recordings({ id }, new AbortController().signal);
  assert.equal(exportReply.synthetic, 'abc'.repeat(8000));
  assert.ok(receivedBytes > 32768, 'covers data-channel fragmentation and bounded reply decoding');
  assert.equal((await request({ op: 'stats' })).result, 3, 'status, receipt, recording each dispatched once');
  await request({ op: 'retire' });
  await assert.rejects(quest.receipt(id));
  assert.equal((await request({ op: 'stats' })).result, 3, 'retired owner accepted no additional dispatch');
  quest.close();
});
