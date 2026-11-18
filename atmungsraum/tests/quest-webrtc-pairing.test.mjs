import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHmac, randomBytes } from 'node:crypto';
import { createInterface } from 'node:readline';
import test from 'node:test';
import { openQuestWebRtc } from '../quest-webrtc.js';
import { bindQuestRtcChannel } from '../quest-webrtc-channel.js';
import { VdoNinjaTransport } from '../vendor/quest-webrtc/vdo-ninja-transport.js';

const room = 'a'.repeat(32);
const options = { skip: !process.env.RQ_WEBRTC_PAIRING_CLASSPATH && 'Compile the actual WebRTC gateway pairing host' };
const emit = (target, type, detail) => target.dispatchEvent(new CustomEvent(type, { detail }));

async function nativePeer(t) {
  const child = spawn(process.env.RQ_PAIRING_JAVA || 'java', ['-cp', process.env.RQ_WEBRTC_PAIRING_CLASSPATH,
    'io.github.mesmerprism.rustyquest.native_renderer.WebRtcPairingHarness', '--serve'],
  { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });
  const pending = [];
  const lines = createInterface({ input: child.stdout });
  lines.on('line', text => pending.shift()?.resolve(JSON.parse(text)));
  child.on('error', error => pending.splice(0).forEach(request => request.reject(error)));
  child.on('exit', code => pending.splice(0).forEach(request => request.reject(Error(`Native pairing host exited ${code}`))));
  t.after(() => { child.stdin.end(); child.kill(); lines.close(); });
  const raw = value => new Promise((resolve, reject) => {
    pending.push({ resolve, reject }); child.stdin.write(JSON.stringify(value) + '\n');
  });
  const call = async value => {
    const response = await raw(value);
    assert.equal(response.ok, true, response.error);
    return response.result;
  };
  return { call, owner: await call({ op: 'identity' }) };
}

async function wiredNative(t, beforeFrame) {
    const native = await nativePeer(t);
    const target = new EventTarget();
    const frames = [];
    let controller;
    let closed = 0;
    t.mock.method(globalThis, 'fetch', () => assert.fail('fresh Bluetooth-selected pairing must not use LAN or managed commissioning'));
    t.mock.method(VdoNinjaTransport.prototype, 'start', async function () {
      controller = this;
      queueMicrotask(() => emit(this, 'peeropen', { peerKey: 'quest' }));
    });
    t.mock.method(VdoNinjaTransport.prototype, 'stop', async () => { closed++; });
    t.mock.method(VdoNinjaTransport.prototype, 'sendControl', (_peer, data) => {
      queueMicrotask(() => emit(target, 'controlmessage', { peerKey: 'phone', data })); return true;
    });
    target.sendControl = (_peer, data) => {
      queueMicrotask(() => emit(controller, 'controlmessage', { peerKey: 'quest', data })); return true;
    };
    target.closePeer = () => emit(controller, 'peerclose', { peerKey: 'quest' });
    const exchange = async frame => {
      frames.push(frame);
      await beforeFrame?.(frame, native);
      return native.call({ op: 'frame', frame, room });
    };
    const targetChannel = bindQuestRtcChannel(target, 'phone', exchange);
    t.after(() => targetChannel.close());
    const signal = new AbortController().signal;
    return { native, frames, signal, get closed() { return closed; },
      discovery: { owner: native.owner, rtc: { r: room, p: 'b'.repeat(64) }, endpoints: [] } };
}

async function authenticatedStatus(transport, owner, { controller, key }, signal) {
  const nonce = randomBytes(16).toString('hex');
  const mac = text => createHmac('sha256', key).update(text).digest('hex');
  const payload = '{}';
  const reply = await (await transport.request({
    v: 1, controller, epoch: owner.epoch, nonce, action: 'status', payload,
    mac: mac(`RQG1|${controller}|${owner.epoch}|${nonce}|status|${payload}`),
  }, signal)).json();
  assert.equal(reply.mac, mac(`RQG1-REPLY|${controller}|${owner.device}|${owner.epoch}|${nonce}|status|${reply.payload}`));
  assert.deepEqual(JSON.parse(reply.payload), { phase: 'idle', action: 'status' });
}

for (const order of ['phone-first', 'headset-first']) {
  test(`fresh RTC-only selection uses two buttons and saves the native ticket (${order})`, options, async t => {
    const fixture = await wiredNative(t, async (frame, native) => {
      if (frame.request?.action === 'pair-status' && order === 'phone-first') {
        const prompt = await native.call({ op: 'prompt' });
        if (prompt?.phone_confirmed) await native.call({ op: 'decide', id: prompt.pairing_id, allow: true });
      }
    });
    const { native, frames, signal } = fixture;
    let saved;
    let prompts = 0;
    const session = await openQuestWebRtc(fixture.discovery,
      undefined, signal, {
        async onCode({ code, pairingId }) {
          prompts++;
          const prompt = await native.call({ op: 'prompt' });
          assert.equal(prompt.pairing_id, pairingId);
          assert.equal(prompt.code, code);
          assert.equal((await native.call({ op: 'stats' })).stores, 0);
          if (order === 'headset-first') await native.call({ op: 'decide', id: pairingId, allow: true });
          return true;
        },
        onEnrollment(value) { saved = value; },
      });
    assert.equal(prompts, 1);
    assert.equal((await native.call({ op: 'stats' })).stores, 1);
    assert.deepEqual(saved, { controller: '3'.repeat(32), key: '4'.repeat(64) });
    assert.deepEqual(session.enrollment, saved);
    assert.ok(frames.some(frame => frame.v === 3 && frame.request?.action === 'pair-reveal'));
    await authenticatedStatus(session.transport, native.owner, saved, signal);
    session.transport.close();
    await new Promise(setImmediate);
    assert.ok(fixture.closed > 0);
    const previousFrames = frames.length;
    const reconnected = await openQuestWebRtc(fixture.discovery, saved, signal);
    await authenticatedStatus(reconnected.transport, native.owner, saved, signal);
    assert.deepEqual(frames.slice(previousFrames).map(frame => frame.v), [1]);
    assert.equal((await native.call({ op: 'stats' })).stores, 1);
    reconnected.transport.close();
  });
}

test('an aborted enrollment callback closes RTC even when the native ticket committed', options, async t => {
  const fixture = await wiredNative(t);
  const attempt = new AbortController();
  await assert.rejects(openQuestWebRtc(fixture.discovery, undefined, attempt.signal, {
    async onCode({ pairingId }) {
      await fixture.native.call({ op: 'decide', id: pairingId, allow: true });
      return true;
    },
    async onEnrollment() { attempt.abort(); },
  }), error => error.name === 'AbortError' && error.pairingAttempted === true);
  assert.equal((await fixture.native.call({ op: 'stats' })).stores, 1);
  assert.ok(fixture.closed > 0);
});

for (const outcome of ['headset-rejected', 'expired', 'frozen-callback-error']) {
  test(`RTC pairing preserves the terminal ${outcome} outcome without a second enrollment`, options, async t => {
    const fixture = await wiredNative(t);
    let saved = 0;
    await assert.rejects(openQuestWebRtc(fixture.discovery, undefined, fixture.signal, {
      async onCode({ pairingId }) {
        if (outcome === 'headset-rejected') await fixture.native.call({ op: 'decide', id: pairingId, allow: false });
        else if (outcome === 'expired') await fixture.native.call({ op: 'expire' });
        else throw Object.freeze(Error('User cancelled this approval'));
        return true;
      },
      onEnrollment() { saved++; },
    }), error => error.pairingAttempted === true);
    assert.equal(saved, 0);
    assert.equal((await fixture.native.call({ op: 'stats' })).stores, 0);
    assert.equal(fixture.frames.filter(frame => frame.request?.action === 'pair-begin').length, 1);
    assert.ok(fixture.closed > 0);
  });
}
