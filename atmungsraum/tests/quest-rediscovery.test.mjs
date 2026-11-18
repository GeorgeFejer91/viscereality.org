import test from 'node:test';
import assert from 'node:assert/strict';
import { companionAccessDenied, rediscoverQuest } from '../quest-gateway.js';

const device = 'a'.repeat(32);
const serial = 'QUEST-TEST';
const signal = new AbortController().signal;
const settings = { device, serial };
const identity = { v: 1, device, serial, epoch: '1', native_epoch: '2', mode: 'remote',
  gateways: ['192.168.10.52'] };

test('only a denied loopback permission identifies a blocked companion', async () => {
  for (const state of ['granted', 'prompt', 'denied']) {
    assert.equal(await companionAccessDenied({
      async query(descriptor) {
        assert.deepEqual(descriptor, { name: 'loopback-network' });
        return { state };
      },
    }), state === 'denied');
  }
  assert.equal(await companionAccessDenied({ query() { throw new TypeError('Unsupported permission'); } }), false);
});

test('native discovery reports a disabled radio and recovers the remembered Quest on retry', async () => {
  const states = [
    { state: 'bluetooth-off' },
    { state: 'searching' },
    { state: 'ready', metadata: identity }
  ];
  const fetcher = async (url, init) => {
    assert.equal(String(url), 'http://127.0.0.1:8767/v1/quest/' + device);
    assert.equal(init.redirect, 'error');
    assert.equal(init.credentials, 'omit');
    return Response.json({ v: 1, ...states.shift() });
  };
  assert.equal((await rediscoverQuest(settings, signal, fetcher)).state, 'bluetooth-off');
  assert.equal((await rediscoverQuest(settings, signal, fetcher)).state, 'searching');
  const found = await rediscoverQuest(settings, signal, fetcher);
  assert.equal(found.owner.device, device);
  assert.deepEqual(found.endpoints, ['http://192.168.10.52:8766/v1/gateway']);
});

test('rediscovery rejects a different headset and a public address', async () => {
  for (const metadata of [{ ...identity, device: 'b'.repeat(32) }, { ...identity, gateways: ['8.8.8.8'] }]) {
    await assert.rejects(rediscoverQuest(settings, signal, async () =>
      Response.json({ v: 1, state: 'ready', metadata })));
  }
});

test('an enrolled Quest can advertise the WebRTC room through companion discovery without a LAN address', async () => {
  const rtc = { r: 'c'.repeat(32), p: 'd'.repeat(64) };
  const actor = { device, serial: '', identity_kind: 'app-installation' };
  const metadata = { ...identity, ...actor, v: 2, pairing: 1, gateways: [], rtc };
  const reply = details => async () => Response.json({ v: 1, state: 'ready', metadata: details });
  const found = await rediscoverQuest(actor, signal, reply(metadata));
  assert.deepEqual(found.rtc, rtc);
  assert.deepEqual(found.endpoints, []);
  const hardwareActor = await rediscoverQuest({ ...actor, commissioned: true }, signal,
    reply({ ...metadata, serial: 'HARDWARE-TEST' }));
  assert.equal(hardwareActor.owner.serial, 'HARDWARE-TEST');
  await assert.rejects(rediscoverQuest({ ...actor, commissioned: true }, signal,
    reply({ ...metadata, device: 'e'.repeat(32), serial: 'HARDWARE-TEST' })));
  await assert.rejects(rediscoverQuest(actor, signal, reply({ ...metadata, rtc: { ...rtc, p: 'too-short' } })));
  await assert.rejects(rediscoverQuest(actor, signal, reply({ ...metadata, rtc: { ...rtc, host: 'custom.example' } })));
});
