import test from 'node:test';
import assert from 'node:assert/strict';
import { rediscoverQuest } from '../quest-gateway.js';

const device = 'a'.repeat(32);
const serial = 'QUEST-TEST';
const signal = new AbortController().signal;
const settings = { device, serial };
const identity = { v: 1, device, serial, epoch: '1', native_epoch: '2', mode: 'remote',
  gateways: ['192.168.10.52'] };

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
