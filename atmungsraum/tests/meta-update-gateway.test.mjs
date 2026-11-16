import test from 'node:test';
import assert from 'node:assert/strict';
import { createHmac } from 'node:crypto';
import { joinQuest } from '../quest-gateway.js';

const owner = { device: 'a'.repeat(32), serial: 'TEST', epoch: '7', native_epoch: '8' };
const controller = 'b'.repeat(32), key = 'c'.repeat(64), build = 'd'.repeat(64);
const mac = text => createHmac('sha256', key).update(text).digest('hex');

async function peer({ version = 2, control = 1, waiting = false } = {}) {
  const calls = [];
  const fetcher = async (_url, init) => {
    if (!init.body) return Response.json({ ...owner, v: 1, enabled: true, remote_available: true });
    const request = JSON.parse(init.body), { action, nonce } = request;
    assert.equal(request.mac, mac(`RQG1|${controller}|7|${nonce}|${action}|${request.payload}`));
    let value;
    if (action === 'status') value = { v: 1, wifi_snapshot: 1, control_mode: 'remote', recordings: 1,
      build: { v: 1, packaging_fingerprint: build }, p: 'IDLE', g: 0, r: 0, ms: 0, pending: false, rec: false,
      o: { v: 1, s: 'observed', e: '8', a: 5, i: '1', g: 0, r: 0 }, meta_update_control: control,
      meta_update: { v: version }, controller_access: { v: 1, role: waiting ? 'waiting' : 'active', active_phone: 'Test', revision: 2 } };
    else {
      assert.equal(action, 'meta-update'); calls.push(JSON.parse(request.payload));
      value = { v: 1, id: calls.at(-1).id, state: 'accepted', cycle: '2', revision: '8', attempt: '20', reason: '' };
    }
    const payload = JSON.stringify(value);
    return Response.json({ v: 1, device: owner.device, epoch: '7', nonce, action, payload,
      mac: mac(`RQG1-REPLY|${controller}|${owner.device}|7|${nonce}|${action}|${payload}`) });
  };
  return { calls, gateway: await joinQuest({ endpoint: '192.168.0.10', serial: owner.serial, build, enrollment: { controller, key }, fetcher }) };
}

test('enrolled update action authenticates the exact bounded check body and original receipt', async () => {
  const { gateway, calls } = await peer();
  const body = { v: 1, id: 'f'.repeat(16), op: 'check-and-update', cycle: '2', revision: '7' };
  assert.equal((await gateway.metaUpdate(body)).state, 'accepted');
  await gateway.metaUpdate({ v: 1, id: body.id, op: 'receipt' });
  assert.deepEqual(calls, [body, { v: 1, id: body.id, op: 'receipt' }]);
  for (const bad of [{ ...body, url: 'https://invalid/' }, { ...body, target: 20 }, { ...body, op: 'install' },
    { ...body, cycle: 2 }, { ...body, revision: '0' }, { ...body, id: 'x' }]) assert.throws(() => gateway.metaUpdate(bad));
  assert.equal(calls.length, 2); gateway.close();
});

test('legacy advertisement and a waiting controller cannot invoke the update action', async () => {
  for (const options of [{ version: 1 }, { control: 0 }]) {
    const { gateway } = await peer(options); assert.equal(gateway.metaUpdate, undefined); gateway.close();
  }
  const { gateway, calls } = await peer({ waiting: true });
  assert.throws(() => gateway.metaUpdate({ v: 1, id: 'f'.repeat(16), op: 'receipt' }));
  assert.equal(calls.length, 0); gateway.close();
});
