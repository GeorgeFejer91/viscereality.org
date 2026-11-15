import test from 'node:test';
import assert from 'node:assert/strict';
import * as contract from '../kiosk-control.js';

const owner = { device: 'a'.repeat(32), serial: 'TEST', epoch: '7', native_epoch: '8' };
const kiosk = (patch = {}) => ({ v: 1, g: 2, requested: true, effective: false,
  state: 'interrupted', service: 'failed', blocked: 'service-failed', detail: 'not connected', retry_available: true, ...patch });
const state = (patch = {}) => ({ status: { p: 'IDLE', rec: false, pending: false, kiosk: kiosk(),
  o: { v: 1, s: 'observed', e: '8', a: 5, i: '100' }, ...patch }, busy: false, pending: false, unknown: false });
const storage = () => {
  const values = new Map();
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
};
function fixture() {
  const calls = [];
  let result = 'pending';
  const gateway = { owner, kiosk: async command => { calls.push(command); if (result === 'lost') throw Error('lost');
    return { id: command.id, state: result, detail: 'observed' }; } };
  return { gateway, calls, reply: value => { result = value; } };
}

test('kiosk separates requested from effective and gates bounded advertised retry capability', () => {
  const model = new contract.KioskControl(); const peer = fixture();
  let view = model.update(peer.gateway, state());
  assert.equal(view.canRetry, true);
  assert.match(view.summary, /angefordert.*nicht wirksam/);
  for (const patch of [{ effective: 1 }, { g: -1 }, { service: 'x'.repeat(49) }, { detail: 'x'.repeat(241) }, { retry_available: null }]) {
    assert.equal(model.update(peer.gateway, state({ kiosk: kiosk(patch) })).available, false);
  }
  for (const change of [{ rec: true }, { pending: true }, { kiosk: kiosk({ retry_available: false }) },
    { controller_access: { v: 1, role: 'waiting', active_phone: 'Other', revision: 2 } }]) {
    assert.equal(model.update(peer.gateway, state(change)).canRetry, false);
  }
  view = model.update(peer.gateway, state({ kiosk: kiosk({ effective: true, retry_available: false }) }));
  assert.match(view.summary, /wirksam/);
  assert.equal(view.canRetry, false);
});

test('kiosk retry journals its CAS generation and checks one original ID without replay after reload', async () => {
  let now = 0;
  const store = storage(); const peer = fixture();
  const model = new contract.KioskControl({ storage: store, now: () => now });
  model.update(peer.gateway, state()); await model.retry();
  assert.equal(peer.calls.length, 1);
  assert.deepEqual(Object.keys(peer.calls[0]).sort(), ['g', 'id', 'op', 'v']);
  assert.equal(peer.calls[0].g, 2);
  assert.equal(peer.calls[0].op, 'retry');
  const restored = new contract.KioskControl({ storage: store, now: () => now });
  restored.update(peer.gateway, state()); await restored.retry();
  await restored.refresh();
  assert.equal(peer.calls.length, 2);
  assert.deepEqual(peer.calls[1], { v: 1, id: peer.calls[0].id, op: 'receipt' });
  now = 13000; await restored.refresh();
  let view = restored.update(peer.gateway, state());
  assert.equal(view.canRetry, false);
  assert.equal(view.canCheck, true);
  assert.match(view.message, /Wartezeit.*abgelaufen/);
  peer.reply('confirmed'); restored.recheck(); await restored.refresh();
  view = restored.update(peer.gateway, state());
  assert.equal(restored.pending, null);
  assert.match(view.summary, /nicht wirksam/, 'a service receipt never manufactures effective kiosk');
  assert.match(view.message, /Dienst.*verbunden/);
});

test('wrong kiosk IDs, current actor loss and pending control retain the fence', async () => {
  const model = new contract.KioskControl(); const peer = fixture(); peer.reply('lost');
  model.update(peer.gateway, state()); await model.retry();
  const id = model.pending.command.id;
  peer.gateway.kiosk = async command => ({ id: 'b'.repeat(16), state: 'confirmed', detail: '' });
  await model.refresh(); assert.equal(model.pending.command.id, id);
  model.update({ ...peer.gateway, owner: { ...owner, device: 'c'.repeat(32) } }, state());
  assert.equal(model.view.canCheck, false);
  assert.equal(model.view.canAcknowledge, false);
  model.update({ ...peer.gateway, owner: { ...owner, native_epoch: '9' } }, state({ o: { v: 1, s: 'observed', e: '9', a: 0, i: '101' } }));
  assert.equal(model.view.canAcknowledge, true);
  model.acknowledge(); assert.equal(model.pending, null);
});
