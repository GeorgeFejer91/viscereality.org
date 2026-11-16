import test from 'node:test';
import assert from 'node:assert/strict';
import { MetaUpdateControl } from '../meta-update-control.js';

const owner = { device: 'a'.repeat(32), serial: 'TEST', epoch: '7', native_epoch: '8' };
const storage = () => { const m = new Map(); return { getItem: k => m.get(k) ?? null,
  setItem: (k, v) => m.set(k, v), removeItem: k => m.delete(k) }; };
const state = (patch = {}) => ({ status: { meta_update_control: 1, p: 'IDLE', rec: false, pending: false,
  controller_access: { v: 1, role: 'active', active_phone: 'This phone', revision: 2 },
  o: { v: 1, s: 'observed', e: '8', a: 5, i: '100' }, meta_update: { v: 2, state: 'current', reason: '',
    installed_code: 18, available_code: 18, required_code: 18, downloaded_bytes: 0, total_bytes: null,
    session_allowed: true, cycle: '2', revision: '7', attempt: '19', can_check: true, busy_reason: '' }, ...patch },
  busy: false, pending: false, unknown: false });
function fixture() {
  const calls = []; let mode = 'accepted';
  const gateway = { owner, metaUpdate: async cmd => { calls.push(cmd); if (mode === 'lost') throw Error('lost');
    return { v: 1, id: cmd.id, state: mode, cycle: '2', revision: '8', attempt: '20', reason: '' }; } };
  return { gateway, calls, mode: v => { mode = v; } };
}

test('rendering and status refresh never issue an update; manual request binds native cycle/revision', async () => {
  const peer = fixture(), model = new MetaUpdateControl();
  for (let i = 0; i < 3; i++) { assert.equal(model.update(peer.gateway, state()).canRequest, true); await model.refresh(); }
  assert.equal(peer.calls.length, 0);
  await model.request();
  assert.equal(peer.calls.length, 1);
  assert.deepEqual(Object.keys(peer.calls[0]).sort(), ['cycle', 'id', 'op', 'revision', 'v']);
  assert.equal(peer.calls[0].op, 'check-and-update');
  assert.equal(peer.calls[0].cycle, '2'); assert.equal(peer.calls[0].revision, '7');
  assert.match(peer.calls[0].id, /^[a-f0-9]{16}$/);
  assert.equal(model.update(peer.gateway, state()).canRequest, false);
  await model.request(); assert.equal(peer.calls.length, 1);
  const observed = state(); Object.assign(observed.status.meta_update, { attempt: '20', revision: '9', state: 'checking', can_check: false, session_allowed: false });
  model.update(peer.gateway, observed); assert.equal(model.pending, null);
});

test('only a fresh native advertised safe boundary and active controller permit a manual update', async () => {
  for (const patch of [{ meta_update_control: 0 }, { meta_update: undefined }, { controller_access: undefined }, { rec: true }, { pending: true },
    { p: 'RUNNING' }, { o: { ...state().status.o, a: 1501 } }, { o: { ...state().status.o, e: '9' } },
    { controller_access: { v: 1, role: 'waiting', active_phone: 'Other', revision: 2 } },
    { meta_update: { ...state().status.meta_update, can_check: false, busy_reason: 'preparation-active' } }]) {
    const peer = fixture(), model = new MetaUpdateControl();
    assert.equal(model.update(peer.gateway, state(patch)).canRequest, false);
    await model.request(); assert.equal(peer.calls.length, 0);
  }
  const model = new MetaUpdateControl(), peer = fixture();
  assert.equal(model.update(peer.gateway, { ...state(), unknown: true }).canRequest, false);
  assert.equal(model.update(peer.gateway, { ...state(), busy: true }).canRequest, false);
  assert.equal(model.update(peer.gateway, { ...state(), pending: true }).canRequest, false);
});

test('lost response survives reload and only reads the original receipt', async () => {
  const saved = storage(), peer = fixture(), first = new MetaUpdateControl({ storage: saved });
  peer.mode('lost'); first.update(peer.gateway, state()); await first.request();
  const id = peer.calls[0].id;
  const next = new MetaUpdateControl({ storage: saved }); next.update(peer.gateway, state());
  await next.request(); assert.equal(peer.calls.length, 1);
  peer.mode('accepted'); await next.refresh();
  assert.deepEqual(peer.calls[1], { v: 1, id, op: 'receipt' });
  assert.equal(next.pending.command.id, id);
  assert.equal(next.update(peer.gateway, state()).canRequest, false);
});

test('wrong or malformed receipts cannot retire an unresolved command or claim installation', async () => {
  const peer = fixture(), model = new MetaUpdateControl(); peer.mode('lost');
  model.update(peer.gateway, state()); await model.request();
  const original = model.pending;
  for (const patch of [{ id: '0'.repeat(16) }, { state: 'updated' }, { cycle: 2 }, { revision: '0' },
    { attempt: null }, { reason: 'x'.repeat(97) }]) {
    model.observe({ v: 1, id: original.command.id, state: 'accepted', cycle: '2', revision: '8', attempt: '20', reason: '', ...patch }, peer.gateway);
    assert.equal(model.pending, original);
  }
  assert.doesNotMatch(model.update(peer.gateway, state()).message, /erfolgreich installiert/);
});

test('a retired owner requires explicit local acknowledgment and cannot replay to another headset', async () => {
  const peer = fixture(), model = new MetaUpdateControl(); peer.mode('lost');
  model.update(peer.gateway, state()); await model.request();
  const next = { ...peer.gateway, owner: { ...owner, epoch: '9', native_epoch: '10' } };
  const s = state({ o: { ...state().status.o, e: '10' } });
  assert.equal(model.update(next, s).canAcknowledge, true);
  await model.refresh(); assert.equal(peer.calls.length, 1);
  model.acknowledge(); assert.equal(model.pending, null);
  assert.equal(model.update(next, s).canRequest, true);
});

test('a stale original command can be retired after the native cycle advances, without a replacement request', async () => {
  const peer = fixture(), model = new MetaUpdateControl(); peer.mode('lost');
  model.update(peer.gateway, state()); await model.request();
  model.observe({ v: 1, id: model.pending.command.id, state: 'stale', cycle: '3', revision: '10', attempt: null, reason: 'cycle-changed' }, peer.gateway);
  assert.equal(model.pending, null);
  assert.equal(peer.calls.length, 1);
  assert.match(model.message, /geändert/);
});

test('accepted receipt polling remains finite and retains the unresolved original attempt', async () => {
  const peer = fixture(); let now = 0;
  const model = new MetaUpdateControl({ now: () => now }); model.update(peer.gateway, state()); await model.request();
  for (let i = 0; i < 20; i++) { model.update(peer.gateway, state()); await model.refresh(); now += 1000; }
  assert.ok(peer.calls.length <= 13);
  assert.equal(model.expired, true);
  assert.equal(model.pending.accepted.attempt, '20');
  assert.ok(peer.calls.slice(1).every(c => c.op === 'receipt'));
});

test('a completed native check clears accepted-request copy and idle instructions do not claim a check is running', async () => {
  const peer = fixture(), model = new MetaUpdateControl();
  model.update(peer.gateway, state()); await model.request();
  const completed = state();
  Object.assign(completed.status.meta_update, { attempt: '20', revision: '9' });
  const view = model.update(peer.gateway, completed);
  assert.equal(model.pending, null);
  assert.equal(view.message, '');
  assert.equal(view.canRequest, true);
  assert.doesNotMatch(view.prerequisites, /^Die Quest prüft die erforderliche Version/);
  assert.equal(peer.calls.length, 1);
});
