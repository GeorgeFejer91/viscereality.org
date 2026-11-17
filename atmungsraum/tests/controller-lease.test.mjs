import test from 'node:test';
import assert from 'node:assert/strict';
import { ControllerLease } from '../controller-lease.js';

function fixture() {
  let time = 1000, calls = [], reads = 0;
  const storage = new Map();
  const saved = { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) };
  const gateway = { owner: { device: 'd'.repeat(32), serial: 'Q', epoch: '7', native_epoch: '8' },
    controllerAccess: async value => { calls.push(value); return { v: 1, role: 'active', active_phone: 'Phone', revision: 2 }; } };
  const state = { statusTiming: { receivedAt: time }, status: { p: 'IDLE', rec: false, pending: false,
    controller_access: { v: 1, role: 'waiting', active_phone: '', revision: 1 } } };
  const model = new ControllerLease({ now: () => time, storage: saved });
  const refresh = async () => { reads++; state.statusTiming = { receivedAt: time }; };
  return { model, gateway, state, refresh, saved, calls, get reads() { return reads; }, advance: ms => { time += ms; } };
}

test('automatically claims a fresh free native lease, but only status readback grants control', async () => {
  const f = fixture();
  await f.model.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0].expected_control_revision, 1);
  assert.equal(f.calls[0].op, 'claim');
  assert.equal(f.reads, 2);
  assert.equal(f.model.update(f.gateway, f.state).active, false, 'a claim reply alone must never paint green');
  f.state.status.controller_access = { v: 1, role: 'active', active_phone: 'Phone', revision: 2 };
  assert.equal(f.model.update(f.gateway, f.state).active, true);
  f.advance(3000);
  assert.equal(f.model.update(f.gateway, f.state).active, false, 'old active status expires');
});

test('busy owners and hidden/blocked/native-pending states only observe, never steal control', async () => {
  for (const change of [f => { f.state.status.controller_access.active_phone = 'Other'; },
    f => { f.state.status.pending = true; }, f => { f.state.status.setup = { stage: 'recovering' }; },
    f => { f.state.busy = true; }]) {
    const f = fixture(); change(f);
    await f.model.acquire(f.gateway, f.state, f.refresh);
    assert.equal(f.calls.length, 0);
  }
  for (const options of [{ visible: false }, { blocked: true }]) {
    const f = fixture(); await f.model.acquire(f.gateway, f.state, f.refresh, options);
    assert.equal(f.calls.length, 0);
  }
});

test('fresh preflight rechecks ownership and never sends a claim after a race', async () => {
  const f = fixture();
  await f.model.acquire(f.gateway, f.state, async () => { f.state.status.controller_access.active_phone = 'Other'; });
  assert.equal(f.calls.length, 0);
});

test('unknown claim is persisted and only read until a native revision fence resolves it', async () => {
  const f = fixture(); f.gateway.controllerAccess = async value => { f.calls.push(value); throw Error('lost response'); };
  await f.model.acquire(f.gateway, f.state, f.refresh);
  f.advance(60000); await f.refresh();
  const resumed = new ControllerLease({ now: () => 61000, storage: f.saved });
  assert.equal(resumed.update(f.gateway, f.state).state, 'unknown');
  await resumed.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 1);
  f.state.status.controller_access.revision = 2;
  await resumed.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 2);
  assert.notEqual(f.calls[0].id, f.calls[1].id);
});

test('a confirmed non-grant retries at a bounded cadence during recording too', async () => {
  const f = fixture(); f.gateway.controllerAccess = async value => { f.calls.push(value); return { v: 1, role: 'waiting', active_phone: '', revision: 1 }; };
  await f.model.acquire(f.gateway, f.state, f.refresh);
  await f.model.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 1);
  f.advance(5000); await f.refresh();
  await f.model.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 2);
  f.state.status.rec = true; f.state.status.p = 'RUNNING';
  f.advance(5000); await f.refresh();
  await f.model.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 3);
  f.advance(5000); await f.refresh();
  await f.model.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 3, 'recording retry respects the thirty-second cooldown');
  f.advance(25000); await f.refresh();
  await f.model.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 4);
});

test('explicit release pauses automatic acquisition until the operator requests control again', async () => {
  const f = fixture(); f.model.release();
  await f.model.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 0);
  assert.equal(f.model.update(f.gateway, f.state).state, 'released');
  f.model.request();
  await f.model.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 1);
});

test('control claims may recover a free recording lease without replaying an unresolved session command', async () => {
  const f = fixture(); f.state.status.rec = true; f.state.status.p = 'RUNNING'; f.state.pending = f.state.unknown = true;
  await f.model.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 1);
  assert.equal(f.state.pending, true);
  assert.equal(f.state.unknown, true);
});

test('lost active status is visible and a delayed native response cannot grant green control', () => {
  const f = fixture();
  f.state.status.controller_access = { v: 1, role: 'active', active_phone: 'Phone', revision: 2 };
  assert.equal(f.model.update(f.gateway, f.state).active, true);
  f.state.status = null;
  assert.equal(f.model.update(f.gateway, f.state).state, 'lost');
  f.state.status = { controller_access: { v: 1, role: 'active', active_phone: 'Phone', revision: 2 } };
  f.state.statusTiming.transportMs = 3000;
  assert.equal(f.model.update(f.gateway, f.state).active, false);
});

test('conclusive non-grants continue after the third attempt with a thirty-second cooldown', async () => {
  const f = fixture(); f.gateway.controllerAccess = async value => { f.calls.push(value); return structuredClone(f.state.status.controller_access); };
  for (let i = 0; i < 8; i++) {
    await f.model.acquire(f.gateway, f.state, f.refresh);
    f.advance(5000); await f.refresh();
  }
  assert.equal(f.calls.length, 3);
  assert.equal(f.model.update(f.gateway, f.state).state, 'retrying');
  await f.model.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 4);
  for (let i = 0; i < 10; i++) await f.model.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 4, 'ordinary ticks cannot spin additional requests');
  f.advance(30000); await f.refresh();
  await f.model.acquire(f.gateway, f.state, f.refresh);
  assert.equal(f.calls.length, 5);
  assert.equal(new Set(f.calls.map(value => value.id)).size, 5);
});
