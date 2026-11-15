import test from 'node:test';
import assert from 'node:assert/strict';
import { BreathingSetup, breathingSetup } from '../breathing-setup.js';

const owner = { device: 'a'.repeat(32), serial: 'TEST-QUEST', epoch: '7', native_epoch: '8' };
const cycle = 705600;
const native = (patch = {}) => ({
  v: 1, g: 1, r: 0, stage: 'testing', input: 'automatic', phase: 'inhale',
  position_frames: 0, cycle_frames: cycle, target_frames: 1411200, completed_frames: 0,
  attempt: 0, calibration: 'idle', failure: '', ready: false, playing: true,
  experiment_elapsed_ms: 0, ...patch,
});
const snapshot = (setup = native(), patch = {}) => ({
  breath_setup: 1, p: 'IDLE', h: 'vr-preparation', rec: false, pending: false,
  pi: { r: true, f: true }, o: { v: 1, s: 'observed', e: '8', a: 5, i: '100' },
  setup, ...patch,
});
const state = (status = snapshot(), patch = {}) => ({ status, busy: false, pending: false, unknown: false, ...patch });
function fixture(status = snapshot()) {
  const calls = [];
  let reply = 'confirmed';
  let failure = false;
  const gateway = {
    owner, initial: status,
    async breathSetup(command) {
      calls.push(structuredClone(command));
      if (failure) throw Error('Connection lost');
      return { v: 1, id: command.id, state: reply };
    },
    async receipt(id) { return { v: 1, id, state: reply }; },
  };
  return { gateway, calls, reply(value) { reply = value; }, fail(value) { failure = value; } };
}
const storage = () => {
  const data = new Map();
  return { getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value), removeItem: key => data.delete(key) };
};

test('accepts only bounded native setup observations from the current native epoch', () => {
  assert.deepEqual(breathingSetup(snapshot(), owner), native());
  for (const patch of [
    { v: 2 }, { g: 0 }, { r: -1 }, { stage: 'invented' }, { input: 'browser' },
    { phase: 'exhale' }, { position_frames: -1 }, { cycle_frames: 7 },
    { target_frames: cycle }, { completed_frames: 1411201 }, { attempt: -1 },
    { calibration: 'success' }, { failure: 'x'.repeat(257) }, { ready: 1 },
    { playing: null }, { experiment_elapsed_ms: 900001 },
    { calibration: 'collecting', stage: 'calibrating', ready: true },
    { calibration: 'collecting', stage: 'calibrating', input: 'polar-tilt' },
    { ready: true, input: 'automatic' },
  ]) assert.equal(breathingSetup(snapshot(native(patch)), owner), null, JSON.stringify(patch));
  for (const status of [snapshot(undefined, { breath_setup: 0 }), snapshot(undefined, { o: { e: '9' } }),
    snapshot(undefined, { o: { v: 1, s: 'observed', e: '8', a: 1501, i: '100' } })]) {
    assert.equal(breathingSetup(status, owner), null);
  }
});

test('an explicit installation identity permits an unavailable serial but waiting phones cannot calibrate', () => {
  const installation = { ...owner, serial: '', identity_kind: 'app-installation' };
  assert.deepEqual(breathingSetup(snapshot(), installation), native());
  assert.equal(breathingSetup(snapshot(), { ...owner, serial: '' }), null);
  const ui = new BreathingSetup(); const peer = fixture(); peer.gateway.owner = installation;
  const waiting = state(snapshot(native(), { controller_access: { v: 1, role: 'waiting', active_phone: 'Other', revision: 1 } }));
  const view = ui.update(peer.gateway, waiting, true);
  assert.equal(view.available, true);
  for (const operation of ['start', 'calibrate', 'controller', 'continue']) assert.equal(view.actions[operation], false);
});

test('projects only native four-phase positions and progress through exactly two cycles', () => {
  const model = new BreathingSetup();
  const { gateway } = fixture();
  for (const [position, phase, label, radius] of [
    [0, 'inhale', 'Einatmen', .46], [176400, 'hold_in', 'Atem halten', 1],
    [352800, 'exhale', 'Ausatmen', 1], [529200, 'hold_out', 'Atem halten', .46],
  ]) {
    const view = model.update(gateway, state(snapshot(native({ position_frames: position, phase }))), true);
    assert.equal(view.phaseLabel, label);
    assert.equal(view.radius, radius);
  }
  for (const [completed, value] of [[0, 0], [705600, 1], [1411200, 2]]) {
    const view = model.update(gateway, state(snapshot(native({ r: 1, stage: 'calibrating', calibration: 'collecting',
      position_frames: cycle, completed_frames: completed, attempt: 1 }))), true);
    assert.deepEqual(view.progress, { value, max: 2 });
    assert.equal(view.actions.continue, false);
  }
});

test('reported tracking phase is independent of the intended audio phase', () => {
  const model = new BreathingSetup();
  const { gateway } = fixture();
  const measured = native({ input_phase: 'exhale', input_volume01: .32 });
  const view = model.update(gateway, state(snapshot(measured)), true);
  assert.equal(view.phaseLabel, 'Einatmen');
  assert.equal(view.inputPhaseLabel, 'Ausatmen');
  assert.equal(view.inputVolume, .32);
  assert.equal(model.update(gateway, state(), true).inputPhaseLabel, '');
  for (const patch of [{ input_phase: 'invented' }, { input_volume01: NaN }, { input_volume01: 1.01 }]) {
    assert.equal(breathingSetup(snapshot(native(patch)), owner), null);
  }
});

test('elapsed polling and calibration completion alone never unlock selection', () => {
  const model = new BreathingSetup();
  const { gateway } = fixture();
  for (let index = 0; index < 10; index++) {
    const view = model.update(gateway, state(snapshot(native({ position_frames: cycle * index,
      stage: 'calibrating', calibration: 'collecting', completed_frames: 1411200, attempt: 1 }))), true);
    assert.equal(view.actions.continue, false);
    assert.equal(view.inputLabel, 'Automatic');
  }
  const ready = native({ r: 1, position_frames: cycle * 10, calibration: 'ready', input: 'polar-tilt', ready: true, attempt: 1 });
  assert.equal(model.update(gateway, state(snapshot(ready)), true).actions.continue, true);
});

test('native failure retains Automatic, gives its reason and permits retry or controller fallback', () => {
  const model = new BreathingSetup();
  const { gateway } = fixture();
  const failed = native({ r: 1, attempt: 1, calibration: 'failed', failure: 'insufficient-range', completed_frames: 1411200 });
  const view = model.update(gateway, state(snapshot(failed)), true);
  assert.equal(view.failure, 'insufficient-range');
  assert.equal(view.inputLabel, 'Automatic');
  assert.equal(view.actions.calibrate, true);
  assert.equal(view.actions.controller, true);
  assert.equal(view.actions.continue, false);
  const fallback = native({ ...failed, r: 2, input: 'controller-state', ready: true });
  assert.equal(model.update(gateway, state(snapshot(fallback)), true).actions.continue, true);
});

test('preserves participant freshness, owner, idle-run and pending action gates', () => {
  const ready = snapshot(native({ input: 'polar-tilt', ready: true, calibration: 'ready', attempt: 1 }));
  for (const [status, flags, participantReady, gatewayPatch] of [
    [ready, { busy: true }, true, {}], [ready, { pending: true }, true, {}],
    [ready, { unknown: true }, true, {}], [snapshot(ready.setup, { pending: true }), {}, true, {}],
    [snapshot(ready.setup, { p: 'RUNNING', rec: true, h: 'vr-session' }), {}, true, {}],
    [ready, {}, true, { owner: { ...owner, native_epoch: '9' } }],
    [ready, {}, true, { breathSetup: null }],
  ]) {
    const model = new BreathingSetup();
    const { gateway } = fixture(ready);
    const view = model.update({ ...gateway, ...gatewayPatch }, state(status, flags), participantReady);
    assert.equal(Object.values(view.actions).some(Boolean), false);
  }
  const model = new BreathingSetup();
  const { gateway } = fixture(ready);
  assert.equal(model.update(gateway, state(ready), true).actions.continue, true);
  assert.equal(model.update(gateway, state(ready), false).actions.continue, false);
  assert.match(model.update(gateway, state(snapshot(ready.setup, { pi: { r: true, f: false } })), true).blockedReason, /Polar/);
});

test('old APK and malformed or regressing projections require native recovery', () => {
  const model = new BreathingSetup();
  const { gateway } = fixture();
  const old = model.update(gateway, state(snapshot(undefined, { breath_setup: undefined })), true);
  assert.equal(old.updateRequired, true);
  assert.equal(old.available, false);
  assert.equal(model.update(gateway, state(snapshot(native({ r: 4, position_frames: cycle }))), true).available, true);
  assert.equal(model.update(gateway, state(snapshot(native({ r: 3, position_frames: cycle }))), true).available, false);
  assert.equal(model.update(gateway, state(snapshot(native({ r: 4, position_frames: 0 }))), true).available, false);
  assert.equal(model.update(gateway, state(null), true).available, false);
  assert.equal(model.update(gateway, state(snapshot(native({ r: 5, position_frames: cycle * 2 }))), true).available, true);
});

test('calibrate emits one closed journaled native command and waits for fresh receipt/status', async () => {
  const saved = storage();
  const model = new BreathingSetup({ storage: saved });
  const { gateway, calls } = fixture();
  model.update(gateway, state(), true);
  await model.run('calibrate');
  assert.equal(calls.length, 1);
  assert.deepEqual(Object.keys(calls[0]).sort(), ['g', 'id', 'op', 'r', 'v']);
  assert.match(calls[0].id, /^[a-f0-9]{16}$/);
  assert.deepEqual({ ...calls[0], id: 'ID' }, { v: 1, id: 'ID', g: 1, r: 0, op: 'calibrate' });
  assert.equal(model.update(gateway, state(), true).pending, true);
  await model.run('calibrate');
  assert.equal(calls.length, 1);
  const collecting = snapshot(native({ r: 1, stage: 'calibrating', calibration: 'collecting', attempt: 1 }));
  model.update(gateway, state(collecting), true);
  await model.refresh();
  assert.equal(model.update(gateway, state(collecting), true).pending, false);
});

test('disconnect and reload preserve an uncertain command ID without resending it', async () => {
  const saved = storage();
  let model = new BreathingSetup({ storage: saved });
  const peer = fixture();
  peer.fail(true);
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  const id = peer.calls[0].id;
  assert.equal(model.update(null, state(null), true).unknown, true);
  model = new BreathingSetup({ storage: saved });
  const newer = snapshot(native({ r: 1, calibration: 'ready', input: 'polar-tilt', ready: true, attempt: 1 }));
  peer.reply('outcome_unknown');
  assert.equal(model.update(peer.gateway, state(newer), true).actions.continue, false);
  await model.refresh();
  await model.run('calibrate');
  assert.equal(peer.calls.length, 1);
  assert.equal(model.pending.command.id, id);
  peer.reply('confirmed');
  await model.refresh();
  assert.equal(model.update(peer.gateway, state(newer), true).actions.continue, true);
  assert.equal(peer.calls.length, 1);
});

test('a receipt from another owner or epoch cannot settle an uncertain command', async () => {
  const saved = storage();
  const model = new BreathingSetup({ storage: saved });
  const peer = fixture();
  peer.fail(true);
  model.update(peer.gateway, state(), true);
  await model.run('controller');
  const changed = { ...peer.gateway, owner: { ...owner, epoch: '9' } };
  const view = model.update(changed, state(snapshot(native({ r: 1, input: 'controller-state', ready: true }))), true);
  await model.refresh();
  assert.equal(view.unknown, true);
  assert.equal(view.actions.continue, false);
  assert.ok(model.pending);
});

test('explicit native rejection permits a fresh request but malformed receipt stays unknown', async () => {
  const model = new BreathingSetup();
  const peer = fixture();
  peer.reply('rejected');
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  assert.equal(model.update(peer.gateway, state(), true).actions.calibrate, true);
  peer.reply('unrecognized');
  await model.run('controller');
  assert.equal(model.update(peer.gateway, state(), true).unknown, true);
  assert.equal(peer.calls.length, 2);
});

test('controller fallback remains native-ready after a previous successful Polar calibration', () => {
  const model = new BreathingSetup();
  const { gateway } = fixture();
  const previous = native({ r: 2, input: 'controller-state', ready: true, calibration: 'ready',
    completed_frames: 1411200, attempt: 1 });
  const view = model.update(gateway, state(snapshot(previous)), true);
  assert.equal(view.actions.continue, true);
  assert.equal(view.actions.calibrate, true);
  assert.equal(view.actions.controller, false);
});

test('waiting calibration remains Automatic until native success, even across phone clock changes', () => {
  const model = new BreathingSetup();
  const { gateway } = fixture();
  const waiting = snapshot(native({ stage: 'waiting', calibration: 'waiting', attempt: 1 }));
  const before = model.update(gateway, state(waiting), true);
  const original = Date.now;
  try {
    Date.now = () => original() + 3600000;
    const after = model.update(gateway, state(waiting), true);
    assert.equal(after.phaseLabel, before.phaseLabel);
    assert.equal(after.radius, before.radius);
    assert.deepEqual(after.progress, { value: 0, max: 2 });
    assert.equal(after.actions.continue, false);
  } finally { Date.now = original; }
});

test('audio generation replacement cannot confirm an earlier outstanding mutation', async () => {
  const model = new BreathingSetup();
  const peer = fixture();
  model.update(peer.gateway, state(), true);
  await model.run('controller');
  const replacement = snapshot(native({ g: 2, r: 5 }));
  model.update(peer.gateway, state(replacement), true);
  await model.refresh();
  assert.equal(model.update(peer.gateway, state(replacement), true).unknown, true);
  assert.ok(model.pending);
});

test('retired audio commands permit explicit acknowledgement without replaying the old mutation', async () => {
  const model = new BreathingSetup({ storage: storage() });
  const peer = fixture();
  peer.fail(true);
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  assert.equal(model.update(peer.gateway, state(), true).actions.acknowledge, false);
  const retired = snapshot(native({ g: 2, r: 0 }));
  assert.equal(model.update(peer.gateway, state(retired), true).actions.acknowledge, true);
  model.acknowledgeRetired();
  const recovered = model.update(peer.gateway, state(retired), true);
  assert.equal(recovered.unknown, false);
  assert.equal(recovered.actions.calibrate, true);
  assert.equal(peer.calls.length, 1);
  assert.equal(model.pending, null);
});

test('controller fallback cancels preparation despite lost Polar data while continuation stays gated', () => {
  const model = new BreathingSetup();
  const peer = fixture();
  const lost = snapshot(native({ stage: 'calibrating', calibration: 'collecting', attempt: 1 }),
    { pi: { r: true, f: false } });
  const view = model.update(peer.gateway, state(lost), false);
  assert.equal(view.actions.controller, true);
  assert.equal(view.actions.calibrate, false);
  assert.equal(view.actions.continue, false);
});

test('controller re-enrollment permits acknowledging an old gateway journal without replay', async () => {
  const model = new BreathingSetup({ storage: storage() });
  const peer = fixture();
  peer.fail(true);
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  const renewed = { ...peer.gateway, owner: { ...owner, epoch: '9' } };
  assert.equal(model.update(renewed, state(), true).actions.acknowledge, true);
  model.acknowledgeRetired();
  assert.equal(model.update(renewed, state(), true).actions.calibrate, true);
  assert.equal(peer.calls.length, 1);
});

test('native idle permits start, paused playback blocks continuation and an aborted action never dispatches', async () => {
  const model = new BreathingSetup();
  const peer = fixture();
  const idle = snapshot(native({ stage: 'idle', playing: false }));
  assert.equal(model.update(peer.gateway, state(idle), true).actions.start, true);
  const aborted = AbortSignal.abort();
  await assert.rejects(model.run('start', aborted));
  assert.equal(peer.calls.length, 0);
  assert.equal(model.pending, null);
  await model.run('start');
  assert.equal(peer.calls[0].op, 'start');
  const next = snapshot(native({ r: 1 }));
  model.update(peer.gateway, state(next), true);
  await model.refresh();
  const paused = snapshot(native({ r: 2, playing: false, calibration: 'ready', input: 'polar-tilt', ready: true }));
  const view = model.update(peer.gateway, state(paused), true);
  assert.equal(view.actions.continue, false);
  assert.equal(view.actions.calibrate, false);
});
