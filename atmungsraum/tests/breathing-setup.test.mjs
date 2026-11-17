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
    async breathSetupReceipt(id) { return { v: 1, id, state: reply }; },
  };
  return { gateway, calls, reply(value) { reply = value; }, fail(value) { failure = value; } };
}
const storage = () => {
  const data = new Map();
  return { getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value), removeItem: key => data.delete(key) };
};

test('only a bound explicit non-accepted stale rejection permits fresh selection retry', async () => {
  for (const patch of [{}, {accepted:undefined}, {accepted:true}, {reason:'busy'}, {id:'f'.repeat(16)}, {state:'outcome_unknown'}]) {
    const model=new BreathingSetup(), peer=fixture();
    peer.gateway.breathSetup=async command=>({v:1,id:command.id,state:'rejected',accepted:false,reason:'stale-revision',...patch});
    model.update(peer.gateway,state(),true);
    const result=await model.run('controller');
    assert.equal(result, Object.keys(patch).length ? undefined : 'stale-revision');
  }
});

test('review binding survives sample readiness but retires on a selected sensor change', () => {
  const model=new BreathingSetup(), peer=fixture();
  const y={s:'b'.repeat(32),g:8,x:'c'.repeat(12)};
  const original=model.update(peer.gateway,state(snapshot(native({input:'controller-state',ready:true}),{y})),true).acceptance;
  assert.equal(model.update(peer.gateway,state(snapshot(native({input:'controller-state',ready:false,r:1}),{y})),true).acceptance,original);
  assert.notEqual(model.update(peer.gateway,state(snapshot(native({input:'controller-state',ready:true,r:2}),{y:{...y,x:'d'.repeat(12)}})),true).acceptance,original);
});

test('confirmed Controller mode hides Polar progress and reports a requested switch without claiming it happened', async () => {
  const model = new BreathingSetup(); const peer = fixture(); peer.reply('pending');
  const controller = state(snapshot(native({ input: 'controller-state', ready: true, attempt: 2,
    calibration: 'failed', failure: 'clock_stalled', completed_frames: 100 })));
  let view = model.update(peer.gateway, controller, true);
  assert.equal(view.title, 'Controller testen');
  assert.equal(view.showPolarCalibration, false);
  assert.equal(view.failureLabel, '');
  assert.equal(view.qualityWarning, '');
  assert.equal(view.actions.calibrate, true, 'explicit Polar setup remains available');
  await model.run('calibrate');
  view = model.update(peer.gateway, controller, true);
  assert.equal(view.inputLabel, 'Controller State · feste Ausrichtung');
  assert.equal(view.pendingInputLabel, 'Polar einrichten');
  assert.match(view.recovery.detail, /Bestätigung/);
  assert.equal(peer.calls.length, 1);
});

test('ACTUAL radius is an optional bounded renderer projection and never an input-volume fallback', () => {
  const model = new BreathingSetup(); const peer = fixture();
  assert.equal(model.update(peer.gateway, state(snapshot(native({ input_volume01: .9 }))), true).renderer, null);
  const renderer = { v: 1, progress01: .2, radius_m: 1.4, frame: 7, render_generation: 2,
    settings_revision: 0, observed_ns: '1200000000', age_ns: 10000000 };
  assert.deepEqual(model.update(peer.gateway, state(snapshot(native({ renderer }))), true).renderer, renderer);
  for (const patch of [{ v: 2 }, { progress01: 1.01 }, { radius_m: 0 }, { frame: -1 },
    { render_generation: 0 }, { settings_revision: -1 }, { observed_ns: '1e9' }, { age_ns: 500000001 }]) {
    assert.equal(breathingSetup(snapshot(native({ renderer: { ...renderer, ...patch } })), owner), null);
  }
  assert.ok(breathingSetup(snapshot(native({ renderer: null })), owner));
});

test('an expired unknown setup exposes read-only checking and explains a local acknowledgement separately', async () => {
  let now = 0;
  const model = new BreathingSetup({ now: () => now }); const peer = fixture(); peer.reply('outcome_unknown');
  model.update(peer.gateway, state(), true); await model.run('controller');
  await model.refresh(); now = 13000; await model.refresh();
  let view = model.update(peer.gateway, state(), true);
  assert.equal(view.recovery.action, 'Bestätigung erneut abfragen');
  assert.match(view.recovery.detail, /Controller/);
  assert.match(view.recovery.detail, /nicht erneut gesendet/);
  assert.equal(view.actions.acknowledge, false);
  const next = { ...peer.gateway, owner: { ...owner, native_epoch: '9' } };
  view = model.update(next, state(snapshot(native(), { o: { ...snapshot().o, e: '9' } })), true);
  assert.equal(view.recovery.action, 'Aktuellen Quest-Zustand übernehmen');
  assert.match(view.recovery.detail, /setzt die Quest nicht zurück/);
});

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

test('fresh parked native Automatic admits one calibration or controller choice without a separate start', async () => {
  for (const op of ['calibrate', 'controller']) {
    const model = new BreathingSetup(); const peer = fixture();
    const parked = state(snapshot(native({ stage: 'idle', playing: false, input_volume01: 0 })));
    const view = model.update(peer.gateway, parked, true);
    assert.equal(view.actions.start, false);
    assert.equal(view.actions[op], true);
    assert.equal(peer.calls.length, 0);
    await model.run(op);
    await model.run(op);
    assert.equal(peer.calls.length, 1);
    assert.equal(peer.calls[0].op, op);
  }
});

test('optional source clock readiness gates calibration but preserves the fixed controller alternative', () => {
  const model = new BreathingSetup(); const peer = fixture();
  const setup = native({ stage: 'idle', playing: false, source_clock_ready: false });
  const view = model.update(peer.gateway, state(snapshot(setup)), true);
  assert.equal(view.actions.calibrate, false);
  assert.equal(view.actions.controller, true);
  assert.match(view.blockedReason, /Bewegungsdaten/);
  assert.equal(model.update(peer.gateway, state(snapshot({ ...setup, source_clock_ready: true })), true).actions.calibrate, true);
  for (const invalid of [null, 0, 1, 'true']) {
    assert.equal(breathingSetup(snapshot({ ...setup, source_clock_ready: invalid }), owner), null);
  }
});

test('bounded optional quality warns during collection without fabricating failure or success', () => {
  const quality = { accepted: 150, masked: 8, motion: 5, timing: 2, invalid: 1, usable_fraction: .9, warning: 'motion' };
  const setup = native({ stage: 'calibrating', calibration: 'collecting', quality });
  const model = new BreathingSetup(); const peer = fixture();
  const view = model.update(peer.gateway, state(snapshot(setup)), true);
  assert.match(view.qualityWarning, /Oberkörper ruhig/);
  assert.equal(view.failure, '');
  assert.equal(view.actions.continue, false);
  for (const invalid of [null, [], ...['accepted', 'masked', 'motion', 'timing', 'invalid'].map(key => ({ ...quality, [key]: 801 })), { ...quality, masked: -1 },
    { ...quality, usable_fraction: 1.1 }, { ...quality, warning: 'invented' }, { ...quality, extra: 1 }]) {
    assert.equal(breathingSetup(snapshot({ ...setup, quality: invalid }), owner), null);
  }
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

test('native terminal reasons get plain wording while retaining the exact failure token', () => {
  const model = new BreathingSetup(); const peer = fixture();
  const view = model.update(peer.gateway, state(snapshot(native({ calibration:'failed',failure:'audio_paused' }))), true);
  assert.equal(view.failure, 'audio_paused');
  assert.equal(view.failureLabel, 'Die Atemschleife wurde angehalten');
  assert.equal(view.actions.continue, false);
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
  let now = 0, model = new BreathingSetup({ storage: saved, now: () => now });
  const peer = fixture();
  peer.fail(true);
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  const id = peer.calls[0].id;
  assert.equal(model.update(null, state(null), true).unknown, true);
  model = new BreathingSetup({ storage: saved, now: () => now });
  const newer = snapshot(native({ r: 1, calibration: 'ready', input: 'polar-tilt', ready: true, attempt: 1 }));
  peer.reply('outcome_unknown');
  assert.equal(model.update(peer.gateway, state(newer), true).actions.continue, false);
  await model.refresh();
  await model.run('calibrate');
  assert.equal(peer.calls.length, 1);
  assert.equal(model.pending.command.id, id);
  peer.reply('confirmed');
  now += 1000;
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

test('same-actor audio replacement permits only receipt reconciliation without discarding or replaying the old mutation', async () => {
  const model = new BreathingSetup({ storage: storage() });
  const peer = fixture();
  peer.fail(true);
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  assert.equal(model.update(peer.gateway, state(), true).actions.acknowledge, false);
  const replaced = snapshot(native({ g: 2, r: 0 }));
  const view = model.update(peer.gateway, state(replaced), true);
  assert.equal(view.actions.acknowledge, false);
  assert.equal(view.actions.checkReceipt, true);
  assert.equal(model.acknowledgeRetired(), false);
  const unresolved = model.update(peer.gateway, state(replaced), true);
  assert.equal(unresolved.unknown, true);
  assert.equal(unresolved.actions.calibrate, false);
  assert.equal(peer.calls.length, 1);
  assert.equal(model.pending.command.id, peer.calls[0].id);
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

test('native idle permits calibration, paused playback blocks continuation and an aborted action never dispatches', async () => {
  const model = new BreathingSetup();
  const peer = fixture();
  const idle = snapshot(native({ stage: 'idle', playing: false }));
  assert.equal(model.update(peer.gateway, state(idle), true).actions.calibrate, true);
  const aborted = AbortSignal.abort();
  await assert.rejects(model.run('calibrate', aborted));
  assert.equal(peer.calls.length, 0);
  assert.equal(model.pending, null);
  await model.run('calibrate');
  assert.equal(peer.calls[0].op, 'calibrate');
  const next = snapshot(native({ r: 1 }));
  model.update(peer.gateway, state(next), true);
  await model.refresh();
  const paused = snapshot(native({ r: 2, playing: false, calibration: 'ready', input: 'polar-tilt', ready: true }));
  const view = model.update(peer.gateway, state(paused), true);
  assert.equal(view.actions.continue, false);
  assert.equal(view.actions.calibrate, false);
});

const recovery = (id, patch = {}) => ({ id, state: 'pending', deadline_ns: 2000000000, target_generation: null, ...patch });
const recovering = (id, patch = {}) => native({ stage: 'recovering', ready: false, clock_valid: false,
  recovery: recovery(id), ...patch });
const receipt = (id, patch = {}) => ({ v: 1, id, state: 'pending', accepted: true, reason: 'audio-recovery-pending', ...patch });

test('recovering status is bounded, prevents every setup admission and never implies completed calibration', () => {
  const id = 'a'.repeat(16), model = new BreathingSetup(), peer = fixture();
  const setup = recovering(id);
  assert.equal(breathingSetup(snapshot(setup), owner), setup);
  const view = model.update(peer.gateway, state(snapshot(setup)), true);
  assert.equal(view.recovering, true);
  assert.deepEqual(view.progress, { value: 0, max: 2 });
  for (const action of ['calibrate', 'controller', 'continue', 'acknowledge']) assert.equal(view.actions[action], false);
  for (const patch of [
    { ready: true }, { recovery: null }, { recovery: recovery('A'.repeat(16)) },
    { recovery: recovery(id, { deadline_ns: -1 }) }, { recovery: recovery(id, { state: 'confirmed' }) },
    { recovery: recovery(id, { target_generation: 0 }) }, { clock_valid: 'false' },
  ]) assert.equal(breathingSetup(snapshot({ ...setup, ...patch }), owner), null, JSON.stringify(patch));
  assert.ok(breathingSetup(snapshot(native()), owner), 'older APK statuses remain valid without recovery telemetry');
});

test('pending recovery resolves the original ID after generation change and still waits for native two-cycle collection', async () => {
  const model = new BreathingSetup({ storage: storage() }), peer = fixture(), reads = [];
  peer.gateway.breathSetup = async command => { peer.calls.push(command); return receipt(command.id); };
  peer.gateway.breathSetupReceipt = async id => { reads.push(id); return receipt(id, { state: 'confirmed', reason: 'accepted', recovery_generation: 2 }); };
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  const id = peer.calls[0].id;
  let view = model.update(peer.gateway, state(snapshot(recovering(id))), true);
  assert.equal(view.pending, true);
  assert.equal(view.recovering, true);
  await model.run('controller');
  assert.equal(peer.calls.length, 1);
  const waiting = snapshot(native({ g: 2, r: 1, stage: 'waiting', calibration: 'waiting', attempt: 1 }));
  model.update(peer.gateway, state(waiting), true);
  await model.refresh();
  view = model.update(peer.gateway, state(waiting), true);
  assert.deepEqual(reads, [id]);
  assert.equal(view.pending, false);
  assert.equal(view.unknown, false);
  assert.equal(view.actions.continue, false);
  assert.deepEqual(view.progress, { value: 0, max: 2 });
  assert.equal(model.pending, null);
});

test('terminal recovery rejection permits an explicit retry and retains a practical failure explanation', async () => {
  for (const reason of ['audio-recovery-failed', 'audio-recovery-timeout', 'audio-recovery-retired']) {
    const model = new BreathingSetup(), peer = fixture();
    peer.gateway.breathSetup = async command => { peer.calls.push(command); return receipt(command.id); };
    peer.gateway.breathSetupReceipt = async id => receipt(id, { state: 'rejected', accepted: false, reason });
    model.update(peer.gateway, state(), true);
    await model.run('controller');
    await model.refresh();
    const view = model.update(peer.gateway, state(), true);
    assert.equal(view.pending, false);
    assert.equal(view.actions.calibrate, true);
    assert.equal(view.actions.controller, true);
    assert.match(view.message, /Atemschleife.*erneut/i);
    assert.equal(peer.calls.length, 1);
  }
});

test('wrong IDs and malformed recovery receipt shapes preserve uncertainty and the original mutation', async () => {
  for (const patch of [
    { id: 'f'.repeat(16) }, { id: 'A'.repeat(16) }, { accepted: false },
    { accepted: 'true' }, { state: 'confirmed', accepted: false, reason: 'accepted' },
    { state: 'rejected', accepted: true, reason: 'audio-recovery-failed' },
    { state: 'confirmed', reason: 'audio-recovery-pending' }, { reason: 'bad\nreason' },
  ]) {
    const model = new BreathingSetup(), peer = fixture();
    peer.gateway.breathSetup = async command => { peer.calls.push(command); return receipt(command.id, patch); };
    model.update(peer.gateway, state(), true);
    await model.run('calibrate');
    const view = model.update(peer.gateway, state(), true);
    assert.equal(view.pending, true, JSON.stringify(patch));
    assert.equal(view.unknown, true, JSON.stringify(patch));
    assert.equal(view.actions.calibrate, false);
    assert.equal(model.pending.command.id, peer.calls[0].id);
  }
});

test('a lost recovery admission reconciles through its full terminal receipt without replay after reload', async () => {
  const saved = storage(), peer = fixture();
  let now = 0, model = new BreathingSetup({ storage: saved, now: () => now });
  peer.fail(true);
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  const id = peer.calls[0].id;
  model.update(null, state(null), true);
  model = new BreathingSetup({ storage: saved, now: () => now });
  const reads = [];
  let result = { v: 1, id, state: 'confirmed' };
  peer.gateway.breathSetupReceipt = async queryId => { reads.push(queryId); return result; };
  const waiting = state(snapshot(native({ g: 2, r: 1, stage: 'waiting', calibration: 'waiting', attempt: 1, recovery: null })));
  let view = model.update(peer.gateway, waiting, true);
  assert.equal(model.pending.recovery, undefined, 'the initial pending admission was never observed');
  assert.equal(view.actions.acknowledge, false);
  assert.equal(view.actions.checkReceipt, true);
  assert.equal(model.acknowledgeRetired(), false);
  assert.equal(model.pending.command.id, id);
  assert.equal(model.recheckReceipt(), true);
  await model.refresh();
  view = model.update(peer.gateway, waiting, true);
  assert.equal(view.unknown, true, 'a field-light confirmation cannot retire the original ID');
  assert.equal(view.actions.acknowledge, false);
  now += 20000;
  await model.refresh();
  view = model.update(peer.gateway, waiting, true);
  assert.equal(view.actions.acknowledge, false);
  assert.equal(view.actions.checkReceipt, true);
  assert.equal(model.acknowledgeRetired(), false);
  assert.equal(model.pending.command.id, id, 'local waiting expiry does not prove native actor retirement');
  result = receipt(id, { state: 'confirmed', reason: 'accepted', recovery_generation: 3 });
  assert.equal(model.recheckReceipt(), true);
  await model.refresh();
  view = model.update(peer.gateway, waiting, true);
  assert.equal(view.unknown, true);
  assert.equal(view.actions.acknowledge, false);
  assert.equal(model.pending.command.id, id, 'a wrong target generation cannot discard the ID');
  now += 1000;
  result = receipt(id, { state: 'confirmed', reason: 'accepted', recovery_generation: 2 });
  await model.refresh();
  assert.equal(model.update(peer.gateway, waiting, true).unknown, false);
  assert.equal(model.pending, null);
  assert.equal(peer.calls.length, 1);
  assert.deepEqual(reads, [id, id, id]);
  assert.match(id, /^[a-f0-9]{16}$/);
});

test('receipt waiting deduplicates reads, bounds polling and offers only an explicit read-only recheck', async () => {
  let now = 0, release, count = 0;
  const model = new BreathingSetup({ now: () => now }), peer = fixture();
  peer.gateway.breathSetup = async command => { peer.calls.push(command); return receipt(command.id); };
  peer.gateway.breathSetupReceipt = async id => { count++; await new Promise(resolve => { release = resolve; }); return receipt(id); };
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  const first = model.refresh();
  assert.equal(model.refresh(), first);
  assert.equal(count, 1);
  release(); await first;
  await model.refresh();
  assert.equal(count, 1, 'the existing UI tick cannot create an immediate duplicate read');
  now = 20000;
  await model.refresh();
  let view = model.update(peer.gateway, state(), true);
  assert.equal(count, 1, 'local waiting expiry must not keep polling indefinitely');
  assert.equal(view.unknown, true);
  assert.equal(view.actions.checkReceipt, true);
  assert.equal(view.actions.calibrate, false);
  assert.equal(model.recheckReceipt(), true);
  const second = model.refresh();
  assert.equal(count, 2);
  release(); await second;
  assert.equal(peer.calls.length, 1);
  assert.ok(model.pending);
});

test('cancelling a receipt read preserves its pending ID and reconciles on reconnect without a mutation replay', async () => {
  const saved = storage(), model = new BreathingSetup({ storage: saved }), peer = fixture();
  let readingSignal;
  peer.gateway.breathSetup = async command => { peer.calls.push(command); return receipt(command.id); };
  peer.gateway.breathSetupReceipt = (id, signal) => new Promise((resolve, reject) => {
    readingSignal = signal;
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  });
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  const id = model.pending.command.id, read = model.refresh();
  model.cancel();
  assert.equal(readingSignal.aborted, true);
  await read;
  assert.equal(model.pending.command.id, id);
  assert.equal(model.unknown, true);
  model.update(null, state(null), true);
  const rejoined = { ...peer.gateway, breathSetupReceipt: async queryId => receipt(queryId, { state: 'rejected', accepted: false, reason: 'audio-recovery-timeout' }) };
  model.update(rejoined, state(), true);
  await model.refresh();
  assert.equal(model.pending, null);
  assert.equal(peer.calls.length, 1);
});

test('an ordinary full receipt or wrong recovery generation cannot settle across an unrelated audio replacement', async () => {
  for (const proof of [{}, { recovery_generation: null }, { recovery_generation: 1 }, { recovery_generation: 3 },
    { recovery_generation: '2' }, { recovery_generation: -1 }]) {
    const model = new BreathingSetup(), peer = fixture();
    peer.gateway.breathSetupReceipt = async id => receipt(id, { state: 'confirmed', reason: 'accepted', ...proof });
    model.update(peer.gateway, state(), true);
    await model.run('calibrate');
    const replacement = state(snapshot(native({ g: 2, r: 1 })));
    model.update(peer.gateway, replacement, true);
    await model.refresh();
    assert.equal(model.update(peer.gateway, replacement, true).unknown, true, JSON.stringify(proof));
    assert.ok(model.pending);
  }
});

test('explicit native audio recovery availability admits repair while preserving participant, stream and controller barriers', () => {
  const model = new BreathingSetup(), peer = fixture();
  const failed = native({ calibration: 'failed', failure: 'clock_stalled', clock_valid: false, source_clock_ready: false });
  assert.equal(model.update(peer.gateway, state(snapshot(failed)), true).actions.calibrate, false);
  const eligible = { ...failed, audio_recovery_available: true };
  assert.equal(model.update(peer.gateway, state(snapshot(eligible)), true).actions.calibrate, true);
  assert.equal(model.update(peer.gateway, state(snapshot({ ...eligible, playing: false })), true).actions.calibrate, true);
  assert.equal(model.update(peer.gateway, state(snapshot({ ...eligible, playing: false })), true).actions.controller, true);
  assert.equal(model.update(peer.gateway, state(snapshot({ ...eligible, stage: 'main' })), true).actions.calibrate, false);
  assert.equal(model.update(peer.gateway, state(snapshot(eligible, { p: 'RUNNING', h: 'vr-session', rec: true })), true).actions.calibrate, false);
  assert.equal(model.update(peer.gateway, state(snapshot(eligible)), false).actions.calibrate, false);
  assert.equal(model.update(peer.gateway, state(snapshot(eligible, { pi: { r: true, f: false } })), true).actions.calibrate, false);
  assert.equal(model.update(peer.gateway, state(snapshot(eligible, { controller_access: { v: 1, role: 'waiting', active_phone: 'Other', revision: 1 } })), true).actions.calibrate, false);
  assert.equal(model.update(peer.gateway, state(snapshot({ ...eligible, audio_recovery_available: false })), true).actions.calibrate, false);
  assert.equal(model.update(peer.gateway, state(snapshot({ ...eligible, input: 'controller-state' })), true).actions.controller, true);
  const ready = { ...eligible, ready: true, input: 'polar-tilt', calibration: 'ready' };
  assert.equal(model.update(peer.gateway, state(snapshot(ready)), true).actions.continue, false);
  for (const value of [null, 'true', 0, 1]) assert.equal(breathingSetup(snapshot({ ...eligible, audio_recovery_available: value }), owner), null);
});

test('waiting phones never poll setup receipts or accept another setup command', async () => {
  const model = new BreathingSetup(), peer = fixture();
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  let reads = 0;
  peer.gateway.breathSetupReceipt = async id => { reads++; return receipt(id); };
  const status = snapshot(native(), { controller_access: { v: 1, role: 'waiting', active_phone: 'Other', revision: 1 } });
  model.update(peer.gateway, state(status), true);
  await model.refresh();
  assert.equal(reads, 0);
  assert.ok(model.pending);
});

test('recovery admission alone cannot settle from a field-light confirmation or a local wait deadline', async () => {
  let now = 0;
  const model = new BreathingSetup({ now: () => now }), peer = fixture();
  peer.gateway.breathSetup = async command => { peer.calls.push(command); return receipt(command.id); };
  peer.gateway.breathSetupReceipt = async id => ({ v: 1, id, state: 'confirmed' });
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  const changedRevision = state(snapshot(native({ r: 1, stage: 'waiting', calibration: 'waiting', attempt: 1 })));
  model.update(peer.gateway, changedRevision, true);
  await model.refresh();
  assert.ok(model.pending, 'a recovery requires its native target generation proof');
  now = 20000;
  await model.refresh();
  assert.ok(model.pending);
  assert.equal(peer.calls.length, 1);
  assert.equal(model.update(peer.gateway, changedRevision, true).actions.continue, false);
});

test('an admitted recovery remains pending without becoming uncertain merely because its native audio generation advances', async () => {
  const model = new BreathingSetup(), peer = fixture();
  peer.gateway.breathSetup = async command => { peer.calls.push(command); return receipt(command.id); };
  peer.gateway.breathSetupReceipt = async id => receipt(id, { recovery_generation: 2 });
  model.update(peer.gateway, state(), true);
  await model.run('controller');
  const id = peer.calls[0].id;
  const advancing = state(snapshot(recovering(id, { g: 2, r: 1, recovery: recovery(id, { target_generation: 2 }) })));
  assert.equal(model.update(peer.gateway, advancing, true).unknown, false);
  await model.refresh();
  assert.equal(model.update(peer.gateway, advancing, true).unknown, false);
  assert.ok(model.pending);
});

test('an actually retired native owner permits explicit recovery acknowledgement while an in-owner generation change does not', async () => {
  const model = new BreathingSetup(), peer = fixture();
  peer.gateway.breathSetup = async command => { peer.calls.push(command); return receipt(command.id); };
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  const replacedAudio = state(snapshot(native({ g: 2 })));
  assert.equal(model.update(peer.gateway, replacedAudio, true).actions.acknowledge, false);
  const renewed = { ...peer.gateway, owner: { ...owner, epoch: '9' } };
  assert.equal(model.update(renewed, state(), true).actions.acknowledge, true);
  assert.equal(model.acknowledgeRetired(), true);
  assert.equal(model.update(renewed, state(), true).actions.calibrate, true);
  assert.equal(peer.calls.length, 1);
});

test('a receipt budget caps rapid but spaced polls even before its local waiting deadline', async () => {
  let now = 0, reads = 0;
  const model = new BreathingSetup({ now: () => now }), peer = fixture();
  peer.gateway.breathSetup = async command => { peer.calls.push(command); return receipt(command.id); };
  peer.gateway.breathSetupReceipt = async id => { reads++; return receipt(id); };
  model.update(peer.gateway, state(), true);
  await model.run('controller');
  for (let index = 0; index < 20; index++, now += 500) await model.refresh();
  assert.equal(reads, 12);
  assert.ok(model.pending);
  assert.equal(model.update(peer.gateway, state(), true).actions.checkReceipt, true);
});

test('reload rejects malformed pending operation IDs before reconciliation can use them', () => {
  for (const id of [1111111111111111, 'A'.repeat(16), 'f'.repeat(15), null]) {
    const saved = storage();
    saved.setItem('atmungsraum-breathing-setup-pending', JSON.stringify({ owner,
      command: { v: 1, id, g: 1, r: 0, op: 'calibrate' }, recovery: true }));
    const model = new BreathingSetup({ storage: saved });
    assert.equal(model.pending, null);
  }
});

test('a new native owner that reports recovery unavailable disables retries even when the last source projection was ready', () => {
  const model = new BreathingSetup(), peer = fixture();
  const unavailable = native({ clock_valid: false, audio_recovery_available: false, source_clock_ready: true });
  const view = model.update(peer.gateway, state(snapshot(unavailable)), true);
  assert.equal(view.actions.calibrate, false);
  assert.equal(view.actions.controller, false);
  assert.equal(view.actions.continue, false);
  assert.match(view.blockedReason, /Quest-App erneut/);
  assert.equal(model.update(peer.gateway, state(snapshot({ ...unavailable, clock_valid: true })), true).actions.calibrate, true);
  const legacy = { ...unavailable }; delete legacy.audio_recovery_available;
  assert.equal(model.update(peer.gateway, state(snapshot(legacy)), true).actions.calibrate, true);
});

test('native Polar capabilities admit builtin STATE choices without participant calibration', async () => {
  for (const [op, input, word] of [['polar-fixed', 'polar-tilt-fixed', 'feste Achse'], ['polar-adaptive', 'polar-tilt-adaptive', 'dynamische Achse']]) {
    const model = new BreathingSetup(), peer = fixture();
    const polar_modes = ['calibrated', 'fixed', 'adaptive'];
    const parked = state(snapshot(native({ polar_modes, stage: 'idle', playing: false })));
    let view = model.update(peer.gateway, parked, true);
    assert.equal(view.actions[op], true);
    assert.equal(peer.calls.length, 0, 'Automatic stays parked until an explicit choice');
    peer.reply('pending'); await model.run(op);
    view = model.update(peer.gateway, parked, true);
    assert.equal(view.inputLabel, 'Automatic');
    assert.match(view.pendingInputLabel, new RegExp(word));
    assert.equal(peer.calls.length, 1);
    assert.deepEqual(Object.keys(peer.calls[0]).sort(), ['g', 'id', 'op', 'r', 'v']);
    const confirmed = state(snapshot(native({ polar_modes, r: 1, input, ready: true, calibration: 'idle' })));
    model.update(peer.gateway, confirmed, true); peer.reply('confirmed'); await model.refresh();
    view = model.update(peer.gateway, confirmed, true);
    assert.equal(view.pending, false);
    assert.match(view.inputLabel, new RegExp(word));
    assert.equal(view.showPolarCalibration, false);
    assert.equal(view.failureLabel, '');
    assert.equal(view.actions.continue, true);
    assert.equal(view.actions.calibrate, true, 'explicit return to participant fit');
    assert.equal(view.actions[op], false, 'already selected mode is not reissued');
  }
});

test('absent or malformed Polar capability never offers or confirms builtin modes', async () => {
  for (const polar_modes of [undefined, null, 'fixed', ['fixed', 'fixed'], ['fixed', 'invented'], [1], {}]) {
    const patch = polar_modes === undefined ? {} : { polar_modes };
    const model = new BreathingSetup(), peer = fixture();
    const view = model.update(peer.gateway, state(snapshot(native(patch))), true);
    assert.equal(view.available, true, 'existing participant calibration remains compatible');
    assert.equal(view.polarModesUpdateRequired, true);
    assert.equal(view.actions['polar-fixed'], false);
    assert.equal(view.actions['polar-adaptive'], false);
    await model.run('polar-fixed'); assert.equal(peer.calls.length, 0);
    assert.equal(breathingSetup(snapshot(native({ ...patch, input: 'polar-tilt-fixed', ready: true })), owner), null);
  }
});

test('builtin modes retain participant, source, recording, authorization and pending barriers', async () => {
  const polar_modes = ['calibrated', 'fixed', 'adaptive'];
  for (const op of ['polar-fixed', 'polar-adaptive']) {
    for (const [setupPatch, statusPatch, statePatch, participantReady] of [
      [{ source_clock_ready: false }, {}, {}, true], [{ stage: 'main' }, {}, {}, true],
      [{}, { rec: true, p: 'RUNNING', h: 'vr-session' }, {}, true],
      [{}, { controller_access: { v: 1, role: 'waiting', active_phone: 'Other', revision: 1 } }, {}, true],
      [{}, { pi: { r: true, f: false } }, {}, true], [{}, {}, {}, false],
      [{}, {}, { busy: true }, true], [{}, {}, { pending: true }, true], [{}, {}, { unknown: true }, true],
    ]) {
      const model = new BreathingSetup(), peer = fixture();
      const view = model.update(peer.gateway, state(snapshot(native({ polar_modes, ...setupPatch }), statusPatch), statePatch), participantReady);
      assert.equal(view.actions[op], false);
      await model.run(op); assert.equal(peer.calls.length, 0);
    }
  }
});

test('unknown builtin switches survive reload with their original ID and reconcile read-only', async () => {
  for (const [op, input] of [['polar-fixed', 'polar-tilt-fixed'], ['polar-adaptive', 'polar-tilt-adaptive']]) {
    const saved = storage(), peer = fixture(); const polar_modes = ['calibrated', 'fixed', 'adaptive'];
    let model = new BreathingSetup({ storage: saved });
    model.update(peer.gateway, state(snapshot(native({ polar_modes }))), true);
    peer.fail(true); await model.run(op);
    const id = peer.calls[0].id; model = new BreathingSetup({ storage: saved });
    const confirmed = state(snapshot(native({ polar_modes, r: 1, input, ready: true })));
    peer.reply('outcome_unknown'); model.update(peer.gateway, confirmed, true); await model.refresh();
    assert.equal(model.pending.command.id, id);
    assert.equal(model.view.actions.continue, false);
    await model.run(op); assert.equal(peer.calls.length, 1);
    model.observeReceipt({ v: 1, id: 'b'.repeat(16), state: 'confirmed' });
    assert.equal(model.pending.command.id, id, 'wrong receipt cannot resolve switch');
    model.observeReceipt({ v: 1, id, state: 'rejected', accepted: false, reason: 'source-not-ready' });
    assert.equal(model.pending, null);
    assert.equal(peer.calls.length, 1);
  }
});

test('choosing a builtin mode hides personal-fit presentation before confirmation without changing confirmed input', async () => {
  const model = new BreathingSetup(), peer = fixture(); peer.reply('pending');
  const original = state(snapshot(native({ polar_modes: ['calibrated', 'fixed', 'adaptive'],
    calibration: 'failed', failure: 'clock_stalled', attempt: 1 })));
  model.update(peer.gateway, original, true); await model.run('polar-fixed');
  const view = model.update(peer.gateway, original, true);
  assert.equal(view.inputLabel, 'Automatic'); assert.equal(view.showPolarCalibration, false);
  assert.equal(view.failureLabel, ''); assert.match(view.pendingInputLabel, /feste Achse/);
});

test('native update gate denies preparation independently of IDLE and keeps safe failure usable', async () => {
  const update = (state, session_allowed) => ({ v: 1, state, reason: '', installed_code: 15,
    available_code: 16, downloaded_bytes: 0, total_bytes: null, session_allowed });
  for (const meta_update of [null, {}, update('checking', false), update('downloading', true),
    update('installing', false), update('waiting', false), update('restart_required', true),
    update('current', false), { ...update('current', true), downloaded_bytes: -1 }]) {
    const model = new BreathingSetup(); const peer = fixture();
    const view = model.update(peer.gateway, state(snapshot(native({ stage: 'idle', playing: false }), { meta_update })), true);
    for (const op of ['calibrate', 'controller', 'continue']) assert.equal(view.actions[op], false, JSON.stringify(meta_update));
    await model.run('controller');
    assert.equal(peer.calls.length, 0);
  }
  for (const name of ['current', 'updated', 'failed']) {
    const model = new BreathingSetup(); const peer = fixture();
    assert.equal(model.update(peer.gateway, state(snapshot(native({ stage: 'idle', playing: false }), {
      meta_update: update(name, true) })), true).actions.controller, true);
  }
});

test('Quest calibration narration keeps the original command pending and offers target-bound cancellation', async () => {
  let now = 0;
  const model = new BreathingSetup({ now: () => now });
  const peer = fixture();
  let target;
  let cancelled = false;
  peer.gateway.breathSetup = async command => {
    if (command.op === 'calibrate') {
      target = command.id;
      return { v: 1, id: target, state: 'pending', accepted: true, reason: 'calibration-instruction-pending' };
    }
    assert.equal(command.op, 'cancel-instruction');
    assert.equal(command.target, target);
    assert.notEqual(command.id, target);
    cancelled = true;
    return { v: 1, id: command.id, state: 'confirmed', accepted: true };
  };
  peer.gateway.breathSetupReceipt = async id => cancelled
    ? { v: 1, id, state: 'rejected', accepted: false, reason: 'calibration-instruction-cancelled' }
    : { v: 1, id, state: 'pending', accepted: true, reason: 'calibration-instruction-pending' };
  model.update(peer.gateway, state(), true);
  await model.run('calibrate');
  await model.refresh();
  now = 35_000;
  await model.refresh();
  const view = model.update(peer.gateway, state(snapshot(native({ instruction_id: target }), { pending: true })), true);
  assert.equal(view.instructionPending, true);
  assert.equal(view.actions.cancelInstruction, true);
  assert.equal(model.waitExpired, false);
  assert.equal(model.setup.attempt, 0);
  await model.cancelInstruction();
  now += 550;
  await model.refresh();
  assert.equal(model.pending, null);
  assert.equal(model.setup.attempt, 0);
});
