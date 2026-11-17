import test from 'node:test';
import assert from 'node:assert/strict';
import { MetaUpdateHandoff } from '../meta-update-handoff.js';

const owner = { device: 'a'.repeat(32), serial: 'TEST', epoch: '7', native_epoch: '8', identity_kind: 'app-installation' };
const packageName = 'org.viscereality.atmungsraum.developer';
const hash = 'b'.repeat(64);
const detail = 'Fortschritt auf diesem Smartphone nicht verfügbar. Prüfe das Update im Headset. Sobald Atmungsraum dort wieder geöffnet ist, verbinde dieses Smartphone erneut.';
function storage() { const data = new Map(); return { data, getItem: k => data.get(k) ?? null,
  setItem: (k, v) => data.set(k, v), removeItem: k => data.delete(k) }; }
const gateway = (patch = {}) => ({ owner: { ...owner, ...patch }, metaUpdate() { throw Error('No actions permitted'); }, command() { throw Error('No actions permitted'); } });
function state(patch = {}, { code = 20, epoch = '8', observation = '100', status = {}, local = {} } = {}) {
  return { status: { p: 'IDLE', rec: false, pending: false, meta_update_control: 1,
    o: { v: 1, s: 'observed', e: epoch, i: observation, a: 5 },
    build: { v: 1, packaging_fingerprint: hash },
    apk: { v: 1, package_name: packageName, version_code: code, version_name: '0.2.19-dev', build_id: hash },
    meta_update: { v: 2, state: 'installing', reason: 'queued-meta-update-handoff', installed_code: code,
      available_code: 21, required_code: 21, downloaded_bytes: 0, total_bytes: null, session_allowed: false,
      cycle: '2', revision: '7', attempt: '19', can_check: false, busy_reason: '', ...patch }, ...status },
    busy: false, pending: false, unknown: false, ...local };
}
const current = (options = {}) => state({ state: 'current', reason: '', installed_code: options.code ?? 21,
  required_code: options.code ?? 21, available_code: options.code ?? 21, session_allowed: true },
{ code: 21, observation: '101', ...options });

test('confirmed close hint survives loss and reload until fresh same Quest higher APK is ready', () => {
  const saved = storage(), first = new MetaUpdateHandoff({ storage: saved });
  assert.equal(first.observe(gateway(), state()), true);
  assert.equal(first.view(owner), null, 'native connected UI owns presentation');
  first.disconnected('connection-lost');
  assert.equal(first.view(owner).confirmedHandoff, true);
  assert.equal(first.view(owner).progress, null);
  assert.equal(first.view(owner).detail, detail);
  const restored = new MetaUpdateHandoff({ storage: saved });
  assert.equal(restored.view(owner).targetCode, 21);
  const newer = gateway({ epoch: '9', native_epoch: '10' });
  assert.equal(restored.observe(newer, current({ epoch: '10', observation: '1' })), true);
  assert.equal(restored.view(owner), null);
  assert.equal(saved.data.size, 0);
});

test('unconfirmed checking/download/request losses never claim installation or invent progress', () => {
  for (const last of ['checking', 'downloading', 'installing']) {
    const model = new MetaUpdateHandoff();
    model.observe(gateway(), state({ state: last, reason: '', downloaded_bytes: 55, total_bytes: 100 }));
    model.disconnected('secret diagnostic transport text');
    const view = model.view(owner);
    assert.equal(view.confirmedHandoff, false);
    assert.equal(view.lastState, last);
    assert.equal(view.progress, null);
    assert.match(view.summary, /unbekannt/);
    assert.doesNotMatch(JSON.stringify(view), /55|100|secret|ETA|%/);
  }
  const model = new MetaUpdateHandoff();
  assert.equal(model.manualRequest(gateway(), current({ code: 20, observation: '100' })), true);
  model.disconnected();
  assert.equal(model.view(owner).lastState, 'requested');
  assert.equal(model.view(owner).confirmedHandoff, false);
});

test('only exact fresh native installing handoff reasons confirm transfer', () => {
  for (const reason of ['queued-meta-update-handoff', 'installing-meta-channel-update']) {
    const model = new MetaUpdateHandoff(); model.observe(gateway(), state({ reason })); model.disconnected();
    assert.equal(model.view(owner).confirmedHandoff, true);
  }
  for (const candidate of [state({ state: 'checking' }), state({}, { status: { o: { ...state().status.o, a: 1501 } } }),
    state({}, { status: { apk: undefined } }), state({}, { status: { build: { v: 1, packaging_fingerprint: 'c'.repeat(64) } } })]) {
    const model = new MetaUpdateHandoff(); model.observe(gateway(), candidate); model.disconnected();
    assert.notEqual(model.view(owner)?.confirmedHandoff, true);
  }
});

test('lower/failed/stale/wrong identities/diagnostic SUCCESS and blocked readiness cannot complete', () => {
  const variants = [
    [gateway(), current({ code: 20 })],
    [gateway(), current({ status: { meta_update: { ...current().status.meta_update, state: 'failed' } } })],
    [gateway(), current({ status: { o: { ...current().status.o, a: 1501 } } })],
    [gateway(), current({ observation: '100' })],
    [gateway({ native_epoch: '10' }), current()],
    [gateway({ device: 'c'.repeat(32) }), current()],
    [gateway({ serial: 'OTHER' }), current()],
    [gateway(), current({ status: { apk: { ...current().status.apk, package_name: 'org.other.app' } } })],
    [gateway(), current({ status: { meta_update: { ...current().status.meta_update, sdk_error: { v: 1, stage: 5, code: 7, http: null, install_result: 'SUCCESS' } } } })],
    [gateway(), current({ status: { rec: true } })],
    [gateway(), current({ status: { p: 'RUNNING' } })],
    [gateway(), current({ local: { unknown: true } })],
    [gateway(), current({ status: { pending: true } })],
  ];
  for (const [peer, observed] of variants) {
    const saved = storage(), model = new MetaUpdateHandoff({ storage: saved });
    model.observe(gateway(), state()); model.disconnected();
    model.observe(peer, observed); model.disconnected();
    assert.notEqual(model.view(owner), null, JSON.stringify(observed));
    assert.equal(new MetaUpdateHandoff({ storage: saved }).view(owner).targetCode, 21);
  }
});

test('same-epoch lineage cannot regress or replay; remembered target never lowers', () => {
  const model = new MetaUpdateHandoff(); model.observe(gateway(), state());
  assert.equal(model.observe(gateway(), state({ revision: '6' }, { observation: '101' })), false);
  assert.equal(model.observe(gateway(), state({ cycle: '1' }, { observation: '102' })), false);
  model.observe(gateway(), state({ state: 'checking', reason: '', required_code: 20, available_code: 20 }, { observation: '103' }));
  model.disconnected(); assert.equal(model.view(owner).targetCode, 21);
});

test('manual accepted context is read-only and bound to the original native cycle', () => {
  const model = new MetaUpdateHandoff();
  const receipt = { v: 1, id: 'a'.repeat(16), state: 'accepted', cycle: '2', revision: '8', attempt: '22', reason: '' };
  assert.equal(model.manualRequest(gateway(), current({ code: 20, observation: '100' }), { ...receipt, cycle: '9' }), false);
  assert.equal(model.manualRequest(gateway(), current({ code: 20, observation: '100' }), receipt), true);
  model.disconnected(); assert.equal(model.view(owner).lastState, 'requested');
  assert.equal(model.observe(gateway(), state({ state: 'checking', reason: '', revision: '7' }, { observation: '101' })), false);
});

test('stored schema is closed, contains no status/private payload, and selection mismatch leaks nothing', () => {
  const saved = storage(), model = new MetaUpdateHandoff({ storage: saved });
  model.observe(gateway(), state({}, { status: { enrollment: { key: 'SECRET' }, ecg: ['PARTICIPANT'], recording: 'PRIVATE' } }));
  model.disconnected();
  const [[key, raw]] = [...saved.data];
  assert.doesNotMatch(raw, /SECRET|PARTICIPANT|PRIVATE|enrollment|ecg|recording|downloaded_bytes/);
  assert.equal(model.view({ ...owner, device: 'f'.repeat(32) }), null);
  saved.setItem(key, JSON.stringify({ ...JSON.parse(raw), command: { op: 'install' } }));
  assert.equal(new MetaUpdateHandoff({ storage: saved }).view(owner), null);
});

test('no elapsed time, reload, or unavailable storage finishes a pending handoff', () => {
  const saved = storage(), model = new MetaUpdateHandoff({ storage: saved });
  model.observe(gateway(), state()); model.disconnected();
  const raw = [...saved.data.values()][0];
  for (let i = 0; i < 30; i++) assert.equal(new MetaUpdateHandoff({ storage: saved }).view(owner).confirmedHandoff, true);
  assert.equal([...saved.data.values()][0], raw);
  const broken = new MetaUpdateHandoff({ storage: { getItem() { throw Error(); }, setItem() { throw Error(); } } });
  broken.observe(gateway(), state()); broken.disconnected(); assert.equal(broken.view(owner).confirmedHandoff, true);
});

test('a disconnected saved selection needs only the exact device and serial; journal remains independent', () => {
  const saved = storage(); saved.setItem('atmungsraum-meta-update-pending', 'INDEPENDENT_COMMAND_JOURNAL');
  const model = new MetaUpdateHandoff({ storage: saved }); model.observe(gateway(), state()); model.disconnected();
  assert.equal(model.view({ device: owner.device, serial: owner.serial }).confirmedHandoff, true);
  model.observe(gateway(), current());
  assert.equal(saved.getItem('atmungsraum-meta-update-pending'), 'INDEPENDENT_COMMAND_JOURNAL');
});

test('installing without native attempt lineage cannot confirm a headset handoff', () => {
  const model = new MetaUpdateHandoff(); model.observe(gateway(), state({ attempt: null })); model.disconnected();
  assert.equal(model.view(owner).confirmedHandoff, false);
});

test('known target can rise but a later lower native floor never completes it', () => {
  const model = new MetaUpdateHandoff(); model.observe(gateway(), state());
  model.observe(gateway(), state({ state: 'downloading', reason: '', required_code: 22, available_code: 22 }, { observation: '101' }));
  model.observe(gateway(), current({ observation: '102' })); model.disconnected();
  assert.equal(model.view(owner).targetCode, 22);
});
