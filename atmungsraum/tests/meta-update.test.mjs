import test from 'node:test';
import assert from 'node:assert/strict';
import { nativeMetaUpdate, nativeSessionAllowed, metaUpdateView, updatePresentation } from '../meta-update.js';

const value = (state = 'current', patch = {}) => ({ v: 1, state, reason: '', installed_code: 15,
  available_code: 16, downloaded_bytes: 0, total_bytes: null, session_allowed: true, ...patch });
const status = meta_update => ({ meta_update, o: { v: 1, s: 'observed', e: '8', i: '100', a: 5 } });

const cycle = (patch = {}) => ({ ...status(value('current', { v: 2, available_code: 15,
  required_code: 15, cycle: '2', revision: '7', attempt: '19', can_check: true, busy_reason: '', ...patch })),
  meta_update_control: 1 });

test('an actionable required update has one status and its primary action, while specific blockers stay visible', () => {
  const version = { state: 'required' };
  const controls = { available: true, canRequest: true, requestOp: 'check-and-update' };
  const required = metaUpdateView(cycle({ state: 'update_required', reason: 'native-owner-retired',
    available_code: 16, required_code: 16, session_allowed: false }));
  assert.deepEqual(updatePresentation(version, required, controls, false), {
    summary: 'Update der Quest-App erforderlich.', instruction: '', checkLabel: 'Jetzt aktualisieren' });
  const unavailable = metaUpdateView(cycle({ state: 'update_required', reason: 'required-update-not-yet-on-channel',
    installed_code: 25, available_code: 26, required_code: 27, session_allowed: false }));
  assert.deepEqual(updatePresentation(version, unavailable, controls, false), {
    summary: 'Meta bietet das benötigte Update noch nicht an.', instruction: '', checkLabel: 'Erneut prüfen' });
  for (const prerequisites of ['Übernimm zuerst die Steuerung.', 'Eine frühere Sitzungsanfrage ist unklar.']) {
    assert.equal(updatePresentation(version, required, { ...controls, canRequest: false, prerequisites }, false).instruction, prerequisites);
  }
  const maintenance = metaUpdateView(cycle({ state: 'restart_required', reason: 'prior-install-unconfirmed-maintenance-required' }));
  const pending = updatePresentation(version, maintenance, { ...controls, canRequest: false, message: 'Ursprüngliche Bestätigung offen.' }, true);
  assert.match(pending.instruction, /Ursprüngliche Bestätigung.*Meta-Bibliothek/);
});

test('only a native narration release wait shows the headset app recovery steps', () => {
  const held = { state: 'checking', reason: 'ending-preparation', busy_reason: 'narration-park-pending',
    session_allowed: false, can_check: false };
  const waiting = metaUpdateView(cycle(held));
  assert.equal(waiting.summary, 'Anleitung wird beendet.');
  assert.match(waiting.detail, /Bleibt diese Meldung stehen, schließe Atmungsraum im Headset/);
  assert.match(waiting.detail, /öffne die App erneut und verbinde das Smartphone wieder/);
  assert.equal(waiting.allowed, false);
  assert.equal(waiting.progress, null);
  const loop = metaUpdateView(cycle({ ...held, busy_reason: 'setup-park-pending' }));
  assert.match(loop.detail, /Atemtakt.*Ruheposition/);
  for (const patch of [{ reason: '' }, { state: 'installing' }, { state: 'failed' }]) {
    const other = metaUpdateView(cycle({ ...held, ...patch }));
    assert.notEqual(other.summary, waiting.summary);
    assert.doesNotMatch(other.detail, /schließe Atmungsraum/);
  }
});

test('SDK failure provenance is bounded and never grants an update or session', () => {
  const error = { v: 1, stage: 3, code: 1971054, http: null, install_result: 'DUPLICATE_REQUEST' };
  const required = patch => cycle({ state: 'update_required', required_code: 20, available_code: 20,
    session_allowed: false, sdk_error: { ...error, ...patch } });
  assert.equal(nativeSessionAllowed(required()), false);
  assert.match(metaUpdateView(required()).detail, /bereits.*Anfrage/);
  assert.equal(nativeSessionAllowed(required({ install_result: 'SUCCESS' })), false);
  for (const patch of [{ v: 2 }, { stage: 0 }, { stage: 7 }, { code: 0 }, { code: 2147483648 },
    { code: '7' }, { http: 0 }, { http: 600 }, { install_result: 'raw error text' }, { raw_message: 'secret' }]) {
    assert.equal(metaUpdateView(required(patch)).state, 'invalid');
    assert.equal(nativeSessionAllowed(required(patch)), false);
  }
  assert.equal(metaUpdateView(cycle({ sdk_error: null })).state, 'current');
  for (const reason of ['checking-duplicate-download', 'restarting-duplicate-download']) {
    const recovering = metaUpdateView(cycle({ state: 'checking', reason,
      session_allowed: false, sdk_error: error }));
    assert.equal(recovering.allowed, false);
    assert.match(recovering.summary, /vorgemerkten|erneut/);
  }
  const queued = metaUpdateView(cycle({ state: 'installing', reason: 'queued-meta-update-handoff',
    session_allowed: false, sdk_error: error, required_code: 20 }));
  assert.equal(queued.allowed, false);
  assert.match(queued.summary, /schließt die App/);
  assert.doesNotMatch(queued.summary, /Installation angefragt/);
  assert.match(queued.detail, /überwacht den installierten App-Stand/);
});

test('repeatable native update status requires a fresh cycle and preserves known required versions', () => {
  assert.equal(nativeSessionAllowed(cycle()), true);
  assert.equal(metaUpdateView(cycle()).value.cycle, '2');
  for (const patch of [{ cycle: '0' }, { revision: 7 }, { attempt: '' }, { required_code: -1 },
    { can_check: 1 }, { busy_reason: 'x'.repeat(97) }]) {
    assert.equal(metaUpdateView(cycle(patch)).state, 'invalid');
    assert.equal(nativeSessionAllowed(cycle(patch)), false);
  }
  assert.equal(nativeSessionAllowed(cycle({ required_code: 18 })), false);
  assert.equal(nativeSessionAllowed(cycle({ available_code: 18 })), false);
  const required = metaUpdateView(cycle({ state: 'update_required', required_code: 18,
    available_code: 17, session_allowed: false, reason: 'required-update-not-yet-on-channel' }));
  assert.equal(required.allowed, false);
  assert.match(required.summary, /Update.*erforderlich/);
  assert.match(required.detail, /18/);
  assert.match(required.detail, /Meta.*17/);
});

test('selected repeatable updater cannot become legacy through absent status; offline warning is conditional', () => {
  const missing = metaUpdateView({ o: status().o, meta_update_control: 1 });
  assert.equal(missing.state, 'invalid');
  assert.equal(missing.allowed, false);
  const warn = metaUpdateView(cycle({ state: 'failed', reason: 'version-check-timeout' }));
  assert.equal(warn.allowed, true);
  assert.match(warn.detail, /Updates prüfen und installieren/);
  assert.doesNotMatch(warn.detail, /Quest neu starten/);
  const known = metaUpdateView(cycle({ state: 'failed', required_code: 18, available_code: 18 }));
  assert.equal(known.allowed, false);
});

test('advertised repeatable capability cannot downgrade to a version-one warning or malformed legacy fallback', () => {
  for (const state of ['current', 'failed']) {
    const mixed = { ...status(value(state, { installed_code: 18, available_code: 20, reason: 'offline' })), meta_update_control: 1 };
    assert.equal(metaUpdateView(mixed).state, 'invalid');
    assert.equal(nativeSessionAllowed(mixed), false);
    assert.equal(nativeSessionAllowed(status(value(state))), true);
  }
  for (const capability of [0, 2, '1', null]) {
    assert.equal(nativeSessionAllowed({ o: status().o, meta_update_control: capability }), false);
    assert.equal(nativeSessionAllowed({ ...status(value()), meta_update_control: capability }), false);
  }
});

test('malformed JSON numeric fields return blocked status without coercion exceptions', () => {
  for (const field of ['required_code', 'available_code', 'downloaded_bytes', 'total_bytes']) {
    const malformed = cycle({ [field]: JSON.parse('{"toString":null}') });
    assert.doesNotThrow(() => metaUpdateView(malformed));
    assert.equal(metaUpdateView(malformed).state, 'invalid');
    assert.equal(nativeSessionAllowed(malformed), false);
  }
});

test('process initialization failure explains the specific restart exception to manual retries', () => {
  const view = metaUpdateView(cycle({ state: 'failed', reason: 'meta-platform-process-init-failed-restart-required' }));
  assert.equal(view.allowed, true);
  assert.match(view.detail, /Quest.*neu starten/);
  assert.doesNotMatch(view.detail, /Wähle „Updates prüfen/);
});

test('read-only Meta status distinguishes all native states and absent legacy feature', () => {
  assert.equal(nativeSessionAllowed({}), true);
  assert.equal(metaUpdateView({}).state, 'legacy');
  for (const state of ['waiting', 'checking', 'downloading', 'installing', 'restart_required']) {
    assert.equal(nativeSessionAllowed(status(value(state))), false, state);
    assert.match(metaUpdateView(status(value(state))).summary, /gesperrt/);
  }
  for (const state of ['current', 'updated', 'failed']) {
    assert.equal(nativeSessionAllowed(status(value(state))), true, state);
    assert.equal(nativeSessionAllowed(status(value(state, { session_allowed: false }))), false, state);
    assert.match(metaUpdateView(status(value(state, { session_allowed: false }))).summary, /gesperrt/);
  }
});

test('malformed advertised native status and stale observations fail closed', () => {
  for (const malformed of [null, [], {}, value('invented'), value('current', { v: 2 }),
    value('current', { session_allowed: 1 }), value('current', { installed_code: 0 }),
    value('current', { available_code: -1 }), value('current', { downloaded_bytes: -1 }),
    value('current', { downloaded_bytes: Number.MAX_SAFE_INTEGER + 1 }),
    value('current', { total_bytes: 0 }), value('current', { total_bytes: 2, downloaded_bytes: 3 }),
    value('current', { reason: 'x'.repeat(97) }), value('current', { reason: 'line\nfeed' })]) {
    const view = metaUpdateView(status(malformed));
    assert.equal(view.state, 'invalid', JSON.stringify(malformed));
    assert.equal(view.allowed, false);
  }
  for (const o of [null, { ...status().o, a: 1501 }, { ...status().o, s: 'unavailable' }]) {
    assert.equal(nativeSessionAllowed({ ...status(value()), o }), false);
  }
  assert.equal(nativeSessionAllowed({ ...status(value()), apk: { version_code: 14 } }), false);
});

test('download progress uses native byte counts and leaves unknown totals indeterminate', () => {
  const view = metaUpdateView(status(value('downloading', { downloaded_bytes: 1024 })));
  assert.deepEqual(view.progress, { value: 1024, max: null });
  assert.match(view.detail, /1.024 Byte.*unbekannt/);
  const known = metaUpdateView(status(value('downloading', { downloaded_bytes: 1024, total_bytes: 2048 })));
  assert.deepEqual(known.progress, { value: 1024, max: 2048 });
  assert.match(known.detail, /1.024.*2.048/);
});

test('safe failure and uncertain install have distinct recovery without claiming installed target', () => {
  const failed = metaUpdateView(status(value('failed', { reason: 'download-timeout' })));
  assert.equal(failed.allowed, true);
  assert.match(failed.summary, /installierte APK.*verwendet/);
  assert.match(failed.detail, /Quest neu starten/);
  assert.match(failed.detail, /zu lange/);
  assert.doesNotMatch(failed.detail, /download-timeout/);
  const install = metaUpdateView(status(value('installing', { session_allowed: false })));
  assert.match(install.summary, /angefragt/);
  assert.match(install.detail, /bestätigt keine neue APK/);
  const updated = metaUpdateView(status(value('updated')));
  assert.match(updated.summary, /Code 15/);
  assert.doesNotMatch(updated.summary, /16/);
  assert.equal(nativeMetaUpdate(status(value())).value.installed_code, 15);
});

test('native failure reasons use readable wording and hide unknown technical codes', () => {
  for (const [reason, expected] of [['offline', /Internetverbindung/], ['storage', /Speicherplatz/],
    ['authentication', /Meta-Konto/], ['mystery_' + 'x'.repeat(70), /erneute Prüfung/]]) {
    const view = metaUpdateView(status(value('failed', { reason })));
    assert.match(view.detail, expected);
    assert.equal(view.detail.includes(reason), false);
  }
});

test('persisted unconfirmed install requires Meta library and operator maintenance, not another retry', () => {
  for (const reason of ['prior-install-unconfirmed-maintenance-required', 'pending-install-record-invalid',
    'pending-install-record-unreadable', 'pending-install-record-clear-failed']) {
    const view = metaUpdateView(status(value('restart_required', { reason, session_allowed: false })));
    assert.equal(view.allowed, false);
    assert.match(view.detail, /Meta-Bibliothek/);
    assert.match(view.detail, /Betreiberwartung/);
    assert.doesNotMatch(view.summary, /neu starten/);
  }
});

test('a warm native app relaunch requires a headset restart for another check', () => {
  const view = metaUpdateView(status(value('failed', { reason: 'fresh-process-required-for-update-check' })));
  assert.equal(view.allowed, true);
  assert.match(view.detail, /Quest neu starten/);
  assert.match(view.detail, /reicht dafür nicht aus/);
});
