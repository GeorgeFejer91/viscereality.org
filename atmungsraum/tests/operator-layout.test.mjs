import test from 'node:test';
import assert from 'node:assert/strict';
import { navigateOperatorView, resolveOperatorView, operatorCompatibilityWarning,
  restoreOperatorView, operatorSessionHeading } from '../operator-layout.js';

const connected = { connected: true, participantRequired: true, participantComplete: true };

test('verified offline archives remain reachable beside a preparation receipt without granting setup navigation', () => {
  const context = { connected: false, recovery: true, offlineArchives: true };
  assert.equal(resolveOperatorView('recordings', context), 'recordings');
  for (const step of ['input', 'session', 'running']) assert.equal(resolveOperatorView(step, context), 'connect');
  assert.equal(resolveOperatorView('recordings', { ...context, offlineArchives: false }), 'connect');
  assert.equal(resolveOperatorView('recordings', { ...context, connected: true }), 'recovery');
  assert.equal(resolveOperatorView('recordings', { ...context, connected: true, active: true }), 'running');
});
test('an actionable APK update is a preparation step, while current APKs skip it', () => {
  assert.equal(resolveOperatorView('input', { ...connected, updateRequired: true }), 'update');
  assert.equal(resolveOperatorView('input', { ...connected, updateRequired: false }), 'input');
  assert.equal(resolveOperatorView('connect', { ...connected, updateRequired: true }), 'connect');
  assert.equal(navigateOperatorView({step:'update'}, {type:'return'}).step, 'connect');
  const manual = navigateOperatorView({step:'settings',returnTo:'input'}, {type:'go',step:'update'});
  assert.equal(navigateOperatorView(manual, {type:'return'}).step, 'input');
  assert.equal(resolveOperatorView('input', { ...connected, updateRequired: true, active: true }), 'running');
  assert.equal(resolveOperatorView('settings', { ...connected, updateRequired: true }), 'settings');
  assert.equal(resolveOperatorView('update', connected), 'update');
});
test('local Back, Next and settings preserve all external experiment state', () => {
  const experiment = { participant: 'abc_23_m', acceptance: 'owner/epoch/attempt/input', journal: { id: 'original' }, draft: 'abc' };
  const before = structuredClone(experiment);
  let view = { step: 'participant' };
  view = navigateOperatorView(view, { type: 'go', step: 'input' });
  assert.equal(view.step, 'input');
  view = navigateOperatorView(view, { type: 'go', step: 'settings' });
  assert.equal(view.returnTo, 'input');
  view = navigateOperatorView(view, { type: 'return' });
  assert.equal(view.step, 'input');
  view = navigateOperatorView(view, { type: 'back' });
  assert.equal(view.step, 'participant');
  assert.deepEqual(experiment, before);
});
test('missing streams never block input choice or Controller fallback', () => {
  assert.equal(resolveOperatorView('input', { ...connected, signalsReady: false }), 'input');
});
test('native active recording and recovery override obsolete preparation and settings', () => {
  for (const requested of ['participant', 'signals', 'input', 'recordings']) {
    assert.equal(resolveOperatorView(requested, { ...connected, active: true }), 'running');
    assert.equal(resolveOperatorView(requested, { ...connected, recovery: true }), 'recovery');
  }
  assert.equal(resolveOperatorView('participant', { ...connected, participantComplete: false, recovery: true }), 'recovery');
});
test('domain acceptance admits selection after local battery review; changing input is local', () => {
  assert.equal(resolveOperatorView('session', { ...connected, selectionAllowed: false }), 'tracking');
  assert.equal(resolveOperatorView('session', { ...connected, selectionAllowed: true, batteryReviewed: true }), 'session');
  assert.equal(resolveOperatorView('input', { ...connected, selectionAllowed: true }), 'input');
});
test('waiting phone stays at connection and settings return is bounded to known step names', () => {
  assert.equal(resolveOperatorView('input', { ...connected, waitingForControl: true }), 'connect');
  assert.equal(resolveOperatorView('not-a-step', connected), 'connect');
  assert.equal(navigateOperatorView({ step: 'input' }, { type: 'go', step: 'abc_23_m' }).step, 'input');
});

test('reconnect and read-only recordings stay available without a control lease', () => {
  assert.equal(resolveOperatorView('session', { connected: false, recovery: true }), 'connect');
  assert.equal(resolveOperatorView('recordings', { connected: false }), 'recordings');
  assert.equal(resolveOperatorView('recordings', { connected: true, waitingForControl: true }), 'recordings');
});

test('stored enrollment still starts at connection before participant entry', () => {
  assert.equal(resolveOperatorView('connect', connected), 'connect');
  assert.equal(resolveOperatorView(null, { ...connected, participantComplete: false }), 'connect');
  assert.equal(resolveOperatorView(null, connected), 'connect');
});
test('utility navigation cannot create a settings and recordings return loop', () => {
  let view = navigateOperatorView({ step: 'recordings', returnTo: 'input' }, { type: 'go', step: 'settings' });
  view = navigateOperatorView(view, { type: 'go', step: 'recordings' });
  assert.equal(view.step, 'recordings');
  assert.notEqual(view.returnTo, 'settings');
});
test('native recording keeps session controls while simultaneous recovery remains required', () => {
  assert.equal(resolveOperatorView('input', { ...connected, active: true, recovery: true }), 'running');
});

test('unverified APK compatibility remains visible without adding an experiment gate', () => {
  for (const state of ['required', 'check-unavailable', 'unknown']) assert.equal(operatorCompatibilityWarning(state), true);
  for (const state of ['compatible', 'recommended', 'loading']) assert.equal(operatorCompatibilityWarning(state), false);
});

test('settings remains reachable for native diagnostics during recording and recovery', () => {
  assert.equal(resolveOperatorView('settings', { ...connected, active: true, recovery: true }), 'settings');
});

test('history restores each utility entry return target instead of retaining the next page target', () => {
  const recordings = navigateOperatorView({ step: 'input' }, { type: 'go', step: 'recordings' });
  const settings = navigateOperatorView(recordings, { type: 'go', step: 'settings' });
  assert.equal(settings.returnTo, 'recordings');
  const restored = restoreOperatorView('#recordings', { operatorStep: recordings.step, operatorReturnTo: recordings.returnTo });
  assert.equal(navigateOperatorView(restored, { type: 'return' }).step, 'input');
  assert.deepEqual(restoreOperatorView('#input', { operatorStep: 'settings', operatorReturnTo: 'recordings' }), { step: 'input', returnTo: null });
  assert.equal(navigateOperatorView({ step: 'recordings', returnTo: 'recordings' }, { type: 'return' }).step, 'signals');
  assert.deepEqual(restoreOperatorView('#private-identity', {}), { step: null, returnTo: null });
});

test('native paused and finalizing states cannot be labelled as running', () => {
  assert.equal(operatorSessionHeading('RUNNING'), 'Sitzung läuft');
  assert.equal(operatorSessionHeading('PAUSED'), 'Sitzung pausiert');
  assert.equal(operatorSessionHeading('FINALIZING'), 'Sitzung wird beendet');
  assert.equal(operatorSessionHeading('SAVING'), 'Aufnahme wird gespeichert');
});
test('battery review precedes Session without authorizing native commands or blocking on optional telemetry', () => {
  const ready = { ...connected, selectionAllowed: true, batteryReviewed: false };
  assert.equal(resolveOperatorView('session', ready), 'battery');
  assert.equal(resolveOperatorView('battery', ready), 'battery');
  assert.equal(resolveOperatorView('session', { ...ready, batteryReviewed: true }), 'session');
  assert.equal(resolveOperatorView('battery', { ...ready, selectionAllowed: false }), 'tracking');
  assert.equal(navigateOperatorView({ step: 'session' }, { type: 'back' }).step, 'battery');
  assert.equal(navigateOperatorView({ step: 'battery' }, { type: 'back' }).step, 'tracking');
  assert.equal(resolveOperatorView('battery', { ...ready, active: true }), 'running');
  assert.equal(resolveOperatorView('battery', { ...ready, recovery: true }), 'recovery');
});
