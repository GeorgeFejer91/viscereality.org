import test from 'node:test';
import assert from 'node:assert/strict';
import { OperatorDeviceHealth } from '../operator-device-health.js';

const owner = { device: 'd'.repeat(32), serial: '', epoch: '7', native_epoch: '8' };
const health = patch => ({ v: 1, link: true, connection_generation: '1', observed_at_ns: '1000000000',
  ecg: { age_ms: 0, fresh: true }, acc: { age_ms: 0, fresh: true },
  contact: { state: 'detected', supported: true, detected: true, age_ms: 0 },
  battery: { percent: 73, age_ms: 0 }, ...patch });
const state = (polar = health(), power = { v: 1, quest: { percent: 82, charging: true }, controllers: [] }, patch = {}) => ({
  status: { p: 'IDLE', pi: { f: true }, o: { e: '8', a: 10 }, polar_health: polar, power, ...patch },
  statusTiming: { receivedAt: 1000, transportMs: 20 },
});

test('fresh contact plus linked ECG and motion confirms ongoing Polar health in active sessions', () => {
  const model = new OperatorDeviceHealth({ now: () => 1000 });
  for (const p of ['IDLE', 'ARMED', 'RUNNING', 'PAUSED']) {
    const view = model.update(owner, state(health(), undefined, { p }));
    assert.equal(view.polar.confirmed, true);
    assert.match(view.polar.detail, /Tragekontakt.*bestätigt/);
    assert.equal(view.battery.quest, '82 % · lädt');
    assert.equal(view.battery.polar, '73 %');
    assert.match(view.battery.controllers, /keine Akkudaten/i);
  }
});

test('contact is never inferred from sample freshness or legacy pi.f', () => {
  const model = new OperatorDeviceHealth({ now: () => 1000 });
  assert.equal(model.update(owner, state(undefined)).polar.confirmed, true);
  const missing = state(); delete missing.status.polar_health;
  assert.equal(model.update(owner, missing).polar.confirmed, null);
  assert.match(model.update(owner, missing).polar.detail, /nicht gemeldet/);
  for (const contact of [
    { state: 'unknown', supported: null, detected: null, age_ms: null },
    { state: 'unsupported', supported: false, detected: null, age_ms: 0 },
    { state: 'stale', supported: true, detected: true, age_ms: 5001 },
  ]) assert.equal(model.update(owner, state(health({ contact }))).polar.confirmed, null);
});

test('loss, stale streams and absent contact remove green, and a fresh native observation can recover', () => {
  const model = new OperatorDeviceHealth({ now: () => 1000 });
  assert.equal(model.update(owner, state()).polar.confirmed, true);
  for (const patch of [{ link: false }, { ecg: { age_ms: 0, fresh: false } },
    { acc: { age_ms: 5001, fresh: true } },
    { contact: { state: 'not-detected', supported: true, detected: false, age_ms: 0 } }]) {
    assert.equal(model.update(owner, state(health(patch))).polar.confirmed, false);
  }
  assert.equal(model.update(owner, state(health({ observed_at_ns: '1000000001' }))).polar.confirmed, true);
});

test('native observation age, transport and elapsed phone time expire health and battery without polls', () => {
  let now = 1000;
  const model = new OperatorDeviceHealth({ now: () => now });
  const observed = state(health({ ecg: { age_ms: 4900, fresh: true } }));
  assert.equal(model.update(owner, observed).polar.confirmed, true);
  now = 1071;
  assert.equal(model.update(owner, observed).polar.confirmed, false);
  now = 6000;
  const expired = model.update(owner, observed);
  assert.equal(expired.battery.quest, 'unbekannt');
  assert.equal(expired.battery.polar, 'unbekannt');
});

test('repeated health witnesses cannot renew stale fields, and old connection data cannot return green', () => {
  let now = 1000;
  const model = new OperatorDeviceHealth({ now: () => now });
  assert.equal(model.update(owner, state()).polar.confirmed, true);
  now = 6100;
  const repeated = state(); repeated.statusTiming.receivedAt = now;
  assert.notEqual(model.update(owner, repeated).polar.confirmed, true);
  const newer = state(health({ connection_generation: '2', observed_at_ns: '1000000002' })); newer.statusTiming.receivedAt = now;
  assert.equal(model.update(owner, newer).polar.confirmed, true);
  assert.notEqual(model.update(owner, state()).polar.confirmed, true);
});

test('battery facts are optional, never invented, and not reused across a lost Polar link', () => {
  const model = new OperatorDeviceHealth({ now: () => 1000 });
  const missing = state(null, undefined); delete missing.status.power;
  const view = model.update(owner, missing);
  assert.equal(view.battery.quest, 'unbekannt');
  assert.equal(view.battery.polar, 'unbekannt');
  assert.match(view.battery.controllers, /keine Akkudaten/i);
  const optional = state(health({ link: false }), { v: 1, quest: { percent: 0, charging: false },
    controllers: [{ name: 'Links', percent: null }, { name: 'Rechts', percent: 12 }] });
  const facts = model.update(owner, optional);
  assert.equal(facts.battery.quest, '0 %');
  assert.equal(facts.battery.controllers, 'Controller 1: unbekannt\nController 2: 12 %');
  assert.equal(facts.battery.polar, 'nicht verbunden');
  assert.equal(model.update(owner, state(health({ battery: { percent: 73, age_ms: 70000 } }))).battery.polar, '73 %');
  assert.equal(model.update(owner, state(health({ battery: { percent: 73, age_ms: 120000 } }))).battery.polar, 'unbekannt');
});

test('foreign, malformed or unavailable observations cannot claim health or battery', () => {
  const model = new OperatorDeviceHealth({ now: () => 1000 });
  const malformed = state(health({ ecg: { age_ms: null, fresh: 'true' } }), { v: 1, quest: { percent: 101, charging: false }, controllers: [] });
  assert.equal(model.update(owner, malformed).polar.confirmed, null);
  assert.equal(model.update(owner, malformed).battery.quest, 'unbekannt');
  assert.equal(model.update({ ...owner, native_epoch: '9' }, state()).polar.confirmed, null);
  assert.equal(model.update(owner, { status: null }).polar.confirmed, null);
});
