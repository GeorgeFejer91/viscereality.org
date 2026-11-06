import assert from 'node:assert/strict';
import { state, choices, hasSignal, setSensor, route, start, reset, confirmECG, updateReconnect } from '../../ui-logic.js';
import * as ui from '../../ui-logic.js';
const routes = ['h10', 'patient', 'runde', 'probe-runde', 'haupt-runde', ...Object.entries(choices).flatMap(([branch, values]) => values.map((value) => `start-${branch}-${value}`))];
reset();
for (const name of routes) assert.equal(route(name), 'patient', 'patient entry must precede the ECG gate, including cold deep links');
assert.equal(start(), false);
setSensor('ecg');
for (const name of routes) assert.equal(route(name), 'patient', 'exercise routes must collect patient details first');
assert.equal(start(), false, 'ECG alone must not permit playback');
const patient = { surnamePrefix: 'Mül', age: '42', gender: 'female' };
for (const invalid of [
  { surnamePrefix: '' }, { surnamePrefix: 'Mu' }, { surnamePrefix: 'Müller' },
  { surnamePrefix: 'M1l' }, { surnamePrefix: 'M-l' }, { age: '' }, { age: ' ' },
  { age: '-1' }, { age: '4.5' }, { age: 'NaN' }, { age: 'Infinity' },
  { gender: '' }, { gender: 'invalid' },
]) {
  assert.equal(ui.confirmPatient({ ...patient, ...invalid }), false);
  assert.equal(route('start-haupt-15'), 'patient', 'invalid input must leave the gate locked');
}
for (const gender of ['male', 'female', 'other', 'prefer-not-to-say']) {
  assert.equal(ui.confirmPatient({ ...patient, surnamePrefix: ' Mu\u0308l ', gender }), true);
  assert.deepEqual(state.patient, { surnamePrefix: 'Mül', age: 42, gender });
}
assert.equal(ui.confirmPatient({ ...patient, age: '0' }), true);
assert.equal(state.patient.age, 0);
assert.equal(ui.confirmPatient(patient), true);
for (const signal of ['searching', 'found', 'connecting', 'no-ecg', 'disconnected']) {
  setSensor(signal);
  assert.equal(hasSignal(), false);
  assert.equal(route('runde'), 'h10', 'valid details do not bypass connection');
}
setSensor('ecg');
assert.equal(route('runde'), 'runde');
assert.equal(route('patient'), 'patient', 'details remain editable');
for (const [branch, values] of Object.entries(choices)) {
  assert.deepEqual(values, branch === 'probe' ? ['anleitung', '5', '10', '15'] : ['15']);
  for (const value of values) {
    assert.equal(route(`start-${branch}-${value}`), 'session-start');
    const minutes = value === 'anleitung' ? null : Number(value);
    assert.deepEqual(state.selection, { branch, value, minutes, boxMinutes: minutes });
    assert.equal(start(), true);
    assert.equal(start(), false, 'a double click must not start twice');
    setSensor('no-ecg');
    assert.equal(state.started, false);
    assert.equal(route(`start-${branch}-${value}`), 'h10');
    assert.equal(start(), false);
    setSensor('ecg');
  }
}
for (const invalid of ['start-haupt-10', 'start-probe-20', 'start-haupt-5', 'start-haupt-20', 'start-haupt-anleitung', 'start-probe-frei', 'session-start', 'unknown']) assert.equal(route(invalid), 'h10');
reset();
assert.equal(state.selection, null);
assert.equal(state.patient, null, 'reset must clear the patient');
assert.equal(confirmECG(), false);
assert.equal(ui.confirmPatient(patient), true, 'details are submitted before connection');
assert.equal(route('runde'), 'h10');
setSensor('ecg');
assert.equal(confirmECG(), true);
route('start-haupt-15'); start();
setSensor('disconnected', 1000);
assert.equal(state.started, true, 'a short loss must not abort the current session');
for (let now = 1000; now <= 31000; now += 5000) {
  assert.equal(updateReconnect(now), 'attempt');
  assert.equal(route('start-haupt-15'), 'session-start', 'stay in the current flow through exactly 30 seconds');
}
assert.equal(state.reconnectAttempts, 7);
assert.equal(updateReconnect(31000), null);
assert.equal(updateReconnect(31001), 'expired');
assert.equal(route('start-haupt-15'), 'h10');
assert.equal(state.started, false);
assert.deepEqual(state.patient, { surnamePrefix: 'Mül', age: 42, gender: 'female' }, 'reconnects retain details within the page session');
setSensor('ecg'); confirmECG();
setSensor('no-ecg', 40000);
assert.equal(updateReconnect(40000), 'attempt');
setSensor('ecg', 69999);
assert.equal(updateReconnect(80000), null, 'recovery cancels the loss deadline and retry counter');
assert.equal(route('runde'), 'runde');
assert.equal(state.lostSince, null);
assert.equal(state.reconnectAttempts, 0);
reset();
console.log('PASS: sensor and patient gates, surname/age/four-gender validation, editing/reset, five session routes, direct 15-minute main round, duplicate-start guard, 30-second reconnection grace and recovery.');
