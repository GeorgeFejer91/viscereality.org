import assert from 'node:assert/strict';
import { state, choices, hasSignal, setSensor, route, start, reset, confirmECG, updateReconnect } from '../../ui-logic.js';
const routes = ['runde', 'probe-runde', 'haupt-runde', ...Object.entries(choices).flatMap(([branch, values]) => values.map((value) => `start-${branch}-${value}`))];
reset();
for (const name of routes) assert.equal(route(name), 'h10', 'cold deep links must respect the ECG gate');
assert.equal(start(), false);
for (const signal of ['searching', 'found', 'connecting', 'no-ecg', 'disconnected']) {
  setSensor(signal);
  assert.equal(hasSignal(), false);
  assert.equal(route('runde'), 'h10');
}
setSensor('ecg');
assert.equal(route('runde'), 'runde');
for (const [branch, values] of Object.entries(choices)) {
  assert.deepEqual(values, branch === 'probe' ? ['anleitung', '5', '10', '15'] : ['10', '15']);
  for (const value of values) {
    assert.equal(route(`start-${branch}-${value}`), 'session-start');
    const minutes = value === 'anleitung' ? null : Number(value);
    const baselineMinutes = branch === 'haupt' ? 5 : 0;
    assert.deepEqual(state.selection, { branch, value, minutes, baselineMinutes,
      boxMinutes: minutes === null ? null : minutes - baselineMinutes });
    assert.equal(start(), true);
    assert.equal(start(), false, 'a double click must not start twice');
    setSensor('no-ecg');
    assert.equal(state.started, false);
    assert.equal(route(`start-${branch}-${value}`), 'h10');
    assert.equal(start(), false);
    setSensor('ecg');
  }
}
for (const invalid of ['start-probe-20', 'start-haupt-5', 'start-haupt-20', 'start-haupt-anleitung', 'start-probe-frei', 'session-start', 'unknown']) assert.equal(route(invalid), 'h10');
reset();
assert.equal(state.selection, null);
assert.equal(confirmECG(), false);
setSensor('ecg');
assert.equal(confirmECG(), true);
route('start-haupt-10'); start();
setSensor('disconnected', 1000);
assert.equal(state.started, true, 'a short loss must not abort the current session');
for (let now = 1000; now <= 31000; now += 5000) {
  assert.equal(updateReconnect(now), 'attempt');
  assert.equal(route('start-haupt-10'), 'session-start', 'stay in the current flow through exactly 30 seconds');
}
assert.equal(state.reconnectAttempts, 7);
assert.equal(updateReconnect(31000), null);
assert.equal(updateReconnect(31001), 'expired');
assert.equal(route('start-haupt-10'), 'h10');
assert.equal(state.started, false);
setSensor('ecg'); confirmECG();
setSensor('no-ecg', 40000);
assert.equal(updateReconnect(40000), 'attempt');
setSensor('ecg', 69999);
assert.equal(updateReconnect(80000), null, 'recovery cancels the loss deadline and retry counter');
assert.equal(route('runde'), 'runde');
assert.equal(state.lostSince, null);
assert.equal(state.reconnectAttempts, 0);
reset();
console.log('PASS: sensor gate, six session routes, main-session baseline/totals, unlimited instructions, reset, duplicate-start guard, 30-second reconnection grace, retries, expiry, recovery.');
