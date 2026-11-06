import test from 'node:test';
import assert from 'node:assert/strict';
import { state, reset, confirmPatient, confirmECG, setSensor, route, start, updateReconnect } from '../ui-logic.js';

function playing(confirm = true) {
  reset();
  confirmPatient({ surnamePrefix: 'Mül', age: '42', gender: 'female' });
  setSensor('ecg');
  if (confirm) confirmECG();
  route('start-haupt-15');
  start();
  setSensor('disconnected', 1000);
}

test('A scenario started through a valid-ECG deep link also arms the loss deadline', () => {
  try {
    playing(false);
    assert.equal(state.started, true, 'a short loss preserves the directly started run');
    assert.equal(updateReconnect(31001), 'expired');
    assert.equal(route('start-haupt-15'), 'h10');
    assert.equal(state.started, false);
  } finally { reset(); }
});

test('Polar loss expires after 30 seconds even if the first reconnect poll is delayed', () => {
  try {
    playing();
    assert.equal(updateReconnect(31001), 'expired');
    assert.equal(route('start-haupt-15'), 'h10');
    assert.equal(state.started, false);
    assert.equal(state.selection, null);
    assert.equal(state.patient.surnamePrefix, 'Mül');
  } finally { reset(); }
});

test('ECG arriving after the deadline cannot revive the expired scenario', () => {
  try {
    playing();
    updateReconnect(1000);
    assert.equal(setSensor('ecg', 31001), 'expired');
    assert.equal(state.started, false);
    assert.equal(state.selection, null);
    assert.equal(state.entered, false);
    assert.equal(start(), false);
  } finally { reset(); }
});

test('Recovery through exactly 30 seconds preserves the current scenario', () => {
  try {
    playing();
    assert.notEqual(setSensor('ecg', 31000), 'expired');
    assert.equal(state.started, true);
    assert.equal(state.selection.minutes, 15);
    assert.equal(updateReconnect(60000), null);
  } finally { reset(); }
});
