// This prototype only emulates headset verification and ECG reception.
import { participantIdentity } from './data-logging.js?v=20261105-data';
export const choices = { probe: ['anleitung', '5', '10', '15'], haupt: ['15'] };
export const state = { sensor: 'disconnected', patient: null, selection: null, started: false, entered: false, lostSince: null, reconnectAttempts: 0, lastReconnect: null };
export function hasSignal() { return state.sensor === 'ecg'; }
export function canNavigate() { return Boolean(state.patient && (hasSignal() || state.entered)); }
export function confirmECG() {
  if (!state.patient || !hasSignal()) return false;
  state.entered = true;
  return true;
}
export function confirmPatient({ surnamePrefix = '', age = '', gender = '' } = {}) {
  let identity;
  try { identity = participantIdentity({ surnamePrefix, age, gender }); } catch { return false; }
  state.patient = { surnamePrefix: identity.name, age: identity.age, gender };
  state.selection = null;
  state.started = false;
  return true;
}
export function setSensor(sensor, now = performance.now()) {
  if (!['searching', 'found', 'connecting', 'ecg', 'no-ecg', 'disconnected'].includes(sensor)) return;
  // A late recovery must expire the old scenario before clearing its loss clock.
  const reconnect = sensor === 'ecg' ? updateReconnect(now) : null;
  state.sensor = sensor;
  if (hasSignal()) {
    state.lostSince = null;
    state.reconnectAttempts = 0;
    state.lastReconnect = null;
  } else if (state.entered && state.lostSince === null) {
    state.lostSince = now;
    state.lastReconnect = now - 5000;
  } else if (!state.entered) state.started = false;
  return reconnect;
}
export function updateReconnect(now = performance.now()) {
  if (!state.entered || hasSignal() || state.lostSince === null) return null;
  if (now - state.lostSince > 30000) {
    state.entered = false;
    state.started = false;
    state.selection = null;
    return 'expired';
  }
  if (now - state.lastReconnect >= 5000) {
    state.lastReconnect = now;
    state.reconnectAttempts++;
    return 'attempt';
  }
  return null;
}
export function route(requested) {
  if (requested === 'haupt-runde') requested = 'start-haupt-15';
  if (requested === 'daten') return 'daten';
  if (!state.patient || requested === 'patient') return 'patient';
  if (!canNavigate()) return 'h10';
  if (requested === 'h10') return 'h10';
  const [, branch, value] = /^start-(probe|haupt)-(anleitung|\d+)$/.exec(requested) ?? [];
  const menu = ['runde', 'probe-runde'].includes(requested);
  if (!menu && !choices[branch]?.includes(value)) return 'h10';
  if (menu) return requested;
  if (state.selection?.branch !== branch || state.selection?.value !== value) state.started = false;
  const minutes = value === 'anleitung' ? null : Number(value);
  state.selection = { branch, value, minutes, boxMinutes: minutes };
  return 'session-start';
}
export function start() {
  if (!hasSignal() || !state.patient || !state.selection || state.started) return false;
  state.entered = true;
  state.started = true;
  return true;
}
export function reset() { Object.assign(state, { sensor: 'disconnected', patient: null, selection: null, started: false, entered: false, lostSince: null, reconnectAttempts: 0, lastReconnect: null }); }
