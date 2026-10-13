// This prototype only emulates headset verification and ECG reception.
export const choices = { probe: ['anleitung', '5', '10', '15'], haupt: ['10', '15'] };
export const state = { sensor: 'disconnected', selection: null, started: false, entered: false, lostSince: null, reconnectAttempts: 0, lastReconnect: null };
export function hasSignal() { return state.sensor === 'ecg'; }
export function canNavigate() { return hasSignal() || state.entered; }
export function confirmECG() {
  if (!hasSignal()) return false;
  state.entered = true;
  return true;
}
export function setSensor(sensor, now = Date.now()) {
  if (!['searching', 'found', 'connecting', 'ecg', 'no-ecg', 'disconnected'].includes(sensor)) return;
  state.sensor = sensor;
  if (hasSignal()) {
    state.lostSince = null;
    state.reconnectAttempts = 0;
    state.lastReconnect = null;
  } else if (state.entered && state.lostSince === null) {
    state.lostSince = now;
    state.lastReconnect = now - 5000;
  } else if (!state.entered) state.started = false;
}
export function updateReconnect(now = Date.now()) {
  if (!state.entered || hasSignal() || state.lostSince === null) return null;
  if (now - state.lostSince > 30000 && state.reconnectAttempts > 0) {
    state.entered = false;
    state.started = false;
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
  if (!canNavigate()) return 'h10';
  if (['h10', 'runde', 'probe-runde', 'haupt-runde'].includes(requested)) return requested;
  const [, branch, value] = /^start-(probe|haupt)-(anleitung|\d+)$/.exec(requested) ?? [];
  if (!choices[branch]?.includes(value)) return 'h10';
  if (state.selection?.branch !== branch || state.selection?.value !== value) state.started = false;
  const minutes = value === 'anleitung' ? null : Number(value);
  const baselineMinutes = branch === 'haupt' ? 5 : 0;
  state.selection = { branch, value, minutes, baselineMinutes,
    boxMinutes: minutes === null ? null : minutes - baselineMinutes };
  return 'session-start';
}
export function start() {
  if (!hasSignal() || !state.selection || state.started) return false;
  state.started = true;
  return true;
}
export function reset() { Object.assign(state, { sensor: 'disconnected', selection: null, started: false, entered: false, lostSince: null, reconnectAttempts: 0, lastReconnect: null }); }
