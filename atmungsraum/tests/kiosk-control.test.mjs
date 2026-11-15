import test from 'node:test';
import assert from 'node:assert/strict';
import * as contract from '../kiosk-control.js';

const owner = { device: 'a'.repeat(32), serial: 'TEST', epoch: '7', native_epoch: '8' };
const kiosk = (patch = {}) => ({ v: 1, g: 2, requested: true, effective: false,
  state: 'watchdog-failed', service: 'start-failed', blocked: 'watchdog-failed', detail: 'not connected', retry_available: true, ...patch });
const state = (patch = {}) => ({ status: { p: 'IDLE', rec: false, pending: false, kiosk: kiosk(),
  o: { v: 1, s: 'observed', e: '8', a: 5, i: '100' }, ...patch }, busy: false, pending: false, unknown: false });
const storage = () => {
  const values = new Map();
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
};
function fixture() {
  const calls = [];
  let result = 'pending';
  const gateway = { owner, kiosk: async command => { calls.push(command); if (result === 'lost') throw Error('lost');
    return { id: command.id, state: result, detail: 'observed' }; } };
  return { gateway, calls, reply: value => { result = value; } };
}

test('kiosk separates requested from effective and gates bounded advertised retry capability', () => {
  const model = new contract.KioskControl(); const peer = fixture();
  let view = model.update(peer.gateway, state());
  assert.equal(view.canRetry, true);
  assert.match(view.summary, /angefordert.*nicht wirksam/);
  for (const patch of [{ effective: 1 }, { g: -1 }, { service: 'x'.repeat(49) }, { detail: 'x'.repeat(241) }, { retry_available: null }]) {
    assert.equal(model.update(peer.gateway, state({ kiosk: kiosk(patch) })).available, false);
  }
  for (const change of [{ rec: true }, { pending: true }, { kiosk: kiosk({ retry_available: false }) },
    { controller_access: { v: 1, role: 'waiting', active_phone: 'Other', revision: 2 } }]) {
    assert.equal(model.update(peer.gateway, state(change)).canRetry, false);
  }
  view = model.update(peer.gateway, state({ kiosk: kiosk({ effective: true, retry_available: false }) }));
  assert.match(view.summary, /wirksam/);
  assert.equal(view.canRetry, false);
});

test('actual native kiosk blockers give their own German action without inventing Accessibility requirements', () => {
  // NativeRendererSoftKioskCoordinator effectiveness/service enums and
  // BreathCompositionPanelModule.remoteKioskProjection blocked precedence.
  const cases = [
    [{ state: 'watchdog-starting', service: 'start-pending', blocked: 'watchdog-starting', retry_available: false }, /wird.*gestartet.*warten/i],
    [{ state: 'watchdog-failed', service: 'start-failed', blocked: 'watchdog-failed' }, /konnte nicht starten.*Kiosk-Dienst erneut verbinden/],
    [{ state: 'watchdog-missing', service: 'missing-or-disabled', blocked: 'watchdog-missing' }, /nicht verbunden.*Kiosk-Dienst erneut verbinden/],
    [{ state: 'interrupted', service: 'interrupted', blocked: 'interrupted' }, /unterbrochen.*Kiosk-Dienst erneut verbinden/],
    [{ state: 'watchdog-failed', service: 'start-failed', blocked: 'watchdog-failed', retry_available: false }, /Kehre.*Headset zurück.*keinen Dienst-Neustart/],
    [{ state: 'ready-armed', service: 'connected', blocked: 'presentation-stale', retry_available: false }, /aktuelle Ansicht.*Headset.*Atmungsraum-App/],
    [{ state: 'watchdog-awaiting-presentation', service: 'connected', blocked: 'watchdog-awaiting-presentation', retry_available: false }, /aktuelle Ansicht.*Headset.*Atmungsraum-App/],
    [{ state: 'ready-armed', service: 'connected', blocked: 'system-prompt-or-transition', retry_available: false }, /Systemdialog.*Headset.*VR-App zurück/],
    [{ state: 'ready-armed', service: 'connected', blocked: 'recovery-exhausted', retry_available: false }, /ausgeschöpft.*Beende.*App-Bibliothek.*Smartphone/],
    [{ state: 'terminal', service: 'connected', blocked: 'terminal', requested: false, retry_available: false }, /beendet.*App-Bibliothek/],
    [{ state: 'ready-disarmed', service: 'missing-or-disabled', blocked: 'not-requested', requested: false, retry_available: false }, /nicht angefordert.*App-Bibliothek/],
    [{ state: 'revoked', service: 'revoked', blocked: 'revoked', retry_available: false }, /entzogen.*Meldung im Headset/],
    [{ state: 'degraded-home-resolving', service: 'connected', blocked: 'degraded-home-resolving', retry_available: false }, /Home-Oberfläche.*warten/i],
    [{ state: 'unavailable-home-surface', service: 'connected', blocked: 'unavailable-home-surface', retry_available: false }, /Home-Oberfläche.*nicht verfügbar.*Headset/],
  ];
  const model = new contract.KioskControl(); const peer = fixture();
  for (const [patch, expected] of cases) {
    const view = model.update(peer.gateway, state({ kiosk: kiosk(patch) }));
    assert.match(view.prerequisites, expected, JSON.stringify(patch));
    assert.doesNotMatch(view.prerequisites, /Bedienungshilfe|Accessibility/, JSON.stringify(patch));
    assert.equal(view.canRetry, patch.retry_available !== false, JSON.stringify(patch));
  }
  const needs = model.update(peer.gateway, state({ kiosk: kiosk({ state: 'needs-accessibility-setup', service: 'missing-or-disabled', blocked: 'needs-accessibility-setup', retry_available: false }) }));
  assert.match(needs.prerequisites, /meldet.*Bedienungshilfe.*Headset/);
});

test('kiosk retry journals its CAS generation and checks one original ID without replay after reload', async () => {
  let now = 0;
  const store = storage(); const peer = fixture();
  const model = new contract.KioskControl({ storage: store, now: () => now });
  model.update(peer.gateway, state()); await model.retry();
  assert.equal(peer.calls.length, 1);
  assert.deepEqual(Object.keys(peer.calls[0]).sort(), ['g', 'id', 'op', 'v']);
  assert.equal(peer.calls[0].g, 2);
  assert.equal(peer.calls[0].op, 'retry');
  const restored = new contract.KioskControl({ storage: store, now: () => now });
  restored.update(peer.gateway, state()); await restored.retry();
  await restored.refresh();
  assert.equal(peer.calls.length, 2);
  assert.deepEqual(peer.calls[1], { v: 1, id: peer.calls[0].id, op: 'receipt' });
  now = 13000; await restored.refresh();
  let view = restored.update(peer.gateway, state());
  assert.equal(view.canRetry, false);
  assert.equal(view.canCheck, true);
  assert.match(view.message, /Wartezeit.*abgelaufen/);
  peer.reply('confirmed'); restored.recheck(); await restored.refresh();
  view = restored.update(peer.gateway, state());
  assert.equal(restored.pending, null);
  assert.match(view.summary, /nicht wirksam/, 'a service receipt never manufactures effective kiosk');
  assert.match(view.message, /Dienst.*verbunden/);
});

test('wrong kiosk IDs, current actor loss and pending control retain the fence', async () => {
  const model = new contract.KioskControl(); const peer = fixture(); peer.reply('lost');
  model.update(peer.gateway, state()); await model.retry();
  const id = model.pending.command.id;
  peer.gateway.kiosk = async command => ({ id: 'b'.repeat(16), state: 'confirmed', detail: '' });
  await model.refresh(); assert.equal(model.pending.command.id, id);
  model.update({ ...peer.gateway, owner: { ...owner, device: 'c'.repeat(32) } }, state());
  assert.equal(model.view.canCheck, false);
  assert.equal(model.view.canAcknowledge, false);
  model.update({ ...peer.gateway, owner: { ...owner, native_epoch: '9' } }, state({ o: { v: 1, s: 'observed', e: '9', a: 0, i: '101' } }));
  assert.equal(model.view.canAcknowledge, true);
  model.acknowledge(); assert.equal(model.pending, null);
});
