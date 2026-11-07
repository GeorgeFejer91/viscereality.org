import { discoverQuest, joinQuest } from './quest-gateway.js?v=1';
import { polarSetup, QuestRehearsal, rehearsalActions } from './quest-rehearsal.js?v=2';

const byId = (id) => document.getElementById(id);
const text = (element, value) => { if (element.textContent !== value) element.textContent = value; };
const labels = {
  IDLE: 'Quest bereit.', ARMED: 'Probelauf vorbereitet.', RUNNING: 'Probelauf läuft.',
  PAUSED: 'Probelauf pausiert.', FINALIZING: 'Quest beendet den Probelauf.',
  SAVING: 'Quest speichert den Probelauf.', ERROR: 'Die Quest meldet einen Fehler.',
};

export function mountQuestRehearsal() {
  const form = byId('quest-join');
  const statusText = byId('quest-status');
  const clock = byId('quest-clock');
  const polar = byId('quest-polar');
  const polarStatus = byId('quest-polar-status');
  const polarConnect = byId('quest-polar-connect');
  const play = byId('quest-play');
  const finish = byId('quest-finish');
  const acknowledge = byId('quest-acknowledge');
  const disconnect = byId('quest-disconnect');
  const demo = byId('quest-demo');
  let connecting = null;
  let userAction = null;
  let joinMessage = '';
  let storage;
  try { storage = sessionStorage; } catch { /* The current connection still works. */ }
  const rehearsal = new QuestRehearsal({ storage, onChange: paint });

  function paint({ status, busy, pending, unknown, message } = rehearsal.state) {
    const actions = rehearsalActions(status);
    const active = Boolean(status?.rec);
    const waiting = Boolean(connecting || busy || pending || status?.pending);
    const wasJoining = !form.hidden;
    const hadFinishFocus = document.activeElement === finish;
    if (status && !connecting) joinMessage = '';
    form.hidden = Boolean(status);
    form.querySelectorAll('input, button').forEach((element) => { element.disabled = Boolean(connecting); });
    const hint = !status ? 'Quest ist nicht verbunden.'
      : status.h === 'panel' ? 'Schließe das Bedienfeld in der Quest und kehre zur VR-Szene zurück.'
      : status.pi?.r !== false ? 'Diese Sitzung ist nicht für den Probelauf freigegeben.'
      : status.p === 'IDLE' && !actions.play ? 'Warte auf die VR-Szene und den Probelauf.'
      : labels[status.p] ?? 'Quest bereitet den Probelauf vor.';
    text(statusText, joinMessage || (unknown
      ? 'Ergebnis unklar. Prüfe die Quest. Es wird kein Befehl wiederholt.'
      : waiting ? 'Warte auf Bestätigung der Quest …' : message || hint));
    if (wasJoining && form.hidden && form.contains(document.activeElement)) statusText.focus({ preventScroll: true });
    clock.hidden = !status;
    if (status) {
      const seconds = Math.floor(status.ms / 1000);
      text(clock, [Math.floor(seconds / 60), seconds % 60].map((part) => String(part).padStart(2, '0')).join(':'));
    }
    const setup = polarSetup(status);
    const ready = setup?.p && setup.b && setup.c && setup.e;
    const hadPolarFocus = document.activeElement === polarConnect;
    polar.hidden = status?.p !== 'IDLE' || active;
    text(polarStatus, !setup ? 'Polar-Einrichtung ist auf der Quest gerade nicht verfügbar.'
      : !setup.p ? 'Erlaube „Geräte in der Nähe“ für Viscereality auf der Quest.'
      : !setup.b ? 'Schalte Bluetooth auf der Quest ein.'
      : setup.sc ? 'Quest sucht nach dem Polar H10 …'
      : ready ? 'Polar H10 verbunden. Quest empfängt EKG.'
      : setup.c ? 'Polar H10 verbunden. Warte auf EKG von der Quest.'
      : 'Lege den Gurt an. Quest verbindet sich mit dem Polar H10.');
    polarConnect.disabled = waiting || unknown || !setup?.p || !setup.b || setup.sc || ready;
    if (polar.hidden && hadPolarFocus) statusText.focus({ preventScroll: true });
    play.hidden = !status;
    play.disabled = waiting || unknown || !actions.play;
    text(play.firstElementChild, status?.p === 'RUNNING' ? 'Pause' : status?.p === 'PAUSED' ? 'Fortsetzen' : 'Starten');
    finish.hidden = !active;
    if (finish.hidden && hadFinishFocus) statusText.focus({ preventScroll: true });
    finish.disabled = waiting || unknown || !actions.finish;
    acknowledge.hidden = !rehearsal.canAcknowledgeIdle();
    acknowledge.disabled = busy;
    disconnect.hidden = !status || active || waiting || unknown;
    demo.hidden = Boolean(status || pending || connecting);
  }

  async function connect(useBluetooth) {
    if (connecting || !form.reportValidity()) return;
    const attempt = new AbortController();
    connecting = attempt;
    joinMessage = '';
    const serial = byId('quest-serial').value.trim();
    const build = byId('quest-build').value.trim();
    const endpoint = byId('quest-address').value;
    paint();
    try {
      const discovery = useBluetooth ? await discoverQuest(navigator.bluetooth, attempt.signal) : null;
      attempt.signal.throwIfAborted();
      if (discovery && discovery.owner.serial !== serial) throw Error('Die gewählte Quest hat eine andere Seriennummer.');
      const gateway = await joinQuest({ endpoint: discovery?.endpoints[0] ?? endpoint, serial, build,
        discovery: discovery?.owner, signal: attempt.signal });
      if (connecting !== attempt) { gateway.close(); return; }
      rehearsal.attach(gateway);
      statusText.focus({ preventScroll: true });
      if (rehearsal.state.pending) await rehearsal.refresh();
    } catch (error) {
      if (connecting !== attempt) return;
      joinMessage = error instanceof TypeError || ['TimeoutError', 'NotFoundError', 'SecurityError'].includes(error?.name)
        ? 'Verbindung fehlgeschlagen. Prüfe WLAN, Fernmodus und Browser-Berechtigungen.'
        : error?.name === 'Error' ? error.message : 'Die Quest konnte nicht verbunden werden.';
    } finally {
      if (connecting === attempt) { connecting = null; paint(); }
    }
  }

  async function act(action) {
    if (userAction) return;
    const attempt = new AbortController();
    userAction = attempt;
    try { await rehearsal.run(action, attempt.signal); }
    finally { if (userAction === attempt) userAction = null; }
  }

  form.addEventListener('submit', (event) => { event.preventDefault(); connect(false); });
  byId('quest-find').addEventListener('click', () => connect(true));
  polarConnect.addEventListener('click', () => act('polar'));
  play.addEventListener('click', () => act('play'));
  finish.addEventListener('click', () => act('finish'));
  acknowledge.addEventListener('click', () => rehearsal.acknowledgeIdle());
  disconnect.addEventListener('click', () => { rehearsal.attach(null); byId('quest-serial').focus(); });
  const timer = setInterval(() => { if (!document.hidden && !connecting) rehearsal.refresh(); }, 1000);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) userAction?.abort();
    else rehearsal.refresh();
  });
  window.addEventListener('pagehide', (event) => {
    userAction?.abort();
    connecting?.abort();
    connecting = null;
    rehearsal.attach(null);
    if (!event.persisted) clearInterval(timer);
  });
  paint();
}
