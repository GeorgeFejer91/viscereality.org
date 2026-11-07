import { discoverQuest, joinQuest, sameQuest } from './quest-gateway.js?v=2';
import { polarSetup, QuestRehearsal, rehearsalActions } from './quest-rehearsal.js?v=3';
import { ecgTrace, questRecording } from './quest-data.js?v=1';

const byId = (id) => document.getElementById(id);
const text = (element, value) => { if (element.textContent !== value) element.textContent = value; };
const recordingId = /^utc-ns-[0-9]{20}$/;
const recordingStorageKey = 'atmungsraum-quest-recording';
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
  const condition = byId('quest-condition');
  const data = byId('quest-data');
  const ecgStatus = byId('quest-ecg-status');
  const graph = byId('quest-ecg');
  const trace = byId('quest-ecg-trace');
  const scale = byId('quest-ecg-scale');
  const exportStatus = byId('quest-export-status');
  const retry = byId('quest-export-retry');
  const download = byId('quest-export-download');
  let ecgTimer, ecgAttempt, exportAttempt, recording, fetchedRecording, exportMessage = '';
  let connecting = null;
  let userAction = null;
  let joinMessage = '';
  let storage;
  try { storage = sessionStorage; } catch { /* The current connection still works. */ }
  const rehearsal = new QuestRehearsal({ storage, onChange: paint });

  function paint({ status, busy, pending, unknown, message } = rehearsal.state) {
    const actions = rehearsalActions(status, condition.value);
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
    condition.parentElement.hidden = status?.ecg_preview !== 1 || status.p !== 'IDLE';
    condition.disabled = waiting || unknown || Boolean(exportAttempt);
    play.disabled = waiting || unknown || Boolean(exportAttempt) || !actions.play;
    text(play.firstElementChild, status?.p === 'RUNNING' ? 'Pause' : status?.p === 'PAUSED' ? 'Fortsetzen' : 'Starten');
    finish.hidden = !active;
    if (finish.hidden && hadFinishFocus) statusText.focus({ preventScroll: true });
    finish.disabled = waiting || unknown || !actions.finish;
    acknowledge.hidden = !rehearsal.canAcknowledgeIdle();
    acknowledge.disabled = busy;
    disconnect.hidden = !status || active || waiting || unknown;
    demo.hidden = Boolean(status || pending || connecting);
    if (rehearsal.gateway?.recordings && status?.rec && recordingId.test(status.recording_id)
      && recording !== status.recording_id) {
      clearRun();
      recording = status.recording_id;
      const { device, serial } = rehearsal.gateway.owner;
      try { storage?.setItem(recordingStorageKey, JSON.stringify({ device, serial, id: recording })); }
      catch { /* Recovery remains available while this page is open. */ }
    }
    data.hidden = status?.ecg_preview !== 1;
    if (data.hidden) showTrace(null);
    exportStatus.hidden = !exportMessage;
    text(exportStatus, exportMessage);
    retry.hidden = !rehearsal.gateway?.recordings || !recording || fetchedRecording === recording
      || Boolean(exportAttempt) || status?.p !== 'IDLE';
    retry.disabled = waiting || unknown;
  }

  function showTrace(value) {
    clearTimeout(ecgTimer);
    graph.hidden = scale.hidden = !value;
    trace.setAttribute('d', value?.path ?? '');
    text(ecgStatus, value ? 'Live-EKG vom Polar H10' : rehearsal.gateway?.ecg
      ? 'Warte auf aktuelle EKG-Daten von der Quest.' : 'Für Live-EKG mit dem Quest-Schlüssel verbinden.');
    if (value) {
      text(scale, `2 s · 130 Hz · ±${value.range} µV`);
      ecgTimer = setTimeout(() => showTrace(null), value.ttl);
    }
  }

  function clearDownload() {
    if (download.hasAttribute('href')) URL.revokeObjectURL(download.href);
    download.removeAttribute('href');
    download.hidden = true;
  }

  function clearRun() {
    exportAttempt?.abort();
    exportAttempt = null;
    recording = fetchedRecording = null;
    exportMessage = '';
    clearDownload();
  }

  async function saveRecording() {
    const gateway = rehearsal.gateway;
    if (!gateway?.recordings || !recording || exportAttempt || rehearsal.state.status?.p !== 'IDLE') return;
    const id = recording;
    const attempt = new AbortController();
    exportAttempt = attempt;
    clearDownload();
    exportMessage = 'Quest-Aufnahme wird geprüft und übertragen …';
    paint();
    try {
      const archive = await questRecording(gateway, id, attempt.signal);
      if (exportAttempt !== attempt || rehearsal.gateway !== gateway) return;
      download.href = URL.createObjectURL(archive);
      download.download = `quest-${id}.zip`;
      download.hidden = false;
      fetchedRecording = id;
      exportMessage = 'Quest-Aufnahme geprüft. Der Download ist bereit.';
      download.click();
    } catch {
      if (exportAttempt === attempt) exportMessage = 'Aufnahme konnte nicht übertragen werden. Versuche es erneut.';
    } finally {
      if (exportAttempt === attempt) { exportAttempt = null; paint(); }
    }
  }

  async function tick() {
    if (document.hidden || connecting || userAction) return;
    await rehearsal.refresh();
    const gateway = rehearsal.gateway;
    const { status, busy, pending, unknown } = rehearsal.state;
    if (!gateway || !status || busy || pending || unknown || userAction || document.hidden) return;
    if (status.p === 'IDLE' && recording && fetchedRecording !== recording && !exportMessage && !exportAttempt) {
      void saveRecording();
    }
    if (!gateway.ecg || ecgAttempt || exportAttempt) return;
    const attempt = new AbortController();
    ecgAttempt = attempt;
    const start = performance.now();
    try {
      const frame = await gateway.ecg(attempt.signal);
      if (ecgAttempt === attempt && rehearsal.gateway === gateway && !document.hidden && !attempt.signal.aborted) {
        showTrace(ecgTrace(frame, performance.now() - start));
      }
    } catch {
      if (ecgAttempt === attempt && !attempt.signal.aborted) showTrace(null);
    } finally {
      if (ecgAttempt === attempt) ecgAttempt = null;
    }
  }

  async function connect(useBluetooth) {
    if (connecting || !form.reportValidity()) return;
    const attempt = new AbortController();
    connecting = attempt;
    joinMessage = '';
    const serial = byId('quest-serial').value.trim();
    const build = byId('quest-build').value.trim();
    const endpoint = byId('quest-address').value;
    const controller = byId('quest-controller');
    const key = byId('quest-key');
    const enrollment = controller.value || key.value ? { controller: controller.value.trim(), key: key.value.trim() } : undefined;
    controller.value = key.value = '';
    paint();
    try {
      const discovery = useBluetooth ? await discoverQuest(navigator.bluetooth, attempt.signal) : null;
      attempt.signal.throwIfAborted();
      if (discovery && discovery.owner.serial !== serial) throw Error('Die gewählte Quest hat eine andere Seriennummer.');
      const gateway = await joinQuest({ endpoint: discovery?.endpoints[0] ?? endpoint, serial, build,
        discovery: discovery?.owner, enrollment, signal: attempt.signal });
      if (connecting !== attempt) { gateway.close(); return; }
      if (!sameQuest(rehearsal.gateway?.owner, gateway.owner)) {
        clearRun();
        try {
          const saved = JSON.parse(storage?.getItem(recordingStorageKey) ?? 'null');
          if (gateway.recordings && saved?.device === gateway.owner.device && saved.serial === gateway.owner.serial
            && recordingId.test(saved.id)) {
            recording = saved.id;
            exportMessage = 'Die letzte beobachtete Quest-Aufnahme kann erneut abgerufen werden.';
          }
        } catch { /* Treat malformed or unavailable storage as no observed run. */ }
      }
      rehearsal.attach(gateway);
      showTrace(null);
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
    ecgAttempt?.abort();
    const attempt = new AbortController();
    userAction = attempt;
    try { await rehearsal.run(action, attempt.signal, condition.value); }
    finally { if (userAction === attempt) userAction = null; }
  }

  form.addEventListener('submit', (event) => { event.preventDefault(); connect(false); });
  byId('quest-find').addEventListener('click', () => connect(true));
  polarConnect.addEventListener('click', () => act('polar'));
  condition.addEventListener('change', () => paint());
  play.addEventListener('click', () => act('play'));
  finish.addEventListener('click', () => act('finish'));
  acknowledge.addEventListener('click', () => rehearsal.acknowledgeIdle());
  retry.addEventListener('click', saveRecording);
  function detach() {
    clearRun();
    rehearsal.attach(null);
  }
  disconnect.addEventListener('click', () => { detach(); byId('quest-serial').focus(); });
  const timer = setInterval(tick, 1000);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      userAction?.abort();
      ecgAttempt?.abort();
      exportAttempt?.abort();
      showTrace(null);
    } else tick();
  });
  window.addEventListener('pagehide', (event) => {
    userAction?.abort();
    connecting?.abort();
    connecting = null;
    detach();
    if (!event.persisted) clearInterval(timer);
  });
  paint();
}
