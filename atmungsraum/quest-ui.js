import { discoverQuest, joinQuest, sameQuest } from './quest-gateway.js?v=3';
import { polarSetup, QuestRehearsal, rehearsalActions } from './quest-rehearsal.js?v=4';
import { ecgTrace, questRecording, questRecordings } from './quest-data.js?v=3';
import { openQuestRecordingStore } from './quest-recording-store.js?v=1';

const byId = (id) => document.getElementById(id);
const text = (element, value) => { if (element.textContent !== value) element.textContent = value; };
const archiveDate = new Intl.DateTimeFormat('de-DE', { dateStyle: 'short', timeStyle: 'medium' });

function idleForSync({ status, busy, pending, unknown }) {
  return status?.p === 'IDLE' && !status.rec && !status.pending && !busy && !pending && !unknown;
}
const labels = {
  IDLE: 'Quest bereit.', ARMED: 'Probelauf vorbereitet.', RUNNING: 'Probelauf läuft.',
  PAUSED: 'Probelauf pausiert.', FINALIZING: 'Quest beendet den Probelauf.',
  SAVING: 'Quest speichert den Probelauf.', ERROR: 'Die Quest meldet einen Fehler.',
};

export function mountQuestRehearsal(participantIdentity) {
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
  const participant = byId('quest-participant');
  const participantLabel = byId('quest-participant-label');
  const participantForm = byId('patient-form');
  const participantFields = byId('patient-fields');
  const participantSubmit = byId('patient-next');
  let participantId = '';
  participant.append(participantForm);
  participantForm.append(participantSubmit);
  participantFields.setAttribute('aria-labelledby', participantLabel.id);
  text(participantSubmit.firstElementChild, 'Teilnehmer-ID übernehmen');
  const data = byId('quest-data');
  const ecgStatus = byId('quest-ecg-status');
  const graph = byId('quest-ecg');
  const trace = byId('quest-ecg-trace');
  const scale = byId('quest-ecg-scale');
  const exportStatus = byId('quest-export-status');
  const retry = byId('quest-export-retry');
  const download = byId('quest-export-download');
  const archivePicker = byId('quest-archives');
  let ecgTimer, ecgAttempt, exportAttempt, nextSync = 0, exportMessage = '';
  let archiveRead = 0;
  let connecting = null;
  let userAction = null;
  let joinMessage = '';
  let storage;
  try { storage = sessionStorage; } catch { /* The current connection still works. */ }
  const rehearsal = new QuestRehearsal({ storage, onChange: paint });

  function paint({ status, busy, pending, unknown, message } = rehearsal.state) {
    const actions = rehearsalActions(status, condition.value);
    const active = Boolean(status?.rec);
    const requiresParticipant = status?.participant_arm === 1;
    const armedWithParticipant = Boolean(rehearsal.gateway?.participantArm);
    const waiting = Boolean(connecting || busy || pending || status?.pending);
    const wasJoining = !form.hidden;
    const hadFinishFocus = document.activeElement === finish;
    if (status && !connecting) joinMessage = '';
    form.hidden = Boolean(status);
    form.querySelectorAll('input, button').forEach((element) => { element.disabled = Boolean(connecting); });
    const hint = !status ? 'Quest ist nicht verbunden.'
      : status.h === 'panel' ? 'Schließe das Bedienfeld in der Quest und kehre zur VR-Szene zurück.'
      : status.pi?.r !== false ? 'Diese Sitzung ist nicht für den Probelauf freigegeben.'
      : requiresParticipant && !armedWithParticipant ? 'Verbinde dich mit dem Quest-Schlüssel, um einen Probelauf zu starten.'
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
    const showParticipant = requiresParticipant && armedWithParticipant && status.p === 'IDLE' && !active;
    if (!showParticipant && !participant.hidden && participant.contains(document.activeElement)) {
      statusText.focus({ preventScroll: true });
    }
    participant.hidden = !showParticipant;
    participantFields.disabled = !showParticipant || waiting || unknown;
    participantSubmit.disabled = participantFields.disabled;
    text(participantLabel, participantId ? `Teilnehmer-ID: ${participantId}` : 'Teilnehmer-ID eingeben');
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
    condition.disabled = waiting || unknown;
    play.disabled = waiting || unknown || !actions.play
      || actions.play === 'arm' && requiresParticipant && (!armedWithParticipant || !participantId || !ready);
    text(play.firstElementChild, status?.p === 'RUNNING' ? 'Pause' : status?.p === 'PAUSED' ? 'Fortsetzen' : 'Starten');
    finish.hidden = !active;
    if (finish.hidden && hadFinishFocus) statusText.focus({ preventScroll: true });
    finish.disabled = waiting || unknown || !actions.finish;
    acknowledge.hidden = !rehearsal.canAcknowledgeIdle();
    acknowledge.disabled = busy;
    disconnect.hidden = !status || active || waiting || unknown;
    demo.hidden = Boolean(status || pending || connecting);
    if (!idleForSync(rehearsal.state)) {
      exportAttempt?.abort();
      nextSync = 0;
    }
    data.hidden = status?.ecg_preview !== 1;
    if (data.hidden) showTrace(null);
    exportStatus.hidden = !exportMessage || !status;
    text(exportStatus, exportMessage);
    retry.hidden = !rehearsal.gateway?.recordings || status?.p !== 'IDLE';
    retry.disabled = waiting || unknown || Boolean(exportAttempt);
    archivePicker.parentElement.hidden = !status || !archivePicker.options.length;
    download.hidden = !status || !download.hasAttribute('href');
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

  function clearArchives() {
    exportAttempt?.abort();
    archiveRead++;
    nextSync = 0;
    exportMessage = '';
    archivePicker.replaceChildren();
    clearDownload();
  }

  async function selectArchive() {
    const gateway = rehearsal.gateway;
    const id = archivePicker.value;
    const read = ++archiveRead;
    clearDownload();
    if (!gateway?.recordings || !id) return;
    let store;
    try {
      store = await openQuestRecordingStore(gateway.owner);
      const archive = await store.get(id);
      if (read !== archiveRead || rehearsal.gateway !== gateway) return;
      if (!(archive instanceof Blob)) throw Error('Aufnahme fehlt.');
      download.href = URL.createObjectURL(archive);
      download.download = archive instanceof File ? archive.name : `quest-${id}.zip`;
    } catch {
      if (read === archiveRead) exportMessage = 'Die gespeicherte Aufnahme konnte nicht geöffnet werden.';
    } finally {
      store?.close();
      if (read === archiveRead) paint();
    }
  }

  function showArchives(ids) {
    if (ids.length === archivePicker.options.length && ids.every((id, index) => archivePicker.options[index].value === id)) {
      if (ids.length && !download.hasAttribute('href')) void selectArchive();
      return;
    }
    const selected = archivePicker.value;
    archivePicker.replaceChildren(...ids.map(id => {
      const date = new Date(Number(BigInt(id.slice(7)) / 1000000n));
      return new Option(archiveDate.format(date), id);
    }));
    if (ids.includes(selected)) archivePicker.value = selected;
    void selectArchive();
  }

  async function syncRecordings(gateway) {
    const attempt = new AbortController();
    const { signal } = attempt;
    exportAttempt = attempt;
    nextSync = performance.now() + 15000;
    paint();
    try {
      await navigator.locks.request(`atmungsraum-quest-sync/${gateway.owner.device}`, { ifAvailable: true }, async lock => {
        if (!lock) return;
        signal.throwIfAborted();
        const store = await openQuestRecordingStore(gateway.owner);
        try {
          const saved = await store.ids();
          signal.throwIfAborted();
          showArchives(saved);
          const known = new Set(saved);
          exportMessage = 'Quest-Aufnahmen werden abgeglichen …';
          paint();
          for await (const { id, participantCsv } of questRecordings(gateway, signal)) {
            const existing = known.has(id);
            if (existing && (!participantCsv || await store.get(id) instanceof File)) continue;
            const archive = await questRecording(gateway, id, signal);
            if (participantCsv && !(archive instanceof File)) throw Error('Quest-CSV fehlt.');
            signal.throwIfAborted();
            await store.put(id, archive);
            signal.throwIfAborted();
            known.add(id);
            showArchives([...known].sort().reverse());
            if (existing && archivePicker.value === id) void selectArchive();
          }
          exportMessage = known.size ? 'Aufnahmen sind in diesem Browser gespeichert.' : 'Keine gespeicherten Aufnahmen auf der Quest.';
          nextSync = performance.now() + 60000;
        } finally { store.close(); }
      });
    } catch {
      if (!signal.aborted) exportMessage = 'Abgleich fehlgeschlagen. Die App versucht es erneut, wenn die Quest bereit ist.';
    } finally {
      if (exportAttempt === attempt) {
        if (signal.aborted) exportMessage = '';
        exportAttempt = null;
        paint();
      }
    }
  }

  async function tick() {
    if (document.hidden || connecting || userAction) return;
    await rehearsal.refresh();
    const gateway = rehearsal.gateway;
    const { status, busy, pending, unknown } = rehearsal.state;
    if (!gateway || !status || busy || pending || unknown || userAction || document.hidden) return;
    if (gateway.recordings && idleForSync(rehearsal.state) && !exportAttempt && performance.now() >= nextSync) {
      void syncRecordings(gateway);
    }
    if (!gateway.ecg || ecgAttempt) return;
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
      if (!sameQuest(rehearsal.gateway?.owner, gateway.owner)) clearArchives();
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
    exportAttempt?.abort();
    nextSync = 0;
    const attempt = new AbortController();
    userAction = attempt;
    try { await rehearsal.run(action, attempt.signal, condition.value, participantId); }
    finally { if (userAction === attempt) userAction = null; }
  }

  form.addEventListener('submit', (event) => { event.preventDefault(); connect(false); });
  participantForm.addEventListener('input', () => {
    if (!participantId) return;
    participantId = '';
    paint();
  });
  participantForm.addEventListener('submit', (event) => {
    event.preventDefault();
    if (participantFields.disabled) return;
    try {
      participantId = participantIdentity(Object.fromEntries(new FormData(participantForm))).id;
      participant.open = false;
      participant.querySelector('summary').focus({ preventScroll: true });
      paint();
    } catch {
      participantId = '';
      paint();
    }
  });
  byId('quest-find').addEventListener('click', () => connect(true));
  polarConnect.addEventListener('click', () => act('polar'));
  condition.addEventListener('change', () => paint());
  play.addEventListener('click', () => act('play'));
  finish.addEventListener('click', () => act('finish'));
  acknowledge.addEventListener('click', () => rehearsal.acknowledgeIdle());
  retry.addEventListener('click', () => { nextSync = 0; void tick(); });
  archivePicker.addEventListener('change', selectArchive);
  function detach() {
    clearArchives();
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
    } else { nextSync = 0; void tick(); }
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
