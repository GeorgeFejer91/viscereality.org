import { selectQuest, readQuest, joinQuest, sameQuest } from './quest-gateway.js?v=7';
import { canRecenter, polarSetup, QuestRehearsal, rehearsalActions, rehearsalPolarAllowed, rehearsalScenarios } from './quest-rehearsal.js?v=7';
import { ecgTrace, questRecording, questRecordings } from './quest-data.js?v=18';
import { openQuestRecordingStore } from './quest-recording-store.js?v=2';
import { mountQuestDrive } from './quest-drive-ui.js?v=1';

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

export function mountQuestRehearsal(participantIdentity, savedPair) {
  const drive = mountQuestDrive();
  const form = byId('quest-join');
  const find = byId('quest-find');
  const repair = byId('quest-repair');
  const statusText = byId('quest-status');
  const clock = byId('quest-clock');
  const recenter = byId('recenter-vr');
  const recenterNote = byId('recenter-vr-note');
  const polar = byId('quest-polar');
  const polarStatus = byId('quest-polar-status');
  const polarConnect = byId('quest-polar-connect');
  const play = byId('quest-play');
  const finish = byId('quest-finish');
  const acknowledge = byId('quest-acknowledge');
  const disconnect = byId('quest-disconnect');
  const condition = byId('quest-condition');
  const conditionName = byId('quest-condition-name');
  const scenario = byId('quest-scenario');
  const scenarioOptions = Array.from(scenario.options);
  const participant = byId('quest-participant');
  const participantLabel = byId('quest-participant-label');
  const participantForm = byId('patient-form');
  const participantFields = byId('patient-fields');
  const participantSubmit = byId('patient-next');
  const participantOutput = byId('patient-id');
  let participantId = '';
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
  let pairingDevice = null;
  let rememberedDevice = null;
  let nextConnect = 0;
  let autoConnect = true;
  let canRetry = false;
  let pairWarning = '';
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
    const recenterAvailable = Boolean(rehearsal.gateway?.recenter && canRecenter(status) && !waiting && !unknown);
    recenter.setAttribute('aria-disabled', String(!recenterAvailable));
    text(recenterNote, recenterAvailable ? 'Richtet die VR-Szene an der aktuellen Blickrichtung aus.'
      : 'Verbinde dich mit einer Quest im Fernmodus, um die VR-Szene zu zentrieren.');
    const connectingFocus = [find, repair].includes(document.activeElement) || form.contains(document.activeElement);
    const hadFinishFocus = document.activeElement === finish;
    if (status && !connecting) joinMessage = '';
    form.hidden = Boolean(status) || !pairingDevice;
    form.querySelectorAll('input, button').forEach((element) => { element.disabled = Boolean(connecting); });
    find.hidden = Boolean(status || pairingDevice || (savedPair && autoConnect && !canRetry));
    find.disabled = Boolean(connecting);
    text(find.firstElementChild, savedPair ? 'Erneut verbinden' : 'Quest verbinden');
    repair.hidden = Boolean(status || !savedPair || !canRetry || pairingDevice || connecting);
    const hint = !status ? (pairingDevice ? 'Quest ausgewählt.'
      : savedPair && autoConnect
      ? 'Warte auf die gespeicherte Quest.'
      : 'Starte die Quest im Fernmodus und schalte Bluetooth am Smartphone ein.')
      : status.h === 'panel' ? 'Schließe das Bedienfeld in der Quest und kehre zur VR-Szene zurück.'
      : !rehearsalPolarAllowed(status) ? 'Diese Sitzung ist nicht für den Probelauf freigegeben.'
      : requiresParticipant && !armedWithParticipant ? 'Verbinde dich mit dem Quest-Schlüssel, um einen Probelauf zu starten.'
      : status.pi?.r === true && status.pi.f !== true && ['ARMED', 'PAUSED'].includes(status.p)
        ? 'Warte auf aktuelle Polar-H10-Daten von der Quest.'
      : status.p === 'IDLE' && !actions.play ? 'Warte auf die VR-Szene und den Probelauf.'
      : labels[status.p] ?? 'Quest bereitet den Probelauf vor.';
    text(statusText, joinMessage || pairWarning || (unknown
      ? 'Ergebnis unklar. Prüfe die Quest. Es wird kein Befehl wiederholt.'
      : connecting ? 'Quest wird verbunden …' : waiting ? 'Warte auf Bestätigung der Quest …' : message || hint));
    if (status && connectingFocus) statusText.focus({ preventScroll: true });
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
    const available = rehearsal.gateway?.scenarioArm ? rehearsalScenarios(status) : [];
    const showScenario = available.length > 0 && status?.p === 'IDLE';
    if (!showScenario && document.activeElement === scenario) statusText.focus({ preventScroll: true });
    scenario.parentElement.hidden = !showScenario;
    scenario.disabled = !showScenario || waiting || unknown;
    if (showScenario) {
      for (const option of scenarioOptions) option.hidden = option.disabled = !available.includes(option.value);
      if (!available.includes(scenario.value)) scenario.value = scenarioOptions.find(option => !option.disabled).value;
    }
    condition.parentElement.hidden = status?.ecg_preview !== 1 || status.p !== 'IDLE';
    condition.disabled = waiting || unknown;
    text(conditionName, showScenario ? 'VR-Bedingung' : 'Szenario');
    text(condition.options[0], showScenario ? 'Bedingung A' : 'Szenario A');
    text(condition.options[1], showScenario ? 'Bedingung B' : 'Szenario B');
    play.disabled = waiting || unknown || !actions.play
      || actions.play === 'arm' && requiresParticipant && (!armedWithParticipant || !participantId || !ready);
    text(play.firstElementChild, status?.p === 'RUNNING' ? 'Pause' : status?.p === 'PAUSED' ? 'Fortsetzen' : 'Starten');
    finish.hidden = !active;
    if (finish.hidden && hadFinishFocus) statusText.focus({ preventScroll: true });
    finish.disabled = waiting || unknown || !actions.finish;
    acknowledge.hidden = !rehearsal.canAcknowledgeIdle();
    acknowledge.disabled = busy;
    disconnect.hidden = !status || active || waiting || unknown;
    if (!idleForSync(rehearsal.state)) {
      exportAttempt?.abort();
      drive.cancel();
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
    graph.toggleAttribute('hidden', !value);
    scale.hidden = !value;
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
      const stored = await store.get(id);
      const archive = stored instanceof Blob ? stored : stored?.download;
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
            if (existing) {
              const stored = await store.get(id);
              if (stored?.bundleVersion === 1 && stored.bundle instanceof File
                && (!participantCsv || (stored.download instanceof File && stored.download.name.endsWith('.csv')))) continue;
            }
            const archive = await questRecording(gateway, id, signal, { includeSensorBundle: true });
            if (participantCsv && !archive.download.name.endsWith('.csv')) throw Error('Quest-CSV fehlt.');
            signal.throwIfAborted();
            await store.put(id, archive);
            signal.throwIfAborted();
            drive.schedule();
            void drive.sync();
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
    if (document.hidden || connecting || userAction) return;
    if (!rehearsal.state.status && savedPair && autoConnect && !pairingDevice && performance.now() >= nextConnect) {
      void connect(savedPair);
      return;
    }
    const gateway = rehearsal.gateway;
    const { status, busy, pending, unknown } = rehearsal.state;
    if (!gateway || !status || busy || pending || unknown || userAction || document.hidden) return;
    if (gateway.recordings && idleForSync(rehearsal.state) && !exportAttempt && performance.now() >= nextSync) {
      void syncRecordings(gateway);
    }
    if (idleForSync(rehearsal.state)) void drive.sync();
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

  async function connect(settings, chooseDevice = false) {
    if (connecting) return;
    const attempt = new AbortController();
    const { signal } = attempt;
    connecting = attempt;
    joinMessage = '';
    const setup = {
      ...settings,
      onApproval() {
        if (connecting !== attempt) return;
        joinMessage = 'Bestätige die WLAN-Verbindung im Headset.';
        paint();
      }
    };
    try {
      const selection = chooseDevice ? selectQuest(navigator.bluetooth) : null;
      paint();
      const join = (endpoint, discovery) => joinQuest({ endpoint, serial: discovery?.owner.serial ?? settings.serial,
        device: settings.device, build: settings.build, enrollment: settings.enrollment, discovery: discovery?.owner, signal });
      let peer = selection ? await selection : pairingDevice;
      let gateway;
      let endpoint = settings.endpoint;
      if (endpoint && !peer) gateway = await join(endpoint).catch(() => null);
      if (!gateway) {
        signal.throwIfAborted();
        peer ??= rememberedDevice;
        if (!peer) {
          const allowed = await navigator.bluetooth?.getDevices?.() ?? [];
          peer = allowed.find(device => device.id === settings.bluetoothId);
        }
        if (!peer) throw Error('Erlaube den Bluetooth-Zugriff auf die Quest erneut.');
        const discovery = await readQuest(peer, signal, setup);
        for (const candidate of discovery.endpoints) {
          endpoint = candidate;
          try { gateway = await join(endpoint, discovery); break; }
          catch (error) {
            signal.throwIfAborted();
            joinMessage = error.message;
          }
        }
        if (!gateway && settings.hotspot) {
          const hotspot = await readQuest(peer, signal, { ...setup, ...discovery.owner, forceHotspot: true });
          endpoint = hotspot.endpoints[0];
          gateway = await join(endpoint, hotspot);
        }
        if (!gateway) throw Error(joinMessage || 'Die Quest antwortet nicht im WLAN.');
      }
      if (signal.aborted || connecting !== attempt) { gateway.close(); return; }
      rememberPair({ serial: gateway.owner.serial, device: gateway.owner.device,
        build: gateway.initial.build.packaging_fingerprint, enrollment: settings.enrollment, hotspot: settings.hotspot,
        bluetoothId: peer?.id ?? settings.bluetoothId, endpoint });
      if (peer) rememberedDevice = peer;
      pairingDevice = null;
      form.reset();
      autoConnect = true;
      canRetry = false;
      joinMessage = '';
      if (!sameQuest(rehearsal.gateway?.owner, gateway.owner)) clearArchives();
      rehearsal.attach(gateway);
      void drive.setOwner(gateway.recordings ? gateway.owner : null);
      showTrace(null);
      if (rehearsal.state.pending) await rehearsal.refresh();
    } catch (error) {
      if (connecting !== attempt) return;
      canRetry = true;
      joinMessage = error instanceof TypeError || ['TimeoutError', 'NotFoundError', 'SecurityError'].includes(error?.name)
        ? 'Quest nicht erreichbar. Prüfe Bluetooth, WLAN und den Fernmodus.'
        : error?.name === 'Error' ? error.message : 'Die Quest konnte nicht verbunden werden.';
    } finally {
      if (connecting === attempt) {
        nextConnect = performance.now() + 8000;
        connecting = null;
        paint();
      }
    }
  }

  function rememberPair(settings) {
    const serialized = JSON.stringify(settings);
    if (serialized !== JSON.stringify(savedPair) || pairWarning) {
      try {
        localStorage.setItem('atmungsraum-quest-pair', serialized);
        pairWarning = '';
      }
      catch { pairWarning = 'Die Verbindung kann in diesem Browser nicht gespeichert werden.'; }
    }
    savedPair = settings;
  }

  async function beginPairing() {
    if (connecting) return;
    const attempt = new AbortController();
    connecting = attempt;
    joinMessage = '';
    try {
      const selection = selectQuest(navigator.bluetooth);
      paint();
      const peer = await selection;
      if (attempt.signal.aborted) return;
      pairingDevice = peer;
    } catch (error) {
      if (!attempt.signal.aborted) joinMessage = error?.name === 'NotFoundError'
        ? 'Wähle die Quest aus, um sie zu verbinden.' : error.message;
    } finally {
      if (connecting === attempt) {
        connecting = null;
        paint();
        if (pairingDevice) byId('quest-controller').focus({ preventScroll: true });
      }
    }
  }

  async function act(action) {
    if (userAction) return;
    ecgAttempt?.abort();
    exportAttempt?.abort();
    drive.cancel();
    nextSync = 0;
    const attempt = new AbortController();
    userAction = attempt;
    const selected = scenario.parentElement.hidden ? '' : scenario.value;
    try { await rehearsal.run(action, attempt.signal, condition.value, participantId, selected); }
    finally { if (userAction === attempt) userAction = null; }
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const controller = byId('quest-controller').value;
    const key = byId('quest-key').value;
    const ssid = byId('quest-hotspot-ssid').value;
    const passphrase = byId('quest-hotspot-password').value;
    void connect({ enrollment: { controller, key }, hotspot: ssid || passphrase ? { ssid, passphrase } : undefined });
  });
  participantForm.addEventListener('input', () => {
    try { text(participantOutput, participantIdentity(Object.fromEntries(new FormData(participantForm))).id); }
    catch { text(participantOutput, '—'); }
    if (participantId) {
      participantId = '';
      paint();
    }
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
  find.addEventListener('click', () => {
    if (!savedPair) return void beginPairing();
    autoConnect = true;
    void connect(savedPair, true);
  });
  repair.addEventListener('click', beginPairing);
  polarConnect.addEventListener('click', () => act('polar'));
  condition.addEventListener('change', () => paint());
  play.addEventListener('click', () => act('play'));
  recenter.addEventListener('click', () => {
    if (!document.hidden && recenter.getAttribute('aria-disabled') === 'false') void act('recenter');
  });
  finish.addEventListener('click', () => act('finish'));
  acknowledge.addEventListener('click', () => rehearsal.acknowledgeIdle());
  retry.addEventListener('click', () => { nextSync = 0; void tick(); });
  archivePicker.addEventListener('change', selectArchive);
  function detach() {
    clearArchives();
    void drive.setOwner(null);
    rehearsal.attach(null);
  }
  disconnect.addEventListener('click', () => { autoConnect = false; detach(); find.focus(); });
  const timer = setInterval(tick, 1000);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      userAction?.abort();
      ecgAttempt?.abort();
      exportAttempt?.abort();
      drive.cancel();
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
  window.addEventListener('pageshow', () => { nextConnect = 0; void tick(); });
  window.addEventListener('online', () => { nextConnect = 0; void tick(); });
  paint();
  void tick();
}
