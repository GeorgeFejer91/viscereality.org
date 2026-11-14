import { selectQuest, readQuest, joinQuest, sameQuest } from './quest-gateway.js?v=8';
import { canRecenter, polarSetup, QuestRehearsal, rehearsalActions, rehearsalPolarAllowed, rehearsalScenarios } from './quest-rehearsal.js?v=20';
import { ecgTrace, questRecording, questRecordings, questPrefixBatches } from './quest-data.js?v=25';
import { openQuestRecordingStore } from './quest-recording-store.js?v=4';
import { mountQuestDrive } from './quest-drive-ui.js?v=5';
import { BreathingSetup } from './breathing-setup.js?v=2';

const byId = (id) => document.getElementById(id);
const text = (element, value) => { if (element.textContent !== value) element.textContent = value; };
const archiveDate = new Intl.DateTimeFormat('de-DE', { dateStyle: 'short', timeStyle: 'medium' });
const liveSync = new URLSearchParams(location.search).get('live-sync') !== 'off';

function idleForSync({ status, busy, pending, unknown }) {
  return status?.p === 'IDLE' && !status.rec && !status.pending && !busy && !pending && !unknown;
}
function activeForSync({ status, busy, pending, unknown }) {
  return liveSync && status?.live_recording === 1 && status.rec && !status.pending && !busy && !pending && !unknown
    && ['RUNNING', 'PAUSED'].includes(status.p);
}
const labels = {
  IDLE: 'Quest bereit.', ARMED: 'Probelauf vorbereitet.', RUNNING: 'Probelauf läuft.',
  PAUSED: 'Probelauf pausiert.', FINALIZING: 'Quest beendet den Probelauf.',
  SAVING: 'Quest speichert den Probelauf.', ERROR: 'Die Quest meldet einen Fehler.',
};

export function mountQuestRehearsal(participantIdentity, savedPair) {
  const drive = mountQuestDrive();
  const questScreen = byId('quest');
  const preview = byId('breathing-preview');
  const previewTitle = byId('preview-title');
  const previewCircle = byId('preview-circle');
  const previewStatus = byId('preview-status');
  const previewInput = byId('preview-input');
  const previewDetail = byId('preview-detail');
  const previewProgressGroup = byId('preview-progress-group');
  const previewProgress = byId('preview-progress');
  const previewProgressText = byId('preview-progress-text');
  const previewStart = byId('preview-start');
  const previewCalibrate = byId('preview-calibrate');
  const previewController = byId('preview-controller');
  const previewContinue = byId('preview-continue');
  const previewAcknowledge = byId('preview-acknowledge');
  const find = byId('quest-find');
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
  let previewAccepted = '';
  let attemptedSetupStart = '';
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
  let liveAttempt, liveGeneration, nextLive = 0;
  let archiveRead = 0;
  let connecting = null;
  let rememberedDevice = null;
  let nextConnect = 0;
  let autoConnect = true;
  let pairWarning = '';
  let userAction = null;
  let joinMessage = '';
  let storage;
  try { storage = sessionStorage; } catch { /* The current connection still works. */ }
  const breathing = new BreathingSetup({ storage });
  const rehearsal = new QuestRehearsal({ storage, onChange: paint });

  function paint({ status, busy, pending, unknown, message } = rehearsal.state) {
    const actions = rehearsalActions(status, condition.value);
    const active = Boolean(status?.rec);
    const requiresParticipant = status?.participant_arm === 1;
    const armedWithParticipant = Boolean(rehearsal.gateway?.participantArm);
    const waiting = Boolean(connecting || userAction || busy || pending || status?.pending || breathing.busy || breathing.pending);
    const recenterAvailable = Boolean(rehearsal.gateway?.recenter && canRecenter(status) && !waiting && !unknown);
    recenter.hidden = !status;
    recenter.setAttribute('aria-disabled', String(!recenterAvailable));
    text(recenterNote, recenterAvailable ? 'Richtet die VR-Szene an der aktuellen Blickrichtung aus.'
      : 'Verbinde dich mit einer Quest im Fernmodus, um die VR-Szene zu zentrieren.');
    const connectingFocus = document.activeElement === find;
    const hadFinishFocus = document.activeElement === finish;
    if (status && !connecting) joinMessage = '';
    find.hidden = Boolean(status);
    find.setAttribute('aria-disabled', String(Boolean(connecting)));
    const hint = !status ? (savedPair?.enrollment && autoConnect
      ? 'Warte auf die gespeicherte Quest.'
      : 'Starte die Quest im Fernmodus und schalte Bluetooth am Smartphone ein.')
      : status.h === 'panel' ? 'Schließe das Bedienfeld in der Quest und kehre zur VR-Szene zurück.'
      : !rehearsalPolarAllowed(status) ? 'Diese Sitzung ist nicht für den Probelauf freigegeben.'
      : requiresParticipant && !armedWithParticipant ? 'Dieses Smartphone ist für den Probelauf noch nicht freigegeben.'
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
    const ready = setup?.p && setup.b && setup.c && setup.e && setup.a;
    const idle = status?.p === 'IDLE' && !active;
    const participantComplete = Boolean(!requiresParticipant || armedWithParticipant && participantId);
    const setupComplete = Boolean(ready && participantComplete);
    const breath = breathing.update(rehearsal.gateway, { status, busy, pending, unknown }, setupComplete);
    if (!breath.actions.continue || previewAccepted !== breath.acceptance) previewAccepted = '';
    const selectionAllowed = idle && setupComplete && Boolean(previewAccepted) && breath.actions.continue;
    const recoveringSetup = breath.pending || breath.unknown;
    const showPreview = idle && (participantComplete || recoveringSetup) && (setupComplete || !preview.hidden || breath.setup?.attempt > 0 || recoveringSetup)
      && !previewAccepted && status.h === 'vr-preparation'
      && rehearsalPolarAllowed(status) && !unknown && (!waiting || !preview.hidden || recoveringSetup);
    paintBreathingSetup(breath, waiting);
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
      : ready ? 'Polar H10 verbunden. Quest empfängt EKG und Beschleunigungsdaten.'
      : setup.c ? 'Polar H10 verbunden. Warte auf beide Messsignale.'
      : 'Lege den Gurt an. Quest verbindet sich mit dem Polar H10.');
    polarConnect.disabled = waiting || unknown || !setup?.p || !setup.b || setup.sc || ready;
    if (polar.hidden && hadPolarFocus) statusText.focus({ preventScroll: true });
    play.hidden = !status || idle && !selectionAllowed;
    const available = rehearsal.gateway?.scenarioArm ? rehearsalScenarios(status) : [];
    const showScenario = selectionAllowed && available.length > 0;
    if (!showScenario && document.activeElement === scenario) statusText.focus({ preventScroll: true });
    scenario.parentElement.hidden = !showScenario;
    scenario.disabled = !showScenario || waiting || unknown;
    if (showScenario) {
      for (const option of scenarioOptions) option.hidden = option.disabled = !available.includes(option.value);
      if (!available.includes(scenario.value)) scenario.value = scenarioOptions.find(option => !option.disabled).value;
    }
    const showCondition = selectionAllowed && status.ecg_preview === 1;
    if (!showCondition && document.activeElement === condition) statusText.focus({ preventScroll: true });
    condition.parentElement.hidden = !showCondition;
    condition.disabled = waiting || unknown;
    text(conditionName, showScenario ? 'VR-Bedingung' : 'Szenario');
    text(condition.options[0], showScenario ? 'Bedingung A' : 'Szenario A');
    text(condition.options[1], showScenario ? 'Bedingung B' : 'Szenario B');
    play.disabled = waiting || unknown || !actions.play || actions.play === 'arm' && !selectionAllowed;
    text(play.firstElementChild, status?.p === 'RUNNING' ? 'Pause' : status?.p === 'PAUSED' ? 'Fortsetzen' : 'Starten');
    finish.hidden = !active;
    if (finish.hidden && hadFinishFocus) statusText.focus({ preventScroll: true });
    finish.disabled = waiting || unknown || !actions.finish;
    acknowledge.hidden = !rehearsal.canAcknowledgeIdle();
    acknowledge.disabled = busy;
    disconnect.hidden = !status || active || Boolean(connecting || userAction || busy);
    if (!idleForSync(rehearsal.state)) {
      exportAttempt?.abort();
      nextSync = 0;
    }
    if (!activeForSync(rehearsal.state)) liveAttempt?.abort();
    if (!idleForSync(rehearsal.state) && !activeForSync(rehearsal.state)) drive.cancel();
    data.hidden = status?.ecg_preview !== 1;
    if (data.hidden) showTrace(null);
    exportStatus.hidden = !exportMessage || !status;
    text(exportStatus, exportMessage);
    retry.hidden = !rehearsal.gateway?.recordings || status?.p !== 'IDLE';
    retry.disabled = waiting || unknown || Boolean(exportAttempt);
    archivePicker.parentElement.hidden = !status || !archivePicker.options.length;
    download.hidden = !status || !download.hasAttribute('href');
    if (showPreview !== !preview.hidden) {
      const hadPreviewFocus = preview.contains(document.activeElement);
      preview.hidden = !showPreview;
      questScreen.hidden = showPreview;
      if (showPreview) previewTitle.focus();
      else {
        if (hadPreviewFocus || document.activeElement === document.body) {
          if (showCondition) condition.focus();
          else if (showScenario) scenario.focus();
          else statusText.focus();
        }
      }
    }
  }

  function paintBreathingSetup(view, waiting) {
    const native = view.setup;
    previewCircle.hidden = !view.available;
    if (view.available) previewCircle.style.transform = `scale(${view.radius})`;
    text(previewStatus, !view.available ? 'Warte auf den Atemtakt der Quest.'
      : !native.playing ? 'Atemschleife auf der Quest angehalten.' : `Atemtakt: ${view.phaseLabel}`);
    const measured = view.inputPhaseLabel ? ` · Erkannt: ${view.inputPhaseLabel}` : '';
    const volume = view.inputVolume === null ? '' : ` · ${Math.round(view.inputVolume * 100)} %`;
    text(previewInput, view.available ? `Eingabe: ${view.inputLabel}${measured}${volume}` : 'Eingabe auf der Quest nicht verfügbar.');
    const collecting = ['waiting', 'collecting'].includes(native?.calibration);
    const detail = view.updateRequired ? 'Quest-App aktualisieren: Dieser Build unterstützt die Atemeinrichtung noch nicht.'
      : view.actions.acknowledge ? 'Die Quest-Verbindung wurde erneuert. Prüfe den aktuellen Zustand und bestätige mit „Quest geprüft“. Kein Befehl wird wiederholt.'
      : view.unknown ? 'Ergebnis unklar. Prüfe die Quest. Die Aktion wird nicht wiederholt; warte auf den bestätigten Status.'
      : view.pending ? 'Warte auf Bestätigung der Quest …'
      : !view.available ? 'Kein aktueller Atemstatus verfügbar. Prüfe die Verbindung zur Quest.'
      : view.failure && native.input === 'automatic' ? `Einrichtung fehlgeschlagen: ${view.failure}. Automatic bleibt aktiv. Du kannst erneut einrichten oder Controller State wählen.`
      : view.blockedReason ? view.blockedReason
      : view.message ? view.message
      : native.calibration === 'waiting' ? 'Automatic ist aktiv. Die Einrichtung beginnt mit dem nächsten vollständigen Atemzyklus.'
      : native.calibration === 'collecting' ? 'Automatic ist aktiv. Folge dem Atemtakt für zwei vollständige Atemzyklen.'
      : view.actions.continue ? 'Prüfe die Bewegung im Headset. Wenn die Eingabe passt, gehe weiter zur Auswahl.'
      : !native.playing ? 'Warte auf die Atemschleife der Quest.'
      : 'Die Atemschleife läuft. Richte die Atmung ein oder wähle Controller State mit fester Ausrichtung.';
    text(previewDetail, detail);
    previewProgressGroup.hidden = !view.available || !(collecting || native?.attempt > 0);
    previewProgress.value = view.progress.value;
    text(previewProgressText, `${new Intl.NumberFormat('de-DE', { maximumFractionDigits: 2 }).format(view.progress.value)} von 2 Atemzyklen`);
    previewStart.hidden = native?.stage !== 'idle';
    previewStart.disabled = waiting || !view.actions.start;
    previewCalibrate.disabled = waiting || !view.actions.calibrate;
    text(previewCalibrate.firstElementChild, native?.attempt > 0 ? 'Erneut einrichten' : 'Atmung einrichten');
    previewController.disabled = waiting || !view.actions.controller;
    previewContinue.disabled = waiting || !view.actions.continue;
    previewAcknowledge.hidden = !view.actions.acknowledge;
    previewAcknowledge.disabled = !view.actions.acknowledge;
  }

  function resetPreview() {
    previewAccepted = '';
  }

  async function setupAct(op) {
    if (userAction || connecting || document.hidden || !breathing.view?.actions[op]) return;
    const attempt = new AbortController();
    userAction = attempt;
    const request = breathing.run(op, attempt.signal);
    paint();
    try { await request; }
    finally {
      if (userAction === attempt) userAction = null;
      await rehearsal.refresh();
      await breathing.refresh();
      paint();
    }
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
    liveAttempt?.abort();
    liveGeneration = null;
    nextLive = 0;
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
            const archive = await questRecording(gateway, id, signal, { includeSensorBundle: true, store });
            if (participantCsv && !archive.download.name.endsWith('.csv')) throw Error('Quest-CSV fehlt.');
            signal.throwIfAborted();
            await store.finalizePrefix(id, archive);
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

  async function syncLive(gateway, generation) {
    const attempt = new AbortController();
    liveAttempt = attempt;
    const { signal } = attempt;
    try {
      await navigator.locks.request('atmungsraum-quest-sync/' + gateway.owner.device,
        { ifAvailable: true }, async lock => {
          if (!lock) return;
          signal.throwIfAborted();
          const store = await openQuestRecordingStore(gateway.owner);
          try {
            for await (const prefix of questPrefixBatches(gateway, store, signal)) {
              signal.throwIfAborted();
              if (rehearsal.state.status?.g !== generation) break;
              if (!exportMessage.startsWith('Laufende Quest-Aufnahme')) {
                exportMessage = 'Laufende Quest-Aufnahme wird auf diesem Gerät gespeichert.';
                paint();
              }
              void drive.sync(prefix);
            }
          } finally { store.close(); }
        });
    } catch {
      if (!signal.aborted) {
        nextLive = performance.now() + 3000;
        exportMessage = 'Live-Abgleich unterbrochen. Die App versucht es erneut.';
        paint();
      }
    } finally {
      if (liveAttempt === attempt) liveAttempt = null;
    }
  }

  async function tick() {
    if (document.hidden || connecting || userAction) return;
    await rehearsal.refresh();
    if (document.hidden || connecting || userAction) return;
    await breathing.refresh();
    paint();
    if (document.hidden || connecting || userAction) return;
    if (!rehearsal.state.status && savedPair?.enrollment && autoConnect && performance.now() >= nextConnect) {
      void connect(savedPair);
      return;
    }
    const setupStart = breathing.view?.setup;
    const setupStartKey = setupStart && `${breathing.view.acceptance}/${setupStart.g}`;
    if (breathing.view?.actions.start && attemptedSetupStart !== setupStartKey) {
      attemptedSetupStart = setupStartKey;
      void setupAct('start');
      return;
    }
    const gateway = rehearsal.gateway;
    const { status, busy, pending, unknown } = rehearsal.state;
    if (!gateway || !status || busy || pending || unknown || userAction || document.hidden) return;
    if (gateway.recordings && idleForSync(rehearsal.state) && !exportAttempt && performance.now() >= nextSync) {
      void syncRecordings(gateway);
    }
    if (idleForSync(rehearsal.state)) void drive.sync();
    if (activeForSync(rehearsal.state)) {
      if (liveGeneration !== status.g) {
        liveGeneration = status.g;
        nextLive = 0;
        drive.schedule();
      }
      if (!liveAttempt && performance.now() >= nextLive) void syncLive(gateway, status.g);
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

  async function connect(settings, chooseDevice = false) {
    if (connecting) return;
    settings ??= {};
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
        device: settings.device, build: chooseDevice ? undefined : settings.build,
        enrollment: settings.enrollment, discovery: discovery?.owner, signal });
      let peer = selection ? await selection : null;
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
        if (!settings.enrollment) {
          throw Error('Dieses Smartphone ist für diese Quest noch nicht freigegeben.');
        }
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
      if (savedPair && (savedPair.serial !== gateway.owner.serial || savedPair.device !== gateway.owner.device)) {
        resetPreview();
      }
      rememberPair({ serial: gateway.owner.serial, device: gateway.owner.device,
        build: gateway.initial.build.packaging_fingerprint, enrollment: settings.enrollment, hotspot: settings.hotspot,
        bluetoothId: peer?.id ?? settings.bluetoothId, endpoint });
      if (peer) rememberedDevice = peer;
      autoConnect = true;
      joinMessage = '';
      if (!sameQuest(rehearsal.gateway?.owner, gateway.owner)) clearArchives();
      rehearsal.attach(gateway);
      void drive.setOwner(gateway.recordings ? gateway.owner : null);
      showTrace(null);
      if (rehearsal.state.pending) await rehearsal.refresh();
    } catch (error) {
      if (connecting !== attempt) return;
      joinMessage = error?.name === 'NotFoundError' && chooseDevice
        ? 'Schalte Bluetooth ein und wähle die Quest aus.'
        : error instanceof TypeError || ['TimeoutError', 'NotFoundError', 'SecurityError'].includes(error?.name)
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

  async function act(action) {
    if (userAction) return;
    ecgAttempt?.abort();
    exportAttempt?.abort();
    liveAttempt?.abort();
    drive.cancel();
    nextSync = 0;
    const attempt = new AbortController();
    userAction = attempt;
    const selected = scenario.parentElement.hidden ? '' : scenario.value;
    try { await rehearsal.run(action, attempt.signal, condition.value, participantId, selected); }
    finally { if (userAction === attempt) userAction = null; }
  }

  participantForm.addEventListener('input', () => {
    try { text(participantOutput, participantIdentity(Object.fromEntries(new FormData(participantForm))).id); }
    catch { text(participantOutput, '—'); }
    if (participantId) {
      participantId = '';
      resetPreview();
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
    autoConnect = true;
    void connect(savedPair, true);
  });
  polarConnect.addEventListener('click', () => act('polar'));
  previewStart.addEventListener('click', () => setupAct('start'));
  previewCalibrate.addEventListener('click', () => setupAct('calibrate'));
  previewController.addEventListener('click', () => setupAct('controller'));
  previewAcknowledge.addEventListener('click', () => {
    if (breathing.acknowledgeRetired()) { resetPreview(); paint(); }
  });
  previewContinue.addEventListener('click', () => {
    if (preview.hidden || previewContinue.disabled || !breathing.view?.actions.continue) return;
    previewAccepted = breathing.view.acceptance;
    paint();
  });
  condition.addEventListener('change', () => paint());
  play.addEventListener('click', () => {
    if (!play.hidden && !play.disabled) void act('play');
  });
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
  disconnect.addEventListener('click', () => {
    autoConnect = false;
    resetPreview();
    detach();
    find.focus();
  });
  const timer = setInterval(tick, 1000);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      userAction?.abort();
      ecgAttempt?.abort();
      exportAttempt?.abort();
      liveAttempt?.abort();
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
  window.addEventListener('online', () => { nextConnect = nextLive = 0; drive.schedule(); void tick(); });
  paint();
  void tick();
}
