import { operatorSteps, navigateOperatorView, resolveOperatorView, operatorCompatibilityWarning,
  restoreOperatorView, operatorSessionHeading } from './operator-layout.js?v=3';
import { selectQuest, readQuest, rediscoverQuest, joinQuest, sameQuest, operationId,
  controllerAccessState } from './quest-gateway.js?v=13';
import { pairPhone } from './quest-pairing.js?v=2';
import { canRecenter, polarSetup, polarReadiness, QuestRehearsal, rehearsalActions, rehearsalPolarAllowed, rehearsalScenarios } from './quest-rehearsal.js?v=30';
import { ecgTrace, questRecording, questRecordings, questPrefixBatches } from './quest-data.js?v=26';
import { openQuestRecordingStore } from './quest-recording-store.js?v=4';
import { mountQuestDrive } from './quest-drive-ui.js?v=5';
import { BreathingSetup } from './breathing-setup.js?v=10';
import { LiveBreathDisplay } from './live-breath-display.js?v=5';
import { LiveEcgDisplay } from './live-ecg-display.js?v=1';
import { KioskControl } from './kiosk-control.js?v=3';
import { ApkRelease } from './quest-release.js?v=1';
import { nativeSessionAllowed, metaUpdateView } from './meta-update.js?v=2';
import { MetaUpdateControl } from './meta-update-control.js?v=1';

const byId = (id) => document.getElementById(id);
const text = (element, value) => { if (element.textContent !== value) element.textContent = value; };
const needsDialogFocus = dialog => !document.activeElement || document.activeElement === document.body
  || dialog.contains(document.activeElement);
const mainCondition = 'condition-a';
const runTime = ms => {
  const seconds = Math.floor(ms / 1000);
  return [Math.floor(seconds / 60), seconds % 60].map(part => String(part).padStart(2, '0')).join(':');
};
const archiveDate = new Intl.DateTimeFormat('de-DE', { dateStyle: 'short', timeStyle: 'medium' });
const liveSync = new URLSearchParams(location.search).get('live-sync') !== 'off';

function idleForSync({ status, busy, pending, unknown }) {
  return status?.p === 'IDLE' && nativeSessionAllowed(status) && !status.rec && !status.pending && !busy && !pending && !unknown;
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
  const preview = byId('breathing-preview');
  const versionPanel = byId('quest-version');
  const previewCircle = byId('preview-circle');
  const actualCircle = byId('preview-actual-circle');
  const actualStatus = byId('preview-actual-status');
  const actualVolume = byId('preview-actual-volume');
  const previewStatus = byId('preview-status');
  const previewInputLabel = byId('preview-input-label');
  const previewInputVolume = byId('preview-input-volume');
  const previewDetail = byId('preview-detail');
  const previewProgressGroup = byId('preview-progress-group');
  const previewProgress = byId('preview-progress');
  const previewProgressFill = byId('preview-progress-fill');
  const previewProgressText = byId('preview-progress-text');
  const previewCalibrate = byId('preview-calibrate');
  const previewRestart = byId('preview-restart');
  const previewRetry = byId('preview-retry');
  let calibrationBaseline;
  const previewPersonalReturn = byId('preview-personal-return');
  const previewPolarFixed = byId('preview-polar-fixed');
  const previewPolarAdaptive = byId('preview-polar-adaptive');
  const previewController = byId('preview-controller');
  const previewContinue = byId('preview-continue');
  const previewAcknowledge = byId('preview-acknowledge');
  const previewSwitch = byId('preview-switch');
  const find = byId('quest-find');
  const statusText = byId('quest-status');
  const pairing = byId('quest-pairing');
  const pairingCode = byId('quest-pairing-code');
  const pairingConfirm = byId('quest-pairing-confirm');
  const pairingCancel = byId('quest-pairing-cancel');
  const controlPanel = byId('quest-control-access');
  const controlStatus = byId('quest-control-status');
  const controlClaim = byId('quest-control-claim');
  const controlRelease = byId('quest-control-release');
  let pairingChoice = null;
  const playback = byId('quest-playback');
  const clock = byId('quest-clock');
  const progress = byId('quest-progress');
  const recenter = byId('recenter-vr');
  const recenterNote = byId('recenter-vr-note');
  const polar = byId('quest-polar');
  const polarStatus = byId('quest-polar-status');
  const polarConnect = byId('quest-polar-connect');
  const play = byId('quest-play');
  const finish = byId('quest-finish');
  const acknowledge = byId('quest-acknowledge');
  const disconnect = byId('quest-disconnect');
  const sessionChoices = byId('session-choices');
  const scenarioChoices = byId('quest-scenario');
  const scenarioOptions = Array.from(scenarioChoices.querySelectorAll('input'));
  const condition = byId('quest-condition');
  const legacyOptions = byId('quest-legacy-options');
  const legacyScenarios = byId('quest-legacy-scenarios');
  const participant = byId('quest-participant');
  const participantLabel = byId('quest-participant-label');
  const participantForm = byId('patient-form');
  const participantFields = byId('patient-fields');
  const participantSubmit = byId('patient-next');
  const participantOutput = byId('patient-id');
  let participantId = '';
  let previewAccepted = '';
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
  let nextSavedProbe = 0;
  let discoveryInProgress = false;
  let bluetoothCheck = null;
  let bridgeFailed = false;
  const diagnostics = byId('connection-diagnostic');
  const driveDialog = byId('drive-dialog');
  const driveOpen = byId('drive-open');
  const drivePanel = byId('quest-drive');
  const driveStatus = byId('quest-drive-status');
  const driveStatusHost = byId('drive-status-host');
  function driveActionsHost() {
    return byId(displayedStep === 'recordings' ? 'recordings-drive-actions' : 'connection-drive-actions');
  }
  driveOpen.addEventListener('click', () => {
    if (drivePanel.hidden) return;
    driveDialog.querySelector('.screen-body').append(driveStatus);
    drivePanel.open = true;
    driveDialog.showModal();
    byId('quest-drive-client').focus({ preventScroll: true });
  });
  drivePanel.addEventListener('toggle', () => {
    if (driveDialog.open && !drivePanel.open) driveDialog.close();
  });
  driveDialog.addEventListener('close', () => {
    if (driveDialog.open) return;
    drivePanel.open = false;
    if (driveDialog.contains(driveStatus)) {
      (displayedStep === 'recordings' ? byId('recording-files') : driveStatusHost).prepend(driveStatus);
    }
    if (needsDialogFocus(driveDialog) && !driveOpen.hidden && driveOpen.getClientRects().length) {
      driveOpen.focus({ preventScroll: true });
    }
  });
  const checks = Array.from(document.querySelectorAll('.connection-checks button'));
  const diagnosticState = new Map();
  let diagnosticTrigger;
  diagnostics.addEventListener('close', () => {
    if (!needsDialogFocus(diagnostics)) return;
    if (diagnosticTrigger?.getClientRects().length && !diagnosticTrigger.disabled) diagnosticTrigger.focus({ preventScroll: true });
    else if (!find.hidden) find.focus({ preventScroll: true });
    else byId('quest-title').focus({ preventScroll: true });
  });
  for (const button of checks) button.addEventListener('click', () => {
    const entry = diagnosticState.get(button.id);
    diagnosticTrigger = button;
    text(byId('connection-diagnostic-title'), entry.label);
    text(byId('connection-diagnostic-message'), entry.detail);
    diagnostics.showModal();
  });

  function paintCheck(name, confirmed, label, detail, neutral = 'wird geprüft') {
    const button = byId('check-' + name);
    diagnosticState.set(button.id, { label, detail });
    button.dataset.check = confirmed === null ? 'checking' : confirmed ? 'ok' : 'unconfirmed';
    button.disabled = confirmed !== false;
    const accessible = label + (confirmed === null ? ': ' + neutral : confirmed ? ': bestätigt' : ': nicht bestätigt');
    button.title = accessible;
    button.setAttribute('aria-label', accessible);
    text(button.querySelector('.check-symbol'), confirmed === null ? '' : confirmed ? '✓' : '!');
    if (diagnostics.open && diagnosticTrigger === button) text(byId('connection-diagnostic-message'), detail);
  }
  let autoConnect = true;
  let pairWarning = '';
  let userAction = null;
  let joinMessage = '';
  let storage;
  try { storage = sessionStorage; } catch { /* The current connection still works. */ }
  const breathing = new BreathingSetup({ storage });
  const kiosk = new KioskControl({ storage });
  const metaControl = new MetaUpdateControl({ storage });
  const apkRelease = new ApkRelease({ onChange: () => paint() });
  const breathDisplay = new LiveBreathDisplay();
  const ecgDisplay = new LiveEcgDisplay();
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let animationFrame = 0;
  let ecgOwner = '';
  let ecgGeneration = null;
  let ecgSensor = '';
  let breathTimer;
  let versionGeometryKey = '';
  let updateGeometryKey = '';
  const rehearsal = new QuestRehearsal({ storage, onChange: paint });

  let localView = restoreOperatorView(location.hash, history.state);
  let displayedStep = '';
  let wasActive = false;
  let settingsReturnFocus = null;

  function livePreviewVisible() {
    return ['calibration', 'tracking'].includes(displayedStep) && location.hash !== '#daten';
  }
  function navigateLocal(event) {
    if (event.step === 'settings') settingsReturnFocus = document.activeElement;
    localView = navigateOperatorView({ ...localView, step: displayedStep || localView.step }, event);
    const hash = `#${localView.step}`;
    if (location.hash !== hash) history.pushState({ operatorStep: localView.step, operatorReturnTo: localView.returnTo }, '', hash);
  }
  function paintOperatorLayout(context) {
    const nativeActive = context.active || Boolean(context.status && ['ARMED', 'RUNNING', 'PAUSED', 'FINALIZING', 'SAVING'].includes(context.status.p));
    if (location.hash === '#daten' && !nativeActive && !context.recovery) {
      if (diagnostics.open) diagnostics.close();
      if (driveDialog.open) driveDialog.close();
      for (const id of Object.values(operatorSteps)) if (!byId(id).hidden) byId(id).hidden = true;
      if (byId('daten').hidden) byId('daten').hidden = false;
      document.documentElement.dataset.operatorStep = 'daten';
      displayedStep = '';
      return;
    }
    // Rewriting an unchanged screen flag triggers the layout observer, whose
    // bounded-layout retry can clamp document scroll before reflow completes.
    if (!byId('daten').hidden) byId('daten').hidden = true;
    if (context.status) {
      if (wasActive && !nativeActive && context.status.p === 'IDLE') localView = { ...localView, step: 'recordings' };
      wasActive = nativeActive;
    }
    const step = resolveOperatorView(localView.step, { ...context, connected: Boolean(context.status),
      participantRequired: context.requiresParticipant, active: nativeActive });
    if (localView.step === 'calibration' && step === 'tracking') localView = { ...localView, step };
    const page = byId(operatorSteps[step]);
    const body = page.querySelector('.operator-body');
    const move = (id, host) => { const element = byId(id); if (element.parentElement !== host) host.prepend(element); };
    const activeControls = step === 'running' || step === 'settings' && nativeActive;
    if (['connect', 'signals'].includes(step) || activeControls) move('quest-data', body);
    if (['connect', 'signals'].includes(step)) move('quest-polar', body);
    byId('corner-actions').hidden = step !== 'connect' || byId('install-app').hidden;
    driveOpen.hidden = drivePanel.hidden;
    if (step !== 'connect' && diagnostics.open) diagnostics.close();
    if ((!['connect', 'recordings'].includes(step) || drivePanel.hidden) && driveDialog.open) driveDialog.close();
    const recordingStatusHost = activeControls ? body : byId('recording-files');
    move('quest-export-status', recordingStatusHost);
    if (!driveDialog.open) move('quest-drive-status', activeControls || step === 'recordings' ? recordingStatusHost : driveStatusHost);
    const controllerHost = byId('preview-controller-host');
    const fallback = byId('preview-controller-fallback');
    const offerFallback = context.polarAttempts >= 2 && ['calibration', 'tracking'].includes(step);
    move('preview-controller', offerFallback ? fallback : controllerHost);
    fallback.hidden = !offerFallback;
    if (activeControls) move('quest-playback', body);
    // Only transitions move shared nodes; polling never detaches a focused editor.
    if (step !== displayedStep) {
      if (['input', 'calibration', 'tracking', 'recovery'].includes(step)) move('operator-input-status', body);
      if (['calibration', 'tracking'].includes(step)) move('operator-live', body);
      move('operator-common', body);
      for (const id of Object.values(operatorSteps)) byId(id).hidden = id !== page.id;
      const returning = displayedStep === 'settings' && step !== 'settings';
      displayedStep = step;
      if (['connect', 'recordings'].includes(step)) {
        driveActionsHost().append(driveOpen, driveStatusHost);
      } else if (driveDialog.open) driveDialog.close();
      document.documentElement.dataset.operatorStep = step;
      if (returning && settingsReturnFocus?.isConnected && settingsReturnFocus.getClientRects().length) {
        settingsReturnFocus.focus();
      } else page.querySelector('h1').focus();
    }
    if (step === 'session' || activeControls) {
      move('operator-session-actions', page.querySelector('.actions'));
    }
    const fullConnection = ['connect', 'settings'].includes(step);
    const routineStatus = ['IDLE', 'RUNNING'].includes(context.status?.p) && Object.values(labels).includes(statusText.textContent);
    text(byId('running-screen-title'), operatorSessionHeading(context.status?.p));
    const access = controllerAccessState(context.status);
    statusText.hidden = (!fullConnection || step === 'connect' && access?.role === 'active') && routineStatus;
    byId('quest-control-host').hidden = step !== 'settings' && !context.waitingForControl || !access;
    text(byId('operator-connection-summary'), controllerAccessState(context.status)?.role === 'active'
      ? 'Quest verbunden · Steuerung aktiv' : 'Quest verbunden');
    byId('operator-input-status').hidden = step === 'tracking' && context.inputReady;
    previewProgressGroup.hidden ||= step !== 'calibration';
    byId('operator-connection-summary').hidden = step !== 'connect' || !context.status || context.recovery || !routineStatus;
    byId('quest-control-release').hidden = step !== 'settings' || controllerAccessState(context.status)?.role !== 'active';
    const warning = byId('operator-compatibility');
    const version = byId('quest-version').dataset.versionState;
    warning.hidden = !context.status || !operatorCompatibilityWarning(version) || step === 'settings';
    text(warning, byId('apk-update-status').textContent);
    // An overriding native operation cannot be escaped through utility links.
    for (const control of document.querySelectorAll('[data-operator-go], [data-operator-back], [data-operator-return]')) {
      control.disabled = ['running', 'recovery'].includes(step) && control.dataset.operatorGo !== 'settings';
    }
    const polarAlreadyReady = step === 'connect' && byId('check-polar').dataset.check === 'ok';
    polarStatus.hidden = polarAlreadyReady;
    if (polarAlreadyReady) polarConnect.hidden = true;
    byId('operator-connect-next').disabled = context.status?.wifi_snapshot !== 1 || context.waitingForControl;
    byId('operator-next-session').disabled = !context.status || context.waitingForControl;
  }
  document.addEventListener('click', event => {
    const control = event.target.closest('[data-operator-go], [data-operator-back], [data-operator-return]');
    if (!control || control.disabled) return;
    navigateLocal(control.hasAttribute('data-operator-back') ? { type: 'back' }
      : control.hasAttribute('data-operator-return') ? { type: 'return' }
      : { type: 'go', step: control.dataset.operatorGo });
    paint();
  });
  window.addEventListener('popstate', () => {
    localView = restoreOperatorView(location.hash, history.state);
    paint();
  });
  window.addEventListener('hashchange', () => {
    localView = restoreOperatorView(location.hash, history.state);
    paint();
  });
  byId('recordings-screen').querySelector('a[href="#daten"]').addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    history.pushState({ browserExportsFromRecordings: true }, '', '#daten');
    paint();
  });

  function selectedScenario() {
    return scenarioChoices.querySelector('input:checked:not([disabled])')?.value ?? '';
  }

  function paint({ status, busy, pending, unknown, message } = rehearsal.state) {
    const supportsScenarios = Boolean(rehearsal.gateway?.scenarioArm);
    const available = supportsScenarios ? rehearsalScenarios(status) : [];
    const advertised = status?.scenarios !== undefined;
    const focusedChoice = sessionChoices.contains(document.activeElement) ? document.activeElement : null;
    for (const option of scenarioOptions) {
      option.closest('label').hidden = option.disabled = !available.includes(option.value);
    }
    if (!selectedScenario()) {
      const firstOffered = scenarioOptions.find(option => !option.disabled);
      if (firstOffered) firstOffered.checked = true;
    }
    const scenario = selectedScenario();
    for (const [index, option] of Array.from(condition.options).entries()) {
      option.disabled = status?.a?.[index] !== 'track-ready';
    }
    const [trackA, trackB] = condition.options;
    const mainSelected = scenario === 'haupt-15';
    const actions = rehearsalActions(status, mainSelected ? mainCondition : condition.value);
    const active = Boolean(status?.rec);
    const requiresParticipant = status?.participant_arm === 1;
    const armedWithParticipant = Boolean(rehearsal.gateway?.participantArm);
    const access = controllerAccessState(status);
    const waitingForControl = access?.role === 'waiting';
    const nativeRecovering = status?.setup?.stage === 'recovering';
    const metaUpdate = metaUpdateView(status);
    const common = byId('operator-common');
    const updateKey = status && metaUpdate.advertised ? `${rehearsal.gateway?.owner.device}/${rehearsal.gateway?.owner.epoch}/${rehearsal.gateway?.owner.native_epoch}/${displayedStep}/${common.clientWidth}/${getComputedStyle(common).fontSize}` : '';
    if (updateGeometryKey !== updateKey) {
      common.style.minBlockSize = '';
      updateGeometryKey = updateKey;
    } else if (updateKey && common.getClientRects().length) {
      // Both the native notice and the shared status can shorten at completion.
      // Preserve their combined geometry before writing either text node.
      common.style.minBlockSize = `${Math.max(common.getBoundingClientRect().height, parseFloat(common.style.minBlockSize) || 0)}px`;
    }
    const commandWaiting = Boolean(connecting || userAction || busy || pending || status?.pending
      || breathing.busy || breathing.pending || nativeRecovering);
    const waiting = waitingForControl || commandWaiting;
    controlPanel.hidden = !access;
    controlClaim.hidden = !waitingForControl;
    controlRelease.hidden = access?.role !== 'active';
    controlClaim.disabled = !canChangeControl('claim');
    controlRelease.disabled = !canChangeControl('release');
    text(controlStatus, waitingForControl ? (access.active_phone
      ? `${access.active_phone} steuert die Quest. Warte auf die Freigabe.` : 'Die Quest-Steuerung ist frei.')
      : 'Dieses Smartphone steuert die Quest.');
    const recenterAvailable = Boolean(rehearsal.gateway?.recenter && canRecenter(status) && !waiting && !unknown);
    recenter.hidden = !status;
    recenter.setAttribute('aria-disabled', String(!recenterAvailable));
    text(recenterNote, recenterAvailable ? 'Richtet die VR-Szene an der aktuellen Blickrichtung aus.'
      : 'Verbinde dich mit einer Quest im Fernmodus, um die VR-Szene zu zentrieren.');
    const connectingFocus = document.activeElement === find;
    const hadFinishFocus = document.activeElement === finish;
    if (status && !connecting) joinMessage = '';
    find.hidden = Boolean(status || pairingChoice);
    find.setAttribute('aria-disabled', String(Boolean(connecting)));
    const hint = !status ? (savedPair?.enrollment && autoConnect
      ? 'Warte auf die gespeicherte Quest.'
      : 'Starte die Quest im Fernmodus und schalte Bluetooth am Smartphone ein.')
      : waitingForControl ? (access.active_phone ? 'Ein anderes Smartphone steuert die Quest.' : 'Übernimm die Quest-Steuerung auf diesem Smartphone.')
      : status.h === 'panel' ? 'Schließe das Bedienfeld in der Quest und kehre zur VR-Szene zurück.'
      : !rehearsalPolarAllowed(status) ? 'Diese Sitzung ist nicht für den Probelauf freigegeben.'
      : requiresParticipant && !armedWithParticipant ? 'Dieses Smartphone ist für den Probelauf noch nicht freigegeben.'
      : status.pi?.r === true && status.pi.f !== true && ['ARMED', 'PAUSED'].includes(status.p)
        ? 'Warte auf aktuelle Polar-H10-Daten von der Quest.'
      : status.p === 'IDLE' && !actions.play ? 'Warte auf die VR-Szene und den Probelauf.'
      : labels[status.p] ?? 'Quest bereitet den Probelauf vor.';
    text(statusText, joinMessage || pairWarning || (unknown
      ? 'Ergebnis unklar. Prüfe die Quest. Es wird kein Befehl wiederholt.'
      : connecting ? 'Quest wird verbunden …' : commandWaiting ? 'Warte auf Bestätigung der Quest …'
        : waitingForControl ? hint : !metaUpdate.allowed ? metaUpdate.summary : message || hint));
    if (status && connectingFocus) statusText.focus({ preventScroll: true });
    playback.hidden = !status;
    const duration = status?.duration_ms;
    const timed = active && Number.isSafeInteger(duration) && duration > 0;
    progress.hidden = !timed;
    if (timed) {
      progress.max = duration;
      progress.value = Math.min(status.ms, duration);
    }
    if (status) text(clock, timed ? `${runTime(progress.value)} / ${runTime(duration)}` : runTime(status.ms));
    const setup = polarSetup(status);
    const source = rehearsal.gateway?.owner;
    if (calibrationBaseline && source && !sameQuest(calibrationBaseline.owner, source)) resetPreview();
    const ecgSource = source && `${source.device}/${source.epoch}/${source.native_epoch}`;
    let sourceChanged = ecgOwner !== ecgSource;
    if (sourceChanged) { ecgOwner = ecgSource; ecgGeneration = null; ecgSensor = ''; }
    if (status) {
      const sensor = status.y ? `${status.y.s}/${status.y.x}` : ecgSensor;
      sourceChanged ||= ecgGeneration !== status.g || ecgSensor !== sensor;
      ecgGeneration = status.g;
      ecgSensor = sensor;
    }
    if (sourceChanged) ecgDisplay.reset();
    // An unavailable status clears the image, but cannot retire its replay fence.
    if (sourceChanged || !status) showTrace(null);
    const ready = setup?.p && setup.b && setup.c && setup.e && setup.a;
    const idle = status?.p === 'IDLE' && !active;
    const participantComplete = Boolean(!requiresParticipant || armedWithParticipant && participantId);
    const setupComplete = Boolean(ready && participantComplete);
    const breath = breathing.update(rehearsal.gateway, { status, busy, pending, unknown }, setupComplete);
    if (!breath.actions.continue || previewAccepted !== breath.acceptance) previewAccepted = '';
    const selectionAllowed = idle && setupComplete && Boolean(previewAccepted) && breath.actions.continue;
    // Authority and readiness remain owned by the existing domain models.
    byId('quest-control-host').hidden = !access;
    const version = apkRelease.update(rehearsal.gateway, status);
    const versionKey = status ? `${rehearsal.gateway?.owner.device}/${rehearsal.gateway?.owner.epoch}/${rehearsal.gateway?.owner.native_epoch}/${versionPanel.clientWidth}/${getComputedStyle(versionPanel).fontSize}/${byId('apk-identity').open}` : '';
    if (versionGeometryKey !== versionKey) {
      versionPanel.style.minBlockSize = '';
      versionGeometryKey = versionKey;
    } else if (!versionPanel.hidden && versionPanel.getClientRects().length) {
      // A read-only policy refresh must not temporarily shrink the document
      // and clamp an operator's scroll position near its bottom.
      versionPanel.style.minBlockSize = `${Math.max(versionPanel.getBoundingClientRect().height, parseFloat(versionPanel.style.minBlockSize) || 0)}px`;
    }
    versionPanel.hidden = !status;
    versionPanel.dataset.versionState = version.state;
    text(byId('apk-update-status'), version.message);
    text(byId('apk-installed'), version.installedLabel);
    text(byId('apk-required'), version.required);
    text(byId('apk-full-identity'), version.identity);
    const updateNotice = byId('operator-meta-update');
    // Stable nodes only: progress never replaces controls, editors or journals.
    const noticeHidden = !status || !metaUpdate.advertised;
    if (updateNotice.hidden !== noticeHidden) updateNotice.hidden = noticeHidden;
    for (const panel of [updateNotice, byId('meta-update')]) {
      panel.dataset.updateState = metaUpdate.state;
      panel.dataset.updateAllowed = String(metaUpdate.allowed);
    }
    text(byId('meta-update-status'), metaUpdate.summary);
    text(byId('meta-update-detail'), metaUpdate.detail);
    text(byId('operator-meta-summary'), metaUpdate.summary);
    text(byId('operator-meta-detail'), metaUpdate.detail);
    const updateControls = metaControl.update(rehearsal.gateway, { status, busy: waiting, pending, unknown });
    byId('meta-update-controls').hidden = !updateControls.available;
    byId('meta-update-check').disabled = !updateControls.canRequest;
    text(byId('meta-update-prerequisites'), updateControls.prerequisites);
    text(byId('meta-update-message'), updateControls.message);
    byId('meta-update-message').hidden = !updateControls.message;
    byId('meta-update-receipt').hidden = !updateControls.canReadReceipt || !metaControl.expired;
    byId('meta-update-acknowledge').hidden = !updateControls.canAcknowledge;
    const updateProgress = byId('meta-update-progress');
    updateProgress.hidden = !metaUpdate.progress;
    if (metaUpdate.progress && metaUpdate.progress.max !== null) {
      updateProgress.max = metaUpdate.progress.max;
      updateProgress.value = metaUpdate.progress.value;
    } else {
      updateProgress.removeAttribute('value');
      updateProgress.removeAttribute('max');
    }
    byId('quest-settings').hidden = !status;
    const kioskView = kiosk.update(rehearsal.gateway, { status, busy: waiting, pending, unknown });
    text(byId('kiosk-summary'), kioskView.summary);
    text(byId('kiosk-prerequisites'), kioskView.prerequisites);
    text(byId('kiosk-detail'), kioskView.value ? `Quest-Status: ${kioskView.value.state} · Dienst: ${kioskView.value.service}${kioskView.value.blocked ? ` · Blockiert: ${kioskView.value.blocked}` : ''}${kioskView.value.detail ? ` · ${kioskView.value.detail}` : ''}` : '');
    text(byId('kiosk-message'), kioskView.message);
    byId('kiosk-message').hidden = !kioskView.message;
    byId('kiosk-retry').hidden = !kioskView.canRetry;
    byId('kiosk-check').hidden = !kioskView.canCheck;
    byId('kiosk-acknowledge').hidden = !kioskView.canAcknowledge;
    const polarAttempts = breath.setup && calibrationBaseline ? breath.setup.attempt - calibrationBaseline.attempt : 0;
    paintBreathingSetup(breath, waiting);
    paintCheck('bluetooth', bluetoothCheck, 'Bluetooth', bluetoothCheck
      ? 'Die Quest wurde über Bluetooth gefunden.' : 'Prüfe Bluetooth am Smartphone und den Fernmodus der Quest.',
      status ? 'für diese WLAN-Verbindung nicht benötigt' : connecting ? 'wird geprüft' : 'nicht geprüft');
    paintCheck('bridge', status ? true : bridgeFailed ? false : null, 'WLAN-Brücke', status
      ? 'Die WLAN-Brücke liefert bestätigte Quest-Daten.' : joinMessage || 'Prüfe, ob Smartphone und Quest im selben WLAN sind.');
    paintCheck('apk', status && !['checking', 'loading'].includes(version.state) ? version.compatible : null, 'Quest-App', version.message);
    const polarChecking = setup?.sc || ['scanning', 'backoff', 'connecting', 'streams'].includes(setup?.auto?.state);
    paintCheck('polar', status && !polarChecking ? Boolean(ready || active && status.pi?.f === true) : null,
      'Polar H10', polarReadiness(setup));
    const showParticipant = requiresParticipant && armedWithParticipant && status.p === 'IDLE' && !active && !waitingForControl;
    if (!showParticipant && !participant.hidden && participant.contains(document.activeElement)) {
      statusText.focus({ preventScroll: true });
    }
    participant.hidden = !showParticipant;
    participantFields.disabled = !showParticipant || waiting || unknown || !metaUpdate.allowed;
    participantSubmit.disabled = participantFields.disabled;
    text(participantLabel, participantId ? `Teilnehmer-ID: ${participantId}` : 'Teilnehmer-ID eingeben');
    const hadPolarFocus = document.activeElement === polarConnect;
    const recoveringStreams = Boolean(setup?.c && !ready && (setup.ar === true || setup.auto?.state === 'streams'));
    const automaticConnection = ['scanning', 'backoff', 'connecting'].includes(setup?.auto?.state);
    polar.hidden = status?.p !== 'IDLE' || active || waitingForControl;
    text(polarStatus, polarReadiness(setup));
    text(byId('preview-polar-status'), polarStatus.textContent);
    text(byId('operator-signal-warning'), polarStatus.textContent);
    byId('operator-signal-warning').hidden = Boolean(ready);
    text(polarConnect.firstElementChild, ready ? 'Messsignale bereit'
      : setup?.c ? 'Fehlende Messsignale starten' : 'Polar H10 verbinden');
    polarConnect.hidden = recoveringStreams || automaticConnection;
    polarConnect.disabled = !metaUpdate.allowed || recoveringStreams || waiting || unknown || !setup?.p || !setup.b || !setup.c && setup.l === 'disabled' || setup.sc || ready;
    if ((polar.hidden || polarConnect.hidden) && hadPolarFocus) statusText.focus({ preventScroll: true });
    play.hidden = !status || idle && !selectionAllowed;
    const onlyTrackB = trackA.disabled && !trackB.disabled;
    const showConditions = !mainSelected && (!advertised || !trackB.disabled || condition.value === trackB.value);
    const hasLegacyScenario = scenarioOptions.some(option => !option.disabled && legacyScenarios.contains(option));
    const hasPrimaryScenario = scenarioOptions.some(option => !option.disabled && !legacyScenarios.contains(option));
    const showLegacyOptions = selectionAllowed && (hasLegacyScenario || showConditions);
    legacyScenarios.hidden = !hasLegacyScenario;
    byId('quest-condition-label').hidden = !showConditions;
    if (showLegacyOptions && legacyOptions.hidden) legacyOptions.open = !hasPrimaryScenario || onlyTrackB;
    legacyOptions.hidden = !showLegacyOptions;
    sessionChoices.hidden = !selectionAllowed;
    scenarioChoices.hidden = !selectionAllowed || advertised && !scenario;
    byId('quest-scenario-unavailable').hidden = !advertised || Boolean(scenario);
    scenarioChoices.disabled = waiting || unknown;
    if (focusedChoice && (focusedChoice.closest('[hidden]') || focusedChoice.matches(':disabled'))) {
      byId('session-screen-title').focus({ preventScroll: true });
    }
    play.disabled = waiting || unknown || !actions.play
      || actions.play === 'arm' && (!selectionAllowed || advertised && !scenario);
    text(play.firstElementChild, status?.p === 'RUNNING' ? 'Pause' : status?.p === 'PAUSED' ? 'Fortsetzen' : 'Starten');
    finish.hidden = !active;
    if (finish.hidden && hadFinishFocus) statusText.focus({ preventScroll: true });
    finish.disabled = waiting || unknown || !actions.finish;
    const recovery = rehearsal.recovery();
    byId('quest-recovery').hidden = !recovery;
    if (recovery) {
      text(byId('quest-recovery-title'), recovery.title);
      text(byId('quest-recovery-detail'), recovery.detail);
      text(acknowledge.firstElementChild, recovery.action);
    }
    acknowledge.hidden = !recovery?.action;
    acknowledge.disabled = busy || Boolean(userAction);
    disconnect.hidden = !status || active && !waitingForControl || Boolean(connecting || userAction || busy);
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
    paintOperatorLayout({ status, active, waitingForControl, participantComplete,
      requiresParticipant, selectionAllowed, polarAttempts, inputReady: breath.actions.continue, recovery: Boolean(recovery || breath.recovery
        || unknown || pending || status?.pending || nativeRecovering || breath.unknown),
      calibrationComplete: Boolean(breath.actions.continue && breath.showPolarCalibration) });
    renderLive(performance.now());
    scheduleBreathExpiry();
    scheduleAnimation();
  }

  function paintBreathingSetup(view, waiting) {
    const native = view.setup;
    text(byId('operator-confirmed-input'), view.available ? `Bestätigte Eingabe: ${view.inputLabel}` : 'Kein aktueller Eingabestatus.');
    previewCircle.hidden = !view.available;
    if (view.available) previewCircle.style.transform = `scale(${view.radius})`;
    text(previewStatus, !view.available ? 'Warte auf den Atemtakt der Quest.'
      : !native.playing ? native.stage === 'idle' ? 'Automatic ist bereit.' : 'Atemschleife auf der Quest angehalten.' : `Atemtakt: ${view.phaseLabel}`);
    const measured = view.inputPhaseLabel ? ` · Erkannt: ${view.inputPhaseLabel}` : '';
    const hasVolume = !view.renderer && (native?.input === 'automatic' || view.inputVolume !== null);
    text(previewInputLabel, view.available ? `Bestätigte Eingabe: ${view.inputLabel}${measured}${hasVolume ? ' · Rohwert der Eingabe: ' : ''}` : 'Eingabe auf der Quest nicht verfügbar.');
    previewSwitch.hidden = !view.pendingInputLabel;
    text(previewSwitch, view.pendingInputLabel ? `Angefragt: ${view.pendingInputLabel}. Die bestätigte Eingabe gilt bis zur Rückmeldung der Quest.` : '');
    const collecting = view.showPolarCalibration && ['waiting', 'collecting'].includes(native?.calibration);
    const detail = view.updateRequired ? 'Quest-App aktualisieren: Dieser Build unterstützt die Atemeinrichtung noch nicht.'
      : view.recovery ? ''
      : view.unknown ? 'Ergebnis unklar. Prüfe die Quest. Die Aktion wird nicht wiederholt; warte auf den bestätigten Status.'
      : view.recovering ? 'Die Quest prüft die Atemschleife und startet sie neu. Warte auf die Bestätigung. Die Einrichtung beginnt erst danach.'
      : view.pending ? 'Warte auf Bestätigung der Quest …'
      : !view.available ? 'Kein aktueller Atemstatus verfügbar. Prüfe die Verbindung zur Quest.'
      : view.recoveryUnavailable ? view.blockedReason
      : view.message ? view.message
      : view.failure && native.input === 'automatic' ? `Einrichtung fehlgeschlagen: ${view.failureLabel}. Automatic bleibt aktiv. Du kannst erneut einrichten oder Controller State wählen.`
      : view.blockedReason ? view.blockedReason
      : view.showPolarCalibration && native.calibration === 'waiting' ? 'Automatic ist aktiv. Die Einrichtung beginnt mit dem nächsten vollständigen Atemzyklus.'
      : view.showPolarCalibration && native.calibration === 'collecting' ? `Automatic ist aktiv. Folge dem Atemtakt für zwei vollständige Atemzyklen.${view.qualityWarning ? ` ${view.qualityWarning}` : ''}`
      : view.actions.continue ? 'Prüfe die Bewegung im Headset. Wenn die Eingabe passt, gehe weiter zur Auswahl.'
      : !native.playing ? native.stage === 'idle'
        ? 'Wähle eine Eingabe. Die Atemschleife beginnt erst danach auf der Quest.' : 'Warte auf die Atemschleife der Quest.'
      : 'Die Atemschleife läuft. Wähle eine Eingabe oder richte Polar tilt persönlich ein.';
    text(previewDetail, detail);
    previewDetail.hidden = Boolean(view.recovery);
    previewProgressGroup.hidden = !view.showPolarCalibration || !view.available || view.recovering || !(collecting || native?.attempt > 0);
    previewProgress.value = view.progress.value;
    const now = performance.now();
    const observation = rehearsal.state.status?.o;
    const timing = rehearsal.state.statusTiming;
    const age = timing ? observation?.a + timing.transportMs + Math.max(0, now - timing.receivedAt) : 1501;
    breathDisplay.update(view,
      observation && { ...observation, a: age }, now);
    text(previewProgressText, `${new Intl.NumberFormat('de-DE', { maximumFractionDigits: 2 }).format(view.progress.value)} von 2 Atemzyklen`);
    previewCalibrate.disabled = waiting || !view.actions.calibrate;
    previewCalibrate.hidden = !view.showPolarCalibration;
    text(previewCalibrate.firstElementChild, native?.attempt > 0 ? 'Erneut einrichten' : 'Einrichtung starten');
    previewPersonalReturn.hidden = view.showPolarCalibration;
    previewPersonalReturn.disabled = waiting || !view.actions.calibrate;
    for (const [button, mode, input] of [[previewPolarFixed, 'fixed', 'polar-tilt-fixed'], [previewPolarAdaptive, 'adaptive', 'polar-tilt-adaptive']]) {
      button.hidden = !view.supportedModes.includes(mode);
      button.disabled = waiting || !view.actions[`polar-${mode}`];
      button.setAttribute('aria-pressed', String(native?.input === input));
    }
    byId('preview-modes-compatibility').hidden = !view.polarModesUpdateRequired;
    previewController.hidden = native?.input === 'controller-state';
    previewController.disabled = waiting || !view.actions.controller;
    previewContinue.disabled = waiting || !view.actions.continue;
    previewContinue.hidden = !view.actions.continue;
    const reviewPrompt = byId('preview-review');
    reviewPrompt.hidden = !view.actions.continue;
    text(reviewPrompt, native?.input === 'controller-state'
      ? 'Prüfe, ob die Partikel im Headset auf die Bewegung des Controllers reagieren.'
      : 'Prüfe bei der teilnehmenden Person, ob sich die Partikel im Headset mit der Atmung bewegen.');
    previewRestart.hidden = !view.actions.continue || !view.showPolarCalibration;
    previewRetry.hidden = !view.failure || collecting;
    previewRestart.disabled = previewRetry.disabled = waiting || !view.actions.calibrate;
    byId('preview-recovery').hidden = !view.recovery;
    if (view.recovery) {
      text(byId('preview-recovery-title'), view.recovery.title);
      text(byId('preview-recovery-detail'), view.recovery.detail);
    }
    previewAcknowledge.hidden = !view.recovery?.action;
    previewAcknowledge.disabled = !(view.actions.acknowledge || view.actions.checkReceipt);
    text(previewAcknowledge.firstElementChild, view.recovery?.action || '');
  }

  function resetPreview() {
    previewAccepted = '';
    calibrationBaseline = undefined;
  }

  async function setupAct(op) {
    if (userAction || connecting || document.hidden || !breathing.view?.actions[op]) return;
    const attempt = new AbortController();
    userAction = attempt;
    if (op === 'calibrate') calibrationBaseline ??= { owner: breathing.gateway.owner, attempt: breathing.view.setup.attempt };
    const request = breathing.run(op, attempt.signal);
    navigateLocal({ type: 'go', step: op === 'calibrate' ? 'calibration' : 'tracking' });
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
    const now = performance.now();
    ecgDisplay.push(value, now);
    const current = ecgDisplay.frame(now);
    renderEcg(current);
    if (current) ecgTimer = setTimeout(() => showTrace(null), current.ttl);
    scheduleAnimation();
  }

  function renderEcg(current) {
    graph.toggleAttribute('hidden', !current);
    scale.hidden = !current;
    trace.setAttribute('d', current?.path ?? '');
    text(ecgStatus, current ? 'Live-EKG vom Polar H10' : rehearsal.gateway?.ecg
      ? 'Warte auf aktuelle EKG-Daten von der Quest.' : 'Für Live-EKG mit dem Quest-Schlüssel verbinden.');
    if (current) text(scale, `2 s · 130 Hz · ±${current.range} µV`);
  }

  function renderLive(now) {
    let moving = false;
    if (livePreviewVisible()) {
      const current = breathDisplay.frame(now, reducedMotion.matches);
      actualCircle.hidden = current?.actualRadius === null || !current;
      actualVolume.hidden = actualCircle.hidden;
      if (!actualCircle.hidden) {
        actualCircle.style.transform = `scale(${current.actualRadius})`;
        actualCircle.toggleAttribute('data-stale', !current.actualLive);
        text(actualVolume, `${Math.round(current.actualProgress * 100)} %`);
        text(actualStatus, current.actualLive ? 'IST-Radius · aktuell: ' : 'IST-Radius · eingefroren, warte auf Renderer: ');
      } else text(actualStatus, breathing.view?.setup && !Object.hasOwn(breathing.view.setup, 'renderer')
        ? 'IST-Radius nicht verfügbar · Quest-App aktualisieren.' : 'IST-Radius nicht verfügbar · warte auf aktuellen Renderer.');
      previewCircle.hidden = !current?.desiredAvailable;
      if (current) {
        previewCircle.style.transform = `scale(${current.radius})`;
        previewProgressFill.style.transform = `scaleX(${current.progress / 2})`;
        if (current.desiredAvailable && !breathing.view?.renderer && typeof current.inputVolume === 'number') {
          if (previewInputVolume.hidden) previewInputVolume.hidden = false;
          text(previewInputVolume, `${Math.round(Math.max(0, Math.min(1, current.inputVolume)) * 100)} %`);
        } else if (!previewInputVolume.hidden) previewInputVolume.hidden = true;
        const native = breathing.view?.setup;
        if (native?.playing && Number.isSafeInteger(native.audio_generation)) {
          text(previewStatus, current.live ? `Atemtakt: ${current.phaseLabel}` : 'Warte auf den aktuellen Atemtakt der Quest.');
        }
        moving = current.animating;
      } else {
        if (!previewInputVolume.hidden) previewInputVolume.hidden = true;
        text(previewStatus, breathing.view?.recovering ? 'Atemschleife wird neu gestartet. Bitte warten.'
          : 'Warte auf den aktuellen Atemtakt der Quest.');
      }
    }
    if (!graph.hasAttribute('hidden')) {
      const current = ecgDisplay.frame(now);
      renderEcg(current);
      moving ||= Boolean(current);
    }
    return moving;
  }

  function scheduleAnimation() {
    if (animationFrame || document.hidden || reducedMotion.matches || !livePreviewVisible() && graph.hasAttribute('hidden')) return;
    animationFrame = requestAnimationFrame(now => {
      animationFrame = 0;
      if (!document.hidden && !reducedMotion.matches && renderLive(now)) scheduleAnimation();
    });
  }

  function scheduleBreathExpiry() {
    clearTimeout(breathTimer);
    const current = breathDisplay.frame(performance.now(), reducedMotion.matches);
    const expiries = [current?.live ? current.ttl : null, current?.actualLive ? current.actualTTL : null].filter(value => value !== null && value !== undefined);
    if (!document.hidden && expiries.length) {
      breathTimer = setTimeout(() => { renderLive(performance.now()); scheduleBreathExpiry(); }, Math.min(...expiries) + 1);
    }
  }

  function clearDownload() {
    if (download.hasAttribute('href')) URL.revokeObjectURL(download.href);
    download.removeAttribute('href');
    download.hidden = true;
  }

  function clearArchives() {
    ecgDisplay.reset();
    breathDisplay.reset();
    showTrace(null);
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
    await kiosk.refresh();
    await metaControl.refresh();
    paint();
    if (document.hidden || connecting || userAction) return;
    if (!rehearsal.state.status && savedPair?.enrollment && autoConnect && performance.now() >= nextConnect) {
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
    if (chooseDevice) bluetoothCheck = null;
    bridgeFailed = false;
    if (chooseDevice || !discoveryInProgress) joinMessage = '';
    discoveryInProgress = false;
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
        identity_kind: discovery?.owner.identity_kind ?? settings.identity_kind,
        device: discovery?.owner.device ?? settings.device, build: chooseDevice || discovery ? undefined : settings.build,
        enrollment: settings.enrollment, discovery: discovery?.owner, signal });
      let peer = selection ? await selection : null;
      let gateway;
      let endpoint = settings.endpoint;
      if (endpoint && !peer && performance.now() >= nextSavedProbe) {
        gateway = await join(endpoint).catch(() => null);
        if (!gateway) nextSavedProbe = performance.now() + 8000;
      }
      if (!gateway) {
        signal.throwIfAborted();
        peer ??= rememberedDevice;
        if (!peer) {
          const allowed = await navigator.bluetooth?.getDevices?.() ?? [];
          peer = allowed.find(device => device.id === settings.bluetoothId);
        }
        const discovery = peer ? await readQuest(peer, signal, chooseDevice ? {} : setup)
          : await rediscoverQuest(settings, signal);
        if (discovery?.state && discovery.state !== 'ready') {
          discoveryInProgress = true;
          bluetoothCheck = discovery.state === 'searching' ? null : false;
          joinMessage = {
            'bluetooth-off': 'Schalte Bluetooth ein. Die Quest verbindet sich automatisch.',
            'searching': 'Suche die gespeicherte Quest über Bluetooth …',
            'permission-required': 'Erlaube der Bluetooth-Suche den Zugriff auf Geräte in der Nähe.',
          }[discovery.state];
          return;
        }
        if (!discovery) throw Error('Erlaube den Bluetooth-Zugriff auf die Quest erneut.');
        bluetoothCheck = true;
        let pairedThisAttempt = false;
        const pairNewPhone = async () => {
          if (!chooseDevice || discovery.pairing !== 1) throw Error('Dieses Smartphone ist für diese Quest noch nicht freigegeben.');
          const candidate = discovery.endpoints[0];
          if (!candidate) throw Error('Verbinde Quest und Smartphone mit demselben WLAN.');
          const ticket = await pairPhone({ endpoint: candidate, owner: discovery.owner, signal,
            onCode: prompt => showPairingCode(prompt, attempt) });
          signal.throwIfAborted();
          pairedThisAttempt = true;
          settings = { ...discovery.owner, endpoint: candidate, bluetoothId: peer?.id,
            hotspot: settings.device === discovery.owner.device ? settings.hotspot : undefined,
            enrollment: { controller: ticket.controller, key: ticket.key } };
          // Human approval already created this independent membership. Preserve
          // it if the following authenticated status request loses its reply.
          rememberPair(settings);
        };
        if (!settings.enrollment || settings.device && settings.device !== discovery.owner.device) await pairNewPhone();
        let authenticationRejected = false;
        for (const candidate of discovery.endpoints) {
          endpoint = candidate;
          try { gateway = await join(endpoint, discovery); break; }
          catch (error) {
            signal.throwIfAborted();
            authenticationRejected ||= error?.status === 403;
            joinMessage = error.message;
          }
        }
        if (!gateway && authenticationRejected && !pairedThisAttempt && chooseDevice && discovery.pairing === 1) {
          await pairNewPhone();
          endpoint = discovery.endpoints[0];
          gateway = await join(endpoint, discovery);
        }
        if (!gateway && peer && settings.hotspot && settings.device === discovery.owner.device) {
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
        ...(gateway.owner.identity_kind ? { identity_kind: gateway.owner.identity_kind } : {}),
        build: gateway.initial.build.packaging_fingerprint, enrollment: settings.enrollment, hotspot: settings.hotspot,
        bluetoothId: peer?.id ?? settings.bluetoothId, endpoint });
      if (peer) rememberedDevice = peer;
      autoConnect = true;
      nextSavedProbe = 0;
      joinMessage = '';
      if (!sameQuest(rehearsal.gateway?.owner, gateway.owner)) clearArchives();
      rehearsal.attach(gateway);
      void apkRelease.refresh();
      void drive.setOwner(gateway.recordings ? gateway.owner : null);
      showTrace(null);
      if (rehearsal.state.pending) await rehearsal.refresh();
    } catch (error) {
      if (connecting !== attempt) return;
      if (!signal.aborted) {
        if (bluetoothCheck === true || settings.endpoint && !chooseDevice) bridgeFailed = true;
        if (chooseDevice && bluetoothCheck !== true) bluetoothCheck = false;
      }
      joinMessage = error?.name === 'NotFoundError' && chooseDevice
        ? 'Schalte Bluetooth ein und wähle die Quest aus.'
        : error instanceof TypeError || ['TimeoutError', 'NotFoundError', 'SecurityError'].includes(error?.name)
          ? 'Quest nicht erreichbar. Prüfe Bluetooth, WLAN und den Fernmodus.'
          : error?.name === 'Error' ? error.message : 'Die Quest konnte nicht verbunden werden.';
    } finally {
      let restorePairingFocus = false;
      if (pairingChoice?.attempt === attempt) {
        restorePairingFocus = pairing.contains(document.activeElement) || document.activeElement === document.body;
        pairingChoice = null;
        pairing.hidden = true;
        text(pairingCode, '');
      }
      if (connecting === attempt) {
        nextConnect = discoveryInProgress ? 0 : performance.now() + 8000;
        connecting = null;
        paint();
        if (restorePairingFocus && !document.hidden) {
          if (!rehearsal.state.status) find.focus({ preventScroll: true });
          else if (preview.hidden) statusText.focus({ preventScroll: true });
        }
      }
    }
  }

  function showPairingCode({ code, signal }, attempt) {
    signal.throwIfAborted();
    if (connecting !== attempt || !/^[0-9]{6}$/.test(code)) throw Error('Die Verbindungsanfrage ist nicht mehr aktuell.');
    return new Promise(resolve => {
      pairingChoice = { attempt, resolve };
      text(pairingCode, code);
      pairing.hidden = false;
      pairingConfirm.disabled = false;
      joinMessage = 'Vergleiche den Code und bestätige auf Smartphone und Headset.';
      paint();
      pairingConfirm.focus({ preventScroll: true });
    });
  }

  function canChangeControl(op) {
    const { status, busy, pending, unknown } = rehearsal.state;
    const access = controllerAccessState(status);
    if (!rehearsal.gateway?.controllerAccess || !access || userAction || connecting || document.hidden
      || busy || status.pending || status.setup?.stage === 'recovering') return false;
    // A trusted phone can reclaim free control to read an earlier receipt.
    // The pending operation still guards every session and setup command.
    return op === 'claim' ? access.role === 'waiting' && !access.active_phone
      : access.role === 'active' && !pending && !unknown && !breathing.pending && !breathing.view?.recovering;
  }

  async function changeControl(op) {
    if (!canChangeControl(op)) return;
    const { gateway, state } = rehearsal;
    const access = controllerAccessState(state.status);
    const attempt = new AbortController(); userAction = attempt; paint();
    try {
      await gateway.controllerAccess({ v: 1, id: operationId(), op, expected_control_revision: access.revision }, attempt.signal);
    } catch { /* Authority readback is decisive; a lost response is never replayed. */ }
    finally {
      if (userAction === attempt) userAction = null;
      await rehearsal.refresh();
      paint();
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
    if (userAction || breathing.pending || breathing.view?.recovering) return;
    ecgAttempt?.abort();
    exportAttempt?.abort();
    liveAttempt?.abort();
    drive.cancel();
    nextSync = 0;
    const attempt = new AbortController();
    userAction = attempt;
    const scenario = scenarioChoices.hidden ? '' : selectedScenario();
    try { await rehearsal.run(action, attempt.signal, scenario === 'haupt-15' ? mainCondition : condition.value, participantId, scenario); }
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
      participant.open = true;
      navigateLocal({ type: 'go', step: 'input' });
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
  polarConnect.addEventListener('click', () => { if (!polarConnect.hidden && !polarConnect.disabled) void act('polar'); });
  pairingConfirm.addEventListener('click', () => {
    if (!pairingChoice || pairingConfirm.disabled) return;
    pairingConfirm.disabled = true;
    pairingChoice.resolve(true);
    joinMessage = 'Warte auf die Freigabe im Headset …';
    paint();
  });
  pairingCancel.addEventListener('click', () => pairingChoice?.attempt.abort());
  controlClaim.addEventListener('click', () => { if (!controlClaim.disabled) void changeControl('claim'); });
  controlRelease.addEventListener('click', () => { if (!controlRelease.disabled) void changeControl('release'); });
  byId('apk-identity').addEventListener('toggle', () => {
    byId('quest-version').style.minBlockSize = '';
    versionGeometryKey = '';
  });
  byId('meta-update-check').addEventListener('click', async () => {
    if (userAction || connecting || document.hidden || !metaControl.view?.canRequest) return;
    const attempt = new AbortController(); userAction = attempt;
    const request = metaControl.request(attempt.signal); paint();
    try { await request; }
    finally { if (userAction === attempt) userAction = null; paint(); }
  });
  byId('meta-update-receipt').addEventListener('click', () => {
    if (userAction || connecting || document.hidden || !metaControl.view?.canReadReceipt) return;
    metaControl.recheck(); void metaControl.refresh().finally(paint); paint();
  });
  byId('meta-update-acknowledge').addEventListener('click', () => {
    if (userAction || connecting || document.hidden) return;
    metaControl.acknowledge(); paint();
  });
  byId('kiosk-retry').addEventListener('click', async () => {
    if (userAction || connecting || document.hidden || !kiosk.view?.canRetry) return;
    const attempt = new AbortController(); userAction = attempt;
    const request = kiosk.retry(attempt.signal); paint();
    try { await request; }
    finally { if (userAction === attempt) userAction = null; await rehearsal.refresh(); paint(); }
  });
  byId('kiosk-check').addEventListener('click', () => {
    if (userAction || connecting || document.hidden || !kiosk.view?.canCheck) return;
    kiosk.recheck(); void kiosk.refresh().finally(paint); paint();
  });
  byId('kiosk-acknowledge').addEventListener('click', () => {
    if (userAction || connecting || document.hidden) return;
    kiosk.acknowledge(); paint();
  });
  for (const button of [previewCalibrate, previewRestart, previewRetry]) {
    button.addEventListener('click', () => setupAct('calibrate'));
  }
  previewPersonalReturn.addEventListener('click', () => setupAct('calibrate'));
  previewPolarFixed.addEventListener('click', () => setupAct('polar-fixed'));
  previewPolarAdaptive.addEventListener('click', () => setupAct('polar-adaptive'));
  previewController.addEventListener('click', () => setupAct('controller'));
  previewAcknowledge.addEventListener('click', () => {
    if (breathing.recheckReceipt()) { void breathing.refresh().finally(paint); paint(); return; }
    if (breathing.acknowledgeRetired()) { resetPreview(); paint(); }
  });
  previewContinue.addEventListener('click', () => {
    if (preview.hidden || previewContinue.disabled || !breathing.view?.actions.continue) return;
    previewAccepted = breathing.view.acceptance;
    navigateLocal({ type: 'go', step: 'session' });
    paint();
  });
  sessionChoices.addEventListener('change', () => paint());
  play.addEventListener('click', () => {
    if (!play.hidden && !play.disabled) void act('play');
  });
  recenter.addEventListener('click', () => {
    if (!document.hidden && recenter.getAttribute('aria-disabled') === 'false') void act('recenter');
  });
  finish.addEventListener('click', () => act('finish'));
  acknowledge.addEventListener('click', () => {
    if (userAction || connecting || document.hidden) return;
    if (rehearsal.canAcknowledgeIdle()) rehearsal.acknowledgeIdle();
    else if (rehearsal.recovery()?.action) void rehearsal.refresh();
  });
  retry.addEventListener('click', () => { nextSync = 0; void tick(); });
  archivePicker.addEventListener('change', selectArchive);
  function detach() {
    breathing.cancel();
    clearArchives();
    void drive.setOwner(null);
    rehearsal.attach(null);
  }
  disconnect.addEventListener('click', () => {
    autoConnect = false;
    resetPreview();
    navigateLocal({ type: 'go', step: 'connect' });
    detach();
    find.focus();
  });
  const timer = setInterval(tick, 1000);
  // Renderer samples expire after 500 ms. Read the visible preview every 100 ms
  // to leave room for observed-sample playout; other work keeps the ordinary tick.
  const previewTimer = setInterval(() => {
    if (document.hidden || !livePreviewVisible() || connecting || userAction || !rehearsal.gateway?.owner) return;
    void rehearsal.refresh({ statusOnly: true });
  }, 100);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      clearTimeout(breathTimer);
      cancelAnimationFrame(animationFrame);
      animationFrame = 0;
      breathDisplay.reset();
      renderLive(performance.now());
      if (pairingChoice) connecting?.abort();
      userAction?.abort();
      breathing.cancel();
      ecgAttempt?.abort();
      exportAttempt?.abort();
      liveAttempt?.abort();
      drive.cancel();
      showTrace(null);
    } else { renderLive(performance.now()); nextSync = 0; void apkRelease.refresh(); void tick(); }
  });
  reducedMotion.addEventListener('change', () => {
    cancelAnimationFrame(animationFrame);
    animationFrame = 0;
    renderLive(performance.now());
    scheduleBreathExpiry();
    scheduleAnimation();
  });
  window.addEventListener('pagehide', (event) => {
    clearTimeout(breathTimer);
    cancelAnimationFrame(animationFrame);
    animationFrame = 0;
    userAction?.abort();
    connecting?.abort();
    connecting = null;
    detach();
    if (!event.persisted) { clearInterval(timer); clearInterval(previewTimer); }
  });
  window.addEventListener('pageshow', () => { nextConnect = nextSavedProbe = 0; void tick(); });
  window.addEventListener('online', () => { nextConnect = nextSavedProbe = nextLive = 0; drive.schedule(); void tick(); });
  paint();
  void apkRelease.refresh();
  void tick();
}
