import { operatorSteps, navigateOperatorView, resolveOperatorView, operatorCompatibilityWarning,
  restoreOperatorView, operatorSessionHeading } from './operator-layout.js?v=10';
import { selectQuest, readQuest, rediscoverQuest, joinQuest, sameQuest, operationId,
  controllerAccessState } from './quest-gateway.js?v=20';
import { pairPhone } from './quest-pairing.js?v=9';
import { canRecenter, polarSetup, polarReadiness, probeInstructionPlaying, QuestRehearsal, rehearsalActions, rehearsalDuration, rehearsalPolarAllowed, rehearsalScenarios } from './quest-rehearsal.js?v=44';
import { ecgTrace, questRecording, questRecordings, questPrefixBatches } from './quest-data.js?v=30';
import { openQuestRecordingStore } from './quest-recording-store.js?v=5';
import { mountQuestDrive } from './quest-drive-ui.js?v=9';
import { questionnaireFile } from './data-logging.js?v=20261118-questionnaire-completion';
import { mountQuestionnaireForm } from './questionnaire-form.js?v=2';
import { captureUtc } from './collection-time.js?v=1';
import { BreathingSetup } from './breathing-setup.js?v=24';
import { selectBreathingMethod } from './breathing-selection.js?v=1';
import { LiveBreathDisplay } from './live-breath-display.js?v=6';
import { SessionParticles } from './session-particles.js?v=1';
import { LiveEcgDisplay } from './live-ecg-display.js?v=2';
import { KioskControl } from './kiosk-control.js?v=10';
import { ApkRelease } from './quest-release.js?v=2';
import { nativeSessionAllowed, metaUpdateView, updatePresentation } from './meta-update.js?v=6';
import { MetaUpdateControl } from './meta-update-control.js?v=12';
import { MetaUpdateHandoff } from './meta-update-handoff.js?v=4';
import { ControllerLease } from './controller-lease.js?v=8';
import { OperatorDeviceHealth } from './operator-device-health.js?v=2';

const byId = (id) => document.getElementById(id);
const text = (element, value) => { if (element.textContent !== value) element.textContent = value; };
const needsDialogFocus = dialog => !document.activeElement || document.activeElement === document.body
  || dialog.contains(document.activeElement);
const operatorCondition = 'condition-a';
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

export function mountQuestRehearsal(participantIdentity, savedPair, canAutoConnect) {
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
  const instructionCancel = byId('calibration-instruction-cancel');
  const calibrationChangeInput = byId('calibration-change-input');
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
  const scenarioUnavailable = byId('quest-scenario-unavailable');
  const participant = byId('quest-participant');
  const participantLabel = byId('quest-participant-label');
  const participantForm = byId('patient-form');
  const participantFields = byId('patient-fields');
  const participantSubmit = byId('patient-next');
  const participantOutput = byId('patient-id');
  let participantId = '';
  let previewAccepted = '';
  let batteryAccepted = '';
  const data = byId('quest-data');
  const ecgStatus = byId('quest-ecg-status');
  const graph = byId('quest-ecg');
  const trace = byId('quest-ecg-trace');
  const scale = byId('quest-ecg-scale');
  const exportStatus = byId('quest-export-status');
  const retry = byId('quest-export-retry');
  const download = byId('quest-export-download');
  const archivePicker = byId('quest-archives');
  const recordingsView = byId('recordings-view');
  const recordingsScreen = byId('recordings-screen');
  function setRecordingsView(view) {
    recordingsView.value = view;
    if (recordingsScreen.dataset.view !== view) recordingsScreen.dataset.view = view;
  }
  recordingsView.addEventListener('change', () => setRecordingsView(recordingsView.value));
  recordingsScreen.addEventListener('focusin', event => {
    const view = event.target.closest('[data-recordings-view]')?.dataset.recordingsView;
    if (view) setRecordingsView(view);
  });
  let ecgTimer, ecgAttempt, exportAttempt, nextSync = 0, exportMessage = '';
  let liveAttempt, liveGeneration, nextLive = 0;
  let archiveRead = 0;
  let archiveListRead = 0, archiveScope = '', archivesSuspended = false;
  let archiveListLoaded = false;
  let connecting = null;
  let rememberedDevice = null;
  let nextConnect = 0;
  let autoConnectInFlight = false;
  let nextCommissioningProbe = 0;
  let nextSavedProbe = 0;
  let discoveryInProgress = false;
  let bluetoothCheck = null;
  let bluetoothHint = '';
  const bluetooth = navigator.bluetooth;
  const enableBluetooth = 'Schalte Bluetooth ein. Die Quest verbindet sich automatisch.';
  const allowCompanion = 'Erlaube „Apps auf dem Gerät“ in Chrome.';
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
  const toolbar = byId('operator-toolbar');
  const utilities = { settings: byId('settings-open'), recordings: byId('recordings-open') };
  const toolbarActions = byId('toolbar-actions');
  const toolbarDialog = byId('toolbar-dialog');
  const compactToolbar = matchMedia('(max-width: 389px)');
  function placeToolbarActions() {
    const active = document.activeElement;
    if (toolbarDialog.open && !compactToolbar.matches) toolbarDialog.close();
    toolbarActions.hidden = !compactToolbar.matches;
    for (const button of Object.values(utilities)) {
      if (compactToolbar.matches) toolbarDialog.querySelector('.screen-body').append(button);
      else toolbar.insertBefore(button, toolbarActions);
    }
    if (Object.values(utilities).includes(active)) {
      (compactToolbar.matches ? toolbarActions : active).focus({ preventScroll: true });
    } else if (!compactToolbar.matches && active === toolbarActions) utilities.settings.focus({ preventScroll: true });
  }
  compactToolbar.addEventListener('change', placeToolbarActions);
  placeToolbarActions();
  toolbarActions.addEventListener('click', () => toolbarDialog.showModal());
  const apkDetails = byId('apk-identity');
  const apkView = byId('apk-detail-view');
  const settingsEcg = byId('settings-ecg');
  const settingsConnection = byId('settings-connection');
  const settingsKiosk = byId('quest-settings');
  const connectionView = byId('settings-connection-view');
  const sessionReceiptView = connectionView.querySelector('[value="run-receipt"]');
  const setupReceiptView = connectionView.querySelector('[value="setup-receipt"]');
  connectionView.addEventListener('change', () => { settingsConnection.dataset.view = connectionView.value; });
  function showApkDetails() {
    for (const option of apkView.options) byId('apk-detail-' + option.value).hidden = option.value !== apkView.value;
  }
  apkView.addEventListener('change', showApkDetails);
  for (const disclosure of [apkDetails, settingsEcg, settingsConnection, settingsKiosk, byId('operator-input-diagnostics')]) {
    const dialog = disclosure.querySelector('dialog');
    disclosure.addEventListener('toggle', () => {
      if (disclosure.open && !dialog.open) dialog.showModal();
      else if (!disclosure.open && dialog.open) dialog.close();
    });
    dialog.addEventListener('close', () => { disclosure.open = false; });
  }
  const checks = toolbar.querySelectorAll('.check-icon');
  const diagnosticState = new Map();
  let diagnosticTrigger;
  diagnostics.addEventListener('close', () => {
    if (!needsDialogFocus(diagnostics)) return;
    if (diagnosticTrigger?.getClientRects().length && !diagnosticTrigger.disabled) diagnosticTrigger.focus({ preventScroll: true });
    else document.querySelector('.screen:not([hidden]) h1')?.focus({ preventScroll: true });
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
  const lookingForSavedQuest = 'Suche die gespeicherte Quest über Bluetooth …';
  let storage;
  try { storage = sessionStorage; } catch { /* The current connection still works. */ }
  const breathing = new BreathingSetup({ storage });
  const controllerLease = new ControllerLease({ storage });
  const deviceHealth = new OperatorDeviceHealth();
  const kiosk = new KioskControl({ storage });
  const metaControl = new MetaUpdateControl({ storage });
  const metaHandoff = new MetaUpdateHandoff({ storage });
  const apkRelease = new ApkRelease({ onChange: () => paint() });
  const breathDisplay = new LiveBreathDisplay();
  const sessionScene = byId('session-scene');
  const sceneStatus = byId('session-scene-status');
  const sceneRadius = byId('session-scene-radius');
  const particles = new SessionParticles(sessionScene.querySelector('canvas'));
  const ecgDisplay = new LiveEcgDisplay();
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let animationFrame = 0;
  let ecgOwner = '';
  let ecgGeneration = null;
  let ecgSensor = '';
  let breathTimer;
  let controlTimer;
  let deviceTimer;
  let versionGeometryKey = '';
  let updateGeometryKey = '';
  let saveForUpdate = false;
  let saveUpdateMessage = '';
  let savedUpdateIntent = null;
  let savedUpdateReconciling = false;
  const rehearsal = new QuestRehearsal({ storage, onChange: paint });

  let localView = restoreOperatorView(location.hash, history.state);
  let displayedStep = '';
  let wasActive = false;
  const questionnaireForm = byId('questionnaire-form');
  const answerStatus = byId('questionnaire-status');
  const answerSubmit = byId('questionnaire-submit');
  const answerFields = mountQuestionnaireForm(questionnaireForm, answerStatus);
  const answerLink = byId('questionnaire-download');
  const answerOpen = byId('questionnaire-open');
  let questionnaire, selectedQuestionnaire, savingAnswers = false;

  function setQuestionnaire(form) {
    if (form && !localArchiveOwner()) return;
    if (questionnaire?.id === form?.id && questionnaire?.run.session_id === form?.run.session_id) return;
    questionnaire = form;
    answerFields.reset();
  }

  answerOpen.addEventListener('click', () => {
    if (!selectedQuestionnaire || selectedQuestionnaire.file || questionnaire?.required) return;
    setQuestionnaire(selectedQuestionnaire);
    navigateLocal({ type: 'go', step: 'questionnaire' });
    paint();
  });

  questionnaireForm.addEventListener('submit', async event => {
    event.preventDefault();
    const form = questionnaire;
    const owner = localArchiveOwner();
    if (!form || !owner || savingAnswers || displayedStep !== 'questionnaire') return;
    const answers = answerFields.read();
    if (!answers) return;
    savingAnswers = answerSubmit.disabled = questionnaireForm.inert = true;
    let store;
    try {
      const file = questionnaireFile(form.run, answers, await captureUtc());
      store = await openQuestRecordingStore(owner);
      await store.saveQuestionnaire(form.id, form.run.session_id, file);
      drive.schedule();
      const nextForm = await store.pendingQuestionnaire();
      if (questionnaire !== form) return;
      setQuestionnaire(nextForm);
      localView = { step: 'recordings', returnTo: null };
      setRecordingsView('files');
      history.replaceState({ operatorStep: 'recordings' }, '', '#recordings');
      archivePicker.value = form.id;
      void selectArchive();
    } catch {
      if (questionnaire === form) {
        text(answerStatus, 'Speichern fehlgeschlagen. Bitte versuche es erneut.');
        answerStatus.hidden = false;
      }
    } finally {
      store?.close();
      savingAnswers = answerSubmit.disabled = questionnaireForm.inert = false;
      paint();
    }
  });
  let utilityReturnFocus = null;

  function livePreviewVisible() {
    return ['calibration', 'tracking'].includes(displayedStep) && location.hash !== '#daten';
  }
  function runningScene() {
    const status = rehearsal.state.status;
    return displayedStep === 'running' && location.hash !== '#daten' && status?.rec
      && (status.p === 'RUNNING' || status.p === 'PAUSED') ? status : null;
  }
  function navigateLocal(event) {
    utilityReturnFocus = event.type === 'return' && utilities[displayedStep]
      ? (compactToolbar.matches ? toolbarActions : utilities[displayedStep]) : null;
    localView = navigateOperatorView({ ...localView, step: displayedStep || localView.step }, event,
      rehearsal.state.status?.participant_arm === 1);
    const hash = `#${localView.step}`;
    if (location.hash !== hash) history.pushState({ operatorStep: localView.step, operatorReturnTo: localView.returnTo }, '', hash);
  }
  function paintOperatorLayout(context) {
    const nativeActive = context.active || Boolean(context.status && ['ARMED', 'RUNNING', 'PAUSED', 'FINALIZING', 'SAVING'].includes(context.status.p));
    if (context.status) {
      if (wasActive && !nativeActive && context.status.p === 'IDLE') {
        localView = { step: saveForUpdate ? 'update' : 'recordings', returnTo: null };
        // Native completion changes the current view. History must reflect it
        // before a link can open the older browser recordings and return.
        history.replaceState({ operatorStep: localView.step }, '', `#${localView.step}`);
      }
      wasActive = nativeActive;
    }
    const showQuestionnaire = archiveScope && questionnaire && (questionnaire.required || localView.step === 'questionnaire');
    if (location.hash === '#daten' && !nativeActive && !context.recovery && !showQuestionnaire) {
      utilities.settings.disabled = utilities.recordings.disabled = false;
      for (const button of Object.values(utilities)) button.removeAttribute('aria-current');
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
    const step = resolveOperatorView(localView.step, { ...context, connected: Boolean(context.status),
      participantRequired: context.requiresParticipant, active: nativeActive, questionnaire: showQuestionnaire });
    text(statusText, step === 'recordings' && context.statusMessage === lookingForSavedQuest
      ? 'Quest wird gesucht …' : context.statusMessage);
    if (step === 'update' && localView.step !== 'update') {
      // Once an exception needs attention, retain its page until Next. A
      // terminal update poll must not pull focus out of an open diagnostic.
      localView = { step, returnTo: 'connect' };
      history.replaceState({ operatorStep: step, operatorReturnTo: localView.returnTo }, '', '#update');
    }
    if (localView.step === 'calibration' && step === 'tracking') localView = { ...localView, step };
    const page = byId(operatorSteps[step]);
    const body = page.querySelector('.operator-body');
    const move = (id, host) => { const element = byId(id); if (element.parentElement !== host) host.prepend(element); };
    const activeControls = step === 'running' || step === 'settings' && nativeActive;
    settingsEcg.hidden = !activeControls || step !== 'settings' || data.hidden;
    if (settingsEcg.hidden && settingsEcg.open) settingsEcg.open = false;
    if (['connect', 'signals'].includes(step) || step === 'running') move('quest-data', body);
    else if (step === 'settings' && activeControls) move('quest-data', settingsEcg.querySelector('.screen-body'));
    if (['connect', 'signals'].includes(step)) move('quest-polar', body);
    byId('corner-actions').hidden = step !== 'connect' || byId('install-app').hidden;
    driveOpen.hidden = drivePanel.hidden;
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
      if (['input', 'calibration', 'tracking'].includes(step)) move('operator-input-status', body);
      if (['calibration', 'tracking'].includes(step)) move('operator-live', body);
      if (step === 'settings') connectionView.after(byId('operator-common'));
      else if (step !== 'questionnaire') move('operator-common', body);
      if (step === 'questionnaire' && toolbarDialog.open) toolbarDialog.close();
      if (step !== 'settings') settingsConnection.open = false;
      if (step !== 'settings') settingsKiosk.open = false;
      const preparationRecovery = byId('preview-recovery');
      if (step === 'update') byId('apk-detail-receipt').append(preparationRecovery);
      else if (preparationRecovery.parentElement !== byId('operator-common')) byId('quest-recovery').after(preparationRecovery);
      if (step !== 'update') apkDetails.open = false;
      for (const id of Object.values(operatorSteps)) byId(id).hidden = id !== page.id;
      displayedStep = step;
      if (['connect', 'recordings'].includes(step)) {
        driveActionsHost().append(driveOpen, driveStatusHost);
      } else if (driveDialog.open) driveDialog.close();
      document.documentElement.dataset.operatorStep = step;
      if (utilityReturnFocus) {
        utilityReturnFocus.disabled = false;
        utilityReturnFocus.focus();
        utilityReturnFocus = null;
      } else page.querySelector('h1').focus();
    }
    if (step === 'session' || activeControls) {
      move('operator-session-actions', page.querySelector(':scope > .actions'));
    }
    const updateNotice = byId('operator-meta-update');
    const nativeUpdate = metaUpdateView(context.status);
    const noticeHidden = !context.status || !nativeUpdate.advertised || step === 'update'
      || nativeUpdate.allowed && ['current', 'updated'].includes(nativeUpdate.state) && !metaControl.pending;
    if (updateNotice.hidden !== noticeHidden) updateNotice.hidden = noticeHidden;
    const fullConnection = ['connect', 'settings'].includes(step);
    const routineStatus = ['IDLE', 'RUNNING'].includes(context.status?.p) && Object.values(labels).includes(statusText.textContent);
    text(byId('running-screen-title'), probeInstructionPlaying(context.status) ? 'Anleitung zur Probe-Runde' : operatorSessionHeading(context.status?.p));
    const access = controllerAccessState(context.status);
    const duplicateUpdateStatus = (step === 'update' || !noticeHidden) && statusText.textContent === nativeUpdate.summary;
    const duplicateControlStatus = context.waitingForControl && access && statusText.textContent === controlStatus.textContent;
    statusText.hidden = duplicateUpdateStatus || duplicateControlStatus
      || step === 'update' && !rehearsal.recovery() && !context.waitingForControl
      || (!fullConnection || step === 'connect' && access?.role === 'active') && routineStatus;
    byId('quest-control-host').hidden = step !== 'settings' && !context.waitingForControl || !access;
    text(byId('operator-connection-summary'), controllerLease.view?.active
      ? 'Quest verbunden · Steuerung aktiv' : 'Quest verbunden');
    // Readiness gates actions, never the existence of the confirmed method.
    // Hiding this block on ready/unready polls made it flash above the circles.
    if (byId('operator-input-status').hidden) byId('operator-input-status').hidden = false;
    previewProgressGroup.hidden ||= step !== 'calibration';
    byId('operator-connection-summary').hidden = step !== 'connect' || !context.status || context.recovery || !routineStatus;
    byId('quest-control-release').hidden = step !== 'settings' || controllerAccessState(context.status)?.role !== 'active';
    const warning = byId('operator-compatibility');
    const version = byId('quest-version').dataset.versionState;
    warning.hidden = !context.status || !operatorCompatibilityWarning(version) || step === 'update'
      || version === 'required' && nativeUpdate.state === 'update_required' && !noticeHidden;
    text(warning, version === 'required' ? 'Update der Quest-App erforderlich.' : byId('apk-update-status').textContent);
    // An overriding native operation cannot be escaped through utility links.
    for (const control of document.querySelectorAll('[data-operator-go], [data-operator-back], [data-operator-return]')) {
      control.disabled = step === 'questionnaire'
        || ['running', 'recovery'].includes(step) && control.dataset.operatorGo !== 'settings';
    }
    byId('operator-update-next').disabled = nativeActive || context.updateRequired || context.waitingForControl || !context.status;
    byId('operator-update-open').disabled = !context.status;
    for (const [name, button] of Object.entries(utilities)) {
      if (name === step && !button.hasAttribute('aria-current')) button.setAttribute('aria-current', 'page');
      else if (name !== step) button.removeAttribute('aria-current');
    }
    utilities.recordings.disabled = step === 'questionnaire' || nativeActive || Boolean(context.recovery && !context.offlineArchives);
    const polarAlreadyReady = step === 'connect' && byId('check-polar').dataset.check === 'ok';
    polarStatus.hidden = polarAlreadyReady;
    if (polarAlreadyReady) polarConnect.hidden = true;
    byId('operator-connect-next').disabled = context.status?.wifi_snapshot !== 1 || context.waitingForControl;
    text(byId('operator-connect-next').querySelector('[data-measure]'), context.updateRequired
      ? nativeUpdate.value?.reason === 'required-update-not-yet-on-channel' ? 'Update-Prüfung öffnen' : 'Quest-App aktualisieren' : 'Weiter');
    byId('operator-connect-next').dataset.operatorGo = context.updateRequired ? 'update' : 'participant';
    byId('operator-next-session').disabled = !context.status || context.waitingForControl;
    if (step === 'settings') {
      sessionReceiptView.disabled = byId('quest-recovery').hidden;
      setupReceiptView.disabled = byId('preview-recovery').hidden;
      if (connectionView.selectedOptions[0]?.disabled) connectionView.value = 'control';
      settingsConnection.dataset.view = connectionView.value;
    }
  }
  document.addEventListener('click', event => {
    const control = event.target.closest('[data-operator-go], [data-operator-back], [data-operator-return]');
    if (!control || control.disabled || control.getAttribute('aria-current') === 'page') return;
    if (toolbarDialog.open && toolbarDialog.contains(control)) toolbarDialog.close();
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
    refreshLocalArchives();
    if (status) metaHandoff.observe(rehearsal.gateway, { status, busy, pending, unknown });
    else metaHandoff.disconnected();
    const handoff = metaHandoff.view(savedPair);
    const handoffNotice = byId('operator-meta-handoff');
    handoffNotice.hidden = !handoff;
    text(byId('operator-meta-handoff-summary'), handoff?.summary ?? '');
    text(byId('operator-meta-handoff-detail'), handoff?.detail ?? '');
    const supportsScenarios = Boolean(rehearsal.gateway?.scenarioArm);
    const available = supportsScenarios ? rehearsalScenarios(status)
      .filter(id => id !== 'free-exploration' || rehearsal.gateway.participantArm) : [];
    const focusedChoice = sessionChoices.contains(document.activeElement) ? document.activeElement : null;
    for (const option of scenarioOptions) {
      option.closest('label').hidden = option.disabled = !available.includes(option.value);
    }
    if (!selectedScenario()) {
      const firstOffered = scenarioOptions.find(option => !option.disabled);
      if (firstOffered) firstOffered.checked = true;
    }
    const scenario = selectedScenario();
    const actions = rehearsalActions(status, operatorCondition);
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
    const control = controllerLease.update(rehearsal.gateway, rehearsal.state, controlContext());
    clearTimeout(controlTimer);
    if (control.active) controlTimer = setTimeout(() => paint(), control.expiresIn + 1);
    const localCommandWaiting = Boolean(connecting || userAction || controllerLease.inFlight || controllerLease.pending
      || busy || pending || breathing.busy || breathing.cancellingInstruction || nativeRecovering || savedUpdateIntent);
    const liveCommandWaiting = localCommandWaiting || status?.pending;
    const preparationWaiting = Boolean(breathing.pending || breathing.view?.recovering);
    const commandWaiting = liveCommandWaiting || preparationWaiting;
    const waiting = waitingForControl || commandWaiting;
    // An old preparation receipt cannot own a different current recording.
    // Native recovery and uncertain experiment/control requests still fence it.
    const sessionWaiting = waitingForControl || liveCommandWaiting
      || preparationWaiting && !currentSessionControls(status, control);
    controlPanel.hidden = !access;
    controlClaim.hidden = !waitingForControl;
    controlRelease.hidden = access?.role !== 'active';
    controlClaim.disabled = !canChangeControl('claim');
    controlRelease.disabled = !canChangeControl('release');
    const controlSummary = control.active ? 'Dieses Smartphone steuert die Quest.'
      : control.state === 'occupied' ? 'Anderes Smartphone steuert die Quest.' : control.message;
    text(controlStatus, controlSummary);
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
      : waitingForControl ? controlSummary
      : status.h === 'panel' ? 'Schließe das Bedienfeld in der Quest und kehre zur VR-Szene zurück.'
      : !rehearsalPolarAllowed(status) ? 'Diese Sitzung ist nicht für den Probelauf freigegeben.'
      : requiresParticipant && !armedWithParticipant ? 'Dieses Smartphone ist für den Probelauf noch nicht freigegeben.'
      : status.pi?.r === true && status.pi.f !== true && ['ARMED', 'PAUSED'].includes(status.p)
        ? 'Warte auf aktuelle Polar-H10-Daten von der Quest.'
      : status.p === 'IDLE' && !actions.play ? 'Warte auf die VR-Szene und den Probelauf.'
      : labels[status.p] ?? 'Quest bereitet den Probelauf vor.';
    const statusMessage = joinMessage || pairWarning || (unknown
      ? 'Ergebnis unklar. Prüfe die Quest. Es wird kein Befehl wiederholt.'
      : connecting ? 'Quest wird verbunden …' : probeInstructionPlaying(status) ? 'Die fünf Minuten beginnen nach der Anleitung im Headset.'
        : commandWaiting ? 'Warte auf Bestätigung der Quest …'
        : waitingForControl ? hint : !metaUpdate.allowed ? metaUpdate.summary : message || hint);
    if (status && connectingFocus) statusText.focus({ preventScroll: true });
    playback.hidden = !status;
    const duration = rehearsalDuration(status, available);
    if (progress.hidden !== (duration === null)) progress.hidden = duration === null;
    if (duration !== null) {
      if (progress.max !== duration) progress.max = duration;
      const elapsed = Math.min(status.ms, duration);
      if (progress.value !== elapsed) progress.value = elapsed;
    }
    if (status) text(clock, duration === null ? runTime(status.ms)
      : `${runTime(progress.value)} / ${runTime(duration)}`);
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
    // This is the operator's judgement of this exact method/attempt, not a
    // sample-level readiness grant. A brief unknown phase cannot erase it.
    const reviewable = breath.available && breath.setup.stage === 'testing' && breath.setup.playing
      && breath.setup.input !== 'automatic'
      && (breath.setup.input !== 'polar-tilt' || breath.setup.calibration === 'ready');
    if (previewAccepted !== breath.acceptance || !reviewable || !idle) previewAccepted = '';
    const batteryBinding = previewAccepted && `${previewAccepted}/${participantId}`;
    if (!batteryBinding || batteryAccepted !== batteryBinding) batteryAccepted = '';
    const batteryReviewed = Boolean(batteryAccepted && batteryAccepted === batteryBinding);
    const selectionAllowed = idle && setupComplete && Boolean(previewAccepted) && reviewable;
    const device = deviceHealth.update(rehearsal.gateway?.owner, rehearsal.state);
    clearTimeout(deviceTimer);
    if (!document.hidden && device.expiresIn !== null) deviceTimer = setTimeout(() => paint(), device.expiresIn + 1);
    text(byId('battery-quest'), device.battery.quest);
    text(byId('battery-controllers'), device.battery.controllers);
    text(byId('battery-polar'), device.battery.polar);
    byId('battery-next').disabled = !selectionAllowed || waiting || unknown;
    // Authority and readiness remain owned by the existing domain models.
    byId('quest-control-host').hidden = !access;
    const version = apkRelease.update(rehearsal.gateway, status);
    const updateRequired = version.state === 'required' || Boolean(metaUpdate.advertised && !metaUpdate.allowed);
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
    // Full values stay available in the collapsed diagnostic disclosure.
    for (const id of ['apk-installed', 'apk-required']) if (byId(id).hidden) byId(id).hidden = false;
    const unknownIdentity = 'Keine bestätigte APK-Kennung verfügbar.';
    text(byId('apk-package'), version.installed?.package_name ?? unknownIdentity);
    text(byId('apk-full-identity'), version.installed?.build_id ?? unknownIdentity);
    const updateNotice = byId('operator-meta-update');
    // Stable nodes only: progress never replaces controls, editors or journals.
    for (const panel of [updateNotice, byId('meta-update')]) {
      panel.dataset.updateState = metaUpdate.state;
      panel.dataset.updateAllowed = String(metaUpdate.allowed);
    }
    text(byId('meta-update-diagnostic-status'), metaUpdate.summary);
    const currentMetaRelease = metaUpdate.allowed && ['current', 'updated'].includes(metaUpdate.state) && metaUpdate.value;
    const channelCode = currentMetaRelease && metaUpdate.value.available_code;
    const requiredCode = currentMetaRelease && metaUpdate.value.required_code;
    text(byId('meta-update-detail'), currentMetaRelease
      ? ['Vorbereitung freigegeben.', channelCode != null && `Meta-Code ${channelCode}.`,
        requiredCode != null && `Mindestens Code ${requiredCode}.`].filter(Boolean).join(' ')
      : metaUpdate.detail);
    // An old setup journal stays read-only; native dynamic admission independently
    // fences ending preparation. Actual experiment/control uncertainty still blocks.
    const updateControls = metaControl.update(rehearsal.gateway, { status, busy: localCommandWaiting || waitingForControl || !control.active,
      calibrationInstruction: breath.setup?.instruction_id, pending, unknown });
    byId('meta-update-controls').hidden = !updateControls.available;
    byId('meta-update-check').disabled = !updateControls.canRequest;
    text(byId('meta-update-prerequisites'), updateControls.prerequisites);
    text(byId('meta-update-message'), updateControls.message);
    byId('meta-update-message').hidden = !updateControls.message;
    byId('meta-update-receipt').hidden = !updateControls.canReadReceipt || !metaControl.expired;
    byId('meta-update-acknowledge').hidden = !updateControls.canAcknowledge;
    const updateCopy = updatePresentation(version, metaUpdate, updateControls, Boolean(metaControl.pending));
    text(byId('operator-meta-summary'), updateCopy.summary);
    const noticeDetail = byId('operator-meta-detail');
    text(noticeDetail, updateCopy.instruction);
    if (noticeDetail.hidden !== !updateCopy.instruction) noticeDetail.hidden = !updateCopy.instruction;
    const finishUpdate = byId('meta-update-finish');
    const recording = Boolean(status?.rec || status?.p === 'ARMED' && actions.finish === 'save-next');
    finishUpdate.hidden = !recording;
    finishUpdate.disabled = !recording || sessionWaiting || unknown || actions.finish !== 'save-next' || !control.active || Boolean(metaControl.pending);
    const pendingProbeStart = probeInstructionPlaying(status) && rehearsal.pending?.op === 'start'
      && rehearsal.pending.generation === status.g && sameQuest(rehearsal.pending.owner, rehearsal.gateway?.owner);
    if (recording && !metaControl.pending) {
      if (pendingProbeStart && !unknown) {
        updateCopy.instruction = 'Die Anleitung zur Probe-Runde läuft. Nach dem bestätigten Sitzungsstart kannst du die Aufnahme speichern.';
      } else if (!finishUpdate.disabled) {
        if (['current', 'updated'].includes(metaUpdate.state) && version.compatible) {
          updateCopy.summary = status.p === 'PAUSED' ? 'Aufnahme pausiert.' : status.p === 'ARMED' ? 'Sitzung vorbereitet.' : 'Aufnahme läuft.';
        }
        updateCopy.instruction = metaUpdate.instruction ?? updateCopy.instruction;
      }
    }
    if (saveForUpdate) {
      updateCopy.summary = 'Quest beendet und speichert die Sitzung.';
      updateCopy.instruction = metaUpdate.instruction ?? '';
    }
    if (saveUpdateMessage) {
      updateCopy.summary = 'Update noch nicht bestätigt.';
      updateCopy.instruction = saveUpdateMessage;
    }
    if (savedUpdateIntent) updateCopy.summary = 'Aufnahme gespeichert. Quest bereitet das Update vor.';
    byId('meta-update-save-retry').hidden = !savedUpdateIntent;
    byId('meta-update-save-retry').disabled = Boolean(userAction || connecting || savedUpdateReconciling);
    byId('meta-update-check').hidden = recording;
    text(byId('meta-update-status'), updateCopy.summary);
    text(byId('update-instruction'), updateCopy.instruction);
    if (byId('update-instruction').hidden !== !updateCopy.instruction) byId('update-instruction').hidden = !updateCopy.instruction;
    text(byId('meta-update-check').querySelector('[data-measure]'), updateCopy.checkLabel);
    const updateProgress = byId('meta-update-progress');
    updateProgress.hidden = !metaUpdate.progress;
    if (metaUpdate.progress && metaUpdate.progress.max !== null) {
      updateProgress.max = metaUpdate.progress.max;
      updateProgress.value = metaUpdate.progress.value;
    } else {
      updateProgress.removeAttribute('value');
      updateProgress.removeAttribute('max');
    }
    settingsKiosk.hidden = !status;
    if (!status && settingsKiosk.open) settingsKiosk.open = false;
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
    paintCheck('bluetooth', status ? true : bluetoothCheck, status ? 'Quest-Verbindung' : 'Bluetooth', status
      ? 'Die Quest-Verbindung ist bestätigt. Bluetooth dient zum Finden und Koppeln; die laufende Steuerung verwendet die WLAN-Brücke.'
      : bluetoothCheck ? 'Die Quest wurde über Bluetooth gefunden.' : bluetoothHint || 'Prüfe Bluetooth am Smartphone und den Fernmodus der Quest.',
      status ? 'für diese WLAN-Verbindung nicht benötigt' : connecting ? 'wird geprüft' : 'nicht geprüft');
    paintCheck('bridge', status ? true : bridgeFailed ? false : null, 'WLAN-Brücke', status
      ? 'Die WLAN-Brücke liefert bestätigte Quest-Daten.' : joinMessage || 'Prüfe, ob Smartphone und Quest im selben WLAN sind.');
    paintCheck('control', control.active ? true : control.fresh || control.state === 'lost' ? false : null,
      'Quest-Steuerung', control.message, 'wird bestätigt');
    byId('check-control').dataset.controlState = control.state;
    const polarConfirmation = document.hidden ? null : device.polar.confirmed;
    paintCheck('polar', polarConfirmation, 'Polar H10', device.polar.detail, device.polar.neutral);
    byId('check-polar').dataset.healthState = device.polar.state;
    if (polarConfirmation === null && status && !document.hidden) {
      byId('check-polar').disabled = false;
      text(byId('check-polar').querySelector('.check-symbol'), '?');
    }
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
    text(byId('operator-signal-warning'), 'Polar-Signale sind noch nicht bereit.');
    byId('operator-signal-warning').hidden = Boolean(ready || breath.blockedReason && !previewDetail.hidden);
    text(polarConnect.firstElementChild, ready ? 'Messsignale bereit'
      : setup?.c ? 'Fehlende Messsignale starten' : 'Polar H10 verbinden');
    polarConnect.hidden = recoveringStreams || automaticConnection;
    polarConnect.disabled = !metaUpdate.allowed || recoveringStreams || waiting || unknown || !setup?.p || !setup.b || !setup.c && setup.l === 'disabled' || setup.sc || ready;
    if ((polar.hidden || polarConnect.hidden) && hadPolarFocus) statusText.focus({ preventScroll: true });
    play.hidden = !status || idle && !selectionAllowed;
    const scenarioIssue = !scenario ? 'Die Quest bietet keinen der hier verfügbaren Abläufe an. Prüfe die Quest-App unter Einstellungen.'
      : status?.a?.[0] !== 'track-ready' ? 'VR-Bedingung A ist auf der Quest nicht verfügbar.' : '';
    sessionChoices.hidden = !selectionAllowed;
    scenarioChoices.hidden = !selectionAllowed || !scenario;
    scenarioUnavailable.hidden = !scenarioIssue;
    text(scenarioUnavailable, scenarioIssue);
    scenarioChoices.disabled = waiting || unknown;
    if (focusedChoice && (focusedChoice.closest('[hidden]') || focusedChoice.matches(':disabled'))) {
      byId('session-screen-title').focus({ preventScroll: true });
    }
    const currentPlay = ['pause', 'resume'].includes(actions.play);
    play.disabled = (currentPlay ? sessionWaiting : waiting) || unknown || !actions.play
      || actions.play === 'arm' && (!selectionAllowed || !batteryReviewed || !scenario || !breath.actions.continue);
    text(play.firstElementChild, status?.p === 'RUNNING' ? 'Pause' : status?.p === 'PAUSED' ? 'Fortsetzen' : 'Starten');
    finish.hidden = !active && actions.finish !== 'save-next';
    if (finish.hidden && hadFinishFocus) statusText.focus({ preventScroll: true });
    finish.disabled = sessionWaiting || unknown || !actions.finish;
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
    exportStatus.hidden = !exportMessage || !status && !archiveScope;
    text(exportStatus, exportMessage === 'Keine gespeicherten Aufnahmen auf der Quest.' ? 'Keine gespeicherten Aufnahmen.' : exportMessage);
    retry.hidden = !rehearsal.gateway?.recordings || status?.p !== 'IDLE';
    retry.disabled = waiting || unknown || Boolean(exportAttempt);
    archivePicker.parentElement.hidden = !archiveScope || !archivePicker.options.length;
    download.hidden = !archiveScope || !download.hasAttribute('href');
    paintOperatorLayout({ status, statusMessage, active, waitingForControl, participantComplete,
      updateBlocked: Boolean(recovery && !pendingProbeStart || unknown || nativeRecovering || controllerLease.pending),
      offlineArchives: Boolean(!status && archiveScope && !recovery && !busy && !unknown && !pending && !breathing.busy),
      requiresParticipant, selectionAllowed, batteryReviewed, polarAttempts, updateRequired, inputReady: breath.actions.continue, recovery: Boolean(recovery || breath.recovery
        || unknown || pending || status?.pending || nativeRecovering || breath.unknown),
      instructionPending: breath.instructionPending,
      calibrationComplete: Boolean(breath.actions.continue && breath.showPolarCalibration) });
    renderLive(performance.now());
    scheduleBreathExpiry();
    scheduleAnimation();
  }

  function paintBreathingSetup(view, waiting) {
    const native = view.setup;
    text(byId('calibration-screen-title'), view.instructionPending ? 'Anleitung zur Atemübung' : 'Zwei Atemzyklen');
    if (instructionCancel.hidden === view.instructionPending) instructionCancel.hidden = !view.instructionPending;
    if (calibrationChangeInput.hidden !== view.instructionPending) calibrationChangeInput.hidden = view.instructionPending;
    instructionCancel.disabled = !view.actions.cancelInstruction || connecting || controllerLease.inFlight || Boolean(userAction);
    const testingMethod = view.available && native.stage === 'testing' && native.playing && native.input !== 'automatic';
    const method = {
      'polar-tilt': 'Polar kalibriert',
      'polar-tilt-fixed': 'Polar · feste Achse',
      'polar-tilt-adaptive': 'Polar · dynamische Achse',
      'controller-state': 'Controller State',
    }[native?.input] ?? view.inputLabel;
    text(byId('operator-confirmed-input'), view.available ? `Aktiv: ${method}` : 'Keine bestätigte Messmethode.');
    previewCircle.hidden = !view.available;
    if (view.available) previewCircle.style.transform = `scale(${view.radius})`;
    text(previewStatus, !view.available ? 'Warte auf den Atemtakt der Quest.'
      : !native.playing ? native.stage === 'idle' ? 'Automatic ist bereit.' : 'Atemschleife auf der Quest angehalten.' : `Atemtakt: ${view.phaseLabel}`);
    previewSwitch.hidden = !view.pendingInputLabel || Boolean(view.recovery);
    text(previewSwitch, view.pendingInputLabel ? `Angefragt: ${view.pendingInputLabel}. Warte auf die Quest.` : '');
    const collecting = view.showPolarCalibration && ['waiting', 'collecting'].includes(native?.calibration);
    const detail = view.updateRequired ? 'Quest-App aktualisieren: Dieser Build unterstützt die Atemeinrichtung noch nicht.'
      : view.recovery ? ''
      : view.unknown ? 'Ergebnis unklar. Prüfe die Quest. Die Aktion wird nicht wiederholt; warte auf den bestätigten Status.'
      : view.recovering ? 'Die Quest prüft die Atemschleife und startet sie neu. Warte auf die Bestätigung. Die Einrichtung beginnt erst danach.'
      : view.pending ? 'Warte auf Bestätigung der Quest …'
      : !view.available ? 'Kein aktueller Atemstatus verfügbar. Prüfe die Verbindung zur Quest.'
      : view.recoveryUnavailable ? view.blockedReason
      : view.message ? view.message
      : view.failure && native.input === 'automatic' ? `Einrichtung fehlgeschlagen: ${view.failureLabel}.`
      : view.blockedReason ? view.blockedReason
      : view.showPolarCalibration && native.calibration === 'waiting' ? 'Die Einrichtung beginnt mit dem nächsten Atemzyklus.'
      : view.showPolarCalibration && native.calibration === 'collecting' ? `Folge zwei Atemzyklen.${view.qualityWarning ? ` ${view.qualityWarning}` : ''}`
      : testingMethod ? 'Prüfe die Bewegung im Headset und bestätige die Messmethode.'
      : !native.playing ? native.stage === 'idle'
        ? 'Wähle eine Methode, um die Atemschleife zu starten.' : 'Warte auf die Atemschleife der Quest.'
      : 'Die Atemschleife läuft. Wähle eine Messmethode.';
    text(previewDetail, detail);
    // Review has its own permanent instruction beside the circles. Repeat
    // only an actionable warning/error here, never sample-level readiness.
    previewDetail.hidden = Boolean(view.recovery) || testingMethod && !view.message && !view.failure
      && !view.blockedReason && !view.pending && !view.unknown && !view.recovering;
    previewProgressGroup.hidden = !view.showPolarCalibration || !view.available || view.recovering || !(collecting || native?.attempt > 0);
    previewProgress.value = view.progress.value;
    const now = performance.now();
    const { status, statusTiming: timing } = rehearsal.state;
    const observation = status?.o;
    const age = timing ? observation?.a + timing.transportMs + Math.max(0, now - timing.receivedAt) : 1501;
    const sessionGeneration = status?.rec && status.p !== 'ARMED' ? status.g : null;
    breathDisplay.update(view, observation && { ...observation, a: age }, now, sessionGeneration);
    text(previewProgressText, `${new Intl.NumberFormat('de-DE', { maximumFractionDigits: 2 }).format(view.progress.value)} von 2 Atemzyklen`);
    previewCalibrate.disabled = waiting || !view.actions.calibrate;
    previewCalibrate.hidden = !view.showPolarCalibration;
    text(previewCalibrate.firstElementChild, native?.attempt > 0 ? 'Polar erneut einrichten' : 'Polar persönlich einrichten');
    previewPersonalReturn.hidden = view.showPolarCalibration;
    previewPersonalReturn.disabled = waiting || !view.actions.calibrate;
    for (const [button, mode, input] of [[previewPolarFixed, 'fixed', 'polar-tilt-fixed'], [previewPolarAdaptive, 'adaptive', 'polar-tilt-adaptive']]) {
      button.hidden = !view.supportedModes.includes(mode);
      button.disabled = waiting || !view.actions[`polar-${mode}`];
      button.setAttribute('aria-pressed', String(native?.input === input));
    }
    byId('preview-modes-compatibility').hidden = !view.polarModesUpdateRequired;
    previewController.hidden = false;
    previewController.setAttribute('aria-pressed', String(native?.input === 'controller-state'));
    previewCalibrate.setAttribute('aria-pressed', String(native?.input === 'polar-tilt'));
    previewPersonalReturn.setAttribute('aria-pressed', String(native?.input === 'polar-tilt'));
    previewController.disabled = waiting || !view.actions.controller;
    previewContinue.disabled = waiting || !view.actions.continue;
    previewContinue.hidden = !testingMethod;
    const reviewPrompt = byId('preview-review');
    reviewPrompt.hidden = !testingMethod;
    text(reviewPrompt, native?.input === 'controller-state'
      ? 'Folgen die Partikel dem Controller?'
      : 'Folgen die Partikel im Headset der Atmung?');
    previewRestart.hidden = !testingMethod || !view.showPolarCalibration;
    previewRetry.hidden = !(view.failure || view.instructionFailed) || collecting || view.instructionPending;
    previewRestart.disabled = previewRetry.disabled = waiting || !view.actions.calibrate;
    byId('preview-recovery').hidden = !view.recovery;
    apkView.querySelector('[value="receipt"]').disabled = !view.recovery;
    if (!view.recovery && apkView.value === 'receipt') {
      apkView.value = 'version';
      showApkDetails();
    }
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
    batteryAccepted = '';
    calibrationBaseline = undefined;
  }

  async function setupAct(op) {
    if (userAction || controllerLease.inFlight || connecting || document.hidden || !breathing.view?.actions[op]) return;
    const attempt = new AbortController();
    userAction = attempt;
    if (op === 'calibrate') calibrationBaseline ??= { owner: breathing.gateway.owner, attempt: breathing.view.setup.attempt };
    const request = selectBreathingMethod(breathing, op, { signal: attempt.signal,
      refresh: async () => { await rehearsal.refresh(); await breathing.refresh(); paint(); },
      onPending: paint });
    navigateLocal({ type: 'go', step: op === 'calibrate' ? 'calibration' : 'tracking' });
    paint();
    try { await request; }
    catch (error) { if (!attempt.signal.aborted) throw error; }
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
    if (current) text(scale, `2 s · 130 Hz · ±${current.range} µV · Basislinie zentriert`);
  }

  function renderSessionScene(session, current, now) {
    const owner = rehearsal.gateway?.owner;
    const sceneVisible = Boolean(session && owner && !document.hidden);
    if (sessionScene.hidden === sceneVisible) sessionScene.hidden = !sceneVisible;
    const paused = session?.p === 'PAUSED';
    const radius = particles.render({
      key: sceneVisible ? `${owner.device}/${owner.epoch}/${owner.native_epoch}/${session.g}` : '',
      progress: current?.actualLive ? current.actualProgress : null,
      paused,
      reducedMotion: reducedMotion.matches,
      now,
    });
    if (sceneVisible) {
      text(sceneStatus, radius === null ? 'Kein aktueller Partikelradius'
        : paused ? 'Partikelvorschau pausiert' : 'Partikelradius in der Quest');
      if (sceneRadius.hidden !== (radius === null)) sceneRadius.hidden = radius === null;
      if (radius !== null) text(sceneRadius, `${Math.round(radius * 100)} %`);
      return !paused && Boolean(current?.actualAnimating);
    }
    return false;
  }

  function renderLive(now) {
    const previewVisible = livePreviewVisible();
    const view = breathing.view;
    const session = runningScene();
    const current = previewVisible || session || displayedStep === 'input'
      ? breathDisplay.frame(now, (!previewVisible && !session) || reducedMotion.matches) : null;
    let moving = renderSessionScene(session, current, now);
    const showInputVolume = Boolean(current?.desiredAvailable) && !view?.renderer
      && typeof current.inputVolume === 'number';
    if (view?.available) {
      const measured = view.inputPhaseLabel ? ` · Erkannt: ${view.inputPhaseLabel}` : '';
      const rawLabel = showInputVolume ? ' · Rohwert der Messmethode: ' : '';
      text(previewInputLabel, `Aktive Atmungsmessmethode: ${view.inputLabel}${measured}${rawLabel}`);
    } else text(previewInputLabel, 'Atmungsmessmethode auf der Quest nicht verfügbar.');
    if (previewInputVolume.hidden === showInputVolume) previewInputVolume.hidden = !showInputVolume;
    if (showInputVolume) text(previewInputVolume, `${Math.round(Math.max(0, Math.min(1, current.inputVolume)) * 100)} %`);
    if (previewVisible) {
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
        const native = breathing.view?.setup;
        if (native?.playing && Number.isSafeInteger(native.audio_generation)) {
          text(previewStatus, current.live ? `Atemtakt: ${current.phaseLabel}` : 'Warte auf den aktuellen Atemtakt der Quest.');
        }
        moving = current.animating;
      } else {
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
    if (animationFrame || document.hidden || reducedMotion.matches || !livePreviewVisible() && !runningScene() && graph.hasAttribute('hidden')) return;
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
    for (const link of [download, answerLink]) {
      if (link.hasAttribute('href')) URL.revokeObjectURL(link.href);
      link.removeAttribute('href');
      link.hidden = true;
    }
    selectedQuestionnaire = undefined;
    answerOpen.hidden = true;
  }

  function localArchiveOwner() {
    // rememberPair writes build only after an authenticated join. Its value is
    // provenance, not part of the recording key: APK/actor updates retain files.
    const pair = savedPair;
    if (archivesSuspended || !pair || !/^[a-f0-9]{32}$/.test(pair.device)
      || !/^[a-f0-9]{64}$/.test(pair.build) || typeof pair.serial !== 'string'
      || !(/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(pair.serial)
        || pair.identity_kind === 'app-installation' && pair.serial === '')) return null;
    const live = rehearsal.gateway?.owner;
    if (live && (live.device !== pair.device || live.serial !== pair.serial
      || live.identity_kind !== pair.identity_kind)) return null;
    return { device: pair.device, serial: pair.serial, identity_kind: pair.identity_kind };
  }

  const localArchiveKey = () => {
    const owner = localArchiveOwner();
    return owner ? `${owner.device}/${owner.serial}/${owner.identity_kind ?? ''}` : '';
  };

  function refreshLocalArchives() {
    const scope = localArchiveKey();
    if (archiveScope === scope) return;
    archiveScope = scope;
    archiveListLoaded = false;
    const read = ++archiveListRead;
    archiveRead++;
    archivePicker.replaceChildren();
    clearDownload();
    setQuestionnaire(undefined);
    if (!scope) return;
    const owner = localArchiveOwner();
    void (async () => {
      let store;
      try {
        store = await openQuestRecordingStore(owner);
        const [ids, pendingForm] = await Promise.all([store.ids(), store.pendingQuestionnaire()]);
        if (read !== archiveListRead || scope !== localArchiveKey()) return;
        if (pendingForm) setQuestionnaire(pendingForm);
        showArchives(ids);
        archiveListLoaded = true;
      } catch {
        if (read === archiveListRead) exportMessage = 'Die gespeicherten Aufnahmen konnten nicht geladen werden.';
      } finally {
        store?.close();
        if (read === archiveListRead) paint();
      }
    })();
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
    archiveListRead++;
    archiveScope = '';
    archiveListLoaded = false;
    nextSync = 0;
    exportMessage = '';
    archivePicker.replaceChildren();
    clearDownload();
    setQuestionnaire(undefined);
  }

  async function selectArchive() {
    const owner = localArchiveOwner();
    const scope = localArchiveKey();
    const id = archivePicker.value;
    const read = ++archiveRead;
    clearDownload();
    if (!owner || !scope || !id) return;
    let store;
    try {
      store = await openQuestRecordingStore(owner);
      const [stored, form] = await Promise.all([store.get(id), store.getQuestionnaire(id)]);
      const archive = stored instanceof Blob ? stored : stored?.download;
      if (read !== archiveRead || scope !== localArchiveKey()) return;
      if (!(archive instanceof Blob)) throw Error('Aufnahme fehlt.');
      download.href = URL.createObjectURL(archive);
      download.download = archive instanceof File ? archive.name : `quest-${id}.zip`;
      text(download.firstElementChild, form ? 'Aufnahme' : 'Aufnahme herunterladen');
      selectedQuestionnaire = form;
      answerOpen.hidden = !form || Boolean(form.file);
      if (form?.file instanceof File) {
        answerLink.href = URL.createObjectURL(form.file);
        answerLink.download = form.file.name;
        answerLink.hidden = false;
      }
    } catch {
      if (read === archiveRead) exportMessage = 'Die gespeicherte Aufnahme konnte nicht geöffnet werden.';
    } finally {
      store?.close();
      if (read === archiveRead) paint();
    }
  }

  function showArchives(ids) {
    const timestamp = id => {
      const nanos = /^(?:record-([1-9][0-9]{0,19})|utc-ns-([0-9]{20}))$/.exec(id);
      return nanos ? Number(BigInt(nanos[1] ?? nanos[2]) / 1000000n) : NaN;
    };
    ids = ids.filter(id => timestamp(id) <= 8640000000000000);
    if (ids.length === archivePicker.options.length && ids.every((id, index) => archivePicker.options[index].value === id)) {
      if (ids.length && !download.hasAttribute('href')) void selectArchive();
      return;
    }
    const selected = archivePicker.value;
    archivePicker.replaceChildren(...ids.map(id => {
      const date = new Date(timestamp(id));
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
          const [saved, pendingForm] = await Promise.all([store.ids(), store.pendingQuestionnaire()]);
          signal.throwIfAborted();
          if (pendingForm && !questionnaire?.required) setQuestionnaire(pendingForm);
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
            if (!archive.collectionObserved && archive.download.type === 'text/csv') {
              archive.collectionObserved = { ...await captureUtc({ signal }), stage: 'final-import' };
            }
            signal.throwIfAborted();
            await store.finalizePrefix(id, archive);
            const pendingForm = archive.questionnaire?.required && await store.pendingQuestionnaire();
            signal.throwIfAborted();
            if (pendingForm && !questionnaire?.required) setQuestionnaire(pendingForm);
            drive.schedule();
            void drive.sync();
            known.add(id);
            showArchives([...known].sort().reverse());
            if (existing && archivePicker.value === id) void selectArchive();
            if (pendingForm) paint();
          }
          exportMessage = known.size ? 'Aufnahmen gespeichert.' : 'Keine gespeicherten Aufnahmen auf der Quest.';
          nextSync = performance.now() + 60000;
        } finally { store.close(); }
      });
    } catch {
      if (!signal.aborted) exportMessage = 'Abgleich fehlgeschlagen.';
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
            for await (const prefix of questPrefixBatches(gateway, store, signal, () => captureUtc({ signal }))) {
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

  function canConnectAutomatically() {
    return !document.hidden && !archivesSuspended && !userAction && !rehearsal.state.status
      && autoConnect && canAutoConnect() && location.hash !== '#daten';
  }

  async function connectAutomatically() {
    if (autoConnectInFlight) return;
    autoConnectInFlight = true;
    const previous = savedPair;
    const canCommission = () => savedPair === previous && canConnectAutomatically() && (!previous?.enrollment
      || !previous.identity_kind && !rehearsal.gateway && !rehearsal.pending && !breathing.pending
      && !kiosk.pending && !controllerLease.pending && !metaControl.pending && !savedUpdateIntent
      && !metaHandoff.view(previous) && (!archiveScope || archiveListLoaded && !questionnaire && !savingAnswers));
    try {
      if (previous?.enrollment) await connect(previous);
      else nextConnect = performance.now() + 8000;
      if (!canCommission() || connecting || performance.now() < nextCommissioningProbe) return;
      nextCommissioningProbe = performance.now() + 8000;
      const { readCommissionedQuest } = await import('./quest-commissioning.js?v=5');
      const commissioned = await readCommissionedQuest();
      if (!canCommission() || connecting) return;
      if (commissioned?.state === 'loopback-denied') {
        if (!joinMessage || joinMessage === allowCompanion) {
          joinMessage = bluetoothHint = allowCompanion;
          paint();
        }
      } else {
        if (joinMessage === allowCompanion) {
          joinMessage = bluetoothHint = '';
          paint();
        }
        if (commissioned) await connect(commissioned, false, canCommission);
      }
    } finally { autoConnectInFlight = false; }
  }

  async function tick() {
    if (document.hidden || connecting || userAction) return;
    await rehearsal.refresh();
    if (document.hidden || connecting || userAction) return;
    const acquisition = controllerLease.acquire(rehearsal.gateway, rehearsal.state, () => rehearsal.refresh(),
      { ...controlContext(), current: () => ({ gateway: rehearsal.gateway, state: rehearsal.state, ...controlContext() }) });
    if (controllerLease.inFlight) paint();
    await acquisition;
    if (document.hidden || connecting || userAction) return;
    await breathing.refresh();
    await kiosk.refresh();
    await reconcileSavedUpdate();
    await metaControl.refresh();
    paint();
    if (document.hidden || connecting || userAction) return;
    if (canConnectAutomatically() && performance.now() >= nextConnect) {
      void connectAutomatically();
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

  async function connect(settings, chooseDevice = false, allowAttach) {
    if (connecting) return;
    settings ??= {};
    const attempt = new AbortController();
    const { signal } = attempt;
    connecting = attempt;
    if (chooseDevice) {
      bluetoothCheck = null;
      bluetoothHint = joinMessage = '';
    }
    bridgeFailed = false;
    discoveryInProgress = false;
    const setup = { ...settings, forceHotspot: false };
    try {
      const selection = chooseDevice ? selectQuest(bluetooth) : null;
      paint();
      const join = (endpoint, discovery, transport) => joinQuest({ endpoint, transport, serial: discovery?.owner.serial ?? settings.serial,
        identity_kind: discovery?.owner.identity_kind ?? settings.identity_kind,
        device: discovery?.owner.device ?? settings.device, build: chooseDevice || discovery ? undefined : settings.build,
        enrollment: settings.enrollment, discovery: discovery?.owner, signal });
      let peer = selection ? await selection : null;
      let gateway;
      let endpoint = settings.endpoint;
      let discovery;
      if (chooseDevice) {
        discovery = await readQuest(peer, signal, {});
        if (discovery.owner.device !== settings.device) endpoint = undefined;
      } else {
        // Read the saved headset's current address before trying the old one.
        // An unpaired automatic attempt requires an explicitly commissioned installation.
        bluetoothHint = '';
        try {
          if (await bluetooth?.getAvailability?.() === false) {
            bluetoothHint = enableBluetooth;
          } else {
            if (settings.bluetoothId) {
              const allowed = await bluetooth?.getDevices?.();
              peer = allowed ? allowed.find(device => device.id === settings.bluetoothId) : rememberedDevice;
            }
            if (peer) discovery = await readQuest(peer, signal, setup);
          }
        } catch {
          signal.throwIfAborted();
          bluetoothHint = 'Schalte Bluetooth ein und prüfe den Fernmodus der Quest.';
        }
        signal.throwIfAborted();
        const trySavedAddress = endpoint && performance.now() >= nextSavedProbe;
        if (!discovery) {
          // Chrome may not expose saved grants. The Android companion can start
          // a scan without a chooser. Don't hold up a usable saved Wi-Fi route
          // if the companion or its browser permission is slow to respond.
          const lookup = trySavedAddress ? AbortSignal.any([signal, AbortSignal.timeout(1500)]) : signal;
          discovery = await rediscoverQuest(settings, lookup).catch(error => {
            signal.throwIfAborted();
            if (!trySavedAddress) throw error;
            return null;
          });
          if (discovery?.state !== 'ready' && trySavedAddress) {
            gateway = await join(endpoint).catch(() => null);
            signal.throwIfAborted();
            if (!gateway) nextSavedProbe = performance.now() + 8000;
          }
        }
      }
      if (!gateway) {
        if (discovery?.state && discovery.state !== 'ready') {
          discoveryInProgress = discovery.state !== 'loopback-denied';
          bluetoothCheck = discovery.state === 'searching' ? null : false;
          joinMessage = bluetoothHint = {
            'bluetooth-off': enableBluetooth,
            'searching': lookingForSavedQuest,
            'permission-required': 'Erlaube der Bluetooth-Suche den Zugriff auf Geräte in der Nähe.',
            'loopback-denied': allowCompanion,
          }[discovery.state];
          return;
        }
        if (!discovery) {
          bluetoothCheck = false;
          bluetoothHint ||= 'Schalte Bluetooth ein oder wähle die Quest erneut über „Mit Bluetooth verbinden“.';
          throw Error(bluetoothHint);
        }
        bluetoothCheck = true;
        bluetoothHint = joinMessage = '';
        let pairedThisAttempt = false;
        let connectionFailure;
        const retainEnrollment = enrollment => {
          pairedThisAttempt = true;
          endpoint = discovery.endpoints[0];
          settings = { ...discovery.owner, endpoint, bluetoothId: peer?.id,
            hotspot: settings.device === discovery.owner.device ? settings.hotspot : undefined, enrollment };
          rememberPair(settings);
        };
        const pairNewPhone = async () => {
          if (!chooseDevice || discovery.pairing !== 1) throw Error('Dieses Smartphone ist für diese Quest noch nicht freigegeben.');
          const candidate = discovery.endpoints[0];
          if (!candidate) throw connectionFailure ?? Error('Prüfe Internetzugang und Quest-App für diese Verbindung.');
          const ticket = await pairPhone({ endpoint: candidate, owner: discovery.owner, signal,
            onCode: prompt => showPairingCode(prompt, attempt) });
          signal.throwIfAborted();
          // Human approval already created this independent membership. Preserve
          // it if the following authenticated status request loses its reply.
          retainEnrollment({ controller: ticket.controller, key: ticket.key });
        };
        const duringCommissioning = settings.commissioned === true && !chooseDevice && !settings.enrollment;
        const needsPairing = !settings.enrollment || settings.device && settings.device !== discovery.owner.device;
        if (duringCommissioning) {
          if (!discovery.rtc || settings.device !== discovery.owner.device) {
            throw Error('Die freigegebene Quest ist über WebRTC noch nicht erreichbar.');
          }
        } else if (needsPairing && !discovery.rtc) await pairNewPhone();
        const connectRtc = async enrollment => {
          const { openQuestWebRtc } = await import('./quest-webrtc.js?v=7');
          const session = await openQuestWebRtc(discovery, enrollment, signal,
            chooseDevice && discovery.pairing === 1 ? {
              onCode: prompt => showPairingCode(prompt, attempt),
              onEnrollment: retainEnrollment,
            } : undefined);
          settings = { ...settings, enrollment: session.enrollment };
          gateway = await join(undefined, discovery, session.transport);
          endpoint = discovery.endpoints[0] ?? endpoint;
        };
        let authenticationRejected = false;
        if (discovery.rtc) {
          try {
            await connectRtc(needsPairing && !pairedThisAttempt ? undefined : settings.enrollment);
          } catch (error) {
            signal.throwIfAborted();
            if (duringCommissioning || error.pairingAttempted || error.status === 403 && !chooseDevice) throw error;
            authenticationRejected = error.status === 403;
            connectionFailure = error;
          }
        }
        if (!gateway && !duringCommissioning && needsPairing && !pairedThisAttempt) await pairNewPhone();
        for (const candidate of gateway ? [] : discovery.endpoints) {
          endpoint = candidate;
          try { gateway = await join(endpoint, discovery); break; }
          catch (error) {
            signal.throwIfAborted();
            authenticationRejected ||= error?.status === 403;
            connectionFailure = error;
          }
        }
        if (!gateway && !pairedThisAttempt && chooseDevice && discovery.pairing === 1 && discovery.rtc) {
          try { await connectRtc(undefined); }
          catch (error) {
            signal.throwIfAborted();
            if (error.pairingAttempted) throw error;
            connectionFailure = error;
          }
        }
        if (!gateway && authenticationRejected && !pairedThisAttempt && chooseDevice && discovery.pairing === 1) {
          await pairNewPhone();
          endpoint = discovery.endpoints[0];
          gateway = await join(endpoint, discovery);
        }

        if (!gateway) throw connectionFailure ?? Error('Die Quest antwortet nicht im WLAN.');
      }
      if (signal.aborted || connecting !== attempt || allowAttach && !allowAttach()) { gateway.close(); return; }
      if (savedPair && (savedPair.serial !== gateway.owner.serial || savedPair.device !== gateway.owner.device)) {
        resetPreview();
      }
      rememberPair({ serial: gateway.owner.serial, device: gateway.owner.device,
        ...(gateway.owner.identity_kind ? { identity_kind: gateway.owner.identity_kind } : {}),
        build: gateway.initial.build.packaging_fingerprint, enrollment: settings.enrollment, hotspot: settings.hotspot,
        bluetoothId: peer?.id ?? settings.bluetoothId, endpoint });
      if (peer) rememberedDevice = peer;
      autoConnect = true;
      bluetoothHint = '';
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

  function controlContext() {
    return { visible: !document.hidden,
      blocked: Boolean(userAction || connecting || breathing.busy) };
  }

  function canChangeControl(op) {
    const { status, busy, pending, unknown } = rehearsal.state;
    const access = controllerAccessState(status);
    if (!rehearsal.gateway?.controllerAccess || !access || userAction || controllerLease.inFlight || connecting || document.hidden
      || busy || status.pending || status.setup?.stage === 'recovering') return false;
    // A trusted phone can reclaim free control to read an earlier receipt.
    // The pending operation still guards every session and setup command.
    return op === 'claim' ? access.role === 'waiting' && !access.active_phone && !controllerLease.pending
      : access.role === 'active' && !pending && !unknown && !breathing.pending && !breathing.view?.recovering;
  }

  async function changeControl(op) {
    if (!canChangeControl(op)) return;
    if (op === 'claim') {
      controllerLease.request();
      await controllerLease.acquire(rehearsal.gateway, rehearsal.state, () => rehearsal.refresh(),
        { ...controlContext(), current: () => ({ gateway: rehearsal.gateway, state: rehearsal.state, ...controlContext() }) });
      paint();
      return;
    }
    // Releasing control is intentional. Polling must not immediately undo it.
    controllerLease.release();
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
    refreshLocalArchives();
  }

  function currentSessionControls(status, control = controllerLease.update(rehearsal.gateway, rehearsal.state, controlContext())) {
    return Boolean((status?.rec && ['RUNNING', 'PAUSED', 'RECORDING'].includes(status.p)
      || status?.p === 'ARMED' && rehearsalActions(status, operatorCondition).finish === 'save-next')
      && control.active && !controllerLease.pending && !document.hidden);
  }

  async function act(action) {
    const { status, busy, pending, unknown } = rehearsal.state;
    const op = rehearsalActions(status, operatorCondition)[action];
    const independentSession = ['pause', 'resume', 'save-next'].includes(op) && currentSessionControls(status);
    if (document.hidden || connecting || userAction || controllerLease.inFlight || controllerLease.pending
      || busy || pending || unknown || status?.pending || breathing.busy || status?.setup?.stage === 'recovering'
      || (breathing.pending || breathing.view?.recovering) && !independentSession) return;
    ecgAttempt?.abort();
    exportAttempt?.abort();
    liveAttempt?.abort();
    drive.cancel();
    nextSync = 0;
    const attempt = new AbortController();
    userAction = attempt;
    const scenario = scenarioChoices.hidden ? '' : selectedScenario();
    try { await rehearsal.run(action, attempt.signal, operatorCondition, participantId, scenario); }
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
    saveUpdateMessage = '';
    const attempt = new AbortController(); userAction = attempt;
    const request = metaControl.request(attempt.signal);
    if (metaControl.busy) metaHandoff.manualRequest(rehearsal.gateway, rehearsal.state);
    paint();
    try { await request; }
    finally { if (userAction === attempt) userAction = null; paint(); }
  });
  byId('meta-update-finish').addEventListener('click', async () => {
    if (byId('meta-update-finish').disabled || userAction || connecting || document.hidden) return;
    const gateway = rehearsal.gateway, clicked = rehearsal.state.status, owner = gateway && { ...gateway.owner };
    const actor = { ...controllerAccessState(clicked) }, original = metaControl.view?.value && { ...metaControl.view.value };
    if (!gateway || !original || !currentSessionControls(clicked) || rehearsalActions(clicked, operatorCondition).finish !== 'save-next') return;
    const attempt = new AbortController(); userAction = attempt;
    saveForUpdate = true; saveUpdateMessage = '';
    ecgAttempt?.abort(); exportAttempt?.abort(); liveAttempt?.abort(); drive.cancel();
    paint();
    try {
      const confirmed = await rehearsal.run('finish', attempt.signal, operatorCondition, participantId, '', { confirmSave: true });
      if (!confirmed || attempt.signal.aborted || document.hidden || rehearsal.gateway !== gateway) {
        saveUpdateMessage = 'Der Speicherabschluss ist noch nicht bestätigt. Prüfe die ursprüngliche Sitzungsanfrage; das Update wird nicht erneut angefragt.';
        return;
      }
      savedUpdateIntent = { gateway, owner, actor, previousCycle: original.cycle,
        cycle: String(BigInt(original.cycle) + 1n), previousAttempt: original.attempt, signal: attempt.signal };
      await rehearsal.refresh();
      await reconcileSavedUpdate();
    } finally {
      saveForUpdate = false;
      if (userAction === attempt) userAction = null;
      paint();
    }
  });
  async function reconcileSavedUpdate() {
    const intent = savedUpdateIntent;
    if (!intent || savedUpdateReconciling || document.hidden || connecting) return;
    const state = rehearsal.state, gateway = rehearsal.gateway;
    const sameConnection = !intent.signal.aborted && gateway === intent.gateway && sameQuest(gateway.owner, intent.owner);
    if (sameConnection && !state.status) {
      saveUpdateMessage = 'Warte auf den aktuellen Quest-Stand.';
      return;
    }
    const access = controllerAccessState(state.status), value = metaUpdateView(state.status).value;
    if (!sameConnection
      || state.status?.o?.e !== intent.owner.native_epoch
      || state.unknown || state.pending || state.status?.rec || state.status?.pending
      || access?.role !== 'active' || access.revision !== intent.actor.revision || access.active_phone !== intent.actor.active_phone
      || value && ![intent.previousCycle, intent.cycle].includes(value.cycle)) {
      savedUpdateIntent = null;
      saveUpdateMessage = 'Sitzung gespeichert; der aktuelle Quest-Stand muss erneut bestätigt werden. Prüfe die Verbindung und wähle das Update erneut.';
      return;
    }
    saveUpdateMessage = 'Warte auf die bestätigte Ruheposition der Quest.';
    if (!value || value.cycle !== intent.cycle || state.status.p !== 'IDLE' || state.busy
      || !controllerLease.update(gateway, state, controlContext()).active) return;
    // A next-cycle old "current" value cannot prove a new check while the real
    // audio owner is still parking. Follow only this cycle's admitted attempt.
    if (value.attempt !== null && value.attempt !== intent.previousAttempt
      && ['checking', 'downloading', 'installing', 'current', 'updated', 'failed', 'restart_required', 'update_required'].includes(value.state)) {
      savedUpdateIntent = null; saveUpdateMessage = '';
      return;
    }
    const controls = metaControl.update(gateway, state);
    if (!controls.canRequest) return;
    savedUpdateReconciling = true;
    savedUpdateIntent = null; saveUpdateMessage = '';
    try {
      // Native can_check is the actual park/writer boundary. The journal owns
      // the one new request before transport; receipt loss never resends it.
      await metaControl.request(intent.signal);
      if (metaControl.pending) {
        metaHandoff.manualRequest(gateway, state, metaControl.pending.accepted);
      }
    } finally { savedUpdateReconciling = false; }
  }
  byId('meta-update-save-retry').addEventListener('click', async () => {
    if (byId('meta-update-save-retry').disabled || document.hidden) return;
    await rehearsal.refresh(); await reconcileSavedUpdate(); paint();
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
  instructionCancel.addEventListener('click', async () => {
    if (instructionCancel.disabled || document.hidden) return;
    const cancel = breathing.cancelInstruction();
    paint();
    await cancel;
    await rehearsal.refresh();
    await breathing.refresh();
    paint();
  });
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
    navigateLocal({ type: 'go', step: 'battery' });
    paint();
  });
  byId('battery-next').addEventListener('click', () => {
    if (document.hidden || displayedStep !== 'battery' || byId('battery-next').disabled
      || !previewAccepted || previewAccepted !== breathing.view?.acceptance) return;
    batteryAccepted = `${previewAccepted}/${participantId}`;
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
    const liveScene = runningScene()?.p === 'RUNNING' && Boolean(breathing.view?.setup?.renderer);
    if (document.hidden || (!livePreviewVisible() && !liveScene) || connecting || userAction || !rehearsal.gateway?.owner) return;
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
      if (savedUpdateIntent) {
        savedUpdateIntent = null;
        saveUpdateMessage = 'Sitzung gespeichert. Wähle das Update erneut, sobald die Quest-Steuerung bestätigt ist.';
      }
      breathing.cancel();
      ecgAttempt?.abort();
      exportAttempt?.abort();
      liveAttempt?.abort();
      drive.cancel();
      showTrace(null);
      paint();
    } else {
      nextConnect = nextSavedProbe = nextSync = 0;
      paint();
      renderLive(performance.now());
      void apkRelease.refresh();
      void tick();
    }
  });
  bluetooth?.addEventListener?.('availabilitychanged', () => {
    nextConnect = nextSavedProbe = 0;
    void tick();
  });
  reducedMotion.addEventListener('change', () => {
    cancelAnimationFrame(animationFrame);
    animationFrame = 0;
    renderLive(performance.now());
    scheduleBreathExpiry();
    scheduleAnimation();
  });
  window.addEventListener('pagehide', (event) => {
    archivesSuspended = true;
    clearTimeout(breathTimer);
    clearTimeout(controlTimer);
    clearTimeout(deviceTimer);
    cancelAnimationFrame(animationFrame);
    animationFrame = 0;
    userAction?.abort();
    connecting?.abort();
    connecting = null;
    detach();
    if (!event.persisted) { clearInterval(timer); clearInterval(previewTimer); }
  });
  window.addEventListener('pageshow', () => { archivesSuspended = false; paint(); nextConnect = nextSavedProbe = 0; void tick(); });
  window.addEventListener('storage', event => {
    if (event.key !== null && event.key !== 'atmungsraum-quest-pair') return;
    try { savedPair = JSON.parse(localStorage.getItem('atmungsraum-quest-pair')); }
    catch { savedPair = null; }
    paint();
  });
  window.addEventListener('online', () => { nextConnect = nextSavedProbe = nextLive = 0; drive.schedule(); void tick(); });
  paint();
  void apkRelease.refresh();
  void tick();
}
