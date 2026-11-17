import { operationId, sameQuest, canControlQuest } from './quest-gateway.js?v=14';
import { nativeSessionAllowed, metaUpdateView } from './meta-update.js?v=6';

const stages = new Set(['idle', 'recovering', 'waiting', 'calibrating', 'testing', 'main-pending', 'main', 'complete']);
const inputs = { automatic: 'Automatic', 'polar-tilt': 'Polar tilt · persönlich eingerichtet',
  'polar-tilt-fixed': 'Polar tilt STATE · vortrainiert · feste Achse',
  'polar-tilt-adaptive': 'Polar tilt STATE · vortrainiert · dynamische Achse',
  'controller-state': 'Controller State · feste Ausrichtung' };
const polarModes = setup => Array.isArray(setup?.polar_modes) && setup.polar_modes.length <= 3
  && new Set(setup.polar_modes).size === setup.polar_modes.length
  && setup.polar_modes.every(mode => ['calibrated', 'fixed', 'adaptive'].includes(mode)) ? setup.polar_modes : [];
const phases = ['inhale', 'hold_in', 'exhale', 'hold_out'];
const phaseLabels = ['Einatmen', 'Atem halten', 'Ausatmen', 'Atem halten'];
const inputPhases = { inhale: 'Einatmen', exhale: 'Ausatmen', hold: 'Atem halten', unknown: 'noch nicht erkannt', 'bad-tracking': 'Tracking unzuverlässig' };
const calibrations = new Set(['idle', 'waiting', 'collecting', 'ready', 'failed']);
const operations = new Set(['start', 'calibrate', 'controller', 'polar-fixed', 'polar-adaptive']);
const outcomes = new Set(['accepted', 'pending', 'confirmed', 'rejected', 'outcome_unknown']);
const journalKey = 'atmungsraum-breathing-setup-pending';
const nonnegative = value => Number.isSafeInteger(value) && value >= 0;
const validRenderer = value => value?.v === 1 && typeof value.progress01 === 'number' && Number.isFinite(value.progress01)
  && value.progress01 >= 0 && value.progress01 <= 1 && typeof value.radius_m === 'number' && Number.isFinite(value.radius_m) && value.radius_m > 0
  && nonnegative(value.frame) && nonnegative(value.render_generation) && value.render_generation > 0
  && nonnegative(value.settings_revision) && typeof value.observed_ns === 'string' && /^(0|[1-9][0-9]{0,19})$/.test(value.observed_ns)
  && nonnegative(value.age_ns) && value.age_ns <= 500000000;
const receiptId = value => typeof value === 'string' && /^[a-f0-9]{16}$/.test(value);
const recoveryFailures = new Set(['audio-recovery-failed', 'audio-recovery-timeout', 'audio-recovery-retired']);
const validRecovery = value => value && !Array.isArray(value) && Object.keys(value).length === 4
  && receiptId(value.id) && value.state === 'pending' && nonnegative(value.deadline_ns)
  && (value.target_generation === null || nonnegative(value.target_generation) && value.target_generation > 0);
const validReceipt = value => value && !Array.isArray(value) && value.v === 1 && receiptId(value.id)
  && outcomes.has(value.state)
  && Object.keys(value).every(key => ['v', 'id', 'state', 'accepted', 'reason', 'recovery_generation'].includes(key))
  && (!Object.hasOwn(value, 'recovery_generation') || value.recovery_generation === null
    || nonnegative(value.recovery_generation) && value.recovery_generation > 0)
  && (!Object.hasOwn(value, 'accepted') || typeof value.accepted === 'boolean'
    && (value.state === 'rejected' ? !value.accepted : value.state === 'outcome_unknown' || value.accepted))
  && (!Object.hasOwn(value, 'reason') || typeof value.reason === 'string' && value.reason.length <= 256
    && !/[\x00-\x1f\x7f]/.test(value.reason))
  && (!['audio-recovery-pending', 'calibration-instruction-pending'].includes(value.reason)
    || value.state === 'pending' && value.accepted === true)
  && (!recoveryFailures.has(value.reason) || value.state === 'rejected' && value.accepted === false);
const qualityWarnings = {
  '': '', motion: 'Bewegung erkannt. Halte den Oberkörper ruhig und atme weiter mit.',
  timing: 'Einige Messdaten kommen unregelmäßig an. Atme weiter mit.',
  invalid: 'Einige Messdaten sind nicht nutzbar. Prüfe den Sitz des Gurts und atme weiter mit.',
};
const failureLabels = {
  insufficient_coverage: 'Zu wenige nutzbare Atembewegungen wurden erfasst',
  insufficient_range: 'Die Atembewegung war für die Einrichtung zu klein',
  poor_guided_motion: 'Die Bewegung passte noch nicht zuverlässig zum Atemtakt',
  excessive_motion: 'Zu viel Bewegung',
  invalid_sample: 'Zu viele Messdaten waren nicht nutzbar',
  signal_gap: 'Zu viele Messdaten fehlten oder kamen unregelmäßig an',
  audio_paused: 'Die Atemschleife wurde angehalten',
  source_replaced: 'Die Verbindung zum Gurt wurde während der Einrichtung erneuert',
  invalid_source: 'Die Messdaten konnten dem Gurt nicht zuverlässig zugeordnet werden',
  clock_retired: 'Die Atemschleife wurde während der Einrichtung erneuert',
  clock_reversed: 'Die Zeitzuordnung der Messdaten wurde unterbrochen',
  clock_stale: 'Der aktuelle Atemtakt war nicht mehr verfügbar',
  clock_stalled: 'Die Atemschleife lief nicht weiter',
  frame_overflow: 'Die Zeitzuordnung der Atemschleife war nicht mehr nutzbar',
};
const validQuality = quality => quality && !Array.isArray(quality) && Object.keys(quality).length === 7
  && ['accepted', 'masked', 'motion', 'timing', 'invalid'].every(key => nonnegative(quality[key]) && quality[key] <= 800)
  && typeof quality.usable_fraction === 'number'
  && Number.isFinite(quality.usable_fraction) && quality.usable_fraction >= 0 && quality.usable_fraction <= 1
  && typeof quality.warning === 'string' && Object.hasOwn(qualityWarnings, quality.warning);
const validOwner = owner => owner && /^[a-f0-9]{32}$/.test(owner.device)
  && typeof owner.serial === 'string' && (/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(owner.serial)
    || owner.identity_kind === 'app-installation' && owner.serial === '')
  && ['epoch', 'native_epoch'].every(key => typeof owner[key] === 'string' && /^[1-9][0-9]{0,18}$/.test(owner[key]));

// Status is already authenticated by the gateway. Recheck the bounded native
// projection here; neither a phone clock nor a completed frame count proves a
// successful tilt model.
export function breathingSetup(status, owner) {
  const setup = status?.setup;
  const observation = status?.o;
  if (status?.breath_setup !== 1 || !validOwner(owner) || observation?.v !== 1
    || observation.s !== 'observed' || observation.e !== owner.native_epoch
    || !nonnegative(observation.a) || observation.a > 1500
    || typeof observation.i !== 'string' || !/^[1-9][0-9]{0,18}$/.test(observation.i)
    || setup?.v !== 1 || !nonnegative(setup.g) || setup.g === 0 || !nonnegative(setup.r)
    || !stages.has(setup.stage) || !Object.hasOwn(inputs, setup.input) || !phases.includes(setup.phase)
    || !nonnegative(setup.position_frames) || !nonnegative(setup.cycle_frames)
    || setup.cycle_frames === 0 || setup.cycle_frames % 4 !== 0 || !Number.isSafeInteger(setup.cycle_frames * 2)
    || !nonnegative(setup.target_frames) || ![0, setup.cycle_frames * 2].includes(setup.target_frames)
    || !nonnegative(setup.completed_frames) || setup.completed_frames > setup.target_frames
    || Object.hasOwn(setup, 'instruction_id') && (!receiptId(setup.instruction_id) || setup.attempt !== 0)
    || !nonnegative(setup.attempt) || !calibrations.has(setup.calibration)
    || typeof setup.failure !== 'string' || setup.failure.length > 256 || /[\x00-\x1f\x7f]/.test(setup.failure)
    || typeof setup.ready !== 'boolean' || typeof setup.playing !== 'boolean'
    || Object.hasOwn(setup, 'source_clock_ready') && typeof setup.source_clock_ready !== 'boolean'
    || Object.hasOwn(setup, 'clock_valid') && typeof setup.clock_valid !== 'boolean'
    || Object.hasOwn(setup, 'audio_recovery_available') && typeof setup.audio_recovery_available !== 'boolean'
    || Object.hasOwn(setup, 'recovery') && setup.recovery !== null && !validRecovery(setup.recovery)
    || Object.hasOwn(setup, 'parked') && typeof setup.parked !== 'boolean'
    || Object.hasOwn(setup, 'quality') && !validQuality(setup.quality)
    || Object.hasOwn(setup, 'renderer') && setup.renderer !== null && !validRenderer(setup.renderer)
    || !nonnegative(setup.experiment_elapsed_ms) || setup.experiment_elapsed_ms > 900000
    || Object.hasOwn(setup, 'input_phase') && !Object.hasOwn(inputPhases, setup.input_phase)
    || Object.hasOwn(setup, 'input_volume01') && setup.input_volume01 !== null
      && (typeof setup.input_volume01 !== 'number' || !Number.isFinite(setup.input_volume01)
        || setup.input_volume01 < 0 || setup.input_volume01 > 1)) return null;
  const phase = Math.floor((setup.position_frames % setup.cycle_frames) / (setup.cycle_frames / 4));
  if (setup.phase !== phases[phase]
    || setup.stage === 'recovering' && (!setup.recovery || setup.ready)
    || setup.recovery && setup.stage !== 'recovering'
    || ['waiting', 'collecting'].includes(setup.calibration) && (setup.input !== 'automatic' || setup.ready)
    || setup.calibration === 'collecting' && setup.target_frames !== setup.cycle_frames * 2
    || setup.ready && (setup.input === 'automatic' || !['testing', 'main-pending', 'main', 'complete'].includes(setup.stage))
    || setup.input === 'polar-tilt' && setup.ready && setup.calibration !== 'ready'
    || setup.input === 'polar-tilt-fixed' && !polarModes(setup).includes('fixed')
    || setup.input === 'polar-tilt-adaptive' && !polarModes(setup).includes('adaptive')) return null;
  return setup;
}

function validJournal(saved) {
  const command = saved?.command;
  return validOwner(saved?.owner) && command?.v === 1 && receiptId(command.id)
    && nonnegative(command.g) && command.g > 0 && nonnegative(command.r) && operations.has(command.op)
    && Object.keys(command).length === 5
    && (!Object.hasOwn(saved, 'recovery') || saved.recovery === true)
    && (!Object.hasOwn(saved, 'instruction') || saved.instruction === true && command.op === 'calibrate');
}

export class BreathingSetup {
  pending = null;
  busy = false;
  unknown = false;
  message = '';
  gateway = null;
  setup = null;
  cursor = null;
  refreshing = null;
  refreshAttempt = null;
  waitBudget = null;
  waitExpired = false;
  cancellingInstruction = false;
  instructionFailed = false;

  constructor({ storage, now = () => performance.now() } = {}) {
    this.storage = storage;
    this.now = now;
    try {
      const saved = JSON.parse(storage?.getItem(journalKey) ?? 'null');
      if (validJournal(saved)) {
        this.pending = saved;
        this.unknown = true;
      }
    } catch { /* Browser storage is optional; in-memory commands remain bound. */ }
  }

  remember(pending) {
    if (this.pending?.command.id !== pending?.command.id) {
      this.waitBudget = null;
      this.waitExpired = false;
    }
    this.pending = pending;
    try {
      if (pending) this.storage?.setItem(journalKey, JSON.stringify(pending));
      else this.storage?.removeItem(journalKey);
    } catch { /* Never replace or resend an in-memory command after storage failure. */ }
  }

  update(gateway, state, participantReady) {
    if (this.gateway !== gateway) {
      this.cancel();
      this.waitBudget = null;
      this.waitExpired = false;
    }
    this.gateway = gateway;
    const { status, busy, pending, unknown } = state;
    this.status = status;
    let setup = breathingSetup(status, gateway?.owner);
    if (this.cursor && sameQuest(this.cursor.owner, gateway?.owner) && setup) {
      if (setup.g < this.cursor.g || setup.g === this.cursor.g
        && (setup.r < this.cursor.r || setup.position_frames < this.cursor.position)) setup = null;
    }
    if (setup) this.cursor = { owner: { ...gateway.owner }, g: setup.g, r: setup.r, position: setup.position_frames };
    this.setup = setup;
    if (this.pending && sameQuest(this.pending.owner, gateway?.owner)
      && setup?.recovery?.id === this.pending.command.id && !this.pending.recovery) {
      this.remember({ ...this.pending, recovery: true });
    }
    const instructionPending = Boolean(setup?.instruction_id);
    if (instructionPending && this.pending?.command.op === 'calibrate'
      && setup.instruction_id === this.pending.command.id && sameQuest(this.pending.owner, gateway?.owner)) {
      this.observeInstruction();
    }
    if (this.pending && (!sameQuest(this.pending.owner, gateway?.owner)
      || setup && setup.g !== this.pending.command.g && !this.pending.recovery)) this.unknown = true;
    const available = Boolean(setup && typeof gateway?.breathSetup === 'function');
    const ownerIdle = available && canControlQuest(status)
      && status.p === 'IDLE' && status.h === 'vr-preparation' && !status.rec && !status.pending
      && !busy && !pending && !unknown && !this.busy;
    const recovering = Boolean(setup?.recovery || this.pending?.recovery);
    const recoveryUnavailable = setup?.clock_valid === false && setup.audio_recovery_available === false;
    const updateAllowed = nativeSessionAllowed(status);
    const idle = ownerIdle && updateAllowed && !this.pending && !this.unknown && !recovering && !recoveryUnavailable;
    const gated = idle && participantReady
      && (status.pi?.r === false || status.pi?.r === true && status.pi.f === true);
    const ownerRetired = this.pending && gateway?.owner
      && this.pending.owner.device === gateway.owner.device && this.pending.owner.serial === gateway.owner.serial
      && (this.pending.owner.epoch !== gateway.owner.epoch
        || this.pending.owner.native_epoch !== gateway.owner.native_epoch);
    const testing = setup?.stage === 'testing' && setup.playing;
    const preparing = setup && ['waiting', 'calibrating', 'testing'].includes(setup.stage) && setup.playing;
    const parked = setup?.stage === 'idle' && !setup.playing;
    const recoverable = setup?.audio_recovery_available === true
      && ['idle', 'waiting', 'calibrating', 'testing'].includes(setup.stage);
    const phaseIndex = setup ? phases.indexOf(setup.phase) : -1;
    const phaseFraction = setup ? (setup.position_frames % (setup.cycle_frames / 4)) / (setup.cycle_frames / 4) : 0;
    const radius = [.46 + .54 * phaseFraction, 1, 1 - .54 * phaseFraction, .46][phaseIndex] ?? .46;
    const controller = setup?.input === 'controller-state';
    const builtin = ['polar-tilt-fixed', 'polar-tilt-adaptive'].includes(setup?.input);
    const choosingAlternative = ['polar-fixed', 'polar-adaptive', 'controller'].includes(this.pending?.command.op);
    const showPolarCalibration = !controller && !builtin && !choosingAlternative;
    const supportedModes = polarModes(setup);
    const polarChoice = Boolean(gated && (preparing || parked || recoverable)
      && (setup.source_clock_ready !== false || recoverable));
    const pendingInputLabel = this.pending ? { calibrate: 'Polar einrichten', controller: 'Controller auswählen', start: 'Atemschleife starten',
      'polar-fixed': inputs['polar-tilt-fixed'], 'polar-adaptive': inputs['polar-tilt-adaptive'] }[this.pending.command.op] : '';
    this.view = {
      available, updateRequired: Boolean(status && status.breath_setup !== 1), setup,
      pending: Boolean(this.pending || this.busy), unknown: this.unknown, recovering, recoveryUnavailable,
      instructionPending, instructionFailed: this.instructionFailed,
      phaseLabel: phaseLabels[phaseIndex] ?? '', radius,
      inputLabel: setup ? inputs[setup.input] : '',
      title: controller ? 'Controller testen' : builtin ? 'Polar STATE testen' : 'Atmung einrichten',
      showPolarCalibration, supportedModes,
      polarModesUpdateRequired: Boolean(setup && (!supportedModes.includes('fixed') || !supportedModes.includes('adaptive'))),
      pendingInputLabel,
      inputPhaseLabel: setup && inputPhases[setup.input_phase] || '',
      inputVolume: typeof setup?.input_volume01 === 'number' ? setup.input_volume01 : null,
      renderer: setup?.renderer ?? null,
      failure: showPolarCalibration && setup?.calibration === 'failed' ? setup.failure : '',
      failureLabel: showPolarCalibration && setup?.calibration === 'failed' ? failureLabels[setup.failure] ?? setup.failure : '',
      qualityWarning: showPolarCalibration && setup?.calibration === 'collecting' ? qualityWarnings[setup.quality?.warning ?? ''] : '',
      progress: setup?.target_frames ? { value: 2 * setup.completed_frames / setup.target_frames, max: 2 } : { value: 0, max: 2 },
      acceptance: setup ? `${gateway.owner.device}/${gateway.owner.epoch}/${gateway.owner.native_epoch}/${setup.g}/${setup.attempt}/${setup.input}/${status.y?.s ?? ''}/${status.y?.g ?? ''}/${status.y?.x ?? ''}` : '',
      message: this.message,
      blockedReason: !canControlQuest(status) ? 'Übernimm zuerst die Quest-Steuerung auf diesem Smartphone.'
        : !updateAllowed ? metaUpdateView(status).summary
        : recoveryUnavailable ? 'Die Atemschleife ist nicht bereit. Öffne die Quest-App erneut und verbinde dieses Smartphone wieder.'
        : !participantReady ? 'Teilnehmer-ID und Polar prüfen.'
        : status?.pi?.r === true && status.pi.f !== true ? 'Warte auf aktuelle Polar-H10-Daten von der Quest.'
        : setup?.audio_recovery_available === true ? 'Die Atemschleife läuft nicht weiter. Wähle erneut „Atmung einrichten“ oder Controller State.'
        : setup?.input === 'automatic' && setup.source_clock_ready === false
          ? 'Die Messsignale sind da. Warte kurz auf gleichmäßige Bewegungsdaten vom Gurt.' : '',
      actions: {
        start: false,
        cancelInstruction: Boolean(instructionPending && available && canControlQuest(status)
          && status.p === 'IDLE' && !status.rec && !this.busy && !this.cancellingInstruction),
        calibrate: polarChoice,
        'polar-fixed': Boolean(polarChoice && supportedModes.includes('fixed')
          && (setup.input !== 'polar-tilt-fixed' || recoverable)),
        'polar-adaptive': Boolean(polarChoice && supportedModes.includes('adaptive')
          && (setup.input !== 'polar-tilt-adaptive' || recoverable)),
        controller: Boolean(idle && (preparing || parked || recoverable)
          && (setup.input !== 'controller-state' || recoverable)),
        continue: Boolean(gated && testing && setup.ready && setup.clock_valid !== false
          && ['polar-tilt', 'polar-tilt-fixed', 'polar-tilt-adaptive', 'controller-state'].includes(setup.input)),
        // Audio generation changes can belong to an unobserved recovery of
        // this exact operation. Only actual actor retirement permits discard.
        acknowledge: Boolean(ownerIdle && ownerRetired),
        checkReceipt: Boolean(available && canControlQuest(status) && this.pending && !this.busy
          && !this.refreshing && (this.waitExpired || this.unknown && setup.g > this.pending.command.g)
          && sameQuest(this.pending.owner, gateway?.owner)),
      },
    };
    const { actions } = this.view;
    this.view.recovery = instructionPending ? {
      title: 'Anleitung im Headset', detail: 'Die Atemzyklen beginnen nach der Anleitung.', action: '',
    } : actions.acknowledge ? {
      title: 'Die Quest-App wurde erneuert', action: 'Aktuellen Quest-Zustand übernehmen',
      detail: 'Die frühere Anfrage gehört zur beendeten Verbindung. Übernimm den aktuellen Zustand, um weiterzumachen. Das bestätigt nur lokal und setzt die Quest nicht zurück. Kein Befehl wird wiederholt.',
    } : this.pending || recovering ? {
      title: this.waitExpired ? 'Bestätigung noch offen' : this.unknown ? 'Ergebnis der Anfrage unklar' : recovering ? 'Atemschleife wird wiederhergestellt' : 'Quest bestätigt die Anfrage',
      action: actions.checkReceipt ? 'Bestätigung erneut abfragen' : '',
      detail: `${pendingInputLabel || 'Atemschleife wiederherstellen'}: ${recovering ? 'Die Quest startet den Takt neu.' : 'Warte auf Bestätigung.'} ${this.waitExpired ? 'Frage die Bestätigung ab. Bleibt sie offen, starte Atmungsraum im Headset neu und verbinde dieses Smartphone wieder. ' : ''}Die Anfrage wird nicht erneut gesendet.`,
    } : recoveryUnavailable ? {
      title: 'Atemschleife benötigt einen App-Neustart', action: '',
      detail: 'Die Quest meldet, dass eine automatische Audio-Wiederherstellung nicht verfügbar ist. Beende und öffne die Atmungsraum-App im Headset erneut; verbinde danach dieses Smartphone. Das startet die App tatsächlich neu. Eine lokale Bestätigung kann diesen Schritt nicht ersetzen.',
    } : null;
    return this.view;
  }

  acknowledgeRetired() {
    if (!this.view?.actions.acknowledge) return false;
    this.remember(null);
    this.unknown = false;
    this.message = 'Die frühere Bestätigung wurde verworfen. Der aktuelle Quest-Status gilt; kein Befehl wird wiederholt.';
    return true;
  }

  observeInstruction() {
    if (this.pending?.command.op !== 'calibrate' || this.pending.instruction) return;
    this.remember({ ...this.pending, instruction: true });
    this.waitExpired = false;
    this.waitBudget = null;
  }

  observeReceipt(receipt) {
    const saved = this.pending;
    if (!saved || !sameQuest(saved.owner, this.gateway?.owner)) return;
    if (!validReceipt(receipt) || receipt.id !== saved.command.id
      || receipt.state === 'outcome_unknown') {
      this.unknown = true;
      return;
    }
    if (receipt.state === 'rejected') {
      this.instructionFailed = Boolean(receipt.reason?.startsWith('calibration-instruction-'));
      this.remember(null);
      this.unknown = false;
      this.message = recoveryFailures.has(receipt.reason)
        ? 'Die Atemschleife konnte nicht neu gestartet werden. Prüfe die Quest und versuche es erneut.'
        : receipt.reason === 'calibration-instruction-cancelled' ? 'Die Anleitung wurde abgebrochen.'
        : receipt.reason?.startsWith('calibration-instruction-') ? 'Die Quest konnte die Anleitung nicht abschließen. Versuche es erneut.'
        : receipt.reason === 'stale-revision' ? 'Der Quest-Status hat sich während des Wechsels geändert. Wähle die Atmungsmessmethode bitte erneut.'
        : 'Die Quest hat die Aktion abgelehnt. Prüfe den aktuellen Status.';
    } else {
      if (receipt.reason === 'audio-recovery-pending' && !saved.recovery) this.remember({ ...saved, recovery: true });
      if (receipt.reason === 'calibration-instruction-pending') this.observeInstruction();
      // The authenticated native lookup binds this terminal result to the exact
      // operation, even when recovery advanced the audio generation while the
      // original admission reply was lost. It does not establish calibration.
      const resolvedRecovery = receipt.state === 'confirmed' && receipt.accepted === true && receipt.reason === 'accepted'
        && receipt.recovery_generation > saved.command.g && receipt.recovery_generation === this.setup?.g;
      this.unknown = !this.setup || this.setup.g !== saved.command.g && !this.pending.recovery;
      if (resolvedRecovery || !saved.recovery && receipt.state === 'confirmed' && !this.unknown && this.setup.r > saved.command.r) {
        this.remember(null);
        this.unknown = false;
        this.message = '';
      }
    }
  }

  async run(op, signal) {
    if (!operations.has(op) || !this.view?.actions[op] || this.busy || this.pending || this.unknown) return;
    signal?.throwIfAborted();
    const gateway = this.gateway;
    const command = { v: 1, id: operationId(), g: this.setup.g, r: this.setup.r, op };
    this.remember({ owner: { ...gateway.owner }, command });
    this.busy = true;
    this.message = '';
    this.instructionFailed = false;
    try {
      const receipt = await gateway.breathSetup(command, signal);
      if (this.gateway === gateway) {
        this.observeReceipt(receipt);
        if (validReceipt(receipt) && receipt.id === command.id && receipt.state === 'rejected'
          && receipt.accepted === false && receipt.reason === 'stale-revision'
          && !this.pending && !this.unknown) return 'stale-revision';
      }
    } catch { if (this.pending?.command.id === command.id) this.unknown = true; }
    finally { this.busy = false; }
  }

  async cancelInstruction(signal) {
    if (!this.view?.actions.cancelInstruction || this.cancellingInstruction || signal?.aborted) return;
    const gateway = this.gateway;
    const command = { v: 1, id: operationId(), op: 'cancel-instruction', target: this.setup.instruction_id };
    this.cancellingInstruction = true;
    try {
      const receipt = await gateway.breathSetup(command, signal);
      if (this.gateway === gateway && validReceipt(receipt) && receipt.id === command.id
        && receipt.state === 'confirmed' && receipt.accepted === true) return true;
    } catch {
      if (this.gateway === gateway && this.pending?.command.id === command.target) this.unknown = true;
    } finally { this.cancellingInstruction = false; }
    return false;
  }

  // Read receipts only. A reconnect never retransmits the journaled mutation.
  refresh(signal) {
    const gateway = this.gateway;
    if (!this.pending || !sameQuest(this.pending.owner, gateway?.owner) || this.busy || this.refreshing
      || typeof gateway?.breathSetupReceipt !== 'function' || !canControlQuest(this.status)) return this.refreshing;
    if (signal?.aborted) { this.unknown = true; return; }
    const id = this.pending.command.id;
    const now = this.now();
    this.waitBudget ??= { started: now, last: -Infinity, reads: 0 };
    const instruction = this.pending.instruction;
    if (now - this.waitBudget.started >= (instruction ? 65000 : 12000)
      || this.waitBudget.reads >= (instruction ? 130 : 12)) {
      this.waitExpired = true;
      this.unknown = true;
      return;
    }
    if (now - this.waitBudget.last < 500) return;
    this.waitBudget.last = now;
    this.waitBudget.reads++;
    const attempt = new AbortController();
    this.refreshAttempt = attempt;
    this.refreshing = (async () => {
      try {
        const receipt = await gateway.breathSetupReceipt(id, signal ? AbortSignal.any([signal, attempt.signal]) : attempt.signal);
        if (this.gateway === gateway && this.pending?.command.id === id && !attempt.signal.aborted) this.observeReceipt(receipt);
      } catch { if (this.gateway === gateway && this.pending?.command.id === id) this.unknown = true; }
    })().finally(() => {
      if (this.refreshAttempt === attempt) this.refreshAttempt = null;
      this.refreshing = null;
    });
    return this.refreshing;
  }

  recheckReceipt() {
    if (!this.view?.actions.checkReceipt) return false;
    this.waitBudget = null;
    this.waitExpired = false;
    return true;
  }

  cancel() {
    this.refreshAttempt?.abort();
    if (this.pending) this.unknown = true;
  }
}
