import { operationId, sameQuest } from './quest-gateway.js?v=9';

const stages = new Set(['idle', 'waiting', 'calibrating', 'testing', 'main-pending', 'main', 'complete']);
const inputs = { automatic: 'Automatic', 'polar-tilt': 'Polar tilt', 'controller-state': 'Controller State · feste Ausrichtung' };
const phases = ['inhale', 'hold_in', 'exhale', 'hold_out'];
const phaseLabels = ['Einatmen', 'Atem halten', 'Ausatmen', 'Atem halten'];
const inputPhases = { inhale: 'Einatmen', exhale: 'Ausatmen', hold: 'Atem halten', unknown: 'noch nicht erkannt', 'bad-tracking': 'Tracking unzuverlässig' };
const calibrations = new Set(['idle', 'waiting', 'collecting', 'ready', 'failed']);
const operations = new Set(['start', 'calibrate', 'controller']);
const outcomes = new Set(['accepted', 'pending', 'confirmed', 'rejected', 'outcome_unknown']);
const journalKey = 'atmungsraum-breathing-setup-pending';
const nonnegative = value => Number.isSafeInteger(value) && value >= 0;
const validOwner = owner => owner && /^[a-f0-9]{32}$/.test(owner.device)
  && /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(owner.serial)
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
    || !nonnegative(setup.attempt) || !calibrations.has(setup.calibration)
    || typeof setup.failure !== 'string' || setup.failure.length > 256 || /[\x00-\x1f\x7f]/.test(setup.failure)
    || typeof setup.ready !== 'boolean' || typeof setup.playing !== 'boolean'
    || !nonnegative(setup.experiment_elapsed_ms) || setup.experiment_elapsed_ms > 900000
    || Object.hasOwn(setup, 'input_phase') && !Object.hasOwn(inputPhases, setup.input_phase)
    || Object.hasOwn(setup, 'input_volume01') && setup.input_volume01 !== null
      && (typeof setup.input_volume01 !== 'number' || !Number.isFinite(setup.input_volume01)
        || setup.input_volume01 < 0 || setup.input_volume01 > 1)) return null;
  const phase = Math.floor((setup.position_frames % setup.cycle_frames) / (setup.cycle_frames / 4));
  if (setup.phase !== phases[phase]
    || ['waiting', 'collecting'].includes(setup.calibration) && (setup.input !== 'automatic' || setup.ready)
    || setup.calibration === 'collecting' && setup.target_frames !== setup.cycle_frames * 2
    || setup.ready && (setup.input === 'automatic' || !['testing', 'main-pending', 'main', 'complete'].includes(setup.stage))
    || setup.input === 'polar-tilt' && setup.ready && setup.calibration !== 'ready') return null;
  return setup;
}

function validJournal(saved) {
  const command = saved?.command;
  return validOwner(saved?.owner) && command?.v === 1 && /^[a-f0-9]{16}$/.test(command.id)
    && nonnegative(command.g) && command.g > 0 && nonnegative(command.r) && operations.has(command.op)
    && Object.keys(command).length === 5;
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

  constructor({ storage } = {}) {
    this.storage = storage;
    try {
      const saved = JSON.parse(storage?.getItem(journalKey) ?? 'null');
      if (validJournal(saved)) {
        this.pending = saved;
        this.unknown = true;
      }
    } catch { /* Browser storage is optional; in-memory commands remain bound. */ }
  }

  remember(pending) {
    this.pending = pending;
    try {
      if (pending) this.storage?.setItem(journalKey, JSON.stringify(pending));
      else this.storage?.removeItem(journalKey);
    } catch { /* Never replace or resend an in-memory command after storage failure. */ }
  }

  update(gateway, state, participantReady) {
    this.gateway = gateway;
    const { status, busy, pending, unknown } = state;
    let setup = breathingSetup(status, gateway?.owner);
    if (this.cursor && sameQuest(this.cursor.owner, gateway?.owner) && setup) {
      if (setup.g < this.cursor.g || setup.g === this.cursor.g
        && (setup.r < this.cursor.r || setup.position_frames < this.cursor.position)) setup = null;
    }
    if (setup) this.cursor = { owner: { ...gateway.owner }, g: setup.g, r: setup.r, position: setup.position_frames };
    this.setup = setup;
    if (this.pending && (!sameQuest(this.pending.owner, gateway?.owner)
      || setup && setup.g !== this.pending.command.g)) this.unknown = true;
    const available = Boolean(setup && typeof gateway?.breathSetup === 'function');
    const ownerIdle = available
      && status.p === 'IDLE' && status.h === 'vr-preparation' && !status.rec && !status.pending
      && !busy && !pending && !unknown && !this.busy;
    const idle = ownerIdle && !this.pending && !this.unknown;
    const gated = idle && participantReady
      && (status.pi?.r === false || status.pi?.r === true && status.pi.f === true);
    const retired = this.pending && gateway?.owner
      && this.pending.owner.device === gateway.owner.device && this.pending.owner.serial === gateway.owner.serial
      && (this.pending.owner.epoch !== gateway.owner.epoch
        || this.pending.owner.native_epoch !== gateway.owner.native_epoch || setup?.g > this.pending.command.g);
    const testing = setup?.stage === 'testing' && setup.playing;
    const preparing = setup && ['waiting', 'calibrating', 'testing'].includes(setup.stage) && setup.playing;
    const phaseIndex = setup ? phases.indexOf(setup.phase) : -1;
    const phaseFraction = setup ? (setup.position_frames % (setup.cycle_frames / 4)) / (setup.cycle_frames / 4) : 0;
    const radius = [.46 + .54 * phaseFraction, 1, 1 - .54 * phaseFraction, .46][phaseIndex] ?? .46;
    this.view = {
      available, updateRequired: Boolean(status && status.breath_setup !== 1), setup,
      pending: Boolean(this.pending || this.busy), unknown: this.unknown,
      phaseLabel: phaseLabels[phaseIndex] ?? '', radius,
      inputLabel: setup ? inputs[setup.input] : '',
      inputPhaseLabel: setup && inputPhases[setup.input_phase] || '',
      inputVolume: typeof setup?.input_volume01 === 'number' ? setup.input_volume01 : null,
      failure: setup?.calibration === 'failed' ? setup.failure : '',
      progress: setup?.target_frames ? { value: 2 * setup.completed_frames / setup.target_frames, max: 2 } : { value: 0, max: 2 },
      acceptance: setup ? `${gateway.owner.device}/${gateway.owner.epoch}/${gateway.owner.native_epoch}/${setup.g}/${setup.attempt}/${setup.input}` : '',
      message: this.message,
      blockedReason: !participantReady ? 'Prüfe die Teilnehmer-ID und die Polar-H10-Verbindung.'
        : status?.pi?.r === true && status.pi.f !== true ? 'Warte auf aktuelle Polar-H10-Daten von der Quest.' : '',
      actions: {
        start: Boolean(gated && setup.stage === 'idle'),
        calibrate: Boolean(gated && preparing),
        controller: Boolean(idle && preparing && setup.input !== 'controller-state'),
        continue: Boolean(gated && testing && setup.ready && ['polar-tilt', 'controller-state'].includes(setup.input)),
        acknowledge: Boolean(ownerIdle && retired),
      },
    };
    return this.view;
  }

  acknowledgeRetired() {
    if (!this.view?.actions.acknowledge) return false;
    this.remember(null);
    this.unknown = false;
    this.message = 'Die frühere Bestätigung wurde verworfen. Der aktuelle Quest-Status gilt; kein Befehl wird wiederholt.';
    return true;
  }

  observeReceipt(receipt) {
    const saved = this.pending;
    if (!saved || !sameQuest(saved.owner, this.gateway?.owner)) return;
    if (receipt?.v !== 1 || receipt.id !== saved.command.id || !outcomes.has(receipt.state)
      || receipt.state === 'outcome_unknown') {
      this.unknown = true;
      return;
    }
    if (receipt.state === 'rejected') {
      this.remember(null);
      this.unknown = false;
      this.message = 'Die Quest hat die Aktion abgelehnt. Prüfe den aktuellen Status.';
    } else {
      this.unknown = !this.setup || this.setup.g !== saved.command.g;
      if (receipt.state === 'confirmed' && !this.unknown && this.setup.r > saved.command.r) {
        this.remember(null);
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
    try {
      const receipt = await gateway.breathSetup(command, signal);
      if (this.gateway === gateway) this.observeReceipt(receipt);
    } catch { this.unknown = true; }
    finally { this.busy = false; }
  }

  // Read receipts only. A reconnect never retransmits the journaled mutation.
  refresh() {
    const gateway = this.gateway;
    if (!this.pending || !sameQuest(this.pending.owner, gateway?.owner) || this.busy || this.refreshing) return this.refreshing;
    const id = this.pending.command.id;
    this.refreshing = (async () => {
      try {
        const receipt = await gateway.receipt(id);
        if (this.gateway === gateway && this.pending?.command.id === id) this.observeReceipt(receipt);
      } catch { if (this.gateway === gateway) this.unknown = true; }
    })().finally(() => { this.refreshing = null; });
    return this.refreshing;
  }
}
