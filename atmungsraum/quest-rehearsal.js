import { operationId, sameQuest } from './quest-gateway.js?v=5';

const conditions = ['condition-a', 'condition-b'];
const outcomes = new Set(['accepted', 'pending', 'confirmed', 'rejected', 'outcome_unknown']);
const desiredPhase = { arm: 'ARMED', start: 'RUNNING', pause: 'PAUSED', resume: 'RUNNING', 'save-next': 'IDLE' };
const setupOperations = new Set(['scan', 'select', 'connect', 'start-ecg']);
const setupReference = /^[a-f0-9]{32}$/;
const candidateId = /^[a-f0-9]{12}$/;
const scenarios = new Set(['probe-anleitung', 'probe-5', 'probe-10', 'probe-15', 'haupt-15', 'box-20']);

export function canRecenter(status) {
  return status?.recenter === 1 && !status.pending
    && (status.h === 'vr-preparation' && status.p === 'IDLE'
      || status.h === 'vr-session' && ['ARMED', 'RUNNING', 'PAUSED', 'RECORDING'].includes(status.p));
}

export function rehearsalScenarios(status) {
  const offered = status?.scenarios;
  return Array.isArray(offered) && offered.length <= scenarios.size
    && new Set(offered).size === offered.length && offered.every(id => scenarios.has(id)) ? offered : [];
}

export function polarSetup(status) {
  const setup = status?.y;
  if (status?.p !== 'IDLE' || status.rec || !setup || typeof setup.s !== 'string' || !setupReference.test(setup.s)
    || !Number.isSafeInteger(setup.g) || setup.g <= 0 || String(setup.g) !== status.o?.e
    || !Number.isSafeInteger(setup.r) || setup.r < 0
    || !['p', 'b', 'c', 'a', 'e', 'sc'].every(flag => typeof setup[flag] === 'boolean')
    || typeof setup.x !== 'string' || setup.x !== '' && !candidateId.test(setup.x)
    || setup.c && !setup.x || setup.e && !setup.c
    || !Array.isArray(setup.d) || setup.d.length > 4 || new Set(setup.d).size !== setup.d.length
    || !setup.d.every(id => typeof id === 'string' && candidateId.test(id))) return null;
  return setup;
}

function setupConfirmed(status, { op, setupRef, candidate }) {
  const setup = polarSetup(status);
  if (!setup || setup.s !== setupRef || !setup.p || !setup.b || setup.sc) return false;
  if (op === 'scan') return true;
  if (setup.x !== candidate) return false;
  if (op === 'select') return true;
  return setup.c && (op === 'connect' || setup.e);
}

export function rehearsalPolarAllowed(status) {
  return status?.pi?.r === false || status?.pi?.r === true && status.participant_arm === 1;
}

export function rehearsalActions(status, condition = conditions[0]) {
  if (!rehearsalPolarAllowed(status) || status.pending || polarSetup(status)?.sc) return {};
  const selected = conditions.indexOf(condition);
  if (status.p === 'IDLE' && status.h === 'vr-preparation' && !status.rec
    && selected >= 0 && status.a?.[selected] === 'track-ready') return { play: 'arm' };
  if (!status.rec || !conditions.includes(status.c)) return {};
  const next = { ARMED: 'start', RUNNING: 'pause', PAUSED: 'resume' }[status.p];
  const grant = status.q;
  const allowed = status.h === 'vr-session' && next && grant?.v === 2 && grant.g === status.g
    && typeof grant.s === 'string' && /^[a-f0-9]{32}$/.test(grant.s) && Array.isArray(grant.o) && grant.o.includes(next);
  const canPlay = allowed && (next === 'pause' || status.pi.r === false || status.pi.f === true);
  return { play: canPlay ? next : undefined, finish: next || status.p === 'RECORDING' ? 'save-next' : undefined };
}

export class QuestRehearsal {
  gateway = null;
  pending = null;
  refreshing = null;
  state = { status: null, busy: false, pending: false, unknown: false, message: '' };

  constructor({ onChange = () => {}, storage, wait = () => new Promise(resolve => setTimeout(resolve, 350)) } = {}) {
    this.onChange = onChange;
    this.storage = storage;
    this.wait = wait;
    try {
      const saved = JSON.parse(storage?.getItem('atmungsraum-quest-pending') ?? 'null');
      const validOperation = saved && (saved.op === 'recenter' || Object.hasOwn(desiredPhase, saved.op)
        || setupOperations.has(saved.op) && setupReference.test(saved.setupRef)
          && (saved.op === 'scan' ? saved.candidate === '' : candidateId.test(saved.candidate)));
      if (validOperation && /^[a-f0-9]{16}$/.test(saved.id)
        && Number.isSafeInteger(saved.generation) && saved.generation >= 0 && saved.owner) {
        this.pending = saved;
        this.state.unknown = true;
      }
    } catch { /* Browser storage is optional. */ }
  }

  update(change) {
    Object.assign(this.state, change, { pending: Boolean(this.pending) });
    this.onChange(this.state);
  }

  remember(pending) {
    this.pending = pending;
    try {
      if (pending) this.storage?.setItem('atmungsraum-quest-pending', JSON.stringify(pending));
      else this.storage?.removeItem('atmungsraum-quest-pending');
    } catch { /* Keep the original in memory when storage is unavailable. */ }
  }

  attach(gateway) {
    if (this.gateway !== gateway) this.gateway?.close();
    this.gateway = gateway;
    this.update({ status: gateway?.initial ?? null, busy: false, message: '', unknown: Boolean(this.pending) });
  }

  canAcknowledgeIdle() {
    const { busy, unknown, status } = this.state;
    return !busy && unknown && status?.p === 'IDLE' && !status.pending && !status.rec && !polarSetup(status)?.sc
      && this.pending?.owner.device === this.gateway?.owner.device
      && this.pending?.owner.serial === this.gateway?.owner.serial;
  }

  acknowledgeIdle() {
    if (!this.canAcknowledgeIdle()) return;
    this.remember(null);
    this.update({ unknown: false, message: '' });
  }

  async observe(gateway, receipt) {
    const status = await gateway.status();
    if (this.gateway !== gateway) return false;
    this.update({ status });
    if (!this.pending) return true;
    const { id, op, generation, owner } = this.pending;
    if (!sameQuest(owner, gateway.owner)) return false;
    if (op === 'recenter' && receipt?.v === 1 && receipt.id === id && receipt.state === 'outcome_unknown') {
      this.remember(null);
      this.update({ unknown: false, message: 'Die Quest konnte das Zentrieren nicht bestätigen.' });
      return false;
    }
    if (receipt?.v !== 1 || receipt.id !== id || !outcomes.has(receipt.state) || receipt.state === 'outcome_unknown') {
      this.update({ unknown: true });
      return false;
    }
    if (receipt.state === 'rejected') {
      this.remember(null);
      this.update({ unknown: false, message: 'Die Quest hat die Aktion abgelehnt.' });
      return false;
    }
    if (receipt.state !== 'confirmed') return false;
    if (op !== 'recenter' && status.pending) return false;
    if (setupOperations.has(op)) {
      if (status.g !== generation || !setupConfirmed(status, this.pending)) return false;
    } else if (op !== 'recenter') {
      const generationMatches = op === 'arm' ? status.g > generation
        : op === 'save-next' ? !status.rec : status.g === generation;
      if (status.p !== desiredPhase[op] || !generationMatches
        || op !== 'save-next' && status.c !== (this.pending.condition ?? conditions[0])) return false;
    }
    this.remember(null);
    this.update({ unknown: false, message: op === 'recenter' ? 'VR-Szene zentriert.'
      : op === 'save-next' ? 'Probelauf beendet.' : '' });
    return true;
  }

  refresh() {
    const gateway = this.gateway;
    if (!gateway || this.state.busy || this.refreshing) return this.refreshing;
    this.refreshing = (async () => {
      try {
        const receipt = this.pending && sameQuest(this.pending.owner, gateway.owner)
          ? await gateway.receipt(this.pending.id).catch(() => null) : null;
        await this.observe(gateway, receipt);
      } catch {
        if (this.gateway === gateway) this.update({ status: null, unknown: Boolean(this.pending) });
      }
    })().finally(() => { this.refreshing = null; });
    return this.refreshing;
  }

  async dispatch(gateway, op, status, signal, target = '', participantId = '', scenario = '') {
    signal?.throwIfAborted();
    const id = operationId();
    const pending = { id, op, owner: gateway.owner, generation: status.g };
    let command;
    if (setupOperations.has(op)) {
      const { s, g, r } = polarSetup(status);
      Object.assign(pending, { setupRef: s, candidate: target });
      command = { v: 4, id, op, s, g, r, d: op === 'select' ? target : '', n: operationId() };
    } else if (op === 'recenter') {
      command = { v: 1, id, g: status.g };
    } else if (op === 'arm' || op === 'save-next') {
      if (op === 'arm') pending.condition = target;
      command = { v: 1, id, op, condition: op === 'arm' ? target : '', bias: 0, nonce: operationId() };
    } else {
      pending.condition = status.c;
      command = { v: 2, id, op, s: status.q.s, g: status.q.g, n: operationId() };
    }
    this.remember(pending);
    this.update({ message: '' });
    let receipt;
    if (op === 'recenter') {
      receipt = await gateway.recenter(command, signal);
    } else if (op === 'arm' && status.participant_arm === 1) {
      const selected = scenario ? { scenario } : {};
      receipt = await gateway.participantArm({ ...command, ...selected, g: status.g, participant_id: participantId }, signal);
    } else if (op === 'arm' && scenario) {
      receipt = await gateway.scenarioArm({ ...command, g: status.g, scenario }, signal);
    } else {
      receipt = await gateway.command(command, signal);
    }
    return this.observe(gateway, receipt);
  }

  async dispatchAndConfirm(gateway, op, signal, timeout, target, participantId, scenario) {
    let confirmed = await this.dispatch(gateway, op, this.state.status, signal, target, participantId, scenario);
    const deadline = performance.now() + timeout;
    while (!confirmed && this.pending && !this.state.unknown && this.gateway === gateway
      && !signal?.aborted && performance.now() < deadline) {
      await this.wait();
      if (this.gateway !== gateway || signal?.aborted || performance.now() >= deadline) break;
      confirmed = await this.observe(gateway, await gateway.receipt(this.pending.id));
    }
    if (!confirmed && this.gateway === gateway && this.pending) this.update({ unknown: true });
    return confirmed && this.gateway === gateway && !signal?.aborted;
  }

  async connectPolar(gateway, clicked, signal) {
    const original = polarSetup(clicked);
    const current = () => {
      const { status } = this.state;
      const setup = polarSetup(status);
      return this.gateway === gateway && !signal?.aborted && !status?.pending && status?.g === clicked.g
        && setup?.s === original.s && setup.p && setup.b && !setup.sc ? setup : null;
    };
    let setup = current();
    if (!setup || setup.e) return;
    if (!setup.c) {
      if (!await this.dispatchAndConfirm(gateway, 'scan', signal, 62000)) return;
      setup = current();
      if (!setup) return;
      if (setup.d.length !== 1) {
        this.update({ message: setup.d.length === 0
          ? 'Kein Polar H10 gefunden. Lege den Gurt an und suche erneut.'
          : 'Mehrere Gurte gefunden. Lass nur den gewünschten Gurt aktiv und suche erneut.' });
        return;
      }
      const candidate = setup.d[0];
      if (!await this.dispatchAndConfirm(gateway, 'select', signal, 32000, candidate) || !current()) return;
      if (!await this.dispatchAndConfirm(gateway, 'connect', signal, 32000, candidate)) return;
      setup = current();
    }
    if (setup?.c && !setup.e) await this.dispatchAndConfirm(gateway, 'start-ecg', signal, 32000, setup.x);
  }

  async run(action, signal, condition = conditions[0], participantId = '', scenario = '') {
    const gateway = this.gateway;
    const clicked = this.state.status;
    const requested = action === 'recenter' && gateway?.recenter && canRecenter(clicked)
      ? 'recenter' : rehearsalActions(clicked, condition)[action];
    const setup = action === 'polar' ? polarSetup(clicked) : null;
    if (!gateway || clicked?.pending || this.state.busy || this.pending
      || (!requested && (!setup?.p || !setup.b || setup.sc || setup.e))) return;
    this.update({ busy: true, message: '' });
    try {
      await this.refreshing;
      if (this.gateway !== gateway || signal?.aborted) return;
      if (!await this.observe(gateway) || signal?.aborted) return;
      if (action === 'polar') {
        await this.connectPolar(gateway, clicked, signal);
        return;
      }
      const status = this.state.status;
      if (requested === 'recenter') {
        if (status.g === clicked.g && canRecenter(status)) {
          await this.dispatchAndConfirm(gateway, 'recenter', signal, 5000);
        }
        return;
      }
      const op = rehearsalActions(status, condition)[action];
      if (op !== requested || status.g !== clicked.g
        || op !== 'arm' && op !== 'save-next' && status.q?.s !== clicked.q?.s) return;
      if (op !== 'arm') {
        await this.dispatch(gateway, op, status, signal);
        return;
      }
      if (status.participant_arm === 1 && (!participantId || !gateway.participantArm || !polarSetup(status)?.e)) return;
      if (scenario && (!gateway.scenarioArm || !rehearsalScenarios(status).includes(scenario))) return;
      // Only a confirmed Arm in the same foreground action may immediately lead to Start.
      // Polling and reconnect only read the original receipt.
      if (await this.dispatchAndConfirm(gateway, 'arm', signal, 10000, condition, participantId, scenario)
        && rehearsalActions(this.state.status).play === 'start') {
        await this.dispatch(gateway, 'start', this.state.status, signal);
      }
    } catch {
      if (this.gateway === gateway) this.update({ status: null, unknown: Boolean(this.pending) });
    } finally {
      if (this.gateway === gateway) this.update({ busy: false });
    }
  }
}
