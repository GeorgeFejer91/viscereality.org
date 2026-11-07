import { operationId, sameQuest } from './quest-gateway.js?v=1';

const condition = 'condition-a';
const outcomes = new Set(['accepted', 'pending', 'confirmed', 'rejected', 'outcome_unknown']);
const desiredPhase = { arm: 'ARMED', start: 'RUNNING', pause: 'PAUSED', resume: 'RUNNING', 'save-next': 'IDLE' };

export function rehearsalActions(status) {
  if (!status || status.pending || status.pi?.r !== false) return {};
  if (status.p === 'IDLE' && status.h === 'vr-preparation' && !status.rec && status.a?.[0] === 'track-ready') return { play: 'arm' };
  if (!status.rec || status.c !== condition) return {};
  const next = { ARMED: 'start', RUNNING: 'pause', PAUSED: 'resume' }[status.p];
  const grant = status.q;
  const allowed = status.h === 'vr-session' && next && grant?.v === 2 && grant.g === status.g
    && typeof grant.s === 'string' && /^[a-f0-9]{32}$/.test(grant.s) && Array.isArray(grant.o) && grant.o.includes(next);
  return { play: allowed ? next : undefined, finish: next || status.p === 'RECORDING' ? 'save-next' : undefined };
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
      if (saved && /^[a-f0-9]{16}$/.test(saved.id) && Object.hasOwn(desiredPhase, saved.op)
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
    return !busy && unknown && status?.p === 'IDLE' && !status.pending && !status.rec
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
    const generationMatches = op === 'arm' ? status.g > generation
      : op === 'save-next' ? !status.rec : status.g === generation;
    if (status.pending || status.p !== desiredPhase[op] || !generationMatches
      || op !== 'save-next' && status.c !== condition) return false;
    this.remember(null);
    this.update({ unknown: false, message: op === 'save-next' ? 'Probelauf beendet.' : '' });
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

  async dispatch(gateway, op, status, signal) {
    signal?.throwIfAborted();
    const id = operationId();
    const command = op === 'arm' || op === 'save-next'
      ? { v: 1, id, op, condition: op === 'arm' ? condition : '', bias: 0, nonce: operationId() }
      : { v: 2, id, op, s: status.q.s, g: status.q.g, n: operationId() };
    this.remember({ id, op, owner: gateway.owner, generation: status.g });
    this.update({ pending: true, message: '' });
    return this.observe(gateway, await gateway.command(command, signal));
  }

  async run(action, signal) {
    const gateway = this.gateway;
    if (!gateway || this.state.busy || this.pending || !rehearsalActions(this.state.status)[action]) return;
    this.update({ busy: true, message: '' });
    try {
      const clicked = this.state.status;
      const requested = rehearsalActions(clicked)[action];
      await this.refreshing;
      if (this.gateway !== gateway || signal?.aborted) return;
      if (!await this.observe(gateway) || signal?.aborted) return;
      const status = this.state.status;
      const op = rehearsalActions(status)[action];
      if (op !== requested || status.g !== clicked.g
        || op !== 'arm' && op !== 'save-next' && status.q?.s !== clicked.q?.s) return;
      let confirmed = await this.dispatch(gateway, op, status, signal);
      if (op !== 'arm') return;
      const deadline = performance.now() + 10000;
      while (!confirmed && this.pending && !this.state.unknown && this.gateway === gateway
        && !signal?.aborted && performance.now() < deadline) {
        await this.wait();
        if (this.gateway !== gateway || signal?.aborted || performance.now() >= deadline) break;
        confirmed = await this.observe(gateway, await gateway.receipt(this.pending.id));
      }
      // Only a confirmed Arm in the same foreground action may immediately lead to Start.
      // Polling and reconnect only read the original receipt.
      if (confirmed && this.gateway === gateway && !signal?.aborted
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
