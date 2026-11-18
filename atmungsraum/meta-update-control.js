import { controllerAccessState, operationId, sameQuest } from './quest-gateway.js?v=18';
import { nativeMetaUpdate } from './meta-update.js?v=6';

const key = 'atmungsraum-meta-update-pending';
const lineage = v => typeof v === 'string' && /^[1-9][0-9]{0,18}$/.test(v);
const id = v => typeof v === 'string' && /^[a-f0-9]{16}$/.test(v);
const reason = v => typeof v === 'string' && v.length <= 96 && !/[^\x20-\x7e]/.test(v);
const ownerValid = o => o && /^[a-f0-9]{32}$/.test(o.device) && typeof o.serial === 'string'
  && o.serial.length <= 64 && !/[\x00-\x1f\x7f]/.test(o.serial) && lineage(o.epoch) && lineage(o.native_epoch);
const commandValid = c => c?.v === 1 && id(c.id) && ['check-and-update', 'end-preparation-and-check'].includes(c.op)
  && lineage(c.cycle) && lineage(c.revision) && Object.keys(c).length === 5;
const receiptValid = r => r?.v === 1 && id(r.id) && ['accepted', 'busy', 'stale', 'unsupported', 'rejected'].includes(r.state)
  && lineage(r.cycle) && lineage(r.revision) && reason(r.reason)
  && (r.state === 'accepted' ? lineage(r.attempt) : r.attempt === null || lineage(r.attempt));

const guidance = {
  'preparation-active': 'Diese Quest-App kann die Vorbereitung nicht für das Update beenden. Aktualisiere Atmungsraum in der Meta-Bibliothek im Headset.',
  'session-active': 'Die Sitzung läuft. Beende und speichere sie, bevor du ein Update prüfst.',
  'writer-active': 'Die Quest verarbeitet eine Aufnahme. Warte auf den bestätigten Abschluss.',
  'finalizing': 'Die Quest speichert die Sitzung. Warte auf den bestätigten Abschluss.',
  'setup-park-pending': 'Die Quest hält den Atemtakt an. Warte auf die bestätigte Ruheposition; danach beginnt die Update-Prüfung.',
  'install-uncertain': 'Der Installationsstand ist unklar. Prüfe die Meta-Bibliothek im Headset.',
};
const successor = value => {
  if (!lineage(value)) return null;
  const next = String(BigInt(value) + 1n);
  return lineage(next) ? next : null;
};
const targetCycle = command => command.op === 'end-preparation-and-check'
  ? successor(command.cycle) : command.cycle;
const acceptanceValid = (command, receipt) => receiptValid(receipt) && receipt.state === 'accepted'
  && receipt.id === command.id && receipt.cycle === targetCycle(command)
  && (command.op === 'end-preparation-and-check' ? receipt.revision === successor(command.revision)
    : BigInt(receipt.revision) > BigInt(command.revision));

export class MetaUpdateControl {
  pending = null;
  busy = false;
  refreshing = null;
  message = '';
  expired = false;
  budget = null;
  receiptConflict = false;

  constructor({ storage, now = () => performance.now() } = {}) {
    this.storage = storage; this.now = now;
    try {
      const saved = JSON.parse(storage?.getItem(key) ?? 'null');
      if (ownerValid(saved?.owner) && commandValid(saved.command)) {
        this.pending = { owner: saved.owner, command: saved.command };
        if (saved.accepted && acceptanceValid(saved.command, saved.accepted)) this.pending.accepted = saved.accepted;
        else if (saved.accepted) this.receiptConflict = true;
      }
    } catch { /* In-memory lineage remains usable without storage. */ }
  }

  remember(value) {
    this.pending = value; this.budget = null; this.expired = false; this.receiptConflict = false;
    try { if (value) this.storage?.setItem(key, JSON.stringify(value)); else this.storage?.removeItem(key); }
    catch { /* A storage failure never resends an operation. */ }
  }

  update(gateway, state) {
    this.gateway = gateway; this.state = state;
    const parsed = nativeMetaUpdate(state.status), value = parsed.value;
    const available = Boolean(value?.v === 2 && state.status.meta_update_control === 1
      && state.status.o.e === gateway?.owner.native_epoch && gateway.metaUpdate);
    let active = false;
    try { active = available && controllerAccessState(state.status)?.role === 'active'; } catch { /* Refuse malformed roles. */ }
    if (available && this.pending?.accepted && sameQuest(this.pending.owner, gateway.owner)
      && value.attempt === this.pending.accepted.attempt && value.cycle === targetCycle(this.pending.command)
      && BigInt(value.revision) >= BigInt(this.pending.accepted.revision)) {
      this.remember(null); this.message = '';
    }
    const canEndPreparation = state.status?.meta_update_preparation === 1 && value?.can_end_preparation === true;
    const requestOp = canEndPreparation && !value?.can_check ? 'end-preparation-and-check' : 'check-and-update';
    // Native pending covers both experiment commands and the optional calibration introduction.
    // Only the explicitly admitted preparation-end operation can retire that introduction.
    const endingInstruction = requestOp === 'end-preparation-and-check' && state.status?.h === 'vr-preparation'
      && id(state.calibrationInstruction) && state.status.setup?.instruction_id === state.calibrationInstruction;
    const pendingOnQuest = state.status?.pending && !endingInstruction;
    const idle = active && state.status.p === 'IDLE' && !state.status.rec && !pendingOnQuest
      && !state.busy && !state.pending && !state.unknown;
    const boundedRequest = requestOp !== 'end-preparation-and-check' || successor(value?.cycle) && successor(value?.revision);
    const retired = this.pending && gateway?.owner.device === this.pending.owner.device
      && gateway.owner.serial === this.pending.owner.serial
      && (!sameQuest(this.pending.owner, gateway.owner) || available
        && value.cycle !== this.pending.command.cycle && value.cycle !== targetCycle(this.pending.command));
    this.view = {
      available, value: available ? value : null,
      canRequest: Boolean(idle && boundedRequest && (value.can_check || canEndPreparation) && !this.pending && !this.busy),
      requestOp,
      canReadReceipt: Boolean(active && this.pending && sameQuest(this.pending.owner, gateway.owner) && !this.busy && !this.refreshing),
      canAcknowledge: Boolean(active && retired && !this.busy),
      prerequisites: !available ? '' : !active ? 'Übernimm zuerst die Quest-Steuerung auf diesem Smartphone.'
        : state.unknown ? 'Eine frühere Sitzungsanfrage ist unklar. Frage ihre ursprüngliche Bestätigung ab, bevor du aktualisierst.'
        : state.pending || pendingOnQuest ? 'Eine Sitzungsanfrage wartet auf Bestätigung. Warte auf das ursprüngliche Ergebnis.'
        : state.busy ? 'Eine Quest-Aktion wird bestätigt. Warte auf ihren Abschluss.'
        : !boundedRequest ? 'Der Quest-Stand kann nicht sicher weitergeführt werden. Verbinde das Smartphone erneut und prüfe den aktuellen Stand.'
        : canEndPreparation && !value.can_check ? 'Die Vorbereitung wird beendet. Erst wenn die Quest den Atemtakt sicher angehalten hat, beginnt die Update-Prüfung.'
        : !value.can_check ? (state.status.p === 'IDLE' && !state.status.rec && ['session-active', 'preparation-active'].includes(value.busy_reason)
          ? state.status.meta_update_preparation === 1
            ? 'Die Vorbereitung kann gerade nicht sicher beendet werden. Warte auf die Bestätigung der Quest.'
            : guidance['preparation-active'] : guidance[value.busy_reason]) || 'Die Quest gibt eine neue Update-Prüfung gerade nicht frei. Warte auf einen sicheren Abschluss.'
        : 'Mit „Updates prüfen und installieren“ startest du eine neue Prüfung. Eine nötige Installation kann die Quest-App vorübergehend schließen.',
      message: this.pending ? this.receiptConflict
        ? 'Die Quest meldet widersprüchliche Bestätigungen. Die ursprüngliche Update-Anfrage bleibt gespeichert. Frage ihren Stand erneut ab.'
        : this.expired
        ? 'Die Bestätigung ist noch offen. Frage das Ergebnis der ursprünglichen Anfrage erneut ab. Die Aktualisierung wird nicht erneut angefragt.'
        : 'Warte auf die Bestätigung der Quest. Die ursprüngliche Update-Anfrage bleibt gespeichert.' : this.message,
    };
    return this.view;
  }

  observe(receipt, gateway) {
    if (!this.pending || this.gateway !== gateway || !sameQuest(this.pending.owner, gateway.owner)
      || !receiptValid(receipt) || receipt.id !== this.pending.command.id) return;
    if (this.pending.accepted) {
      if (['v', 'id', 'state', 'cycle', 'revision', 'attempt', 'reason']
        .some(field => receipt[field] !== this.pending.accepted[field])) this.receiptConflict = true;
      return;
    }
    if (receipt.state === 'accepted') {
      if (!acceptanceValid(this.pending.command, receipt)) return;
      if (!this.pending.accepted || JSON.stringify(this.pending.accepted) !== JSON.stringify(receipt)) {
        this.remember({ ...this.pending, accepted: receipt });
      }
    }
    else {
      this.remember(null);
      this.message = receipt.state === 'busy' ? 'Die Quest ist beschäftigt und hat keine neue Update-Prüfung gestartet.'
        : receipt.state === 'stale' ? 'Der Quest-Stand hat sich geändert. Warte auf die aktuelle Meldung und prüfe erneut.'
          : 'Die Quest hat die Update-Anfrage abgelehnt. Prüfe ihren aktuellen Status.';
    }
  }

  async request(signal) {
    if (!this.view?.canRequest || this.pending || this.busy) return;
    signal?.throwIfAborted();
    const gateway = this.gateway;
    const command = { v: 1, id: operationId(), op: this.view.requestOp, cycle: this.view.value.cycle, revision: this.view.value.revision };
    this.remember({ owner: { ...gateway.owner }, command }); this.busy = true; this.message = '';
    try { this.observe(await gateway.metaUpdate(command, signal), gateway); }
    catch { /* Resolve the original ID, never submit a replacement mutation. */ }
    finally { this.busy = false; }
  }

  refresh(signal) {
    if (!this.pending || !this.view?.canReadReceipt || this.busy || this.refreshing || signal?.aborted) return this.refreshing;
    const now = this.now(); this.budget ??= { started: now, last: -Infinity, reads: 0 };
    if (now - this.budget.started >= 12000 || this.budget.reads >= 12) { this.expired = true; return; }
    if (now - this.budget.last < 500) return;
    this.budget.last = now; this.budget.reads++;
    const gateway = this.gateway, original = this.pending.command.id;
    this.refreshing = (async () => {
      try { this.observe(await gateway.metaUpdate({ v: 1, id: original, op: 'receipt' }, signal), gateway); }
      catch { /* Read-only reconciliation can be retried without another install request. */ }
      finally { this.refreshing = null; }
    })();
    return this.refreshing;
  }

  recheck() { this.budget = null; this.expired = false; }
  acknowledge() { if (this.view?.canAcknowledge) { this.remember(null); this.message = 'Der aktuelle native Update-Stand gilt. Die frühere Telefonanfrage wurde lokal abgelegt.'; } }
}
