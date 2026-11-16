import { controllerAccessState, operationId, sameQuest } from './quest-gateway.js?v=13';
import { nativeMetaUpdate } from './meta-update.js?v=2';

const key = 'atmungsraum-meta-update-pending';
const lineage = v => typeof v === 'string' && /^[1-9][0-9]{0,18}$/.test(v);
const id = v => typeof v === 'string' && /^[a-f0-9]{16}$/.test(v);
const reason = v => typeof v === 'string' && v.length <= 96 && !/[^\x20-\x7e]/.test(v);
const ownerValid = o => o && /^[a-f0-9]{32}$/.test(o.device) && typeof o.serial === 'string'
  && o.serial.length <= 64 && !/[\x00-\x1f\x7f]/.test(o.serial) && lineage(o.epoch) && lineage(o.native_epoch);
const commandValid = c => c?.v === 1 && id(c.id) && c.op === 'check-and-update'
  && lineage(c.cycle) && lineage(c.revision) && Object.keys(c).length === 5;
const receiptValid = r => r?.v === 1 && id(r.id) && ['accepted', 'busy', 'stale', 'unsupported', 'rejected'].includes(r.state)
  && lineage(r.cycle) && lineage(r.revision) && reason(r.reason)
  && (r.state === 'accepted' ? lineage(r.attempt) : r.attempt === null || lineage(r.attempt));

const guidance = {
  'preparation-active': 'Die Atemvorbereitung läuft. Updates sind erst nach einer sicher beendeten Sitzung möglich.',
  'session-active': 'Die Sitzung läuft. Beende und speichere sie, bevor du ein Update prüfst.',
  'writer-active': 'Die Quest verarbeitet eine Aufnahme. Warte auf den bestätigten Abschluss.',
  'finalizing': 'Die Quest speichert die Sitzung. Warte auf den bestätigten Abschluss.',
  'install-uncertain': 'Der Installationsstand ist unklar. Prüfe die Meta-Bibliothek im Headset.',
};

export class MetaUpdateControl {
  pending = null;
  busy = false;
  refreshing = null;
  message = '';
  expired = false;
  budget = null;

  constructor({ storage, now = () => performance.now() } = {}) {
    this.storage = storage; this.now = now;
    try {
      const saved = JSON.parse(storage?.getItem(key) ?? 'null');
      if (ownerValid(saved?.owner) && commandValid(saved.command)
        && (!saved.accepted || receiptValid(saved.accepted) && saved.accepted.state === 'accepted'
          && saved.accepted.id === saved.command.id && saved.accepted.cycle === saved.command.cycle)) this.pending = saved;
    } catch { /* In-memory lineage remains usable without storage. */ }
  }

  remember(value) {
    this.pending = value; this.budget = null; this.expired = false;
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
      && value.attempt === this.pending.accepted.attempt && value.cycle === this.pending.accepted.cycle
      && BigInt(value.revision) >= BigInt(this.pending.accepted.revision)) {
      this.remember(null); this.message = 'Die Quest hat die Update-Prüfung übernommen. Der aktuelle Quest-Status zeigt den Fortschritt.';
    }
    const idle = active && state.status.p === 'IDLE' && !state.status.rec && !state.status.pending
      && !state.busy && !state.pending && !state.unknown;
    const retired = this.pending && gateway?.owner.device === this.pending.owner.device
      && gateway.owner.serial === this.pending.owner.serial
      && (!sameQuest(this.pending.owner, gateway.owner) || available && value.cycle !== this.pending.command.cycle);
    this.view = {
      available, value: available ? value : null,
      canRequest: Boolean(idle && value.can_check && !this.pending && !this.busy),
      canReadReceipt: Boolean(active && this.pending && sameQuest(this.pending.owner, gateway.owner) && !this.busy && !this.refreshing),
      canAcknowledge: Boolean(active && retired && !this.busy),
      prerequisites: !available ? '' : !active ? 'Übernimm zuerst die Quest-Steuerung auf diesem Smartphone.'
        : !value.can_check ? guidance[value.busy_reason] || 'Die Quest gibt eine neue Update-Prüfung gerade nicht frei. Warte auf einen sicheren Abschluss.'
        : 'Die Quest prüft die erforderliche Version und den Meta-Kanal. Eine nötige Aktualisierung beendet die App vorübergehend und startet sie danach neu.',
      message: this.pending ? this.expired
        ? 'Die Bestätigung ist noch offen. Frage das Ergebnis der ursprünglichen Anfrage erneut ab. Die Aktualisierung wird nicht erneut angefragt.'
        : 'Warte auf die Bestätigung der Quest. Die ursprüngliche Update-Anfrage bleibt gespeichert.' : this.message,
    };
    return this.view;
  }

  observe(receipt, gateway) {
    if (!this.pending || this.gateway !== gateway || !sameQuest(this.pending.owner, gateway.owner)
      || !receiptValid(receipt) || receipt.id !== this.pending.command.id) return;
    if (receipt.state === 'accepted') {
      if (receipt.cycle !== this.pending.command.cycle
        || BigInt(receipt.revision) <= BigInt(this.pending.command.revision)) return;
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
    const command = { v: 1, id: operationId(), op: 'check-and-update', cycle: this.view.value.cycle, revision: this.view.value.revision };
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
