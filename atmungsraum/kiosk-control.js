import { canControlQuest, operationId, sameQuest } from './quest-gateway.js?v=12';

const journalKey = 'atmungsraum-kiosk-pending';
const code = value => typeof value === 'string' && /^[a-z][a-z-]{0,47}$/.test(value);
const boundedText = value => typeof value === 'string' && value.length <= 240 && !/[\x00-\x1f\x7f]/.test(value);
const integer = value => Number.isSafeInteger(value) && value >= 0;
const id = value => typeof value === 'string' && /^[a-f0-9]{16}$/.test(value);
export function kioskStatus(status) {
  const value = status?.kiosk;
  return value?.v === 1 && integer(value.g) && ['requested', 'effective', 'retry_available'].every(key => typeof value[key] === 'boolean')
    && code(value.state) && code(value.service) && (value.blocked === '' || code(value.blocked)) && boundedText(value.detail) ? value : null;
}
const validJournal = saved => saved?.owner && /^[a-f0-9]{32}$/.test(saved.owner.device)
  && typeof saved.owner.serial === 'string' && saved.owner.serial.length <= 64
  && ['epoch', 'native_epoch'].every(key => /^[1-9][0-9]{0,18}$/.test(saved.owner[key]))
  && saved.command?.v === 1 && id(saved.command.id) && saved.command.op === 'retry' && integer(saved.command.g)
  && Object.keys(saved.command).length === 4;

export class KioskControl {
  pending = null;
  busy = false;
  refreshing = null;
  expired = false;
  message = '';
  budget = null;

  constructor({ storage, now = () => performance.now() } = {}) {
    this.storage = storage; this.now = now;
    try {
      const saved = JSON.parse(storage?.getItem(journalKey) ?? 'null');
      if (validJournal(saved)) this.pending = saved;
    } catch { /* Keep current-session ownership when storage is unavailable. */ }
  }

  remember(value) {
    this.pending = value; this.budget = null; this.expired = false;
    try {
      if (value) this.storage?.setItem(journalKey, JSON.stringify(value));
      else this.storage?.removeItem(journalKey);
    } catch { /* A storage error must never resend an in-memory operation. */ }
  }

  update(gateway, state) {
    this.gateway = gateway; this.state = state;
    const status = state.status;
    const value = kioskStatus(status);
    const observed = status?.o;
    const available = Boolean(value && observed?.v === 1 && observed.s === 'observed' && observed.e === gateway?.owner.native_epoch
      && integer(observed.a) && observed.a <= 1500);
    const active = available && canControlQuest(status);
    const idle = active && status.p === 'IDLE' && !status.rec && !status.pending && !state.busy && !state.pending && !state.unknown;
    const retired = this.pending && this.pending.owner.device === gateway?.owner.device
      && this.pending.owner.serial === gateway?.owner.serial && !sameQuest(this.pending.owner, gateway?.owner);
    this.view = {
      available, value: available ? value : null,
      summary: !available ? 'Dieser Quest-Build meldet keinen aktuellen Kiosk-Status.'
        : `Kiosk: ${value.requested ? 'angefordert' : 'nicht angefordert'} · ${value.effective ? 'wirksam' : 'nicht wirksam'}.`,
      prerequisites: !available ? 'Aktualisiere die Quest-App für die Kiosk-Diagnose.'
        : value.effective ? 'Die Quest bestätigt den wirksamen Kiosk-Schutz.'
          : value.requested ? 'Kiosk ist angefordert. Prüfe im Headset die Android-Bedienungshilfe und bestätige nötige Systemdialoge. Die Quest meldet die Wirksamkeit separat.'
            : 'Kiosk muss im Bedienfeld der Quest eingerichtet und angefordert werden. Dieses Smartphone kann Systemfreigaben nicht erteilen.',
      canRetry: Boolean(idle && gateway?.kiosk && value.retry_available && !this.pending && !this.busy),
      canCheck: Boolean(active && gateway?.kiosk && this.pending && sameQuest(this.pending.owner, gateway.owner) && !this.busy && !this.refreshing),
      canAcknowledge: Boolean(idle && retired && !this.busy),
      message: this.pending ? this.expired
        ? 'Die Wartezeit ist abgelaufen. Frage die Bestätigung erneut ab. Der Kiosk-Dienst wird nicht erneut gestartet. Bleibt das Ergebnis unklar, öffne die Quest-App im Headset erneut und verbinde dieses Smartphone wieder.'
        : 'Warte auf die Bestätigung, dass der Kiosk-Dienst wieder verbunden ist. Die Anfrage wird nicht erneut gesendet.' : this.message,
    };
    return this.view;
  }

  observe(receipt, gateway) {
    if (!this.pending || this.gateway !== gateway || !sameQuest(this.pending.owner, gateway.owner)
      || receipt?.id !== this.pending.command.id || !boundedText(receipt.detail)
      || !['pending', 'confirmed', 'rejected', 'outcome_unknown'].includes(receipt.state)) return;
    if (receipt.state === 'confirmed' || receipt.state === 'rejected') {
      this.message = receipt.state === 'confirmed' ? 'Kiosk-Dienst wieder verbunden. Ob der Schutz wirksam ist, zeigt der aktuelle Quest-Status.'
        : 'Die Quest hat den Neustart des Kiosk-Dienstes abgelehnt oder abgebrochen. Prüfe den aktuellen Status und die Systemfreigaben im Headset.';
      this.remember(null);
    }
  }

  async retry(signal) {
    if (!this.view?.canRetry || this.busy || this.pending) return;
    signal?.throwIfAborted();
    const gateway = this.gateway;
    const command = { v: 1, id: operationId(), op: 'retry', g: this.view.value.g };
    this.remember({ owner: { ...gateway.owner }, command }); this.busy = true; this.message = '';
    try { this.observe(await gateway.kiosk(command, signal), gateway); }
    catch { /* Keep the exact ID for read-only reconciliation. */ }
    finally { this.busy = false; }
  }

  refresh(signal) {
    const gateway = this.gateway;
    if (!this.pending || !this.view?.canCheck || this.busy || this.refreshing || signal?.aborted) return this.refreshing;
    const now = this.now();
    this.budget ??= { started: now, last: -Infinity, reads: 0 };
    if (now - this.budget.started >= 12000 || this.budget.reads >= 12) { this.expired = true; return; }
    if (now - this.budget.last < 500) return;
    this.budget.last = now; this.budget.reads++;
    const operation = this.pending.command.id;
    this.refreshing = (async () => {
      try { this.observe(await gateway.kiosk({ v: 1, id: operation, op: 'receipt' }, signal), gateway); }
      catch { /* Only an exact native terminal receipt can retire this request. */ }
    })().finally(() => { this.refreshing = null; });
    return this.refreshing;
  }

  recheck() { if (this.view?.canCheck) { this.budget = null; this.expired = false; } }
  acknowledge() {
    if (!this.view?.canAcknowledge) return;
    this.remember(null); this.message = 'Aktueller Quest-Zustand übernommen. Kein Kiosk-Befehl wurde wiederholt.';
  }
}
