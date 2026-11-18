import { canControlQuest, operationId, sameQuest } from './quest-gateway.js?v=20';

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

function kioskGuidance(value) {
  if (!value) return 'Aktualisiere die Quest-App für die Kiosk-Diagnose.';
  if (value.effective) return 'Die Quest bestätigt den wirksamen Kiosk-Schutz.';
  const blocker = value.blocked || value.state;
  if (blocker === 'terminal' || value.state === 'terminal') return 'Die Quest hat den Kiosk-Schutz beim Beenden der App beendet. Öffne die Atmungsraum-App über die App-Bibliothek der Quest, wenn du einen neuen Durchlauf starten möchtest.';
  if (!value.requested) return 'Kiosk ist nicht angefordert. Öffne die Atmungsraum-App regulär über die App-Bibliothek der Quest. Dieses Smartphone kann den Kiosk-Schutz nicht aktivieren.';
  if (blocker === 'recovery-exhausted') return 'Die automatischen Rückkehrversuche sind ausgeschöpft. Beende die Atmungsraum-App im Headset und öffne sie erneut über die App-Bibliothek der Quest. Verbinde danach dieses Smartphone wieder.';
  if (blocker === 'system-prompt-or-transition') return 'Ein Systemdialog oder App-Wechsel unterbricht den Schutz vorübergehend. Bearbeite den sichtbaren Dialog im Headset und kehre zur VR-App zurück. Warte danach auf die Bestätigung der Quest.';
  if (blocker === 'presentation-stale' || value.state === 'watchdog-awaiting-presentation') return 'Die Quest bestätigt noch keine aktuelle Ansicht der VR-App. Setze das Headset auf und kehre zur Atmungsraum-App zurück. Warte, bis die VR-Szene oder ihr Bedienfeld wieder sichtbar ist.';
  if (value.state === 'needs-accessibility-setup') return 'Dieser Quest-Modus meldet eine fehlende Bedienungshilfe. Richte die von der Quest-App angeforderte Android-Bedienungshilfe im Headset ein. Warte danach auf den bestätigten Kiosk-Status.';
  if (value.state === 'watchdog-starting' || value.service === 'start-pending') return 'Der Kiosk-Dienst wird auf der Quest gestartet. Bitte warten. Erst ein verbundener Dienst und die sichtbare VR-App machen den Schutz wirksam.';
  if (value.state === 'revoked' || value.service === 'revoked') return 'Die Quest hat den Kiosk-Dienst entzogen. Kehre zur Atmungsraum-App zurück und prüfe die dortige Meldung im Headset. Der Schutz ist noch nicht wirksam.';
  if (['watchdog-failed', 'watchdog-missing', 'interrupted'].includes(value.state)
    || ['start-failed', 'missing-or-disabled', 'interrupted'].includes(value.service)) {
    const reason = value.state === 'watchdog-failed' || value.service === 'start-failed' ? 'Der Kiosk-Dienst konnte nicht starten.'
      : value.state === 'interrupted' || value.service === 'interrupted' ? 'Die Verbindung zum Kiosk-Dienst wurde unterbrochen.' : 'Der Kiosk-Dienst ist nicht verbunden.';
    return `${reason} ${value.retry_available ? 'Wenn dieses Smartphone steuert und keine Anfrage offen ist, wähle „Kiosk-Dienst erneut verbinden“.'
      : 'Kehre zur Atmungsraum-App im Headset zurück und prüfe den Status erneut. Die Quest bietet gerade keinen Dienst-Neustart an.'}`;
  }
  if (value.state === 'degraded-home-resolving') return 'Die Quest bestimmt noch ihre Home-Oberfläche. Bitte warten, bis die Quest den Kiosk-Status bestätigt.';
  if (value.state === 'unavailable-home-surface') return 'Die Home-Oberfläche der Quest ist nicht verfügbar. Prüfe die Meldung im Bedienfeld der App im Headset. Der Schutz ist noch nicht wirksam.';
  return 'Die Quest meldet eine Kiosk-Unterbrechung. Prüfe die Meldung im Bedienfeld der Quest. Die Wirksamkeit ist noch nicht bestätigt.';
}

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
      prerequisites: kioskGuidance(available ? value : null),
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
        : 'Die Quest hat den Neustart des Kiosk-Dienstes abgelehnt oder abgebrochen. Prüfe die angezeigte Kiosk-Meldung der Quest.';
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
