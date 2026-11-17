import { operationId, sameQuest, canControlQuest, controllerAccessState } from './quest-gateway.js?v=14';
import { nativeSessionAllowed } from './meta-update.js?v=4';

const conditions = ['condition-a', 'condition-b'];
const outcomes = new Set(['accepted', 'pending', 'confirmed', 'rejected', 'outcome_unknown']);
const desiredPhase = { arm: 'ARMED', start: 'RUNNING', pause: 'PAUSED', resume: 'RUNNING', 'save-next': 'IDLE' };
const setupOperations = new Set(['scan', 'select', 'connect', 'start-ecg', 'start-acc']);
const setupReference = /^[a-f0-9]{32}$/;
const candidateId = /^[a-f0-9]{12}$/;
const scenarios = new Set(['probe-anleitung', 'probe-5', 'probe-10', 'probe-15', 'haupt-15', 'box-20']);
const audioRecovering = status => status?.setup?.stage === 'recovering';
const automationStates = new Set(['scanning', 'backoff', 'connecting', 'streams', 'ready', 'blocked', 'stopped', 'selection-required']);
const validAutomation = value => value?.v === 1 && automationStates.has(value.state)
  && Number.isSafeInteger(value.retry_ms) && value.retry_ms >= 0 && value.retry_ms <= 10000
  && typeof value.reason === 'string' && value.reason.length <= 240 && !/[\x00-\x1f\x7f]/.test(value.reason);

export function canRecenter(status) {
  return canControlQuest(status) && status?.recenter === 1 && !status.pending
    && (status.h === 'vr-preparation' && status.p === 'IDLE'
      || status.h === 'vr-session' && ['ARMED', 'RUNNING', 'PAUSED', 'RECORDING'].includes(status.p));
}

export function rehearsalScenarios(status) {
  const offered = status?.scenarios;
  return Array.isArray(offered) && offered.length <= scenarios.size
    && new Set(offered).size === offered.length && offered.every(id => scenarios.has(id)) ? offered : [];
}

export function rehearsalDuration(status, offered = rehearsalScenarios(status)) {
  if (status?.rec !== true) return null;
  if (status.duration_ms !== undefined) {
    return Number.isSafeInteger(status.duration_ms) && status.duration_ms > 0 ? status.duration_ms : null;
  }
  // The first shared-loop APKs report a start frame and the main-only
  // catalog, but omit duration_ms. Their native completion boundary is 900 s.
  const start = status.setup?.main_start_frame;
  return status.breath_setup === 1 && status.setup?.v === 1
    && Number.isSafeInteger(start) && start >= 0
    && offered.length === 1 && offered[0] === 'haupt-15' ? 900000 : null;
}

export function polarSetup(status) {
  const setup = status?.y;
  if (status?.p !== 'IDLE' || status.rec || !setup || typeof setup.s !== 'string' || !setupReference.test(setup.s)
    || !Number.isSafeInteger(setup.g) || setup.g <= 0 || String(setup.g) !== status.o?.e
    || !Number.isSafeInteger(setup.r) || setup.r < 0
    || !['p', 'b', 'c', 'a', 'e', 'sc'].every(flag => typeof setup[flag] === 'boolean')
    || Object.hasOwn(setup, 'ar') && typeof setup.ar !== 'boolean'
    || Object.hasOwn(setup, 'auto') && !validAutomation(setup.auto)
    || setup.l !== undefined && !['enabled', 'disabled', 'unavailable', 'legacy-unobserved', 'permission-blocked', 'unknown'].includes(setup.l)
    || typeof setup.x !== 'string' || setup.x !== '' && !candidateId.test(setup.x)
    || setup.c && !setup.x || (setup.a || setup.e) && !setup.c
    || !Array.isArray(setup.d) || setup.d.length > 4 || new Set(setup.d).size !== setup.d.length
    || !setup.d.every(id => typeof id === 'string' && candidateId.test(id))) return null;
  return setup;
}

export function polarReadiness(setup) {
  if (!setup) return 'Polar-Einrichtung ist auf der Quest gerade nicht verfügbar.';
  if (!setup.p) return 'Erlaube „Geräte in der Nähe“ für Viscereality auf der Quest.';
  if (!setup.b) return 'Schalte Bluetooth auf der Quest ein.';
  if (!setup.c && setup.l === 'disabled') return 'Aktiviere die Standortdienste in den Quest-Einstellungen, damit die Quest den H10 finden kann.';
  if (setup.c && setup.e && setup.a) return 'Polar H10 verbunden. Quest empfängt EKG und Beschleunigungsdaten.';
  const automation = setup.auto?.state;
  if (setup.c) return `Polar H10 per Bluetooth verbunden. ${setup.e ? 'EKG bereit; Bewegungsdaten fehlen.'
    : setup.a ? 'Bewegungsdaten bereit; EKG fehlt.' : 'EKG und Bewegungsdaten fehlen.'}${automation === 'streams' || setup.ar === true
      ? ' Die Quest startet fehlende Messsignale automatisch erneut.' : automation === 'stopped' ? ' Automatische Verbindung ist gestoppt.' : ''}`;
  if (setup.sc || automation === 'scanning') return 'Quest sucht automatisch nach dem Polar H10 … Lege den Gurt an.';
  if (automation === 'backoff') return `Kein H10 verbunden. Quest sucht automatisch in ${Math.ceil(setup.auto.retry_ms / 1000)} s erneut. Lege den Gurt an.`;
  if (automation === 'connecting') return 'Quest verbindet sich automatisch mit dem Polar H10 …';
  if (automation === 'selection-required') return 'Mehrere H10 gefunden. Wähle den richtigen Gurt im Bedienfeld der Quest aus.';
  if (automation === 'stopped') return 'Automatische Verbindung ist gestoppt. Verbinde den Polar H10 ausdrücklich erneut.';
  if (automation === 'blocked') return 'Automatische H10-Verbindung blockiert. Prüfe Bluetooth und die Berechtigungen auf der Quest.';
  return 'Lege den Gurt an. Automatische Verbindung nicht gemeldet; du kannst den Polar H10 verbinden.';
}

function setupConfirmed(status, { op, setupRef, candidate }) {
  const setup = polarSetup(status);
  if (!setup || setup.s !== setupRef || !setup.p || !setup.b || setup.sc) return false;
  if (op === 'scan') return true;
  if (setup.x !== candidate) return false;
  if (op === 'select') return true;
  return setup.c && (op === 'connect' || (op === 'start-ecg' ? setup.e : setup.a));
}

export function rehearsalPolarAllowed(status) {
  return canControlQuest(status) && (status?.pi?.r === false || status?.pi?.r === true && status.participant_arm === 1);
}

export function rehearsalActions(status, condition = conditions[0]) {
  if (!rehearsalPolarAllowed(status) || status.pending || audioRecovering(status) || polarSetup(status)?.sc) return {};
  const selected = conditions.indexOf(condition);
  if (status.p === 'IDLE' && status.h === 'vr-preparation' && !status.rec
    && nativeSessionAllowed(status) && selected >= 0 && status.a?.[selected] === 'track-ready') return { play: 'arm' };
  const preparedWriter = status.p === 'ARMED' && status.h === 'vr-session'
    && status.q?.v === 2 && status.q.g === status.g && /^[a-f0-9]{32}$/.test(status.q.s);
  if ((!status.rec && !preparedWriter) || !conditions.includes(status.c)) return {};
  const next = { ARMED: 'start', RUNNING: 'pause', PAUSED: 'resume' }[status.p];
  const grant = status.q;
  const allowed = status.h === 'vr-session' && next && grant?.v === 2 && grant.g === status.g
    && typeof grant.s === 'string' && /^[a-f0-9]{32}$/.test(grant.s) && Array.isArray(grant.o) && grant.o.includes(next);
  const canPlay = allowed && (next === 'pause' || nativeSessionAllowed(status))
    && (next === 'pause' || status.pi.r === false || status.pi.f === true);
  return { play: canPlay ? next : undefined, finish: next || status.p === 'RECORDING' ? 'save-next' : undefined };
}

export class QuestRehearsal {
  gateway = null;
  pending = null;
  refreshing = null;
  state = { status: null, statusTiming: null, busy: false, pending: false, unknown: false, message: '' };

  constructor({ onChange = () => {}, storage, wait = () => new Promise(resolve => setTimeout(resolve, 350)), now = () => performance.now() } = {}) {
    this.onChange = onChange;
    this.storage = storage;
    this.wait = wait;
    this.now = now;
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
    this.update({ status: gateway?.initial ?? null, statusTiming: null, busy: false, message: '', unknown: Boolean(this.pending) });
  }

  canAcknowledgeIdle() {
    const { busy, unknown, status } = this.state;
    return !busy && unknown && canControlQuest(status) && status?.p === 'IDLE' && !status.pending && !status.rec && !polarSetup(status)?.sc
      && status.o?.v === 1 && status.o.s === 'observed' && status.o.e === this.gateway?.owner.native_epoch
      && Number.isSafeInteger(status.o.a) && status.o.a >= 0 && status.o.a <= 1500
      && this.pending?.owner.device === this.gateway?.owner.device
      && this.pending?.owner.serial === this.gateway?.owner.serial
      && !sameQuest(this.pending.owner, this.gateway.owner);
  }

  recovery() {
    if (!this.pending) return null;
    const operation = { arm: 'Vorbereiten', start: 'Starten', pause: 'Pause', resume: 'Fortsetzen',
      'save-next': 'Beenden und Speichern', recenter: 'VR zentrieren', scan: 'Polar suchen',
      select: 'Polar auswählen', connect: 'Polar verbinden', 'start-ecg': 'EKG starten', 'start-acc': 'Bewegungsdaten starten' }[this.pending.op];
    if (this.canAcknowledgeIdle()) return {
      title: 'Die Quest-Verbindung wurde erneuert', action: 'Aktuellen Quest-Zustand übernehmen',
      detail: `${operation}: Die frühere Anfrage gehört zur beendeten Verbindung. Übernimm den jetzt angezeigten Zustand. Das bestätigt nur lokal und setzt die Quest nicht zurück. Kein Befehl wird wiederholt.`,
    };
    const canRead = !this.state.busy && canControlQuest(this.state.status) && sameQuest(this.pending.owner, this.gateway?.owner);
    return {
      title: this.state.unknown ? 'Ergebnis der Anfrage unklar' : 'Bestätigung noch offen',
      action: canRead ? 'Bestätigung erneut abfragen' : '',
      detail: `${operation}: Die Bestätigung der Quest steht noch aus. ${canRead ? 'Frage den Status erneut ab; dabei wird die Anfrage nur gelesen.' : 'Verbinde die ursprüngliche Quest und übernimm bei Bedarf die Steuerung, um die Bestätigung abzufragen.'} ${this.state.unknown ? 'Bleibt das Ergebnis unklar, beende und öffne die Atmungsraum-App im Headset erneut und verbinde dieses Smartphone wieder. Übernimm erst den bestätigten neuen App-Zustand.' : ''} Die Aktion wird nicht erneut gesendet.`,
    };
  }

  acknowledgeIdle() {
    if (!this.canAcknowledgeIdle()) return;
    this.remember(null);
    this.update({ unknown: false, message: '' });
  }

  async observe(gateway, receipt, { reconcile = true } = {}) {
    const started = this.now();
    const status = await gateway.status();
    if (this.gateway !== gateway) return false;
    const receivedAt = this.now();
    this.update({ status, statusTiming: { receivedAt, transportMs: Math.max(0, receivedAt - started) } });
    if (!reconcile || !this.pending) return true;
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
    const polarHold = (op === 'start' || op === 'resume') && status.p === 'PAUSED'
      && status.rec && status.pi?.r === true && status.pi.h === true;
    if (setupOperations.has(op)) {
      if (status.g !== generation || !setupConfirmed(status, this.pending)) return false;
    } else if (op !== 'recenter') {
      const generationMatches = op === 'arm' ? status.g > generation
        : op === 'save-next' ? !status.rec : status.g === generation;
      if ((status.p !== desiredPhase[op] && !polarHold) || !generationMatches
        || op !== 'save-next' && status.c !== (this.pending.condition ?? conditions[0])) return false;
    }
    this.remember(null);
    this.update({ unknown: false, message: op === 'recenter' ? 'VR-Szene zentriert.'
      : op === 'save-next' ? 'Probelauf beendet.' : '' });
    return true;
  }

  refresh({ statusOnly = false } = {}) {
    const gateway = this.gateway;
    if (!gateway || this.state.busy) return this.refreshing;
    if (this.refreshing) {
      if (statusOnly || !this.refreshingStatusOnly) return this.refreshing;
      // A normal tick must still reconcile its receipt after joining a fast
      // preview read. Concurrent normal callers coalesce into that next flight.
      return this.refreshing.then(() => this.gateway === gateway ? this.refresh() : undefined);
    }
    this.refreshingStatusOnly = statusOnly;
    this.refreshing = (async () => {
      try {
        const receipt = !statusOnly && this.pending && sameQuest(this.pending.owner, gateway.owner)
          ? await gateway.receipt(this.pending.id).catch(() => null) : null;
        await this.observe(gateway, receipt, { reconcile: !statusOnly });
      } catch {
        if (this.gateway === gateway) this.update({ status: null, unknown: Boolean(this.pending) });
      }
    })().finally(() => { this.refreshing = null; this.refreshingStatusOnly = false; });
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
      return this.gateway === gateway && !signal?.aborted && nativeSessionAllowed(status) && !status?.pending && !audioRecovering(status) && status?.g === clicked.g
        && setup?.s === original.s && setup.p && setup.b && !setup.sc && (setup.c || setup.l !== 'disabled') ? setup : null;
    };
    let setup = current();
    if (!setup || (setup.e && setup.a)) return;
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
    if (!setup?.c) return;
    if (!setup.e && !await this.dispatchAndConfirm(gateway, 'start-ecg', signal, 32000, setup.x)) return;
    setup = current();
    if (setup?.c && !setup.a) await this.dispatchAndConfirm(gateway, 'start-acc', signal, 32000, setup.x);
  }

  async run(action, signal, condition = conditions[0], participantId = '', scenario = '', { confirmSave = false } = {}) {
    const gateway = this.gateway;
    const clicked = this.state.status;
    const requested = action === 'recenter' && gateway?.recenter && canRecenter(clicked)
      ? 'recenter' : rehearsalActions(clicked, condition)[action];
    const setup = action === 'polar' ? polarSetup(clicked) : null;
    if (!gateway || action === 'polar' && !nativeSessionAllowed(clicked) || clicked?.pending || audioRecovering(clicked) || this.state.busy || this.pending
      || (!requested && (!setup?.p || !setup.b || !setup.c && setup.l === 'disabled' || setup.sc || (setup.e && setup.a)))) return;
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
        || op !== 'arm' && status.q?.s !== clicked.q?.s) return;
      if (['pause', 'resume', 'save-next'].includes(op)) {
        const before = controllerAccessState(clicked), after = controllerAccessState(status);
        const age = this.now() - this.state.statusTiming.receivedAt + this.state.statusTiming.transportMs;
        if (!Number.isFinite(age) || age < 0 || age > 2500 || status.r !== clicked.r || before?.revision !== after?.revision
          || before?.active_phone !== after?.active_phone || before?.role !== after?.role) return;
      }
      if (op !== 'arm') {
        if (op === 'save-next' && confirmSave) {
          return await this.dispatchAndConfirm(gateway, op, signal, 15000);
        }
        await this.dispatch(gateway, op, status, signal);
        return;
      }
      const streams = polarSetup(status);
      if (status.participant_arm === 1 && (!participantId || !gateway.participantArm || !streams?.e || !streams.a)) return;
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
