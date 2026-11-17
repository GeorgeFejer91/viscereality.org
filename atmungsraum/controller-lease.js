import { controllerAccessState, operationId, sameQuest } from './quest-gateway.js?v=13';

const journalKey = 'atmungsraum-controller-lease';
const freshFor = 2500;
const retryAfter = 5000;
const retryCooldown = 30000;

// This policy requests the existing native lease. It never grants authority,
// retries an unknown operation, or changes the session/recording journal.
export class ControllerLease {
  constructor({ now = () => performance.now(), storage } = {}) {
    this.now = now;
    this.storage = storage;
    this.desired = true;
    this.attempts = 0;
    this.nextAttempt = 0;
    try {
      const saved = JSON.parse(storage?.getItem(journalKey) ?? 'null');
      if (saved?.owner && /^[a-f0-9]{16}$/.test(saved.id)
        && Number.isSafeInteger(saved.revision) && saved.revision >= 0) this.pending = saved;
    } catch { /* The in-memory unknown fence remains effective without storage. */ }
  }

  remember(pending) {
    this.pending = pending;
    try {
      if (pending) this.storage?.setItem(journalKey, JSON.stringify(pending));
      else this.storage?.removeItem(journalKey);
    } catch { /* Storage is optional; never replay the in-memory request. */ }
  }

  release() { this.desired = false; }
  request() { this.desired = true; this.attempts = 0; this.nextAttempt = 0; }

  update(gateway, state, { visible = true, blocked = false } = {}) {
    const status = state?.status;
    let access;
    try { access = controllerAccessState(status); } catch { /* Invalid roles cannot grant control. */ }
    const age = this.now() - state?.statusTiming?.receivedAt + (state?.statusTiming?.transportMs ?? 0);
    const fresh = Boolean(gateway && access && Number.isFinite(age) && age >= 0 && age <= freshFor);
    if (fresh) {
      if (this.pending && (!sameQuest(this.pending.owner, gateway.owner)
        || access.role === 'active' || access.revision > this.pending.revision)) this.remember(null);
      const binding = `${gateway.owner.device}/${gateway.owner.epoch}/${gateway.owner.native_epoch}/${access.revision}`;
      if (binding !== this.binding) { this.binding = binding; this.attempts = 0; this.nextAttempt = 0; }
    }
    const active = Boolean(visible && fresh && access.role === 'active');
    if (active) this.wasActive = true;
    const busy = blocked || state?.busy || status?.pending || status?.setup?.stage === 'recovering';
    const eligible = Boolean(visible && fresh && gateway.controllerAccess && this.desired && !busy
      && access.role === 'waiting' && !access.active_phone && !this.pending && !this.inFlight
      && this.now() >= this.nextAttempt);
    const phase = !visible ? 'checking' : !fresh ? this.wasActive ? 'lost' : 'checking' : active ? 'active' : this.inFlight ? 'requesting'
      : this.pending ? 'unknown' : !this.desired ? 'released' : access.active_phone ? 'occupied'
        : busy ? 'blocked' : 'retrying';
    const message = {
      checking: 'Steuerung nicht bestätigt. Die App prüft den aktuellen Quest-Status.',
      lost: 'Steuerung nicht mehr bestätigt. Die Quest-Verbindung wird erneut geprüft; dieses Smartphone übernimmt eine freigegebene Steuerung automatisch.',
      active: 'Dieses Smartphone steuert die Quest. Die Quest hat die Steuerung bestätigt.',
      requesting: 'Die App übernimmt die freie Quest-Steuerung und wartet auf den aktuellen Status.',
      unknown: 'Übernahme unklar. Die App liest nur den Quest-Status; die Anfrage wird nicht wiederholt. Öffne bei Bedarf die Atmungsraum-App im Headset neu und verbinde dieses Smartphone erneut.',
      released: 'Steuerung freigegeben. Übernimm sie hier erneut, wenn dieses Smartphone wieder steuern soll.',
      occupied: `${access?.active_phone} steuert die Quest. Schließe die andere Bedien-App oder gib dort die Steuerung frei. Dieses Smartphone übernimmt, sobald die Quest sie freigibt.`,
      blocked: 'Die Quest bearbeitet noch eine Anfrage. Die App prüft weiter den Status und übernimmt danach die freie Steuerung.',
      retrying: this.attempts >= 3
        ? 'Steuerung noch nicht bestätigt. Die App prüft weiter den Status und versucht die freie Steuerung alle dreißig Sekunden erneut zu übernehmen.'
        : 'Die App übernimmt die freie Quest-Steuerung automatisch. Falls nötig, versucht sie es nach fünf Sekunden erneut.',
    }[phase];
    this.view = { active, eligible, state: phase, message, access, fresh,
      expiresIn: active ? Math.max(0, freshFor - age) : null };
    return this.view;
  }

  async acquire(gateway, state, refresh, options = {}) {
    if (!this.update(gateway, state, options).eligible) return;
    // Serialize callers before the preflight read, and recheck the live UI and
    // native state after it. A status read may reveal a new owner or recording.
    const flight = {};
    this.inFlight = flight;
    try {
      await refresh();
      if (this.inFlight !== flight) return;
      this.inFlight = null;
      const current = typeof options.current === 'function' ? options.current() : { gateway, state, ...options };
      if (current.gateway !== gateway || !this.update(gateway, current.state, current).eligible) return;
      this.inFlight = flight;
      const access = this.view.access;
      const pending = { owner: gateway.owner, id: operationId(), revision: access.revision };
      this.remember(pending);
      this.attempts = Math.min(3, this.attempts + 1);
      this.nextAttempt = this.now() + (this.attempts >= 3 ? retryCooldown : retryAfter);
      const reply = await gateway.controllerAccess({ v: 1, id: pending.id, op: 'claim', expected_control_revision: pending.revision }, options.signal);
      // A valid waiting response is a conclusive non-grant. Everything else
      // remains fenced until status proves active control or a newer revision.
      let result;
      try { result = controllerAccessState({ controller_access: reply }); } catch { /* Unknown reply. */ }
      if (result?.role === 'waiting' && result.revision >= pending.revision) this.remember(null);
    } catch { /* Lost/malformed replies stay unknown, with no new-ID replay. */ }
    finally {
      if (this.inFlight === flight) this.inFlight = null;
      await refresh();
    }
  }
}
