const liveFor = 5000;
const batteryFor = 120000;
const decimal = value => typeof value === 'string' && /^(0|[1-9][0-9]{0,18})$/.test(value);
const age = value => value === null || Number.isFinite(value) && value >= 0;
const percent = value => value === null || Number.isInteger(value) && value >= 0 && value <= 100;
const nullableBoolean = value => value === null || typeof value === 'boolean';
const signal = value => value && age(value.age_ms) && typeof value.fresh === 'boolean';
const validHealth = value => value?.v === 1 && typeof value.link === 'boolean'
  && decimal(value.connection_generation) && decimal(value.observed_at_ns)
  && signal(value.ecg) && signal(value.acc) && value.contact
  && ['detected', 'not-detected', 'unsupported', 'stale', 'unknown'].includes(value.contact.state)
  && nullableBoolean(value.contact.supported) && nullableBoolean(value.contact.detected) && age(value.contact.age_ms)
  && value.battery && percent(value.battery.percent) && age(value.battery.age_ms);
const validPower = value => value?.v === 1 && value.quest && percent(value.quest.percent)
  && nullableBoolean(value.quest.charging) && Array.isArray(value.controllers) && value.controllers.length <= 8
  && value.controllers.every(item => item && typeof item.name === 'string' && item.name.trim()
    && item.name.length <= 128 && !/[\x00-\x1f\x7f]/.test(item.name) && percent(item.percent));
const valueText = value => value === null ? 'unbekannt' : `${value} %`;

// Presentation only. These observations never admit input, Arm, recording or
// session actions, and sample activity is never evidence of body contact.
export class OperatorDeviceHealth {
  constructor({ now = () => performance.now() } = {}) { this.now = now; }

  update(owner, { status, statusTiming } = {}) {
    const now = this.now();
    const timingAge = now - statusTiming?.receivedAt + (statusTiming?.transportMs ?? 0) + (status?.o?.a ?? NaN);
    const observed = Boolean(owner && status?.o?.e === owner.native_epoch && Number.isFinite(timingAge) && timingAge >= 0);
    const fresh = observed && timingAge < liveFor;
    const binding = owner && `${owner.device}/${owner.serial}/${owner.epoch}/${owner.native_epoch}`;
    if (binding !== this.binding) {
      this.binding = binding; this.generation = this.witness = this.firstSeen = null;
    }
    let health = validHealth(status?.polar_health) ? status.polar_health : null;
    let overhead = timingAge;
    if (fresh && health) {
      const generation = BigInt(health.connection_generation), witness = BigInt(health.observed_at_ns);
      if (this.generation !== null && (generation < this.generation
        || generation === this.generation && witness < this.witness)) health = null;
      else {
        if (generation !== this.generation || witness !== this.witness) {
          this.generation = generation; this.witness = witness;
          this.firstSeen = { at: statusTiming.receivedAt, overhead: timingAge - (now - statusTiming.receivedAt) };
        }
        // Repeated witnesses cannot renew cached health. Do not subtract Java
        // nanoTime from a phone clock, or count elapsed phone time twice.
        overhead = Math.max(timingAge, now - this.firstSeen.at + this.firstSeen.overhead);
      }
    }
    if (!fresh || overhead >= liveFor) health = null;
    const remaining = [];
    if (fresh) remaining.push(liveFor - timingAge);
    const fieldFresh = (fieldAge, limit = liveFor) => {
      if (!health || fieldAge === null || fieldAge + overhead >= limit) return false;
      remaining.push(limit - fieldAge - overhead);
      return true;
    };
    let polar = { confirmed: null, state: 'unknown', neutral: 'Quest-App meldet keinen Polar-Status',
      detail: 'Aktueller Polar-Status wird von dieser Quest-App nicht gemeldet. Quest-App aktualisieren, um die Statusprüfung zu nutzen. Tragekontakt unbekannt; EKG und Bewegung allein bestätigen ihn nicht.' };
    if (health) {
      const ecg = health.ecg.fresh && fieldFresh(health.ecg.age_ms);
      const acc = health.acc.fresh && fieldFresh(health.acc.age_ms);
      const contactFresh = fieldFresh(health.contact.age_ms);
      if (!health.link) polar = { confirmed: false, state: 'disconnected', detail: 'Polar H10 nicht verbunden. Prüfe den Gurt und die Quest-Verbindung.' };
      else if (!ecg || !acc) polar = { confirmed: false, state: 'signals', detail: `Polar H10 verbunden. ${!ecg ? 'Aktuelles EKG fehlt. ' : ''}${!acc ? 'Aktuelle Bewegungsdaten fehlen. ' : ''}Tragekontakt wird separat geprüft.` };
      else if (contactFresh && health.contact.state === 'detected' && health.contact.supported === true && health.contact.detected === true)
        polar = { confirmed: true, state: 'ready', detail: 'Polar H10 verbunden. EKG und Bewegung aktuell; Tragekontakt bestätigt.' };
      else if (contactFresh && health.contact.state === 'not-detected' && health.contact.supported === true && health.contact.detected === false)
        polar = { confirmed: false, state: 'contact', detail: 'Polar H10 meldet keinen Tragekontakt. Prüfe den Sitz und die Elektroden des Gurts.' };
      else polar = { confirmed: null, state: 'contact-unknown', neutral: 'Tragekontakt unbekannt',
        detail: contactFresh && health.contact.state === 'unsupported' && health.contact.supported === false
          ? 'EKG und Bewegung aktuell. Dieser Polar H10 meldet keinen prüfbaren Tragekontakt; der Tragezustand bleibt unbekannt.'
          : 'EKG und Bewegung aktuell. Ein aktueller Tragekontakt ist nicht bestätigt.' };
      remaining.push(liveFor - overhead);
    } else if (status?.polar_health) polar.detail = 'Polar-Status nicht aktuell oder nicht verlässlich. Tragekontakt unbekannt; die App prüft weiter.';
    const power = fresh && validPower(status?.power) ? status.power : null;
    const battery = {
      quest: power ? valueText(power.quest.percent) + (power.quest.charging === true ? ' · lädt' : '') : 'unbekannt',
      controllers: power?.controllers.length ? power.controllers.map((item, index) =>
        `${power.controllers.length === 1 ? 'Controller' : `Controller ${index + 1}`}: ${valueText(item.percent)}`).join('\n') : 'keine Akkudaten',
      polar: health?.link ? fieldFresh(health.battery.age_ms, batteryFor) ? valueText(health.battery.percent) : 'unbekannt'
        : health ? 'nicht verbunden' : 'unbekannt',
    };
    const polarConnectionGeneration = health && health.connection_generation !== '0'
      ? health.connection_generation : null;
    this.view = { polar, battery, polarConnectionGeneration,
      expiresIn: remaining.length ? Math.max(1, Math.min(...remaining)) : null };
    return this.view;
  }
}
