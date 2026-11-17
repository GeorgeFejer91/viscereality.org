import { nativeMetaUpdate } from './meta-update.js?v=3';
import { installedApk } from './quest-release.js';

const key = 'atmungsraum-meta-update-handoff';
const detail = 'Fortschritt auf diesem Smartphone nicht verfügbar. Prüfe das Update im Headset. Sobald Atmungsraum dort wieder geöffnet ist, verbinde dieses Smartphone erneut.';
const lineage = value => typeof value === 'string' && /^[1-9][0-9]{0,18}$/.test(value);
const positive = value => Number.isSafeInteger(value) && value > 0;
const exact = (value, fields) => value && !Array.isArray(value)
  && Object.keys(value).sort().join(',') === fields.slice().sort().join(',');
const ownerFields = ['device', 'serial', 'epoch', 'native_epoch', 'identity_kind'];
const fields = ['v', 'owner', 'package', 'installed', 'target', 'attempt', 'cycle', 'revision', 'observation', 'lastState', 'confirmed'];
const phases = new Set(['requested', 'waiting', 'checking', 'downloading', 'installing', 'updated', 'current', 'failed', 'restart_required', 'update_required']);
const active = new Set(['checking', 'downloading', 'installing', 'restart_required', 'update_required']);
const handoffReasons = new Set(['queued-meta-update-handoff', 'installing-meta-channel-update']);
const labels = { requested: 'Update-Prüfung angefragt', waiting: 'Prüfung ausstehend', checking: 'Update-Prüfung',
  downloading: 'Download', installing: 'Installationsanfrage', current: 'Versionsprüfung', updated: 'APK-Rückmeldung',
  failed: 'Prüfung fehlgeschlagen', restart_required: 'Installationsstand unklar', update_required: 'Update erforderlich' };
const ownerValid = owner => exact(owner, ownerFields)
  && typeof owner.device === 'string' && /^[a-f0-9]{32}$/.test(owner.device)
  && typeof owner.serial === 'string' && (owner.serial === '' && owner.identity_kind === 'app-installation'
    || /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(owner.serial))
  && lineage(owner.epoch) && lineage(owner.native_epoch) && ['', 'app-installation'].includes(owner.identity_kind);
const packageValid = value => typeof value === 'string' && value.length <= 200
  && /^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+$/.test(value);
const recordValid = record => exact(record, fields) && record.v === 1 && ownerValid(record.owner)
  && packageValid(record.package) && positive(record.installed) && (record.target === null || positive(record.target))
  && (record.attempt === null || lineage(record.attempt)) && lineage(record.cycle) && lineage(record.revision)
  && lineage(record.observation) && phases.has(record.lastState) && typeof record.confirmed === 'boolean';
const sameDevice = (a, b) => a?.device === b?.device && a?.serial === b?.serial
  && (a?.identity_kind ?? '') === (b?.identity_kind ?? '');
const sameEpoch = (a, b) => a.epoch === b.epoch && a.native_epoch === b.native_epoch;

// This separate local hint is presentation only. It never reads or owns the
// mutation journal, calls a gateway, changes readiness or schedules recovery.
export class MetaUpdateHandoff {
  constructor({ storage } = {}) {
    this.storage = storage;
    this.record = null;
    this.connected = false;
    try {
      const raw = storage?.getItem(key);
      if (typeof raw === 'string' && raw.length <= 1400) {
        const record = JSON.parse(raw);
        if (recordValid(record)) this.record = record;
      }
    } catch { /* An unavailable local hint has no native effect. */ }
  }

  remember(record) {
    this.record = record;
    try {
      if (record) this.storage?.setItem(key, JSON.stringify(record));
      else this.storage?.removeItem(key);
    } catch { /* Keep the in-memory hint without issuing any operation. */ }
  }

  observation(gateway, state) {
    const source = gateway?.owner;
    const owner = source && { device: source.device, serial: source.serial, epoch: source.epoch,
      native_epoch: source.native_epoch, identity_kind: source.identity_kind ?? '' };
    if (!ownerValid(owner)) return null;
    const apk = installedApk(state?.status, owner);
    const parsed = nativeMetaUpdate(state?.status);
    const value = parsed.value;
    if (!apk || value?.v !== 2 || value.installed_code !== apk.version_code) return null;
    const previous = this.record;
    if (previous) {
      if (!sameDevice(previous.owner, owner) || previous.package !== apk.package_name) return null;
      if (sameEpoch(previous.owner, owner)) {
        if (BigInt(state.status.o.i) <= BigInt(previous.observation)
          || BigInt(value.cycle) < BigInt(previous.cycle)
          || value.cycle === previous.cycle && BigInt(value.revision) < BigInt(previous.revision)
          || value.cycle === previous.cycle && value.attempt !== previous.attempt
            && BigInt(value.revision) <= BigInt(previous.revision)) return null;
      }
    }
    return { owner, apk, parsed, value, observation: state.status.o.i };
  }

  observe(gateway, state) {
    const observation = this.observation(gateway, state);
    if (!observation) return false;
    const { owner, apk, parsed, value } = observation;
    const previous = this.record;
    const target = Math.max(previous?.target ?? 0, value.required_code ?? 0, value.available_code ?? 0) || null;
    const ready = parsed.allowed && ['current', 'updated'].includes(value.state) && !value.sdk_error
      && state.status.p === 'IDLE' && state.status.rec === false && state.status.pending === false
      && !state.busy && !state.pending && !state.unknown;
    this.connected = true;
    if (previous && ready && apk.version_code >= Math.max(previous.installed, target ?? 0)) {
      this.remember(null);
      return true;
    }
    if (!previous && !active.has(value.state)) return true;
    const sameAttempt = previous && sameEpoch(previous.owner, owner)
      && value.cycle === previous.cycle && value.attempt === previous.attempt;
    this.remember({ v: 1, owner, package: apk.package_name, installed: previous?.installed ?? apk.version_code,
      target, attempt: value.attempt, cycle: value.cycle, revision: value.revision,
      observation: observation.observation, lastState: value.state,
      confirmed: Boolean(lineage(value.attempt) && (value.state === 'installing' && handoffReasons.has(value.reason)
        || sameAttempt && previous.confirmed)) });
    return true;
  }

  // Call only when the existing command owner actually sent its request. An
  // optional original accepted receipt is copied as lineage, never replayed.
  manualRequest(gateway, state, accepted = null) {
    const observation = this.observation(gateway, state);
    if (!observation) return false;
    const { owner, apk, value } = observation;
    if (accepted && (!exact(accepted, ['v', 'id', 'state', 'cycle', 'revision', 'attempt', 'reason'])
      || accepted.v !== 1 || typeof accepted.id !== 'string' || !/^[a-f0-9]{16}$/.test(accepted.id)
      || accepted.state !== 'accepted' || accepted.cycle !== value.cycle || !lineage(accepted.revision)
      || BigInt(accepted.revision) <= BigInt(value.revision) || !lineage(accepted.attempt)
      || typeof accepted.reason !== 'string' || accepted.reason.length > 96 || /[^\x20-\x7e]/.test(accepted.reason))) return false;
    this.connected = true;
    this.remember({ v: 1, owner, package: apk.package_name, installed: this.record?.installed ?? apk.version_code,
      target: Math.max(this.record?.target ?? 0, value.required_code ?? 0, value.available_code ?? 0) || null,
      cycle: value.cycle, revision: accepted?.revision ?? value.revision, attempt: accepted?.attempt ?? value.attempt,
      observation: observation.observation, lastState: 'requested', confirmed: false });
    return true;
  }

  disconnected() { this.connected = false; }

  view(selectedIdentity) {
    const record = this.record;
    if (!record || this.connected || record.owner.device !== selectedIdentity?.device
      || record.owner.serial !== selectedIdentity?.serial
      || selectedIdentity.identity_kind !== undefined && selectedIdentity.identity_kind !== record.owner.identity_kind
      || selectedIdentity.package_name && selectedIdentity.package_name !== record.package) return null;
    return {
      state: 'connection-unavailable', confirmedHandoff: record.confirmed, lastState: record.lastState,
      targetCode: record.target, progress: null,
      summary: record.confirmed
        ? 'Die Quest hat Atmungsraum für ein Meta-Update übergeben. Der aktuelle Stand ist auf diesem Smartphone unbekannt.'
        : `Zuletzt bestätigt: ${labels[record.lastState]}. Der aktuelle Stand ist auf diesem Smartphone unbekannt.`,
      detail,
    };
  }
}
