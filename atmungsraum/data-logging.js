import { SCENARIOS } from './audio/scenario-protocol.js?v=20261106-placeholders';

// All participant-data writers must share this identity and filename protocol.
export function participantIdentity({ surnamePrefix = '', age = '', gender = '' } = {}) {
  if (typeof surnamePrefix !== 'string' || !['string', 'number'].includes(typeof age)) throw new Error('Invalid participant');
  let name = surnamePrefix.trim().normalize('NFC');
  if (!/^\p{L}{3}$/u.test(name)) throw new Error('Enter three surname letters');
  const lower = name.toLocaleLowerCase('de-DE');
  const styled = lower[0].toLocaleUpperCase('de-DE') + lower.slice(1);
  if (/^\p{L}{3}$/u.test(styled)) name = styled;
  const years = String(age).trim() === '' ? NaN : Number(age);
  const genders = { male: 'M', female: 'F', other: 'O', 'prefer-not-to-say': 'N' };
  const genderCode = Object.hasOwn(genders, gender) ? genders[gender] : null;
  if (!Number.isSafeInteger(years) || years < 0 || !genderCode) throw new Error('Invalid age or gender');
  return { name, age: years, genderCode, id: `${name}_${years}_${genderCode}` };
}
const conditionFor = branch => {
  const condition = branch === 'probe' ? 'probe' : branch === 'haupt' ? 'main' : null;
  if (!condition) throw new Error('Unregistered condition');
  return condition;
};
export function sessionFilename(patient, branch, runIndex, scenario, startedAt) {
  if (!Number.isSafeInteger(runIndex) || runIndex < 1) throw new Error('Invalid run index');
  if (!SCENARIOS.includes(scenario) || scenario.startsWith('haupt-') !== (conditionFor(branch) === 'main')) {
    throw new Error('Unregistered session');
  }
  const date = new Date(startedAt).toISOString().slice(0, 10);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error('Invalid recording date');
  return `${participantIdentity(patient).id}_${date}_${scenario}_r${runIndex}.csv`;
}

export function isDatedSessionFilename(name) {
  const parts = /^\p{L}{3}_(0|[1-9][0-9]{0,15})_[MFON]_(\d{4}-\d{2}-\d{2})_([a-z0-9-]+)_r[1-9][0-9]{0,15}\.csv$/u.exec(name);
  if (!parts || !SCENARIOS.includes(parts[3])) return false;
  const date = new Date(parts[2]);
  return Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === parts[2];
}
const cell = value => {
  let text = String(value ?? '');
  if (/^[=+@\-]/.test(text) && !/^-?\d+(\.\d+)?$/.test(text)) text = `'${text}`;
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};

export class SessionLog {
  constructor({ patient, selection, runIndex, sessionId, signal, protocol, audioStartDelayMs = 0,
    now = () => performance.now(), utc = () => Date.now() }) {
    this.patient = { ...patient };
    this.identity = participantIdentity(patient);
    this.selection = { ...selection };
    this.condition = conditionFor(selection.branch);
    this.runIndex = runIndex;
    this.id = sessionId;
    this.signal = signal;
    this.protocol = protocol;
    this.now = now;
    this.started = now();
    this.utc = utc();
    this.sourceStart = signal.indexAt(this.started);
    this.marks = [{ at: 0, mode: 'playing', audio: -audioStartDelayMs }];
    this.sensorMarks = [{ at: 0, state: 'ecg' }];
  }
  mark(mode, audioSeconds) {
    if (!this.result && this.ended == null) {
      this.marks.push({ at: this.now() - this.started, mode, audio: audioSeconds * 1000 });
      this.onChange?.();
    }
  }
  markSensor(state) {
    if (!this.result && this.ended == null && this.sensorMarks.at(-1).state !== state) {
      this.sensorMarks.push({ at: this.now() - this.started, state });
      this.onChange?.();
    }
  }
  stop() { if (this.ended == null) { this.ended = this.now(); this.onChange?.(); } }
  checkpoint() {
    return { session_id: this.id, patient: this.patient, selection: this.selection, runIndex: this.runIndex,
      protocol: this.protocol, utc: this.utc, sourceStart: this.sourceStart, sourceHash: this.signal.manifest.sha256,
      elapsed: Math.max(0, (this.ended ?? this.now()) - this.started), marks: this.marks, sensorMarks: this.sensorMarks };
  }
  static recover(saved, signal) {
    const log = new SessionLog({ patient: saved.patient, selection: saved.selection, runIndex: saved.runIndex,
      sessionId: saved.session_id, protocol: saved.protocol, signal, now: () => 0, utc: () => saved.utc });
    Object.assign(log, { sourceStart: saved.sourceStart, marks: saved.marks, sensorMarks: saved.sensorMarks, ended: saved.elapsed });
    return log.finish('aborted', 'page-reload-or-interrupted');
  }
  finish(status, reason) {
    if (this.result) return this.result;
    this.stop();
    const elapsed = Math.max(0, this.ended - this.started);
    const { identity: who, signal, selection, protocol } = this;
    const filename = sessionFilename(this.patient, selection.branch, this.runIndex, protocol.scenario_id, this.utc);
    const headers = ['schema_version', 'participant_id', 'name_prefix', 'age_years', 'gender_code', 'condition', 'run_index',
      'session_id', 'scenario_id', 'scenario_sha256', 'planned_duration_seconds', 'session_started_utc',
      'sample_index', 'elapsed_ms', 'timestamp_utc', 'audio_elapsed_ms', 'playback_state', 'phase',
      'ecg_value', 'ecg_unit', 'signal_source', 'signal_state', 'signal_sha256', 'sample_rate_hz', 'source_sample_index', 'source_loop',
      'session_status', 'end_reason'];
    const prefix = [1, who.id, who.name, who.age, who.genderCode, this.condition, this.runIndex, this.id,
      protocol.scenario_id, protocol.scenario_sha256, protocol.total_frames / protocol.sample_rate, new Date(this.utc).toISOString()];
    const lines = [headers.join(',')];
    let markIndex = 0, phaseIndex = 0, sensorIndex = 0;
    const count = Math.max(1, Math.ceil(elapsed * signal.rate / 1000));
    for (let i = 0; i < count; i++) {
      const ms = i * 1000 / signal.rate;
      while (markIndex + 1 < this.marks.length && this.marks[markIndex + 1].at <= ms) markIndex++;
      while (sensorIndex + 1 < this.sensorMarks.length && this.sensorMarks[sensorIndex + 1].at <= ms) sensorIndex++;
      const sensor = this.sensorMarks[sensorIndex].state;
      const mark = this.marks[markIndex];
      const signedAudio = mark.audio + (mark.mode === 'playing' ? ms - mark.at : 0);
      const audio = Math.max(0, Math.min(protocol.total_frames / protocol.sample_rate * 1000, signedAudio));
      const frame = audio * protocol.sample_rate / 1000;
      while (phaseIndex + 1 < protocol.events.length && protocol.events[phaseIndex + 1].offset_frames <= frame) phaseIndex++;
      const event = protocol.events[phaseIndex];
      const source = this.sourceStart + i;
      const sourceIndex = source % signal.samples.length;
      const phase = signedAudio < 0 ? 'waiting' : event?.id ?? 'unknown';
      lines.push([...prefix, i, ms.toFixed(3), new Date(this.utc + ms).toISOString(), audio.toFixed(3), mark.mode, phase,
        sensor === 'ecg' ? signal.samples[sourceIndex].toFixed(6) : '', signal.manifest.unit, signal.manifest.source, sensor, signal.manifest.sha256,
        signal.rate, sourceIndex, Math.floor(source / signal.samples.length), status, reason].map(cell).join(','));
    }
    this.result = { session_id: this.id, filename, csv: lines.join('\r\n') + '\r\n', created_at: new Date(this.utc).toISOString() };
    return this.result;
  }
}

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('atmungsraum-session-data', 2);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('counters')) db.createObjectStore('counters');
      if (!db.objectStoreNames.contains('exports')) db.createObjectStore('exports', { keyPath: 'session_id' });
      db.createObjectStore('active', { keyPath: 'session_id' });
      db.createObjectStore('sources');
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error('Session database blocked'));
  });
}
function readStore(db, store, key) {
  return new Promise((resolve, reject) => {
    const records = db.transaction(store).objectStore(store);
    const request = key === undefined ? records.getAll() : records.get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
export async function savedBrowserExports() {
  const db = await openDatabase();
  try {
    const records = await readStore(db, 'exports');
    return records.sort((first, second) => second.created_at.localeCompare(first.created_at));
  } finally { db.close(); }
}
