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
export function sessionFilename(patient, branch, runIndex) {
  if (!Number.isSafeInteger(runIndex) || runIndex < 1) throw new Error('Invalid run index');
  return `${participantIdentity(patient).id}_${conditionFor(branch)}${runIndex}.csv`;
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
    const filename = sessionFilename(this.patient, selection.branch, this.runIndex);
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
function reserveRun(db, key) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('counters', 'readwrite');
    const store = tx.objectStore('counters');
    const request = store.get(key);
    let index;
    request.onsuccess = () => {
      index = (request.result ?? 0) + 1;
      if (!Number.isSafeInteger(index) || index < 1) { tx.abort(); return; }
      store.put(index, key);
    };
    tx.oncomplete = () => resolve(index);
    tx.onabort = tx.onerror = () => reject(tx.error ?? new Error('Run counter unavailable'));
  });
}
function storeExport(db, record) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(['exports', 'active'], 'readwrite');
    tx.objectStore('exports').put(record);
    tx.objectStore('active').delete(record.session_id);
    tx.oncomplete = resolve;
    tx.onabort = tx.onerror = () => reject(tx.error ?? new Error('CSV storage unavailable'));
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
function checkpoint(db, log, includeSource = false) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(includeSource ? ['active', 'sources'] : ['active'], 'readwrite');
    tx.objectStore('active').put(log.checkpoint());
    if (includeSource) tx.objectStore('sources').put({ samples: log.signal.samples, manifest: log.signal.manifest, rate: log.signal.rate }, log.signal.manifest.sha256);
    tx.oncomplete = resolve;
    tx.onabort = tx.onerror = () => reject(tx.error ?? new Error('Checkpoint unavailable'));
  });
}
const lockName = id => `atmungsraum-session:${id}`;
function holdSession(id) {
  return new Promise((resolve, reject) => {
    navigator.locks.request(lockName(id), () => new Promise(release => { resolve(release); })).catch(reject);
  });
}

export function mountSessionData(signal) {
  const list = document.getElementById('data-files');
  const status = document.getElementById('data-status');
  const links = new Map();
  let db, preparation;
  function show(record, download = false) {
    let link = links.get(record.session_id);
    if (!link) {
      link = document.createElement('a');
      link.href = URL.createObjectURL(new Blob([record.csv], { type: 'text/csv;charset=utf-8' }));
      link.download = record.filename;
      const label = document.createElement('span');
      label.dataset.measure = '';
      label.textContent = record.filename;
      link.append(label);
      const item = document.createElement('li');
      item.append(link); list.prepend(item);
      links.set(record.session_id, link);
    }
    if (download) link.click();
    status.textContent = 'CSV-Datei zum Herunterladen auswählen.';
    document.getElementById('data-recovery').hidden = false;
  }
  return {
    prepare() {
      if (!navigator.locks) return Promise.reject(new Error('Safe session recovery requires Web Locks'));
      preparation ??= openDatabase().then(database => {
        db = database;
        return readStore(db, 'exports').then(async records => {
          records.sort((a,b) => a.created_at.localeCompare(b.created_at)).forEach(record => show(record));
          for (const saved of await readStore(db, 'active')) {
            await navigator.locks.request(lockName(saved.session_id), { ifAvailable: true }, async lock => {
              if (!lock) return; // Another tab still owns this live recording.
              const current = await readStore(db, 'active', saved.session_id);
              if (!current) return;
              const source = await readStore(db, 'sources', current.sourceHash);
              if (!source) throw new Error('Interrupted session source unavailable');
              const record = SessionLog.recover(current, { ...source, indexAt: () => 0 });
              await storeExport(db, record);
              show(record);
            });
          }
        });
      }).catch(error => { preparation = null; throw error; });
      return preparation;
    },
    begin(options) {
      if (!db || !signal.samples) throw new Error('Session data not ready');
      const log = new SessionLog({ ...options, signal });
      log.ready = reserveRun(db, `${log.identity.id}_${log.condition}`).then(async index => {
        log.runIndex = index;
        log.releaseLock = await holdSession(log.id);
        try { await checkpoint(db, log, true); }
        catch (error) { log.releaseLock(); throw error; }
        const saveCheckpoint = () => checkpoint(db, log).catch(error => {
          status.textContent = 'Zwischenstand konnte nicht gesichert werden. Session beenden und CSV herunterladen.';
          console.warn('Session checkpoint failed.', error);
        });
        if (log.ended == null) {
          log.onChange = saveCheckpoint;
          log.checkpointTimer = setInterval(saveCheckpoint, 1000);
        }
      });
      return log;
    },
    finish(receipt, attempt) {
      if (!attempt.recording?.runIndex || attempt.dataFinished) return;
      attempt.dataFinished = true;
      clearInterval(attempt.recording.checkpointTimer);
      attempt.recording.onChange = null;
      const record = attempt.recording.finish(receipt.status, receipt.records?.findLast(event => event.type === 'abort')?.reason ?? '');
      show(record, true);
      storeExport(db, record).catch(error => {
        status.textContent = 'CSV konnte nicht im Browser gesichert werden. Bitte jetzt herunterladen.';
        console.warn('Session CSV storage failed.', error);
      }).finally(() => attempt.recording.releaseLock?.());
    },
  };
}
