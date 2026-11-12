import { createExportsZip } from './data-files.js?v=20261108-quest-data';

const recordingId = /^utc-ns-[0-9]{20}$/;
const digest = /^[a-f0-9]{64}$/;
const files = ['manifest.json', 'final.json', 'events.jsonl', 'polar-acc.jsonl', 'polar-ecg.jsonl',
  'polar-hr-rr.jsonl', 'breath.jsonl', 'radius.jsonl'];
const sensorFiles = new Set(['polar-acc.jsonl', 'polar-ecg.jsonl']);
const maxBytes = 128 * 1024 * 1024;

export function ecgTrace(frame, transportMs) {
  if (frame?.v !== 1 || frame.unit !== 'uV' || frame.rate !== 130
    || !Number.isSafeInteger(frame.age_ms) || frame.age_ms < 0 || frame.age_ms + transportMs > 1500
    || !Array.isArray(frame.samples) || frame.samples.length < 2 || frame.samples.length > 260) return null;
  let previous = 0n;
  const samples = [];
  for (const point of frame.samples) {
    if (!Array.isArray(point) || point.length !== 2 || typeof point[0] !== 'string'
      || !/^[1-9][0-9]{0,18}$/.test(point[0]) || !Number.isSafeInteger(point[1])
      || point[1] < -8388608 || point[1] > 8388607) return null;
    const timestamp = BigInt(point[0]);
    if (timestamp <= previous) return null;
    samples.push({ timestamp, value: point[1], gap: timestamp - previous > 12000000n });
    previous = timestamp;
  }
  const end = samples.at(-1).timestamp;
  const window = samples.filter(({ timestamp }) => end - timestamp <= 2000000000n);
  if (window.length < 2) return null;
  const range = Math.max(100, ...window.map(({ value }) => Math.abs(value)));
  const path = window.map(({ timestamp, value, gap }, index) => {
    const x = 400 - Number(end - timestamp) / 5000000;
    const y = 46 - value / range * 43;
    return `${index === 0 || gap ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(' ');
  return { range, path, ttl: 2500 - frame.age_ms - transportMs };
}

async function preparedRecordingQuery(gateway, query, signal) {
  const deadline = performance.now() + 90000;
  signal.throwIfAborted();
  let response = await gateway.recordings(query, signal);
  while (response?.state === 'preparing' && performance.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, 650));
    signal.throwIfAborted();
    response = await gateway.recordings(query, signal);
  }
  signal.throwIfAborted();
  return response;
}

export async function* questRecordings(gateway, signal) {
  const limit = 16;
  let before = '';
  do {
    const response = await preparedRecordingQuery(gateway, { query: 'list', before, limit }, signal);
    const entries = response?.entries;
    if (response?.v !== 1 || response.state !== 'ready' || !Array.isArray(entries) || entries.length > limit
      || entries.some((entry, i) => typeof entry?.id !== 'string' || !recordingId.test(entry.id)
        || typeof entry.exportable !== 'boolean'
        || entry.participant_csv !== undefined && typeof entry.participant_csv !== 'boolean'
        || (i > 0 ? entry.id >= entries[i - 1].id : before && entry.id >= before))
      || (response.next_before !== null && response.next_before !== entries.at(-1)?.id)
      || response.next_before === undefined) {
      throw Error('Ungültige Quest-Aufnahmeliste.');
    }
    for (const entry of entries) {
      signal.throwIfAborted();
      if (entry.exportable) yield { id: entry.id, participantCsv: entry.participant_csv === true };
    }
    before = response.next_before;
  } while (before);
}

function validRecovery(manifest) {
  const { recovery, disposition, participant_csv } = manifest;
  if (!participant_csv || !['interrupted-early', 'interrupted-completed'].includes(disposition)
    || recovery?.v !== 1 || Object.keys(recovery).length !== 4
    || !Number.isSafeInteger(recovery.checkpoint_ordinal) || recovery.checkpoint_ordinal < 1
    || typeof recovery.checkpoint_sha256 !== 'string' || !digest.test(recovery.checkpoint_sha256)
    || !recovery.stream_sha256 || Object.keys(recovery.stream_sha256).length !== 6) return false;
  return manifest.files.filter(file => file.name.endsWith('.jsonl'))
    .every(file => recovery.stream_sha256[file.name] === file.sha256);
}

export async function questRecording(gateway, id, signal, { includeSensorBundle = false } = {}) {
  if (!recordingId.test(id)) throw Error('Ungültige Quest-Aufnahme.');
  const response = await preparedRecordingQuery(gateway, { query: 'describe', id }, signal);
  const manifest = response?.manifest;
  if (response?.state !== 'ready' || manifest?.schema !== 'rusty.quest.recording_export.v1'
    || manifest.id !== id || manifest.device !== gateway.owner.device || manifest.serial !== gateway.owner.serial
    || manifest.saved !== true
    || !digest.test(manifest.manifest_id) || !Array.isArray(manifest.files) || manifest.files.length !== files.length
    || !files.every(name => manifest.files.some(file => file?.name === name))
    || !manifest.files.every(file => Number.isSafeInteger(file.size) && file.size >= 0 && digest.test(file.sha256))) {
    throw Error('Quest hat keine verifizierte, abgeschlossene Aufnahme geliefert.');
  }
  const validFiles = manifest.file_set_policy === 'core8-original-no-checkpoints-or-index'
    ? manifest.recovery === undefined
    : manifest.file_set_policy === 'core8-recovered-checkpoint' && validRecovery(manifest);
  if (!validFiles) throw Error('Quest hat keine verifizierte Aufnahmedatei geliefert.');
  const total = manifest.files.reduce((sum, file) => sum + file.size, 0);
  if (total !== manifest.total_bytes || total > maxBytes) throw Error('Ungültige Größe der Quest-Aufnahme.');
  const csv = manifest.participant_csv;
  if (csv !== undefined && (typeof csv?.name !== 'string' || csv.name !== csv.name.normalize('NFC')
    || !/^\p{L}{3}_(0|[1-9][0-9]{0,15})_[MFON]_(probe|main)[1-9][0-9]{0,15}\.csv$/u.test(csv.name)
    || !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(csv.session_id)
    || !Number.isSafeInteger(csv.size) || csv.size < 1 || csv.size > maxBytes || !digest.test(csv.sha256))) {
    throw Error('Ungültige Quest-CSV.');
  }
  const records = [];
  const selected = csv ? [csv] : manifest.files;
  if (csv && includeSensorBundle) selected.push(...manifest.files.filter(file => sensorFiles.has(file.name)));
  for (const file of selected) {
    const bytes = new Uint8Array(file.size);
    for (let offset = 0; offset < bytes.length; offset += 32768) {
      signal.throwIfAborted();
      const length = Math.min(32768, bytes.length - offset);
      const chunk = await gateway.recordings({ query: 'read', manifest_id: manifest.manifest_id, file: file.name, offset, length }, signal);
      if (chunk?.v !== 1 || chunk.state !== 'chunk' || chunk.manifest_id !== manifest.manifest_id
        || chunk.file !== file.name || chunk.offset !== offset || chunk.length !== length
        || chunk.sha256 !== file.sha256 || typeof chunk.data !== 'string'
        || chunk.data.length !== 4 * Math.ceil(length / 3)) throw Error('Unvollständiger Quest-Dateiabschnitt.');
      const raw = Uint8Array.from(atob(chunk.data), char => char.charCodeAt(0));
      if (raw.length !== length) throw Error('Ungültige Abschnittsgröße.');
      bytes.set(raw, offset);
    }
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
      byte => byte.toString(16).padStart(2, '0')).join('');
    if (hash !== file.sha256) throw Error('Quest-Datei stimmt nicht mit der Prüfsumme überein.');
    records.push({ filename: file.name, session_id: id, bytes });
  }
  signal.throwIfAborted();
  const participantFile = csv ? new File([records[0].bytes], csv.name, { type: 'text/csv' }) : null;
  if (participantFile && !includeSensorBundle) return participantFile;
  records.push({ filename: 'export-manifest.json', session_id: id, csv: JSON.stringify(manifest, null, 2) });
  const archive = createExportsZip(records);
  if (!includeSensorBundle) return archive;
  const filename = csv ? `${csv.name.slice(0, -4)}.zip` : `quest-${id}.zip`;
  const sensors = csv ? archive : createExportsZip(records.filter(({ filename }) =>
    sensorFiles.has(filename) || filename === 'export-manifest.json'));
  const bundle = new File([sensors], filename, { type: sensors.type });
  return { download: participantFile ?? archive, bundle, bundleVersion: 1 };
}
