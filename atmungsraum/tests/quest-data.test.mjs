import test from 'node:test';
import assert from 'node:assert/strict';
import { createCipheriv, createHash, createHmac } from 'node:crypto';
import { joinQuest } from '../quest-gateway.js';
import { ecgTrace, questRecording, questPrefixBatches } from '../quest-data.js';

const owner = { device: 'a'.repeat(32), serial: 'TEST-QUEST', epoch: '7', native_epoch: '8' };
const controller = 'b'.repeat(32);
const key = 'c'.repeat(64);
const build = 'd'.repeat(64);
const endpoint = '192.168.0.10';
const mac = value => createHmac('sha256', key).update(value).digest('hex');
const frame = { v: 1, rate: 130, unit: 'uV', age_ms: 100,
  samples: [['900719925474099100', 22], ['900719925481791407', -40], ['900719925510000000', 30]] };

test('only a verified enrolled connection decrypts the current native ECG waveform', async () => {
  let tamper = false;
  const fetcher = async (url, init) => {
    assert.equal(url, 'http://192.168.0.10:8766/v1/gateway');
    if (!init.body) return Response.json({ ...owner, v: 1, enabled: true, remote_available: true });
    const request = JSON.parse(init.body);
    const { action, nonce } = request;
    assert.equal(request.mac, mac(`RQG1|${controller}|7|${nonce}|${action}|${request.payload}`));
    assert.equal(request.payload, '{}');
    let value = { v: 1, wifi_snapshot: 1, control_mode: 'remote', recordings: 1, ecg_preview: 1,
      build: { v: 1, packaging_fingerprint: build }, p: 'IDLE', g: 0, r: 0, ms: 0, pending: false, rec: false,
      o: { v: 1, s: 'observed', e: '8', a: 5, i: '1', g: 0, r: 0 } };
    if (action === 'ecg-preview') {
      const iv = Buffer.alloc(12, 8);
      const cipher = createCipheriv('aes-256-gcm', Buffer.from(mac(`RQG1-PRIVATE-KEY|${controller}|${owner.device}|7`), 'hex'), iv);
      cipher.setAAD(Buffer.from(`RQG1-PRIVATE|${action}|${owner.device}|7|${tamper ? 'f'.repeat(32) : nonce}`));
      const data = Buffer.concat([cipher.update(JSON.stringify(frame)), cipher.final(), cipher.getAuthTag()]);
      value = { v: 1, iv: iv.toString('base64'), data: data.toString('base64') };
    } else assert.equal(action, 'status');
    const payload = JSON.stringify(value);
    return Response.json({ v: 1, device: owner.device, epoch: '7', nonce, action, payload,
      mac: mac(`RQG1-REPLY|${controller}|${owner.device}|7|${nonce}|${action}|${payload}`) });
  };
  const gateway = await joinQuest({ endpoint, serial: owner.serial, build, enrollment: { controller, key }, fetcher });
  assert.deepEqual(await gateway.ecg(), frame);
  assert.equal(ecgTrace(frame, 60).path.match(/M/g).length, 2);
  assert.equal(ecgTrace(frame, 1500), null);
  assert.equal(ecgTrace({ ...frame, samples: [...frame.samples].reverse() }, 60), null);
  tamper = true;
  await assert.rejects(() => gateway.ecg());
  gateway.close();
});

test('Quest export matches the observed run and every byte before making the ZIP', async () => {
  const id = 'utc-ns-00000000000000000042';
  const manifestId = 'e'.repeat(64);
  const names = ['manifest.json', 'final.json', 'events.jsonl', 'polar-acc.jsonl', 'polar-ecg.jsonl',
    'polar-hr-rr.jsonl', 'breath.jsonl', 'radius.jsonl'];
  const samples = Buffer.from('{"microvolts":42}\r\n');
  const hash = data => createHash('sha256').update(data).digest('hex');
  let bad = false;
  let participant = false;
  const csvName = 'Mül_35_N_probe2.csv';
  const gateway = { owner, recordings: async query => {
    if (query.query === 'describe') {
      assert.equal(query.id, id);
      return { state: 'ready', manifest: { schema: 'rusty.quest.recording_export.v1', id, saved: true,
        file_set_policy: 'core8-original-no-checkpoints-or-index', device: owner.device, serial: owner.serial,
        total_bytes: samples.length, manifest_id: manifestId,
        ...(participant ? { participant_csv: { name: csvName, size: samples.length,
          sha256: hash(samples), session_id: '01234567-89ab-4def-8abc-0123456789ab' } } : {}),
        files: names.map(name => ({ name, size: name === 'polar-ecg.jsonl' ? samples.length : 0,
          sha256: hash(name === 'polar-ecg.jsonl' ? samples : Buffer.alloc(0)) })) } };
    }
    assert.deepEqual(query, { query: 'read', manifest_id: manifestId,
      file: participant ? csvName : 'polar-ecg.jsonl', offset: 0, length: samples.length });
    return { v: 1, state: 'chunk', ...query, sha256: hash(samples),
      data: (bad ? Buffer.alloc(samples.length) : samples).toString('base64') };
  } };
  const signal = new AbortController().signal;
  const zip = Buffer.from(await (await questRecording(gateway, id, signal)).arrayBuffer());
  assert.ok(zip.includes(samples));
  assert.ok(zip.includes('export-manifest.json'));
  bad = true;
  await assert.rejects(() => questRecording(gateway, id, signal), /Prüfsumme/);
  bad = false;
  participant = true;
  const csv = await questRecording(gateway, id, signal);
  assert.equal(csv.name, csvName);
  assert.deepEqual(Buffer.from(await csv.arrayBuffer()), samples);
});

test('participant exports retain the CSV and both original sensor streams through chunked transfer', async () => {
  const id = 'utc-ns-00000000000000000043';
  const stem = 'Tes_30_N_2026-11-13_probe-5_r2';
  const csvName = `${stem}.csv`;
  const ecg = Buffer.from('{"sensor_time_ns":100,"microvolts":42}\n'.repeat(1300));
  const acc = Buffer.from('{"sensor_time_ns":100,"x_mg":1,"y_mg":2,"z_mg":3}\n'.repeat(900));
  const csv = Buffer.from('sensor,voltage\r\n100,42\r\n');
  const streams = new Map([
    ['manifest.json', Buffer.alloc(0)], ['final.json', Buffer.alloc(0)], ['events.jsonl', Buffer.alloc(0)],
    ['polar-acc.jsonl', acc], ['polar-ecg.jsonl', ecg], ['polar-hr-rr.jsonl', Buffer.alloc(0)],
    ['breath.jsonl', Buffer.alloc(0)], ['radius.jsonl', Buffer.alloc(0)],
  ]);
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const manifest = { schema: 'rusty.quest.recording_export.v1', id, saved: true,
    device: owner.device, serial: owner.serial, manifest_id: 'f'.repeat(64),
    file_set_policy: 'core8-original-no-checkpoints-or-index', total_bytes: ecg.length + acc.length,
    files: [...streams].map(([name, bytes]) => ({ name, size: bytes.length, sha256: hash(bytes) })),
    participant_csv: { name: csvName, size: csv.length, sha256: hash(csv),
      session_id: '01234567-89ab-4def-8abc-0123456789ab' } };
  let corrupt = false;
  const gateway = { owner, recordings: async query => {
    if (query.query === 'describe') return { state: 'ready', manifest };
    const bytes = query.file === csvName ? csv : streams.get(query.file);
    const block = Buffer.from(bytes.subarray(query.offset, query.offset + query.length));
    if (corrupt && query.file === 'polar-acc.jsonl' && query.offset > 0) block[0] ^= 0xff;
    return { v: 1, state: 'chunk', ...query, sha256: hash(bytes), data: block.toString('base64') };
  } };
  const signal = new AbortController().signal;
  const { download, bundle } = await questRecording(gateway, id, signal, { includeSensorBundle: true });
  assert.equal(download.name, csvName);
  assert.deepEqual(Buffer.from(await download.arrayBuffer()), csv);
  assert.equal(bundle.name, `${stem}.zip`);
  const zipped = Buffer.from(await bundle.arrayBuffer());
  const contents = new Map();
  for (let offset = 0; zipped.readUInt32LE(offset) === 0x04034b50;) {
    const nameLength = zipped.readUInt16LE(offset + 26);
    const length = zipped.readUInt32LE(offset + 18);
    const start = offset + 30 + nameLength;
    contents.set(zipped.toString('utf8', offset + 30, start), zipped.subarray(start, start + length));
    offset = start + length;
  }
  const readRecording = gateway.recordings;
  gateway.recordings = async query => {
    if (query.query === 'read') assert.equal(query.file, csvName);
    return readRecording(query);
  };
  for (const schemaVersion of [2, 3, 4]) {
    manifest.participant_csv.schema_version = schemaVersion;
    const unified = await questRecording(gateway, id, signal, { includeSensorBundle: true });
    assert.equal(unified.bundle.name, csvName);
    assert.equal(unified.bundle.type, 'text/csv');
    assert.deepEqual(Buffer.from(await unified.bundle.arrayBuffer()), csv);
  }
  const resumedOffsets = [];
  gateway.recordings = async query => {
    if (query.query === 'read') resumedOffsets.push(query.offset);
    return readRecording(query);
  };
  const prefixBytes = 19;
  const store = {
    readPrefix: async () => ({ session_id: manifest.participant_csv.session_id, name: csvName, offset: prefixBytes }),
    readPrefixBytes: async () => new Blob([csv.subarray(0, prefixBytes)]),
  };
  const finalized = await questRecording(gateway, id, signal, { includeSensorBundle: true, store });
  assert.deepEqual(Buffer.from(await finalized.bundle.arrayBuffer()), csv);
  assert.equal(resumedOffsets[0], prefixBytes);
  for (const damaged of [
    async () => new Blob([Buffer.alloc(prefixBytes, 0)]),
    async () => { throw Error('Missing staged range'); },
  ]) {
    resumedOffsets.length = 0;
    const repaired = await questRecording(gateway, id, signal, { includeSensorBundle: true,
      store: { ...store, readPrefixBytes: damaged } });
    assert.deepEqual(Buffer.from(await repaired.bundle.arrayBuffer()), csv);
    assert.equal(resumedOffsets.at(-1), 0);
    assert.equal(repaired.resetDriveUpload, true);
  }
  gateway.recordings = readRecording;
  delete manifest.participant_csv.schema_version;
  assert.deepEqual(contents.get(`${stem}_polar-ecg.jsonl`), ecg);
  assert.deepEqual(contents.get(`${stem}_polar-acc.jsonl`), acc);
  assert.deepEqual(JSON.parse(contents.get(`${stem}_export-manifest.json`)), manifest);
  const internal = manifest.files.find(file => file.name === 'radius.jsonl');
  internal.size = 140 * 1024 * 1024;
  manifest.total_bytes += internal.size;
  const largeInternal = await questRecording(gateway, id, signal, { includeSensorBundle: true });
  assert.equal(largeInternal.bundle.name, bundle.name);
  const sensor = manifest.files.find(file => file.name === 'polar-acc.jsonl');
  manifest.total_bytes += 128 * 1024 * 1024;
  sensor.size += 128 * 1024 * 1024;
  await assert.rejects(() => questRecording(gateway, id, signal, { includeSensorBundle: true }), /Ungültige Größe/);
  sensor.size -= 128 * 1024 * 1024;
  manifest.total_bytes -= 128 * 1024 * 1024 + internal.size;
  internal.size = 0;
  manifest.participant_csv.name = 'Tes_30_N_2026-02-30_probe-5_r2.csv';
  await assert.rejects(() => questRecording(gateway, id, signal), /Ungültige Quest-CSV/);
  manifest.participant_csv.name = csvName;
  manifest.file_set_policy = 'core8-recovered-checkpoint';
  manifest.disposition = 'interrupted-early';
  manifest.recovery = {
    v: 1, checkpoint_ordinal: 4, checkpoint_sha256: 'a'.repeat(64),
    stream_sha256: Object.fromEntries(manifest.files.filter(file => file.name.endsWith('.jsonl'))
      .map(file => [file.name, file.sha256]))
  };
  const recovered = await questRecording(gateway, id, signal, { includeSensorBundle: true });
  assert.deepEqual(Buffer.from(await recovered.download.arrayBuffer()), csv);
  assert.ok(Buffer.from(await recovered.bundle.arrayBuffer()).includes('core8-recovered-checkpoint'));
  const ecgHash = manifest.recovery.stream_sha256['polar-ecg.jsonl'];
  manifest.recovery.stream_sha256['polar-ecg.jsonl'] = '0'.repeat(64);
  await assert.rejects(() => questRecording(gateway, id, signal), /verifizierte Aufnahmedatei/);
  manifest.recovery.stream_sha256['polar-ecg.jsonl'] = ecgHash;
  corrupt = true;
  await assert.rejects(() => questRecording(gateway, id, signal, { includeSensorBundle: true }), /Prüfsumme/);
});
test('the live feed resumes at the committed offset and rejects a corrupted range before storage', async () => {
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const id = 'utc-ns-01725000000000000000';
  const session_id = 'd2d9ddfe-2717-40b9-9995-1bd3f48ff440';
  const name = 'Tes_30_N_2023-11-14_haupt-15_r1.csv';
  const bytes = Buffer.alloc(320 * 1024, 37);
  const descriptors = { v: 1, state: 'live', schema_version: 4, id, session_id, name, size: bytes.length,
    sha256: hash(bytes), checkpoint: 3 };
  const offsets = [];
  let staged, corrupt = false;
  const store = {
    readPrefix: async () => staged,
    appendPrefix: async (prefix, offset, parts) => {
      assert.equal(offset, staged?.offset ?? 0);
      staged = { ...prefix, offset: offset + parts.reduce((size, part) => size + part.size, 0) };
    },
  };
  const gateway = { recordings: async ({ query, offset, length }) => {
    if (query === 'live-describe') return descriptors;
    offsets.push(offset);
    const data = bytes.subarray(offset, offset + length);
    return { v: 1, state: 'live-chunk', id, session_id, offset, length,
      sha256: corrupt ? '0'.repeat(64) : hash(data), data: data.toString('base64') };
  } };
  const signal = new AbortController().signal;
  const stage = async () => {
    for await (const prefix of questPrefixBatches(gateway, store, signal)) return prefix;
  };
  const first = await stage();
  assert.equal(first.offset, 128 * 1024);
  corrupt = true;
  await assert.rejects(stage, /Prüfsumme/);
  assert.equal(staged.offset, first.offset);
  corrupt = false;
  const restarted = await stage();
  assert.equal(restarted.offset, 256 * 1024);
  assert.equal(offsets[4], first.offset);
  assert.equal(offsets[5], first.offset);
});

test('catch-up drains bounded batches from one published checkpoint and stops on cancellation', async () => {
  const hash = data => createHash('sha256').update(data).digest('hex');
  const bytes = Buffer.alloc(650 * 1024, 9);
  const id = 'utc-ns-01725000000000000000';
  const session_id = 'd2d9ddfe-2717-40b9-9995-1bd3f48ff440';
  const descriptor = { v: 1, state: 'live', schema_version: 4, id, session_id,
    name: 'Tes_30_N_2023-11-14_haupt-15_r1.csv', size: bytes.length, checkpoint: 8, sha256: hash(bytes) };
  const stages = [];
  let saved, describes = 0;
  const gateway = { async recordings({ query, offset, length }) {
    if (query === 'live-describe') { describes++; return descriptor; }
    const data = bytes.subarray(offset, offset + length);
    return { v: 1, state: 'live-chunk', id, session_id, offset, length,
      sha256: hash(data), data: data.toString('base64') };
  } };
  const store = {
    readPrefix: async () => saved,
    async appendPrefix(prefix, start, parts) {
      assert.ok(parts.length <= 4);
      saved = { ...prefix, offset: start + parts.reduce((sum, part) => sum + part.size, 0) };
      stages.push(saved.offset);
    },
  };
  const canceled = new AbortController();
  await assert.rejects(async () => {
    for await (const prefix of questPrefixBatches(gateway, store, canceled.signal)) {
      assert.equal(prefix.offset, 128 * 1024);
      canceled.abort();
    }
  }, { name: 'AbortError' });
  for await (const prefix of questPrefixBatches(gateway, store, new AbortController().signal)) {
    assert.equal(prefix.offset, saved.offset);
  }
  assert.deepEqual(stages, [128, 256, 384, 512, 640, 650].map(size => size * 1024));
  assert.equal(describes, 2);
});
