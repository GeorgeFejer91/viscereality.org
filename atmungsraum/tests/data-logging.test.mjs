import test from 'node:test';
import assert from 'node:assert/strict';
import { participantIdentity, sessionFilename, SessionLog } from '../data-logging.js';
import { MockECG } from '../mock-ecg.js';
import { readFile } from 'node:fs/promises';

test('All filenames bind the same normalized identity to registered conditions and positive run numbers', () => {
  const patient = { surnamePrefix: ' FeJ ', age: '35', gender: 'male' };
  assert.deepEqual(participantIdentity(patient), { name: 'Fej', age: 35, genderCode: 'M', id: 'Fej_35_M' });
  assert.equal(sessionFilename(patient, 'probe', 1), 'Fej_35_M_probe1.csv');
  assert.equal(sessionFilename(patient, 'haupt', 2), 'Fej_35_M_main2.csv');
  assert.equal(sessionFilename({ ...patient, surnamePrefix: 'ßab' }, 'haupt', 1), 'ßab_35_M_main1.csv');
  for (const [gender, code] of [['male','M'], ['female','F'], ['other','O'], ['prefer-not-to-say','N']]) {
    assert.equal(sessionFilename({ ...patient, gender }, 'probe', 1), `Fej_35_${code}_probe1.csv`);
  }
  for (const bad of [{ surnamePrefix: '../' }, { surnamePrefix: '=1+' }, { surnamePrefix: 'Fejer' }, { age: '' }, { age: '-1' }, { age: '3.5' }, { gender: '' }, { gender: '__proto__' }]) {
    assert.throws(() => sessionFilename({ ...patient, ...bad }, 'probe', 1));
  }
  assert.throws(() => sessionFilename(patient, 'unknown', 1));
  assert.throws(() => sessionFilename(patient, '__proto__', 1));
  assert.throws(() => sessionFilename(patient, 'probe', 0));
});

test('CSV samples share identity, monotonic wall time, exact source values and pause-aware protocol phases', () => {
  let now = 1000;
  const signal = { rate: 4, epoch: 0, manifest: { source: 'mock-neurokit2', unit: 'a.u.', sha256: 'test-hash' },
    samples: Float32Array.from([.1, .2, .3, .4, .5, .6, .7, .8]), indexAt: t => Math.floor(t * 4 / 1000) };
  const log = new SessionLog({ patient: { surnamePrefix: 'Fej', age: 35, gender: 'male' },
    selection: { branch: 'haupt', value: '10', minutes: 10, baselineMinutes: 5, boxMinutes: 5 },
    runIndex: 1, sessionId: 'run-1', signal, now: () => now, utc: () => 0,
    protocol: { sample_rate: 4, total_frames: 2400, scenario_id: 'haupt-10', scenario_sha256: 'scenario-hash',
      events: [{ id: 'baseline', offset_frames: 0, duration_frames: 1200 }, { id: 'instruction', offset_frames: 1200, duration_frames: 160 }] } });
  now = 1500; log.mark('paused', .5);
  now = 2000; log.mark('playing', .5); log.markSensor('disconnected');
  now = 2500; log.stop();
  now = 3500; // A delayed storage reservation must not extend an aborted recording.
  const csv = log.finish('aborted', 'user-abort').csv;
  const [headers, ...lines] = csv.trim().split('\r\n').map(line => line.split(','));
  const rows = lines.map(cells => Object.fromEntries(headers.map((key,i) => [key,cells[i]])));
  assert.equal(rows.length, 6);
  assert.deepEqual(rows.map(r => Number(r.sample_index)), [0,1,2,3,4,5]);
  assert.deepEqual(rows.map(r => Number(r.elapsed_ms)), [0,250,500,750,1000,1250]);
  assert.deepEqual(rows.map(r => Number(r.audio_elapsed_ms)), [0,250,500,500,500,750]);
  assert.deepEqual(rows.map(r => r.playback_state), ['playing','playing','paused','paused','playing','playing']);
  assert.deepEqual(rows.map(r => Number(r.source_sample_index)), [4,5,6,7,0,1]);
  assert.deepEqual(rows.map(r => Number(r.source_loop)), [0,0,0,0,1,1]);
  assert.equal(rows[0].ecg_value, signal.samples[4].toFixed(6));
  assert.equal(rows[4].ecg_value, '', 'missing signal must never be exported as live samples');
  assert.equal(rows[4].signal_state, 'disconnected');
  assert.equal(rows[0].timestamp_utc, '1970-01-01T00:00:00.000Z');
  for (const row of rows) {
    assert.equal(row.participant_id, 'Fej_35_M');
    assert.equal(row.condition, 'main');
    assert.equal(row.run_index, '1');
    assert.equal(row.scenario_id, 'haupt-10');
    assert.equal(row.phase, 'baseline');
    assert.equal(row.session_status, 'aborted');
    assert.equal(row.signal_source, 'mock-neurokit2');
  }
  assert.equal(log.finish('completed', '').csv, csv, 'finishing twice must retain the original export');
});

test('The shipped NeuroKit trace passes its integrity contract; corrupted or missing samples cannot become ECG-ready', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async url => new Response(await readFile(url));
    const signal = new MockECG();
    await signal.prepare();
    assert.equal(signal.rate, 256);
    assert.equal(signal.samples.length, 460800);
    assert.equal(signal.manifest.neurokit2_version, '0.2.13');
    assert.equal(signal.manifest.source, 'mock-neurokit2');
    assert.equal(signal.sampleAt(signal.samples.length + 3), signal.samples[3]);
    globalThis.fetch = async url => String(url).endsWith('.f32') ? new Response(new Uint8Array(8)) : new Response(await readFile(url));
    const corrupted = new MockECG();
    await assert.rejects(corrupted.prepare(), /integrity/);
    assert.equal(corrupted.samples, undefined);
    globalThis.fetch = async () => new Response('', { status: 404 });
    await assert.rejects(new MockECG().prepare(), /unavailable/);
  } finally { globalThis.fetch = original; }
});

test('Starting and stopping recordings never restarts the continuous background ECG clock', async () => {
  const signal = Object.assign(new MockECG(), { epoch: 100, rate: 4,
    samples: Float32Array.from({ length: 80 }, (_, i) => i / 100),
    manifest: { source: 'mock-neurokit2', unit: 'a.u.', sha256: 'continuous-source' } });
  const samples = signal.samples;
  let now = 1100;
  const record = runIndex => new SessionLog({
    patient: { surnamePrefix: 'Fej', age: 35, gender: 'male' },
    selection: { branch: 'probe', value: 'anleitung' }, runIndex,
    sessionId: `background-${runIndex}`, signal, now: () => now, utc: () => 0,
    protocol: { sample_rate: 4, total_frames: 40, scenario_id: 'probe-anleitung',
      events: [{ id: 'instruction', offset_frames: 0, duration_frames: 40 }] },
  });
  const first = record(1);
  assert.equal(first.sourceStart, 4, 'the stream has advanced before Start');
  now = 1600;
  first.stop();
  const firstCSV = first.finish('aborted', 'user-abort').csv;
  now = 5100; // Menus/idle time advances the background source without recording.
  await signal.prepare();
  const second = record(2);
  assert.equal(second.sourceStart, 20, 'the next Start records the current stream position');
  assert.equal(signal.epoch, 100);
  assert.equal(signal.samples, samples);
  now = 5600;
  const secondCSV = second.finish('aborted', 'user-abort').csv;
  const rows = csv => {
    const [header, ...raw] = csv.trim().split('\r\n').map(line => line.split(','));
    return raw.map(row => Object.fromEntries(header.map((key, i) => [key, row[i]])));
  };
  assert.deepEqual(rows(firstCSV).map(row => Number(row.source_sample_index)), [4, 5]);
  assert.deepEqual(rows(secondCSV).map(row => Number(row.source_sample_index)), [20, 21]);
  assert.equal(first.finish('aborted', 'user-abort').csv, firstCSV, 'stopped exports exclude idle time');
});

test('A pause during scheduling lead preserves waiting time on pause and resume, and checkpoints recover the same samples', () => {
  let now = 0;
  const signal = { rate: 4, samples: new Float32Array(32), indexAt: () => 0,
    manifest: { source: 'mock-neurokit2', unit: 'a.u.', sha256: 'seed' } };
  const log = new SessionLog({ patient: { surnamePrefix: 'Fej', age: 35, gender: 'male' },
    selection: { branch: 'probe', value: 'anleitung' }, runIndex: 1, sessionId: 'lead-run', signal, now: () => now, utc: () => 0,
    audioStartDelayMs: 1000, protocol: { sample_rate: 4, total_frames: 16, scenario_id: 'probe-anleitung',
      events: [{ id: 'instruction', offset_frames: 0, duration_frames: 16 }] } });
  now = 250; log.mark('paused', -.75);
  now = 1250; log.mark('playing', -.75);
  now = 2250; log.stop();
  const saved = log.checkpoint();
  const result = log.finish('aborted', 'page-reload-or-interrupted');
  assert.equal(SessionLog.recover(saved, signal).csv, result.csv);
  const [header, ...raw] = result.csv.trim().split('\r\n').map(x => x.split(','));
  const rows = raw.map(row => Object.fromEntries(header.map((h,i) => [h,row[i]])));
  assert.deepEqual(rows.slice(0,8).map(r=>r.phase), Array(8).fill('waiting'));
  assert.equal(rows[8].phase, 'instruction');
  assert.ok(rows.filter(r=>r.playback_state==='paused').every(r=>r.audio_elapsed_ms==='0.000'));
});
