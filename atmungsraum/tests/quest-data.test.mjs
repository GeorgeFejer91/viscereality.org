import test from 'node:test';
import assert from 'node:assert/strict';
import { createCipheriv, createHash, createHmac } from 'node:crypto';
import { joinQuest } from '../quest-gateway.js';
import { ecgTrace, questRecording } from '../quest-data.js';

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
