import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createQuestDrive, deliverQuestBundle, deliverQuestPrefix } from '../quest-drive.js';
import { datasetDestination, drivePairIdentity } from '../quest-drive-datasets.js';

const signal = new AbortController().signal;
const folderId = 'private-folder';
const account = { emailAddress: 'research@example.invalid', permissionId: 'owner-1' };
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

function simulatedDrive() {
  const files = new Map();
  const sessions = new Map();
  let reservations = 0;
  let interrupt = false;
  let corrupt = false;
  let foreignSession = false;
  let sharedFolder = false;
  let quota = false;
  let loseFolder = false;
  const fetcher = async (input, options) => {
    const url = new URL(input);
    assert.equal(url.origin, 'https://www.googleapis.com');
    assert.equal(options.headers.get('Authorization'), 'Bearer fixture-token');
    if (url.pathname === '/drive/v3/about') return Response.json({ user: account });
    if (url.pathname.endsWith('/generateIds')) return Response.json({ ids: [`file-${++reservations}`] });
    if (url.pathname === '/drive/v3/files' && options.method === 'GET') {
      const [, parent, name] = /^'([^']+)' in parents and name = '([^']+)'/.exec(url.searchParams.get('q'));
      return Response.json({ files: [...files.values()].filter(file => file.parents?.[0] === parent
        && file.name === name).map(file => ({ id: file.id })) });
    }
    if (url.pathname === `/drive/v3/files/${folderId}`) {
      return Response.json({ id: folderId, mimeType: 'application/vnd.google-apps.folder',
        ownedByMe: true, shared: sharedFolder, trashed: false });
    }
    if (url.pathname.startsWith('/drive/v3/files/')) {
      const found = files.get(url.pathname.split('/').at(-1));
      return found ? Response.json(found) : new Response(null, { status: 404 });
    }
    if (options.method === 'POST') {
      const metadata = JSON.parse(options.body);
      assert.ok(metadata.parents[0] === folderId || files.get(metadata.parents[0])?.mimeType === 'application/vnd.google-apps.folder');
      if (quota) { quota = false; return new Response(null, { status: 403 }); }
      if (files.has(metadata.id)) return new Response(null, { status: 409 });
      if (metadata.mimeType === 'application/vnd.google-apps.folder') {
        files.set(metadata.id, { ...metadata, ownedByMe: true, shared: false, trashed: false });
        if (loseFolder) { loseFolder = false; throw new TypeError('lost folder response'); }
        return Response.json({ id: metadata.id });
      }
      sessions.set(metadata.id, metadata);
      const origin = foreignSession ? 'https://unexpected.example.invalid' : url.origin;
      return new Response(null, { headers: { Location: `${origin}/upload/drive/v3/files?upload_id=${metadata.id}` } });
    }
    assert.equal(options.method, 'PUT');
    const id = url.searchParams.get('upload_id');
    const metadata = sessions.get(id);
    const body = Buffer.from(await options.body.arrayBuffer());
    files.set(id, { ...metadata, ownedByMe: true, shared: false, trashed: false,
      size: String(body.length), sha256Checksum: corrupt ? '0'.repeat(64) : hash(body) });
    if (interrupt) { interrupt = false; throw new TypeError('lost upload response'); }
    return Response.json({ id });
  };
  return { files, fetcher, interruptNextUpload: () => { interrupt = true; },
    corruptChecksum: () => { corrupt = true; }, foreignSession: () => { foreignSession = true; },
    shareFolder: () => { sharedFolder = true; }, quotaOnce: () => { quota = true; },
    interruptFolder: () => { loseFolder = true; } };
}

function localReceipts(rows = new Map()) {
  return { rows, readDelivery: async id => rows.get(id),
    saveDelivery: async (id, receipt) => rows.set(id, structuredClone(receipt)) };
}

test('each Quest dataset gets one folder and sibling files through an ambiguous folder creation and reload', async () => {
  const remote = simulatedDrive();
  const drive = createQuestDrive('fixture-token', remote.fetcher);
  let store = localReceipts();
  const pair = { id: '1'.repeat(24), source: 'usb-hardware-pair' };
  const start = { utc: '2026-11-18T10:00:00.123456Z', source: 'quest-official-start' };
  const observed = { utc: '2026-11-18T10:00:01.010Z', source: 'timeapi.io', round_trip_ms: 120 };
  const ids = ['utc-ns-01794996000000000000', 'utc-ns-01794996000000000001'];
  const folder = id => datasetDestination({ drive, store, root: folderId, pair, id, start, observed, signal });
  remote.interruptFolder();
  await assert.rejects(() => folder(ids[0]), /lost folder response/);
  store = localReceipts(new Map(store.rows));
  const first = await folder(ids[0]);
  const second = await folder(ids[1]);
  assert.notEqual(first, second);
  assert.equal(await folder(ids[0]), first);
  const parent = remote.files.get(first).parents[0];
  assert.deepEqual(remote.files.get(second).parents, [parent]);
  assert.equal(remote.files.get(parent).name, 'Pair_' + pair.id);
  assert.equal(remote.files.get(first).appProperties.phoneObservedAtUtc, observed.utc);
  const file = new File(['fixture only'], 'answers.json', { type: 'application/json' });
  for (const id of [ids[0] + '/first-file', ids[0] + '/questionnaire']) {
    await deliverQuestBundle({ drive, store, folderId: first, id, file, signal, stamp: observed });
    const receipt = store.rows.get(id);
    assert.deepEqual(remote.files.get(receipt.driveId).parents, [first]);
    assert.equal(remote.files.get(receipt.driveId).appProperties.startedAtUtc, observed.utc);
  }
  remote.files.get(first).parents = [second];
  await assert.rejects(() => folder(ids[0]), /Ordner stimmt nicht/);
});

test('hardware pair aliases are only used for their bound Quest installation', async () => {
  const id = 'a'.repeat(24);
  const values = new Map([['atmungsraum-drive-hardware-pairs', JSON.stringify([{ device: 'quest-a', serial: 'a', id }])]]);
  const storage = { getItem: key => values.get(key), setItem: (key, value) => values.set(key, value) };
  assert.deepEqual(await drivePairIdentity({ device: 'quest-a', serial: 'a' }, storage), { id, source: 'usb-hardware-pair' });
  const other = { device: 'quest-b', serial: 'b' };
  const first = await drivePairIdentity(other, storage);
  assert.equal(first.source, 'app-installation-pair');
  assert.notEqual(first.id, id);
  assert.deepEqual(await drivePairIdentity(other, storage), first);
});

test('a lost Drive response leaves one reserved ID and reconciles its exact bytes after restart', async () => {
  const remote = simulatedDrive();
  const firstStore = localReceipts();
  const drive = createQuestDrive('fixture-token', remote.fetcher);
  const file = new File([Buffer.alloc(6 * 1024 * 1024, 17)], 'Tes_30_N_probe1.zip', { type: 'application/zip' });
  const request = { drive, store: firstStore, folderId, id: 'record-1', file, signal };
  remote.interruptNextUpload();
  await assert.rejects(() => deliverQuestBundle(request), /lost upload response/);
  const pending = firstStore.rows.get('record-1');
  assert.equal(pending.driveId, 'file-1');
  assert.equal(pending.verified, false);
  const secondStore = localReceipts(new Map(firstStore.rows));
  await deliverQuestBundle({ ...request, store: secondStore });
  assert.equal(secondStore.rows.get('record-1').verified, true);
  assert.equal(remote.files.size, 1);
  assert.equal(remote.files.get('file-1').sha256Checksum, pending.sha256);
});

test('a wrong Drive checksum leaves the recording pending', async () => {
  const remote = simulatedDrive();
  remote.corruptChecksum();
  const store = localReceipts();
  const drive = createQuestDrive('fixture-token', remote.fetcher);
  const file = new File(['synthetic recording'], 'Tes_30_N_probe1.zip', { type: 'application/zip' });
  await assert.rejects(() => deliverQuestBundle({ drive, store, folderId, id: 'record-1', file, signal }), /Prüfsumme/);
  assert.equal(store.rows.get('record-1').verified, false);
});

test('an upload endpoint outside Google never receives an access token or file', async () => {
  const remote = simulatedDrive();
  remote.foreignSession();
  const store = localReceipts();
  const drive = createQuestDrive('fixture-token', remote.fetcher);
  const file = new File(['synthetic recording'], 'Tes_30_N_probe1.zip', { type: 'application/zip' });
  await assert.rejects(() => deliverQuestBundle({ drive, store, folderId, id: 'record-1', file, signal }), /Upload-Adresse/);
  assert.equal(remote.files.size, 0);
  assert.equal(store.rows.get('record-1').verified, false);
});

test('a quota rejection retains the reserved ID for a successful retry', async () => {
  const remote = simulatedDrive();
  const store = localReceipts();
  const drive = createQuestDrive('fixture-token', remote.fetcher);
  const file = new File(['synthetic recording'], 'Tes_30_N_probe1.zip', { type: 'application/zip' });
  const request = { drive, store, folderId, id: 'record-1', file, signal };
  remote.quotaOnce();
  await assert.rejects(() => deliverQuestBundle(request), { status: 403 });
  assert.equal(store.rows.get('record-1').verified, false);
  assert.equal(remote.files.size, 0);
  await deliverQuestBundle(request);
  assert.equal(store.rows.get('record-1').driveId, 'file-1');
  assert.equal(store.rows.get('record-1').verified, true);
});

test('a folder that becomes shared is refused before a new upload', async () => {
  const remote = simulatedDrive();
  const drive = createQuestDrive('fixture-token', remote.fetcher);
  await drive.ensureFolder(folderId, signal);
  remote.shareFolder();
  await assert.rejects(() => drive.ensureFolder(folderId, signal), /private Google-Drive-Speicherort/);
});

test('forty finalized recording bundles stay unique through repeated lost responses and restarts', async () => {
  const remote = simulatedDrive();
  const drive = createQuestDrive('fixture-token', remote.fetcher);
  let store = localReceipts();
  for (let run = 1; run <= 40; run++) {
    const id = `run-${run}`;
    const bytes = Buffer.alloc(512 * 1024, run);
    const file = new File([bytes], `Tes_30_N_probe${run}.zip`, { type: 'application/zip' });
    if (run % 5 === 0) {
      remote.interruptNextUpload();
      await assert.rejects(() => deliverQuestBundle({ drive, store, folderId, id, file, signal }), /lost upload response/);
      store = localReceipts(new Map(store.rows));
    }
    await deliverQuestBundle({ drive, store, folderId, id, file, signal });
    const receipt = store.rows.get(id);
    assert.equal(receipt.verified, true);
    assert.equal(remote.files.get(receipt.driveId).sha256Checksum, hash(bytes));
  }
  assert.equal(remote.files.size, 40);
  assert.equal(new Set([...store.rows.values()].map(receipt => receipt.driveId)).size, 40);
});

function partialDrive(file) {
  const chunk = 256 * 1024;
  const files = new Map(), sessions = new Map(), ranges = [];
  let reservations = 0, starts = 0, lose = false, invalidRange = false;
  const incomplete = length => new Response(null, { headers: {
    'X-HTTP-Status-Code-Override': '308',
    ...(length ? { Range: invalidRange ? 'bytes=12-14' : 'bytes=0-' + (length - 1) } : {}),
  } });
  const drive = createQuestDrive('fixture-token', async (input, options) => {
    const url = new URL(input);
    assert.equal(url.origin, 'https://www.googleapis.com');
    assert.equal(options.redirect, 'error');
    if (url.pathname.endsWith('/generateIds')) return Response.json({ ids: ['file-' + ++reservations] });
    if (url.pathname === '/drive/v3/files/file-1') {
      return files.has('file-1') ? Response.json(files.get('file-1')) : new Response(null, { status: 404 });
    }
    if (options.method === 'POST') {
      const metadata = JSON.parse(options.body);
      assert.deepEqual(metadata, { id: 'file-1', name: file.name, mimeType: file.type, parents: [folderId] });
      const id = 'session-' + ++starts;
      sessions.set(id, Buffer.alloc(0));
      return new Response(null, { headers: { Location: url.origin + url.pathname + '?upload_id=' + id } });
    }
    assert.equal(options.method, 'PUT');
    assert.equal(options.headers.get('X-GUploader-No-308'), 'yes');
    const id = url.searchParams.get('upload_id'), range = options.headers.get('Content-Range');
    ranges.push(range);
    if (!sessions.has(id)) return new Response(null, { status: 404 });
    const current = sessions.get(id);
    if (range.startsWith('bytes */')) return incomplete(current.length);
    const [, start, end, total] = /^bytes (\d+)-(\d+)\/(\*|\d+)$/.exec(range);
    const payload = Buffer.from(await options.body.arrayBuffer());
    assert.equal(Number(start), current.length);
    assert.equal(Number(end) - Number(start) + 1, payload.length);
    if (total === '*' || Number(end) + 1 < Number(total)) assert.equal(payload.length % chunk, 0);
    const accepted = Buffer.concat([current, lose ? payload.subarray(0, chunk) : payload]);
    sessions.set(id, accepted);
    if (lose) { lose = false; throw new TypeError('lost chunk response'); }
    if (total === '*' || accepted.length < Number(total)) return incomplete(accepted.length);
    assert.equal(accepted.length, Number(total));
    files.set('file-1', { id: 'file-1', name: file.name, mimeType: file.type, parents: [folderId],
      ownedByMe: true, shared: false, trashed: false, size: String(accepted.length),
      sha256Checksum: hash(accepted) });
    return Response.json({ id: 'file-1' });
  });
  const request = { drive, id: 'run-1', folderId, signal };
  return { ranges, files, loseResponseAfterOneChunk: () => { lose = true; },
    expireSessions: () => sessions.clear(), invalidRange: () => { invalidRange = true; },
    prefix: (store, available, read = async (start, end) => file.slice(start, end)) =>
      deliverQuestPrefix({ ...request, store, name: file.name, type: file.type, available, read }),
    finish: (store, resetDriveUpload) => deliverQuestBundle({ ...request, store, file, resetDriveUpload }) };
}

const chunk = 256 * 1024;
const recording = size => new File([Buffer.alloc(size, 19)], 'Tes_30_N_2026-11-13_haupt-15_r1.csv', { type: 'text/csv' });

test('a growing recording keeps a tail and resumes at the acknowledged byte after reload', async () => {
  const remote = partialDrive(recording(chunk * 4));
  const first = localReceipts();
  await remote.prefix(first, chunk);
  assert.equal(first.rows.size, 0);
  remote.loseResponseAfterOneChunk();
  await assert.rejects(() => remote.prefix(first, chunk * 2 + 1), /lost chunk response/);
  assert.equal(remote.files.size, 0);
  const restarted = localReceipts(new Map(first.rows));
  await remote.prefix(restarted, chunk * 3 + 1);
  assert.equal(restarted.rows.get('run-1').offset, chunk * 3);
  await remote.finish(restarted);
  assert.deepEqual(remote.ranges, [
    'bytes 0-524287/*', 'bytes */*', 'bytes 262144-786431/*',
    'bytes 786432-1048575/1048576',
  ]);
  const receipt = restarted.rows.get('run-1');
  assert.equal(receipt.verified, true);
  assert.equal(receipt.uploadUrl, undefined);
  assert.equal(remote.files.size, 1);
  assert.equal(remote.files.get(receipt.driveId).sha256Checksum, hash(Buffer.alloc(chunk * 4, 19)));
});

test('an expired Drive upload session restarts using the same reserved file ID', async () => {
  const remote = partialDrive(recording(chunk + 1));
  const store = localReceipts();
  await remote.prefix(store, chunk + 1);
  remote.expireSessions();
  await assert.rejects(() => remote.finish(store), /abgelaufen/);
  assert.equal(store.rows.get('run-1').offset, 0);
  assert.equal(store.rows.get('run-1').verified, false);
  await remote.finish(store);
  assert.deepEqual(remote.ranges, ['bytes 0-262143/*', 'bytes 262144-262144/262145', 'bytes 0-262144/262145']);
  assert.equal(store.rows.get('run-1').driveId, 'file-1');
  assert.equal(remote.files.size, 1);
});

test('a repaired Quest prefix replaces a suspect Drive session under the same file ID', async () => {
  const remote = partialDrive(recording(chunk + 1));
  const store = localReceipts();
  await remote.prefix(store, chunk + 1, async (start, end) => new Blob([Buffer.alloc(end - start, 6)]));
  const original = store.rows.get('run-1');
  assert.equal(original.offset, chunk);
  await remote.finish(store, true);
  const final = store.rows.get('run-1');
  assert.equal(final.driveId, original.driveId);
  assert.equal(final.verified, true);
  assert.equal(remote.files.size, 1);
  assert.deepEqual(remote.ranges, ['bytes 0-262143/*', 'bytes 0-262144/262145']);
});

test('invalid Drive acknowledgements leave the receipt unverified', async () => {
  const remote = partialDrive(recording(chunk + 1));
  const store = localReceipts();
  remote.invalidRange();
  await assert.rejects(() => remote.prefix(store, chunk + 1), /Bytebereich/);
  assert.equal(store.rows.get('run-1').offset, 0);
  assert.equal(store.rows.get('run-1').verified, false);
});

test('finalized CSV delivery sends a large local file in bounded requests', async () => {
  const remote = partialDrive(recording(5 * chunk + 1));
  const store = localReceipts();
  await remote.finish(store);
  assert.deepEqual(remote.ranges, ['bytes 0-1048575/1310721', 'bytes 1048576-1310720/1310721']);
  assert.equal(store.rows.get('run-1').verified, true);
});
