import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createQuestDrive, deliverQuestBundle } from '../quest-drive.js';

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
  const fetcher = async (input, options) => {
    const url = new URL(input);
    assert.equal(url.origin, 'https://www.googleapis.com');
    assert.equal(options.headers.get('Authorization'), 'Bearer fixture-token');
    if (url.pathname === '/drive/v3/about') return Response.json({ user: account });
    if (url.pathname.endsWith('/generateIds')) return Response.json({ ids: [`file-${++reservations}`] });
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
      assert.deepEqual(metadata.parents, [folderId]);
      if (quota) { quota = false; return new Response(null, { status: 403 }); }
      if (files.has(metadata.id)) return new Response(null, { status: 409 });
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
    shareFolder: () => { sharedFolder = true; }, quotaOnce: () => { quota = true; } };
}

function localReceipts(rows = new Map()) {
  return { rows, readDelivery: async id => rows.get(id),
    saveDelivery: async (id, receipt) => rows.set(id, structuredClone(receipt)) };
}

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
