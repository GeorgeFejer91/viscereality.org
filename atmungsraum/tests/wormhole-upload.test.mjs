import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

const endpoint = 'https://relay.example.workers.dev';
const token = 'A'.repeat(43), writer = Buffer.alloc(16, 7).toString('base64');
const sha1 = bytes => createHash('sha1').update(bytes).digest('hex');
// Independent small reader for the BEP3 metadata emitted by the client.
function torrent(bytes) {
  let at = 0;
  const read = () => {
    const c = bytes[at++];
    if (c === 105) { const end = bytes.indexOf(101, at); const n = Number(bytes.subarray(at, end).toString()); at = end + 1; return n; }
    if (c === 100 || c === 108) {
      const value = c === 100 ? {} : [];
      while (bytes[at] !== 101) {
        if (c === 100) { const key = read().toString(); value[key] = read(); } else value.push(read());
      }
      at++; return value;
    }
    at--; const colon = bytes.indexOf(58, at), size = Number(bytes.subarray(at, colon).toString());
    at = colon + 1; const value = bytes.subarray(at, at + size); at += size; return value;
  };
  return read();
}

async function client() {
  const module = await import('../wormhole-upload.js').catch(error => {
    if (error.code !== 'ERR_MODULE_NOT_FOUND') throw error;
    return {};
  });
  assert.equal(typeof module.uploadWormholeZip, 'function', 'automatic upload client is missing');
  return module.uploadWormholeZip;
}

function service(t, { failFinish = false } = {}) {
  const calls = [], chunks = [];
  let roomData, update;
  t.mock.method(globalThis, 'fetch', async (target, init) => {
    const path = new URL(target).pathname;
    assert.equal(new URL(target).origin, endpoint);
    assert.equal(init.headers.Authorization, `Bearer ${token}`);
    assert.equal(init.redirect, 'error');
    calls.push(path);
    if (path === '/wormhole/api/room') {
      roomData = JSON.parse(init.body);
      return Response.json({ id: 'room01', writerToken: writer, lifetime: 86400, maxDownloads: 100 });
    }
    if (path === '/wormhole/api/room/room01') {
      update = JSON.parse(init.body);
      return new Response(null, { status: 204 });
    }
    if (path.endsWith('/auth-upload')) {
      assert.ok(JSON.parse(init.body).numTokens <= 10);
      return Response.json(Array.from({ length: JSON.parse(init.body).numTokens }, () => ({
        uploadUrl: 'https://pod-001.backblazeb2.com/b2api/v3/b2_upload_file', authorizationToken: 'b2-test-token',
      })));
    }
    if (path === '/wormhole/chunk') {
      const bytes = Buffer.from(await new Response(init.body).arrayBuffer());
      assert.ok(bytes.length <= 8 * 1024 * 1024);
      assert.equal(init.headers['X-Bz-Content-Sha1'], sha1(bytes));
      assert.equal(init.headers['X-Bz-File-Name'], `room01/${chunks.length}`);
      chunks.push(bytes);
      return Response.json({ fileId: `chunk-${chunks.length}` });
    }
    if (path.endsWith('/finish-upload') && failFinish) return new Response('failed', { status: 503 });
    return new Response(null, { status: 204 });
  });
  return { calls, chunks, get roomData() { return roomData; }, get update() { return update; } };
}

test('automatic upload encrypts every ZIP byte, sends bounded chunks and returns a link only after finish acknowledgement', async t => {
  const upload = await client(), remote = service(t);
  const original = Buffer.alloc(9 * 1024 * 1024, 65);
  original.write('SYNTHETIC-PRIVATE-CSV-MARKER');
  const file = new File([original], 'Atmungsraum-Daten.zip', { type: 'application/zip' });
  const progress = [];
  const result = await upload({ file, endpoint, token, onProgress: value => progress.push(value) });
  assert.match(result.url, /^https:\/\/wormhole\.app\/room01#[A-Za-z0-9_-]{22}$/);
  assert.equal(remote.calls.at(-1), '/wormhole/api/room/room01/b2/finish-upload');
  assert.equal(remote.chunks.length, 2);
  assert.ok(progress.some(value => value.phase === 'uploading' && value.completed === value.total));
  const { Keychain } = await import('../vendor/wormhole-crypto/index.js');
  const key = new Keychain(result.url.split('#')[1], remote.roomData.salt);
  assert.equal(await key.authTokenB64(), remote.roomData.readerToken);
  const encrypted = Buffer.concat(remote.chunks);
  assert.equal(encrypted.includes('SYNTHETIC-PRIVATE-CSV-MARKER'), false);
  assert.deepEqual(Buffer.from(await new Response(await key.decryptStream(new Blob([encrypted]).stream())).arrayBuffer()), original);
  const decoded = torrent(Buffer.from(await key.decryptMeta(Buffer.from(remote.update.encryptedTorrentFile, 'base64'))));
  assert.equal(decoded.info.name.toString(), file.name);
  assert.equal(decoded.info.length, encrypted.length);
  assert.equal(decoded.info['piece length'], 8 * 1024 * 1024);
  assert.equal(decoded.info.pieces.length, 40);
  assert.equal(decoded.info.pieces.subarray(0, 20).toString('hex'), sha1(remote.chunks[0]));
  assert.equal(decoded.info.pieces.subarray(20).toString('hex'), sha1(remote.chunks[1]));
});

test('automatic upload fails closed for invalid configuration, cancellation and unacknowledged completion', async t => {
  const upload = await client(), remote = service(t, { failFinish: true });
  const file = new File(['synthetic'], 'Atmungsraum-Daten.zip');
  for (const config of [{ token: 'short' }, { endpoint: 'https://attacker.test' }, { file: new File(['x'], '../unsafe.zip') }]) {
    await assert.rejects(() => upload({ file, endpoint, token, ...config }));
  }
  assert.equal(remote.calls.length, 0);
  await assert.rejects(() => upload({ file, endpoint, token, signal: AbortSignal.abort() }), { name: 'AbortError' });
  assert.equal(remote.calls.length, 0);
  await assert.rejects(() => upload({ file, endpoint, token }), /Upload/);
});
