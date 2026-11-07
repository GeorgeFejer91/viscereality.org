import test from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { createHash } from 'node:crypto';

const networkFetch = fetch;
const origin = 'https://viscereality.org', token = 'A'.repeat(43);
const uploadUrl = 'https://pod-001.backblazeb2.com/b2api/v2/b2_upload_file/bucket/session';

test('Render HTTP adapter exposes readiness and only authenticated upload routes, preserving a full 8 MiB piece', async t => {
  const { createRenderRelay } = await import('../sharing/render.js');
  assert.throws(() => createRenderRelay('short'), /LINK_RELAY_TOKEN/);
  const server = createRenderRelay(token).listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => server.close());
  const endpoint = `http://127.0.0.1:${server.address().port}`;
  const health = await networkFetch(`${endpoint}/health`);
  assert.equal(health.status, 204);
  assert.equal(health.headers.get('Access-Control-Allow-Origin'), origin);
  assert.equal(health.headers.get('Cache-Control'), 'no-store');
  const path = '/wormhole/chunk', upstream = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    upstream.push({ url, init });
    return Response.json({ fileId: 'test-chunk' });
  });
  const preflight = await networkFetch(endpoint + path, { method: 'OPTIONS', headers: { Origin: origin } });
  assert.equal(preflight.status, 204);
  assert.match(preflight.headers.get('Access-Control-Allow-Headers'), /X-Wormhole-Upload-URL/i);
  const headers = { Origin: origin, Authorization: `Bearer ${token}`, 'Content-Type': 'application/octet-stream',
    'X-Wormhole-Upload-URL': uploadUrl, 'X-Wormhole-B2-Authorization': 'test-b2-authorization', 'X-Bz-File-Name': 'room01/0' };
  const bytes = Buffer.alloc(8 * 1024 * 1024, 21);
  headers['X-Bz-Content-Sha1'] = createHash('sha1').update(bytes).digest('hex');
  for (const [route, extra, expected] of [
    ['/notify', {}, 404], [path + '?not-allowed=1', {}, 404],
    [path, { Authorization: '' }, 401], [path, { Origin: 'https://foreign.example' }, 403],
    [path, { 'X-Wormhole-Upload-URL': 'https://foreign.example/b2api/v2/b2_upload_file' }, 400],
  ]) {
    const response = await networkFetch(endpoint + route, { method: 'POST', headers: { ...headers, ...extra }, body: new Uint8Array([1]) });
    assert.equal(response.status, expected);
    await response.text();
  }
  const oversized = await networkFetch(endpoint + path, { method: 'POST', headers, body: Buffer.alloc(bytes.length + 1) });
  assert.equal(oversized.status, 413);
  await oversized.text();
  assert.equal(upstream.length, 0);
  const response = await networkFetch(endpoint + path, { method: 'POST', headers, body: bytes });
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { fileId: 'test-chunk' });
  assert.equal(upstream.length, 1);
  assert.equal(upstream[0].url, uploadUrl);
  assert.equal(upstream[0].init.headers.Authorization, 'test-b2-authorization');
  assert.equal(upstream[0].init.redirect, 'manual');
  assert.deepEqual(Buffer.from(upstream[0].init.body), bytes);
});

test('Render upload waits for a credential-free readiness probe and never starts a cancelled upload', async t => {
  const { uploadWormholeZip } = await import('../wormhole-upload.js');
  const calls = [], progress = [];
  const endpoint = 'https://atmungsraum-test.onrender.com';
  const controller = new AbortController();
  t.mock.method(globalThis, 'fetch', async (target, init) => {
    const path = new URL(target).pathname;
    calls.push(path);
    assert.equal(new URL(target).origin, endpoint);
    if (path === '/health') {
      assert.equal(init.method, 'GET');
      assert.equal(init.credentials, 'omit');
      assert.equal(init.headers, undefined);
      return calls.length === 1 ? new Response('Starting', { status: 200 }) : new Response(null, { status: 204 });
    }
    controller.abort();
    init.signal.throwIfAborted();
  });
  await assert.rejects(() => uploadWormholeZip({ file: new File(['synthetic'], 'test.zip'), endpoint, token,
    signal: controller.signal, onProgress: value => progress.push(value.phase) }), { name: 'AbortError' });
  assert.deepEqual(calls, ['/health', '/health', '/wormhole/api/room']);
  assert.equal(progress[0], 'starting');
  const cancelled = new AbortController();
  t.mock.restoreAll();
  t.mock.method(globalThis, 'fetch', async (target, init) => {
    assert.equal(new URL(target).pathname, '/health');
    cancelled.abort();
    init.signal.throwIfAborted();
  });
  await assert.rejects(() => uploadWormholeZip({ file: new File(['synthetic'], 'test.zip'), endpoint, token,
    signal: cancelled.signal }), { name: 'AbortError' });
});
