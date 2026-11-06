import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../sharing/worker.js';
const env = { ALLOWED_ORIGIN: 'https://viscereality.org', LINK_RELAY_TOKEN: 'A'.repeat(43) };
const writer = `Bearer sync-v1 ${Buffer.alloc(16, 7).toString('base64')}`;
const body = { readerToken: Buffer.alloc(16, 1).toString('base64'), salt: Buffer.alloc(16, 2).toString('base64') };
function req(path, value = body, extra = {}, method = 'POST') {
  return new Request(`https://relay.example.workers.dev${path}`, { method,
    headers: { Origin: env.ALLOWED_ORIGIN, Authorization: `Bearer ${env.LINK_RELAY_TOKEN}`, 'Content-Type': 'application/json', ...extra },
    body: value === undefined ? undefined : value instanceof Uint8Array ? value : JSON.stringify(value),
  });
}

test('restricted Worker uploads use only Wormhole API routes and never forward the operator access token', async t => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => { calls.push({ url, init }); return Response.json({ id: 'room01' }); });
  const response = await worker.fetch(req('/wormhole/api/room', { ...body, url: 'must-not-forward' }), env);
  assert.equal(response.status, 200);
  assert.equal(calls[0].url, 'https://wormhole.app/api/room');
  assert.deepEqual(JSON.parse(calls[0].init.body), body);
  assert.equal(calls[0].init.headers.Authorization, undefined);
  assert.equal(calls[0].init.headers.Origin, 'https://wormhole.app');
  assert.equal(calls[0].init.redirect, 'manual');
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), env.ALLOWED_ORIGIN);
  const next = await worker.fetch(req('/wormhole/api/room/room01/b2/auth-upload', { numTokens: 1 }, { 'X-Wormhole-Authorization': writer }), env);
  assert.equal(next.status, 200);
  assert.equal(calls[1].init.headers.Authorization, writer);
  assert.ok(!JSON.stringify(calls).includes(env.LINK_RELAY_TOKEN));
});

test('chunk relay accepts only bounded encrypted chunks at Backblaze upload endpoints', async t => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => { calls.push({ url, init }); return Response.json({ fileId: 'accepted' }); });
  const headers = { 'Content-Type': 'application/octet-stream',
    'X-Wormhole-Upload-URL': 'https://pod-001.backblazeb2.com/b2api/v2/b2_upload_file/0123456789abcdef01234567/session-12345678901',
    'X-Wormhole-B2-Authorization': 'signed-b2-token', 'X-Bz-File-Name': 'room01/0', 'X-Bz-Content-Sha1': 'a'.repeat(40) };
  const response = await worker.fetch(req('/wormhole/chunk', new Uint8Array([1, 2, 3]), headers), env);
  assert.equal(response.status, 200);
  assert.equal(calls[0].url, headers['X-Wormhole-Upload-URL']);
  assert.equal(calls[0].init.headers.Authorization, 'signed-b2-token');
  assert.deepEqual([...new Uint8Array(calls[0].init.body)], [1, 2, 3]);
  for (const url of ['https://attacker.test/b2api/v3/b2_upload_file', 'https://pod.backblazeb2.com.evil.test/b2api/v3/b2_upload_file',
    'https://pod.backblazeb2.com/other', 'http://pod.backblazeb2.com/b2api/v3/b2_upload_file', 'https://user@pod.backblazeb2.com/b2api/v3/b2_upload_file']) {
    assert.equal((await worker.fetch(req('/wormhole/chunk', new Uint8Array([1]), { ...headers, 'X-Wormhole-Upload-URL': url }), env)).status, 400);
  }
  assert.equal((await worker.fetch(req('/wormhole/chunk', new Uint8Array(8 * 1024 * 1024 + 1), headers), env)).status, 413);
  assert.equal(calls.length, 1);
});

test('unconfigured, unauthenticated and unexpected upload requests fail without reaching a provider', async t => {
  let calls = 0;
  t.mock.method(globalThis, 'fetch', async () => { calls++; return Response.json({}); });
  for (const [request, status] of [
    [req('/wormhole/api/room', body, { Authorization: '' }), 401],
    [req('/wormhole/api/room', body, { Origin: 'https://evil.test' }), 403],
    [req('/wormhole/api/room', { readerToken: 'bad', salt: 'bad' }), 400],
    [req('/wormhole/api/room/room01/b2/auth-upload', { numTokens: 100 }, { 'X-Wormhole-Authorization': writer }), 400],
    [req('/wormhole/api/room/room01', {}, { 'X-Wormhole-Authorization': 'wrong' }, 'PATCH'), 401],
    [req('/wormhole/api/arbitrary', {}), 404],
    [req('/wormhole/api/room/room01/delete', {}), 404],
  ]) assert.equal((await worker.fetch(request, env)).status, status);
  assert.equal(calls, 0);
});

// Workers reject redirect:error before I/O; manual must reject redirects without
// forwarding room/storage credentials to a new host.
test('Worker rejects provider redirects using the supported edge redirect mode', async t => {
  const modes = [];
  t.mock.method(globalThis, 'fetch', async (_url, init) => {
    modes.push(init.redirect);
    if (init.redirect === 'error') throw new TypeError('Unsupported Workers redirect mode');
    assert.equal(init.redirect, 'manual');
    return new Response(null, { status: 302, headers: { Location: 'https://attacker.test/' } });
  });
  const response = await worker.fetch(req('/wormhole/api/room'), env);
  assert.equal(response.status, 502);
  assert.deepEqual(modes, ['manual']);
});
