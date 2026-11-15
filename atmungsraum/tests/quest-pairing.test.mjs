import test from 'node:test';
import assert from 'node:assert/strict';
import { createCipheriv, createHash, createHmac, createPublicKey, diffieHellman,
  generateKeyPairSync, hkdfSync, randomBytes } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { pairPhone } from '../quest-pairing.js';

const owner = { device: 'a'.repeat(32), serial: '', identity_kind: 'app-installation', epoch: '7', native_epoch: '8' };
const endpoint = 'http://192.168.2.48:8766/v1/gateway';
const ticket = { v: 2, ...owner, controller: 'c'.repeat(32), key: 'd'.repeat(64) };
const prefix = 'RQ-PHONE-PAIR-2';
const exact = (value, keys) => assert.deepEqual(Object.keys(value).sort(), keys.sort());
const context = begin => [begin.device, begin.serial, begin.identity_kind, begin.epoch, begin.native_epoch,
  Buffer.from(begin.phone_name, 'utf8').toString('base64url')].join('|');
const commitment = (begin, opening) => createHash('sha256').update(`${prefix}/commit|${context(begin)}|${opening.client_nonce}|${opening.client_public_key}`).digest('hex');
const transcript = (begin, reply, opening) => [prefix, context(begin), reply.pairing_id, begin.client_commitment,
  opening.client_nonce, reply.server_nonce, opening.client_public_key, reply.server_public_key].join('|');

function fixture({ reject = false, tamper = false, wrongOwner = false, headsetApproved = true,
  mutateCommitment = false, badNonce = false, badReveal = false, legacy = false, extra = false } = {}) {
  const state = { requests: [], confirmed: false, cancelled: false, statusCount: 0, revealed: false };
  let native, begin, reply, text, proofKey, aeadKey;
  return Object.assign(state, { async fetch(url, init) {
    assert.equal(url, endpoint); assert.equal(init.credentials, 'omit'); assert.equal(init.redirect, 'error');
    const request = JSON.parse(init.body); state.requests.push(request.action);
    assert.equal(request.v, 2);
    if (request.action === 'pair-begin') {
      exact(request, ['v', 'action', 'device', 'serial', 'identity_kind', 'epoch', 'native_epoch', 'phone_name', 'client_commitment']);
      assert.match(request.client_commitment, /^[a-f0-9]{64}$/);
      assert.equal(request.client_nonce, undefined); assert.equal(request.client_public_key, undefined);
      begin = structuredClone(request); if (mutateCommitment) begin.client_commitment = '0'.repeat(64);
      native = generateKeyPairSync('ec', { namedCurve: 'prime256v1' });
      reply = { v: 2, device: owner.device, epoch: owner.epoch, native_epoch: owner.native_epoch,
        pairing_id: 'e'.repeat(32), client_commitment: request.client_commitment,
        server_public_key: native.publicKey.export({ type: 'spki', format: 'der' }).toString('base64url'),
        server_nonce: randomBytes(16).toString('base64url'), expires_in_ms: 120000 };
      const response = structuredClone(reply);
      if (wrongOwner) response.native_epoch = '99';
      if (badNonce) response.server_nonce = 'f'.repeat(32);
      if (legacy) response.v = 1;
      if (extra) response.extra = true;
      return Response.json(response);
    }
    if (request.action === 'pair-reveal') {
      exact(request, ['v', 'action', 'device', 'epoch', 'native_epoch', 'pairing_id', 'client_public_key', 'client_nonce', 'proof']);
      assert.equal(request.client_nonce.length, 22); assert.equal(Buffer.from(request.client_nonce, 'base64url').length, 16);
      assert.equal(begin.client_commitment, commitment(begin, request), 'opening must match original frozen commitment');
      text = transcript(begin, reply, request);
      const secret = diffieHellman({ privateKey: native.privateKey,
        publicKey: createPublicKey({ key: Buffer.from(request.client_public_key, 'base64url'), type: 'spki', format: 'der' }) });
      const salt = createHash('sha256').update(text).digest();
      const derive = label => Buffer.from(hkdfSync('sha256', secret, salt, `${prefix}/${label}`, 32));
      state.code = String(derive('SAS').readUInt32BE(0) % 1000000).padStart(6, '0');
      proofKey = derive('PROOF'); aeadKey = derive('AEAD');
      assert.equal(request.proof, createHmac('sha256', proofKey).update(text + '|pair-reveal').digest('hex'));
      state.revealed = true;
      return Response.json({ v: 2, pairing_id: reply.pairing_id, state: badReveal ? 'committed' : 'pending' });
    }
    exact(request, ['v', 'action', 'device', 'epoch', 'native_epoch', 'pairing_id', 'proof']);
    assert.equal(state.revealed, true, 'no approval/status before commitment opening');
    assert.equal(request.proof, createHmac('sha256', proofKey).update(text + '|' + request.action).digest('hex'));
    if (request.action === 'pair-cancel') state.cancelled = true;
    if (request.action === 'pair-confirm') state.confirmed = true;
    state.statusCount++;
    if (reject) return Response.json({ v: 2, pairing_id: request.pairing_id, state: 'rejected' });
    if (!headsetApproved || !state.confirmed) return Response.json({ v: 2, pairing_id: request.pairing_id, state: 'pending' });
    const iv = randomBytes(12), cipher = createCipheriv('aes-256-gcm', aeadKey, iv);
    cipher.setAAD(Buffer.from(prefix + '/result|' + text));
    const data = Buffer.concat([cipher.update(JSON.stringify(ticket)), cipher.final(), cipher.getAuthTag()]);
    if (tamper) data[0] ^= 1;
    return Response.json({ v: 2, pairing_id: request.pairing_id, state: 'approved', iv: iv.toString('base64url'), data: data.toString('base64url') });
  } });
}

test('commit and verified reveal precede either matching-code approval and encrypted ticket', async () => {
  const native = fixture(); let prompts = 0;
  const result = await pairPhone({ endpoint, owner, fetcher: native.fetch, wait: async () => {},
    async onCode({ code }) { prompts++; assert.equal(native.revealed, true); assert.equal(code, native.code);
      assert.equal(native.confirmed, false); return true; } });
  assert.equal(prompts, 1); assert.deepEqual(result, ticket);
  assert.deepEqual(native.requests.slice(0, 2), ['pair-begin', 'pair-reveal']);
  assert.ok(native.requests.includes('pair-confirm')); assert.equal(native.cancelled, false);
});

test('rejected, modified or foreign-owner pairing cannot return a saved phone key', async () => {
  for (const options of [{ reject: true }, { tamper: true }, { wrongOwner: true }]) {
    const native = fixture(options);
    await assert.rejects(pairPhone({ endpoint, owner, fetcher: native.fetch, wait: async () => {}, onCode: async () => true }));
  }
});

test('commitment mismatch, old hex nonce and invalid reveal response never show phone SAS', async () => {
  for (const options of [{ mutateCommitment: true }, { badNonce: true }, { badReveal: true }, { legacy: true }, { extra: true }]) {
    const native = fixture(options); let prompts = 0;
    await assert.rejects(pairPhone({ endpoint, owner, fetcher: native.fetch, wait: async () => {}, onCode: async () => { prompts++; return true; } }));
    assert.equal(prompts, 0); assert.equal(native.confirmed, false);
  }
});

test('phone cancellation sends only a possession-bound cancel, with no confirmation or ticket', async () => {
  const native = fixture();
  await assert.rejects(pairPhone({ endpoint, owner, fetcher: native.fetch, wait: async () => {}, onCode: async () => false }));
  assert.equal(native.confirmed, false); assert.equal(native.cancelled, true);
});

test('no pairing request is sent to a public endpoint or after foreground cancellation', async () => {
  const native = fixture(); const aborted = new AbortController(); aborted.abort();
  await assert.rejects(pairPhone({ endpoint, owner, fetcher: native.fetch, signal: aborted.signal, onCode: async () => true }));
  await assert.rejects(pairPhone({ endpoint: 'http://example.com', owner, fetcher: native.fetch, onCode: async () => true }));
  assert.deepEqual(native.requests, []);
});

test('response streaming stops at 8192 bytes before parsing or showing a code', async () => {
  let cancelled = false, prompts = 0, chunks = 0;
  await assert.rejects(pairPhone({ endpoint, owner, onCode: async () => { prompts++; return true; }, fetcher: async () =>
    new Response(new ReadableStream({ pull(controller) { chunks++; controller.enqueue(new Uint8Array(4096)); }, cancel() { cancelled = true; } })) }));
  assert.equal(cancelled, true); assert.equal(prompts, 0); assert.ok(chunks <= 4);
});

// Actual production Java, on its single owner thread. Only synthetic storage
// and the HTTP envelope are replaced; browser WebCrypto runs unchanged.
async function javaPeer(t) {
  const child = spawn(process.env.RQ_PAIRING_JAVA || 'java', ['-cp', process.env.RQ_PAIRING_JAVA_CLASSPATH,
    'io.github.mesmerprism.rustyquest.native_renderer.ExperimentPhonePairingTest', '--serve'], { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });
  const queued = []; let stderr = '';
  child.stderr.setEncoding('utf8'); child.stderr.on('data', chunk => { stderr += chunk; });
  const lines = createInterface({ input: child.stdout });
  lines.on('line', line => { const pending = queued.shift(); if (!pending) throw Error('Unsolicited Java test response'); pending.resolve(JSON.parse(line)); });
  child.on('error', error => { for (const pending of queued.splice(0)) pending.reject(error); });
  child.on('exit', code => { for (const pending of queued.splice(0)) pending.reject(Error(`Java pairing peer exited ${code}: ${stderr}`)); });
  t.after(() => { child.stdin.end(); child.kill(); lines.close(); });
  const raw = value => new Promise((resolve, reject) => { queued.push({ resolve, reject }); child.stdin.write(JSON.stringify(value) + '\n'); });
  const call = async value => { const result = await raw(value); assert.equal(result.ok, true, result.error); return result.result; };
  return { raw, call, owner: await call({ op: 'identity' }) };
}
const javaOptions = { skip: !process.env.RQ_PAIRING_JAVA_CLASSPATH && 'Run native Test-ExperimentPhonePairing.ps1 -WebTest <this-file> for actual Java interoperability' };

for (const order of ['phone-first', 'headset-first']) test(`FULL WebCrypto -> production Java commit/reveal/${order}/status/decrypt`, javaOptions, async t => {
  const java = await javaPeer(t); const actions = []; let prompts = 0;
  const result = await pairPhone({ endpoint, owner: java.owner, phoneName: 'Test phone', wait: async () => {},
    fetcher: async (_url, init) => {
      const request = JSON.parse(init.body); actions.push(request.action);
      const response = await java.call({ op: 'exchange', request });
      if (request.action === 'pair-begin') {
        assert.equal(request.client_nonce, undefined); assert.equal(request.client_public_key, undefined);
        assert.equal(await java.call({ op: 'prompt' }), null); assert.equal((await java.call({ op: 'stats' })).stores, 0);
      }
      if (request.action === 'pair-confirm' && order === 'phone-first') {
        assert.equal(response.state, 'pending'); assert.equal((await java.call({ op: 'stats' })).stores, 0);
        await java.call({ op: 'decide', id: request.pairing_id, allow: true });
      }
      return Response.json(response);
    },
    onCode: async ({ code, pairingId }) => {
      prompts++; const prompt = await java.call({ op: 'prompt' });
      assert.equal(prompt.code, code); assert.equal(prompt.pairing_id, pairingId); assert.equal((await java.call({ op: 'stats' })).stores, 0);
      if (order === 'headset-first') await java.call({ op: 'decide', id: pairingId, allow: true });
      return true;
    } });
  assert.equal(prompts, 1); assert.deepEqual(actions.slice(0, 2), ['pair-begin', 'pair-reveal']);
  assert.ok(actions.includes('pair-confirm')); assert.ok(actions.includes('pair-status'));
  assert.equal(result.v, 2); assert.equal(result.controller, '3'.repeat(32)); assert.equal(result.key, '4'.repeat(64));
  for (const field of Object.keys(java.owner)) assert.equal(result[field], java.owner[field]);
  assert.equal((await java.call({ op: 'stats' })).stores, 1);
});

test('FULL Java rejects changed commitment/opening/proof before either UI', javaOptions, async t => {
  for (const mutation of ['commitment', 'nonce', 'key', 'proof', 'confirm-before-reveal', 'owner-name', 'responder-nonce', 'responder-key']) {
    const java = await javaPeer(t); let prompts = 0, originalReveal;
    await assert.rejects(pairPhone({ endpoint, owner: java.owner, phoneName: 'Test phone', wait: async () => {},
      onCode: async () => { prompts++; return true; }, fetcher: async (_url, init) => {
        const request = JSON.parse(init.body);
        if (request.action === 'pair-begin' && mutation === 'commitment') request.client_commitment = '0'.repeat(64);
        if (request.action === 'pair-begin' && mutation === 'owner-name') request.phone_name = 'Foreign phone';
        if (request.action === 'pair-reveal') {
          originalReveal = structuredClone(request);
          if (mutation === 'nonce') request.client_nonce = randomBytes(16).toString('base64url');
          if (mutation === 'key') request.client_public_key = generateKeyPairSync('ec', { namedCurve: 'prime256v1' }).publicKey.export({ type: 'spki', format: 'der' }).toString('base64url');
          if (mutation === 'proof') request.proof = '0'.repeat(64);
          if (mutation === 'confirm-before-reveal') { request.action = 'pair-confirm'; delete request.client_nonce; delete request.client_public_key; }
        }
        const response = await java.raw({ op: 'exchange', request });
        if (!response.ok) return new Response('', { status: 403 });
        if (request.action === 'pair-begin' && mutation === 'responder-nonce') response.result.server_nonce = randomBytes(16).toString('base64url');
        if (request.action === 'pair-begin' && mutation === 'responder-key') response.result.server_public_key = generateKeyPairSync('ec', { namedCurve: 'prime256v1' }).publicKey.export({ type: 'spki', format: 'der' }).toString('base64url');
        return Response.json(response.result);
      } }));
    assert.equal(prompts, 0); assert.equal(await java.call({ op: 'prompt' }), null); assert.equal((await java.call({ op: 'stats' })).stores, 0);
    if (['nonce', 'key', 'proof', 'owner-name', 'responder-nonce', 'responder-key'].includes(mutation) && originalReveal) {
      const replay = await java.raw({ op: 'exchange', request: originalReveal });
      assert.equal(replay.ok, false, 'failed opening cannot be repaired adaptively under same commitment');
    }
  }
});
