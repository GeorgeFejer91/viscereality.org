import test from 'node:test';
import assert from 'node:assert/strict';

const origin = 'https://viscereality.org';
const url = 'https://wormhole.app/nb7dMz#fnQhVRrakH6ZUgQyUeAFfA';
const token = 'A'.repeat(43);
const request = (body = { url }, headers = {}) => new Request('https://relay.example.workers.dev/notify', {
  method: 'POST', headers: { Origin: origin, Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', ...headers },
  body: typeof body === 'string' ? body : JSON.stringify(body),
});
async function fixture() {
  const { default: worker } = await import('../sharing/worker.js');
  const sent = [];
  const env = { ALLOWED_ORIGIN: origin, LINK_RELAY_TOKEN: token, DESTINATION: 'email', EMAIL_FROM: 'links@viscereality.org', EMAIL_TO: 'owner@example.test', EMAIL: { async send(message) { sent.push(message); return { messageId: 'mail-accepted' }; } } };
  return { worker, env, sent };
}

test('Worker sends only the complete Wormhole link to its configured email recipient and acknowledges acceptance', async () => {
  const { worker, env, sent } = await fixture();
  const response = await worker.fetch(request({ url, to: 'attacker@example.test' }), env);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), origin);
  assert.match(response.headers.get('Cache-Control'), /no-store/);
  assert.deepEqual(await response.json(), { accepted: true, destination: 'email', message_id: 'mail-accepted' });
  assert.equal(sent.length, 1);
  assert.equal(sent[0].to, 'owner@example.test');
  assert.equal(sent[0].from, 'links@viscereality.org');
  assert.ok(sent[0].text.includes(url));
  assert.equal(sent[0].attachments, undefined);
});

test('Worker rejects missing/wrong credentials, foreign origins, unsafe links and oversized bodies without sending', async () => {
  const { worker, env, sent } = await fixture();
  for (const [req, code] of [
    [request({ url }, { Authorization: '' }), 401],
    [request({ url }, { Authorization: `Bearer ${'B'.repeat(43)}` }), 401],
    [request({ url }, { Origin: 'https://attacker.test' }), 403],
    [request({ url: 'https://attacker.test/' }), 400],
    [request({ url: 'https://wormhole.app/nb7dMz' }), 400],
    [request('not-json'), 400],
    [request(' '.repeat(1025)), 413],
    [request({ url }, { 'Content-Type': 'text/plain' }), 415],
  ]) assert.equal((await worker.fetch(req, env)).status, code);
  assert.equal(sent.length, 0);
});

test('Worker answers the allowed preflight and fails closed when unconfigured', async () => {
  const { worker, env, sent } = await fixture();
  const preflight = new Request('https://relay.example.workers.dev/notify', { method: 'OPTIONS', headers: { Origin: origin, 'Access-Control-Request-Method': 'POST' } });
  const response = await worker.fetch(preflight, env);
  assert.equal(response.status, 204);
  assert.match(response.headers.get('Access-Control-Allow-Headers'), /Authorization/i);
  assert.equal((await worker.fetch(request(), { ...env, LINK_RELAY_TOKEN: '' })).status, 503);
  assert.equal((await worker.fetch(request(), { ...env, EMAIL_TO: '' })).status, 503);
  assert.equal((await worker.fetch(request(), { ...env, ALLOWED_ORIGIN: '*' })).status, 503);
  assert.equal(sent.length, 0);
});

test('Provider failures and missing acceptance IDs never report forwarding success or leak link/key', async () => {
  const { worker, env } = await fixture();
  for (const send of [async () => { throw new Error(`private ${url} ${token}`); }, async () => { throw Object.assign(new Error(`private ${url} ${token}`), { status: 400 }); }, async () => ({})]) {
    const response = await worker.fetch(request(), { ...env, EMAIL: { send } });
    assert.equal(response.status, 502);
    const body = await response.text();
    assert.equal(body.includes(url), false);
    assert.equal(body.includes(token), false);
  }
});

test('Discord forwarding uses only the server webhook, disables previews/mentions and waits for message acknowledgement', async t => {
  const { worker, env } = await fixture();
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (target, init) => { calls.push({ target: String(target), init }); return Response.json({ id: 'discord-accepted' }); });
  const response = await worker.fetch(request(), { ...env, DESTINATION: 'discord', DISCORD_WEBHOOK_URL: 'https://discord.com/api/webhooks/123456/TEST_TOKEN' });
  assert.equal(response.status, 200);
  assert.equal((await response.json()).destination, 'discord');
  assert.equal(calls.length, 1);
  assert.equal(calls[0].target, 'https://discord.com/api/webhooks/123456/TEST_TOKEN?wait=true');
  assert.equal(calls[0].init.redirect, 'manual');
  const body = JSON.parse(calls[0].init.body);
  assert.ok(body.content.includes(`<${url}>`));
  assert.deepEqual(body.allowed_mentions, { parse: [] });
});

test('Telegram forwarding supports a configured channel and requires the Telegram success response', async t => {
  const { worker, env } = await fixture();
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (target, init) => { calls.push({ target: String(target), init }); return Response.json({ ok: true, result: { message_id: 42 } }); });
  const response = await worker.fetch(request(), { ...env, DESTINATION: 'telegram', TELEGRAM_BOT_TOKEN: '123456:TEST_TOKEN', TELEGRAM_CHAT_ID: '@study_channel' });
  assert.equal(response.status, 200);
  assert.equal((await response.json()).message_id, '42');
  assert.equal(calls[0].target, 'https://api.telegram.org/bot123456:TEST_TOKEN/sendMessage');
  assert.deepEqual(JSON.parse(calls[0].init.body), { chat_id: '@study_channel', text: `Atmungsraum-Daten\n${url}`, link_preview_options: { is_disabled: true } });
});

test('Browser forwarding validates its endpoint and credentials, sends no recipient and requires an acknowledgement', async t => {
  const { forwardWormholeLink } = await import('../link-relay.js');
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (target, init) => { calls.push({ target, init }); return Response.json({ accepted: true, destination: 'email', message_id: 'mail-accepted' }); });
  const result = await forwardWormholeLink({ endpoint: 'https://relay.example.workers.dev', token, url });
  assert.equal(result.destination, 'email');
  assert.equal(calls[0].target, 'https://relay.example.workers.dev/notify');
  assert.deepEqual(JSON.parse(calls[0].init.body), { url });
  assert.equal(calls[0].init.headers.Authorization, `Bearer ${token}`);
  assert.equal(calls[0].init.redirect, 'error');
  for (const config of [{ endpoint: 'https://attacker.test' }, { endpoint: 'http://relay.example.workers.dev' }, { endpoint: 'https://relay.example.workers.dev/#key' }, { token: 'short' }, { url: 'https://wormhole.app/' }]) {
    await assert.rejects(() => forwardWormholeLink({ endpoint: 'https://relay.example.workers.dev', token, url, ...config }));
  }
  assert.equal(calls.length, 1);
  t.mock.restoreAll();
  t.mock.method(globalThis, 'fetch', async () => Response.json({ success: true }));
  await assert.rejects(() => forwardWormholeLink({ endpoint: 'https://relay.example.workers.dev', token, url }), /bestätigt/);
});


test('Direct browser Discord delivery uses its private webhook with a confirmed response and no relay', async t => {
  const { sendDiscordLink } = await import('../link-relay.js');
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (target, init) => { calls.push({ target, init }); return Response.json({ id: 'direct-ack' }); });
  const webhook = 'https://discord.com/api/webhooks/123456789/TEST_TOKEN';
  const result = await sendDiscordLink({ webhook, url });
  assert.deepEqual(result, { accepted: true, destination: 'discord', message_id: 'direct-ack' });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].target, webhook + '?wait=true');
  assert.deepEqual(JSON.parse(calls[0].init.body), { content: url, flags: 4, allowed_mentions: { parse: [] } });
  assert.equal(calls[0].init.credentials, 'omit');
  assert.equal(calls[0].init.redirect, 'error');
  assert.equal(calls[0].init.referrerPolicy, 'no-referrer');
  assert.equal(calls[0].init.headers.Authorization, undefined);
});

test('Direct Discord rejects unsafe destinations and never exposes a webhook or keyed link in failures', async t => {
  const { sendDiscordLink } = await import('../link-relay.js');
  const webhook = 'https://discord.com/api/webhooks/123456789/TEST_TOKEN';
  let calls = 0;
  t.mock.method(globalThis, 'fetch', async () => { calls++; throw new Error(webhook + url); });
  for (const invalid of ['https://evil.test/api/webhooks/123456789/TEST_TOKEN', webhook + '?wait=true', webhook + '#key', webhook.replace('https:', 'http:'), webhook.replace('discord.com', 'discord.com.evil.test'), webhook.replace('discord.com', 'user@discord.com')]) {
    await assert.rejects(() => sendDiscordLink({ webhook: invalid, url }));
  }
  await assert.rejects(() => sendDiscordLink({ webhook, url: 'https://wormhole.app/' }));
  assert.equal(calls, 0);
  await assert.rejects(() => sendDiscordLink({ webhook, url }), error => !error.message.includes(webhook) && !error.message.includes(url));
  assert.equal(calls, 1);
  t.mock.restoreAll();
  for (const [status, body] of [[200, {}], [204, null], [429, { id: 'not-accepted' }]]) {
    t.mock.method(globalThis, 'fetch', async () => new Response(body === null ? null : JSON.stringify(body), { status }));
    await assert.rejects(() => sendDiscordLink({ webhook, url }), /bestätigt/);
    t.mock.restoreAll();
  }
});
