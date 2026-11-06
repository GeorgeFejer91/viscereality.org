import { validateWormholeUrl } from '../wormhole-link.js';

const tokenPattern = /^[A-Za-z0-9_-]{32,128}$/;
class RequestError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}
const fail = (status, message) => { throw new RequestError(status, message); };
const address = value => typeof value === 'string' && value.length <= 254 && /^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$/.test(value);

async function sameToken(a, b) {
  const hash = text => crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  const [left, right] = await Promise.all([hash(a), hash(b)]);
  return new Uint8Array(left).reduce((difference, byte, i) => difference | (byte ^ new Uint8Array(right)[i]), 0) === 0;
}

async function readBytes(request, limit) {
  if (Number(request.headers.get('Content-Length')) > limit) fail(413, 'Request too large');
  const reader = request.body?.getReader();
  if (!reader) fail(400, 'Link required');
  const parts = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > limit) { await reader.cancel(); fail(413, 'Request too large'); }
    parts.push(value);
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const part of parts) { bytes.set(part, offset); offset += part.byteLength; }
  return bytes;
}

async function readJson(request, limit = 1024) {
  if (request.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase() !== 'application/json') fail(415, 'JSON required');
  const bytes = await readBytes(request, limit);
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)); }
  catch { fail(400, 'Invalid JSON'); }
}

async function wormholeUpload(request, path) {
  let target, headers, body;
  if (path === '/wormhole/chunk') {
    if (request.method !== 'POST') fail(405, 'POST required');
    try { target = new URL(request.headers.get('X-Wormhole-Upload-URL')); } catch { /* Reject below. */ }
    if (!target || target.protocol !== 'https:' || target.port || target.username || target.password || target.search || target.hash
      || !/\.(backblazeb2\.com|backblaze\.com)$/.test(target.hostname)
      || !/^\/b2api\/v[123]\/b2_upload_file(?:\/[A-Za-z0-9_-]{1,256}\/[A-Za-z0-9_-]{1,256})?$/.test(target.pathname)) fail(400, 'Invalid storage endpoint');
    const authorization = request.headers.get('X-Wormhole-B2-Authorization');
    const name = request.headers.get('X-Bz-File-Name'), hash = request.headers.get('X-Bz-Content-Sha1');
    if (!/^[\x21-\x7e]{8,4096}$/.test(authorization ?? '') || !/^[A-Za-z0-9_-]{1,128}\/\d{1,6}$/.test(name ?? '')
      || !/^[a-f0-9]{40}$/.test(hash ?? '')) fail(400, 'Invalid storage authorization');
    if (request.headers.get('Content-Type') !== 'application/octet-stream') fail(415, 'Encrypted binary chunk required');
    body = await readBytes(request, 8 * 1024 * 1024);
    if (!body.length) fail(400, 'Empty chunk');
    headers = { Authorization: authorization, 'Content-Type': 'application/octet-stream',
      'X-Bz-File-Name': name, 'X-Bz-Content-Sha1': hash, 'Content-Length': String(body.length) };
  } else {
    const route = path.match(/^\/wormhole\/api\/room(?:\/([A-Za-z0-9_-]{1,128})(?:\/(uploader-online|b2\/auth-upload|b2\/finish-upload))?)?$/);
    if (!route) fail(404, 'Not found');
    const [, room, action] = route;
    const method = !room || action?.startsWith('b2/') ? 'POST' : 'PATCH';
    if (request.method !== method) fail(405, `${method} required`);
    headers = { 'Content-Type': 'application/json', Origin: 'https://wormhole.app', Referer: 'https://wormhole.app/' };
    if (room) {
      const authorization = request.headers.get('X-Wormhole-Authorization');
      if (!/^Bearer sync-v1 [A-Za-z0-9+/]{22}==$/.test(authorization ?? '')) fail(401, 'Invalid room authorization');
      headers.Authorization = authorization;
    }
    if (!room) {
      const value = await readJson(request);
      if (!/^[A-Za-z0-9+/]{22}==$/.test(value?.readerToken ?? '') || !/^[A-Za-z0-9+/]{22}==$/.test(value?.salt ?? '')) fail(400, 'Invalid room parameters');
      body = { readerToken: value.readerToken, salt: value.salt };
    } else if (!action) {
      const value = await readJson(request, 65536);
      if (!/^[a-f0-9]{40}$/.test(value?.infoHash ?? '') || !/^[A-Za-z0-9+/]+={0,2}$/.test(value?.encryptedTorrentFile ?? '')
        || value.encryptedTorrentFile.length > 60000 || value.multiFile !== false
        || !Number.isInteger(value.sizeMb) || value.sizeMb < 0 || value.sizeMb > 5000) fail(400, 'Invalid encrypted metadata');
      body = { infoHash: value.infoHash, encryptedTorrentFile: value.encryptedTorrentFile, multiFile: false, sizeMb: value.sizeMb };
    } else if (action === 'b2/auth-upload') {
      const value = await readJson(request);
      if (value?.numTokens !== 1) fail(400, 'One chunk authorization required');
      body = { numTokens: 1 };
    } else if (action === 'b2/finish-upload') {
      const value = await readJson(request);
      if (value?.success !== true) fail(400, 'Completion acknowledgement required');
      body = { success: true };
    }
    target = new URL(path.slice('/wormhole'.length), 'https://wormhole.app');
    body = body === undefined ? undefined : JSON.stringify(body);
  }
  const response = await fetch(target.href, { method: request.method, headers, body, redirect: 'manual', signal: AbortSignal.timeout(90000) });
  if (!response.ok) fail(502, 'Wormhole upload not confirmed');
  if (response.status === 204) return { status: 204, body: null };
  return { status: response.status, body: await readBytes(response, 65536) };
}

async function notify(url, env) {
  const destination = env.DESTINATION || 'email';
  if (destination === 'email') {
    if (!env.EMAIL?.send || !address(env.EMAIL_FROM) || !address(env.EMAIL_TO)) fail(503, 'Email is not configured');
    const result = await env.EMAIL.send({
      from: env.EMAIL_FROM, to: env.EMAIL_TO, subject: 'Atmungsraum: geteilter Daten-Link',
      text: `Atmungsraum-Daten\n${url}\n\nDieser Link enthält den Download-Schlüssel. Wormhole bestimmt seine Gültigkeit.`,
    });
    if (typeof result?.messageId !== 'string' || !result.messageId) throw new Error('Missing email acknowledgement');
    return { accepted: true, destination, message_id: result.messageId };
  }
  if (destination === 'discord') {
    let target;
    try { target = new URL(env.DISCORD_WEBHOOK_URL); } catch { /* Validate below. */ }
    if (!target || target.origin !== 'https://discord.com' || target.username || target.password || target.search || target.hash
      || !/^\/api\/webhooks\/\d+\/[A-Za-z0-9._-]+$/.test(target.pathname)) fail(503, 'Discord is not configured');
    target.searchParams.set('wait', 'true');
    const response = await fetch(target.href, {
      method: 'POST', redirect: 'manual', signal: AbortSignal.timeout(10000),
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: `Atmungsraum-Daten\n<${url}>`, allowed_mentions: { parse: [] } }),
    });
    const result = await response.json();
    if (!response.ok || typeof result.id !== 'string' || !result.id) throw new Error('Missing Discord acknowledgement');
    return { accepted: true, destination, message_id: result.id };
  }
  if (destination === 'telegram') {
    if (!/^\d+:[A-Za-z0-9_-]{8,}$/.test(env.TELEGRAM_BOT_TOKEN ?? '')
      || !/^(@[A-Za-z][A-Za-z0-9_]{4,}|-?\d+)$/.test(env.TELEGRAM_CHAT_ID ?? '')) fail(503, 'Telegram is not configured');
    const response = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
      method: 'POST', redirect: 'manual', signal: AbortSignal.timeout(10000),
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chat_id: env.TELEGRAM_CHAT_ID, text: `Atmungsraum-Daten\n${url}`, link_preview_options: { is_disabled: true } }),
    });
    const result = await response.json();
    if (!response.ok || result.ok !== true || !Number.isInteger(result.result?.message_id)) throw new Error('Missing Telegram acknowledgement');
    return { accepted: true, destination, message_id: String(result.result.message_id) };
  }
  fail(503, 'Destination is not configured');
}

export default {
  async fetch(request, env) {
    const headers = { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', Vary: 'Origin' };
    const reply = (body, status = 200) => new Response(body === null ? null : JSON.stringify(body), { status, headers });
    try {
      let site;
      try { site = new URL(env.ALLOWED_ORIGIN); } catch { /* Fail closed below. */ }
      if (!site || site.origin !== env.ALLOWED_ORIGIN || site.protocol !== 'https:') fail(503, 'Origin is not configured');
      if (request.headers.get('Origin') !== site.origin) fail(403, 'Origin not allowed');
      headers['Access-Control-Allow-Origin'] = site.origin;
      const path = new URL(request.url).pathname;
      const uploading = path === '/wormhole/chunk' || path.startsWith('/wormhole/api/room');
      if (path !== '/notify' && !uploading) fail(404, 'Not found');
      if (request.method === 'OPTIONS') {
        headers['Access-Control-Allow-Methods'] = 'POST, PATCH';
        headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type, X-Wormhole-Authorization, X-Wormhole-Upload-URL, X-Wormhole-B2-Authorization, X-Bz-File-Name, X-Bz-Content-Sha1';
        headers['Access-Control-Max-Age'] = '600';
        return reply(null, 204);
      }
      if (!uploading && request.method !== 'POST') fail(405, 'POST required');
      if (!tokenPattern.test(env.LINK_RELAY_TOKEN ?? '')) fail(503, 'Access token is not configured');
      const provided = request.headers.get('Authorization')?.match(/^Bearer ([A-Za-z0-9_-]{32,128})$/)?.[1];
      if (!provided || !await sameToken(provided, env.LINK_RELAY_TOKEN)) fail(401, 'Invalid access token');
      if (uploading) {
        const upstream = await wormholeUpload(request, path);
        return new Response(upstream.body, { status: upstream.status, headers });
      }
      const body = await readJson(request);
      let url;
      try { url = validateWormholeUrl(body?.url); } catch { fail(400, 'Complete Wormhole link required'); }
      return reply(await notify(url, env));
    } catch (error) {
      // Never return/log provider exceptions, request bodies, keys or webhook URLs.
      return reply({ accepted: false, error: error instanceof RequestError ? error.message : 'Forwarding not confirmed' }, error instanceof RequestError ? error.status : 502);
    }
  },
};
