import { validateRelaySettings } from './link-relay.js';
import { validateWormholeUrl } from './wormhole-link.js';

const encoder = new TextEncoder(), PIECE = 8 * 1024 * 1024;
const join = parts => {
  const bytes = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let at = 0;
  for (const part of parts) { bytes.set(part, at); at += part.length; }
  return bytes;
};
function bencode(value) {
  if (typeof value === 'number') return encoder.encode(`i${value}e`);
  if (typeof value === 'string') value = encoder.encode(value);
  if (value instanceof Uint8Array) return join([encoder.encode(`${value.length}:`), value]);
  if (Array.isArray(value)) return join([encoder.encode('l'), ...value.map(bencode), encoder.encode('e')]);
  return join([encoder.encode('d'), ...Object.keys(value).sort().flatMap(key => [bencode(key), bencode(value[key])]), encoder.encode('e')]);
}
const base64 = bytes => {
  let text = '';
  for (let at = 0; at < bytes.length; at += 16384) text += String.fromCharCode(...bytes.subarray(at, at + 16384));
  return btoa(text);
};
const hex = bytes => [...bytes].map(byte => byte.toString(16).padStart(2, '0')).join('');
const sha1 = async bytes => new Uint8Array(await crypto.subtle.digest('SHA-1', bytes));

// The official RFC8188 implementation accepts an explicit per-file salt. Both
// passes read this SAME immutable Blob with this salt/key; no plaintext changes
// may reuse the pair. This avoids retaining a second whole archive in memory.
async function* pieces(file, key, salt, encryptStream, size, signal) {
  const reader = encryptStream(file.stream(), key, undefined, salt).getReader();
  let bytes = new Uint8Array(Math.min(PIECE, size)), used = 0, ended = false;
  try {
    for (;;) {
      signal?.throwIfAborted();
      const { done, value } = await reader.read();
      if (done) { ended = true; break; }
      for (let at = 0; at < value.length;) {
        const length = Math.min(bytes.length - used, value.length - at);
        bytes.set(value.subarray(at, at + length), used); at += length; used += length;
        if (used === bytes.length) { yield bytes; bytes = new Uint8Array(Math.min(PIECE, size)); used = 0; }
      }
    }
    if (used) yield bytes.subarray(0, used);
  } finally {
    if (!ended) await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export async function uploadWormholeZip({ file, endpoint, token, signal, onProgress = () => {} }) {
  const target = validateRelaySettings(endpoint, token);
  if (!(file instanceof Blob) || !file.size || file.size > 0xffffffff
    || typeof file.name !== 'string' || !/^[^\\/\0]{1,128}\.zip$/i.test(file.name)) throw new Error('Gültige ZIP-Datei für den Upload erforderlich.');
  signal?.throwIfAborted();
  const [{ Keychain, encryptedSize }, { encryptStream }] = await Promise.all([
    import('./vendor/wormhole-crypto/index.js'), import('./vendor/wormhole-crypto/lib/ece.js'),
  ]);
  const master = crypto.getRandomValues(new Uint8Array(16)), salt = crypto.getRandomValues(new Uint8Array(16));
  const metadata = new Keychain(master);
  const key = await crypto.subtle.importKey('raw', master, 'HKDF', false, ['deriveBits', 'deriveKey']);
  const size = encryptedSize(file.size), hashes = [];
  let completed = 0;
  for await (const bytes of pieces(file, key, salt, encryptStream, size, signal)) {
    hashes.push(await sha1(bytes)); completed += bytes.length;
    onProgress({ phase: 'encrypting', completed, total: size });
  }
  if (completed !== size) throw new Error('Verschlüsselter Upload hat eine unerwartete Größe.');
  const request = async (path, method, body, extra = {}, json = false) => {
    signal?.throwIfAborted();
    const response = await fetch(new URL(path, target).href, {
      method, redirect: 'error', referrerPolicy: 'no-referrer',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', ...extra },
      body: body instanceof Uint8Array ? body : JSON.stringify(body),
      signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(120000)]) : AbortSignal.timeout(120000),
    });
    if (!response.ok) throw new Error(`Wormhole-Upload nicht bestätigt (HTTP ${response.status}).`);
    if (json) return response.json();
    await response.arrayBuffer();
  };
  const room = await request('/wormhole/api/room', 'POST', { readerToken: await metadata.authTokenB64(), salt: metadata.saltB64 }, {}, true);
  if (!/^[A-Za-z0-9_-]{1,128}$/.test(room?.id ?? '') || !/^[A-Za-z0-9+/]{22}==$/.test(room?.writerToken ?? '')) throw new Error('Wormhole-Upload: ungültige Raum-Antwort.');
  const authorization = { 'X-Wormhole-Authorization': `Bearer sync-v1 ${room.writerToken}` };
  const api = `/wormhole/api/room/${room.id}`;
  await request(`${api}/uploader-online`, 'PATCH', undefined, authorization);
  const info = {
    length: size, name: file.name, nonce: hex(crypto.getRandomValues(new Uint8Array(16))),
    'piece length': PIECE, pieces: join(hashes), private: 1,
  };
  const torrent = bencode({ announce: 'wss://wormhole.app/websocket', 'announce-list': [['wss://wormhole.app/websocket']],
    'created by': 'Atmungsraum', 'creation date': Math.floor(Date.now() / 1000), info, private: 1, 'url-list': [] });
  await request(api, 'PATCH', { infoHash: hex(await sha1(bencode(info))),
    encryptedTorrentFile: base64(await metadata.encryptMeta(torrent)), multiFile: false, sizeMb: Math.ceil(size / 1e6) }, authorization);
  let index = 0; completed = 0;
  for await (const bytes of pieces(file, key, salt, encryptStream, size, signal)) {
    const hash = hex(await sha1(bytes));
    if (hash !== hex(hashes[index])) throw new Error('Wormhole-Upload: verschlüsselte Daten haben sich geändert.');
    const tokens = await request(`${api}/b2/auth-upload`, 'POST', { numTokens: 1 }, authorization, true);
    if (!Array.isArray(tokens) || !tokens[0]?.uploadUrl || !tokens[0]?.authorizationToken) throw new Error('Wormhole-Upload: Upload-Berechtigung fehlt.');
    await request('/wormhole/chunk', 'POST', bytes, {
      'Content-Type': 'application/octet-stream', 'X-Wormhole-Upload-URL': tokens[0].uploadUrl,
      'X-Wormhole-B2-Authorization': tokens[0].authorizationToken,
      'X-Bz-File-Name': `${room.id}/${index}`, 'X-Bz-Content-Sha1': hash,
    });
    index++; completed += bytes.length;
    onProgress({ phase: 'uploading', completed, total: size });
  }
  await request(`${api}/b2/finish-upload`, 'POST', { success: true }, authorization);
  return { url: validateWormholeUrl(`https://wormhole.app/${room.id}#${metadata.keyB64}`),
    roomId: room.id, lifetime: room.lifetime, maxDownloads: room.maxDownloads };
}
