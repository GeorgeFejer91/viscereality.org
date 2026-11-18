const utf8 = new TextEncoder();
const decode = new TextDecoder('utf-8', { fatal: true });
const hex32 = /^[0-9a-f]{32}$/;
const hex64 = /^[0-9a-f]{64}$/;
const epoch = /^[1-9][0-9]{0,18}$/;
const ivBase64 = /^[A-Za-z0-9+/]{16}$/;
const cipherBase64 = /^[A-Za-z0-9+/]+={0,2}$/;
const fromBase64 = text => Uint8Array.from(atob(text), char => char.charCodeAt(0));
const toBase64 = bytes => btoa(String.fromCharCode(...bytes));
const toHex = bytes => Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
const invalid = () => Error('Die sichere WebRTC-Verbindung konnte nicht bestätigt werden.');

export async function createQuestRtcTransport({ owner, enrollment, room, channel }) {
  if (owner?.identity_kind !== 'app-installation' || typeof owner.serial !== 'string'
    || !hex32.test(owner.device) || !epoch.test(owner.epoch) || !epoch.test(owner.native_epoch)
    || !hex32.test(room) || !hex32.test(enrollment?.controller ?? '') || !hex64.test(enrollment?.key ?? '')) {
    throw invalid();
  }
  const identity = {
    v: 2, device: owner.device, serial: owner.serial, identity_kind: owner.identity_kind,
    epoch: owner.epoch, native_epoch: owner.native_epoch, enabled: true, remote_available: true,
  };
  const controller = enrollment.controller;
  const binding = `RQG1-RTC1|${owner.device}|${owner.epoch}|${owner.native_epoch}|${room}`;
  const hmac = await crypto.subtle.importKey('raw', utf8.encode(enrollment.key),
    { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const derive = label => crypto.subtle.sign('HMAC', hmac, utf8.encode(`${binding}|${label}`));
  const [tagBytes, requestBytes, replyBytes] = await Promise.all([derive('member'), derive('request'), derive('reply')]);
  const tag = toHex(new Uint8Array(tagBytes));
  const [requestKey, replyKey] = await Promise.all([
    crypto.subtle.importKey('raw', requestBytes, 'AES-GCM', false, ['encrypt']),
    crypto.subtle.importKey('raw', replyBytes, 'AES-GCM', false, ['decrypt']),
  ]);
  const lifetime = new AbortController();

  return {
    async request(body, signal) {
      const active = AbortSignal.any([lifetime.signal, signal]);
      active.throwIfAborted();
      // BLE provides the candidate identity. joinQuest still requires a verified,
      // fresh native status before the phone can use the gateway.
      if (body === undefined) return Response.json(identity);
      if (body?.v !== 1 || body.controller !== controller || body.epoch !== identity.epoch
        || !hex32.test(body.nonce)) throw invalid();
      const id = body.nonce;
      const payload = utf8.encode(JSON.stringify(body));
      if (payload.byteLength > 8192) throw invalid();
      const iv = crypto.getRandomValues(new Uint8Array(12));
      const data = await crypto.subtle.encrypt({
        name: 'AES-GCM', iv, additionalData: utf8.encode(`${binding}|request|${id}`),
      }, requestKey, payload);
      active.throwIfAborted();
      const frame = await channel.request({ v: 1, tag, id, iv: toBase64(iv), data: toBase64(new Uint8Array(data)) }, active);
      active.throwIfAborted();
      if (frame?.v !== 1 || frame.id !== id || frame.tag !== tag || Object.keys(frame).length !== 5
        || typeof frame.iv !== 'string' || !ivBase64.test(frame.iv)
        || typeof frame.data !== 'string' || frame.data.length < 24 || frame.data.length > 87404
        || !cipherBase64.test(frame.data)) throw invalid();
      const cipher = fromBase64(frame.data);
      if (cipher.byteLength > 65535 + 16 || toBase64(cipher) !== frame.data) throw invalid();
      const reply = await crypto.subtle.decrypt({
        name: 'AES-GCM', iv: fromBase64(frame.iv), additionalData: utf8.encode(`${binding}|reply|${id}`),
      }, replyKey, cipher);
      active.throwIfAborted();
      return new Response(decode.decode(reply), { headers: { 'Content-Type': 'application/json' } });
    },
    close() {
      if (lifetime.signal.aborted) return;
      lifetime.abort();
      channel.close();
    },
  };
}
