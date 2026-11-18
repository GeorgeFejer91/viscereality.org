import { readJson } from './quest-gateway.js?v=17';

const utf8 = new TextEncoder();
const decode = new TextDecoder('utf-8', { fatal: true });
const idValue = /^[a-f0-9]{32}$/;
const keyValue = /^[a-f0-9]{64}$/;
const epochValue = /^[1-9][0-9]{0,18}$/;
const publicKeyValue = /^[A-Za-z0-9_-]{122}$/;
const rejected = () => Error('Die Freigabe für dieses Smartphone konnte nicht geprüft werden.');
const b64 = bytes => btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

function unbase64(text) {
  const bytes = Uint8Array.from(atob(text.replace(/-/g, '+').replace(/_/g, '/')), char => char.charCodeAt(0));
  if (b64(bytes) !== text) throw rejected();
  return bytes;
}

export async function recoverCommissionedEnrollment({ discovery, channel, broker, signal }) {
  const { owner, rtc } = discovery;
  if (owner?.identity_kind !== 'app-installation' || !idValue.test(owner.device) || !idValue.test(rtc?.r)
    || !epochValue.test(owner.epoch) || !epochValue.test(owner.native_epoch)) throw rejected();
  signal.throwIfAborted();
  const keys = await crypto.subtle.generateKey({ name: 'ECDH', namedCurve: 'P-256' }, false, ['deriveBits']);
  const publicKey = b64(new Uint8Array(await crypto.subtle.exportKey('spki', keys.publicKey)));
  const id = Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join('');
  const offer = { device: owner.device, epoch: owner.epoch, native_epoch: owner.native_epoch,
    room: rtc.r, id, public_key: publicKey };
  const frame = await broker.authorize(offer, signal);
  if (frame?.v !== 2 || frame.id !== id || frame.public_key !== publicKey || !keyValue.test(frame.tag)
    || !keyValue.test(frame.proof) || Object.keys(frame).length !== 5) throw rejected();
  signal.throwIfAborted();
  const reply = await channel.request(frame, signal);
  if (reply?.v !== 2 || reply.id !== id || reply.tag !== frame.tag || !publicKeyValue.test(reply.public_key)
    || !/^[A-Za-z0-9_-]{16}$/.test(reply.iv) || !/^[A-Za-z0-9_-]{24,512}$/.test(reply.data)
    || !keyValue.test(reply.proof) || Object.keys(reply).length !== 7) throw rejected();
  if (await broker.verify(frame, reply, signal) !== true) throw rejected();
  signal.throwIfAborted();
  const context = `ATMB1|RQG1-RTC1|${owner.device}|${owner.epoch}|${owner.native_epoch}|${rtc.r}`;
  const transcript = utf8.encode(`${context}|request|${id}|${frame.tag}|${publicKey}|reply|${reply.public_key}`);
  const remote = await crypto.subtle.importKey('spki', unbase64(reply.public_key),
    { name: 'ECDH', namedCurve: 'P-256' }, false, []);
  const shared = new Uint8Array(await crypto.subtle.deriveBits({ name: 'ECDH', public: remote }, keys.privateKey, 256));
  try {
    const material = await crypto.subtle.importKey('raw', shared, 'HKDF', false, ['deriveKey']);
    const key = await crypto.subtle.deriveKey({ name: 'HKDF', hash: 'SHA-256',
      salt: await crypto.subtle.digest('SHA-256', transcript), info: utf8.encode('ATMB1/ticket') },
    material, { name: 'AES-GCM', length: 256 }, false, ['decrypt']);
    const ticket = new Uint8Array(await crypto.subtle.decrypt({ name: 'AES-GCM',
      iv: unbase64(reply.iv), additionalData: transcript }, key, unbase64(reply.data)));
    try {
      signal.throwIfAborted();
      const value = JSON.parse(decode.decode(ticket));
      if (value?.device !== owner.device || value.epoch !== owner.epoch || !idValue.test(value.controller)
        || !keyValue.test(value.key) || Object.keys(value).length !== 4) throw rejected();
      return { controller: value.controller, key: value.key };
    } finally { ticket.fill(0); }
  } finally { shared.fill(0); }
}

const provisioningEndpoint = 'http://127.0.0.1:8767/v1/commissioning';

async function companion(path, body, signal) {
  const requestSignal = AbortSignal.any([signal, AbortSignal.timeout(2000)].filter(Boolean));
  const response = await fetch(provisioningEndpoint + path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: 'no-store', credentials: 'omit', redirect: 'error', targetAddressSpace: 'loopback', signal: requestSignal,
  });
  if (response.status === 404 && body === undefined) return null;
  return readJson(response);
}

export async function readCommissionedQuest() {
  try {
    const result = await companion('');
    if (result?.v !== 1 || result.identity_kind !== 'app-installation' || result.serial !== ''
      || !idValue.test(result.device) || Object.keys(result).length !== 4) return null;
    return { device: result.device, serial: '', identity_kind: 'app-installation', commissioned: true };
  } catch { return null; }
}

export const commissioningBroker = {
  authorize: (offer, signal) => companion('/request', offer, signal),
  async verify(request, reply, signal) {
    const result = await companion('/verify', { request, reply }, signal);
    return result?.verified === true;
  },
};
