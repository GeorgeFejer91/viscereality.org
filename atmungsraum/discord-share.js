import { sendDiscordLink, validateDiscordWebhook, validateRelaySettings } from './link-relay.js?v=20261107-file-delivery';
import { validateWormholeUrl } from './wormhole-link.js';

const encoder = new TextEncoder(), iterations = 600000;
const contexts = {
  1: encoder.encode('Atmungsraum Discord webhook v1'),
  2: encoder.encode('Atmungsraum file delivery v2'),
};
const encode = bytes => btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
function decode(value, max) {
  if (typeof value !== 'string' || !/^[A-Za-z0-9_-]+$/.test(value) || value.length > max) throw new Error('Invalid envelope');
  return Uint8Array.from(atob(value.replace(/-/g, '+').replace(/_/g, '/')), char => char.charCodeAt(0));
}
async function keyFor(password, salt) {
  if (typeof password !== 'string' || !password || password.length > 1024) throw new Error('Passwort erforderlich.');
  const material = await crypto.subtle.importKey('raw', encoder.encode(password), 'PBKDF2', false, ['deriveKey']);
  return crypto.subtle.deriveKey({ name: 'PBKDF2', hash: 'SHA-256', salt, iterations }, material,
    { name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt']);
}

// Build-time helper: only the resulting envelope belongs in public Git.
export async function encryptDiscordWebhook(webhook, password, relay) {
  const target = validateDiscordWebhook(webhook);
  if (typeof password !== 'string' || password.length < 16) throw new Error('Mindestens 16 Zeichen für das Passwort verwenden.');
  const version = relay ? 2 : 1;
  const payload = relay ? JSON.stringify({ webhook: target.href, relay: validateRelay(relay) }) : target.href;
  const salt = crypto.getRandomValues(new Uint8Array(16)), iv = crypto.getRandomValues(new Uint8Array(12));
  const ciphertext = await crypto.subtle.encrypt({ name: 'AES-GCM', iv, additionalData: contexts[version] },
    await keyFor(password, salt), encoder.encode(payload));
  return { version, iterations, salt: encode(salt), iv: encode(iv), ciphertext: encode(new Uint8Array(ciphertext)) };
}

function validateRelay(relay) {
  const { endpoint, token } = relay;
  return { endpoint: validateRelaySettings(endpoint, token).origin, token };
}

async function decryptSharingConfig(config, password) {
  try {
    const version = config?.version;
    if ((version !== 1 && version !== 2) || config.iterations !== iterations) throw new Error('Invalid envelope');
    const salt = decode(config.salt, 24), iv = decode(config.iv, 16), ciphertext = decode(config.ciphertext, 1400);
    if (salt.length !== 16 || iv.length !== 12 || ciphertext.length < 17 || ciphertext.length > 1024) throw new Error('Invalid envelope');
    const plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv, additionalData: contexts[version] },
      await keyFor(password, salt), ciphertext);
    const payload = new TextDecoder('utf-8', { fatal: true }).decode(plain);
    if (version === 1) return { webhook: validateDiscordWebhook(payload).href };
    const { webhook, relay } = JSON.parse(payload);
    return { webhook: validateDiscordWebhook(webhook).href, relay: validateRelay(relay) };
  } catch {
    throw new Error('Passwort falsch oder verschlüsselte Versandkonfiguration ungültig.');
  }
}

export async function decryptDiscordWebhook(config, password) {
  return (await decryptSharingConfig(config, password)).webhook;
}

export async function unlockSharingConfig(password, signal) {
  signal?.throwIfAborted();
  let config;
  try {
    const response = await fetch(new URL('./sharing/discord-config.json', import.meta.url).href, {
      redirect: 'error', credentials: 'omit', referrerPolicy: 'no-referrer', cache: 'no-store',
      signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(15000)]) : AbortSignal.timeout(15000),
    });
    if (!response.ok) throw new Error('Missing configuration');
    config = await response.json();
  } catch {
    throw new Error('Verschlüsselte Versandkonfiguration nicht verfügbar. Bitte erneut versuchen.');
  }
  const access = await decryptSharingConfig(config, password);
  signal?.throwIfAborted();
  return access;
}

export async function unlockDiscordWebhook(password, signal) {
  return (await unlockSharingConfig(password, signal)).webhook;
}

export async function forwardPasswordDiscordLink({ password, url, signal }) {
  const link = validateWormholeUrl(url);
  const webhook = await unlockDiscordWebhook(password, signal);
  return sendDiscordLink({ webhook, url: link, signal });
}
