import { sendDiscordLink, validateDiscordWebhook } from './link-relay.js?v=20261107-discord-password';
import { validateWormholeUrl } from './wormhole-link.js';

const encoder = new TextEncoder(), iterations = 600000;
const additionalData = encoder.encode('Atmungsraum Discord webhook v1');
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
export async function encryptDiscordWebhook(webhook, password) {
  const target = validateDiscordWebhook(webhook);
  if (typeof password !== 'string' || password.length < 16) throw new Error('Mindestens 16 Zeichen für das Passwort verwenden.');
  const salt = crypto.getRandomValues(new Uint8Array(16)), iv = crypto.getRandomValues(new Uint8Array(12));
  const ciphertext = await crypto.subtle.encrypt({ name: 'AES-GCM', iv, additionalData }, await keyFor(password, salt), encoder.encode(target.href));
  return { version: 1, iterations, salt: encode(salt), iv: encode(iv), ciphertext: encode(new Uint8Array(ciphertext)) };
}

export async function decryptDiscordWebhook(config, password) {
  try {
    if (config?.version !== 1 || config.iterations !== iterations) throw new Error('Invalid envelope');
    const salt = decode(config.salt, 24), iv = decode(config.iv, 16), ciphertext = decode(config.ciphertext, 1400);
    if (salt.length !== 16 || iv.length !== 12 || ciphertext.length < 17 || ciphertext.length > 1024) throw new Error('Invalid envelope');
    const plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv, additionalData }, await keyFor(password, salt), ciphertext);
    return validateDiscordWebhook(new TextDecoder('utf-8', { fatal: true }).decode(plain)).href;
  } catch {
    throw new Error('Passwort falsch oder verschlüsselte Discord-Konfiguration ungültig.');
  }
}

export async function forwardPasswordDiscordLink({ password, url, signal }) {
  const link = validateWormholeUrl(url);
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
    throw new Error('Verschlüsselte Discord-Konfiguration nicht verfügbar. Bitte erneut versuchen.');
  }
  const webhook = await decryptDiscordWebhook(config, password);
  signal?.throwIfAborted();
  return sendDiscordLink({ webhook, url: link, signal });
}
