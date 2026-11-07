import { validateWormholeUrl } from './wormhole-link.js';

export function validateDiscordWebhook(value) {
  let target;
  try { target = new URL(value.trim()); } catch { /* Validate below. */ }
  if (!target || target.origin !== 'https://discord.com' || target.username || target.password || target.search || target.hash
    || !/^\/api(?:\/v10)?\/webhooks\/\d{1,20}\/[A-Za-z0-9._-]{8,128}$/.test(target.pathname)) {
    throw new Error('Gültigen Discord-Webhook eintragen.');
  }
  return target;
}

export async function sendDiscordLink({ webhook, url, signal }) {
  const target = validateDiscordWebhook(webhook), link = validateWormholeUrl(url);
  target.searchParams.set('wait', 'true');
  try {
    const response = await fetch(target.href, {
      method: 'POST', redirect: 'error', credentials: 'omit', referrerPolicy: 'no-referrer',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: `Atmungsraum-Daten\n<${link}>`, allowed_mentions: { parse: [] } }),
      signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(15000)]) : AbortSignal.timeout(15000),
    });
    const result = await response.json();
    if (!response.ok || typeof result?.id !== 'string' || !result.id) throw new Error('Missing acknowledgement');
    return { accepted: true, destination: 'discord', message_id: result.id };
  } catch (error) {
    const failure = new Error('Discord-Zustellung nicht bestätigt. Ziel prüfen, bevor du erneut sendest.');
    if (['AbortError', 'TimeoutError'].includes(error.name)) failure.name = error.name;
    throw failure;
  }
}

export function validateRelaySettings(endpoint, token) {
  let target;
  try { target = new URL(endpoint.trim()); } catch { /* Validate below. */ }
  if (!target || target.protocol !== 'https:' || target.port || target.username || target.password
    || target.search || target.hash || !['/', '/notify'].includes(target.pathname)
    || !(target.hostname.endsWith('.onrender.com') || target.hostname.endsWith('.workers.dev')
      || target.hostname === 'viscereality.org' || target.hostname.endsWith('.viscereality.org'))) {
    throw new Error('HTTPS-Adresse des Upload-Dienstes eintragen.');
  }
  if (!/^[A-Za-z0-9_-]{32,128}$/.test(token ?? '')) throw new Error('Zugangsschlüssel des Upload-Dienstes eintragen.');
  target.pathname = '/';
  return target;
}

export async function forwardWormholeLink({ endpoint, token, url, signal }) {
  const target = validateRelaySettings(endpoint, token);
  const link = validateWormholeUrl(url);
  target.pathname = '/notify';
  const response = await fetch(target.href, {
    method: 'POST', redirect: 'error', referrerPolicy: 'no-referrer',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ url: link }),
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(15000)]) : AbortSignal.timeout(15000),
  });
  let result;
  try { result = await response.json(); } catch { /* No acknowledgement. */ }
  if (!response.ok || result?.accepted !== true || !['email', 'discord', 'telegram'].includes(result.destination)
    || typeof result.message_id !== 'string' || !result.message_id) {
    throw new Error('Weiterleitung wurde nicht bestätigt. Einstellungen prüfen und erneut versuchen.');
  }
  return result;
}
