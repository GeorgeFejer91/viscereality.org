import { validateWormholeUrl } from './wormhole-link.js';

export function validateRelaySettings(endpoint, token) {
  let target;
  try { target = new URL(endpoint.trim()); } catch { /* Validate below. */ }
  if (!target || target.protocol !== 'https:' || target.port || target.username || target.password
    || target.search || target.hash || !['/', '/notify'].includes(target.pathname)
    || !(target.hostname.endsWith('.workers.dev') || target.hostname === 'viscereality.org' || target.hostname.endsWith('.viscereality.org'))) {
    throw new Error('HTTPS-Adresse des Cloudflare-Workers eintragen.');
  }
  if (!/^[A-Za-z0-9_-]{32,128}$/.test(token ?? '')) throw new Error('Zugangsschlüssel des Workers eintragen.');
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
