// Shared by the local QR renderer and the optional forwarding Worker.
export function validateWormholeUrl(value) {
  const text = typeof value === 'string' ? value.trim() : '';
  let url;
  try { url = new URL(text); } catch { /* Report the same actionable error below. */ }
  if (!url || /\s/.test(text) || url.origin !== 'https://wormhole.app'
    || url.username || url.password || url.search
    || !/^\/[A-Za-z0-9_-]{1,128}$/.test(url.pathname)
    || !/^#[A-Za-z0-9_-]{22}$/.test(url.hash)) {
    throw new Error('Vollständigen Wormhole-Link einschließlich #Schlüssel einfügen.');
  }
  return url.href;
}
