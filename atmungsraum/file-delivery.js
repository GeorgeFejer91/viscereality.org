export const deliveryKey = 'atmungsraum-file-delivery-v1';

export function readDeliveryHistory(storage) {
  try {
    const raw = storage.getItem(deliveryKey);
    if (raw === null) return { version: 1, sent: {} };
    const history = JSON.parse(raw);
    if (history?.version !== 1 || !history.sent || typeof history.sent !== 'object' || Array.isArray(history.sent)
      || Object.entries(history.sent).some(([key, date]) => !/^[a-f0-9]{64}$/.test(key)
        || typeof date !== 'string' || !Number.isFinite(Date.parse(date)))) throw new Error('Invalid history');
    return history;
  } catch {
    throw new Error('Der Versandverlauf konnte nicht gelesen werden.');
  }
}

export async function deliveryEntries(records, signal) {
  const entries = [];
  for (const { session_id, filename, csv } of records) {
    signal?.throwIfAborted();
    const bytes = await new Blob([JSON.stringify([session_id, filename]), csv]).arrayBuffer();
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
    const key = [...digest].map(byte => byte.toString(16).padStart(2, '0')).join('');
    entries.push({ key, record: { session_id, filename, csv } });
  }
  signal?.throwIfAborted();
  return entries;
}

// Call while holding the browser's delivery lock. Only confirmed entries are added.
export function recordDelivery(storage, history, entries) {
  const sent = { ...history.sent }, date = new Date().toISOString();
  for (const { key } of entries) sent[key] = date;
  try { storage.setItem(deliveryKey, JSON.stringify({ version: 1, sent })); }
  catch { throw new Error('Der Versandverlauf konnte nicht gespeichert werden.'); }
}
