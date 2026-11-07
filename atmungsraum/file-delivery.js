export const deliveryKey = 'atmungsraum-file-delivery-v1';
export const receiptPrefix = deliveryKey + ':receipt:';
const fingerprint = /^[a-f0-9]{64}$/;
const isDate = value => typeof value === 'string' && Number.isFinite(Date.parse(value));

export function readSentFiles(storage) {
  try {
    const sent = new Set();
    const legacy = storage.getItem(deliveryKey);
    if (legacy !== null) {
      const history = JSON.parse(legacy);
      if (history?.version !== 1 || !history.sent || typeof history.sent !== 'object'
        || Array.isArray(history.sent)) throw new Error('Invalid history');
      for (const [key, date] of Object.entries(history.sent)) {
        if (!fingerprint.test(key) || !isDate(date)) throw new Error('Invalid history entry');
        sent.add(key);
      }
    }
    for (let index = 0; index < storage.length; index++) {
      const key = storage.key(index);
      if (!key?.startsWith(receiptPrefix)) continue;
      const receipt = JSON.parse(storage.getItem(key));
      if (receipt?.version !== 1 || !isDate(receipt.acceptedAt) || !Array.isArray(receipt.files)
        || !receipt.files.length || receipt.files.some(file => typeof file !== 'string' || !fingerprint.test(file))) {
        throw new Error('Invalid checkout receipt');
      }
      for (const file of receipt.files) sent.add(file);
    }
    return sent;
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

// Hold the delivery lock from the capacity check through Discord confirmation.
export function prepareDeliveryReceipt(storage, entries) {
  const id = crypto.randomUUID();
  const probeKey = deliveryKey + ':pending:' + id;
  const files = entries.map(entry => entry.key);
  const receiptJSON = () => JSON.stringify({ version: 1, acceptedAt: new Date().toISOString(), files });
  try {
    storage.setItem(probeKey, receiptJSON());
    storage.removeItem(probeKey);
  } catch {
    throw new Error('Der Versandverlauf konnte nicht gespeichert werden.');
  }
  return () => storage.setItem(receiptPrefix + id, receiptJSON());
}
