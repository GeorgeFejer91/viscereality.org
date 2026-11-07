import test from 'node:test';
import assert from 'node:assert/strict';

const first = { session_id: 'one', filename: 'session.csv', csv: 'value\r\n1\r\n' };
const second = { session_id: 'two', filename: 'session.csv', csv: 'value\r\n2\r\n' };
const memory = () => {
  const values = new Map();
  return {
    get length() { return values.size; },
    key: index => [...values.keys()][index] ?? null,
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: key => values.delete(key),
  };
};

test('new files are selected from separate confirmed checkout receipts and the legacy history', async () => {
  const { deliveryKey, receiptPrefix, readSentFiles, deliveryEntries, prepareDeliveryReceipt } = await import('../file-delivery.js');
  const storage = memory();
  const revised = { ...first, csv: first.csv + '3\r\n' };
  const entries = await deliveryEntries([first, second, revised]);
  const [old, newlySent, edited] = entries;
  const legacy = JSON.stringify({ version: 1, sent: { [old.key]: '2026-11-07T09:00:00.000Z' } });
  storage.setItem(deliveryKey, legacy);
  assert.deepEqual(readSentFiles(storage), new Set([old.key]));

  const acknowledge = prepareDeliveryReceipt(storage, [newlySent]);
  assert.equal(storage.length, 1);
  assert.deepEqual(readSentFiles(storage), new Set([old.key]));
  acknowledge();
  const key = storage.key(1);
  assert.ok(key.startsWith(receiptPrefix));
  const raw = storage.getItem(key), json = JSON.parse(raw);
  assert.equal(json.version, 1);
  assert.deepEqual(json.files, [newlySent.key]);
  assert.ok(Number.isFinite(Date.parse(json.acceptedAt)));
  assert.ok(!raw.includes(first.filename));
  assert.deepEqual(entries.filter(entry => !readSentFiles(storage).has(entry.key)), [edited]);

  prepareDeliveryReceipt(storage, entries)();
  assert.equal(storage.length, 3);
  assert.equal(storage.getItem(key), raw);
  assert.equal(storage.getItem(deliveryKey), legacy);
  assert.deepEqual(JSON.parse(storage.getItem(storage.key(2))).files, entries.map(entry => entry.key));
  assert.deepEqual(readSentFiles(storage), new Set(entries.map(entry => entry.key)));
});

test('corrupt JSON or blocked browser storage fails visibly without erasing history', async () => {
  const { deliveryKey, receiptPrefix, readSentFiles, deliveryEntries, prepareDeliveryReceipt } = await import('../file-delivery.js');
  const storage = memory();
  for (const json of ['{', '{"version":2,"sent":{}}', '{"version":1,"sent":{"__proto__":"invalid"}}']) {
    storage.setItem(deliveryKey, json);
    assert.throws(() => readSentFiles(storage), /Versandverlauf/);
    assert.equal(storage.getItem(deliveryKey), json);
  }
  storage.removeItem(deliveryKey);
  storage.setItem(receiptPrefix + 'broken', '{"version":1,"files":["invalid"]}');
  assert.throws(() => readSentFiles(storage), /Versandverlauf/);
  const entries = await deliveryEntries([first]);
  assert.throws(() => prepareDeliveryReceipt({ setItem() { throw Error('quota'); } }, entries), /Versandverlauf/);
});
