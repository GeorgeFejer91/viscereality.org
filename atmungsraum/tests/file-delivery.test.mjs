import test from 'node:test';
import assert from 'node:assert/strict';

const first = { session_id: 'one', filename: 'session.csv', csv: 'value\r\n1\r\n' };
const second = { session_id: 'two', filename: 'session.csv', csv: 'value\r\n2\r\n' };
const memory = () => {
  const values = new Map();
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
};

test('the JSON receipt excludes only confirmed exact file revisions across reloads', async () => {
  const { deliveryKey, readDeliveryHistory, deliveryEntries, recordDelivery } = await import('../file-delivery.js');
  const storage = memory();
  const entries = await deliveryEntries([first, second]);
  const before = readDeliveryHistory(storage);
  assert.deepEqual(entries.filter(entry => !before.sent[entry.key]).map(entry => entry.record), [first, second]);
  recordDelivery(storage, before, entries.slice(0, 1));
  const json = JSON.parse(storage.getItem(deliveryKey));
  assert.equal(json.version, 1);
  assert.deepEqual(Object.keys(json.sent), [entries[0].key]);
  assert.ok(Number.isFinite(Date.parse(json.sent[entries[0].key])));
  assert.ok(!JSON.stringify(json).includes(first.filename));
  const revised = { ...first, csv: first.csv + '3\r\n' };
  const next = await deliveryEntries([first, second, revised]);
  assert.deepEqual(next.filter(entry => !readDeliveryHistory(storage).sent[entry.key]).map(entry => entry.record), [second, revised]);
  recordDelivery(storage, readDeliveryHistory(storage), next);
  assert.ok(next.every(entry => readDeliveryHistory(storage).sent[entry.key]));
});

test('corrupt JSON or blocked browser storage fails visibly without erasing history', async () => {
  const { deliveryKey, readDeliveryHistory, recordDelivery } = await import('../file-delivery.js');
  const storage = memory();
  for (const json of ['{', '{"version":2,"sent":{}}', '{"version":1,"sent":{"__proto__":"invalid"}}']) {
    storage.setItem(deliveryKey, json);
    assert.throws(() => readDeliveryHistory(storage), /Versandverlauf/);
    assert.equal(storage.getItem(deliveryKey), json);
  }
  assert.throws(() => recordDelivery({ setItem() { throw Error('quota'); } }, {version:1,sent:{}}, []), /Versandverlauf/);
});
