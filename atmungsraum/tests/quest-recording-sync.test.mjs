import test from 'node:test';
import assert from 'node:assert/strict';
import { questRecordings } from '../quest-data.js';

const old = 'utc-ns-00000000000000000042';
const recent = 'utc-ns-00000000000000000043';
const signal = new AbortController().signal;

test('lists all exportable saved recordings across idle gateway pages', async () => {
  const calls = [];
  const gateway = { recordings: async query => {
    calls.push(query.before);
    return query.before
      ? { v: 1, state: 'ready', entries: [{ id: old, exportable: false }], next_before: null }
      : { v: 1, state: 'ready', entries: [{ id: recent, exportable: true, participant_csv: true }], next_before: recent };
  } };
  const ids = [];
  for await (const entry of questRecordings(gateway, signal)) ids.push(entry);
  assert.deepEqual(ids, [{ id: recent, participantCsv: true }]);
  assert.deepEqual(calls, ['', recent]);
});

test('rejects pagination that repeats an ID or moves forward', async () => {
  const gateway = { recordings: async () => ({ v: 1, state: 'ready',
    entries: [{ id: recent, exportable: true }], next_before: recent }) };
  await assert.rejects(async () => {
    for await (const entry of questRecordings(gateway, signal)) assert.equal(entry.id, recent);
  }, /Aufnahmeliste/);
});
