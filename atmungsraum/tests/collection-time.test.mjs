import test from 'node:test';
import assert from 'node:assert/strict';
import { captureUtc, collectionStart } from '../collection-time.js';

const session_id = 'd2d9ddfe-2717-40b9-9995-1bd3f48ff440';
const id = 'utc-ns-01725000000000000000';
const name = 'Tes_30_N_2024-08-30_haupt-15_r1.csv';
const start = '2024-08-30T01:02:03.123456Z';
const head = uuid => new TextEncoder().encode('schema_version,participant_id,session_id,run_index,scenario_id,record_type,timestamp_utc,elapsed_us\r\n' +
  '4,Tes_30_N,' + uuid + ',1,haupt-15,official-start,' + start + ',0\r\n');

test('the CSV start supersedes the Quest recording key for the same native run', () => {
  assert.deepEqual(collectionStart({ id, name, session_id }, head(session_id)), { utc: start, source: 'quest-official-start' });
  assert.deepEqual(collectionStart({ id, name, session_id }, head('different-run')),
    { utc: new Date(1725000000000).toISOString(), source: 'quest-recording-key' });
});

test('the UTC request carries no participant data and projects server time to the capture event', async () => {
  const times = [0, 300];
  assert.deepEqual(await captureUtc({ monotonicNow: () => times.shift(), fetcher: async (url, options) => {
    assert.equal(url, 'https://www.timeapi.io/api/Time/current/zone?timeZone=Etc%2FUTC');
    assert.equal(options.credentials, 'omit');
    assert.equal(options.referrerPolicy, 'no-referrer');
    return Response.json({ timeZone: 'Etc/UTC', dateTime: '2026-11-18T09:59:00.1505432' });
  } }), { utc: '2026-11-18T09:59:00.000Z', source: 'timeapi.io', round_trip_ms: 300 });
});

test('a failed or invalid time response falls back to the event wall time', async () => {
  for (const fetcher of [
    async () => { throw new TypeError('network unavailable'); },
    async () => Response.json({ timeZone: 'Europe/Berlin', dateTime: '2026-11-18T10:00:00.000' }),
    async () => Response.json({ timeZone: 'Etc/UTC', dateTime: '2026-02-30T10:00:00.000' }),
  ]) assert.deepEqual(await captureUtc({ fetcher, wallNow: () => 0 }),
    { utc: '1970-01-01T00:00:00.000Z', source: 'phone-system-clock' });
});
