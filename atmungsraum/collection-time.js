import { datedSessionParts } from './data-logging.js?v=20261118-dataset-sync';

const endpoint = 'https://www.timeapi.io/api/Time/current/zone?timeZone=Etc%2FUTC';
export const recordingHeadBytes = 32 * 1024;

function utcMillis(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z$/.test(value)) return NaN;
  const millis = Date.parse(value);
  return Number.isFinite(millis) && new Date(millis).toISOString().slice(0, 19) === value.slice(0, 19) ? millis : NaN;
}

export async function captureUtc({ signal, fetcher = fetch, wallNow = Date.now,
  monotonicNow = () => performance.now() } = {}) {
  const wall = wallNow();
  const before = monotonicNow();
  try {
    const response = await fetcher(endpoint, { cache: 'no-store', credentials: 'omit', redirect: 'error',
      referrerPolicy: 'no-referrer', signal: AbortSignal.any([AbortSignal.timeout(1800), signal].filter(Boolean)) });
    if (response.ok) {
      const { dateTime, timeZone } = await response.json();
      const roundTrip = monotonicNow() - before;
      const epoch = utcMillis(dateTime + 'Z');
      if (timeZone === 'Etc/UTC' && Number.isFinite(epoch) && roundTrip >= 0 && roundTrip <= 1800) {
        return { utc: new Date(epoch - roundTrip / 2).toISOString(), source: 'timeapi.io',
          round_trip_ms: Math.ceil(roundTrip) };
      }
    }
  } catch { /* Use the phone's event time if the reference is unavailable. */ }
  signal?.throwIfAborted();
  return { utc: new Date(wall).toISOString(), source: 'phone-system-clock' };
}

export function collectionStart({ id, name, session_id }, bytes) {
  const nameParts = datedSessionParts(name);
  if (nameParts && bytes) {
    const [, participant, date, scenario, run] = nameParts;
    const [, ...rows] = new TextDecoder().decode(bytes.subarray(0, recordingHeadBytes)).split('\n');
    for (const row of rows) {
      const [version, who, uuid, index, condition, type, utc, elapsed] = row.trimEnd().split(',', 8);
      if (version === '4' && who === participant && (!session_id || uuid === session_id)
        && index === run && condition === scenario && type === 'official-start' && elapsed === '0'
        && utc?.startsWith(date + 'T') && Number.isFinite(utcMillis(utc))) return { utc, source: 'quest-official-start' };
    }
  }
  // Historical exports can lack an explicit official-start row.
  const nanos = /^utc-ns-(\d{20})$/.exec(id)?.[1];
  if (!nanos) throw Error('Ungültiger Quest-Zeitstempel.');
  return { utc: new Date(Number(BigInt(nanos) / 1000000n)).toISOString(), source: 'quest-recording-key' };
}

export async function questionnaireStart(file) {
  const saved = JSON.parse(await file.text());
  const utc = saved.started_at_utc ?? saved.submitted_at_utc;
  if (!Number.isFinite(utcMillis(utc))) throw Error('Ungültiger Fragebogen-Zeitstempel.');
  return { utc, source: saved.clock_source === 'timeapi.io' ? 'timeapi.io' : 'phone-system-clock' };
}
