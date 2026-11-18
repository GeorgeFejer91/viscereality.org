import { captureUtc, collectionStart } from './collection-time.js?v=1';
import { ensureDriveChild } from './quest-drive.js?v=5';

export const hardwarePairKey = 'atmungsraum-drive-hardware-pairs';
const phoneIdKey = 'atmungsraum-drive-phone-id';

export async function drivePairIdentity(owner, storage = localStorage) {
  let pairs;
  try { pairs = JSON.parse(storage.getItem(hardwarePairKey)); } catch { /* The installation identity is still usable. */ }
  const provisioned = Array.isArray(pairs) && pairs.find(pair => pair?.device === owner.device
    && pair.serial === owner.serial && /^[a-f0-9]{24}$/.test(pair.id));
  if (provisioned) return { id: provisioned.id, source: 'usb-hardware-pair' };
  let phone = storage.getItem(phoneIdKey);
  if (!phone) {
    phone = crypto.randomUUID();
    storage.setItem(phoneIdKey, phone);
  }
  const input = new TextEncoder().encode(JSON.stringify(['atmungsraum-installation-pair-v1', owner.device, owner.serial, phone]));
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', input));
  return { id: Array.from(digest.subarray(0, 12), byte => byte.toString(16).padStart(2, '0')).join(''),
    source: 'app-installation-pair' };
}

export function ensureDrivePair({ drive, store, root, pair, signal }) {
  return ensureDriveChild({ drive, store, parent: root, key: 'pair-folder/' + pair.id,
    name: 'Pair_' + pair.id, properties: { pairId: pair.id, pairIdentitySource: pair.source }, signal, reuse: true });
}

export async function datasetDestination({ drive, store, root, pair, id, start, observed, signal }) {
  const key = id + '/dataset-folder';
  const saved = await store.readDelivery(key);
  if (saved) pair = { id: saved.properties.pairId, source: saved.properties.pairIdentitySource };
  const pairFolder = await ensureDrivePair({ drive, store, root, pair, signal });
  if (saved) return ensureDriveChild({ drive, store, key, ...saved, parent: pairFolder, signal });
  start ??= collectionStart({ id });
  const reference = observed ?? await captureUtc({ signal });
  const name = start.utc.replaceAll(':', '-') + '__' + id;
  const properties = { pairId: pair.id, pairIdentitySource: pair.source, questRecordingId: id,
    collectionStartedAtUtc: start.utc, collectionClockSource: start.source,
    phoneObservedAtUtc: reference.utc, phoneClockSource: reference.source,
    phoneObservation: observed ? 'live-prefix' : 'final-import',
    ...(reference.round_trip_ms === undefined ? {} : { phoneClockRoundTripMs: String(reference.round_trip_ms) }) };
  return ensureDriveChild({ drive, store, key, parent: pairFolder, name, properties, signal });
}
