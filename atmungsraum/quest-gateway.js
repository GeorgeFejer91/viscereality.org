const serviceId = '9a7b2001-7d6a-4b7f-9d4a-6f7c0a020101';
const serialId = '9a7b2001-7d6a-4b7f-9d4a-6f7c0a020102';
const metadataId = '9a7b2001-7d6a-4b7f-9d4a-6f7c0a020103';
const encode = new TextEncoder();
const decode = new TextDecoder('utf-8', { fatal: true });
const hex32 = /^[a-f0-9]{32}$/;
const hex64 = /^[a-f0-9]{64}$/;
const serialPattern = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const positive = /^[1-9][0-9]{0,18}$/;
const phases = new Set(['IDLE', 'STARTING', 'ARMING', 'ARMED', 'RUNNING', 'PAUSED', 'RECORDING', 'FINALIZING', 'SAVING', 'RECOVERY', 'ERROR']);
const toHex = (bytes) => Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
export const operationId = () => toHex(crypto.getRandomValues(new Uint8Array(8)));

export function questEndpoint(input) {
  const candidate = input.trim();
  const match = /^(?:http:\/\/)?((?:[0-9]{1,3}\.){3}[0-9]{1,3})(?::8766\/v1\/gateway)?$/.exec(candidate);
  const octets = match?.[1].split('.').map(Number) ?? [];
  const [a, b] = octets;
  const privateAddress = a === 10 || a === 172 && b >= 16 && b <= 31 || a === 192 && b === 168;
  if (!privateAddress || octets.some((octet) => octet > 255) || octets.join('.') !== match[1]) {
    throw Error('Gib die lokale IPv4-Adresse der Quest ein.');
  }
  return `http://${match[1]}:8766/v1/gateway`;
}

function identity(raw) {
  if (raw?.v !== 1 || !hex32.test(raw.device) || !serialPattern.test(raw.serial)
    || typeof raw.epoch !== 'string' || !positive.test(raw.epoch)
    || typeof raw.native_epoch !== 'string' || !positive.test(raw.native_epoch)) {
    throw Error('Die Quest hat keine gültige Verbindungskennung gemeldet.');
  }
  return { device: raw.device, serial: raw.serial, epoch: raw.epoch, native_epoch: raw.native_epoch };
}

export function sameQuest(left, right) {
  return Boolean(left && right && ['device', 'serial', 'epoch', 'native_epoch'].every((key) => left[key] === right[key]));
}

export async function discoverQuest(bluetooth, lifetime) {
  if (!bluetooth) throw Error('Bluetooth-Suche ist hier nicht verfügbar. Gib die Quest-Adresse ein.');
  const device = await bluetooth.requestDevice({ filters: [{ services: [serviceId] }] });
  const signal = AbortSignal.any([lifetime, AbortSignal.timeout(12000)]);
  signal.throwIfAborted();
  const waitForGatt = async (operation) => {
    let abort;
    const cancelled = new Promise((_, reject) => {
      abort = () => reject(signal.reason);
      signal.addEventListener('abort', abort, { once: true });
    });
    try {
      signal.throwIfAborted();
      return await Promise.race([operation, cancelled]);
    } finally { signal.removeEventListener('abort', abort); }
  };
  try {
    const connection = device.gatt.connect().then((gatt) => {
      if (signal.aborted) { gatt.disconnect(); signal.throwIfAborted(); }
      return gatt;
    });
    const gatt = await waitForGatt(connection);
    const service = await waitForGatt(gatt.getPrimaryService(serviceId));
    const serialCharacteristic = await waitForGatt(service.getCharacteristic(serialId));
    const serialBytes = await waitForGatt(serialCharacteristic.readValue());
    const metadataCharacteristic = await waitForGatt(service.getCharacteristic(metadataId));
    const metadataBytes = await waitForGatt(metadataCharacteristic.readValue());
    if (serialBytes.byteLength > 64 || metadataBytes.byteLength > 512) throw Error('Ungültige Quest-Suchdaten.');
    const serial = decode.decode(serialBytes);
    const metadata = JSON.parse(decode.decode(metadataBytes));
    const found = identity(metadata);
    if (metadata.mode !== 'remote' || found.serial !== serial || !Array.isArray(metadata.gateways)
      || metadata.gateways.length < 1 || metadata.gateways.length > 4) {
      throw Error('Diese Quest meldet keine erreichbare WLAN-Adresse im Fernmodus.');
    }
    return { owner: found, endpoints: metadata.gateways.map(questEndpoint) };
  } finally { if (device.gatt.connected) device.gatt.disconnect(); }
}

async function readJson(response) {
  if (!response.ok || !response.body) throw Error('Die Quest-Verbindung wurde abgelehnt.');
  const reader = response.body.getReader();
  const buffer = new Uint8Array(65536);
  let length = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (length + value.byteLength >= buffer.length) throw Error('Ungültige Quest-Antwort.');
      buffer.set(value, length);
      length += value.byteLength;
    }
    return JSON.parse(decode.decode(buffer.subarray(0, length)));
  } finally { await reader.cancel().catch(() => {}); }
}

export async function joinQuest({ endpoint, serial, build, discovery, signal, fetcher = fetch }) {
  const url = questEndpoint(endpoint);
  if (!serialPattern.test(serial) || !hex64.test(build)) throw Error('Prüfe die Seriennummer und den Build-Prüfcode.');
  const lifetime = new AbortController();
  const fetchJson = async (body, requestSignal) => readJson(await fetcher(url, {
    method: body ? 'POST' : 'GET', body: body ? JSON.stringify(body) : undefined,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    mode: 'cors', cache: 'no-store', credentials: 'omit', redirect: 'error',
    referrerPolicy: 'no-referrer', targetAddressSpace: 'local',
    signal: AbortSignal.any([lifetime.signal, AbortSignal.timeout(6500), ...[signal, requestSignal].filter(Boolean)]),
  }));
  try {
    const raw = await fetchJson();
    const owner = identity(raw);
    if (raw.enabled !== true || raw.remote_available !== true || owner.serial !== serial
      || discovery && !sameQuest(owner, discovery)) throw Error('Quest oder Fernmodus stimmt nicht mit der Auswahl überein.');
    const ticket = await fetchJson({ v: 1, action: 'connect', device: owner.device, epoch: owner.epoch });
    if (ticket.v !== 1 || ticket.device !== owner.device || ticket.epoch !== owner.epoch
      || ticket.serial !== owner.serial || !hex32.test(ticket.controller) || !hex64.test(ticket.key)) {
      throw Error('Die Quest-Anmeldung konnte nicht geprüft werden.');
    }
    const controller = ticket.controller;
    const signingKey = await crypto.subtle.importKey('raw', encode.encode(ticket.key), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign', 'verify']);
    const exchange = async (action, body, requestSignal) => {
      requestSignal?.throwIfAborted();
      const nonce = toHex(crypto.getRandomValues(new Uint8Array(16)));
      const payload = JSON.stringify(body);
      const mac = toHex(new Uint8Array(await crypto.subtle.sign('HMAC', signingKey, encode.encode(`RQG1|${controller}|${owner.epoch}|${nonce}|${action}|${payload}`))));
      requestSignal?.throwIfAborted();
      const reply = await fetchJson({ v: 1, controller, epoch: owner.epoch, nonce, action, payload, mac }, requestSignal);
      if (reply?.v !== 1 || reply.device !== owner.device || reply.epoch !== owner.epoch
        || reply.nonce !== nonce || reply.action !== action || typeof reply.payload !== 'string' || !hex64.test(reply.mac)
        || !await crypto.subtle.verify('HMAC', signingKey, Uint8Array.from(reply.mac.match(/../g), (byte) => parseInt(byte, 16)),
          encode.encode(`RQG1-REPLY|${controller}|${owner.device}|${owner.epoch}|${nonce}|${action}|${reply.payload}`))) {
        throw Error('Die Quest-Antwort konnte nicht bestätigt werden.');
      }
      return JSON.parse(reply.payload);
    };
    let witness = 0n;
    const status = async () => {
      const value = await exchange('status', {});
      const observation = value?.o;
      if (value?.v !== 1 || value.wifi_snapshot !== 1 || value.build?.v !== 1
        || value.build.packaging_fingerprint !== build || value.recordings !== 0) {
        throw Error('Der Build-Prüfcode passt nicht zum Probelauf.');
      }
      if (value.control_mode !== 'remote' || observation?.v !== 1 || observation.s !== 'observed'
        || observation.e !== owner.native_epoch || typeof observation.i !== 'string' || !positive.test(observation.i)
        || BigInt(observation.i) <= witness || !Number.isSafeInteger(observation.a) || observation.a < 0 || observation.a > 1500
        || !Number.isSafeInteger(value.g) || value.g < 0 || observation.g !== value.g
        || !Number.isSafeInteger(value.r) || value.r < 0 || observation.r !== value.r
        || !Number.isSafeInteger(value.ms) || value.ms < 0 || !phases.has(value.p)
        || typeof value.pending !== 'boolean' || typeof value.rec !== 'boolean') {
        throw Error('Kein aktueller Status der Quest verfügbar.');
      }
      witness = BigInt(observation.i);
      return value;
    };
    const initial = await status();
    return { owner, initial, status, command: (command, requestSignal) => exchange('command', command, requestSignal),
      receipt: (id) => exchange('receipt', { id }), close: () => lifetime.abort() };
  } catch (error) { lifetime.abort(); throw error; }
}
