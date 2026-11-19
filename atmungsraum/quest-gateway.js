const serviceId = '9a7b2001-7d6a-4b7f-9d4a-6f7c0a020101';
const serialId = '9a7b2001-7d6a-4b7f-9d4a-6f7c0a020102';
const metadataId = '9a7b2001-7d6a-4b7f-9d4a-6f7c0a020103';
const wifiSetupId = '9a7b2001-7d6a-4b7f-9d4a-6f7c0a020104';
const encode = new TextEncoder();
const decode = new TextDecoder('utf-8', { fatal: true });
const hex32 = /^[a-f0-9]{32}$/;
const hex64 = /^[a-f0-9]{64}$/;
const serialPattern = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const positive = /^[1-9][0-9]{0,18}$/;
const phases = new Set(['IDLE', 'STARTING', 'ARMING', 'ARMED', 'RUNNING', 'PAUSED', 'RECORDING', 'FINALIZING', 'SAVING', 'RECOVERY', 'ERROR']);
const toHex = (bytes) => Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
const fromBase64 = (value) => Uint8Array.from(atob(value), (byte) => byte.charCodeAt(0));
export const operationId = () => toHex(crypto.getRandomValues(new Uint8Array(8)));

export function controllerAccessState(status) {
  const access = status?.controller_access;
  if (access === undefined) return null;
  if (!access || Object.keys(access).sort().join(',') !== 'active_phone,revision,role,v'
    || access.v !== 1 || !['active', 'waiting'].includes(access.role)
    || typeof access.active_phone !== 'string' || access.active_phone.length > 64
    || access.role === 'active' && !access.active_phone
    || /[\x00-\x1f\x7f]/.test(access.active_phone)
    || !Number.isSafeInteger(access.revision) || access.revision < 0) {
    throw Error('Die Quest hat keine gültige Steuerungsfreigabe gemeldet.');
  }
  return access;
}
export function canControlQuest(status) {
  try { return controllerAccessState(status)?.role !== 'waiting'; }
  catch { return false; }
}

export function backgroundMusicStatus(value) {
  return value?.v === 1 && ['off', 'starting', 'playing', 'failed'].includes(value.state)
    && Number.isInteger(value.volume) && value.volume >= 0 && value.volume <= 100
    && Number.isSafeInteger(value.revision) && value.revision >= 0 ? value : null;
}

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
  const installation = raw?.v === 2 && raw.identity_kind === 'app-installation';
  if (!(raw?.v === 1 || installation) || !hex32.test(raw.device)
    || typeof raw.serial !== 'string' || !(serialPattern.test(raw.serial) || installation && raw.serial === '')
    || typeof raw.epoch !== 'string' || !positive.test(raw.epoch)
    || typeof raw.native_epoch !== 'string' || !positive.test(raw.native_epoch)) {
    throw Error('Die Quest hat keine gültige Verbindungskennung gemeldet.');
  }
  return { device: raw.device, serial: raw.serial, epoch: raw.epoch, native_epoch: raw.native_epoch,
    ...(installation ? { identity_kind: 'app-installation' } : {}) };
}

export function sameQuest(left, right) {
  return Boolean(left && right && ['device', 'serial', 'identity_kind', 'epoch', 'native_epoch'].every((key) => left[key] === right[key]));
}

function hotspotCredentials(hotspot, enrollment) {
  if (!hex32.test(enrollment?.controller ?? '') || !hex64.test(enrollment?.key ?? '')) {
    throw Error('Gib Controller-ID und Quest-Schlüssel für die WLAN-Einrichtung ein.');
  }
  const ssid = encode.encode(hotspot.ssid);
  if (!ssid.length || ssid.length > 32 || /[\x00-\x1f\x7f-\x9f]/.test(hotspot.ssid)
    || !/^[\x20-\x7e]{8,63}$/.test(hotspot.passphrase)) {
    throw Error('Prüfe den Hotspot-Namen und das WLAN-Passwort.');
  }
  const credentials = new Uint8Array(1 + ssid.length + hotspot.passphrase.length);
  credentials[0] = ssid.length;
  credentials.set(ssid, 1);
  credentials.set(encode.encode(hotspot.passphrase), 1 + ssid.length);
  return credentials;
}

async function setupQuestWifi(service, owner, enrollment, credentials, waitForGatt, signal, onApproval) {
  const characteristic = await waitForGatt(service.getCharacteristic(wifiSetupId)).catch((error) => {
    if (error?.name !== 'NotFoundError') throw error;
    throw Error('Dieser Quest-Build unterstützt keine WLAN-Einrichtung per Bluetooth.');
  });
  const challenge = await waitForGatt(characteristic.readValue());
  if (challenge.byteLength !== 16) throw Error('Die Quest ist noch nicht für die WLAN-Einrichtung bereit.');
  const binding = [enrollment.controller, owner.device, owner.epoch, owner.native_epoch].join('|');
  const mac = await crypto.subtle.importKey('raw', encode.encode(enrollment.key),
    { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const secret = await crypto.subtle.sign('HMAC', mac, encode.encode('RQG1-WIFI-KEY|' + binding));
  const key = await crypto.subtle.importKey('raw', secret, 'AES-GCM', false, ['encrypt']);
  const prefix = encode.encode('RQG1-WIFI|' + binding + '|');
  const aad = new Uint8Array(prefix.length + challenge.byteLength);
  aad.set(prefix);
  aad.set(new Uint8Array(challenge.buffer, challenge.byteOffset, challenge.byteLength), prefix.length);
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const ciphertext = new Uint8Array(await crypto.subtle.encrypt({ name: 'AES-GCM', iv, additionalData: aad }, key, credentials));
  const packet = new Uint8Array(iv.length + ciphertext.length);
  packet.set(iv);
  packet.set(ciphertext, iv.length);
  for (let offset = 0; offset < packet.length; offset += 18) {
    const end = Math.min(packet.length, offset + 18);
    const fragment = new Uint8Array(2 + end - offset);
    fragment[0] = offset / 18;
    fragment[1] = Number(end === packet.length);
    fragment.set(packet.subarray(offset, end), 2);
    await waitForGatt(characteristic.writeValueWithResponse(fragment));
  }
  onApproval?.();
  for (;;) {
    const status = decode.decode(await waitForGatt(characteristic.readValue()));
    if (status !== 'wait') {
      if (status === 'failed') throw Error('Die Quest konnte dem Hotspot nicht beitreten. Prüfe den Hotspot und bestätige den Beitritt im Headset.');
      return questEndpoint(status);
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
    signal.throwIfAborted();
  }
}

export async function selectQuest(bluetooth) {
  if (!bluetooth?.requestDevice) throw Error('Öffne VR-Probe in Chrome auf dem Android-Smartphone.');
  return bluetooth.requestDevice({ filters: [{ services: [serviceId] }] });
}

export async function discoverQuest(bluetooth, lifetime, setup) {
  const device = await selectQuest(bluetooth);
  return readQuest(device, lifetime, setup && { ...setup, forceHotspot: true });
}

export async function readQuest(device, lifetime, setup) {
  const credentials = setup?.forceHotspot ? hotspotCredentials(setup.hotspot, setup.enrollment) : null;
  const signal = AbortSignal.any([lifetime, AbortSignal.timeout(credentials ? 52000 : 12000)]);
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
    const services = typeof gatt.getPrimaryServices === 'function'
      ? await waitForGatt(gatt.getPrimaryServices(serviceId))
      : [await waitForGatt(gatt.getPrimaryService(serviceId))];
    let rejected;
    let lanFallback;
    for (const service of services) {
      let discovery;
      try {
        const serialCharacteristic = await waitForGatt(service.getCharacteristic(serialId));
        const serialBytes = await waitForGatt(serialCharacteristic.readValue());
        const metadataCharacteristic = await waitForGatt(service.getCharacteristic(metadataId));
        const metadataBytes = await waitForGatt(metadataCharacteristic.readValue());
        if (serialBytes.byteLength > 64 || metadataBytes.byteLength > 512) throw Error('Ungültige Quest-Suchdaten.');
        const serial = decode.decode(serialBytes);
        const metadata = JSON.parse(decode.decode(metadataBytes));
        discovery = discoveryMetadata(metadata, serial, setup);
      } catch (error) {
        signal.throwIfAborted();
        rejected ??= error;
        continue;
      }
      if (credentials) {
        const endpoint = await setupQuestWifi(service, discovery.owner, setup.enrollment, credentials, waitForGatt, signal, setup.onApproval);
        return { owner: discovery.owner, endpoints: [endpoint] };
      }
      if (!discovery.endpoints.length && !discovery.rtc && !setup?.hotspot) {
        rejected ??= Error('Prüfe WLAN und Internetzugang der Quest.');
        continue;
      }
      if (setup?.device || discovery.rtc) return discovery;
      lanFallback ??= discovery;
    }
    if (lanFallback) return lanFallback;
    throw rejected ?? Error('Keine Quest-Suchdaten verfügbar.');
  } finally { if (device.gatt.connected) device.gatt.disconnect(); }
}

function discoveryMetadata(metadata, serial, expected) {
  const owner = identity(metadata);
  if (metadata.mode !== 'remote' || owner.serial !== serial || !Array.isArray(metadata.gateways)
    || metadata.gateways.length > 4 || metadata.v === 2 && metadata.pairing !== 1) {
    throw Error('Die Quest hat ungültige Bluetooth-Suchdaten gemeldet.');
  }
  if (expected?.serial && owner.serial !== expected.serial || expected?.device && owner.device !== expected.device) {
    throw Error('Die gewählte Quest passt nicht zur gespeicherten Verbindung.');
  }
  const { rtc } = metadata;
  if (rtc !== undefined && (owner.identity_kind !== 'app-installation' || !rtc || Object.keys(rtc).length !== 2
    || typeof rtc.r !== 'string' || !hex32.test(rtc.r) || typeof rtc.p !== 'string' || !hex64.test(rtc.p))) {
    throw Error('Die Quest hat ungültige WebRTC-Suchdaten gemeldet.');
  }
  return { owner, endpoints: metadata.gateways.map(questEndpoint),
    ...(metadata.v === 2 ? { pairing: 1 } : {}), ...(rtc ? { rtc } : {}) };
}

export async function companionAccessDenied(permissions = globalThis.navigator?.permissions) {
  try { return (await permissions?.query({ name: 'loopback-network' }))?.state === 'denied'; }
  catch { return false; }
}

export async function rediscoverQuest(settings, signal, fetcher = fetch) {
  if (!hex32.test(settings.device ?? '') || !(typeof settings.serial === 'string' && serialPattern.test(settings.serial)
    || settings.identity_kind === 'app-installation' && settings.serial === '')) return null;
  let response;
  try {
    response = await fetcher('http://127.0.0.1:8767/v1/quest/' + settings.device, {
      cache: 'no-store', credentials: 'omit', redirect: 'error', targetAddressSpace: 'loopback',
      signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]),
    });
  } catch {
    signal.throwIfAborted();
    return await companionAccessDenied() ? { state: 'loopback-denied' } : null;
  }
  const result = await readJson(response);
  if (result.v !== 1) throw Error('Ungültige Antwort der Bluetooth-Suche.');
  if (result.state === 'ready') {
    const serial = settings.commissioned && settings.identity_kind === 'app-installation'
      ? result.metadata?.serial : settings.serial;
    return { state: 'ready', ...discoveryMetadata(result.metadata, serial, settings) };
  }
  if (['bluetooth-off', 'searching', 'permission-required'].includes(result.state)) return { state: result.state };
  throw Error('Ungültige Antwort der Bluetooth-Suche.');
}

export async function readJson(response) {
  if (!response.ok || !response.body) {
    const error = Error('Die Quest-Verbindung wurde abgelehnt.');
    error.status = response.status;
    throw error;
  }
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

export async function joinQuest({ endpoint, serial, device, identity_kind, build, discovery, enrollment, signal, fetcher = fetch, transport }) {
  const lifetime = new AbortController();
  const close = () => {
    if (lifetime.signal.aborted) return;
    lifetime.abort();
    if (typeof transport?.close === 'function') transport.close();
  };
  try {
    const installation = identity_kind === 'app-installation' || discovery?.identity_kind === 'app-installation';
    if (transport && (!installation || !enrollment || !hex32.test(device ?? discovery?.device ?? '')
      || typeof transport.request !== 'function' || typeof transport.close !== 'function')) {
      throw Error('Für diesen Verbindungsweg fehlt eine gültige Verbindungsanmeldung.');
    }
    const url = transport ? undefined : questEndpoint(endpoint);
    if (!(typeof serial === 'string' && serialPattern.test(serial) || installation && serial === '')
      || (build === undefined ? !enrollment : !hex64.test(build))) {
      throw Error('Die Einstellungen für die Quest sind ungültig.');
    }
    if (enrollment && (!hex32.test(enrollment.controller) || !hex64.test(enrollment.key))) {
      throw Error('Prüfe Controller-ID und Quest-Schlüssel.');
    }
    const fetchJson = async (body, requestSignal) => {
      const requestLifetime = AbortSignal.any([lifetime.signal, AbortSignal.timeout(6500), ...[signal, requestSignal].filter(Boolean)]);
      requestLifetime.throwIfAborted();
      const response = transport ? await transport.request(body, requestLifetime)
        : await fetcher(url, {
          method: body ? 'POST' : 'GET', body: body ? JSON.stringify(body) : undefined,
          headers: body ? { 'Content-Type': 'application/json' } : undefined,
          mode: 'cors', cache: 'no-store', credentials: 'omit', redirect: 'error',
          referrerPolicy: 'no-referrer', targetAddressSpace: 'local', signal: requestLifetime,
        });
      requestLifetime.throwIfAborted();
      const result = await readJson(response);
      requestLifetime.throwIfAborted();
      return result;
    };
    const raw = await fetchJson();
    const owner = identity(raw);
    if (raw.enabled !== true || raw.remote_available !== true || owner.serial !== serial
      || installation && owner.identity_kind !== 'app-installation' || device && owner.device !== device
      || discovery && !sameQuest(owner, discovery)) throw Error('Quest oder Fernmodus stimmt nicht mit der Auswahl überein.');
    const ticket = enrollment ?? await fetchJson({ v: 1, action: 'connect', device: owner.device, epoch: owner.epoch });
    if (!enrollment && (ticket.v !== 1 || ticket.device !== owner.device || ticket.epoch !== owner.epoch
      || ticket.serial !== owner.serial || !hex32.test(ticket.controller) || !hex64.test(ticket.key))) {
      throw Error('Die Quest-Anmeldung konnte nicht geprüft werden.');
    }
    const controller = ticket.controller;
    const signingKey = await crypto.subtle.importKey('raw', encode.encode(ticket.key), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign', 'verify']);
    let ecgKey, armKey;
    const exchange = async (action, body, requestSignal) => {
      requestSignal?.throwIfAborted();
      const nonce = toHex(crypto.getRandomValues(new Uint8Array(16)));
      if (action === 'participant-arm') {
        const iv = crypto.getRandomValues(new Uint8Array(12));
        const bytes = await crypto.subtle.encrypt({ name: 'AES-GCM', iv,
          additionalData: encode.encode(`RQG1-ARM|${owner.device}|${owner.epoch}|${nonce}`) },
        armKey, encode.encode(JSON.stringify(body)));
        body = { v: 1, iv: btoa(String.fromCharCode(...iv)), data: btoa(String.fromCharCode(...new Uint8Array(bytes))) };
      }
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
      const result = JSON.parse(reply.payload);
      if (action !== 'ecg-preview' && action !== 'recordings') return result;
      if (!ecgKey || result?.v !== 1 || typeof result.iv !== 'string' || !/^[A-Za-z0-9+/]{16}$/.test(result.iv)
        || typeof result.data !== 'string' || result.data.length < 24 || result.data.length > 62500) {
        throw Error('Ungültige verschlüsselte Quest-Antwort.');
      }
      const clear = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: fromBase64(result.iv),
        additionalData: encode.encode(`RQG1-PRIVATE|${action}|${owner.device}|${owner.epoch}|${nonce}`) }, ecgKey, fromBase64(result.data));
      return JSON.parse(decode.decode(clear));
    };
    let witness = 0n;
    let observedStatus;
    const status = async () => {
      const value = await exchange('status', {});
      const observation = value?.o;
      const fingerprint = value?.build?.packaging_fingerprint;
      if (value?.v !== 1 || value.wifi_snapshot !== 1 || value.build?.v !== 1
        || !hex64.test(fingerprint) || build !== undefined && fingerprint !== build || ![0, 1].includes(value.recordings)) {
        throw Error('Der Quest-Build hat sich geändert. Richte die Verbindung erneut ein.');
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
      const access = controllerAccessState(value);
      if (owner.identity_kind === 'app-installation' && !access) throw Error('Die Quest hat keine aktuelle Steuerungsfreigabe gemeldet.');
      observedStatus = value;
      return value;
    };
    const initial = await status();
    build ??= initial.build.packaging_fingerprint;
    if (enrollment && initial.ecg_preview === 1) {
      const bytes = await crypto.subtle.sign('HMAC', signingKey,
        encode.encode(`RQG1-PRIVATE-KEY|${controller}|${owner.device}|${owner.epoch}`));
      ecgKey = await crypto.subtle.importKey('raw', bytes, 'AES-GCM', false, ['decrypt']);
      if (initial.participant_arm === 1 && initial.recordings === 1) {
        const armBytes = await crypto.subtle.sign('HMAC', signingKey,
          encode.encode(`RQG1-ARM-KEY|${controller}|${owner.device}|${owner.epoch}`));
        armKey = await crypto.subtle.importKey('raw', armBytes, 'AES-GCM', false, ['encrypt']);
      }
    }
    const controlled = (action, body, requestSignal) => {
      if (!canControlQuest(observedStatus)) throw Error('Ein anderes Smartphone steuert die Quest. Übernimm zuerst die Steuerung.');
      return exchange(action, body, requestSignal);
    };
    return { owner, initial, status, command: (command, requestSignal) => controlled('command', command, requestSignal),
      participantArm: armKey ? (command, requestSignal) => controlled('participant-arm', command, requestSignal) : null,
      scenarioArm: enrollment ? (command, requestSignal) => controlled('scenario-arm', command, requestSignal) : null,
      breathSetup: enrollment && initial.breath_setup === 1
        ? (command, requestSignal) => controlled('breath-setup', command, requestSignal) : null,
      breathSetupReceipt: enrollment && initial.breath_setup === 1
        ? (id, requestSignal) => {
          if (typeof id !== 'string' || !/^[a-f0-9]{16}$/.test(id)) throw Error('Ungültige Kennung der Atemeinrichtung.');
          return controlled('breath-setup', { v: 1, id, op: 'receipt' }, requestSignal);
        } : null,
      recenter: enrollment ? (command, requestSignal) => controlled('recenter', command, requestSignal) : null,
      backgroundMusic: enrollment && backgroundMusicStatus(initial.background_music) ? async (request, requestSignal) => {
        if (request?.v !== 1 || Object.keys(request).length !== 2
          || !(typeof request.playing === 'boolean' || Number.isInteger(request.volume) && request.volume >= 0 && request.volume <= 100)) {
          throw Error('Ungültige Musikanfrage.');
        }
        const applied = backgroundMusicStatus(await controlled('background-music', request, requestSignal));
        const expected = request.playing ? ['starting', 'playing'] : ['off'];
        if (!applied || (typeof request.playing === 'boolean'
          ? !expected.includes(applied.state) : request.volume !== applied.volume)) {
          throw Error('Die Quest hat die Musikeinstellung nicht bestätigt.');
        }
        return applied;
      } : null,
      kiosk: enrollment && initial.kiosk?.v === 1 ? (command, requestSignal) => {
        if (command?.v !== 1 || typeof command.id !== 'string' || !/^[a-f0-9]{16}$/.test(command.id)
          || !['receipt', 'retry'].includes(command.op)
          || command.op === 'retry' && (!Number.isSafeInteger(command.g) || command.g < 0)
          || Object.keys(command).length !== (command.op === 'retry' ? 4 : 3)) throw Error('Ungültige Kiosk-Anfrage.');
        return controlled('kiosk', command, requestSignal);
      } : null,
      ...(enrollment && initial.meta_update_control === 1 && initial.meta_update?.v === 2 ? { metaUpdate: (command, requestSignal) => {
        if (command?.v !== 1 || typeof command.id !== 'string' || !/^[a-f0-9]{16}$/.test(command.id)
          || !['receipt', 'check-and-update', 'end-preparation-and-check'].includes(command.op)
          || command.op === 'end-preparation-and-check' && initial.meta_update_preparation !== 1
          || command.op !== 'receipt' && !['cycle', 'revision'].every(key => typeof command[key] === 'string' && /^[1-9][0-9]{0,18}$/.test(command[key]))
          || Object.keys(command).length !== (command.op === 'receipt' ? 3 : 5)) throw Error('Ungültige Update-Anfrage.');
        return controlled('meta-update', command, requestSignal);
      } } : {}),
      controllerAccess: enrollment && controllerAccessState(initial)
        ? (command, requestSignal) => exchange('controller-access', command, requestSignal) : null,
      ecg: ecgKey ? (requestSignal) => exchange('ecg-preview', {}, requestSignal) : null,
      recordings: ecgKey && initial.recordings === 1 ? (query, requestSignal) => exchange('recordings', query, requestSignal) : null,
      receipt: (id) => exchange('receipt', { id }), close };
  } catch (error) { close(); throw error; }
}
