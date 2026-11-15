import test from 'node:test';
import assert from 'node:assert/strict';
import { createDecipheriv, createHmac } from 'node:crypto';
import { discoverQuest, joinQuest, questEndpoint, controllerAccessState, canControlQuest, sameQuest } from '../quest-gateway.js';
import { QuestRehearsal, rehearsalActions, polarSetup } from '../quest-rehearsal.js';
import * as rehearsalContract from '../quest-rehearsal.js';

const owner = { device: 'a'.repeat(32), serial: 'TEST-QUEST', epoch: '7', native_epoch: '8' };
const build = 'b'.repeat(64);
const endpoint = 'http://192.168.0.10:8766/v1/gateway';
const key = 'c'.repeat(64);
const controller = 'd'.repeat(32);
const mac = (value) => createHmac('sha256', key).update(value).digest('hex');
const snapshot = (phase = 'IDLE', generation = 0, revision = 0, witness = '100') => ({
  v: 1, wifi_snapshot: 1, control_mode: 'remote', recordings: 0,
  o: { v: 1, s: 'observed', e: owner.native_epoch, a: 5, i: witness, g: generation, r: revision },
  build: { v: 1, packaging_fingerprint: build },
  p: phase, g: generation, r: revision, ms: phase === 'RUNNING' ? 1000 : 0,
  c: phase === 'IDLE' ? '' : 'condition-a', rec: phase !== 'IDLE', pending: false,
  h: phase === 'IDLE' ? 'vr-preparation' : 'vr-session',
  a: ['track-ready'], pi: { r: false, f: false, h: false },
  q: phase === 'IDLE' ? undefined : {
    v: 2, s: 'e'.repeat(32), g: generation, o: [{ ARMED: 'start', RUNNING: 'pause', PAUSED: 'resume' }[phase]],
  },
});

test('Polar automatic recovery is optional and accepts only actual boolean observations', () => {
  const y = { s: 'a'.repeat(32), g: 8, r: 1, p: true, b: true, c: true,
    a: false, e: false, sc: false, x: 'b'.repeat(12), d: [] };
  assert.equal(polarSetup({ ...snapshot(), y }), y);
  for (const ar of [true, false]) {
    assert.equal(polarSetup({ ...snapshot(), y: { ...y, ar } })?.ar, ar);
  }
  for (const ar of [null, 0, 1, 'true', {}]) {
    assert.equal(polarSetup({ ...snapshot(), y: { ...y, ar } }), null);
  }
});

test('native Polar automation distinguishes backoff, missing streams, stopped and selection-required', () => {
  assert.equal(typeof rehearsalContract.polarReadiness, 'function');
  const { polarReadiness } = rehearsalContract;
  const base = { s: 'a'.repeat(32), g: 8, r: 1, p: true, b: true, c: false,
    a: false, e: false, sc: false, x: '', d: [] };
  const auto = { v: 1, state: 'backoff', retry_ms: 2000, reason: 'connection-timeout' };
  const project = (patch = {}) => polarSetup({ ...snapshot(), y: { ...base, ...patch } });
  assert.match(polarReadiness(project({ auto })), /automatisch.*2 s/);
  assert.match(polarReadiness(project({ c: true, x: 'b'.repeat(12), e: true, auto: { ...auto, state: 'streams' } })), /Bewegungsdaten fehlen.*automatisch/);
  assert.match(polarReadiness(project({ auto: { ...auto, state: 'stopped' } })), /gestoppt/);
  assert.match(polarReadiness(project({ auto: { ...auto, state: 'selection-required' } })), /Mehrere.*Quest/);
  for (const malformed of [null, { ...auto, v: 2 }, { ...auto, state: 'invented' }, { ...auto, retry_ms: 10001 }, { ...auto, reason: 'x'.repeat(241) }]) {
    assert.equal(project({ auto: malformed }), null);
  }
  assert.match(polarReadiness(project()), /Automatische Verbindung nicht gemeldet/);
});

test('presentation timing charges status transport without rewriting the native observation', async () => {
  let now = 100;
  const native = snapshot();
  const quest = { owner, initial: native, close() {}, status: async () => { now += 1700; return native; } };
  const rehearsal = new QuestRehearsal({ now: () => now });
  rehearsal.attach(quest);
  assert.equal(rehearsal.state.statusTiming, null, 'an initial join has no fresh display receipt');
  await rehearsal.refresh();
  assert.deepEqual(rehearsal.state.statusTiming, { receivedAt: 1800, transportMs: 1700 });
  assert.equal(rehearsal.state.status, native);
  assert.equal(native.o.a, 5);
  assert.equal(Object.hasOwn(native, 'statusTiming'), false);
  rehearsal.attach(null);
  assert.equal(rehearsal.state.statusTiming, null);
});

function server({ identity = owner, status = snapshot(), tamper = false } = {}) {
  const requests = [];
  return { requests, async fetch(url, init) {
    assert.equal(url, endpoint);
    assert.equal(init.cache, 'no-store');
    assert.equal(init.credentials, 'omit');
    const body = init.body && JSON.parse(init.body);
    requests.push(body?.action ?? 'GET');
    if (!body) return Response.json({ v: 1, ...identity, enabled: true, remote_available: true });
    if (body.action === 'connect') {
      assert.deepEqual(body, { v: 1, action: 'connect', device: owner.device, epoch: owner.epoch });
      return Response.json({ v: 1, ...owner, key, controller });
    }
    assert.equal(body.mac, mac(`RQG1|${controller}|${owner.epoch}|${body.nonce}|${body.action}|${body.payload}`));
    const payload = JSON.stringify(status);
    const proof = mac(`RQG1-REPLY|${controller}|${owner.device}|${owner.epoch}|${body.nonce}|${body.action}|${payload}`);
    return Response.json({ v: 1, device: owner.device, epoch: owner.epoch, nonce: body.nonce, action: body.action, payload, mac: tamper ? '0'.repeat(64) : proof });
  } };
}

test('joins a private Quest, verifies signed observations, and never enrolls the wrong serial', async () => {
  for (const bad of ['8.8.8.8', '127.0.0.1', 'http://192.168.0.10:8767/v1/gateway', endpoint + '?key=x', 'http://u@192.168.0.10:8766/v1/gateway']) {
    assert.throws(() => questEndpoint(bad));
  }
  const valid = server();
  const quest = await joinQuest({ endpoint, serial: owner.serial, build, fetcher: valid.fetch });
  assert.equal(quest.initial.p, 'IDLE');
  assert.deepEqual(valid.requests, ['GET', 'connect', 'status']);
  const wrong = server({ identity: { ...owner, serial: 'OTHER' } });
  await assert.rejects(joinQuest({ endpoint, serial: owner.serial, build, fetcher: wrong.fetch }));
  assert.deepEqual(wrong.requests, ['GET']);
  for (const fixture of [server({ tamper: true }), server({ status: { ...snapshot(), o: { ...snapshot().o, a: 6000 } } }), server({ status: { ...snapshot(), build: { v: 1, packaging_fingerprint: key } } })]) {
    await assert.rejects(joinQuest({ endpoint, serial: owner.serial, build, fetcher: fixture.fetch }));
  }
});

test('v2 identifies an app installation without pretending an unavailable hardware serial exists', async () => {
  const identity = { ...owner, v: 2, identity_kind: 'app-installation', pairing: 1, serial: '' };
  const activeStatus = { ...snapshot(), controller_access: { v: 1, role: 'active', active_phone: 'Pixel', revision: 1 } };
  const valid = server({ identity, status: activeStatus });
  const paired = await joinQuest({ endpoint, serial: '', identity_kind: 'app-installation',
    enrollment: { controller, key }, build, fetcher: valid.fetch });
  assert.equal(paired.owner.serial, '');
  assert.equal(paired.owner.identity_kind, 'app-installation');
  assert.deepEqual(valid.requests, ['GET', 'status']);
  for (const invalid of [{ ...identity, v: 1 }, { ...identity, identity_kind: 'hardware' },
    { ...identity, serial: undefined }]) {
    await assert.rejects(joinQuest({ endpoint, serial: '', identity_kind: 'app-installation',
      enrollment: { controller, key }, build, fetcher: server({ identity: invalid }).fetch }));
  }
  await assert.rejects(joinQuest({ endpoint, serial: '', enrollment: { controller, key }, build, fetcher: valid.fetch }));
});

test('a waiting trusted phone can read status and request control but cannot send experiment mutations', async () => {
  const state = { ...snapshot(), controller_access: { v: 1, role: 'waiting', active_phone: 'Other phone', revision: 4 },
    breath_setup: 1, participant_arm: 1, recordings: 1, ecg_preview: 1, recenter: 1 };
  const fixture = server({ status: state });
  const gateway = await joinQuest({ endpoint, serial: owner.serial, enrollment: { controller, key }, fetcher: fixture.fetch });
  for (const act of [() => gateway.command({ op: 'start' }), () => gateway.breathSetup({ op: 'calibrate' }),
    () => gateway.participantArm({}), () => gateway.recenter({})]) await assert.rejects(async () => act());
  assert.deepEqual(fixture.requests, ['GET', 'status']);
  assert.equal(typeof gateway.controllerAccess, 'function');
  for (const capability of ['command', 'breathSetup', 'participantArm', 'scenarioArm', 'recenter']) {
    assert.equal(typeof gateway[capability], 'function');
  }
  assert.deepEqual(rehearsalActions(state), {});
});

test('control authority rejects malformed roles and identity kind cannot silently change', () => {
  assert.equal(canControlQuest(snapshot()), true);
  const valid = { v: 1, role: 'active', active_phone: 'Pixel', revision: 3 };
  assert.deepEqual(controllerAccessState({ controller_access: valid }), valid);
  for (const invalid of [{ ...valid, active_phone: '' }, { ...valid, revision: -1 },
    { ...valid, role: 'unknown' }, { ...valid, active_phone: 'a'.repeat(65) },
    { ...valid, active_phone: 'Phone\n' }, { ...valid, extra: true }]) {
    assert.throws(() => controllerAccessState({ controller_access: invalid }));
    assert.equal(canControlQuest({ controller_access: invalid }), false);
  }
  assert.equal(sameQuest(owner, { ...owner }), true);
  assert.equal(sameQuest(owner, { ...owner, identity_kind: 'app-installation' }), false);
});

test('learns the signed build during enrollment and pins the saved headset on reconnect', async () => {
  const enrollment = { controller, key };
  const first = server();
  const paired = await joinQuest({ endpoint, serial: owner.serial, enrollment, fetcher: first.fetch });
  assert.equal(paired.initial.build.packaging_fingerprint, build);
  assert.deepEqual(first.requests, ['GET', 'status']);
  paired.close();
  const other = server({ identity: { ...owner, device: 'f'.repeat(32) } });
  await assert.rejects(joinQuest({ endpoint, serial: owner.serial, device: owner.device, build, enrollment, fetcher: other.fetch }));
  assert.deepEqual(other.requests, ['GET']);
  const unverified = server({ tamper: true });
  await assert.rejects(joinQuest({ endpoint, serial: owner.serial, enrollment, fetcher: unverified.fetch }));
  await assert.rejects(joinQuest({ endpoint, serial: owner.serial, fetcher: first.fetch }));
});

test('participant Arm encrypts the identity and observed generation for the enrolled Quest', async () => {
  const state = { ...snapshot(), participant_arm: 1, recordings: 1, ecg_preview: 1 };
  const fixture = server({ status: state });
  const body = { v: 1, id: 'a'.repeat(16), op: 'arm', condition: 'condition-a',
    bias: 0, nonce: 'b'.repeat(16), g: 0, participant_id: 'Mül_35_N' };
  let received;
  const fetcher = async (url, init) => {
    const envelope = init.body && JSON.parse(init.body);
    if (envelope?.action === 'participant-arm') {
      assert.ok(!init.body.includes(body.participant_id));
      const { v, iv, data } = JSON.parse(envelope.payload);
      assert.equal(v, 1);
      const bytes = Buffer.from(data, 'base64');
      const keyBytes = Buffer.from(mac(`RQG1-ARM-KEY|${controller}|${owner.device}|${owner.epoch}`), 'hex');
      const cipher = createDecipheriv('aes-256-gcm', keyBytes, Buffer.from(iv, 'base64'));
      cipher.setAAD(Buffer.from(`RQG1-ARM|${owner.device}|${owner.epoch}|${envelope.nonce}`));
      cipher.setAuthTag(bytes.subarray(-16));
      received = JSON.parse(Buffer.concat([cipher.update(bytes.subarray(0, -16)), cipher.final()]));
    }
    return fixture.fetch(url, init);
  };
  const guest = await joinQuest({ endpoint, serial: owner.serial, build, fetcher });
  assert.equal(guest.participantArm, null);
  const enrolled = await joinQuest({ endpoint, serial: owner.serial, build, enrollment: { controller, key }, fetcher });
  await enrolled.participantArm(body);
  assert.deepEqual(received, body);
});

test('Bluetooth reads both identity characteristics and releases GATT before LAN enrollment', async () => {
  const calls = [];
  const signal = new AbortController().signal;
  const gatt = {
    connected: false,
    async connect() { this.connected = true; return this; },
    async getPrimaryService(service) {
      assert.match(service, /^9a7b2001/);
      return { async getCharacteristic(id) {
        calls.push(id.slice(-2));
        return { async readValue() {
          const value = id.endsWith('02') ? owner.serial
            : JSON.stringify({ ...owner, v: 1, mode: 'remote', gateways: [endpoint] });
          return new TextEncoder().encode(value);
        } };
      } };
    },
    disconnect() { this.connected = false; },
  };
  const found = await discoverQuest({ async requestDevice() { return { gatt }; } }, signal);
  assert.deepEqual(found, { owner, endpoints: [endpoint] });
  assert.deepEqual(calls, ['02', '03']);
  assert.equal(gatt.connected, false);
});

test('fresh app discovery permits an explicitly unavailable serial and reads no enrollment characteristic', async () => {
  const identity = { ...owner, serial: '', identity_kind: 'app-installation' };
  for (const change of [{}, { v: 1 }, { pairing: 0 }, { identity_kind: 'hardware' }]) {
    const calls = [];
    const gatt = { connected: false,
      async connect() { this.connected = true; return this; },
      async getPrimaryService() { return { async getCharacteristic(id) {
        calls.push(id.slice(-2));
        return { async readValue() { return new TextEncoder().encode(id.endsWith('02') ? ''
          : JSON.stringify({ ...identity, v: 2, mode: 'remote', gateways: [endpoint], pairing: 1, ...change })); } };
      } }; }, disconnect() { this.connected = false; } };
    const result = discoverQuest({ async requestDevice() { return { gatt }; } }, new AbortController().signal);
    if (Object.keys(change).length) await assert.rejects(result);
    else assert.deepEqual(await result, { owner: identity, endpoints: [endpoint], pairing: 1 });
    assert.deepEqual(calls, ['02', '03']);
    assert.equal(gatt.connected, false);
  }
});

test('native breathing setup is offered only by an enrolled capable Quest and keeps its closed command', async () => {
  const fixture = server({ status: { ...snapshot(), breath_setup: 1 } });
  const enrolled = await joinQuest({ endpoint, serial: owner.serial, build,
    enrollment: { controller, key }, fetcher: fixture.fetch });
  assert.equal(typeof enrolled.breathSetup, 'function');
  const command = { v: 1, id: 'a'.repeat(16), g: 1, r: 2, op: 'calibrate' };
  await enrolled.breathSetup(command);
  assert.equal(fixture.requests.at(-1), 'breath-setup');
  const guest = await joinQuest({ endpoint, serial: owner.serial, build, fetcher: fixture.fetch });
  assert.equal(guest.breathSetup, null);
  const legacy = server();
  const old = await joinQuest({ endpoint, serial: owner.serial, build,
    enrollment: { controller, key }, fetcher: legacy.fetch });
  assert.equal(old.breathSetup, null);
  enrolled.close(); guest.close(); old.close();
});

test('kiosk capability is private, active-controller fenced and sends closed CAS retry or receipt queries', async () => {
  const offered = { ...snapshot(), kiosk: { v: 1 } };
  const fixture = server({ status: offered }); const payloads = [];
  const gateway = await joinQuest({ endpoint, serial: owner.serial, build, enrollment: { controller, key },
    fetcher: async (url, init) => {
      const request = init.body && JSON.parse(init.body);
      if (request?.action === 'kiosk') payloads.push(JSON.parse(request.payload));
      return fixture.fetch(url, init);
    } });
  assert.equal(typeof gateway.kiosk, 'function');
  const retry = { v: 1, id: 'f'.repeat(16), op: 'retry', g: 3 };
  await gateway.kiosk(retry);
  await gateway.kiosk({ v: 1, id: retry.id, op: 'receipt' });
  assert.deepEqual(payloads, [retry, { v: 1, id: retry.id, op: 'receipt' }]);
  for (const invalid of [{ ...retry, op: 'arm' }, { ...retry, g: -1 }, { ...retry, extra: true },
    { v: 1, id: retry.id, op: 'receipt', g: 3 }]) {
    await assert.rejects(async () => gateway.kiosk(invalid));
  }
  assert.equal(payloads.length, 2);
  offered.controller_access = { v: 1, role: 'waiting', active_phone: 'Other', revision: 2 };
  offered.o.i = '101';
  await gateway.status();
  await assert.rejects(async () => gateway.kiosk(retry), /Steuerung/);
  const guest = await joinQuest({ endpoint, serial: owner.serial, build, fetcher: fixture.fetch });
  assert.equal(guest.kiosk, null);
  const legacy = server();
  const old = await joinQuest({ endpoint, serial: owner.serial, build, enrollment: { controller, key }, fetcher: legacy.fetch });
  assert.equal(old.kiosk, null);
  gateway.close(); guest.close(); old.close();
});

test('setup reconciliation authenticates an exact three-field read-only query and rejects malformed IDs before transport', async () => {
  const fixture = server({ status: { ...snapshot(), breath_setup: 1 } }), payloads = [];
  const gateway = await joinQuest({ endpoint, serial: owner.serial, build, enrollment: { controller, key },
    fetcher: async (url, init) => {
      const request = init.body && JSON.parse(init.body);
      if (request?.action === 'breath-setup') payloads.push(JSON.parse(request.payload));
      return fixture.fetch(url, init);
    } });
  const id = '0123456789abcdef';
  await gateway.breathSetupReceipt(id);
  assert.deepEqual(payloads, [{ v: 1, id, op: 'receipt' }]);
  assert.equal(fixture.requests.at(-1), 'breath-setup');
  for (const invalid of ['', 'A'.repeat(16), 'a'.repeat(15), 'a'.repeat(17), null, {}, 123]) {
    await assert.rejects(async () => gateway.breathSetupReceipt(invalid));
  }
  await assert.rejects(async () => gateway.breathSetupReceipt(id, AbortSignal.abort()), { name: 'AbortError' });
  assert.equal(payloads.length, 1);
  gateway.close();
});

test('a command whose foreground lifetime ends while signing never reaches fetch', async () => {
  const fixture = server();
  const quest = await joinQuest({ endpoint, serial: owner.serial, build, fetcher: fixture.fetch });
  const foreground = new AbortController();
  const original = crypto.subtle.sign;
  crypto.subtle.sign = async function (...args) {
    const proof = await original.apply(this, args);
    foreground.abort();
    return proof;
  };
  try {
    await assert.rejects(quest.command({ op: 'start' }, foreground.signal), { name: 'AbortError' });
    assert.deepEqual(fixture.requests, ['GET', 'connect', 'status']);
  } finally { crypto.subtle.sign = original; quest.close(); }
});

function fakeQuest({ lost = null, delayedArm = false } = {}) {
  let state = 'IDLE', g = 0, r = 0, i = 100;
  const commands = [];
  const receipts = new Map();
  return {
    owner,
    initial: snapshot(),
    commands,
    async status() { return snapshot(state, g, r, String(++i)); },
    async command(request) {
      commands.push(request);
      if (request.op === 'arm') g++;
      r++;
      state = { arm: 'ARMED', start: 'RUNNING', pause: 'PAUSED', resume: 'RUNNING', 'save-next': 'IDLE' }[request.op];
      receipts.set(request.id, { v: 1, id: request.id, state: 'confirmed' });
      if (request.op === lost) throw Error('socket dropped after dispatch');
      return delayedArm && request.op === 'arm' ? { v: 1, id: request.id, state: 'pending' } : receipts.get(request.id);
    },
    async receipt(id) { return receipts.get(id); },
    close() {},
  };
}

test('rehearsal sends one fenced Arm, Start, Pause, Resume and Finish using native receipts', async () => {
  const quest = fakeQuest();
  const rehearsal = new QuestRehearsal();
  rehearsal.attach(quest);
  await Promise.all([rehearsal.run('play'), rehearsal.run('play')]);
  assert.deepEqual(quest.commands.map(({ op }) => op), ['arm', 'start']);
  assert.equal(quest.commands[0].condition, 'condition-a');
  assert.equal(quest.commands[1].s, 'e'.repeat(32));
  assert.match(quest.commands[0].nonce, /^[a-f0-9]{16}$/);
  assert.match(quest.commands[1].n, /^[a-f0-9]{16}$/);
  await rehearsal.run('play');
  assert.equal(rehearsal.state.status.p, 'PAUSED');
  await rehearsal.run('play');
  await rehearsal.run('finish');
  assert.deepEqual(quest.commands.map(({ op }) => op), ['arm', 'start', 'pause', 'resume', 'save-next']);
  assert.equal(rehearsal.state.status.p, 'IDLE');
});

test('participant sessions accept the native Polar requirement and retain pause and finish when signal is stale', () => {
  const armed = { ...snapshot('ARMED', 1, 1), participant_arm: 1, pi: { r: true, f: true, h: false } };
  assert.deepEqual(rehearsalActions(armed), { play: 'start', finish: 'save-next' });
  const stale = { ...armed, pi: { ...armed.pi, f: false } };
  assert.deepEqual(rehearsalActions(stale), { play: undefined, finish: 'save-next' });
  const running = { ...snapshot('RUNNING', 1, 2), participant_arm: 1, pi: stale.pi };
  assert.deepEqual(rehearsalActions(running), { play: 'pause', finish: 'save-next' });
  const paused = { ...snapshot('PAUSED', 1, 3), participant_arm: 1, pi: { ...stale.pi, h: true } };
  assert.deepEqual(rehearsalActions(paused), { play: undefined, finish: 'save-next' });
  assert.deepEqual(rehearsalActions({ ...paused, pi: { ...paused.pi, f: true } }),
    { play: 'resume', finish: 'save-next' });
  assert.deepEqual(rehearsalActions({ ...armed, participant_arm: undefined }), {});
  assert.deepEqual(rehearsalActions({ ...armed, pi: undefined }), {});
});

test('an enrolled scenario Arm binds the native generation and the selected timeline', async () => {
  const quest = fakeQuest();
  const supported = ['probe-anleitung', 'haupt-15'];
  quest.initial = { ...quest.initial, scenarios: supported };
  const read = quest.status;
  quest.status = async () => ({ ...await read(), scenarios: supported });
  const scenarioCommands = [];
  quest.scenarioArm = (command, signal) => {
    scenarioCommands.push(command);
    return quest.command(command, signal);
  };
  const rehearsal = new QuestRehearsal();
  rehearsal.attach(quest);
  await rehearsal.run('play', undefined, 'condition-a', '', 'box-20');
  assert.equal(quest.commands.length, 0);
  await rehearsal.run('play', undefined, 'condition-a', '', 'haupt-15');
  assert.equal(scenarioCommands.length, 1);
  assert.equal(scenarioCommands[0].g, 0);
  assert.equal(scenarioCommands[0].scenario, 'haupt-15');
  assert.deepEqual(quest.commands.map(({ op }) => op), ['arm', 'start']);
});

test('a lost Arm reply is reconciled without starting or replaying after reconnect', async () => {
  const quest = fakeQuest({ lost: 'arm' });
  const rehearsal = new QuestRehearsal();
  rehearsal.attach(quest);
  await rehearsal.run('play');
  assert.equal(rehearsal.state.unknown, true);
  await rehearsal.refresh();
  assert.equal(rehearsal.state.status.p, 'ARMED');
  assert.deepEqual(quest.commands.map(({ op }) => op), ['arm']);
  await rehearsal.run('play');
  assert.deepEqual(quest.commands.map(({ op }) => op), ['arm', 'start']);
});

test('a confirmed Resume followed by a new Polar hold releases the phone controls', async () => {
  const paused = { ...snapshot('PAUSED', 3, 8), participant_arm: 1, pi: { r: true, f: true, h: true } };
  const requests = [];
  const quest = {
    owner, initial: paused, status: async () => paused,
    async command(request) {
      requests.push(request);
      paused.r += 2;
      paused.pi.f = false;
      return { v: 1, id: request.id, state: 'confirmed' };
    },
    close() {},
  };
  const rehearsal = new QuestRehearsal();
  rehearsal.attach(quest);
  await rehearsal.run('play');
  assert.deepEqual(requests.map(request => request.op), ['resume']);
  assert.equal(rehearsal.state.status.p, 'PAUSED');
  assert.equal(rehearsal.state.pending, false);
  assert.equal(rehearsal.state.unknown, false);
});

test('recenter uses a native frame receipt and never replays after a lost reply', async () => {
  const requests = [];
  let outcome = 'pending';
  const ready = { ...snapshot(), recenter: 1 };
  const quest = {
    owner, initial: ready, status: async () => ready,
    recenter: async command => {
      requests.push(command);
      throw Error('reply dropped after admission');
    },
    receipt: async id => ({ v: 1, id, state: outcome }),
    close() {},
  };
  const rehearsal = new QuestRehearsal();
  rehearsal.attach(quest);
  await Promise.all([rehearsal.run('recenter'), rehearsal.run('recenter')]);
  assert.equal(requests.length, 1);
  assert.deepEqual(Object.keys(requests[0]).sort(), ['g', 'id', 'v']);
  assert.equal(requests[0].g, ready.g);
  await rehearsal.refresh();
  assert.equal(rehearsal.state.unknown, true);
  outcome = 'confirmed';
  await rehearsal.refresh();
  assert.equal(rehearsal.state.message, 'VR-Szene zentriert.');
  assert.equal(requests.length, 1);
  ready.h = 'panel';
  await rehearsal.run('recenter');
  assert.equal(requests.length, 1);
});

test('a pending Arm may lead to Start only within the same foreground action', async () => {
  const quest = fakeQuest({ delayedArm: true });
  const connected = new QuestRehearsal({ wait: async () => {} });
  connected.attach(quest);
  await connected.run('play');
  assert.deepEqual(quest.commands.map(({ op }) => op), ['arm', 'start']);
  const other = fakeQuest({ delayedArm: true });
  const foreground = new AbortController();
  const suspended = new QuestRehearsal({ wait: async () => foreground.abort() });
  suspended.attach(other);
  await suspended.run('play', foreground.signal);
  await suspended.refresh();
  assert.deepEqual(other.commands.map(({ op }) => op), ['arm']);
  assert.equal(suspended.state.status.p, 'ARMED');
});

test('a new native owner cannot clear an uncertain Finish or receive a replay', async () => {
  const quest = fakeQuest({ lost: 'save-next' });
  const rehearsal = new QuestRehearsal();
  rehearsal.attach(quest);
  await rehearsal.run('play');
  await rehearsal.run('finish');
  assert.equal(rehearsal.state.unknown, true);
  const replacement = fakeQuest();
  replacement.owner = { ...owner, native_epoch: '9', epoch: '10' };
  const replacementStatus = replacement.status;
  replacement.status = async () => { const value = await replacementStatus(); return { ...value, o: { ...value.o, e: '9' } }; };
  rehearsal.attach(replacement);
  await rehearsal.refresh();
  await rehearsal.run('play');
  assert.equal(rehearsal.state.unknown, true);
  assert.deepEqual(replacement.commands, []);
  rehearsal.acknowledgeIdle();
  assert.equal(rehearsal.state.unknown, false);
});

test('uncertain same-actor operations permit receipt reads but never local acknowledgement or replay', async () => {
  const quest = fakeQuest({ lost: 'save-next' });
  const rehearsal = new QuestRehearsal(); rehearsal.attach(quest);
  await rehearsal.run('play'); await rehearsal.run('finish');
  rehearsal.update({ status: snapshot(), busy: false });
  assert.equal(rehearsal.canAcknowledgeIdle(), false);
  assert.match(rehearsal.recovery().detail, /Bestätigung/);
  assert.equal(rehearsal.recovery().action, 'Bestätigung erneut abfragen');
  rehearsal.acknowledgeIdle();
  assert.equal(rehearsal.state.unknown, true);
  assert.ok(rehearsal.pending);
  const replacement = { ...quest, owner: { ...owner, native_epoch: '9' } };
  rehearsal.attach(replacement);
  assert.equal(rehearsal.canAcknowledgeIdle(), false, 'stale prior-actor status cannot acknowledge the current actor');
  rehearsal.update({ status: { ...snapshot(), o: { ...snapshot().o, e: '9' } } });
  assert.equal(rehearsal.canAcknowledgeIdle(), true);
  rehearsal.update({ status: { ...snapshot(), controller_access: { v: 1, role: 'waiting', active_phone: 'Other', revision: 1 } } });
  assert.equal(rehearsal.canAcknowledgeIdle(), false);
});

function polarQuest({ candidates = ['0123456789ab'], lost, delayedScan = false, permission = true } = {}) {
  const commands = [], receipts = new Map();
  const polar = { s: 'f'.repeat(32), g: Number(owner.native_epoch), r: 1, p: permission, b: true,
    c: false, x: '', a: false, e: false, d: [], sc: false };
  const current = () => ({ ...snapshot(), y: { ...polar } });
  return { owner, initial: current(), commands, polar, close() {}, status: async () => current(),
    async command(command) {
      assert.equal(command.v, 4);
      assert.equal(command.g, polar.g);
      assert.equal(command.s, polar.s);
      assert.equal(command.r, polar.r);
      assert.match(command.n, /^[a-f0-9]{16}$/);
      commands.push(command);
      polar.r++;
      if (command.op === 'scan') polar.d = candidates;
      if (command.op === 'select') polar.x = command.d;
      if (command.op === 'connect') polar.c = true;
      if (command.op === 'start-ecg') polar.e = true;
      if (command.op === 'start-acc') polar.a = true;
      if (command.op !== 'select') assert.equal(command.d, '');
      const receipt = { v: 1, id: command.id, state: 'confirmed' };
      receipts.set(command.id, receipt);
      if (command.op === lost) throw Error('Reply dropped after dispatch');
      return delayedScan && command.op === 'scan' ? { ...receipt, state: 'pending' } : receipt;
    },
    async receipt(id) { return receipts.get(id); },
  };
}

test('one explicit browser action asks Quest to scan, select, connect and start fresh ECG and accelerometer streams', async () => {
  const quest = polarQuest({ delayedScan: true });
  const rehearsal = new QuestRehearsal({ wait: async () => {} });
  rehearsal.attach(quest);
  await Promise.all([rehearsal.run('polar'), rehearsal.run('polar'), rehearsal.run('play')]);
  assert.deepEqual(quest.commands.map(({ op }) => op), ['scan', 'select', 'connect', 'start-ecg', 'start-acc']);
  assert.equal(quest.commands[1].d, '0123456789ab');
  assert.equal(rehearsal.state.status.y.e, true);
  assert.equal(rehearsal.state.status.y.a, true);
  assert.equal(rehearsal.state.pending, false);
});

test('Polar setup refuses missing grants, stale owners and automatic choice among multiple straps', async () => {
  const rehearsal = new QuestRehearsal();
  const ungranted = polarQuest({ permission: false });
  rehearsal.attach(ungranted);
  await rehearsal.run('polar');
  assert.deepEqual(ungranted.commands, []);
  const stale = polarQuest();
  stale.polar.g++;
  rehearsal.attach(stale);
  await rehearsal.run('polar');
  assert.deepEqual(stale.commands, []);
  const multiple = polarQuest({ candidates: ['0123456789ab', '0123456789ac'] });
  rehearsal.attach(multiple);
  await rehearsal.run('polar');
  assert.deepEqual(multiple.commands.map(({ op }) => op), ['scan']);
  assert.match(rehearsal.state.message, /Mehrere/);
});

test('Location service readiness is separate from grants and blocks scanning while preserving an existing connection', async () => {
  const rehearsal = new QuestRehearsal({ wait: async () => {} });
  const disabled = polarQuest();
  disabled.polar.l = 'disabled';
  rehearsal.attach(disabled);
  await rehearsal.run('polar');
  assert.deepEqual(disabled.commands, []);
  for (const state of ['enabled', 'disabled', 'unavailable', 'legacy-unobserved', 'permission-blocked', 'unknown']) {
    disabled.polar.l = state;
    assert.equal(polarSetup(await disabled.status())?.l, state);
  }
  for (const invalid of [false, null, 'ready', {}]) {
    assert.equal(polarSetup({ ...disabled.initial, y: { ...disabled.polar, l: invalid } }), null);
  }
  const connected = polarQuest();
  Object.assign(connected.polar, { l: 'disabled', c: true, x: '0123456789ab' });
  rehearsal.attach(connected);
  await rehearsal.run('polar');
  assert.deepEqual(connected.commands.map(({ op }) => op), ['start-ecg', 'start-acc']);
});

test('recovery only checks the original Polar receipt and never chains the remaining setup steps', async () => {
  const quest = polarQuest({ lost: 'connect' });
  const rehearsal = new QuestRehearsal();
  rehearsal.attach(quest);
  await rehearsal.run('polar');
  assert.equal(rehearsal.state.unknown, true);
  await rehearsal.refresh();
  assert.equal(rehearsal.state.status.y.c, true);
  assert.equal(rehearsal.state.status.y.e, false);
  assert.equal(rehearsal.state.unknown, false);
  assert.deepEqual(quest.commands.map(({ op }) => op), ['scan', 'select', 'connect']);
  await rehearsal.run('polar');
  assert.deepEqual(quest.commands.map(({ op }) => op), ['scan', 'select', 'connect', 'start-ecg', 'start-acc']);
});

test('leaving the foreground during a Polar scan requires a new browser action', async () => {
  const foreground = new AbortController();
  const quest = polarQuest({ delayedScan: true });
  const rehearsal = new QuestRehearsal({ wait: async () => foreground.abort() });
  rehearsal.attach(quest);
  await rehearsal.run('polar', foreground.signal);
  assert.equal(rehearsal.state.unknown, true);
  await rehearsal.refresh();
  assert.equal(rehearsal.state.unknown, false);
  assert.deepEqual(quest.commands.map(({ op }) => op), ['scan']);
});

test('native audio recovery blocks parallel Polar and scenario setup at click and fresh-status admission', async () => {
  for (const beginsRecovering of [true, false]) {
    const quest = polarQuest();
    const healthy = quest.initial, recovering = { ...healthy, setup: { stage: 'recovering' } };
    if (beginsRecovering) quest.initial = recovering;
    quest.status = async () => recovering;
    const rehearsal = new QuestRehearsal();
    rehearsal.attach(quest);
    assert.deepEqual(rehearsalActions(recovering), {});
    await rehearsal.run('polar');
    await rehearsal.run('play');
    assert.deepEqual(quest.commands, []);
  }
});
