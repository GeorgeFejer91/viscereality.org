import test from 'node:test';
import assert from 'node:assert/strict';
import { createHmac } from 'node:crypto';
import { discoverQuest, joinQuest, questEndpoint } from '../quest-gateway.js';
import { QuestRehearsal } from '../quest-rehearsal.js';

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

function server({ identity = owner, status = snapshot(), tamper = false } = {}) {
  const requests = [];
  return { requests, async fetch(url, init) {
    assert.equal(url, endpoint);
    assert.equal(init.cache, 'no-store');
    assert.equal(init.credentials, 'omit');
    const body = init.body && JSON.parse(init.body);
    requests.push(body?.action ?? 'GET');
    if (!body) return Response.json({ ...identity, v: 1, enabled: true, remote_available: true });
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
  rehearsal.attach(replacement);
  await rehearsal.refresh();
  await rehearsal.run('play');
  assert.equal(rehearsal.state.unknown, true);
  assert.deepEqual(replacement.commands, []);
  rehearsal.acknowledgeIdle();
  assert.equal(rehearsal.state.unknown, false);
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
      if (command.op !== 'select') assert.equal(command.d, '');
      const receipt = { v: 1, id: command.id, state: 'confirmed' };
      receipts.set(command.id, receipt);
      if (command.op === lost) throw Error('Reply dropped after dispatch');
      return delayedScan && command.op === 'scan' ? { ...receipt, state: 'pending' } : receipt;
    },
    async receipt(id) { return receipts.get(id); },
  };
}

test('one explicit browser action asks Quest to scan, select, connect and start fresh ECG', async () => {
  const quest = polarQuest({ delayedScan: true });
  const rehearsal = new QuestRehearsal({ wait: async () => {} });
  rehearsal.attach(quest);
  await Promise.all([rehearsal.run('polar'), rehearsal.run('polar'), rehearsal.run('play')]);
  assert.deepEqual(quest.commands.map(({ op }) => op), ['scan', 'select', 'connect', 'start-ecg']);
  assert.equal(quest.commands[1].d, '0123456789ab');
  assert.equal(rehearsal.state.status.y.e, true);
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
  assert.deepEqual(quest.commands.map(({ op }) => op), ['scan', 'select', 'connect', 'start-ecg']);
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
