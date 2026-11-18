import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const target = await readFile(new URL('../quest/webrtc/target.js', import.meta.url), 'utf8');
const moduleSource = target
  .replace('./vdo-ninja-transport.js', new URL('../vendor/quest-webrtc/vdo-ninja-transport.js', import.meta.url).href)
  .replace('./quest-webrtc-channel.js', new URL('../quest-webrtc-channel.js', import.meta.url).href);
const moduleUrl = 'data:text/javascript;base64,' + Buffer.from(moduleSource).toString('base64');
const settle = () => new Promise(setImmediate);

async function loadTarget(t, connect) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = new EventTarget();
  const original = new Map();
  let configuration = 'a'.repeat(32) + ':' + 'b'.repeat(64);
  let attempts = 0;
  let announcements = 0;
  let closed = 0;
  const SDK = class extends EventTarget {
    async connect() { await connect(++attempts); }
    async joinRoom() {}
    async announce() { announcements++; }
    async disconnect() { closed++; }
  };
  const globals = {
    window: page,
    QuestRtcNative: { configuration() { return configuration; } },
    VDONinjaSDK: SDK,
  };
  for (const [name, value] of Object.entries(globals)) {
    original.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  }
  t.after(async () => {
    page.dispatchEvent(new Event('pagehide'));
    await settle();
    for (const [name, descriptor] of original) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  });
  await import(moduleUrl + '#' + encodeURIComponent(t.name));
  await settle();
  return {
    page,
    expire() { configuration = ''; },
    get attempts() { return attempts; },
    get announcements() { return announcements; },
    get closed() { return closed; },
  };
}

test('Quest target retries initial signaling on the same page after the previous attempt closes', async t => {
  const owner = await loadTarget(t, async attempt => {
    if (attempt === 1) throw new Error('signaling temporarily unavailable');
  });
  assert.equal(owner.attempts, 1);
  assert.equal(owner.closed, 1);
  assert.equal(owner.announcements, 0);

  t.mock.timers.tick(1999);
  await settle();
  assert.equal(owner.attempts, 1);
  t.mock.timers.tick(1);
  await settle();
  assert.equal(owner.attempts, 2);
  assert.equal(owner.announcements, 1);

  t.mock.timers.tick(10_000);
  await settle();
  assert.equal(owner.attempts, 2);
});

test('Quest target cancels a scheduled retry on pagehide', async t => {
  const owner = await loadTarget(t, async () => { throw new Error('signaling unavailable'); });
  owner.page.dispatchEvent(new Event('pagehide'));
  t.mock.timers.tick(10_000);
  await settle();
  assert.equal(owner.attempts, 1);
  assert.equal(owner.closed, 1);
});

test('Quest target cannot retry if startup fails after pagehide', async t => {
  let fail;
  const owner = await loadTarget(t, () => new Promise((_, reject) => { fail = reject; }));
  owner.page.dispatchEvent(new Event('pagehide'));
  fail(new Error('late signaling failure'));
  await settle();
  t.mock.timers.tick(10_000);
  await settle();
  assert.equal(owner.attempts, 1);
  assert.equal(owner.closed, 1);
});


test('Quest target leaves an expired native offer retired', async t => {
  const owner = await loadTarget(t, async () => { throw new Error('signaling unavailable'); });
  owner.expire();
  t.mock.timers.tick(20_000);
  await settle();
  assert.equal(owner.attempts, 1);
  assert.equal(owner.announcements, 0);
});
