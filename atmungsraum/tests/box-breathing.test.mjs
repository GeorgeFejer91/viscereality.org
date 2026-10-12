import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { BoxBreathingEngine, createProtocol, PHASES, scheduleProtocol } from '../audio/box-breathing.js';
import { mountSessionAudio } from '../audio/session-ui.js';
import { state as uiState } from '../ui-logic.js';
import { setTimeout as delay } from 'node:timers/promises';

const assetDir = new URL('../assets/audio/box-breathing/de/soft-v2/', import.meta.url);
const manifestBytes = await readFile(new URL('manifest.json', assetDir));
const manifest = JSON.parse(manifestBytes);
const buffers = Object.fromEntries(Object.entries(manifest.blocks).map(([id, block]) =>
  [id, { numberOfChannels: 1, duration: block.frames / manifest.sample_rate }]));

class Context extends EventTarget {
  constructor() { super(); this.currentTime = 10; this.sampleRate = 48000; this.state = 'running'; this.nodes = []; this.destination = {}; }
  async resume() { this.state = 'running'; this.dispatchEvent(new Event('statechange')); }
  async suspend() { this.state = 'suspended'; this.dispatchEvent(new Event('statechange')); }
  createBuffer(channels, frames, rate) { return { numberOfChannels: channels, duration: frames / rate }; }
  createBufferSource() {
    const node = { connect() {}, disconnect() { this.disconnected = true; },
      start(...args) { this.startArgs = args; }, stop(...args) { this.stopArgs = args; } };
    this.nodes.push(node); return node;
  }
  async decodeAudioData(bytes) { return { numberOfChannels: 1, duration: new DataView(bytes).getUint32(40, true) / 2 / 44100 }; }
}

test('All durations contain complete cycles, contiguous slots, and the approved remainder', () => {
  for (const [minutes, cycles, rests, recovery] of [[5, 15, 0, 8], [10, 33, 1, 0], [15, 50, 2, 8], [20, 68, 3, 0]]) {
    const plan = createProtocol(minutes, manifest);
    assert.equal(plan.cycles, cycles);
    assert.equal(plan.rests, rests);
    assert.ok(plan.breathing_blocks.every((count) => count <= 18));
    assert.ok(plan.breathing_blocks.slice(0, -1).every((count) => count === 18));
    assert.equal(plan.recovery_frames / plan.sample_rate, recovery);
    assert.equal(plan.total_frames / plan.sample_rate, minutes * 60);
    let next = 0;
    for (const event of plan.events) {
      assert.equal(event.offset_frames, next);
      if (event.cycle) {
        assert.equal(event.duration_frames, 176400);
        const mixed = event.cycle % 3 === 0 && ['inhale', 'exhale'].includes(event.id);
        assert.equal(event.asset, mixed ? `${event.id}_breath` : event.id);
      }
      if (event.id === 'rest') assert.equal(event.duration_frames, 20 * 44100);
      next += event.duration_frames;
    }
    assert.equal(next, plan.total_frames);
    assert.deepEqual(plan.events.slice(1, 5).map((event) => event.id), PHASES);
    assert.deepEqual(plan.events.filter((event) => event.cycle === cycles).map((event) => event.id), PHASES);
  }
  for (const invalid of [0, 4, 25, '5', NaN, Infinity]) assert.throws(() => createProtocol(invalid, manifest));
  const instruction = createProtocol('anleitung', manifest);
  assert.equal(instruction.total_frames / instruction.sample_rate, 68);
  assert.equal(instruction.cycles, 1);
  assert.equal(instruction.minutes, null);
  assert.equal(instruction.mode, 'instruction-example');
  assert.throws(() => createProtocol(5, { ...manifest, block_frames: 176401 }));
});

test('Actual PCM blocks match their manifest, contain speech, and have quiet joins', async () => {
  for (const [id, block] of Object.entries(manifest.blocks)) {
    const bytes = await readFile(new URL(block.file, assetDir));
    assert.equal(bytes.length, block.bytes, id);
    assert.equal(createHash('sha256').update(bytes).digest('hex'), block.sha256, id);
    assert.equal(bytes.readUInt32LE(24), 44100);
    assert.equal(bytes.readUInt32LE(40), block.frames * 2);
    assert.equal(bytes.readUInt16LE(22), 1);
    assert.equal(bytes.readUInt16LE(34), 16);
    assert.equal(bytes.readInt16LE(44), 0);
    assert.equal(bytes.readInt16LE(bytes.length - 2), 0);
    let peak = 0;
    for (let i = 44; i < bytes.length; i += 2) peak = Math.max(peak, Math.abs(bytes.readInt16LE(i)));
    assert.ok(peak > 100 && peak < 32767, id);
    if (['hold_in', 'hold_out'].includes(id)) {
      assert.equal(block.text, 'Und halten.');
      assert.deepEqual(block.processing.word_onsets_frames, [0]);
      // The rest of a hold is quiet, rather than a spoken count sequence.
      assert.ok(bytes.subarray(44 + 2 * 44100 * 2).every(byte => byte === 0));
    }
  }
  assert.equal(manifest.blocks.inhale.text, 'Einatmen. Zwei. Drei. Vier.');
  assert.equal(manifest.blocks.exhale.text, 'Ausatmen. Zwei. Drei. Vier.');
});

test('Scheduler anchors every slot to one clock, including final silent recovery', () => {
  const context = new Context();
  const plan = createProtocol(15, manifest);
  const nodes = scheduleProtocol(context, buffers, plan, 10.25);
  for (const { source, event } of nodes) {
    assert.deepEqual(source.startArgs, [10.25 + event.offset_frames / 44100, 0, event.duration_frames / 44100]);
    assert.deepEqual(source.stopArgs, [10.25 + (event.offset_frames + event.duration_frames) / 44100]);
  }
  assert.equal(nodes.at(-1).source.stopArgs[0], 910.25);
  assert.throws(() => scheduleProtocol(context, buffers, plan, 9));
  const invalid = new Context();
  assert.throws(() => scheduleProtocol(invalid, { ...buffers, exhale: { numberOfChannels: 1, duration: 4.1 } }, plan, 11));
  assert.ok(invalid.nodes.every((node) => !node.startArgs && node.disconnected));
});

test('Engine verifies assets, rejects retrigger, aborts interrupted clocks, and ignores stale completion', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => new Response(await readFile(url));
  const context = new Context();
  const engine = new BoxBreathingEngine({ audioContext: context, manifestUrl: new URL('manifest.json', assetDir) });
  try {
    await engine.prepare();
    const first = await engine.start(5);
    const stale = engine.run.sources.at(-1).source.onended;
    await assert.rejects(engine.start(10), /already active/);
    context.state = 'suspended'; context.dispatchEvent(new Event('statechange'));
    const interrupted = await first.done;
    assert.equal(interrupted.status, 'aborted');
    assert.equal(interrupted.records.at(-1).reason, 'audio-context-suspended');
    assert.ok(context.nodes.every((node) => node.disconnected && node.stopArgs.length === 0));
    const second = await engine.start(10);
    stale();
    assert.equal(engine.run.status, 'scheduled');
    context.currentTime = second.endAt + 0.2;
    engine.run.sources.at(-1).source.onended();
    const receipt = await second.done;
    assert.equal(receipt.status, 'completed');
    assert.equal(receipt.end_at_audio_seconds - receipt.start_at_audio_seconds, 600);
    const lesson = await engine.start('anleitung');
    context.currentTime = lesson.endAt;
    engine.run.sources.at(-1).source.onended();
    assert.equal((await lesson.done).status, 'completed');
    engine.abort(); assert.equal(engine.receipt().status, 'completed');
    await engine.dispose();
    await assert.rejects(engine.start(5), /Prepare/);
  } finally { globalThis.fetch = originalFetch; await engine.dispose(); }
});

test('Pause/resume keeps the audio plan and clock anchor, and cancelled resume stays stopped', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => new Response(await readFile(url));
  const context = new Context();
  const engine = new BoxBreathingEngine({ audioContext: context, manifestUrl: new URL('manifest.json', assetDir) });
  try {
    await engine.prepare();
    const session = await engine.start(5);
    const nodes = context.nodes.length;
    context.currentTime = session.startAt + 12;
    assert.equal(await engine.pause(), true);
    assert.equal(context.state, 'suspended');
    assert.equal(engine.position().status, 'paused');
    assert.equal(engine.position().elapsed_seconds, 12);
    assert.equal(engine.receipt().records.at(-1).type, 'pause');
    await assert.rejects(engine.start(10), /already active/);
    assert.equal(await engine.resume(), true);
    assert.equal(engine.position().status, 'scheduled');
    assert.equal(engine.position().elapsed_seconds, 12);
    assert.equal(engine.run.start_at_audio_seconds, session.startAt);
    assert.equal(engine.run.end_at_audio_seconds, session.endAt);
    assert.equal(context.nodes.length, nodes);
    assert.equal(engine.receipt().records.at(-1).type, 'resume');
    assert.equal(await engine.pause(), true);
    let unlock;
    context.resume = () => new Promise(resolve => { unlock = resolve; });
    const pending = engine.resume();
    engine.abort('user-abort');
    unlock();
    assert.equal(await pending, false);
    assert.equal((await session.done).status, 'aborted');
    assert.equal(engine.run.status, 'aborted');
    assert.equal(context.nodes.length, nodes);
  } finally { globalThis.fetch = originalFetch; await engine.dispose(); }
});

test('Corrupted audio fails preparation before scheduling', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => new Response(String(url).endsWith('manifest.json') ? manifestBytes : Buffer.alloc(10));
  const context = new Context();
  const engine = new BoxBreathingEngine({ audioContext: context, manifestUrl: new URL('manifest.json', assetDir) });
  try {
    await assert.rejects(engine.prepare(), /integrity mismatch/);
    await assert.rejects(engine.start(5), /Prepare/);
    assert.equal(context.nodes.length, 0);
  } finally { globalThis.fetch = originalFetch; await engine.dispose(); }
});

test('Stopping during browser audio unlock never saves a previous protocol under the new selection', async () => {
  class Element extends EventTarget {
    constructor() { super(); this.dataset = {}; this.textContent = ''; }
    querySelector() { return this.label ??= new Element(); }
    setAttribute(name, value) { this[name] = value; }
  }
  const elements = Object.fromEntries(['session-start', 'play', 'stop', 'session-phase', 'session-time', 'start-status', 'session-controls', 'session-resume', 'session-pause'].map(id => [id, new Element()]));
  const storage = new Map();
  const keys = ['document', 'window', 'AudioContext', 'localStorage', 'requestAnimationFrame', 'cancelAnimationFrame', 'fetch'];
  const originals = Object.fromEntries(keys.map(key => [key, globalThis[key]]));
  const originalUi = { ...uiState };
  let context;
  const doc = new EventTarget(); doc.hidden = false; doc.getElementById = id => elements[id];
  globalThis.document = doc;
  globalThis.window = new EventTarget();
  globalThis.AudioContext = class extends Context { constructor() { super(); context = this; } };
  globalThis.localStorage = { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) };
  globalThis.requestAnimationFrame = () => 1;
  globalThis.cancelAnimationFrame = () => {};
  globalThis.fetch = async url => new Response(await readFile(new URL(url)));
  const ready = async predicate => {
    const deadline = Date.now() + 2000;
    while (!predicate() && Date.now() < deadline) await delay(5);
    assert.ok(predicate(), 'browser controller did not reach the expected state');
  };
  try {
    Object.assign(uiState, { sensor: 'ecg', selection: { branch: 'haupt', value: '5', minutes: 5 }, started: false });
    const controller = mountSessionAudio();
    controller.update('session-start', uiState.selection, true);
    await ready(() => elements['session-start'].dataset.audioState === 'ready');
    elements.play.dispatchEvent(new Event('click'));
    await ready(() => elements['session-start'].dataset.audioState === 'playing');
    assert.equal(elements['session-controls'].hidden, false);
    assert.equal(elements['session-pause'].disabled, false);
    assert.equal(elements['session-resume'].disabled, true);
    const nodesBeforePause = context.nodes.length;
    elements['session-pause'].dispatchEvent(new Event('click'));
    await ready(() => elements['session-start'].dataset.audioState === 'paused');
    assert.equal(context.state, 'suspended');
    assert.equal(elements['session-resume'].disabled, false);
    assert.equal(elements['session-pause'].disabled, true);
    assert.equal(uiState.started, true);
    elements['session-resume'].dispatchEvent(new Event('click'));
    await ready(() => elements['session-start'].dataset.audioState === 'playing');
    assert.equal(context.state, 'running');
    assert.equal(context.nodes.length, nodesBeforePause, 'Start after Pause must not create a new session');
    elements.stop.dispatchEvent(new Event('click'));
    const previous = JSON.parse(storage.get('atmungsraum.audio-receipts'));
    assert.equal(previous.length, 1);
    assert.equal(previous[0].selection.minutes, 5);
    assert.equal(previous[0].status, 'aborted');
    uiState.selection = { branch: 'haupt', value: '10', minutes: 10 };
    controller.update('session-start', uiState.selection, true);
    let unlock;
    context.resume = () => new Promise(resolve => { unlock = resolve; });
    elements.play.dispatchEvent(new Event('click'));
    assert.equal(elements['session-start'].dataset.audioState, 'starting');
    elements.stop.dispatchEvent(new Event('click'));
    unlock(); await delay(0);
    assert.equal(elements['session-start'].dataset.audioState, 'stopped');
    assert.equal(elements.play.disabled, false);
    assert.equal(uiState.started, false);
    assert.deepEqual(JSON.parse(storage.get('atmungsraum.audio-receipts')), previous);
  } finally {
    for (const key of keys) {
      if (originals[key] === undefined) delete globalThis[key]; else globalThis[key] = originals[key];
    }
    Object.assign(uiState, originalUi);
  }
});
