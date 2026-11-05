import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { BoxBreathingEngine, createProtocol } from '../audio/box-breathing.js';
import * as scenarios from '../audio/scenario-protocol.js';

const libraryUrl = new URL('../assets/audio/box-breathing/de/soft-v2/manifest.json', import.meta.url);
const libraryBytes = await readFile(libraryUrl);
const library = JSON.parse(libraryBytes);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const plans = [
  ['probe-anleitung', 'anleitung', 0], ['probe-5', 5, 0], ['probe-10', 10, 0],
  ['probe-15', 15, 0], ['haupt-10', 5, 5], ['haupt-15', 10, 5], ['box-20', 20, 0],
];
const scenarioUrl = id => new URL(`../audio/scenarios/${id}.json`, import.meta.url);

test('Hash-bound library and scenario files retain LF bytes across checkouts', async () => {
  assert.equal(libraryBytes.includes(13), false, 'CRLF changes the pinned library hash on GitHub Pages');
  for (const [id] of plans) {
    for (const extension of ['json', 'csv']) {
      const bytes = await readFile(new URL(`../audio/scenarios/${id}.${extension}`, import.meta.url));
      assert.equal(bytes.includes(13), false, `${id}.${extension} must use LF`);
    }
  }
});

test('Every scenario file preserves the approved sequence and its CSV names the exact shared assets', async () => {
  assert.deepEqual(scenarios.SCENARIOS, plans.map(([id]) => id));
  for (const [id, minutes, baselineMinutes] of plans) {
    const plan = scenarios.validateScenario(JSON.parse(await readFile(scenarioUrl(id))), library,
      { scenarioId: id, manifestHash: hash(libraryBytes) });
    const { schema_version, scenario_id, audio_library, library_manifest_sha256, ...protocol } = plan;
    const original = createProtocol(minutes, library, { baselineMinutes });
    assert.deepEqual(protocol, { ...original, events: original.events.map(event =>
      event.id === 'recovery' ? { ...event, asset: null } : event) });
    assert.equal(scenarios.scenarioId(minutes, { baselineMinutes }), id);
    assert.equal(new URL(audio_library, scenarioUrl(id)).href, libraryUrl.href);
    const csv = await readFile(new URL(`../audio/scenarios/${id}.csv`, import.meta.url), 'utf8');
    assert.equal(csv, scenarios.scenarioCsv(plan, library));
    assert.equal(csv.trim().split('\n').length, plan.events.length + 1);
    const firstInhale = csv.trim().split('\n').map(row => row.split(',')).find(row => row[1] === 'inhale');
    assert.equal(firstInhale[2], 'inhale');
    assert.equal(firstInhale[3], 'assets/audio/box-breathing/de/soft-v2/inhale.wav');
    assert.deepEqual(firstInhale.slice(4, 7), baselineMinutes ? ['340', '4', '344'] : ['40', '4', '44']);
  }
  for (const [minutes, baselineMinutes] of [[5, 4], [15, 5], ['anleitung', 5], ['5', 0]]) {
    assert.throws(() => scenarios.scenarioId(minutes, { baselineMinutes }));
  }
});

test('Invalid scenario timing, assets, identities and library revisions fail before scheduling', async () => {
  const valid = JSON.parse(await readFile(scenarioUrl('probe-5')));
  const check = plan => scenarios.validateScenario(plan, library,
    { scenarioId: 'probe-5', manifestHash: hash(libraryBytes) });
  const invalid = [
    plan => { plan.events[1].asset = 'missing'; },
    plan => { plan.events[1].offset_frames++; },
    plan => { plan.events[1].duration_frames = -1; },
    plan => { plan.events[1].duration_frames = 176400.5; },
    plan => { plan.events[1].asset = null; },
    plan => { plan.events[0].asset = null; },
    plan => { plan.events.at(-1).asset = 'inhale'; },
    plan => { plan.total_frames++; },
    plan => { plan.sample_rate = 48000; },
    plan => { plan.scenario_id = 'haupt-10'; },
    plan => { plan.library_manifest_sha256 = '0'.repeat(64); },
    plan => { plan.audio_library = 'https://example.invalid/library.json'; },
    plan => { plan.cycles++; },
    plan => { plan.events = []; },
  ];
  for (const mutate of invalid) {
    const plan = structuredClone(valid); mutate(plan);
    assert.throws(() => check(plan));
  }
});

class Context extends EventTarget {
  constructor() { super(); this.currentTime = 10; this.sampleRate = 44100; this.state = 'running'; this.nodes = []; this.destination = {}; }
  async resume() { this.state = 'running'; }
  createBuffer(channels, frames, rate) { return { numberOfChannels: channels, duration: frames / rate }; }
  createBufferSource() {
    const node = { connect() {}, disconnect() {}, start(...args) { this.startArgs = args; }, stop() {} };
    this.nodes.push(node); return node;
  }
  async decodeAudioData(bytes) { return { numberOfChannels: 1, duration: new DataView(bytes).getUint32(40, true) / 88200 }; }
}

test('Playback uses the loaded timeline, records its hash, and maps all final choices', async () => {
  const originalFetch = globalThis.fetch;
  let changedBytes;
  globalThis.fetch = async url => {
    const bytes = await readFile(url);
    if (String(url).endsWith('probe-5.json')) {
      const plan = JSON.parse(bytes);
      plan.events[1].asset = 'exhale'; // Same length: proves runtime follows authored data.
      changedBytes = JSON.stringify(plan);
      return new Response(changedBytes);
    }
    return new Response(bytes);
  };
  const context = new Context();
  const engine = new BoxBreathingEngine({ audioContext: context });
  try {
    await engine.prepare();
    for (const [id, minutes, baselineMinutes] of plans) {
      assert.equal(engine.protocol(minutes, { baselineMinutes }).scenario_id, id);
    }
    const session = await engine.start(5);
    assert.equal(session.protocol.events[1].asset, 'exhale');
    assert.equal(context.nodes[1].buffer, engine.buffers.exhale);
    assert.equal(engine.receipt().scenario_id, 'probe-5');
    assert.equal(engine.receipt().scenario_sha256, hash(changedBytes));
    assert.ok(Object.isFrozen(session.protocol.events[1]));
    engine.abort();
    await session.done;
  } finally { globalThis.fetch = originalFetch; await engine.dispose(); }
});

test('A missing or invalid scenario prevents preparation and cannot start fallback audio', async () => {
  const originalFetch = globalThis.fetch;
  try {
    for (const missing of [true, false]) {
      globalThis.fetch = async url => {
        if (String(url).endsWith('haupt-15.json')) {
          return missing ? new Response('', { status: 404 }) : new Response('{}');
        }
        return new Response(await readFile(url));
      };
      const context = new Context();
      const engine = new BoxBreathingEngine({ audioContext: context });
      try {
        await assert.rejects(engine.prepare(), /Scenario|scenario/);
        await assert.rejects(engine.start(5), /Prepare/);
        assert.equal(context.nodes.length, 0);
        assert.equal(engine.buffers, null);
      } finally { await engine.dispose(); }
    }
  } finally { globalThis.fetch = originalFetch; }
});
