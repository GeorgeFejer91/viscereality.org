import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { buildMainStudyScenario, buildProbeStudyScenario, loadStudyInventory, planMainStudy } from '../quest/main-study-audio.mjs';

test('accounts for every attachment and keeps distinct takes in their phase pools', async () => {
  const inventory = await loadStudyInventory();
  const sourceNames = inventory.assets.flatMap(asset => asset.sources);
  assert.equal(new Set(sourceNames).size, 25);
  assert.equal(inventory.assets.length, 24);
  assert.deepEqual(Object.fromEntries(Object.entries(inventory.pools).map(([phase, ids]) => [phase, ids.length])),
    { inhale: 7, exhale: 9, hold: 2 });
  assert.equal(inventory.opening[2], 'main-study-instructions');
  assert.equal(inventory.assets.find(asset => asset.id === 'wisps-of-whorls').text, '');
});

test('matches the controls and rotates the cue pools within the 5 + 10 minute programme', async () => {
  const inventory = await loadStudyInventory();
  const plan = planMainStudy(inventory);
  assert.equal(plan.total_frames, 900 * 44100);
  assert.equal(plan.cycles, 37);
  assert.equal(plan.events[0].duration_frames, 5615640);
  assert.equal(plan.events[0].duration_frames, plan.events[1].duration_frames);
  assert.equal(plan.events[3].id, 'inhale');
  assert.equal(plan.events[3].offset_frames, 300 * 44100);
  assert.equal(plan.total_frames - plan.events[3].offset_frames, 600 * 44100);
  assert.deepEqual(plan.events.slice(-2).map(event => event.id), ['inhale', 'exhale']);
  assert.deepEqual(plan.events.slice(0, 3).map(event => event.asset), inventory.opening);
  let next = 0;
  const used = { inhale: new Set(), exhale: new Set(), hold: new Set() };
  const previous = new Map();
  for (const event of plan.events) {
    assert.equal(event.offset_frames, next);
    next += event.duration_frames;
    if (event.cycle === null) continue;
    const pool = event.id.startsWith('hold_') ? 'hold' : event.id;
    assert.equal(event.duration_frames, 4 * 44100);
    assert.ok(inventory.pools[pool].includes(event.asset));
    used[pool].add(event.asset);
    assert.notEqual(event.asset, previous.get(event.id), 'repeated take for ' + event.id);
    if (event.id === 'hold_out') assert.notEqual(event.asset, previous.get('hold_in'));
    previous.set(event.id, event.asset);
  }
  assert.equal(next, plan.total_frames);
  for (const [pool, ids] of Object.entries(inventory.pools)) assert.deepEqual(used[pool], new Set(ids));
  assert.deepEqual(planMainStudy(inventory), plan);
});

test('stages the probe instruction separately from exactly five minutes of breathwork', async () => {
  const root = fileURLToPath(new URL('../', import.meta.url));
  const original = async id => JSON.parse(await readFile(new URL('../audio/scenarios/' + id + '.json', import.meta.url)));
  const study = await buildMainStudyScenario(root, await original('haupt-15'));
  const { plan, files } = await buildProbeStudyScenario(root, await original('probe-5'), study);
  assert.equal(plan.total_frames, 300 * 44100);
  assert.equal(plan.pre_session_audio.duration_frames, 2002176);
  assert.equal(files.get(plan.pre_session_audio.path).length, 44 + 2 * plan.pre_session_audio.duration_frames);
  assert.match(plan.pre_session_audio.text, /für fünf Minuten/);
  assert.equal(plan.events[0].id, 'inhale');
  assert.equal(plan.events[0].offset_frames, 0);
  assert.deepEqual(plan.events.slice(-3).map(event => event.id), ['inhale', 'hold_in', 'exhale']);
  let next = 0;
  for (const event of plan.events) {
    assert.equal(event.offset_frames, next);
    next += event.duration_frames;
    assert.ok(files.has('assets/audio/scenarios/probe-5/' + plan.audio_assets[event.asset].file));
  }
  assert.equal(next, plan.total_frames);
});
