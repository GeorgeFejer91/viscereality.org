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

test('matches the controls and reuses the three canonical cues in the 5 + 10 minute programme', async () => {
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
  const cueIds = { inhale: 'zen-inhale', hold_in: 'zen-hold', exhale: 'zen-exhale', hold_out: 'zen-hold' };
  for (const event of plan.events) {
    assert.equal(event.offset_frames, next);
    next += event.duration_frames;
    if (event.cycle === null) continue;
    assert.equal(event.duration_frames, 4 * 44100);
    assert.equal(event.asset, cueIds[event.id]);
  }
  assert.equal(next, plan.total_frames);
  assert.deepEqual(planMainStudy(inventory), plan);
});

test('stages the probe instruction separately from exactly five minutes of breathwork', async () => {
  const root = fileURLToPath(new URL('../', import.meta.url));
  const original = async id => JSON.parse(await readFile(new URL('../audio/scenarios/' + id + '.json', import.meta.url)));
  const study = await buildMainStudyScenario(root, await original('haupt-15'));
  const shared = new URL('../assets/audio/box-breathing/de/soft-v2/', import.meta.url);
  const canonical = { 'zen-inhale': 'inhale_breath.wav', 'zen-hold': 'hold_in.wav', 'zen-exhale': 'exhale_breath.wav' };
  for (const [id, file] of Object.entries(canonical)) {
    const asset = study.plan.audio_assets[id];
    assert.deepEqual(study.files.get('assets/audio/scenarios/haupt-15/' + asset.file),
      await readFile(new URL(file, shared)));
  }
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
