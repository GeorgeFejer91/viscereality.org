import test from 'node:test';
import assert from 'node:assert/strict';
import { loadStudyInventory, planMainStudy } from '../quest/main-study-audio.mjs';

test('accounts for every attachment and keeps distinct takes in their phase pools', async () => {
  const inventory = await loadStudyInventory();
  const sourceNames = inventory.assets.flatMap(asset => asset.sources);
  assert.equal(new Set(sourceNames).size, 23);
  assert.equal(inventory.assets.length, 22);
  assert.deepEqual(Object.fromEntries(Object.entries(inventory.pools).map(([phase, ids]) => [phase, ids.length])),
    { inhale: 7, exhale: 9, hold: 2 });
  assert.equal(inventory.opening.length, 3);
  assert.equal(inventory.assets.find(asset => asset.id === 'wisps-of-whorls').text, '');
});

test('plays each instruction once and rotates the correct cue pools on an exact 15-minute timeline', async () => {
  const inventory = await loadStudyInventory();
  const plan = planMainStudy(inventory);
  assert.equal(plan.total_frames, 900 * 44100);
  assert.equal(plan.cycles, 45);
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
  assert.equal(plan.events.at(-1).asset, null);
  for (const [pool, ids] of Object.entries(inventory.pools)) assert.deepEqual(used[pool], new Set(ids));
  assert.deepEqual(planMainStudy(inventory), plan);
});
