import assert from 'node:assert/strict';
import test from 'node:test';
import { LiveSphereVolume } from '../live-sphere-volume.js';

const renderer = (frame, time, changes = {}) => ({
  v: 1, frame, render_generation: 1, settings_revision: 3, observed_ns: String(time * 1e6),
  age_ns: 0, radius_m: 1.5, progress01: .5,
  radius_min_m: 1, radius_max_m: 2, cue_progress01: .25, cue_observed_ns: String((time - 20) * 1e6),
  audio_generation: 2, ...changes,
});
const update = (chart, sample, now, changes = {}) => chart.update({
  key: 'owner/7', paused: false, sample, age: 0, generation: 7, witnessGeneration: 7,
  elapsed: 1500, audio: 2, ...changes,
}, now);

test('plots the same signed radius progress as the preview on a fixed percentage scale', () => {
  const chart = new LiveSphereVolume();
  update(chart, renderer(1, 100), 1000);
  update(chart, renderer(2, 200, {radius_m: 1.7, progress01: .75, cue_progress01: .5}), 1110);
  const result = chart.frame(1110, true);
  assert.equal(result.actual, .75);
  assert.equal(result.intended, .5);
  assert.deepEqual(result.bounds, [0, 1]);
  assert.match(result.actualPath, /^M397\.5 35\.0 h0\.01 L400\.0 19\.5/);
  assert.match(result.intendedPath, /^M397\.0 50\.5 h0\.01 L399\.5 35\.0/);
  update(chart, renderer(3, 300, {progress01: 1.01}), 1220);
  assert.equal(chart.frame(1220).actual, null);
});

test('plots legacy renderer progress without inventing a cue', () => {
  const chart = new LiveSphereVolume();
  update(chart, renderer(1, 100, {radius_min_m: undefined, radius_max_m: undefined, cue_progress01: undefined}), 1000);
  const result = chart.frame(1000);
  assert.equal(result.actual, .5);
  assert.equal(result.intended, null);
  assert.equal(result.intendedPath, '');
});

test('duplicates, old renderer generations and stale observations never refresh the live label', () => {
  const chart = new LiveSphereVolume();
  const sample = renderer(2, 110);
  update(chart, sample, 1000);
  update(chart, sample, 1400);
  update(chart, renderer(1, 105), 1501);
  assert.equal(chart.frame(1501).actual, null);
  const frame = chart.frame(1501);
  assert.match(frame.actualPath, /^M/);
  update(chart, renderer(3, 120, {age_ns: 470000000}), 1600, {age: 40});
  assert.equal(chart.frame(1600).actual, null);
});

test('pause freezes history, resume creates a break, and session replacement drops old data', () => {
  const chart = new LiveSphereVolume();
  update(chart, renderer(1, 100), 1000);
  update(chart, renderer(2, 200), 1100, {paused: true});
  const paused = chart.frame(1100);
  assert.equal(paused.paused, true);
  assert.equal(chart.frame(1600).actualPath, paused.actualPath);
  update(chart, renderer(3, 800), 1700);
  assert.equal((chart.frame(1700).actualPath.match(/M/g) ?? []).length, 2);
  update(chart, renderer(4, 900), 1800, {key: 'owner/8', generation: 8});
  assert.equal(chart.frame(1800).intendedPath, '');
  assert.equal(chart.frame(1800).actualPath, '');
  update(chart, renderer(5, 1000), 1900, {key: 'owner/8', generation: 8, witnessGeneration: 8});
  assert.equal((chart.frame(1900).actualPath.match(/M/g) ?? []).length, 1);
});

test('rejects a cue from another audio generation and opens gaps for missing frames', () => {
  const chart = new LiveSphereVolume();
  update(chart, renderer(1, 100, {audio_generation: 1}), 1000);
  assert.equal(chart.frame(1000).intended, null);
  update(chart, null, 1200);
  assert.equal(chart.frame(1200).actual, null);
  update(chart, renderer(3, 400), 1300);
  assert.equal((chart.frame(1300).actualPath.match(/M/g) ?? []).length, 2);
});

test('incomparable, repeated and pre-run cue timestamps cannot become current', () => {
  const chart = new LiveSphereVolume();
  update(chart, renderer(1, 100, {cue_observed_ns:'101000000'}), 1000);
  assert.equal(chart.frame(1000).intended, null);
  update(chart, renderer(2, 200, {cue_observed_ns:'190000000'}), 1100, {elapsed:5});
  assert.equal(chart.frame(1100).intended, null);
  update(chart, renderer(3, 300), 1200);
  assert.equal(chart.frame(1200).intended, .25);
  update(chart, renderer(4, 500, {cue_observed_ns:'280000000'}), 1400);
  assert.equal((chart.frame(1400).intendedPath.match(/M/g) ?? []).length, 1);
  assert.equal(chart.frame(1690).intended, null);
  assert.equal(chart.frame(1690).actual, .5);
});

test('resume discards the last paused frame and a setting change keeps the same axis', () => {
  const chart = new LiveSphereVolume();
  update(chart,renderer(1,100),1000);
  const pause = renderer(2,200);
  update(chart,pause,1100,{paused:true});
  update(chart,pause,1110);
  assert.equal(chart.frame(1110).actual,null);
  update(chart,renderer(3,300,{
    settings_revision:4, radius_m:.2, radius_min_m:.1, radius_max_m:.3, progress01:1,
  }),1200);
  const frame = chart.frame(1200);
  assert.equal(frame.actual, 1);
  assert.match(frame.actualPath,/^M400\.0 4\.0/);
  assert.equal((frame.actualPath.match(/M/g) ?? []).length,1);
});

test('uneven delivery preserves native spacing without pushing either point into the future', () => {
  const chart = new LiveSphereVolume();
  update(chart,renderer(1,100),1000);
  update(chart,renderer(2,200,{radius_m:1.75,progress01:.75}),1080,{age:4});
  const frame = chart.frame(1080);
  assert.equal(frame.actual,.75);
  assert.match(frame.actualPath,/^M397\.4 .* L399\.9 /);
});
