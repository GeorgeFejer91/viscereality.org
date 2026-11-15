import test from 'node:test';
import assert from 'node:assert/strict';
import { LiveBreathDisplay } from '../live-breath-display.js';

const view = (overrides = {}, progress = 0, inputVolume = null) => ({
  available: true, acceptance: 'device/epoch/native/1/1/automatic',
  radius: .46, phaseLabel: 'Einatmen', progress: { value: progress, max: 2 }, inputVolume,
  setup: { g: 1, attempt: 1, input: 'automatic', playing: true, audio_state: 'playing', audio_generation: 1,
    position_frames: 0, cycle_frames: 705600, ...overrides },
});
const observation = (i, a = 0) => ({ i: String(i), a });

test('Automatic percent follows the circle from the same fresh native phase clock', () => {
  const display = new LiveBreathDisplay();
  display.update(view({ position_frames: 88200 }), observation(1), 0);
  assert.equal(display.frame(0).inputVolume, .5);
  const frame = display.frame(500);
  assert.equal(frame.inputVolume, .625);
  assert.equal(frame.radius, .46 + .54 * frame.inputVolume);
  assert.equal(frame.progress, 0);
  assert.equal(display.frame(500, true).inputVolume, .5);
  const expired = display.frame(1500);
  assert.equal(display.frame(50000).inputVolume, expired.inputVolume);
  display.update(view({ position_frames: 352800 }), observation(2), 51000);
  assert.equal(display.frame(51500).inputVolume, .875);
});

test('parked Automatic stays at minimum with no local clock advance', () => {
  const display = new LiveBreathDisplay();
  display.update(view({ stage: 'idle', playing: false, audio_state: 'paused' }), observation(1), 0);
  assert.equal(display.frame(500).radius, .46);
  assert.equal(display.frame(500).inputVolume, 0);
  assert.equal(display.frame(500).animating, false);
});

test('measured input percent interpolates only between observed samples and stops at expiry', () => {
  for (const input of ['polar-tilt', 'controller-state']) {
    const display = new LiveBreathDisplay();
    display.update(view({ input }, 0, .2), observation(1), 0);
    display.update(view({ input }, 0, .8), observation(2), 1000);
    assert.equal(display.frame(1000).inputVolume, .2);
    assert.ok(Math.abs(display.frame(1500).inputVolume - .5) < 1e-12);
    assert.equal(display.frame(2000).inputVolume, .8);
    assert.equal(display.frame(50000).inputVolume, .8);
    assert.equal(display.frame(1500, true).inputVolume, .8);
    display.update(view({ input }, 0, .1), observation(2), 51000);
    assert.equal(display.frame(51000).inputVolume, .8, 'replay cannot introduce a new measured value');
  }
});

test('breathing presentation follows a fresh native audio anchor through the actual phase boundary', () => {
  const display = new LiveBreathDisplay();
  display.update(view({ position_frames: 154350 }), observation(1), 100);
  const halfSecond = display.frame(600);
  assert.equal(halfSecond.phaseLabel, 'Atem halten');
  assert.equal(halfSecond.radius, 1);
  assert.equal(halfSecond.live, true);
  assert.equal(halfSecond.progress, 0);
});

test('lost observations stop the circle; a duplicate cannot renew its freshness', () => {
  const display = new LiveBreathDisplay();
  display.update(view(), observation(1, 100), 0);
  const fresh = display.frame(1400);
  assert.equal(fresh.live, true);
  display.update(view(), observation(1, 100), 1300);
  const expired = display.frame(1500);
  assert.equal(expired.live, false);
  assert.equal(expired.radius, fresh.radius);
  assert.equal(display.frame(50000).radius, expired.radius);
});

test('calibration fill interpolates only to an observed native frame count', () => {
  const display = new LiveBreathDisplay();
  display.update(view(), observation(1), 0);
  display.update(view({}, .25), observation(2), 1000);
  assert.equal(display.frame(1000).progress, 0);
  assert.equal(display.frame(1500).progress, .125);
  assert.equal(display.frame(2000).progress, .25);
  assert.equal(display.frame(50000).progress, .25);
  display.update(view({ attempt: 2 }, 0), observation(3), 51000);
  assert.equal(display.frame(51000).progress, 0);
});

test('pause, retired audio generation and replacement owners reset presentation', () => {
  const display = new LiveBreathDisplay();
  display.update(view(), observation(1), 0);
  display.update(view({ playing: false, position_frames: 88200 }), observation(2), 1000);
  assert.equal(display.frame(2000).live, false);
  assert.equal(display.frame(2000).radius, .73);
  display.update(view({ audio_generation: 2 }, 0), observation(3), 2200);
  assert.equal(display.frame(2200).radius, .46);
  const replacement = view({}, 0);
  replacement.acceptance = 'device/new-epoch/native/1/1/automatic';
  display.update(replacement, observation(1), 2300);
  assert.equal(display.frame(2300).radius, .46);
  display.update({ available: false }, observation(4), 2400);
  assert.equal(display.frame(2400), null);
});

test('older native projections stay discrete and reduced motion uses the native sample', () => {
  const display = new LiveBreathDisplay();
  display.update(view({ audio_state: undefined, audio_generation: undefined }), observation(1), 0);
  assert.equal(display.frame(500).radius, .46);
  assert.equal(display.frame(500).live, false);
  display.update(view({ position_frames: 88200 }), observation(2), 1000);
  assert.equal(display.frame(1500, true).radius, .73);
  display.reset();
  assert.equal(display.frame(1600), null);
  display.update(view({ position_frames: 88200 }), observation(2), 20000);
  assert.equal(display.frame(20000), null, 'visibility recovery cannot make the previous observation fresh');
  display.update(view({ position_frames: 132300 }), observation(3), 21000);
  assert.equal(display.frame(21000).radius, .865);
});
