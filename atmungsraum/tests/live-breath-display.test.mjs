import test from 'node:test';
import assert from 'node:assert/strict';
import { LiveBreathDisplay } from '../live-breath-display.js';
import { readFileSync } from 'node:fs';

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

const applied = (progress01, frame = 1, patch = {}) => ({ v: 1, progress01, radius_m: 1 + progress01,
  frame, render_generation: 2, settings_revision: 0, observed_ns: String(frame * 200000000), age_ns: 0, ...patch });
const rendered = (renderer, overrides = {}) => ({ ...view(overrides, 0, .9), renderer });

test('an applied renderer frame cannot be replayed across recording boundaries', () => {
  const display = new LiveBreathDisplay();
  display.update(rendered(applied(.9, 1)), observation(1), 200);
  assert.equal(display.frame(200).actualProgress, .9);
  display.update(rendered(applied(.9, 1)), observation(2), 220, 1);
  assert.equal(display.frame(220).actualProgress, null);
  display.update(rendered(applied(.25, 2)), observation(3), 400, 1);
  assert.equal(display.frame(400).actualProgress, .25);
  display.update(rendered(applied(.25, 2)), observation(4), 420, 2);
  assert.equal(display.frame(420).actualProgress, null);
  display.update(rendered(applied(.25, 2)), observation(5), 450, 2);
  assert.equal(display.frame(450).actualProgress, null);
  display.update(rendered(applied(.7, 3)), observation(6), 600, 2);
  assert.equal(display.frame(600).actualProgress, .7);
});

// Anonymous rounded timing only: response completion, full source/transport age,
// and relative GPU timestamp. No phone payload, owner, participant or raw signal.
const capturedCadence = JSON.parse(readFileSync(new URL('./fixtures/actual-radius-cadence.json', import.meta.url)));

test('ACTUAL adjacent-sample playout removes repeated interior ramp stops under captured early/late status timing', () => {
  const display = new LiveBreathDisplay();
  const arrivals = capturedCadence.map(([received, age, stamp], i) => ({ received, age, stamp, i })).sort((a, b) => a.received - b.received);
  let next = 0, previous = null, stopped = 0, longest = 0, run = 0, frames = 0;
  for (let now = 0; now < 17500; now += 10) {
    while (arrivals[next]?.received <= now) {
      const sample = arrivals[next++];
      display.update(rendered(applied(.1 + sample.stamp / 22000, sample.i + 1,
        { observed_ns: String(Math.round(sample.stamp * 1e6)) })), observation(sample.i + 1, sample.age), sample.received);
    }
    const current = display.frame(now);
    if (now > 1000 && current?.actualLive) {
      frames++;
      if (Math.abs(current.actualProgress - previous) < 1e-10) { stopped++; run += 10; longest = Math.max(longest, run); }
      else run = 0;
    }
    previous = current?.actualProgress;
  }
  assert.ok(stopped / frames < .10, `${stopped}/${frames} interior ramp frames stopped; longest ${longest}ms`);
  assert.ok(longest <= 170, `interior ramp stop lasts ${longest}ms`);
});

test('captured timing has continuous exact ACTUAL motion wherever a fresh observed source bracket exists', () => {
  const display = new LiveBreathDisplay();
  const arrivals = capturedCadence.map(([received, age, stamp], i) => ({ received, age, stamp, i })).sort((a, b) => a.received - b.received);
  const accepted = [];
  let next = 0, cursor = -1, covered = 0;
  for (let now = 0; now < 17500; now += 10) {
    while (arrivals[next]?.received <= now) {
      const sample = arrivals[next++];
      display.update(rendered(applied(.1 + sample.stamp / 22000, sample.i + 1,
        { observed_ns: String(Math.round(sample.stamp * 1e6)) })), observation(sample.i + 1, sample.age), sample.received);
      if (sample.i > cursor) { accepted.push(sample); cursor = sample.i; }
    }
    const fresh = accepted.filter(sample => now <= sample.received + 500 - sample.age);
    // First sample is 68.3ms old at receipt. A 280ms source delay therefore
    // requests native time now-211.7, independently of all later HTTP jitter.
    const sourceTime = now - 211.7;
    if (fresh.some((sample, i) => sample.stamp <= sourceTime && sourceTime < fresh[i + 1]?.stamp)) {
      const current = display.frame(now);
      const expected = .1 + sourceTime / 22000;
      assert.ok(Math.abs(current.actualProgress - expected) < 1e-10, `source bracket at ${now}ms`);
      assert.ok(Math.abs(current.actualRadius - (.46 + .54 * expected)) < 1e-10);
      assert.equal(current.actualLive, true);
      covered++;
    }
  }
  assert.ok(covered > 1400, `${covered} frames with real fresh bracket coverage`);
});

test('100ms singleflight reads absorb captured transport jitter without repeated interior ramp stops', () => {
  const display = new LiveBreathDisplay();
  const arrivals = [];
  let start = 0, firstStamp;
  for (const [, age] of capturedCadence) {
    // Reuse the captured age distribution with a neutral 8ms GPU + 1ms carrier
    // age. Requests can start only on the next 100ms tick after completion.
    const transport = Math.max(1, age - 9);
    const stamp = start + transport / 2 - 8;
    firstStamp ??= stamp;
    arrivals.push({ received: start + transport, age, stamp: stamp - firstStamp });
    start = Math.ceil((start + transport) / 100) * 100;
  }
  let next = 0, prior = null, stopped = 0, frames = 0, run = 0, longest = 0;
  for (let now = 0; now < arrivals.at(-1).received; now += 10) {
    while (arrivals[next]?.received <= now) {
      const sample = arrivals[next++];
      display.update(rendered(applied(.1 + sample.stamp / 22000, next,
        { observed_ns: String(Math.round(sample.stamp * 1e6)) })), observation(next, sample.age), sample.received);
    }
    const current = display.frame(now);
    if (now > 1000) {
      frames++;
      assert.equal(current.actualLive, true);
      if (Math.abs(current.actualProgress - prior) < 1e-10) { stopped++; run += 10; longest = Math.max(longest, run); }
      else run = 0;
    }
    prior = current?.actualProgress;
  }
  assert.ok(stopped / frames < .02, `${stopped}/${frames} stopped frames; longest ${longest}ms`);
  assert.ok(longest <= 80, `uncovered interval lasts ${longest}ms`);
});

test('ACTUAL history preserves real holds and never extrapolates an unobserved future radius', () => {
  const display = new LiveBreathDisplay();
  for (const [i, progress] of [[0, .2], [1, .4], [2, .4], [3, .3]]) {
    display.update(rendered(applied(progress, i + 1)), observation(i + 1), i * 200);
  }
  assert.equal(display.frame(680).actualProgress, .4, 'observed hold stays flat while desired audio can move');
  assert.ok(Math.abs(display.frame(750).actualProgress - .365) < 1e-12);
  assert.equal(display.frame(900).actualProgress, .3, 'source playout clamps to its newest actual observation');
  assert.equal(display.frame(1101).actualLive, false);
  assert.equal(display.frame(50000).actualProgress, .3);
  display.update(rendered(applied(.7, 5)), observation(5), 51000);
  assert.equal(display.frame(51000).actualProgress, .7, 'stale recovery snaps rather than sweeping old history');
});

test('ACTUAL blends only renderer-applied progress, with independent age and replay fences', () => {
  const display = new LiveBreathDisplay();
  display.update(rendered(applied(.2)), observation(1), 0);
  assert.equal(display.frame(0).actualProgress, .2);
  assert.equal(display.frame(0).actualRadius, .46 + .54 * .2);
  assert.equal(display.frame(0).actualLive, true);
  display.update(rendered(applied(.8, 2)), observation(2), 1000);
  assert.equal(display.frame(1000).actualProgress, .8, 'fresh resumption cannot animate a stale endpoint as current');
  assert.equal(display.frame(1500).actualProgress, .8);
  assert.equal(display.frame(1501).actualLive, false);
  assert.equal(display.frame(50000).actualProgress, .8, 'stale ACTUAL freezes without waveform extrapolation');
  display.update(rendered(applied(.1, 2)), observation(3), 51000);
  assert.equal(display.frame(51000).actualLive, false, 'new status cannot renew a replayed renderer frame');
  display.update(rendered(applied(.4, 3, { age_ns: 400000000 })), observation(4, 90), 52000);
  assert.equal(display.frame(52010).actualLive, true);
  assert.equal(display.frame(52011).actualLive, false, 'renderer age plus outer age share the 500ms budget');
});

test('ACTUAL source-time playout shares one physical value between circle and percent at 200ms cadence', () => {
  const display = new LiveBreathDisplay();
  display.update(rendered(applied(.2)), observation(1), 0);
  display.update(rendered(applied(.4, 2)), observation(2), 200);
  assert.equal(display.frame(200).actualProgress, .2);
  assert.ok(Math.abs(display.frame(300).actualProgress - .22) < 1e-12);
  assert.ok(Math.abs(display.frame(400).actualProgress - .32) < 1e-12);
  display.update(rendered(applied(.6, 3)), observation(3), 400);
  for (const [now, expected] of [[450, .37], [500, .42], [550, .47], [600, .52]]) {
    const current = display.frame(now);
    assert.ok(Math.abs(current.actualProgress - expected) < 1e-12);
    assert.ok(Math.abs(current.actualRadius - (.46 + .54 * expected)) < 1e-12);
    assert.equal(current.actualLive, true);
  }
  assert.equal(display.frame(500, true).actualProgress, .6, 'reduced motion shows the latest real GPU sample');
});

test('ACTUAL cannot fill a one-second polling gap with imagined or freshened GPU samples', () => {
  const display = new LiveBreathDisplay();
  display.update(rendered(applied(.2)), observation(1), 0);
  display.update(rendered(applied(.8, 2)), observation(2), 1000);
  assert.equal(display.frame(1000).actualProgress, .8, 'resume immediately at the fresh sample after a stale gap');
  assert.equal(display.frame(1499).actualLive, true);
  const expired = display.frame(1500);
  assert.equal(expired.actualProgress, .8);
  assert.equal(display.frame(1750).actualLive, false);
  assert.equal(display.frame(1750).actualProgress, expired.actualProgress);
  assert.equal(display.frame(50000).actualProgress, expired.actualProgress);
});

test('ACTUAL stays smooth at 200ms cadence with 100ms of native and transport age', () => {
  const display = new LiveBreathDisplay();
  display.update(rendered(applied(.2, 1, { age_ns: 60000000 })), observation(1, 40), 0);
  display.update(rendered(applied(.4, 2, { age_ns: 60000000 })), observation(2, 40), 200);
  assert.ok(Math.abs(display.frame(300).actualProgress - .32) < 1e-12);
  assert.equal(display.frame(400).actualProgress, .4);
  display.update(rendered(applied(.6, 3, { age_ns: 60000000 })), observation(3, 40), 400);
  assert.ok(Math.abs(display.frame(500).actualProgress - .52) < 1e-12);
  assert.equal(display.frame(600).actualProgress, .6);
});

test('ACTUAL retires the prior endpoint when its own age budget runs out', () => {
  const display = new LiveBreathDisplay();
  display.update(rendered(applied(.2, 1, { age_ns: 300000000 })), observation(1, 50), 0);
  display.update(rendered(applied(.8, 2, { observed_ns: '650000000' })), observation(2), 100);
  assert.ok(Math.abs(display.frame(125).actualProgress - .46) < 1e-12);
  assert.equal(display.frame(150.01).actualProgress, .8, 'the old endpoint expires at150ms, not the new sample TTL');
  assert.equal(display.frame(150.01).actualLive, true);
});

test('early polls do not refill the oldest support age while an ACTUAL blend is unfinished', () => {
  const display = new LiveBreathDisplay();
  display.update(rendered(applied(.2, 1, { age_ns: 300000000 })), observation(1, 50), 0);
  display.update(rendered(applied(.6, 2, { observed_ns: '650000000' })), observation(2), 100);
  display.update(rendered(applied(.8, 3, { observed_ns: '690000000' })), observation(3), 140);
  assert.equal(display.frame(150.01).actualProgress, .6, 'the expired first endpoint cannot leak through a composite blend');
});

test('ACTUAL cadence smoothing spends transport and carrier age rather than extending sample life', () => {
  const display = new LiveBreathDisplay();
  display.update(rendered(applied(.2)), observation(1), 0);
  display.update(rendered(applied(.8, 2, { age_ns: 300000000 })), observation(2, 100), 600);
  assert.equal(display.frame(600).actualProgress, .8, 'an older, still fresh sample resumes without stale interpolation');
  assert.equal(display.frame(700).actualLive, true);
  assert.equal(display.frame(701).actualLive, false);
  display.update(rendered(applied(.99, 2)), observation(3), 750);
  assert.equal(display.frame(750).actualLive, false, 'status replay does not renew GPU sample life');
  assert.equal(display.frame(750).actualProgress, .8);
});

test('ACTUAL does not blend retired source, renderer or settings lineages', () => {
  const display = new LiveBreathDisplay();
  display.update(rendered(applied(.2)), observation(1), 0);
  const replacement = rendered(applied(.9, 2));
  replacement.acceptance = 'new-owner';
  display.update(replacement, observation(2), 200);
  assert.equal(display.frame(200).actualProgress, .9);
  display.update({ ...replacement, renderer: applied(.4, 3, { settings_revision: 1 }) }, observation(3), 400);
  assert.equal(display.frame(400).actualProgress, .4);
});

test('reordered status observations cannot introduce or freshen an ACTUAL sample', () => {
  const display = new LiveBreathDisplay();
  display.update(rendered(applied(.2)), observation(5), 0);
  display.update(rendered(applied(.8, 2)), observation(4), 200);
  assert.equal(display.frame(200).actualProgress, .2);
  assert.equal(display.frame(501).actualLive, false);
  display.update(rendered(applied(.8, 2)), observation(6), 600);
  assert.equal(display.frame(600).actualLive, true);
});

test('advancing renderer frame numbers cannot renew a repeated or reversed native GPU timestamp', () => {
  for (const stamp of ['200000000', '199999999']) {
    const display = new LiveBreathDisplay();
    display.update(rendered(applied(.2, 1)), observation(1), 0);
    display.update(rendered(applied(.8, 2, { observed_ns: stamp })), observation(2), 1000);
    assert.equal(display.frame(1000).actualRadius, null, 'retired source time cannot become a newly fresh GPU endpoint');
  }
});

test('ACTUAL is unavailable on older APK, null renderer, hidden reset or regressing renderer lineage', () => {
  const display = new LiveBreathDisplay();
  display.update(view({}, 0, .99), observation(1), 0);
  assert.equal(display.frame(0).actualRadius, null, 'desired waveform and learned input cannot manufacture ACTUAL');
  display.update(rendered(applied(.3)), observation(2), 100);
  display.update(rendered(applied(.8, 2, { render_generation: 1 })), observation(3), 200);
  assert.equal(display.frame(200).actualRadius, null);
  display.update(rendered(applied(.4, 2)), observation(4), 300);
  assert.equal(display.frame(300).actualProgress, .4);
  display.update(rendered(null), observation(5), 400);
  assert.equal(display.frame(400).actualRadius, null);
  display.update(rendered(applied(.5, 3)), observation(6), 500);
  display.reset();
  display.update(rendered(applied(.5, 3)), observation(6), 51000);
  assert.equal(display.frame(51000), null, 'visibility reset cannot revive the old native observation');
});

test('fresh applied renderer remains independent when the desired audio clock is invalid', () => {
  const display = new LiveBreathDisplay();
  display.update(rendered(applied(.6), { clock_valid: false }), observation(1), 0);
  assert.equal(display.frame(0).desiredAvailable, false);
  assert.equal(display.frame(0).actualLive, true);
  assert.equal(display.frame(500).radius, .46, 'invalid desired clock cannot advance locally');
  assert.equal(display.frame(0).actualProgress, .6);
});

test('transport-expired renderer packets cannot introduce or change an ACTUAL value', () => {
  const display = new LiveBreathDisplay();
  display.update(rendered(applied(.9)), observation(1, 600), 0);
  assert.equal(display.frame(0).actualRadius, null);
  display.update(rendered(applied(.2, 2)), observation(2), 1000);
  assert.equal(display.frame(1000).actualProgress, .2);
  display.update(rendered(applied(.8, 3)), observation(3, 600), 2000);
  assert.equal(display.frame(2000).actualProgress, .2);
  assert.equal(display.frame(2000).actualLive, false);
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
