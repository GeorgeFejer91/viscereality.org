// Presentation only. Command admission and calibration completion stay native.
const labels = ['Einatmen', 'Atem halten', 'Ausatmen', 'Atem halten'];
// Bounded source-time delay absorbs polling jitter without extending freshness.
const actualDelayMs = 280;

export class LiveBreathDisplay {
  reset() { this.anchor = null; this.actual = null; }

  update(view, observation, now, sessionGeneration = null) {
    const newSession = sessionGeneration !== null && this.sessionGeneration !== sessionGeneration;
    if (this.sessionGeneration !== sessionGeneration) this.actual = null;
    this.sessionGeneration = sessionGeneration;
    const setup = view?.setup;
    if (!view?.available || !setup || !Number.isFinite(now)
      || !Number.isSafeInteger(setup.cycle_frames) || setup.cycle_frames <= 0
      || typeof observation?.i !== 'string' || !Number.isFinite(observation.a)) {
      this.reset();
      return;
    }
    const key = `${view.acceptance}/${setup.g}/${setup.attempt}/${setup.audio_generation}`;
    if (this.lastObservation?.key === key && (this.lastObservation.i === observation.i
      || /^[0-9]+$/.test(this.lastObservation.i) && /^[0-9]+$/.test(observation.i)
        && BigInt(observation.i) < BigInt(this.lastObservation.i))) return;
    this.lastObservation = { key, i: observation.i };
    const prior = this.frame(now);
    const progress = Math.max(0, Math.min(2, view.progress.value));
    const continuing = this.anchor?.key === key;
    this.anchor = {
      key, observation: observation.i, setup: { ...setup }, received: now,
      desiredAvailable: !view.recovering && setup.clock_valid !== false,
      age: Math.max(0, observation.a), progress,
      inputVolume: typeof view.inputVolume === 'number' ? view.inputVolume : null,
      volumeFrom: continuing && typeof prior?.inputVolume === 'number' ? prior.inputVolume : view.inputVolume,
      from: continuing ? Math.min(prior.progress, progress) : progress,
      duration: continuing ? Math.max(1, Math.min(1000, now - this.anchor.received)) : 1,
    };
    const renderer = view.renderer;
    if (!renderer) { this.actual = null; return; }
    const stamp = BigInt(renderer.observed_ns);
    const previous = this.rendererCursor;
    const same = previous?.key === key;
    if (same && (renderer.render_generation < previous.generation
      || renderer.render_generation === previous.generation && (renderer.settings_revision < previous.settings
        || renderer.settings_revision === previous.settings && renderer.frame < previous.frame))) {
      this.actual = null;
      return;
    }
    // A new status witness cannot make an old renderer frame fresh again.
    if (same && renderer.render_generation === previous.generation && renderer.settings_revision === previous.settings
      && renderer.frame === previous.frame) return;
    if (same && renderer.render_generation === previous.generation && renderer.settings_revision === previous.settings
      && stamp <= previous.stamp) { this.actual = null; return; }
    this.rendererCursor = {
      key, generation: renderer.render_generation, settings: renderer.settings_revision, frame: renderer.frame, stamp,
    };
    if (newSession) return;
    const age = Math.max(0, observation.a) + renderer.age_ns / 1000000;
    if (age > 500) {
      if (!same || this.actual?.generation !== renderer.render_generation || this.actual?.settings !== renderer.settings_revision) this.actual = null;
      return;
    }
    const rendererContinuing = same && this.actual?.generation === renderer.render_generation && this.actual.settings === renderer.settings_revision
      && prior.actualLive;
    const old = this.actual;
    const origin = rendererContinuing ? old.origin : stamp;
    const originTime = rendererContinuing ? old.originTime : now - age;
    const time = originTime + Number(stamp - origin) / 1000000;
    if (rendererContinuing && (time <= old.samples.at(-1).time || time > now)) { this.actual = null; return; }
    // Keep adjacent observations, never a recursively blended endpoint. HTTP
    // jitter may append a new point but cannot restart the ongoing playout.
    // Every point retains its original full-age expiry; neither buffering nor
    // a later witness can renew it. Native timestamps preserve source cadence.
    const samples = rendererContinuing ? old.samples.filter(sample => sample.expires >= now) : [];
    samples.push({ time, progress: renderer.progress01, meters: renderer.radius_m, expires: now + 500 - age });
    this.actual = {
      generation: renderer.render_generation, settings: renderer.settings_revision,
      origin, originTime, samples: samples.slice(-32),
    };
  }

  frame(now, reducedMotion = false) {
    const anchor = this.anchor;
    if (!anchor) return null;
    const { setup } = anchor;
    const elapsed = Math.max(0, now - anchor.received);
    const followsAudio = anchor.desiredAvailable && setup.playing && setup.audio_state === 'playing'
      && Number.isSafeInteger(setup.audio_generation) && setup.audio_generation > 0;
    // The accepted native schedule is four phases of four seconds. This clock
    // only paints that observed loop; it cannot train, finish or switch inputs.
    const freshElapsed = Math.min(elapsed, Math.max(0, 1500 - anchor.age));
    const advance = followsAudio && !reducedMotion ? freshElapsed : 0;
    const position = setup.position_frames + advance * setup.cycle_frames / 16000;
    const phaseFrames = setup.cycle_frames / 4;
    const phase = Math.floor(position / phaseFrames) % 4;
    const fraction = position % phaseFrames / phaseFrames;
    const radius = [.46 + .54 * fraction, 1, 1 - .54 * fraction, .46][phase];
    const blend = reducedMotion ? 1 : Math.min(1, elapsed / anchor.duration);
    const volumeBlend = reducedMotion ? 1 : Math.min(1, freshElapsed / anchor.duration);
    const inputVolume = setup.input === 'automatic' ? [fraction, 1, 1 - fraction, 0][phase]
      : anchor.inputVolume === null ? null
        : typeof anchor.volumeFrom !== 'number' ? anchor.inputVolume
          : anchor.volumeFrom + (anchor.inputVolume - anchor.volumeFrom) * volumeBlend;
    const actual = this.actual;
    const fresh = actual?.samples.filter(sample => sample.expires >= now) ?? [];
    const target = now - actualDelayMs;
    let left = fresh[0] ?? actual?.samples.at(-1), right = left;
    for (const sample of fresh) {
      if (sample.time <= target) left = right = sample;
      else { right = sample; break; }
    }
    if (reducedMotion) left = right = actual?.samples.at(-1);
    const actualBlend = left && right !== left ? Math.max(0, Math.min(1, (target - left.time) / (right.time - left.time))) : 0;
    const actualProgress = left ? left.progress + (right.progress - left.progress) * actualBlend : null;
    const actualLive = Boolean(left && now <= left.expires && now <= right.expires);
    const actualTTL = actualLive ? Math.max(0, Math.min(left.expires, right.expires) - now) : 0;
    const actualAnimating = actualLive && !reducedMotion && fresh.length > 1 && target < fresh.at(-1).time;
    return {
      radius, desiredAvailable: anchor.desiredAvailable, phaseLabel: labels[phase], live: Boolean(followsAudio && elapsed + anchor.age <= 1500),
      ttl: Math.max(0, 1500 - anchor.age - elapsed),
      animating: Boolean(actualAnimating || followsAudio && elapsed + anchor.age <= 1500 || blend < 1
        || elapsed + anchor.age <= 1500 && volumeBlend < 1 && anchor.volumeFrom !== anchor.inputVolume),
      inputVolume,
      actualProgress, actualRadius: actualProgress === null ? null : .46 + .54 * actualProgress,
      actualMeters: left ? left.meters + (right.meters - left.meters) * actualBlend : null,
      actualLive, actualTTL, actualAnimating,
      // Never paint progress beyond a frame count actually reported by Quest.
      progress: anchor.from + (anchor.progress - anchor.from) * blend,
    };
  }
}
