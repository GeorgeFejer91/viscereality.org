// Presentation only. Command admission and calibration completion stay native.
const labels = ['Einatmen', 'Atem halten', 'Ausatmen', 'Atem halten'];

export class LiveBreathDisplay {
  reset() { this.anchor = null; this.actual = null; }

  update(view, observation, now) {
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
    this.rendererCursor = { key, generation: renderer.render_generation, settings: renderer.settings_revision, frame: renderer.frame };
    const age = Math.max(0, observation.a) + renderer.age_ns / 1000000;
    if (age > 500) {
      if (!same || this.actual?.generation !== renderer.render_generation || this.actual?.settings !== renderer.settings_revision) this.actual = null;
      return;
    }
    const rendererContinuing = same && this.actual?.generation === renderer.render_generation && this.actual.settings === renderer.settings_revision;
    // Follow the spacing of real GPU observations. A fixed short blend finishes
    // before the next poll and produces a step/plateau even on a steady stream.
    // Both contributing endpoints must remain fresh. An unfinished blend also
    // carries its oldest support expiry through an early subsequent poll.
    const old = this.actual;
    const oldExpiry = rendererContinuing ? (now - old.received < old.duration ? old.supportExpires : old.expires) : now;
    const expires = now + 500 - age;
    const duration = rendererContinuing ? Math.max(0, Math.min(now - old.received, oldExpiry - now, expires - now)) : 0;
    const blending = duration > 0;
    this.actual = {
      progress: renderer.progress01, meters: renderer.radius_m, received: now, age,
      generation: renderer.render_generation, settings: renderer.settings_revision,
      from: blending ? prior.actualProgress : renderer.progress01,
      metersFrom: blending ? prior.actualMeters : renderer.radius_m,
      duration: Math.max(1, duration), expires,
      supportExpires: blending ? Math.min(oldExpiry, expires) : expires,
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
    const actualElapsed = actual ? Math.max(0, now - actual.received) : 0;
    const actualBlend = !actual || reducedMotion ? 1 : Math.min(1, Math.min(actualElapsed, Math.max(0, 500 - actual.age)) / actual.duration);
    const actualProgress = actual ? actual.from + (actual.progress - actual.from) * actualBlend : null;
    const actualLive = Boolean(actual && actualElapsed + actual.age <= 500);
    return {
      radius, desiredAvailable: anchor.desiredAvailable, phaseLabel: labels[phase], live: Boolean(followsAudio && elapsed + anchor.age <= 1500),
      ttl: Math.max(0, 1500 - anchor.age - elapsed),
      animating: Boolean(actualLive && actualBlend < 1 || followsAudio && elapsed + anchor.age <= 1500 || blend < 1
        || elapsed + anchor.age <= 1500 && volumeBlend < 1 && anchor.volumeFrom !== anchor.inputVolume),
      inputVolume,
      actualProgress, actualRadius: actualProgress === null ? null : .46 + .54 * actualProgress,
      actualMeters: actual ? actual.metersFrom + (actual.meters - actual.metersFrom) * actualBlend : null,
      actualLive, actualTTL: actual ? Math.max(0, 500 - actual.age - actualElapsed) : 0,
      // Never paint progress beyond a frame count actually reported by Quest.
      progress: anchor.from + (anchor.progress - anchor.from) * blend,
    };
  }
}
