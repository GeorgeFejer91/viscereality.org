// Presentation only. Command admission and calibration completion stay native.
const labels = ['Einatmen', 'Atem halten', 'Ausatmen', 'Atem halten'];

export class LiveBreathDisplay {
  reset() { this.anchor = null; }

  update(view, observation, now) {
    const setup = view?.setup;
    if (!view?.available || !setup || !Number.isFinite(now)
      || !Number.isSafeInteger(setup.cycle_frames) || setup.cycle_frames <= 0
      || typeof observation?.i !== 'string' || !Number.isFinite(observation.a)) {
      this.reset();
      return;
    }
    const key = `${view.acceptance}/${setup.g}/${setup.attempt}/${setup.audio_generation}`;
    if (this.lastObservation?.key === key && this.lastObservation.i === observation.i) return;
    this.lastObservation = { key, i: observation.i };
    const prior = this.frame(now);
    const progress = Math.max(0, Math.min(2, view.progress.value));
    const continuing = this.anchor?.key === key;
    this.anchor = {
      key, observation: observation.i, setup: { ...setup }, received: now,
      age: Math.max(0, observation.a), progress,
      inputVolume: typeof view.inputVolume === 'number' ? view.inputVolume : null,
      volumeFrom: continuing && typeof prior?.inputVolume === 'number' ? prior.inputVolume : view.inputVolume,
      from: continuing ? Math.min(prior.progress, progress) : progress,
      duration: continuing ? Math.max(1, Math.min(1000, now - this.anchor.received)) : 1,
    };
  }

  frame(now, reducedMotion = false) {
    const anchor = this.anchor;
    if (!anchor) return null;
    const { setup } = anchor;
    const elapsed = Math.max(0, now - anchor.received);
    const followsAudio = setup.playing && setup.audio_state === 'playing'
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
    return {
      radius, phaseLabel: labels[phase], live: Boolean(followsAudio && elapsed + anchor.age <= 1500),
      ttl: Math.max(0, 1500 - anchor.age - elapsed),
      animating: Boolean(followsAudio && elapsed + anchor.age <= 1500 || blend < 1
        || elapsed + anchor.age <= 1500 && volumeBlend < 1 && anchor.volumeFrom !== anchor.inputVolume),
      inputVolume,
      // Never paint progress beyond a frame count actually reported by Quest.
      progress: anchor.from + (anchor.progress - anchor.from) * blend,
    };
  }
}
