// Presentation of authenticated renderer snapshots. The audio clock, radius
// profile, phase and admission all stay on the Quest.
const windowMs = 16000;
const freshnessMs = 500;
const maximumGapMs = 320;
const maximumCueOffsetNs = 250000000n;
const maximumRadius = 100;
const volumeUnit = 4 * Math.PI / 3;
const normalized = value => Number.isFinite(value) && value >= 0 && value <= 1;
const nonnegative = value => Number.isSafeInteger(value) && value >= 0;
const validRadius = value => Number.isFinite(value) && value > 0 && value <= maximumRadius;
const timestamp = value => typeof value === 'string' && /^[1-9][0-9]{0,19}$/.test(value);

export const sphereVolume = radius => volumeUnit * radius ** 3;

function validSample(sample) {
  return sample?.v === 1 && validRadius(sample.radius_m)
    && nonnegative(sample.render_generation) && sample.render_generation > 0
    && nonnegative(sample.frame) && nonnegative(sample.settings_revision)
    && nonnegative(sample.age_ns) && sample.age_ns <= freshnessMs * 1000000 && timestamp(sample.observed_ns);
}

function signedRange(sample) {
  const { radius_min_m: low, radius_max_m: high } = sample;
  return validRadius(low) && validRadius(high) && low <= high ? [low, high] : null;
}

function draw(points, edge, low, high) {
  let path = '';
  let prior = false;
  for (const point of points) {
    const x = 400 * (1 - (edge - point.at) / windowMs);
    const y = 66 - 62 * (point.value - low) / (high - low);
    const location = `${x.toFixed(1)} ${y.toFixed(1)}`;
    path += `${path ? ' ' : ''}${prior && !point.gap ? 'L' : 'M'}${location}`;
    if (!prior || point.gap) path += ' h0.01';
    prior = true;
  }
  return path;
}

export class LiveSphereVolume {
  constructor() { this.reset(); }

  reset() {
    this.key = null;
    this.audio = null;
    this.cursor = null;
    this.pausedAt = null;
    this.pauseStamp = null;
    this.clearHistory();
  }

  clearHistory() {
    this.samples = [];
    this.cues = [];
    this.range = null;
    this.clock = null;
    this.interrupt();
  }

  interrupt() {
    this.interrupted = true;
    this.cueInterrupted = true;
  }

  update(observation, now) {
    const key = observation?.key;
    if (!key) { if (this.key) this.reset(); return; }
    if (!Number.isFinite(now)) return;
    if (key !== this.key) {
      this.reset();
      this.key = key;
    }
    const sample = observation.sample;
    const bound = nonnegative(observation.generation) && observation.generation === observation.witnessGeneration;
    if (observation.paused) {
      this.pausedAt ??= Math.max(now, this.samples.at(-1)?.at ?? now);
      if (bound && validSample(sample)) {
        const stamp = BigInt(sample.observed_ns);
        if (this.pauseStamp === null || stamp > this.pauseStamp) this.pauseStamp = stamp;
      }
      this.interrupt();
      return;
    }
    this.pausedAt = null;
    if (observation.audio !== this.audio) {
      this.audio = observation.audio;
      this.interrupt();
    }
    const age = observation.age + (sample?.age_ns ?? Infinity) / 1000000;
    if (!validSample(sample) || !Number.isFinite(age) || age < 0 || age > freshnessMs || !bound
      || !nonnegative(observation.elapsed)
      || sample.age_ns / 1000000 > observation.elapsed) {
      this.interrupt();
      return;
    }
    const stamp = BigInt(sample.observed_ns);
    if (this.pauseStamp !== null && stamp <= this.pauseStamp) return;
    const prior = this.cursor;
    if (prior && (sample.render_generation < prior.generation
      || sample.render_generation === prior.generation && sample.settings_revision < prior.settings)) return;
    const changed = prior && (sample.render_generation !== prior.generation || sample.settings_revision !== prior.settings);
    if (prior && (stamp <= prior.stamp || !changed && sample.frame <= prior.frame)) return;
    // A new revision has its own physical scale. No prior curve crosses it.
    if (changed) this.clearHistory();
    this.cursor = { generation: sample.render_generation, settings: sample.settings_revision, frame: sample.frame, stamp };
    this.clock ??= { stamp, at: now - age };
    let at = this.clock.at + Number(stamp - this.clock.stamp) / 1000000;
    const correction = at - (now - age);
    if (correction > freshnessMs) { this.interrupt(); return; }
    if (correction > 0) {
      // Read jitter can make a first receipt an overly late host-clock anchor.
      // Move the whole source timeline together, retaining every native interval.
      this.clock.at -= correction;
      for (const point of [...this.samples, ...this.cues]) point.at -= correction;
      at -= correction;
    }
    const previous = this.samples.at(-1);
    if (previous && at <= previous.at) { this.interrupt(); return; }
    const range = signedRange(sample);
    if (this.range && range && (range[0] !== this.range[0] || range[1] !== this.range[1])) {
      this.interrupt();
      return;
    }
    this.range ??= range;
    this.samples.push({
      at, expires: now + freshnessMs - age, value: sphereVolume(sample.radius_m),
      gap: this.interrupted || previous && (at - previous.at > maximumGapMs || previous.expires < now),
    });
    this.samples = this.samples.filter(point => point.at >= at - windowMs - freshnessMs).slice(-256);
    this.interrupted = false;
    this.updateCue(observation, range, stamp, at, age, now);
  }

  updateCue(observation, range, renderStamp, renderTime, renderAge, now) {
    const sample = observation.sample;
    if (!range || !normalized(sample.cue_progress01) || !timestamp(sample.cue_observed_ns)
      || !nonnegative(sample.audio_generation) || sample.audio_generation === 0 || sample.audio_generation !== this.audio) {
      this.cueInterrupted = true;
      return;
    }
    const stamp = BigInt(sample.cue_observed_ns);
    const offset = renderStamp - stamp;
    const delay = Number(offset) / 1000000;
    const age = renderAge + delay;
    if (offset < 0n || offset > maximumCueOffsetNs || age > freshnessMs
      || sample.age_ns / 1000000 + delay > observation.elapsed) {
      this.cueInterrupted = true;
      return;
    }
    const previous = this.cues.at(-1);
    const value = sphereVolume(range[0] + (range[1] - range[0]) * sample.cue_progress01);
    if (this.pauseStamp !== null && stamp <= this.pauseStamp || previous && stamp < previous.stamp) {
      this.cueInterrupted = true;
      return;
    }
    if (previous && stamp === previous.stamp) {
      if (value !== previous.value) this.cueInterrupted = true;
      return;
    }
    const at = renderTime - delay;
    this.cues.push({
      stamp, at, expires: now + freshnessMs - age,
      value,
      gap: this.cueInterrupted || previous && (at - previous.at > maximumGapMs || previous.expires < now),
    });
    this.cues = this.cues.filter(point => point.at >= at - windowMs - freshnessMs).slice(-256);
    this.cueInterrupted = false;
  }

  frame(now, reducedMotion = false) {
    const paused = this.pausedAt !== null;
    const latest = this.samples.at(-1);
    const base = paused ? this.pausedAt : now;
    const edge = reducedMotion && latest && !paused ? latest.at : Math.max(base, latest?.at ?? base);
    const points = this.samples.filter(point => edge - point.at <= windowMs);
    const cues = this.cues.filter(point => edge - point.at <= windowMs);
    const latestCue = cues.at(-1);
    const live = Boolean(!paused && !this.interrupted && latest && now <= latest.expires);
    const cueLive = Boolean(live && !this.cueInterrupted && latestCue && now <= latestCue.expires);
    const upper = Math.max(this.range ? sphereVolume(this.range[1]) : 1, ...points.map(point => point.value));
    const lower = Math.min(this.range ? sphereVolume(this.range[0]) : 0, ...points.map(point => point.value));
    const high = lower === upper ? upper * 1.01 : upper;
    return {
      paused, live, ttl: live ? Math.max(0, Math.min(latest.expires, cueLive ? latestCue.expires : Infinity) - now) : null,
      actual: live ? latest.value : null,
      intended: cueLive ? latestCue.value : null,
      actualPath: draw(points, edge, lower, high),
      intendedPath: draw(cues, edge, lower, high),
      bounds: [lower, high], confirmedBounds: this.range !== null,
    };
  }
}
