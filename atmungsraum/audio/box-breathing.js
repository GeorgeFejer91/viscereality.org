export const PHASES = Object.freeze(['inhale', 'hold_in', 'exhale', 'hold_out']);
export const MINUTES = Object.freeze([5, 10, 15, 20]);
const BLOCKS = ['instruction', ...PHASES, 'ending', 'rest', 'inhale_breath', 'exhale_breath'];
const DEFAULT_MANIFEST = new URL('../assets/audio/box-breathing/de/soft-v2/manifest.json', import.meta.url);

export function validateManifest(manifest) {
  if (manifest?.id !== 'box-breathing-4-4-4-4-de-v2' || manifest.version !== 2
      || manifest.sample_rate !== 44100 || manifest.block_frames !== 176400
      || manifest.language !== 'de' || manifest.format !== 'pcm_s16le_mono_wav'
      || manifest.remainder_policy !== 'quiet-recovery-after-ending'
      || manifest.cycles_per_block !== 18 || manifest.rest_frames !== 882000
      || manifest.overlay_every_cycles !== 3
      || JSON.stringify(manifest.phase_order) !== JSON.stringify(PHASES)
      || JSON.stringify(manifest.supported_minutes) !== JSON.stringify(MINUTES)) {
    throw new Error('Unsupported box-breathing protocol.');
  }
  for (const id of BLOCKS) {
    const block = manifest.blocks?.[id];
    const phase = PHASES.includes(id) || id.endsWith('_breath');
    const framesValid = Number.isSafeInteger(block?.frames) && block.frames > 0
      && (phase ? block.frames === manifest.block_frames
        : id === 'rest' ? block.frames === manifest.rest_frames
          : block.frames % manifest.block_frames === 0 && block.frames <= 180 * manifest.sample_rate);
    if (block?.file !== `${id}.wav` || !framesValid
        || !/^[a-f0-9]{64}$/.test(block.sha256) || typeof block.text !== 'string') {
      throw new Error(`Invalid audio block: ${id}.`);
    }
  }
  return manifest;
}

// Integer source frames are the timing contract; seconds are derived at playback.
export function createProtocol(minutes, manifest) {
  validateManifest(manifest);
  const instruction = minutes === 'anleitung';
  if (!instruction && !MINUTES.includes(minutes)) throw new RangeError('Choose 5, 10, 15, or 20 minutes.');
  const sampleRate = manifest.sample_rate;
  const blockFrames = manifest.block_frames;
  const introFrames = manifest.blocks.instruction.frames;
  const endingFrames = manifest.blocks.ending.frames;
  const cycleFrames = 4 * blockFrames;
  const totalFrames = instruction ? introFrames + cycleFrames + endingFrames : minutes * 60 * sampleRate;
  let remaining = totalFrames - introFrames - endingFrames;
  if (remaining < cycleFrames) throw new RangeError('No complete breathing cycle fits this session.');
  let cycles = 0, rests = 0, block = 1;
  const breathingBlocks = [];
  const events = [];
  let frame = 0;
  const add = (id, frames, cycle = null, asset = id) => {
    events.push(Object.freeze({ id, asset, offset_frames: frame, duration_frames: frames, cycle,
      breathing_block: cycle === null ? null : block }));
    frame += frames;
  };
  add('instruction', introFrames);
  while (remaining >= cycleFrames) {
    const count = Math.min(manifest.cycles_per_block, Math.floor(remaining / cycleFrames));
    breathingBlocks.push(count);
    for (let i = 0; i < count; i++) {
      cycles++;
      for (const id of PHASES) {
        const overlay = cycles % manifest.overlay_every_cycles === 0 && ['inhale', 'exhale'].includes(id);
        add(id, blockFrames, cycles, overlay ? `${id}_breath` : id);
      }
    }
    remaining -= count * cycleFrames;
    // Rest only between breathing blocks; never turn the final remainder into a hold.
    if (remaining >= manifest.rest_frames + cycleFrames) {
      add('rest', manifest.rest_frames); rests++; block++; remaining -= manifest.rest_frames;
    } else break;
  }
  add('ending', endingFrames);
  const recoveryFrames = totalFrames - frame;
  if (recoveryFrames) add('recovery', recoveryFrames);
  return Object.freeze({ id: manifest.id, minutes: instruction ? null : minutes,
    mode: instruction ? 'instruction-example' : 'session', sample_rate: sampleRate,
    total_frames: totalFrames, cycles, rests, breathing_blocks: Object.freeze(breathingBlocks),
    intro_frames: introFrames, ending_frames: endingFrames, recovery_frames: recoveryFrames,
    events: Object.freeze(events) });
}

// The same scheduler supports realtime playback and OfflineAudioContext tests.
// All nodes are queued before t0. Neither JS timers nor ended callbacks start cues.
export function scheduleProtocol(context, buffers, protocol, startAt, destination = context.destination) {
  if (!Number.isFinite(startAt) || startAt < context.currentTime) {
    throw new RangeError('Audio start time is already past or invalid.');
  }
  const sources = [];
  try {
    for (const event of protocol.events) {
      const seconds = event.duration_frames / protocol.sample_rate;
      const buffer = event.id === 'recovery'
        ? context.createBuffer(1, Math.round(seconds * context.sampleRate), context.sampleRate)
        : buffers[event.asset ?? event.id];
      if (!buffer || buffer.numberOfChannels !== 1
          || Math.abs(buffer.duration - seconds) > 1 / context.sampleRate) {
        throw new Error(`Audio duration/channel mismatch: ${event.id}.`);
      }
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(destination);
      sources.push({ source, event });
    }
    // Validate all blocks before any source can sound.
    for (const { source, event } of sources) {
      const when = startAt + event.offset_frames / protocol.sample_rate;
      source.start(when, 0, event.duration_frames / protocol.sample_rate);
      source.stop(when + event.duration_frames / protocol.sample_rate);
    }
    return sources;
  } catch (error) {
    for (const { source } of sources) {
      try { source.stop(); } catch { /* A node may not have started. */ }
      source.disconnect();
    }
    throw error;
  }
}

async function sha256(bytes) {
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(hash)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

export class BoxBreathingEngine {
  constructor({ audioContext, manifestUrl = DEFAULT_MANIFEST } = {}) {
    this.ownsContext = !audioContext;
    this.context = audioContext ?? new AudioContext({ latencyHint: 'interactive' });
    this.manifestUrl = new URL(manifestUrl, DEFAULT_MANIFEST);
    this.run = null;
    this.token = 0;
    this.disposed = false;
    this.pausing = false;
    this.resuming = false;
    this.onStateChange = () => {
      const status = this.run?.status;
      const unexpectedPause = status === 'scheduled' && this.context.state !== 'running' && !this.pausing;
      const unexpectedResume = status === 'paused' && this.context.state !== 'suspended'
        && !(this.context.state === 'running' && this.resuming);
      if (unexpectedPause || unexpectedResume) {
        this.abort(`audio-context-${this.context.state}`);
      }
    };
    this.onVisibilityChange = () => {
      if (document.hidden) this.abort('page-hidden');
    };
    this.context.addEventListener('statechange', this.onStateChange);
    if (typeof document !== 'undefined') document.addEventListener('visibilitychange', this.onVisibilityChange);
  }

  async prepare() {
    if (this.disposed) throw new Error('Engine is disposed.');
    if (this.preparation) return this.preparation;
    this.preparation = (async () => {
      const response = await fetch(this.manifestUrl);
      if (!response.ok) throw new Error(`Manifest HTTP ${response.status}.`);
      const bytes = await response.arrayBuffer();
      this.manifest = validateManifest(JSON.parse(new TextDecoder().decode(bytes)));
      this.manifestHash = await sha256(bytes);
      const entries = await Promise.all(BLOCKS.map(async (id) => {
        const block = this.manifest.blocks[id];
        const audio = await fetch(new URL(block.file, this.manifestUrl));
        if (!audio.ok) throw new Error(`Audio HTTP ${audio.status}: ${id}.`);
        const data = await audio.arrayBuffer();
        if (data.byteLength !== block.bytes || await sha256(data) !== block.sha256) {
          throw new Error(`Audio integrity mismatch: ${id}.`);
        }
        const buffer = await this.context.decodeAudioData(data);
        if (buffer.numberOfChannels !== 1 || Math.abs(buffer.duration - block.frames / this.manifest.sample_rate) > 1 / this.context.sampleRate) {
          throw new Error(`Audio block has the wrong channel count or duration: ${id}.`);
        }
        return [id, buffer];
      }));
      this.buffers = Object.fromEntries(entries);
      return this.manifest;
    })();
    try { return await this.preparation; }
    catch (error) { this.preparation = null; this.buffers = null; throw error; }
  }

  // Call after prepare(), directly from the host's Start action. An explicit
  // startAt lets the rest of an experiment share this audio-clock anchor.
  async start(minutes, { startAt } = {}) {
    if (this.disposed || !this.buffers) throw new Error('Prepare audio before starting.');
    if (this.starting || this.pausing || this.resuming || ['scheduled', 'paused'].includes(this.run?.status)) throw new Error('A session is already active.');
    if (typeof document !== 'undefined' && document.hidden) throw new Error('Keep the experiment page visible.');
    const protocol = createProtocol(minutes, this.manifest);
    const token = ++this.token;
    this.starting = true;
    try {
      await this.context.resume();
      if (token !== this.token || this.disposed) throw new Error('Start was cancelled.');
      if (this.context.state !== 'running') throw new Error('Audio context did not start.');
      const t0 = startAt ?? this.context.currentTime + 0.25;
      if (!Number.isFinite(t0) || t0 < this.context.currentTime + 0.05) {
        throw new RangeError('Start requires at least 50 ms of scheduling lead time.');
      }
      let resolve;
      const done = new Promise((finish) => { resolve = finish; });
      const run = { status: 'scheduled', protocol, start_at_audio_seconds: t0,
        end_at_audio_seconds: t0 + protocol.total_frames / protocol.sample_rate,
        manifest_sha256: this.manifestHash, user_agent: globalThis.navigator?.userAgent ?? null,
        audio_sample_rate: this.context.sampleRate, base_latency_seconds: this.context.baseLatency ?? null,
        output_latency_seconds: this.context.outputLatency ?? null,
        clock_mapping: this.context.getOutputTimestamp?.() ?? null,
        requested_at_performance_ms: performance.now(), records: [], sources: [], done, resolve };
      this.run = run;
      try {
        run.sources = scheduleProtocol(this.context, this.buffers, protocol, t0);
        for (const { source, event } of run.sources) {
          const when = t0 + event.offset_frames / protocol.sample_rate;
          run.records.push({ type: 'scheduled', ...event, audio_time_seconds: when,
            expected_end_audio_seconds: when + event.duration_frames / protocol.sample_rate });
          source.onended = () => {
            source.disconnect();
            if (this.run !== run || !['scheduled', 'paused'].includes(run.status)) return;
            run.records.push({ type: 'ended-notification', id: event.id,
              offset_frames: event.offset_frames, observed_audio_seconds: this.context.currentTime,
              observed_performance_ms: performance.now() });
            if (event === protocol.events.at(-1)) {
              // A delayed main-thread notification never changes the audio plan.
              if (this.context.currentTime + 1 / this.context.sampleRate < run.end_at_audio_seconds) {
                this.abort('premature-end-notification');
              } else {
                run.status = 'completed';
                run.resolve(this.receipt());
              }
            }
          };
        }
        if (this.context.currentTime >= t0) throw new Error('Scheduling missed the start deadline.');
        return { protocol, done, startAt: t0, endAt: run.end_at_audio_seconds };
      } catch (error) { this.abort('scheduling-failed'); throw error; }
    } finally { this.starting = false; }
  }

  position() {
    const run = this.run;
    if (!run) return null;
    const elapsed = this.context.currentTime - run.start_at_audio_seconds;
    const frame = Math.floor(elapsed * run.protocol.sample_rate);
    const event = run.protocol.events.find((item) => frame >= item.offset_frames
      && frame < item.offset_frames + item.duration_frames);
    return { status: run.status, elapsed_seconds: Math.max(0, elapsed), event: event ?? null };
  }

  async pause() {
    const run = this.run;
    if (this.pausing || this.resuming || run?.status !== 'scheduled') return false;
    const token = ++this.token;
    this.pausing = true;
    try {
      await this.context.suspend();
      if (token !== this.token || this.run !== run || run.status !== 'scheduled') return false;
      if (this.context.state !== 'suspended') throw new Error('Audio context did not pause.');
      run.status = 'paused';
      run.records.push({ type: 'pause', observed_audio_seconds: this.context.currentTime,
        observed_performance_ms: performance.now() });
      return true;
    } catch (error) {
      if (this.run === run) this.abort('pause-failed');
      throw error;
    } finally { this.pausing = false; }
  }

  async resume() {
    const run = this.run;
    if (this.pausing || this.resuming || run?.status !== 'paused') return false;
    const token = ++this.token;
    this.resuming = true;
    try {
      await this.context.resume();
      if (token !== this.token || this.run !== run || run.status !== 'paused') return false;
      if (this.context.state !== 'running') throw new Error('Audio context did not resume.');
      run.status = 'scheduled';
      run.records.push({ type: 'resume', observed_audio_seconds: this.context.currentTime,
        observed_performance_ms: performance.now() });
      return true;
    } catch (error) {
      if (this.run === run) this.abort('resume-failed');
      throw error;
    } finally { this.resuming = false; }
  }

  receipt() {
    if (!this.run) return null;
    const { sources, done, resolve, ...receipt } = this.run;
    return structuredClone(receipt);
  }

  abort(reason = 'user-abort') {
    ++this.token; // Also invalidate a start waiting for an autoplay unlock.
    const run = this.run;
    if (!run || !['scheduled', 'paused'].includes(run.status)) return;
    run.status = 'aborted';
    run.records.push({ type: 'abort', reason, observed_audio_seconds: this.context.currentTime,
      observed_performance_ms: performance.now() });
    for (const { source } of run.sources) {
      source.onended = null;
      try { source.stop(); } catch { /* Already ended sources are harmless. */ }
      source.disconnect();
    }
    run.resolve(this.receipt());
  }

  async dispose() {
    this.disposed = true;
    this.abort('engine-disposed');
    this.context.removeEventListener('statechange', this.onStateChange);
    if (typeof document !== 'undefined') document.removeEventListener('visibilitychange', this.onVisibilityChange);
    if (this.ownsContext && this.context.state !== 'closed') await this.context.close();
  }
}
