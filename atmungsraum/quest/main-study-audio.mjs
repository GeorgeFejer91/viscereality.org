import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { once } from 'node:events';
import { mkdtemp, readFile, rmdir, unlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, posix } from 'node:path';
import { fileURLToPath } from 'node:url';

export const STUDY_DIRECTORY = 'assets/audio/main-study/de';
const projectRoot = fileURLToPath(new URL('../', import.meta.url));
const RATE = 44_100;
const PHASE_FRAMES = 4 * RATE;
const TOTAL_FRAMES = 900 * RATE;
const OPENING_FRAMES = 300 * RATE;
const GUIDED_FRAMES = 600 * RATE;
const SOFT_LOOP_DIRECTORY = 'assets/audio/box-breathing/de/soft-v2';

// Source milliseconds for the eyes-closed and eyes-open prompt boundaries.
const LISTENING_BREAKS = [
  { after: [14100, 12140], resumeAt: 28000 }, // Settle and listen.
  { after: [25580, 25320], resumeAt: 57000 }, // Notice the inhale and exhale.
  { after: [44950, 49850], resumeAt: 93000 }, // Return attention to breathing.
];
const PHASES = [
  ['inhale', 'zen-inhale'], ['hold_in', 'zen-hold'], ['exhale', 'zen-exhale'], ['hold_out', 'zen-hold'],
];
const CANONICAL_CUES = { 'zen-inhale': 'inhale_breath', 'zen-hold': 'hold_in', 'zen-exhale': 'exhale_breath' };
const INSTRUCTIONS = ['instruction_1', 'instruction_2', 'instruction_3'];
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const invalid = detail => { throw new Error('Main study audio: ' + detail); };

export async function loadStudyInventory(sourceRoot = projectRoot) {
  const dir = join(sourceRoot, STUDY_DIRECTORY);
  const inventory = JSON.parse(await readFile(join(dir, 'manifest.json'), 'utf8'));
  if (inventory.schema_version !== 1 || inventory.scenario_id !== 'haupt-15'
      || inventory.language !== 'de' || !Array.isArray(inventory.assets)
      || !Array.isArray(inventory.opening) || inventory.opening.length !== INSTRUCTIONS.length) {
    invalid('invalid inventory');
  }
  const byId = new Map();
  const sources = new Set();
  for (const asset of inventory.assets) {
    const { id, text, file, transcript_file: transcript, role } = asset;
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id) || byId.has(id)
        || file !== id + '.mp3' || transcript !== id + '.txt'
        || typeof text !== 'string' || (role === 'background' ? text !== '' : !text)
        || !Array.isArray(asset.sources) || asset.sources.length === 0) {
      invalid('invalid asset identity: ' + id);
    }
    for (const name of asset.sources) {
      if (typeof name !== 'string' || sources.has(name)) invalid('duplicate source: ' + name);
      sources.add(name);
    }
    const [mp3, sidecar] = await Promise.all([readFile(join(dir, file)), readFile(join(dir, transcript), 'utf8')]);
    if (mp3.length !== asset.bytes || sha256(mp3) !== asset.sha256 || sidecar !== (text ? text + '\n' : '')) {
      invalid('asset or transcript differs: ' + id);
    }
    byId.set(id, asset);
  }
  inventory.opening.forEach((id, i) => {
    const asset = byId.get(id);
    if (asset?.role !== INSTRUCTIONS[i] || !Number.isSafeInteger(asset.frames_44100) || asset.frames_44100 < 1) {
      invalid('invalid instruction: ' + id);
    }
  });
  for (const phase of ['inhale', 'exhale', 'hold']) {
    const ids = inventory.pools?.[phase];
    const eligible = inventory.assets.filter(asset => asset.role === phase);
    if (!Array.isArray(ids) || ids.length < 2 || new Set(ids).size !== ids.length
        || ids.length !== eligible.length || ids.some(id => byId.get(id)?.role !== phase)) {
      invalid('invalid ' + phase + ' pool');
    }
    for (const id of ids) {
      const cue = byId.get(id);
      if (!cue.phase_text || (phase !== 'hold' && (!Number.isFinite(cue.phase_source_end_seconds)
          || cue.phase_source_end_seconds <= 0 || cue.phase_source_end_seconds > cue.duration_seconds))) {
        invalid('invalid phase edit: ' + id);
      }
    }
  }
  const probe = byId.get(inventory.probe_instruction);
  if (probe?.role !== 'probe_instruction' || !Number.isSafeInteger(probe.frames_44100) || probe.frames_44100 < 1) {
    invalid('invalid probe instruction');
  }
  if (byId.get(inventory.background)?.role !== 'background') invalid('missing background music');
  return inventory;
}

export function planMainStudy(inventory) {
  const assets = new Map(inventory.assets.map(asset => [asset.id, asset]));
  const events = [];
  let offset = 0;
  const add = (id, asset, frames, cycle = null) => {
    events.push({ id, asset, offset_frames: offset, duration_frames: frames, cycle });
    offset += frames;
  };
  const controls = inventory.opening.slice(0, 2).map(id => assets.get(id));
  const instruction = assets.get(inventory.opening[2]);
  const controlFrames = (OPENING_FRAMES - instruction.frames_44100) / 2;
  if (!Number.isSafeInteger(controlFrames) || controls.some(asset => asset.frames_44100 >= controlFrames)) {
    invalid('opening cannot fit two equal listening controls');
  }
  for (const asset of controls) add(asset.role, asset.id, controlFrames);
  add(instruction.role, instruction.id, instruction.frames_44100);

  const cycles = Math.floor(GUIDED_FRAMES / (PHASES.length * PHASE_FRAMES));
  const addPhase = ([phase, asset], cycle) => add(phase, asset, PHASE_FRAMES, cycle + 1);
  for (let cycle = 0; cycle < cycles; cycle++) {
    for (const phase of PHASES) addPhase(phase, cycle);
  }
  addPhase(PHASES[0], cycles);
  addPhase(PHASES[2], cycles);
  if (offset !== TOTAL_FRAMES) invalid('timed programme does not end at 15 minutes');
  return { scenario_id: 'haupt-15', sample_rate: RATE, total_frames: TOTAL_FRAMES, cycles, events };
}

function decodePcm(input, args = []) {
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn('ffmpeg', ['-nostdin', '-v', 'error', '-i', input,
      ...args, '-vn', '-ac', '1', '-ar', String(RATE), '-f', 's16le', '-'],
    { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    const chunks = [];
    let error = '';
    ffmpeg.stdout.on('data', chunk => chunks.push(chunk));
    ffmpeg.stderr.on('data', chunk => { error += chunk; });
    ffmpeg.once('error', reject);
    ffmpeg.once('close', code => code === 0 ? resolve(Buffer.concat(chunks))
      : reject(new Error('FFmpeg could not decode ' + input + ': ' + error.trim())));
  });
}

function extendControl(samples, control, frames) {
  const chunks = [];
  let lastFrame = 0;
  let addedFrames = 0;
  for (const { after, resumeAt } of LISTENING_BREAKS) {
    const cut = RATE * after[control] / 1000;
    const silence = RATE * resumeAt / 1000 - cut - addedFrames;
    const margin = RATE / 50;
    if (cut <= lastFrame || cut + margin >= samples.length / 2 || silence <= 0) invalid('invalid listening break');
    for (let frame = cut - margin; frame < cut + margin; frame++) {
      if (Math.abs(samples.readInt16LE(frame * 2)) > 128) invalid('listening break overlaps source speech');
    }
    chunks.push(samples.subarray(lastFrame * 2, cut * 2), Buffer.alloc(silence * 2));
    addedFrames += silence;
    lastFrame = cut;
  }
  const tailFrames = frames - samples.length / 2 - addedFrames;
  if (tailFrames <= 0) invalid('no silence remains after the final listening prompt');
  chunks.push(samples.subarray(lastFrame * 2), Buffer.alloc(tailFrames * 2));
  return Buffer.concat(chunks);
}

function wavePcm(wave, frames) {
  if (wave.length !== frames * 2 + 44 || wave.toString('ascii', 36, 40) !== 'data'
      || wave.readUInt32LE(24) !== RATE || wave.readUInt16LE(22) !== 1 || wave.readUInt16LE(34) !== 16) {
    invalid('unexpected mono PCM WAV');
  }
  return wave.subarray(44);
}

function encodeWav(pcm) {
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(pcm.length + 36, 4);
  header.write('WAVEfmt ', 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(RATE, 24);
  header.writeUInt32LE(RATE * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36);
  header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}

function stagedCueFilename(index, event) {
  const seconds = Math.floor(event.offset_frames / RATE);
  const minutes = String(Math.floor(seconds / 60)).padStart(2, '0');
  const remainder = String(seconds % 60).padStart(2, '0');
  return String(index + 1).padStart(4, '0') + '_' + minutes + 'm' + remainder + 's_' + event.asset + '.wav';
}

export async function buildMainStudyScenario(sourceRoot, original) {
  const inventory = await loadStudyInventory(sourceRoot);
  const timeline = planMainStudy(inventory);
  const assets = new Map(inventory.assets.map(asset => [asset.id, asset]));
  const registry = {};
  const files = new Map();
  const prefix = 'assets/audio/scenarios/haupt-15';
  const library = JSON.parse(await readFile(join(sourceRoot, SOFT_LOOP_DIRECTORY, 'manifest.json'), 'utf8'));
  for (const [index, event] of timeline.events.entries()) {
    if (event.asset === null || Object.hasOwn(registry, event.asset)) continue;
    const canonicalId = CANONICAL_CUES[event.asset];
    const cue = canonicalId ? library.blocks[canonicalId] : assets.get(event.asset);
    let wave;
    if (canonicalId) {
      wave = await readFile(join(sourceRoot, SOFT_LOOP_DIRECTORY, cue.file));
      wavePcm(wave, event.duration_frames);
      if (wave.length !== cue.bytes || sha256(wave) !== cue.sha256) {
        invalid('canonical phase integrity mismatch: ' + canonicalId);
      }
    } else {
      let pcm = await decodePcm(join(sourceRoot, STUDY_DIRECTORY, cue.file));
      if (index < 2) {
        if (pcm.length !== cue.frames_44100 * 2) invalid('unexpected control length: ' + cue.id);
        pcm = extendControl(pcm, index, event.duration_frames);
      }
      if (pcm.length !== event.duration_frames * 2 || !pcm.some(byte => byte !== 0)) invalid('unexpected PCM: ' + cue.id);
      wave = encodeWav(pcm);
    }
    const file = stagedCueFilename(index, event);
    registry[event.asset] = {
      id: event.asset, file, frames: event.duration_frames, bytes: wave.length, sha256: sha256(wave),
      placeholder: false, text: cue.text, source_sha256: cue.sha256,
    };
    files.set(posix.join(prefix, file), wave);
  }
  const outro = JSON.parse(await readFile(join(sourceRoot, SOFT_LOOP_DIRECTORY, 'outro.json'), 'utf8'));
  if (outro.file !== 'outro.wav' || outro.sample_rate !== RATE || !Number.isSafeInteger(outro.frames)
      || outro.frames <= 0 || typeof outro.text !== 'string') invalid('invalid post-session outro metadata');
  const outroPath = posix.join(SOFT_LOOP_DIRECTORY, outro.file);
  const outroWav = await readFile(join(sourceRoot, outroPath));
  const outroText = await readFile(join(sourceRoot, SOFT_LOOP_DIRECTORY, 'outro.txt'), 'utf8');
  if (outroWav.length !== outro.bytes || sha256(outroWav) !== outro.sha256 || outroText !== outro.text + '\n') {
    invalid('post-session outro integrity mismatch');
  }
  wavePcm(outroWav, outro.frames);
  const closing = { id: 'outro', asset: 'outro', offset_frames: TOTAL_FRAMES, duration_frames: outro.frames, cycle: null };
  const outroFile = stagedCueFilename(timeline.events.length, closing);
  registry.outro = {
    id: 'outro', file: outroFile, frames: outro.frames, bytes: outro.bytes,
    sha256: outro.sha256, placeholder: false, text: outro.text,
  };
  files.set(posix.join(prefix, outroFile), outroWav);
  const plan = {
    schema_version: original.schema_version,
    library_manifest_sha256: original.library_manifest_sha256,
    audio_assets_base_directory: original.audio_assets_base_directory,
    ...timeline,
    full_round: true,
    total_frames: TOTAL_FRAMES + outro.frames,
    events: [...timeline.events, closing],
    audio_assets: registry,
  };
  return { plan, files };
}

export async function renderMainStudyTrack({ plan, files }, output, { codec = 'pcm_s16le' } = {}) {
  const ffmpeg = spawn('ffmpeg', ['-nostdin', '-v', 'error', '-n', '-f', 's16le', '-ar', String(plan.sample_rate),
    '-ac', '1', '-i', 'pipe:0', '-map_metadata', '-1',
    '-ac', '1', '-sample_fmt', 's16', '-c:a', codec, output], { stdio: ['pipe', 'ignore', 'inherit'], windowsHide: true });
  const result = once(ffmpeg, 'close');
  ffmpeg.stdin.on('error', error => { if (error.code !== 'EPIPE') ffmpeg.kill(); });
  for (const event of plan.events) {
    if (ffmpeg.stdin.destroyed) break;
    const asset = event.asset === null ? null : plan.audio_assets[event.asset];
    const pcm = asset ? files.get('assets/audio/scenarios/haupt-15/' + asset.file).subarray(44)
      : Buffer.alloc(event.duration_frames * 2);
    if (!ffmpeg.stdin.write(pcm)) await Promise.race([once(ffmpeg.stdin, 'drain'), result]);
  }
  ffmpeg.stdin.end();
  const [exitCode] = await result;
  if (exitCode !== 0) invalid('study mix render failed');
}

export async function stageMainStudyTrack(study) {
  const dir = await mkdtemp(join(tmpdir(), 'atmungsraum-main-audio-'));
  const rendered = join(dir, 'haupt-15.flac');
  try {
    await renderMainStudyTrack(study, rendered, { codec: 'flac' });
    const bytes = await readFile(rendered);
    const path = 'assets/audio/scenarios/haupt-15/haupt-15.flac';
    if (bytes.length < 42 || bytes.length > 32 * 1024 * 1024 || bytes.toString('ascii', 0, 4) !== 'fLaC'
        || (bytes.readUInt32BE(4) & 0x7fffffff) !== 34) invalid('invalid timed FLAC header');
    const stream = bytes.readBigUInt64BE(18);
    if (Number(stream >> 44n) !== study.plan.sample_rate
        || Number((stream >> 36n) & 255n) !== 15
        || Number(stream & ((1n << 36n) - 1n)) !== study.plan.total_frames) {
      invalid('timed FLAC does not match the recording plan');
    }
    study.files.set(path, bytes);
    study.plan.timed_audio = { path, bytes: bytes.length, sha256: sha256(bytes), duration_frames: study.plan.total_frames };
  } finally {
    try { await unlink(rendered); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    await rmdir(dir);
  }
}

export async function buildProbeStudyScenario(sourceRoot, original, study) {
  const inventory = await loadStudyInventory(sourceRoot);
  const instruction = inventory.assets.find(asset => asset.id === inventory.probe_instruction);
  const pcm = await decodePcm(join(sourceRoot, STUDY_DIRECTORY, instruction.file));
  if (pcm.length !== instruction.frames_44100 * 2 || !pcm.some(byte => byte !== 0)) invalid('unexpected probe instruction PCM');

  const totalFrames = 300 * RATE;
  if (original.scenario_id !== 'probe-5' || original.total_frames !== totalFrames
      || original.sample_rate !== RATE) invalid('probe must be five minutes');
  const guidance = study.plan.events.filter(event => event.offset_frames >= OPENING_FRAMES
    && event.offset_frames < OPENING_FRAMES + totalFrames);
  if (guidance.length !== totalFrames / PHASE_FRAMES || guidance.at(-1).id !== 'exhale') {
    invalid('probe guidance must end after the final exhale');
  }

  const files = new Map();
  const cueDirectory = 'assets/audio/scenarios/probe-5';
  const instructionFrames = instruction.frames_44100;
  const opening = { id: 'instruction', asset: instruction.id, offset_frames: 0, duration_frames: instructionFrames, cycle: null };
  const instructionFile = stagedCueFilename(0, opening);
  const wav = encodeWav(pcm);
  const registry = {
    [instruction.id]: {
      id: instruction.id, file: instructionFile, frames: instructionFrames, bytes: wav.length,
      sha256: sha256(wav), placeholder: false, text: instruction.text,
    },
  };
  files.set(posix.join(cueDirectory, instructionFile), wav);
  const events = [opening, ...guidance.map(event => ({
    ...event, offset_frames: event.offset_frames - OPENING_FRAMES + instructionFrames,
  }))];
  for (const [index, event] of events.entries()) {
    if (Object.hasOwn(registry, event.asset)) continue;
    const previous = study.plan.audio_assets[event.asset];
    const file = stagedCueFilename(index, event);
    registry[event.asset] = { ...previous, file };
    files.set(posix.join(cueDirectory, file), study.files.get(posix.join('assets/audio/scenarios/haupt-15', previous.file)));
  }

  const plan = {
    schema_version: original.schema_version,
    scenario_id: 'probe-5',
    library_manifest_sha256: original.library_manifest_sha256,
    audio_assets_base_directory: original.audio_assets_base_directory,
    sample_rate: RATE,
    full_round: true,
    total_frames: totalFrames + instructionFrames,
    events,
    audio_assets: registry,
  };
  return { plan, files };
}
