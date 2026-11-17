import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
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
const PHASES = [['inhale', 'inhale'], ['hold_in', 'hold'], ['exhale', 'exhale'], ['hold_out', 'hold']];
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
  const addPhase = ([phase, pool], cycle) => {
    const ids = inventory.pools[pool];
    const take = cycle + (phase === 'hold_out' ? 1 : 0);
    add(phase, ids[take % ids.length], PHASE_FRAMES, cycle + 1);
  };
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

async function loadBreathLayer(sourceRoot, phase) {
  const read = async name => wavePcm(await readFile(join(sourceRoot, SOFT_LOOP_DIRECTORY, name + '.wav')), PHASE_FRAMES);
  const [dry, mixed] = await Promise.all([read(phase), read(phase + '_breath')]);
  return { dry, mixed };
}

function mixPhase(voice, layer) {
  let peak = 0;
  for (let i = 0; i < voice.length; i += 2) peak = Math.max(peak, Math.abs(voice.readInt16LE(i)));
  if (!peak) invalid('silent voice phase');
  const voicePeak = 32768 * 10 ** (-8 / 20);
  for (let i = 0; i < voice.length; i += 2) {
    const breath = layer ? layer.mixed.readInt16LE(i) - layer.dry.readInt16LE(i) : 0;
    const sample = Math.round(voice.readInt16LE(i) * voicePeak / peak + breath);
    if (Math.abs(sample) > 32767) invalid('phase mix clips');
    voice.writeInt16LE(sample, i);
  }
  return voice;
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

export async function buildMainStudyScenario(sourceRoot, original) {
  const inventory = await loadStudyInventory(sourceRoot);
  const timeline = planMainStudy(inventory);
  const assets = new Map(inventory.assets.map(asset => [asset.id, asset]));
  const registry = {};
  const files = new Map();
  const prefix = 'assets/audio/scenarios/haupt-15';
  const layers = {
    inhale: await loadBreathLayer(sourceRoot, 'inhale'),
    exhale: await loadBreathLayer(sourceRoot, 'exhale'),
  };
  for (const [index, event] of timeline.events.entries()) {
    if (event.asset === null || Object.hasOwn(registry, event.asset)) continue;
    const source = assets.get(event.asset);
    const cutoff = source.phase_source_end_seconds;
    const tempo = cutoff ? Math.max(1, cutoff / 3.92) : 1;
    const filters = cutoff
      ? `atrim=end=${cutoff},atempo=${tempo},afade=t=in:d=0.008,afade=t=out:st=3.92:d=0.065,apad`
      : source.role === 'hold' ? 'afade=t=in:d=0.008,apad' : '';
    const args = filters ? ['-af', filters, '-t', '4'] : [];
    let pcm = await decodePcm(join(sourceRoot, STUDY_DIRECTORY, source.file), args);
    if (index < 2) {
      if (pcm.length !== source.frames_44100 * 2) invalid('unexpected control length: ' + source.id);
      pcm = extendControl(pcm, index, event.duration_frames);
    }
    if (event.cycle !== null) pcm = mixPhase(pcm, layers[source.role]);
    if (pcm.length !== event.duration_frames * 2 || !pcm.some(byte => byte !== 0)) invalid('unexpected PCM: ' + source.id);
    const wave = encodeWav(pcm);
    const seconds = Math.floor(event.offset_frames / RATE);
    const firstUse = String(Math.floor(seconds / 60)).padStart(2, '0') + 'm' + String(seconds % 60).padStart(2, '0') + 's';
    const file = String(index + 1).padStart(4, '0') + '_' + firstUse + '_' + source.id + '.wav';
    registry[source.id] = {
      id: source.id, file, frames: event.duration_frames, bytes: wave.length, sha256: sha256(wave),
      placeholder: false, text: source.phase_text ?? source.text, source_sha256: source.sha256,
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
  files.set(outroPath, outroWav);
  const plan = {
    schema_version: original.schema_version,
    library_manifest_sha256: original.library_manifest_sha256,
    audio_assets_base_directory: original.audio_assets_base_directory,
    ...timeline, audio_assets: registry,
    post_session_audio: {
      path: outroPath, offset_frames: TOTAL_FRAMES, duration_frames: outro.frames,
      bytes: outro.bytes, sha256: outro.sha256, text: outro.text,
    },
  };
  const music = assets.get(inventory.background);
  return { plan, files, music: join(sourceRoot, STUDY_DIRECTORY, music.file) };
}
