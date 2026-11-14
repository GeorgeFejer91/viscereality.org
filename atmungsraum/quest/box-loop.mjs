import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const source = fileURLToPath(new URL('../', import.meta.url));
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const fail = reason => { throw new Error('Box loop: ' + reason); };
const integer = value => Number.isSafeInteger(value) && value >= 0;

export function validateBoxLoop(plan) {
  if (!plan || plan.schema !== 'atmungsraum.box_breathing_loop.v1'
      || plan.sample_rate !== 44100 || plan.channels !== 1 || plan.bits_per_sample !== 16
      || plan.main_seconds !== 900 || !Array.isArray(plan.phases) || plan.phases.length !== 4) {
    fail('unsupported clock or PCM contract');
  }
  const ids = ['inhale', 'hold_in', 'exhale', 'hold_out'];
  let offset = 0;
  const paths = new Set();
  for (const [index, phase] of plan.phases.entries()) {
    const cue = phase?.cue;
    if (phase.id !== ids[index] || phase.offset_frames !== offset
        || !integer(phase.duration_frames) || phase.duration_frames !== 4 * plan.sample_rate
        || !cue || cue.frames !== phase.duration_frames || !integer(cue.bytes)
        || cue.bytes < cue.frames * 2 + 44 || cue.bytes > 1024 * 1024
        || !/^[a-f0-9]{64}$/.test(cue.sha256)
        || typeof cue.path !== 'string' || !/^assets\/audio\/[a-zA-Z0-9_./-]+\.wav$/.test(cue.path)
        || cue.path.split('/').some(part => !part || part === '.' || part === '..')
        || paths.has(cue.path)) fail('invalid phase, cue identity or frame coverage');
    paths.add(cue.path);
    offset += phase.duration_frames;
  }
  if (plan.cycle_frames !== offset) fail('cycle frame coverage');
  return plan;
}

export function decodeBoxLoopWav(bytes, cue, plan) {
  if (bytes.length !== cue.bytes || sha256(bytes) !== cue.sha256) fail('cue identity mismatch');
  if (bytes.length < 44 || bytes.toString('ascii', 0, 4) !== 'RIFF'
      || bytes.toString('ascii', 8, 12) !== 'WAVE' || bytes.readUInt32LE(4) !== bytes.length - 8) {
    fail('invalid WAV envelope');
  }
  let format = false;
  let pcm;
  let offset = 12;
  for (; offset + 8 <= bytes.length;) {
    const id = bytes.toString('ascii', offset, offset + 4);
    const length = bytes.readUInt32LE(offset + 4);
    const start = offset + 8;
    if (start + length > bytes.length) fail('truncated WAV chunk');
    if (id === 'fmt ') {
      if (format || length < 16 || bytes.readUInt16LE(start) !== 1
          || bytes.readUInt16LE(start + 2) !== plan.channels
          || bytes.readUInt32LE(start + 4) !== plan.sample_rate
          || bytes.readUInt32LE(start + 8) !== plan.sample_rate * 2
          || bytes.readUInt16LE(start + 12) !== 2 || bytes.readUInt16LE(start + 14) !== 16) {
        fail('unsupported PCM format');
      }
      format = true;
    } else if (id === 'data') {
      if (pcm) fail('duplicate WAV data');
      pcm = bytes.subarray(start, start + length);
    }
    offset = start + length + (length & 1);
    if (offset > bytes.length) fail('truncated WAV padding');
  }
  if (offset !== bytes.length) fail('trailing WAV chunk bytes');
  if (!format || !pcm || pcm.length !== cue.frames * 2) fail('PCM frames mismatch');
  return pcm;
}

export function recordingPlan(plan, hash) {
  validateBoxLoop(plan);
  if (!/^[a-f0-9]{64}$/.test(hash)) fail('invalid loop JSON hash');
  const total = plan.main_seconds * plan.sample_rate;
  const phases = [];
  for (let base = 0; base < total; base += plan.cycle_frames) {
    for (const phase of plan.phases) {
      const offset = base + phase.offset_frames;
      if (offset < total) phases.push([offset, phase.id]);
    }
  }
  return { scenario_id: 'haupt-15', scenario_sha256: hash,
    sample_rate: plan.sample_rate, total_frames: total, phases };
}

/** Paths are relative to the existing atmungsraum-audio packaged asset root. */
export async function loadBoxLoop({ sourceRoot = source } = {}) {
  const path = 'audio/box-loop.json';
  const bytes = await readFile(join(sourceRoot, path));
  const plan = validateBoxLoop(JSON.parse(bytes.toString('utf8')));
  const files = new Map([[path, bytes]]);
  for (const phase of plan.phases) {
    const data = await readFile(join(sourceRoot, phase.cue.path));
    decodeBoxLoopWav(data, phase.cue, plan);
    files.set(phase.cue.path, data);
  }
  return { plan, sha256: sha256(bytes), files };
}

export async function stageBoxLoop({ sourceRoot = source, output }) {
  const loop = await loadBoxLoop({ sourceRoot });
  for (const [path, bytes] of loop.files) {
    const target = join(output, path);
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, bytes, { flag: 'wx' });
  }
  return loop;
}
