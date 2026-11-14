import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadBoxLoop, validateBoxLoop, decodeBoxLoopWav, recordingPlan, stageBoxLoop } from '../quest/box-loop.mjs';

const sourceRoot = fileURLToPath(new URL('../', import.meta.url));
const copy = value => JSON.parse(JSON.stringify(value));

test('verified four cues stage unchanged and recording ends at exactly 900 seconds', async () => {
  const loop = await loadBoxLoop({ sourceRoot });
  assert.equal(loop.files.size, 5);
  assert.deepEqual(loop.plan.phases.map(p => [p.id, p.offset_frames, p.duration_frames]), [
    ['inhale', 0, 176400], ['hold_in', 176400, 176400],
    ['exhale', 352800, 176400], ['hold_out', 529200, 176400],
  ]);
  const recording = recordingPlan(loop.plan, loop.sha256);
  assert.equal(recording.scenario_id, 'haupt-15');
  assert.equal(recording.scenario_sha256, loop.sha256);
  assert.equal(recording.total_frames, 39690000);
  assert.equal(recording.phases.length, 225);
  assert.deepEqual(recording.phases[0], [0, 'inhale']);
  assert.deepEqual(recording.phases[224], [39513600, 'inhale']);
  const output = await mkdtemp(join(tmpdir(), 'box-loop-'));
  try {
    await stageBoxLoop({ sourceRoot, output });
    for (const [path, bytes] of loop.files) assert.deepEqual(await readFile(join(output, path)), bytes);
    await assert.rejects(stageBoxLoop({ sourceRoot, output }), /exist/i);
  } finally { await rm(output, { recursive: true, force: true }); }
});

test('schedule rejects gaps, duplicated phases, unsafe paths and altered clock', async () => {
  const { plan } = await loadBoxLoop({ sourceRoot });
  for (const mutate of [
    p => p.phases[1].offset_frames++, p => p.phases[1].id = 'inhale',
    p => p.phases[0].cue.path = '../inhale.wav', p => p.sample_rate = 48000,
    p => p.main_seconds = 899, p => p.phases[0].cue.frames--,
    p => p.phases[0].cue.sha256 = 'bad', p => p.cycle_frames++,
    p => p.phases[0].duration_frames = 176400.5,
  ]) {
    const invalid = copy(plan); mutate(invalid);
    assert.throws(() => validateBoxLoop(invalid));
  }
});

test('PCM decoder enforces the declared hash, bytes, format and frame count', async () => {
  const { plan, files } = await loadBoxLoop({ sourceRoot });
  const cue = plan.phases[0].cue;
  const bytes = files.get(cue.path);
  assert.equal(decodeBoxLoopWav(bytes, cue, plan).length, 352800);
  const altered = Buffer.from(bytes); altered[100] ^= 1;
  assert.throws(() => decodeBoxLoopWav(altered, cue, plan), /identity/);
  assert.throws(() => decodeBoxLoopWav(bytes, { ...cue, bytes: cue.bytes - 1 }, plan), /identity/);
  const wrongFrames = { ...cue, frames: cue.frames - 1 };
  assert.throws(() => decodeBoxLoopWav(bytes, wrongFrames, plan), /frames/);
  const format = Buffer.from(bytes); format.writeUInt16LE(3, 20);
  const { createHash } = await import('node:crypto');
  assert.throws(() => decodeBoxLoopWav(format, { ...cue,
    sha256: createHash('sha256').update(format).digest('hex') }, plan), /PCM/);
});
