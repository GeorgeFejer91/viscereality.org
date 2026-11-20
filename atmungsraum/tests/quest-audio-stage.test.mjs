import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { access, copyFile, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { SCENARIOS } from '../audio/scenario-protocol.js';
import { stageQuestAudio } from '../quest/stage-audio.mjs';

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const run = promisify(execFile);

test('stages original scenario JSON and every numbered cue with reproducible identities', async t => {
  const scratch = await mkdtemp(join(tmpdir(), 'atmungsraum-quest-audio-'));
  t.after(() => rm(scratch, { recursive: true, force: true }));
  const output = join(scratch, 'package');
  const catalog = await stageQuestAudio({ output });
  assert.deepEqual(catalog.scenarios.map(entry => entry.id), SCENARIOS);
  assert.equal(catalog.files.filter(entry => entry.path.startsWith('assets/audio/scenarios/') && entry.path.endsWith('.wav')).length, 53);
  assert.equal(catalog.box_loop.path, 'audio/box-loop.json');
  assert.equal(catalog.box_loop.cycle_frames, 705600);
  const loop = JSON.parse(await readFile(join(output, catalog.box_loop.path), 'utf8'));
  assert.deepEqual(loop.phases.map(phase => phase.id), ['inhale', 'hold_in', 'exhale', 'hold_out']);
  for (const phase of loop.phases) {
    const cue = await readFile(join(output, phase.cue.path));
    assert.equal(digest(cue), phase.cue.sha256);
    assert.equal(cue.length, phase.cue.bytes);
  }
  assert.deepEqual(JSON.parse(await readFile(join(output, 'scenario-catalog.json'), 'utf8')), catalog);
  const prompts = 'assets/audio/calibration-menu';
  assert.deepEqual(catalog.files.filter(entry => entry.path.endsWith('.mp3')).map(entry => entry.path).sort(),
    [prompts + '/exhale.mp3', prompts + '/inhale.mp3']);
  for (const sidecar of ['manifest.json', 'inhale.txt', 'exhale.txt']) {
    await copyFile(new URL('../' + prompts + '/' + sidecar, import.meta.url), join(output, prompts, sidecar));
  }

  for (const { path, bytes, sha256 } of catalog.files) {
    const stored = await readFile(join(output, path));
    assert.equal(stored.length, bytes, path);
    assert.equal(digest(stored), sha256, path);
  }
  const main = JSON.parse(await readFile(join(output, 'audio/scenarios/haupt-15.json'), 'utf8'));
  assert.equal(main.total_frames, 900 * 44100);
  assert.equal(main.calibration_start_frame, 26 * 44100);
  assert.equal(main.calibration_frames, 15 * 44100);
  await assert.rejects(() => stageQuestAudio({ output }), { code: 'EEXIST' });

  const planPath = join(output, 'audio/scenarios/haupt-15.json');
  const selected = main.audio_assets.instruction_1;
  const wavPath = join(output, 'assets/audio/scenarios/haupt-15', selected.file);
  const wav = await readFile(wavPath);
  wav.writeInt16LE(123, 44);
  await writeFile(wavPath, wav);
  const broken = join(scratch, 'broken');
  await assert.rejects(() => stageQuestAudio({ sourceRoot: output, output: broken }), /integrity/);
  await assert.rejects(() => access(broken), { code: 'ENOENT' });

  Object.assign(selected, {
    sha256: digest(wav), placeholder: false, spoken_text: 'Synthetic test cue', audio_status: 'recorded',
  });
  const newPlan = JSON.stringify(main);
  await writeFile(planPath, newPlan);
  const manifestPath = join(output, 'assets/audio/scenarios/haupt-15/0000_manifest.json');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  manifest.source_scenario_sha256 = digest(newPlan);
  Object.assign(manifest.assets.find(asset => asset.id === selected.id), selected);
  await writeFile(manifestPath, JSON.stringify(manifest));
  const replaced = join(scratch, 'replacement');
  await stageQuestAudio({ sourceRoot: output, output: replaced });
  assert.deepEqual(await readFile(join(replaced, 'assets/audio/scenarios/haupt-15', selected.file)), wav);

  await writeFile(join(output, prompts, 'exhale.txt'), 'Changed instruction.\n');
  const wrongPrompt = join(scratch, 'wrong-prompt');
  await assert.rejects(() => stageQuestAudio({ sourceRoot: output, output: wrongPrompt }), /Calibration prompt integrity/);
  await assert.rejects(() => access(wrongPrompt), { code: 'ENOENT' });
});

test('the authored Quest package binds the entire main mix including the outro, exploration and calibration instruction', async t => {
  const scratch = await mkdtemp(join(tmpdir(), 'atmungsraum-authored-audio-'));
  t.after(() => rm(scratch, { recursive: true, force: true }));
  const output = join(scratch, 'package');
  const catalog = await stageQuestAudio({ output, mainStudy: true });
  const main = JSON.parse(await readFile(join(output, 'audio/scenarios/haupt-15.json'), 'utf8'));
  assert.equal(main.total_frames, 40_527_900);
  assert.equal(main.timed_audio.duration_frames, main.total_frames);
  assert.equal(main.post_session_audio, undefined);
  const outro = main.events.at(-1);
  assert.deepEqual([outro.id, outro.offset_frames, outro.duration_frames], ['outro', 39_690_000, 837_900]);
  const clip = main.timed_audio;
  const bytes = await readFile(join(output, clip.path));
  assert.equal(bytes.length, clip.bytes);
  assert.equal(digest(bytes), clip.sha256);
  assert.ok(catalog.files.some(entry => entry.path === clip.path && entry.sha256 === clip.sha256));
  const review = join(scratch, 'review.wav');
  await run(process.execPath, [fileURLToPath(new URL('../tools/render-main-study.mjs', import.meta.url)), review]);
  const pcmHash = async source => (await run('ffmpeg', ['-v', 'error', '-i', source,
    '-c:a', 'pcm_s16le', '-f', 'hash', '-hash', 'sha256', '-'])).stdout.trim();
  assert.equal(await pcmHash(join(output, main.timed_audio.path)), await pcmHash(review));
  const exploration = catalog.scenarios.find(({ id }) => id === 'free-exploration');
  assert.ok(exploration, 'the headset can discover a packaged exploration timeline');
  assert.equal(exploration.total_frames, 19_715_793);
  const plan = JSON.parse(await readFile(join(output, exploration.path), 'utf8'));
  assert.deepEqual(plan.events, [
    { id: 'instruction', offset_frames: 0, duration_frames: 960_768 },
    { id: 'rest', offset_frames: 960_768, duration_frames: 18_755_025 },
  ]);
  assert.equal(plan.encoded_audio.duration_frames, exploration.total_frames);
  assert.equal(catalog.calibration_instruction.duration_frames, 1_952_640);
  for (const [clip, expected] of [
    [plan.encoded_audio, '1f1e942ba5ec3c74eb680958047b90ba0e8385e3d4dc3eb641514c57b6f6ab2a'],
    [catalog.calibration_instruction, '9d723c7fd5e397c087acf8be5e16c38b04344a7deed4c908057a645ac2e0c67a'],
  ]) {
    const bytes = await readFile(join(output, clip.path));
    assert.equal(bytes.length, clip.bytes);
    assert.equal(digest(bytes), expected);
    assert.equal(clip.sha256, expected);
    assert.ok(catalog.files.some(entry => entry.path === clip.path && entry.sha256 === expected));
  }
});
