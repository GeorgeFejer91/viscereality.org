import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SCENARIOS } from '../audio/scenario-protocol.js';
import { stageQuestAudio } from '../quest/stage-audio.mjs';

const digest = bytes => createHash('sha256').update(bytes).digest('hex');

test('stages original scenario JSON and every numbered cue with reproducible identities', async t => {
  const scratch = await mkdtemp(join(tmpdir(), 'atmungsraum-quest-audio-'));
  t.after(() => rm(scratch, { recursive: true, force: true }));
  const output = join(scratch, 'package');
  const catalog = await stageQuestAudio({ output });
  assert.deepEqual(catalog.scenarios.map(entry => entry.id), SCENARIOS);
  assert.equal(catalog.files.filter(entry => entry.path.endsWith('.wav')).length, 53);
  assert.deepEqual(JSON.parse(await readFile(join(output, 'scenario-catalog.json'), 'utf8')), catalog);

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
});
