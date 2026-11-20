import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildMainStudyScenario, renderMainStudyTrack } from '../quest/main-study-audio.mjs';

const output = process.argv[2];
if (!output || process.argv.length !== 3 || !output.toLowerCase().endsWith('.wav')) {
  console.error('Usage: node tools/render-main-study.mjs <new output.wav>');
  process.exit(2);
}

const root = fileURLToPath(new URL('../', import.meta.url));
const original = JSON.parse(await readFile(new URL('../audio/scenarios/haupt-15.json', import.meta.url), 'utf8'));
const study = await buildMainStudyScenario(root, original);
await renderMainStudyTrack(study, resolve(output));
const { plan } = study;
console.log(`${resolve(output)} — ${plan.total_frames / plan.sample_rate}s including the outro, ${plan.cycles} complete cycles`);
