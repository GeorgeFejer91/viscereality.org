import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { AUDIO_LIBRARY, SCENARIOS, validateScenario, scenarioCsv } from '../audio/scenario-protocol.js';
import { validateManifest } from '../audio/box-breathing.js';

const directory = new URL('../audio/scenarios/', import.meta.url);
const bytes = await readFile(new URL(AUDIO_LIBRARY, directory));
const library = validateManifest(JSON.parse(bytes));
const manifestHash = createHash('sha256').update(bytes).digest('hex');
const check = process.argv.includes('--check');
for (const id of SCENARIOS) {
  const plan = validateScenario(JSON.parse(await readFile(new URL(`${id}.json`, directory))), library,
    { scenarioId: id, manifestHash });
  const csv = scenarioCsv(plan, library);
  const target = new URL(`${id}.csv`, directory);
  if (check) {
    if (await readFile(target, 'utf8') !== csv) throw new Error(`Stale CSV: ${id}. Run pnpm run scenarios:csv.`);
  } else await writeFile(target, csv);
}
console.log(`${check ? 'Checked' : 'Updated'} ${SCENARIOS.length} scenario JSON/CSV pairs.`);
