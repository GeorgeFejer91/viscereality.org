import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, posix, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { validateManifest } from '../audio/box-breathing.js';
import { AUDIO_LIBRARY, SCENARIOS, validateScenario } from '../audio/scenario-protocol.js';

const projectRoot = fileURLToPath(new URL('../', import.meta.url));
const scenarioDirectory = 'audio/scenarios';
const libraryPath = posix.join(scenarioDirectory, AUDIO_LIBRARY);
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const describe = (path, data) => ({ path, bytes: data.length, sha256: sha256(data) });

export async function stageQuestAudio({ sourceRoot = projectRoot, output }) {
  const files = new Map();
  const libraryBytes = await readFile(join(sourceRoot, libraryPath));
  const library = validateManifest(JSON.parse(libraryBytes.toString('utf8')));
  const libraryRecord = describe(libraryPath, libraryBytes);
  files.set(libraryPath, libraryBytes);

  const scenarios = [];
  for (const id of SCENARIOS) {
    const path = posix.join(scenarioDirectory, id + '.json');
    const data = await readFile(join(sourceRoot, path));
    const plan = validateScenario(JSON.parse(data.toString('utf8')), library, {
      scenarioId: id,
      manifestHash: libraryRecord.sha256,
    });
    files.set(path, data);
    scenarios.push({ id, path, total_frames: plan.total_frames });
    const assetDirectory = posix.join(scenarioDirectory, plan.audio_assets_base_directory);
    const manifestPath = posix.join(assetDirectory, '0000_manifest.json');
    const manifestBytes = await readFile(join(sourceRoot, manifestPath));
    const manifest = JSON.parse(manifestBytes.toString('utf8'));
    const expectedIds = new Set(Object.keys(plan.audio_assets));
    if (plan.recovery_frames > 0) expectedIds.add('recovery');
    if (manifest.scenario_id !== id || manifest.sample_rate !== plan.sample_rate
        || manifest.source_scenario_sha256 !== sha256(data)
        || !Array.isArray(manifest.assets) || manifest.assets.length !== expectedIds.size) {
      throw new Error('Quest audio manifest mismatch: ' + manifestPath);
    }
    files.set(manifestPath, manifestBytes);

    for (const asset of manifest.assets) {
      const expected = plan.audio_assets[asset.id];
      if (!expectedIds.delete(asset.id) || !/^\d{4}_\d{2}m\d{2}s_[a-z0-9-]+\.wav$/.test(asset.file)
          || (asset.id === 'recovery'
            ? asset.frames !== plan.recovery_frames || asset.placeholder !== true
            : asset.file !== expected.file || asset.sha256 !== expected.sha256
              || asset.bytes !== expected.bytes || asset.frames !== expected.frames)) {
        throw new Error('Quest audio registry mismatch: ' + manifestPath);
      }
      const assetPath = posix.join(assetDirectory, asset.file);
      const audio = await readFile(join(sourceRoot, assetPath));
      if (files.has(assetPath) || audio.length !== asset.bytes || sha256(audio) !== asset.sha256) {
        throw new Error('Quest audio integrity mismatch: ' + assetPath);
      }
      files.set(assetPath, audio);
    }
  }

  const catalog = {
    schema_version: 1,
    sample_rate: library.sample_rate,
    library: libraryPath,
    scenarios,
    files: Array.from(files, ([path, data]) => describe(path, data)),
  };
  const target = resolve(output);
  await mkdir(dirname(target), { recursive: true });
  await mkdir(target);
  for (const [path, data] of files) {
    const destination = join(target, path);
    await mkdir(dirname(destination), { recursive: true });
    await writeFile(destination, data, { flag: 'wx' });
  }
  // An interrupted stage never receives a catalog that can be used by the packager.
  await writeFile(join(target, 'scenario-catalog.json'), JSON.stringify(catalog, null, 2) + '\n', { flag: 'wx' });
  return catalog;
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  const output = process.argv[2];
  if (!output || process.argv.length !== 3) {
    console.error('Usage: node quest/stage-audio.mjs <new output directory>');
    process.exitCode = 2;
  } else {
    try {
      const catalog = await stageQuestAudio({ output });
      console.log(JSON.stringify({
        output: resolve(output),
        scenarios: catalog.scenarios.length,
        files: catalog.files.length,
      }));
    } catch (error) {
      console.error(error.message);
      process.exitCode = 1;
    }
  }
}
