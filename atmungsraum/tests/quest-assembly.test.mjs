import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { mkdtemp, readFile, readdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const questRepo = process.env.ATMUNGSRAUM_QUEST_REPO;
const assembler = fileURLToPath(new URL('../quest/assemble-apk.ps1', import.meta.url));
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const audioDirectory = 'apps/native-renderer-android/panel-modules/condition-audio/src/main/java/io/github/mesmerprism/rustyquest/native_renderer/';

function invoke(runName, extra = []) {
  return spawnSync('pwsh', ['-NoProfile', '-File', assembler, '-QuestRepo', questRepo,
    '-BundledAssets', '-ResolveOnly', '-RunName', runName, ...extra], { encoding: 'utf8', timeout: 60000 });
}

async function resolvedRun(runName, appId) {
  const output = join(tmpdir(), 'atmungsraum-native', runName, 'resolved', appId);
  const [fingerprint] = await readdir(output);
  const root = join(output, fingerprint);
  return {
    lock: JSON.parse(await readFile(join(root, 'feature-lock.json'), 'utf8')),
    manifest: await readFile(join(root, 'AndroidManifest.xml'), 'utf8'),
    specPath: join(questRepo, 'local-artifacts', 'atmungsraum', runName, 'atmungsraum.app.json'),
    root,
  };
}

test('assembly binds the real looping audio and selected Java definitions before packaging',
  { skip: !questRepo && 'Set ATMUNGSRAUM_QUEST_REPO to a patched native checkout.' }, async () => {
    const runName = `asm-test-${randomUUID().slice(0, 12)}`;
    const result = invoke(runName);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    const run = await resolvedRun(runName, 'atmungsraum_quest2');
    const catalogPath = join(questRepo, 'local-artifacts', 'atmungsraum', runName, 'atmungsraum-audio', 'scenario-catalog.json');
    const catalog = JSON.parse(await readFile(catalogPath, 'utf8'));
    const loop = await readFile(join(catalogPath, '..', catalog.box_loop.path));
    const binding = run.lock.build_inputs.env.filter(entry => entry.name === 'RUSTY_QUEST_NATIVE_RENDERER_BOX_LOOP_SHA256');
    assert.equal(binding.length, 1, 'one authenticated audio-loop input');
    assert.equal(binding[0].value, sha256(loop));
    for (const filename of ['ScenarioAudioPackage.java', 'ScenarioAudioAndroidTracks.java', 'BreathingSetupAudio.java', 'BreathingSetupAudioContract.java']) {
      const path = audioDirectory + filename;
      const selected = run.lock.panel_source_closure.source_files.filter(entry => entry.path === path);
      assert.equal(selected.length, 1, `${filename} definition must be selected exactly once`);
      assert.equal(selected[0].sha256, sha256(await readFile(join(questRepo, path))));
    }
    assert.match(run.manifest, /android:debuggable="true"/);
    assert.match(run.manifest, /android:targetSdkVersion="35"/);
    assert.doesNotMatch(run.manifest, /android:installLocation=/);
    assert.match(run.manifest, /package="org\.viscereality\.atmungsraum\.rehearsal"/);
    assert.match(run.manifest, /android:name="rusty\.quest\.experiment\.breathing_setup\.enabled" android:value="true"/);
    const schemaCheck = spawnSync('pwsh', ['-NoProfile', '-Command',
      'if (!(Test-Json -Json (Get-Content -Raw -LiteralPath $env:ASM_TEST_SPEC) -SchemaFile $env:ASM_TEST_SCHEMA)) { throw "Assembled app violates schema" }; if (!(Test-Json -Json (Get-Content -Raw -LiteralPath $env:ASM_TEST_FEATURE) -SchemaFile $env:ASM_TEST_FEATURE_SCHEMA)) { throw "Scenario feature violates schema" }'],
    { encoding: 'utf8', env: { ...process.env, ASM_TEST_SPEC: run.specPath,
      ASM_TEST_SCHEMA: join(questRepo, 'schemas/rusty.quest.native_app_build.v1.schema.json'),
      ASM_TEST_FEATURE: join(run.specPath, '..', 'features/audio.atmungsraum_scenarios.feature.json'),
      ASM_TEST_FEATURE_SCHEMA: join(questRepo, 'schemas/rusty.quest.native_app_feature.v1.schema.json') } });
    assert.equal(schemaCheck.status, 0, schemaCheck.stdout + schemaCheck.stderr);
  });

test('developer resolution selects a distinct non-debuggable app and requires an explicit external key and version',
  { skip: !questRepo && 'Set ATMUNGSRAUM_QUEST_REPO to a patched native checkout.' }, async () => {
    const runName = `asm-dev-${randomUUID().slice(0, 12)}`;
    const scratch = await mkdtemp(join(tmpdir(), 'assembly-signing-fixture-'));
    // ResolveOnly never opens or signs with this fixture. It must not generate a real key.
    const key = join(scratch, 'not-a-signing-key.keystore');
    await writeFile(key, 'resolve-only path fixture');
    const result = invoke(runName, ['-Developer', '-Keystore', key, '-VersionCode', '30', '-VersionName', '0.1.30-dev']);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    const run = await resolvedRun(runName, 'atmungsraum_developer');
    assert.match(run.manifest, /package="org\.viscereality\.atmungsraum\.developer"/);
    assert.match(run.manifest, /android:label="Atmungsraum Developer"/);
    assert.match(run.manifest, /android:debuggable="false"/);
    assert.match(run.manifest, /android:targetSdkVersion="34"/);
    assert.match(run.manifest, /android:installLocation="auto"/);
    assert.equal(run.lock.android_manifest.debuggable, false);
    assert.equal(run.lock.android_manifest.target_sdk_version, 34);
    assert.equal(run.lock.android_manifest.install_location, 'auto');
    assert.equal(run.lock.android_manifest.experiment_phone_pairing_enabled, true);
    assert.equal(run.lock.android_manifest.experiment_panel_initially_closed, true);
    assert.equal(run.lock.android_manifest.experiment_headset_setup_enabled, false);
    assert.match(run.manifest, /android:name="rusty\.quest\.experiment\.phone_pairing\.enabled" android:value="true"/);
    assert.deepEqual(run.lock.build_inputs.env.filter(entry => entry.name === 'RUSTY_QUEST_NATIVE_RENDERER_PANEL_INITIALLY_CLOSED')
      .map(entry => entry.value), ['true']);
    assert.doesNotMatch(run.manifest, /debug-experiment-control|\.headset-setup/);
    assert.equal(run.lock.headset_setup_sources.length, 0);
    assert.equal(run.lock.debug_experiment_operator_sources.length, 0);
    assert.equal(run.lock.phone_pairing_sources.length, 5);
    for (const source of run.lock.phone_pairing_sources) {
      assert.equal(source.sha256, sha256(await readFile(join(questRepo, source.path))), source.path);
    }
    const noKey = invoke(runName + '-nokey', ['-Developer', '-VersionCode', '31']);
    assert.notEqual(noKey.status, 0);
    assert.match(noKey.stdout + noKey.stderr, /Developer requires an explicit existing -Keystore/);
    const noVersion = invoke(runName + '-nover', ['-Developer', '-Keystore', key]);
    assert.notEqual(noVersion.status, 0);
    assert.match(noVersion.stdout + noVersion.stderr, /Developer requires an explicit -VersionCode/);
    const trackedKey = invoke(runName + '-git', ['-Developer', '-Keystore', resolve(questRepo, 'README.md'), '-VersionCode', '31']);
    assert.notEqual(trackedKey.status, 0);
    assert.match(trackedKey.stdout + trackedKey.stderr, /Developer keystore must be outside Git/);
  });
