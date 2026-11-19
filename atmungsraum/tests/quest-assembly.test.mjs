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

test('Meta startup updates require explicit complete inputs before assembly effects',
  { skip: !questRepo && 'Set ATMUNGSRAUM_QUEST_REPO to a patched native checkout.' }, () => {
    const runName = `asm-meta-${randomUUID().slice(0, 12)}`;
    const missing = invoke(runName, ['-MetaPlatformUpdate']);
    assert.notEqual(missing.status, 0);
    assert.match(missing.stdout + missing.stderr, /MetaPlatformUpdate requires/);
    const ambient = invoke(runName + '-ambient', ['-MetaPlatformAppId', '123']);
    assert.notEqual(ambient.status, 0);
    assert.match(ambient.stdout + ambient.stderr, /require explicit -MetaPlatformUpdate/);
  });

test('Meta update resolution binds the explicit app identity and preserves the external boot helper split',
  { skip: !questRepo && 'Set ATMUNGSRAUM_QUEST_REPO to a patched native checkout.' }, async () => {
    const runName = `asm-meta-on-${randomUUID().slice(0, 10)}`;
    const scratch = await mkdtemp(join(tmpdir(), 'meta-update-resolve-'));
    const key = join(scratch, 'resolve-only.keystore');
    await writeFile(key, 'resolve-only path fixture');
    const result = invoke(runName, ['-Developer', '-Keystore', key, '-VersionCode', '32',
      '-MetaPlatformUpdate', '-MetaPlatformAppId', '1234567890123456',
      '-MetaRequiredVersionManifestUrl', 'https://viscereality.org/atmungsraum/quest-release.json', '-MetaUpdateHelper',
      '-MetaPlatformSdkRoot', scratch, '-MetaPlatformHeadersSha256', 'a'.repeat(64),
      '-MetaPlatformLoaderSha256', 'b'.repeat(64), '-MetaPlatformArchiveSha256', 'c'.repeat(64)]);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    const run = await resolvedRun(runName, 'atmungsraum_developer');
    assert.equal(run.lock.android_manifest.experiment_meta_update_enabled, true);
    assert.equal(run.lock.android_manifest.meta_required_version_manifest_url, 'https://viscereality.org/atmungsraum/quest-release.json');
    assert.equal(run.lock.android_manifest.experiment_meta_update_helper_enabled, true);
    assert.match(run.manifest, /org\.viscereality\.atmungsraum\.boot\.permission\.UPDATE_HANDOFF/);
    assert.match(run.manifest, /android:name="org\.viscereality\.atmungsraum\.boot_protocol" android:value="3"/);
    assert.match(run.manifest, /<package android:name="org\.viscereality\.atmungsraum\.boot"\s*\/>/);
    assert.match(run.manifest, /android:name="com\.oculus\.sdk\.appid" android:value="1234567890123456"/);
    assert.match(run.manifest, /android:name="rusty\.quest\.meta_update\.enabled" android:value="true"/);
    assert.doesNotMatch(run.manifest, /SYSTEM_ALERT_WINDOW|RECEIVE_BOOT_COMPLETED|NativeRendererBootReceiver/);
    await assertControllerOnly(run);
  });

test('the WebRTC lab APK resolves to its own package and locked bridge', {
  skip: !questRepo && 'Set ATMUNGSRAUM_QUEST_REPO to the WebRTC native source checkout.',
}, async () => {
  const runName = `asm-webrtc-${randomUUID().slice(0, 10)}`;
  const result = invoke(runName, ['-WebRtc']);
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.match(result.stdout, /dry-run accepted: atmungsraum_webrtc_bench/);
  const run = await resolvedRun(runName, 'atmungsraum_webrtc_bench');
  assert.match(run.manifest, /package="org\.viscereality\.atmungsraum\.webrtcbench"/);
  assert.match(run.manifest, /android:label="Atmungsraum WebRTC Bench"/);
  assert.doesNotMatch(run.manifest, /android:authorities="org\.viscereality\.atmungsraum\.(rehearsal|developer)\./);
  assert.equal(run.lock.android_manifest.experiment_meta_update_enabled, false);
  assert.ok(run.lock.build_inputs.assets.some(path => path.endsWith('/atmungsraum-webrtc')));
  assert.ok(run.lock.panel_source_closure.source_files.some(file => file.path.endsWith('/ExperimentSessionWebRtc.java')));
  await assertPhonePairing(run);
  await assertBackgroundMusic(run);
});

test('canonical assembly binds automatic pairing and Meta update to the one Developer package', {
  skip: !questRepo && 'Set ATMUNGSRAUM_QUEST_REPO to the WebRTC native source checkout.',
}, async () => {
  const runName = 'asm-canonical-' + randomUUID().slice(0, 10);
  const scratch = await mkdtemp(join(tmpdir(), 'atmungsraum-canonical-resolve-'));
  const key = join(scratch, 'resolve-only.keystore');
  await writeFile(key, 'resolve-only path fixture');
  const old = invoke(runName + '-old', ['-CanonicalRelease', '-VersionCode', '1', '-VersionName', '0.0.1']);
  assert.notEqual(old.status, 0);
  assert.match(old.stdout + old.stderr, /code greater than the advertised Developer release/);
  const conflicting = invoke(runName + '-boot', ['-CanonicalRelease', '-BootAutostart']);
  assert.notEqual(conflicting.status, 0);
  assert.match(conflicting.stdout + conflicting.stderr, /external boot helper/);
  const canonicalArgs = ['-CanonicalRelease', '-Keystore', key, '-VersionCode', '2100000000',
    '-VersionName', '0.0.0-resolve-test',
    '-MetaRequiredVersionManifestUrl', 'https://viscereality.org/atmungsraum/quest-release.json',
    '-MetaPlatformSdkRoot', scratch, '-MetaPlatformHeadersSha256', 'a'.repeat(64),
    '-MetaPlatformLoaderSha256', 'b'.repeat(64), '-MetaPlatformArchiveSha256', 'c'.repeat(64)];
  const foreign = invoke(runName + '-foreign', [...canonicalArgs, '-MetaPlatformAppId', '1234567890123456']);
  assert.notEqual(foreign.status, 0, 'reject a different Meta app before staging');
  assert.match(foreign.stdout + foreign.stderr, /Meta app 1459506690581676/);
  const result = invoke(runName, [...canonicalArgs, '-MetaPlatformAppId', '1459506690581676']);
  assert.equal(result.status, 0, result.stdout + result.stderr);
  const run = await resolvedRun(runName, 'atmungsraum_developer');
  assert.match(run.manifest, /package="org\.viscereality\.atmungsraum\.developer"/);
  assert.match(run.manifest, /android:name="com\.oculus\.sdk\.appid" android:value="1459506690581676"/);
  assert.match(run.manifest, /android:debuggable="false"/);
  assert.doesNotMatch(run.manifest, /SYSTEM_ALERT_WINDOW|RECEIVE_BOOT_COMPLETED|NativeRendererBootReceiver/);
  assert.equal(run.lock.android_manifest.experiment_meta_update_enabled, true);
  assert.equal(run.lock.android_manifest.experiment_meta_update_helper_enabled, true);
  const webRtcAssets = run.lock.build_inputs.assets.find(path => path.endsWith('/atmungsraum-webrtc'));
  assert.ok(webRtcAssets, 'the feature lock selects the WebRTC asset directory');
  const protocol = await readFile(new URL('../quest/webrtc/release-protocol.md', import.meta.url));
  assert.deepEqual(await readFile(join(questRepo, webRtcAssets, 'release-protocol.md')), protocol);
  assert.ok(run.lock.panel_source_closure.source_files.some(file => file.path.endsWith('/ExperimentSessionWebRtc.java')));
  await assertPhonePairing(run);
  await assertBackgroundMusic(run);
  await assertControllerOnly(run);
});

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

async function assertControllerOnly(run) {
  assert.doesNotMatch(run.manifest, /com\.oculus\.permission\.HAND_TRACKING|oculus\.software\.handtracking/);
  assert.deepEqual(run.lock.build_inputs.env.filter(entry => entry.name === 'RUSTY_QUEST_NATIVE_RENDERER_CONTROLLER_ONLY')
    .map(entry => entry.value), ['true']);
  const spec = JSON.parse(await readFile(run.specPath, 'utf8'));
  assert.ok(spec.requested_features.includes('input.controllers_only'));
  for (const feature of ['input.simultaneous_hands_and_controllers', 'hand_mesh_live_input',
    'hand_mesh_visual', 'input.tracked_hand_pointer', 'hand_anchor_particles']) {
    assert.ok(!spec.requested_features.includes(feature), feature);
    assert.ok(spec.denied_features.includes(feature), feature);
  }
  for (const setting of ['hand_adapter.enabled', 'hand_mesh.graft_copies.enabled', 'hand_mesh.real_hands.visible',
    'hand_mesh.visual.diagnostic.enabled', 'hand_mesh.visual.wireframe.enabled', 'simultaneous_hands_controllers.enabled']) {
    assert.equal(spec.runtime_profile.set[`debug.rustyquest.native_renderer.${setting}`], 'false', setting);
    assert.equal(spec.settings_assertions.required_values[`native_renderer.${setting}`], 'false', setting);
  }
  assert.ok(spec.settings_assertions.required_modules.includes('input/controller-only'));
  assert.ok(spec.settings_assertions.required_modules.includes('breath/controller-assessment'));
}

async function assertBackgroundMusic(run) {
  const musicPath = join(run.specPath, '..', 'atmungsraum-music', 'ji3ji3.mp3');
  assert.equal(run.lock.build_inputs.assets.filter(path => path.endsWith('/atmungsraum-music')).length, 1);
  const bundled = await readFile(musicPath);
  const supplied = await readFile(new URL('../assets/audio/background/ji3ji3.mp3', import.meta.url));
  assert.equal(sha256(bundled), sha256(supplied));
}

async function assertPhonePairing(run, initiallyClosed = true) {
  assert.equal(run.lock.android_manifest.experiment_phone_pairing_enabled, true);
  assert.equal(run.lock.android_manifest.experiment_panel_initially_closed, initiallyClosed);
  assert.equal(run.lock.android_manifest.experiment_headset_setup_enabled, false);
  assert.match(run.manifest, /android:name="rusty\.quest\.experiment\.phone_pairing\.enabled" android:value="true"/);
  assert.deepEqual(run.lock.build_inputs.env.filter(entry => entry.name === 'RUSTY_QUEST_NATIVE_RENDERER_PANEL_INITIALLY_CLOSED')
    .map(entry => entry.value), initiallyClosed ? ['true'] : []);
  assert.doesNotMatch(run.manifest, /debug-experiment-control|\.headset-setup/);
  assert.equal(run.lock.headset_setup_sources.length, 0);
  assert.equal(run.lock.debug_experiment_operator_sources.length, 0);
  assert.equal(run.lock.phone_pairing_sources.length, 5);
  for (const source of run.lock.phone_pairing_sources) {
    assert.equal(source.sha256, sha256(await readFile(join(questRepo, source.path))), source.path);
  }
}

test('assembly binds the real looping audio and selected Java definitions before packaging',
  { skip: !questRepo && 'Set ATMUNGSRAUM_QUEST_REPO to a patched native checkout.' }, async () => {
    const runName = `asm-test-${randomUUID().slice(0, 12)}`;
    const result = invoke(runName);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    const run = await resolvedRun(runName, 'atmungsraum_quest2');
    await assertControllerOnly(run);
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
    assert.doesNotMatch(run.manifest, /com\.oculus\.sdk\.appid|rusty\.quest\.meta_update\.enabled/);
    assert.ok(!run.lock.selected_feature_ids.includes('ui.experiment_boot_autostart'));
    assert.doesNotMatch(run.manifest, /NativeRendererBootReceiver|android\.intent\.action\.BOOT_COMPLETED|rusty\.quest\.experiment\.boot_autostart/);
    assert.match(run.manifest, /android:targetSdkVersion="35"/);
    assert.doesNotMatch(run.manifest, /android:installLocation=/);
    assert.match(run.manifest, /package="org\.viscereality\.atmungsraum\.rehearsal"/);
    assert.match(run.manifest, /android:name="rusty\.quest\.experiment\.breathing_setup\.enabled" android:value="true"/);
    await assertBackgroundMusic(run);
    await assertPhonePairing(run, false);
    const schemaCheck = spawnSync('pwsh', ['-NoProfile', '-Command',
      'if (!(Test-Json -Json (Get-Content -Raw -LiteralPath $env:ASM_TEST_SPEC) -SchemaFile $env:ASM_TEST_SCHEMA)) { throw "Assembled app violates schema" }; if (!(Test-Json -Json (Get-Content -Raw -LiteralPath $env:ASM_TEST_FEATURE) -SchemaFile $env:ASM_TEST_FEATURE_SCHEMA)) { throw "Scenario feature violates schema" }'],
    { encoding: 'utf8', env: { ...process.env, ASM_TEST_SPEC: run.specPath,
      ASM_TEST_SCHEMA: join(questRepo, 'schemas/rusty.quest.native_app_build.v1.schema.json'),
      ASM_TEST_FEATURE: join(run.specPath, '..', 'features/audio.atmungsraum_scenarios.feature.json'),
      ASM_TEST_FEATURE_SCHEMA: join(questRepo, 'schemas/rusty.quest.native_app_feature.v1.schema.json') } });
    assert.equal(schemaCheck.status, 0, schemaCheck.stdout + schemaCheck.stderr);
  });

test('rehearsal can opt into the same closed startup and phone pairing as developer',
  { skip: !questRepo && 'Set ATMUNGSRAUM_QUEST_REPO to a patched native checkout.' }, async () => {
    const runName = `asm-pair-${randomUUID().slice(0, 12)}`;
    const result = invoke(runName, ['-PhonePairing']);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    const run = await resolvedRun(runName, 'atmungsraum_quest2');
    assert.match(run.manifest, /package="org\.viscereality\.atmungsraum\.rehearsal"/);
    assert.match(run.manifest, /android:debuggable="true"/);
    await assertPhonePairing(run);
    await assertBackgroundMusic(run);
  });

test('authored-audio APK includes the separate phone-controlled music asset', {
  skip: !questRepo && 'Set ATMUNGSRAUM_QUEST_REPO to a patched native checkout.',
}, async () => {
  const runName = `asm-music-study-${randomUUID().slice(0, 10)}`;
  const result = invoke(runName, ['-MainStudyAudio']);
  assert.equal(result.status, 0, result.stdout + result.stderr);
  const run = await resolvedRun(runName, 'atmungsraum_quest2');
  await assertBackgroundMusic(run);
  await assertPhonePairing(run, false);
});

test('boot autostart opt-in resolves the actual own-app receiver and permission closure',
  { skip: !questRepo && 'Set ATMUNGSRAUM_QUEST_REPO to a patched native checkout.' }, async () => {
    const runName = `asm-boot-${randomUUID().slice(0, 12)}`;
    const result = invoke(runName, ['-BootAutostart']);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    const run = await resolvedRun(runName, 'atmungsraum_quest2');
    await assertControllerOnly(run);
    await assertBackgroundMusic(run);
    await assertPhonePairing(run, false);
    for (const feature of ['ui.experiment_boot_autostart', 'ui.same_apk_soft_kiosk', 'ui.experiment_session_ble_control']) {
      assert.ok(run.lock.selected_feature_ids.includes(feature), feature);
    }
    assert.ok(run.lock.android_manifest.receivers.includes('NativeRendererBootReceiver'));
    assert.match(run.manifest, /<receiver android:name="io\.github\.mesmerprism\.rustyquest\.native_renderer\.NativeRendererBootReceiver" android:exported="false">/);
    assert.match(run.manifest, /android:name="android\.intent\.action\.BOOT_COMPLETED"/);
    assert.match(run.manifest, /android:name="rusty\.quest\.experiment\.boot_autostart" android:value="true"/);
    for (const permission of ['android.permission.RECEIVE_BOOT_COMPLETED', 'android.permission.SYSTEM_ALERT_WINDOW']) {
      assert.ok(run.lock.android_manifest.permissions.includes(permission), permission);
      assert.ok(run.manifest.includes(`android:name="${permission}"`), permission);
    }
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
    await assertControllerOnly(run);
    assert.match(run.manifest, /package="org\.viscereality\.atmungsraum\.developer"/);
    assert.match(run.manifest, /android:label="Atmungsraum Developer"/);
    assert.match(run.manifest, /android:debuggable="false"/);
    assert.match(run.manifest, /android:targetSdkVersion="34"/);
    assert.match(run.manifest, /android:installLocation="auto"/);
    assert.equal(run.lock.android_manifest.debuggable, false);
    assert.equal(run.lock.android_manifest.target_sdk_version, 34);
    assert.equal(run.lock.android_manifest.install_location, 'auto');
    await assertPhonePairing(run);
    await assertBackgroundMusic(run);
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
