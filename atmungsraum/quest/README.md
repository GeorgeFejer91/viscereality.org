# Build Atmungsraum for Quest 2

See the [Quest APK log](apks/README.md) for the current signed build, its
source and checksum, and release history. Follow the
[APK commit workflow](../for-ai/WORKFLOW.md#quest-apk-changes-and-releases)
for every change to a signed Quest APK.

The shared-scheduler developer APK has passed software and packaging checks;
device acceptance is pending. The APK log records its exact source, signer
and checksum. The separate 0.1.30 rehearsal entry retains its original artifact,
source identity and verification history.

## Shared breathing setup

The selected Atmungsraum build uses one looping PCM AudioTrack. Its actual
playback-head frames own the cues, intended phases, automatic visual input,
calibration labels and main boundaries. Inhale, hold-in, exhale and hold-out
each last four seconds. Preparation has no time limit. The phone displays
Quest state and does not advance the schedule with a browser timer.

Automatic is the startup input. Polar calibration begins at the next inhale
boundary and collects two full cycles (32 seconds). Success requires accepted,
fresh directional samples across both cycles; elapsed time alone is not
success. A successful model becomes Polar tilt while the same loop continues.
Retry repeats calibration. Controller State with fixed orientation cancels an
outstanding calibration and uses the existing controller assessment without
restarting the audio. Fresh H10 streams remain required for the participant run.

The selected gateway offers only `haupt-15`. Start retains the chosen input
and model, waits for the next complete cycle boundary, and starts a separate
900-second main run: 39,690,000 audio frames at 44,100 Hz. Preparation is not
main data. The old introductory restart and embedded 15-second calibration
are absent from this path. The archived practice timelines, condition assets
and local calibration menu voices remain available to unselected builds;
they cannot start a second guidance player over the selected loop.

The selected path currently supports zero guidance bias. Nonzero Arm or
live-bias requests are explicitly rejected as
`setup-guidance-bias-unsupported`; no live-bias capability is advertised.
See the [approved setup contract](../for-ai/SHARED-BREATHING-SETUP.md) and
[recording clock rules](../for-ai/DATA-LOGGING.md#shared-scheduler-recording-boundary).

## Build from the public repository

Use Windows with PowerShell 7.6, Git, Node.js, JDK 17, Android SDK Platform 35,
Android Build Tools 36.0.0, Android NDK 27.2.12479018, and Rust with the
`aarch64-linux-android` target. Keep the native checkouts together in a short
ordinary directory. The following commands run from the `atmungsraum/quest`
directory in a complete Atmungsraum checkout.
Set the SDK and JDK paths to your installations.

```powershell
$native = '<absolute short path for the native sibling repositories>'
$env:JAVA_HOME = '<absolute path to JDK 17>'
$env:ANDROID_HOME = '<absolute path to the Android SDK>'
$developerKeystore = '<absolute path to the existing developer keystore outside Git>'
$env:ANDROID_NDK_HOME = Join-Path $env:ANDROID_HOME 'ndk/27.2.12479018'
$env:Path = "$env:JAVA_HOME/bin;$env:Path"
rustup target add aarch64-linux-android

$pins = [ordered]@{
    'rusty-quest' = '99352620fd79ec5cdcee71b47c2ba4056241af16'
    'rusty-lattice' = '2594b2bb0cd9e2e3593011f7dd09f68b59ea1989'
    'rusty-manifold' = '58e32fa1f098b58819a25e7d1fb51e22671a79db'
    'rusty-matter' = '54a18d85ee877f5cb6c8275013da8a76764b140e'
    'rusty-optics' = '23ce99bbb24ca90d7dcd1bf85dbda3a8603543c7'
}
New-Item -ItemType Directory -Force -Path $native | Out-Null
foreach ($name in $pins.Keys) {
    $dir = Join-Path $native $name
    git clone --filter=blob:none --no-checkout "https://github.com/MesmerPrism/$name.git" $dir
    if ($LASTEXITCODE) { throw "Could not clone $name" }
    git -C $dir checkout --detach $pins[$name]
    if ($LASTEXITCODE) { throw "Could not select the pinned revision for $name" }
}

$quest = Join-Path $native 'rusty-quest'
git -C $quest apply --index (Join-Path (Get-Location) 'native-source.patch')
if ($LASTEXITCODE) { throw 'Could not apply the Atmungsraum native patch' }
git -C $quest commit -m 'Apply Atmungsraum Quest 2 source'
if ($LASTEXITCODE) { throw 'A clean native commit is required for the locked build' }

./assemble-apk.ps1 -QuestRepo $quest -BundledAssets -RunName atmungsraum-developer-001 `
    -Developer -Keystore $developerKeystore -VersionCode 1 -VersionName 0.2.0-dev
```

Use your Git author identity for the local patch commit. The cumulative patch
reconstructs source tree `3df3966543186596cb3ab942eb9b29c87ee1cdc3`, matching
the software-verified developer build. Its SHA-256 and native commit are in
the APK log. The recorded pre-upload build used compile SDK 36 and targets
SDK 35. Meta upload validation requires the developer rebuild described below;
its updated source identity and artifact must be recorded after rebuilding.
The script stages the [shared loop JSON](../audio/box-loop.json), its four phase
cues, and the retained scenario library and calibration menu prompts. It
resolves the [bundled Experimenter inputs](build-assets/README.md) and verifies
the scenario asset inventory and ARM64 loader in the APK, then prints the APK
location and SHA-256. The output and its build receipts are under
`rusty-quest/target/aq2/atmungsraum-developer-001/`. Use a fresh run name for a
second build or add `-ResolveOnly` to check the feature lock without compiling.

Assembly binds `RUSTY_QUEST_NATIVE_RENDERER_BOX_LOOP_SHA256` to the exact
packaged `audio/box-loop.json`, selects
`rusty.quest.experiment.breathing_setup.enabled=true`, and includes the audio
owners in the locked Java source closure. Other native products retain their
unselected defaults.

## Developer and rehearsal signing

`-Developer` selects package `org.viscereality.atmungsraum.developer`, label
**Atmungsraum Developer**, `android:debuggable=false`, target SDK `34`, and
`android:installLocation="auto"`. Minimum SDK stays `29`; compile SDK `36`
remains supported. Meta requires apps created since 2026-03-01 to target
exactly SDK 34 and requires automatic install location. See the
[official release manifest requirements](https://developers.meta.com/vr/resources/publish-mobile-manifest/).
The developer flavor requires an explicit existing `-Keystore` outside Git
and an explicit `-VersionCode`,
including with `-ResolveOnly`. The recipe does not generate a developer key.
Keep one durable developer key for every later build. The initial empty
developer application can use version code `1` and name `0.2.0-dev`; every
later upload needs a higher code and the same signing key.

The existing native builder expects alias `androiddebugkey`, store password
`android` and key password `android` for an explicit keystore. This documents
its existing input contract; provision and retain the developer key separately
outside tracked source. Do not replace, regenerate or reuse the lab key for
the separate developer identity.

Without `-Developer`, the default remains
`org.viscereality.atmungsraum.rehearsal`, label **Atmungsraum Quest 2**, with
debugging enabled, target SDK `35` and no explicit install location. Its
existing signing identity must remain unchanged for in-place lab updates.
The default native development key can be created on
first use only in that rehearsal/development route. A new external key does
not become update-compatible by using the same package name. The two packages
can coexist and have separate controller enrollment and app data.

## Build with a separate Experimenter configuration

Use the Experimenter lane in MesmerPrism's shared Viscereality repository. Its
handoff describes the private provider and exact sibling source pins. Use the
consolidated [Rusty Quest source](https://github.com/MesmerPrism/rusty-quest/tree/99352620fd79ec5cdcee71b47c2ba4056241af16)
and apply [the native source patch](native-source.patch) to a clean checkout of
that revision with `git apply --index`. The patch is for the upstream
[AGPL-3.0 project](https://github.com/MesmerPrism/rusty-quest/blob/99352620fd79ec5cdcee71b47c2ba4056241af16/LICENSE).
The Experimenter build binding must point to the compatible consolidated
Rusty Quest revision when deriving the base inputs.

Run the Experimenter resolver with its condition-audio provider, Remote VR
preparation, command gateway and recording export. Keep the derived app spec,
its exact feature lock and its private provider manifest together. Stage the
pinned ARM64 liblsl via the Experimenter repository's
`tools/Stage-LibLslAndroid.ps1`. The assembly also needs JDK 17, Android SDK and
NDK, the Rust `aarch64-linux-android` target and the ARM64 OpenXR loader. Assembly selects the original native product panel so
the app starts in Remote control and advertises its Bluetooth discovery
service. The resolved manifest must select both the native panel host and
command gateway, and the feature lock must select the Remote default. The
Quest still requires trusted controller enrollment before private access.
Use the headset maintenance panel to switch to Direct for the current app session.
Use a native build checkout whose path ancestors are ordinary directories so
the packager can verify its output root.

From this directory, resolve and assemble a new run using PowerShell 7.6
and Node.js.

```powershell
./assemble-apk.ps1 -QuestRepo $questRepo -ExperimentAppSpec $appSpec `
  -ExperimentFeatureLock $featureLock -PrivateAssetProviderManifest $provider `
  -OpenXrLoader $arm64OpenXrLoader -RunName quest2-rehearsal `
  -VersionCode $versionCode -VersionName $versionName -Keystore $keystore
```

Use `-ResolveOnly` for a feature-lock check. Without `-Developer`, the build uses
`org.viscereality.atmungsraum.rehearsal`, targets Quest 2 and admits
`https://viscereality.org` to the native gateway. It verifies each scenario
file in the signed APK by size and SHA-256. ECG previews and recording replies
are encrypted on the enrolled LAN route. Keep the private provider, signing
inputs, APK and device evidence with the native release work.

Add `-Developer` with its durable key and explicit version code to use the
separate developer identity with custom Experimenter inputs.

On startup, the Quest APK requests Android’s turn-on dialog if its Bluetooth
radio is off. The provisioned PWA reconnects using the saved connection. Tap
**Mit Bluetooth verbinden** to rediscover the headset when needed. Chrome
provides the device chooser. The PWA retrieves the identity and gateway
addresses and verifies the signed status with the saved trusted enrollment.

When the phone already has hotspot settings and its hotspot is running, the
PWA tries the shared LAN first and uses the encrypted Bluetooth setup
characteristic if needed. Accept Android’s join prompt in the headset if
shown. The browser continues over the authenticated Wi-Fi gateway after the
headset reports a local address. See the [automatic connection checklist](../for-ai/PROJECT.md#automatic-quest-connection--2026-11-14)
for the remaining first-use provisioning and automatic hotspot work.
