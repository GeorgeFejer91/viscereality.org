# Build Atmungsraum for Quest 2

See the [Quest APK log](apks/README.md) for the current signed build, its
source and checksum, and release history. Follow the
[APK commit workflow](../for-ai/WORKFLOW.md#quest-apk-changes-and-releases)
for every change to a signed Quest APK.

## Build from the public repository

Use Windows with PowerShell 7.6, Git, Node.js, JDK 17, Android SDK Platform 35,
Android Build Tools 36.0.0, Android NDK 27.2.12479018, and Rust with the
`aarch64-linux-android` target. Keep the native checkouts together in a short
ordinary directory such as `C:/src/atmungsraum-native`. The following commands
run from the `atmungsraum/quest` directory in a complete Atmungsraum checkout.
Set the SDK and JDK paths to your installations.

```powershell
$native = 'C:/src/atmungsraum-native'
$env:JAVA_HOME = 'C:/tools/jdk-17'
$env:ANDROID_HOME = 'C:/tools/android-sdk'
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

./assemble-apk.ps1 -QuestRepo $quest -BundledAssets -RunName atmungsraum-030 `
    -VersionCode 30 -VersionName 0.1.30
```

Use your Git author identity for the local patch commit. The current patch
reconstructs source tree `170d6c3def526684d2b27ef8f54c3c3a0e3e8d7a`.
The script stages the six authored scenarios and two calibration menu prompts
from `../assets/audio/` and
resolves the [bundled Experimenter inputs](build-assets/README.md). It verifies
the scenario asset inventory and ARM64 loader in the APK, then prints the APK
location and SHA-256. The output and its build receipts are under
`rusty-quest/target/aq2/atmungsraum-030/`. Use a fresh run name for a second build
or add `-ResolveOnly` to check the feature lock without compiling.

The native packager creates a local development keystore on first use. An
external build has its own signature and build identity. Use a clean
development Quest profile or an appropriately signed in-place update for the
same Android package.

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

Use `-ResolveOnly` for a feature-lock check. The build uses
`org.viscereality.atmungsraum.rehearsal`, targets Quest 2 and admits
`https://viscereality.org` to the native gateway. It verifies each scenario
file in the signed APK by size and SHA-256. ECG previews and recording replies
are encrypted on the enrolled LAN route. Keep the private provider, signing
inputs, APK and device evidence with the native release work.

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
