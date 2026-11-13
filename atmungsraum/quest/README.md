# Build Atmungsraum for Quest 2

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
the app can enter Remote control and advertise its Bluetooth discovery
service. The resolved manifest must select both the native panel host and
command gateway.
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
radio is off. On first setup, **Quest verbinden** opens Chrome's Bluetooth
chooser. Enter the enrolled controller credentials from the headset's network
panel once. VR-Probe learns the serial and signed build and saves the pairing.

To connect a Quest 2 to a Pixel hotspot, turn on the phone’s Wi-Fi hotspot,
open **Smartphone-Hotspot verwenden** during setup, and enter the hotspot
name and password once.
The browser verifies the selected serial and encrypts the credentials for
that headset before sending them. Accept Android’s join prompt in the
headset if shown. The controller continues over the authenticated Wi-Fi
gateway after the headset reports a local address. The connection belongs
to the active Quest app and is released when Remote mode or its owner ends.
The phone reuses the saved settings when reconnecting.

The enrolled phone can recenter the app scene while Quest is in immersive
Remote mode. Quest uses its current eye poses and confirms the action after
submitting the recentered scene in a visible OpenXR frame.

The native audio worker verifies all six scenario timelines and their packaged
cue files before rehearsal audio reports ready. In an enrolled VR-Probe
session, choose the VR condition and the authored timeline under **Ablauf**.
The signed Arm request binds the selection to the observed native generation.
The Quest audio owner prepares the timeline from its packaged WAVs and plays
it alongside the selected condition track. Both tracks follow the native
Start, Pause, Resume and Finish transitions.

For Quest 2 playback qualification, capture the native player receipts for
the test run with `adb -s <quest-serial> logcat -d -s RustyQuestNativeRenderer:I`.
The `channel=condition-audio` lines carry a run generation, receipt revision,
Android monotonic time in nanoseconds and the player's position in milliseconds.
Use one generation to check that position advances through 26,000–41,000 ms
in the main round, holds during a pause and continues from the retained
position after resume. Capture the phone and headset at matching transitions.

For participant recording, the enrolled phone sends an encrypted Arm request
after the H10 reports fresh ECG and the headset is in idle VR preparation.
The app reads the selected plan from the verified audio package. Quest
reserves the attempt before recording and writes the CSV from its ECG and
playback events. A finished run and a run ended early through the app retain
their original filenames and outcomes. The phone verifies each downloaded
file against the signed export descriptor. After an interrupted run, the app
can recover the CSV and durable stream prefixes from the last verified checkpoint
while retaining the original files on Quest.

For final recordings, follow the [scenario asset contract](../audio/scenarios/README.md)
to replace each numbered WAV and refresh its metadata before building again.
See [For-AI](../for-ai/PROJECT.md#quest-2-control-media-and-ui-acceptance--2026-11-12)
for physical qualification and current release evidence.
