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
NDK, the Rust `aarch64-linux-android` target and the ARM64 OpenXR loader.
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

The native audio worker verifies all six scenario timelines and their packaged
cue files before rehearsal audio reports ready. In an enrolled VR-Probe
session, choose the VR condition and the authored timeline under **Ablauf**.
The signed Arm request binds the selection to the observed native generation.
The Quest audio owner prepares the timeline from its packaged WAVs and plays
it alongside the selected condition track. Both tracks follow the native
Start, Pause, Resume and Finish transitions.

For final recordings, follow the [scenario asset contract](../audio/scenarios/README.md)
to replace each numbered WAV and refresh its metadata before building again.
See [For-AI](../for-ai/PROJECT.md#quest-2-control-media-and-ui-acceptance--2026-11-12)
for physical qualification and current release evidence.
