# Atmungsraum Quest 2 APKs

## 0.2.0-dev — separate developer build, 2026-11-14

| Build property | Verified value |
| --- | --- |
| Package / platform | `org.viscereality.atmungsraum.developer`, Quest 2, `arm64-v8a` |
| Release | 0.2.0-dev, Android version code 1 |
| Signed APK | Kept in private developer distribution; not bundled in this repository |
| Size | 65,264,431 bytes |
| SHA-256 | `1a5470f097a06578fc3b1d58e64787b5c586e0204cc0409ce26d5fe04ce3036b` |
| Signing certificate SHA-256 | `372324e933ce701b9e71ff1a24239aeba8589231fa54e298c424e1f1a2cf570e` |
| Native commit | `e8e7715be2ca253d50515878ce3c973ba7636c93` |
| Native source tree | `5962359ba086897f7bf1e9ef0c40568c95010cf5` |
| Upstream Rusty Quest | `99352620fd79ec5cdcee71b47c2ba4056241af16` |
| Atmungsraum source | [1dfa5f0](https://github.com/GeorgeFejer91/viscereality.org/commit/1dfa5f0e16249db2dffebbda8d508d45d4bf1144) |
| Source patch SHA-256 | `fa5905c0a1f923989d2544d34bef01bb7b5b44acd2e1277610567284b458317d` |
| Android packaging | Debugging disabled; min SDK 29, target SDK 34, compile SDK 36, install location auto; v2 and v3 signatures verified |

This build adds one actual audio-frame clock, Automatic startup, unlimited
4–4–4–4 preparation, two-cycle measured tilt calibration, retry and fixed
Controller State fallback. The chosen input continues into a separate,
exactly 900-second main at the next full cycle boundary. Required study H10
freshness and submitted-frame checks remain active in Controller mode. A
held incomplete run cannot claim completion by adopting a later audio head.
Retry cannot reuse an earlier model's classified phase or source timestamp.
Nonzero guidance bias is explicitly unsupported in this selected path.

The locked build verified all 74 catalog/audio files and the ARM64 OpenXR
loader. All five shared-loop JNI exports are present. The cumulative public
patch applies to the pinned upstream and reconstructs the native source tree
above exactly. Native host tests passed 798 with three ignored; all 116 Node
checks, 59 rendered browser states, context checks and scoped code reviews
passed. The Android compiler regression uses the real SDK boot class path.

This is software/build evidence. Worn-Polar calibration quality, actual
audio/visual alignment, a full 900-second Quest 2 trial, installed Pixel
refresh and remote version-to-version replacement remain pending at the
user's request to finish the build without hardware testing now. The
private Meta Developer upload was accepted and passed malware and security
scans. Owner enrollment awaits the account owner's invite consent; remote
installation and version-to-version replacement are not claimed. The developer package
and signing identity are separate from the rehearsal app and preserve its
installation and data. Keep the developer signing key for future updates.

## 0.1.30 — Quest 2 calibration menu

| Build property | Verified value |
| --- | --- |
| Package / platform | `org.viscereality.atmungsraum.rehearsal`, Quest 2, `arm64-v8a` |
| Release | 0.1.30, Android version code 30 |
| Signed APK | [Atmungsraum-Quest2-0.1.30.apk](Atmungsraum-Quest2-0.1.30.apk) |
| Size | 63,760,629 bytes |
| SHA-256 | `580edecca5a51bc2653d561f08dc58c3621df87bc2d06efd7cfe8262c34c1be7` |
| Signing certificate SHA-256 | `6c9b7c04bfbe88cae2509ad1533e6014ebb6abc97d855c666d687eee56a57f7d` (same as 0.1.29) |
| Native commit | `ac5be2669cace0e75146ae3bae90a3026ff3f8d5` |
| Native source tree | `170d6c3def526684d2b27ef8f54c3c3a0e3e8d7a` |
| Upstream Rusty Quest | `99352620fd79ec5cdcee71b47c2ba4056241af16` |
| Atmungsraum source | [b360e99](https://github.com/GeorgeFejer91/viscereality.org/commit/b360e99) |

**Direct → Developer settings → Breath → Calibration** now plays the approved
German inhale cue after an accepted Start and the exhale cue four seconds later
while that calibration is active. Cancel, Reset and leaving the menu release
playback. The two [Secret Tunnel clips and transcripts](../../assets/audio/calibration-menu/manifest.json)
are independent of the six scenario timelines and the phone breathing preview.

The reconstructed native tree, focused Java playback and existing audio tests,
91 Node tests, six scenario pairs, and project context passed. The Android SDK
verified the signature, manifest, package, version and ABI. All 68 files in
the audio catalog matched, including both stored MP3s. The original 66
scenario files and ARM64 OpenXR loader matched 0.1.29 byte for byte. The broader
native panel check still stops at a source assertion already false in 0.1.29.
An attended listening check on Quest 2 remains open.

## 0.1.29 — Quest 2 rehearsal, 2026-11-14

| Build property | Verified value |
| --- | --- |
| Package / platform | `org.viscereality.atmungsraum.rehearsal`, Quest 2, `arm64-v8a` |
| Release | 0.1.29, Android version code 29 |
| Signed APK | [`Atmungsraum-Quest2-0.1.29.apk`](Atmungsraum-Quest2-0.1.29.apk) |
| Size | 63,723,545 bytes |
| SHA-256 | `755ca38022dc7c35d37d3c9e3739d58fe3145872721e37a70d8b59a26a49e734` |
| Signing certificate SHA-256 | `6c9b7c04bfbe88cae2509ad1533e6014ebb6abc97d855c666d687eee56a57f7d` |
| Native commit | `bdfbf8d6a4b5da7776b8b6aebd8de4baedbc6963` |
| Native source tree | `cd2d5a5837d2c642a70ba6b1cf7256f14ebd7969` |
| Upstream Rusty Quest | `99352620fd79ec5cdcee71b47c2ba4056241af16` |
| Atmungsraum source | [`1843da3`](https://github.com/GeorgeFejer91/viscereality.org/commit/1843da3), with the native source patch and assembly recipe |
| Tested Pixel PWA | `20261114-recording-bytes` |

The signed build uses linked Experimenter condition audio, breathing
guidance, the session profile and the Viscereality particle payload. The
repository includes the [build assets](../build-assets/README.md),
[source patch](../native-source.patch) and [assembly recipe](../README.md).
The original signed APK and its capsule are under
`target/aq2/live-recording-029/p-e2740b20fd5b713f/` in the native build
checkout. The capsule records the exact feature lock and sibling dependency
pins. Applying the public patch to the upstream revision above reconstructs
the same native Git tree as the signed build.

### Changes since 0.1.28

The preceding APK was built from native commit
`84bcf7fdf8faaf8cb6214b9ad6f816710f9965ca`. The complete native diff for
0.1.29 spans 11 files and is recorded in the patch at `1843da3`.

- A separate Quest worker projects durable native checkpoints into an
  append-only CSV v4. Measurement rows retain ECG, acceleration, breathing
  phase, applied radius and per-stream signal markers. Explicit playback
  transition rows let consumers reconstruct the audio clock even when a
  pause or a sensor observation arrives late.
- Each committed prefix has a length, SHA-256 and checkpoint ordinal. Finish
  appends the final observations and one terminal row in the same file.
  Checkpoint recovery replays the v4 boundaries and retains older finalized
  CSV formats.
- The native gateway serves the active recording descriptor and bounded
  ranges only to the current enrolled LAN controller. It rejects guest and
  relay access, reads beyond the committed prefix, and reads for a retired
  run. Idle export continues through the finalized recording interface.
- Native recording and Android gateway tests cover prefix stability with
  late observations and backdated Pause, checkpoint recovery, committed
  range bounds and owner replacement. The build passed 78 native recording
  tests and 86 gateway checks.

### Hardware verification

Android SDK tools verified the APK signature, matching signer on 0.1.28,
package, version and ABI. The APK contains the ARM64 OpenXR loader. The
installed Quest base APK matched the signed SHA-256. The qualified
15-minute physical trial verified local Polar-driven scene response,
recording through upload disruptions and final delivery of one matching CSV
across Quest, the installed Pixel PWA and private Drive. See the
[collection procedure and trial results](../../for-ai/QUEST-PHONE-SYNC.md#fresh-polar-response-on-the-updated-pwa).

### External build verification

The public recipe compiled and signed a fresh APK from clean clones at the
pinned revisions. Its 67 scenario files, five provider assets and two ARM64
libraries matched the released APK byte-for-byte. Android SDK tools verified
the rebuilt package, ABI and independent development signature. The
headset trial above used the released lab-signed APK.

Record subsequent releases here using the
[APK commit workflow](../../for-ai/WORKFLOW.md#quest-apk-changes-and-releases).
