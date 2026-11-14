# Atmungsraum Quest 2 APKs

## 0.1.29 — Quest 2 rehearsal, 2026-11-14

| Build property | Verified value |
| --- | --- |
| Package / platform | `org.viscereality.atmungsraum.rehearsal`, Quest 2, `arm64-v8a` |
| Release | 0.1.29, Android version code 29 |
| Signed APK | `atmungsraum/output/quest-apks/Atmungsraum-Quest2-0.1.29.apk` on the lab PC |
| Size | 63,723,545 bytes |
| SHA-256 | `755ca38022dc7c35d37d3c9e3739d58fe3145872721e37a70d8b59a26a49e734` |
| Signing certificate SHA-256 | `6c9b7c04bfbe88cae2509ad1533e6014ebb6abc97d855c666d687eee56a57f7d` |
| Native commit | `bdfbf8d6a4b5da7776b8b6aebd8de4baedbc6963` |
| Native source tree | `cd2d5a5837d2c642a70ba6b1cf7256f14ebd7969` |
| Upstream Rusty Quest | `99352620fd79ec5cdcee71b47c2ba4056241af16` |
| Atmungsraum source | [`1843da3`](https://github.com/GeorgeFejer91/viscereality.org/commit/1843da3), with the native source patch and assembly recipe |
| Tested Pixel PWA | `20261114-recording-bytes` |

The lab build uses linked Experimenter condition audio, breathing guidance
and the session profile. It also includes the private particle payload.
The public Git tree carries the [source patch](../native-source.patch),
[assembly recipe](../README.md) and verification evidence.
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

Record subsequent releases here using the
[APK commit workflow](../../for-ai/WORKFLOW.md#quest-apk-changes-and-releases).
