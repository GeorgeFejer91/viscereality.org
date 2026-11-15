# Atmungsraum Quest 2 APKs

## 0.2.6-dev — participant-led calibration and controllers only, 2026-11-15

| Build property | Verified value |
| --- | --- |
| Package / ABI | `org.viscereality.atmungsraum.developer`, `arm64-v8a` |
| Release | 0.2.6-dev, version code 7; local hardware testing only |
| APK size / SHA-256 | 65,264,431 bytes; `613e536034e720ec0528dd933e22d4bb5bf736d267e451113db58f624c81402d` |
| Certificate SHA-256 | `372324e933ce701b9e71ff1a24239aeba8589231fa54e298c424e1f1a2cf570e` |
| Native commit / tree | `50eaa1ac55eb9161879c48916da82894cfaff9af` / `0d6e505055a77611b2f46a4b50d50d32ce8bc95d` |
| Upstream | `99352620fd79ec5cdcee71b47c2ba4056241af16`, unchanged sibling pins |
| Source patch SHA-256 | `7c27f93464c20dcc7500eb2fe4e784910f47aa616dff659ca4dce731be0fa725` |
| Packaging | Non-debuggable; min 29, target 34, compile 36, automatic install; v2/v3 and 16KiB ZIP alignment checks pass |

Calibration assumes cue adherence and learns the participant's tilt axis,
direction and measured range. Ideal correlation, RMSE, flat-hold and ramp-shape
checks are removed, without introducing recurrence/compliance rejection. Technical
quality, usable coverage and finite mapping remain required. The existing measured
span 0.02–0.50 remains an engineering gain/noise limit; it can reject smaller real
movement and is not a validated physiological cutoff. Participant testing of the
resulting Polar-driven sphere decides satisfaction and whether to repeat setup.

The app now selects the existing locked controller-only policy, strips hand
manifest support and rejects conflicting hand/simultaneous features. Runtime hand
settings and assertions are disabled, while controller breathing and the optional
base input substrate remain. A legacy original-renderer build guard now admits
exactly one locked input mode, preserving its renderer/background/panel checks.
No global headset hand-tracking setting is changed. Private visuals and the
shared 900-second experiment remain unchanged.

Verification: new waveform tests first failed then passed; core50, setup72 and
full native824 pass, three ignored. Actual default/selected resolver, compiled
input/extension selectors, 32 panel controls, real non-resolve builder preflight,
both product assembly checks and independent reviews pass. The first assembly
attempt exposed the legacy input guard before compilation; fresh assembly after
the reviewed fix succeeds. Exact patch/tree reconstruction, signed-APK session
closure, 89 assets, signer, manifest and ZIP checks pass.

Code 7 is installed with matching installed-byte hash, retained UID/grants and
persistent app identity. Actual startup reports `nativeControllerOnlyInput=true`,
`liveHandTrackingExtensionEnabled=false`, `simultaneousHandsControllersSelected=false`
and closed app-panel visibility. The gateway is ready. Location remains enabled
at the user's request. An unworn-headset capture is black and does not establish
wearer rendering. Fresh personal calibration, satisfactory tracking, retry,
Controller fallback and the full trial remain open. No Meta upload occurred.

## 0.2.5-dev — parked startup and full-window quality, 2026-11-15

| Build property | Verified value |
| --- | --- |
| Package / ABI | `org.viscereality.atmungsraum.developer`, `arm64-v8a` |
| Release | 0.2.5-dev, version code 6; local hardware testing only |
| APK size / SHA-256 | 65,280,816 bytes; `25a1b9464bc63f2e9c349a8c899c34ccec84c881eaabc8e6565fc7d7b260c2a8` |
| Certificate SHA-256 | `372324e933ce701b9e71ff1a24239aeba8589231fa54e298c424e1f1a2cf570e` |
| Native commit / tree | `f31156d7832f71b7caeff9ad005c311ea77c0c8e` / `c095a8df92480b6da4c7810ed3332bf3998667c2` |
| Upstream | `99352620fd79ec5cdcee71b47c2ba4056241af16`, unchanged sibling pins |
| Source patch SHA-256 | `eb5f53d397089478f2732ee889c0637ae67ad0fa8b5f9db0524ed8828f02c661` |
| Packaging | Non-debuggable; min 29, target 34, compile 36, automatic install; v2/v3 and 16KiB ZIP alignment checks pass |

Automatic is parked at minimum with prepared silent audio. Accepted calibration
or Controller selection starts the shared loop once; ordinary lifecycle resume
cannot first-start it. Calibration assesses all 32 seconds, masks bounded noisy
segments and fits only with sufficient phase coverage. Fixed sensor/receipt-anchor
estimation avoids callback-derived host reordering, with explicit non-exact
timing provenance and unchanged raw timestamps. Source-clock warmup gates initial
calibration; Controller remains available. See the
[quality and timing contract](../../for-ai/SHARED-BREATHING-SETUP.md).

Connected Polar acquisition establishes and recovers ECG and motion automatically.
Explicit Stop, source ownership and uncertain START outcomes remain fenced. A
single bounded raw-frame diagnostic is restricted to authenticated private LAN;
relay and guest responses exclude it. The current phone release smooths the
percentage and displays native warmup and quality warnings. Private Showcase
visual/coupling inputs, zero tracers/indicator and synthetic heartbeat default
remain identical to code 5.

Checks: 69 focused setup tests; 821 full native passes, three ignored, no failures;
three selected profile and one reset regression; recovery47, source-scope24,
diagnostic15 and full gateway suites; actual Android API-36 compilation and
independent reviews. The cumulative patch reconstructs the exact native tree.
Signed-APK session closure, 89 assets, manifest, signer and ZIP alignment pass.

Code 6 is installed on Quest 2 with exact installed-byte hash readback, retained
UID, grants and persistent app identity. Native frames advance, effective app-panel
visibility is false, and the HTTP gateway is ready. The known Location handoff
can leave Meta system Settings over VR; this is separate from the closed app panel.
System Location remains OFF. Fresh wearer confirmation of silence/minimum startup,
actual ACC format, advancing calibration, fit/retry/fallback and the full trial
remain pending. No calibration was started and no Meta upload was performed.

## 0.2.4-dev — combined Showcase particle update, 2026-11-15

| Build property | Verified value |
| --- | --- |
| Package / ABI | `org.viscereality.atmungsraum.developer`, `arm64-v8a` |
| Release | 0.2.4-dev, version code 5; local hardware testing only |
| APK size / SHA-256 | 65,276,717 bytes; `a54fce521452a6f0f6afdaaed3a081267a2b4285510232eb6f9d96cf84dc4f2a` |
| Certificate SHA-256 | `372324e933ce701b9e71ff1a24239aeba8589231fa54e298c424e1f1a2cf570e` |
| Native commit / tree | `6cd05d6a511f02db426bcdcf8808d662639bd034` / `f78d4cad8b9c0e4521f98ad9993d675b5f2866ab` |
| Upstream | `99352620fd79ec5cdcee71b47c2ba4056241af16`, unchanged sibling pins |
| Source patch SHA-256 | `ac5082fd2605b92ca7a9c116a0b3eee5b52243297b796a5f549d4323c568ad3d` |
| Packaging | Non-debuggable; min 29, target 34, compile 36, automatic install; v2/v3 and 16KiB ZIP alignment checks pass |

The generic opt-in particle adapter is layered on the existing calibration,
buffered-timing and permission source. It uses the full private Showcase visual
and coupling profile while retaining experiment input, audio, recording and
session authority. Experiment tracer drawing and the separate white pole/
retention indicator are zero. The packaged profile survives experiment resets.
Its preserved Showcase heartbeat default is synthetic at 60 BPM; a real Polar
heartbeat selection is not implied by this visual port. Private profile bodies
and shaders remain outside the public website repository.

The selected particle profile digest is
`2c5ce38b5403273d73f9a46444d012c460b0596177d95994985023c79aa17e35`;
the compiled coupling payload matches the retained Showcase baseline. The
private wrapper's late undeclared LSL path override was removed after an actual
build reproduced its rejection. The repaired wrapper verifies the documented
staged library before using the existing locked assembly; ambient-input checks
remain enforced.

The native suite passed 812 tests with three ignored, including actual-profile
and reset checks. Permission diagnostics passed 71 controls, three negative
controls, approval UI45, projection24 and actual API-36 compilation. Independent
reviews found no remaining P1/P2 source issue. Actual signed-APK session closure,
source-tree reconstruction, same signer and manifest/library checks pass.

Code 5 is installed on Quest 2 with exact byte readback and the retained app UID,
grants and memberships. Cold startup shows VR with the app panel closed and a
ready gateway. Live startup capture proves that permission coordination runs
while closed, finds all runtime grants, and dispatches Location recovery; Meta's
relay returns with Location still disabled. This is a known recovery limitation,
not a passed Location UI test. System Location remains at its original OFF
setting. The wearer fit/retry/fallback, full trial, visual parity/performance and
installed-phone refresh remain pending. No calibration has started collecting.
Code 5 has not been uploaded to Meta.

## 0.2.3-dev — local permission and timing repair, 2026-11-15

| Build property | Verified value |
| --- | --- |
| Package / ABI | `org.viscereality.atmungsraum.developer`, `arm64-v8a` |
| Release | 0.2.3-dev, version code 4; local hardware testing only |
| APK size / SHA-256 | 65,297,199 bytes; `7a5afc02e0895357e929ccbf4857a79232c4da64eb7e04d4c803ee05eb52a4bf` |
| Certificate SHA-256 | `372324e933ce701b9e71ff1a24239aeba8589231fa54e298c424e1f1a2cf570e` |
| Native commit / tree | `ff36ec42f4f0e0023ba68086cf6475a15c2ff91f` / `922b085585263bf68aebbd91d3011217aaf09d11` |
| Upstream | `99352620fd79ec5cdcee71b47c2ba4056241af16`, unchanged sibling pins |
| Source patch SHA-256 | `ea2426cc58bbbcfc840d77c0202899e544e1fc7cec90a4b1a922b9bb9c9a9ab0` |
| Packaging | Non-debuggable; min 29, target 34, compile 36, automatic install, v2/v3 signatures |

One selected startup coordinator handles the three existing permission
requesters. It requests the declared runtime union once, then uses an attended
Location settings surface when required. No app approval panel or programmatic
Location toggle is involved. The five-source closure and optional phone `y.l`
readiness field are bound into the build. Waiting phones receive the current
claim-control hint rather than a stale earlier action message.

The native setup owner no longer advances its control revision for unused
idle Automatic tilt diagnostics. Valid buffered samples map into bounded
actual audio history; future training labels wait for a real bracket, and
fitting requires final-window evidence. Sensor gap, driver freshness and drift
guards remain separate from delivery tolerance. Original acquisition timestamps
are unchanged. See the [timing contract](../../for-ai/SHARED-BREATHING-SETUP.md#buffered-polar-timing--2026-11-15).

Full native tests passed 809 with three ignored and zero failures; the focused
setup gate passed 63, Node passed 131 with zero skips, and rendered control/
Location checks passed 18. Permission lifecycle, approval UI, API-36 compilation,
five-source selection and independent reviews passed. The cumulative patch
reconstructed the exact source tree; assembly verified all 74 audio/catalog
files, the ARM64 loader and actual signed-APK session closure.

Installed bytes matched on Quest 2. Cold startup visibly rendered VR with the
panel closed and the gateway ready. Existing grants and phone memberships were
preserved. Wearer calibration, fresh permission dialogs, callback timing and
the full trial remain pending. This APK is withheld from Meta until relevant
local hardware testing passes, as requested by the user.

## 0.2.2-dev — button-only phone pairing, 2026-11-15

| Build property | Verified value |
| --- | --- |
| Package / platform | `org.viscereality.atmungsraum.developer`, Quest 2, `arm64-v8a` |
| Release | 0.2.2-dev, Android version code 3 |
| Signed APK | Private developer distribution; not bundled in this repository |
| Size | 65,293,101 bytes |
| SHA-256 | `86b8e0fa5de95fbde47cd934552f4200352f50fa9a8c32410e6498f499ea6fa9` |
| Signing certificate SHA-256 | `372324e933ce701b9e71ff1a24239aeba8589231fa54e298c424e1f1a2cf570e` |
| Native commit | `603a432db108220f1270695bd0bbda07a0d86371` |
| Native source tree | `8bbc4937ab0fd9ab97d0d6c751dc624260ec0dbe` |
| Upstream Rusty Quest | `99352620fd79ec5cdcee71b47c2ba4056241af16` |
| Source patch SHA-256 | `5059f3c5ac0ba9c20a7d7075ba007cc1a94a99b4a6a4777cd3a3475dc36df9c5` |
| Packaging | Non-debuggable; min 29, target 34, compile 36; automatic install; v2/v3 signatures verified |

The Developer app starts in VR with its panel closed. A verified incoming
phone request opens a matching-code Allow/Reject prompt; no headset typing is
required. Cancellation and expiry dismiss only the prompt-owned opening.
Triple-B still opens the regular app panel. Read-only Bluetooth discovery uses
explicit app-installation identity when hardware serial is unavailable. Up to
eight independent phones can be saved and individually revoked; one explicitly
claimed controller lease admits experiment mutations. Saved reconnection never
claims control. Selected mode rejects public legacy BLE mutations while
retaining its internal receipt carrier. The machine provisioning adapter and
broad debug operator remain unselected.

Pairing v2 uses ordered commitment/reveal, canonical nonce encoding and both
human confirmations before storing membership. Its production Java/browser
exchange and adversarial transcript checks passed. All 130 Node checks,
369 focused crypto checks, gateway/registry/discovery suites, 45 native UI
controls, actual API-36 compilation, source selection and shell policy passed.
Chromium passed 78 screens plus 12 focused final control screens. Independent
source review closed the initial unshipped protocol and BLE authority findings.
See the [exact contract](../../for-ai/PHONE-PAIRING.md).

The cumulative patch reconstructed the exact tree above. Assembly verified the
74 catalog/audio files, ARM64 loader and signed-APK cold-start closure. The
same-key update installed with exact byte readback over Wi-Fi ADB. Actual
Quest 2 startup rendered VR with the panel closed; its public gateway reported
v2 app-installation discovery without a hardware serial. The refreshed Pixel
found the Quest; the user observed matching codes and confirmed on both
devices. Saved reconnection, two-phone handover/revocation, worn calibration
and the full 900-second trial remain pending. Meta accepted code 3 as private
Developer build `1460570757141936`; malware/security scans passed and the owner
is joined. The Store remains unpublished. This device update used ADB and does
not prove remote replacement. Subsequent H10 setup found that Bluetooth Scan
was not requested at startup and had no visible native request option. The
user authorized a one-permission pre-grant on this test installation; readback
confirmed it. Fresh-install startup permissions still require a fix. Further
iterative APKs must be tested locally before Meta submission.

## 0.2.1-dev — cold-start repair, 2026-11-15

Private Developer package `org.viscereality.atmungsraum.developer`, version code
2, uses the same certificate as code 1. APK size is 65,264,429 bytes; SHA-256
`0ee0677944225910dfdd733a5397e746af83d7209870a169e8014e1e0a7347e5`.
Native source is `252f8120cb518763b070b79519175470f1529398`, tree
`fa0e7626958a67850056286effda6fea34a100cd`, against the unchanged upstream pin.
Its cumulative source-patch digest was
`1d2276e347b76aaa67b2bed1ad9ea6740400fd6ed8a7fb84e308ff2135119ead`.
SDKs, ABI, install location, signing identity and audio inventory are unchanged.

An actual code-1 cold start exposed a validator that still required the legacy
30-second runtime. The repair follows the strict selected boolean: exactly
900 seconds for shared breathing setup, 30 seconds for the legacy profile.
The new host probe reads the signed APK's actual assets using the production
Java validator. The probe, shell policy, signatures, manifest, loader and all
74 catalog/audio files passed. Installed bytes matched on Quest 2; foreground
OpenXR rendering at approximately 72 frames/s was observed.

This local ADB update was not uploaded to Meta. It exposed a separate fresh-app
onboarding gap: hardware serial was unavailable, so discovery would not
advertise. Wearer calibration and phone pairing remained pending. The new
phone-pairing update addresses that gap with explicit app-installation identity.

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
