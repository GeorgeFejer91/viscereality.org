# Build Atmungsraum for Quest 2

## Optional background music

Apply [background-music.patch](background-music.patch) after the Developer 32
source overlays. Every Atmungsraum APK assembles with phone enrollment and
packages the supplied `../assets/audio/background/ji3ji3.mp3` under
`assets/atmungsraum-music/`. The assembly verifies the asset by hash in the
signed APK. Developer and WebRTC bench builds start with the panel closed.
Rehearsal keeps its open panel by default and can still select closed
startup with `-PhonePairing`. The optional `-MainStudyAudio` build keeps the
authored session mix and offers the same independent operator-controlled loop.

On the phone, open **Einstellungen → Hintergrundmusik**. The Quest starts
each native run with music off and volume at 35%. Play loops the track in the
APK. Stop releases the player. The volume slider changes the music gain
independently of the breathing guidance. The connected phone with active
control can change playback and volume. Music can keep playing through a
phone reconnect while the native owner remains active.

See the [Quest APK log](apks/README.md) for the current signed build, its
source and checksum, and release history. Follow the
[APK commit workflow](../for-ai/WORKFLOW.md#quest-apk-changes-and-releases)
for every change to a signed Quest APK.

The installed Meta Developer baseline is code33. Its source, signer,
checksum and remaining physical acceptance are recorded in the APK log.
The historical 0.1.30 rehearsal entry retains its original artifact and
verification history.

## Canonical Quest Developer APK

Use `assemble-apk.ps1 -CanonicalRelease` for the single final experiment
APK, `org.viscereality.atmungsraum.developer`. Supply the retained private
app spec, feature lock and asset provider, the existing Developer keystore
and OpenXR loader, and the Meta SDK inputs described below. Set a fresh run
name and the release owner's explicit version code and name after checking
the current Meta channel and public release policy. The profile selects
WebRTC, phone enrollment, Meta updates and the installed helper handoff.
It verifies the actual APK's v3 certificate against the established
Developer signer.

The [connectivity and release protocol](webrtc/release-protocol.md) ships in
`assets/atmungsraum-webrtc/release-protocol.md`. Assembly verifies its signed APK
bytes with the WebRTC runtime assets. Consult it for the exact Bluetooth,
Wi-Fi, TURN and WebRTC checks and the standing Meta Developer sync procedure.

Use the existing shared 900-second loop and the retained private visuals.
Before distribution, qualify the exact signed artifact and repeat the
automatic pairing, Polar, recording custody, and update gates on the lab
headset and installed Pixel PWA. The WebRTC Bench below supplies development
evidence. The APK log identifies the signed release by one version and hash.

## WebRTC candidate

### Reconciled updater source — 2026-11-19

Apply `webrtc-native.patch` to fixed27-r2 first, then apply
[`updater-reconciliation.patch`](updater-reconciliation.patch). The two overlays
reconstruct tree `310ee8870fce0de0108cc05891abdd309d976b09`, native commit
`a7289a89d0ab092a0d5a9a89a6ef67f7cb36ff57`. The separate repair overlay is
23,933 bytes, SHA-256
`91d9ff51790246392eec5f03c98d9fbfb7123a99e0549defc5bd0bcbed5c2302`.
This preserves the exact code 30 source/artifact record below.

SDK 207 progress cannot identify an existing download's target. The repaired
coordinator therefore cancels an ambiguous duplicate, requires a positive cancel
ACK and a fresh version check, and admits one new owned download. Only its
matched success and subsequent fresh progress can reach the existing
native/persistence/helper/kiosk/SDK install sequence. Selecting helper 3 does not
permit adopting the offered version as an unidentified queued payload's target.
The original dirty source checkout is preserved; no helper update is needed.

The defect is reproduced against exact Developer 30 before the repair.
All 16 updater host suites, production Java compilation, 11-source closure
checks, native pairing/gateway controls and all 361 web/assembly tests with
zero skips pass. Independent review is clean. This is source qualification,
not proof of a headset update. Developer 31 is the next local build candidate;
Meta code 27, helper 3 and the held code 28 acceptance order remain unchanged.

### Actual startup-permission kiosk protection

For the subsequent startup repair, apply
[`startup-permission-kiosk.patch`](startup-permission-kiosk.patch) after both
overlays above. It reconstructs native commit
`fe22d7d402d2edfa912412777720685c571b4f5b`, tree
`a99d4eb89b05b170e89a10bfb45f11044807ecbb`. The patch is 30,378 bytes, SHA256
`d432a1435e0c114e45f856cd927b097733a78ed79450ea42f321706561caf70b`.

The attended bench38 permission dialogs outlived the existing60-second generic
prompt protection and kiosk selected its triple-departure exit. The repair
observes actual pending permission/Location ownership for the exact current
native Activity, independent of a timeout or whether kiosk was armed when the
prompt began. Completed, retired, foreign and finishing owners cannot retain
protection. Optional/default-off compositions and generic timed callers retain
their existing behavior. A present selected observer that is temporarily
unavailable suppresses recovery until observation recovers or the owner retires.

The regression is RED on the prior source and GREEN with 94 lifecycle, 42 actual
watchdog and 14 optional-observer controls. The kiosk policy gate, production
compilation and independent review pass. Signed Developer 32 and bench 39
packages pass their actual 900-second session closure, v3 signature and 16 KiB
alignment checks; first-run hardware acceptance remains pending.

Before running the full updater host suite on that source, also apply
[`startup-kiosk-host-fixture.patch`](startup-kiosk-host-fixture.patch). This
test-only follow-up adds the new observer method to the older install-handoff
fixture. It is 1,287 bytes, SHA256
`e81ec8b79788cec8767a8dc13d0eb10383db0b48ab20120ad7d2ec0867541321`,
and reconstructs native commit `30cde1d1aff280a9a6d6f79576df774e506616bc`,
tree `678e7fe27274353344feaab9992196967d0e2eb6`. All 16 updater suites pass
with that fixture correction. Production source and native docs are byte-identical
to the signed `fe22d7d4` packages; their source identity above remains unchanged.

Start with the exact fixed27-r2 source tree from the APK log and apply
[`webrtc-native.patch`](webrtc-native.patch) on top. The resulting native
tree is `da94a0c2c7b1acb60fa734fb9ca5cb4c0857c485`. Commit it in an isolated
source checkout so the build can bind the clean composition. The current
overlay admits the validated WebRTC offer to the Bluetooth advertiser,
advertises LAN after a successful bind, and retries a contended LAN bind
while the native owner stays live. It admits button-confirmed v2 pairing
inside a bounded v3 WebRTC frame. It includes the CSV v4 Controller State
projection, shared-scheduler Polar recording clock support, the exact
AudioTrack completion receipt and Polar host fixtures.
See the APK log for the source revision bound into the installed APK.
Select `-WebRtc` with the phone-pairing APK profile. The assembly locks
the production Java bridge and the bundled browser assets, then compares
every selected WebRTC asset in the signed APK to its input bytes.

The browser uses the pinned VDO.Ninja 1.5.5 data-only transport. The
WebView carries a bounded pairing exchange and encrypted member requests. Native
membership, gateway and runtime epochs, nonces and command receipts remain
authoritative. This path supports fresh Bluetooth-selected pairing, saved
reconnect and commissioned enrollment. The physical network matrix gates
lab release. Keep the installed Meta baseline available for the study and
update validation.

The native verification source includes the Location and live health
projection check (`tools/checks/test_polar_platform_projection.py`),
stream scope and GATT retirement check
(`tools/checks/test_polar_required_stream_scope.py`), and Direct/Remote
startup check (`tools/checks/test_polar_native_mode_startup.py`). Run them
from the patched native checkout with the build JDK and the native host
test dependencies. The startup check takes the JDK bin folder, Android
platform JAR and host JSON JAR explicitly. These checks use the production
Java methods with simulated Android effects.

### Side-by-side Quest 2 bench install

A non-Developer `-WebRtc -PhonePairing` build uses app ID
`atmungsraum_webrtc_bench`, package `org.viscereality.atmungsraum.webrtcbench`
and launcher label Atmungsraum WebRTC Bench. Its feature lock includes the
production Android bridge and the hashed WebRTC assets. The ordinary rehearsal
and the Meta Developer builds use their established package IDs.

## Shared breathing setup

### Start recovery and retained calibration

Apply [main-start-recovery.patch](main-start-recovery.patch) after the
published [background-music.patch](background-music.patch) on the Developer32
source above. It reconstructs native commit
`e71ea147fccff9b6712de14a85d634563ad3b7cf`, tree
`5a868ed5cf4a6ce074bdbf8400d3b8e9fd76ce8e`. The patch is78,345bytes,
SHA256 `f349e8ccc9b43ee2d053f3cfe2bc5afd9cd24babfe2c067a19d823366ee4a87d`.
An independent temporary-index reconstruction matches that exact tree.

Queued Start separates packet-delivery liveness from the short audio-history
window. Fresh input is still required at admission. A missed boundary can move
to a future full cycle at most twice; it cannot start an already-completed
interval. Failed Start cancels only its generation/frame-bound reservation and
retains a still-valid calibration and selected method. A normal session end
still resets preparation. Definitive native rejection uses the original full
session fence; unknown and duplicate outcomes cannot clear a newer session.

The focused regressions include actual Java audio-owner/Rust peer tests with
36-sample/200Hz packets, one delayed packet, revision-race cleanup and unknown
outcomes. Device acceptance belongs to the signed candidate recorded in the
[APK log](apks/README.md); host checks do not establish a completed experiment.

The selected Atmungsraum build uses one looping PCM AudioTrack. Its actual
playback-head frames own the cues, intended phases, automatic visual input,
calibration labels and main boundaries. Inhale, hold-in, exhale and hold-out
each last four seconds. Preparation has no time limit. The phone displays
Quest state and does not advance the schedule with a browser timer.

Automatic is the startup input, parked at minimum expansion and silent until
explicit calibration or Controller selection. Polar calibration begins at the next inhale
boundary and collects two full cycles (32 seconds). Success requires accepted,
fresh directional samples across both cycles; elapsed time alone is not
success. A successful model becomes Polar tilt while the same loop continues.
Retry repeats calibration. Controller State with fixed orientation cancels an
outstanding calibration and uses the existing controller assessment without
restarting the audio. Fresh H10 streams remain required for the participant run.

Calibrated Polar classifies the measured tilt into inhale, exhale or hold. The
existing frame-time integrator drives radius. Between valid classifications it
retains the last confirmed state: inhale keeps expanding, exhale contracting,
hold stays still. Missing/invalid samples do not reset radius. Genuine source,
input/configuration changes or explicit operator recovery still clear that state.
Technical source/projection/audio/model retirement invalidates readiness and
classification evidence while retaining the selected manual visual method and
its last diagnosed phase; it does not silently select Automatic.

If a preparation audio clock is unhealthy, an explicit calibration or Controller
request can reserve a fresh transport generation. Only actual forward playback
and fresh post-claim input admit the new setup; failed training is discarded.
Healthy retries retain the current loop. Recovery cannot restart a main run.
The phone follows the original operation's read-only receipt and measured native
progress. Diagnostics distinguish intent, polling and actual frame advancement;
the initial physical stall trigger is not yet established.

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

Use Windows with PowerShell 7.6, Git, Node.js, JDK 17, Android SDK Platform 36,
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

./assemble-apk.ps1 -QuestRepo $quest -BundledAssets -RunName atmungsraum-developer-009 `
    -Developer -Keystore $developerKeystore -VersionCode 11 -VersionName 0.2.10-dev
```

Use your Git author identity for the local patch commit. The cumulative patch
reconstructs the integrated source tree `358a0c910ba5d0a84c18ca064492f4978e35a4a4`
with SHA-256 `5c6e49967f7150fbb1e30390b92ed02aa81906e646e52d5b25dcc986ff070723`
(2,368,442 bytes), reviewed native source `4edf4b89b754c57bd062d4695233ede34c8de256`.
It preserves optional authored playback and adds explicit preparation-end/actual
audio-park admission for PWA updates. A busy native observation retains the
accepted continuation without SDK effects; confirmed retirement still stops it.
Default builds retain the original73 audio
assets and900-second shared loop; authored playback remains opt-in. The build
uses `core-lambda-stubs.jar` from the selected
Android Build Tools when compiling Java media callbacks. The
[Code 25-r2 repair source](https://github.com/GeorgeFejer91/viscereality.org/blob/0a7fb2439d249436d7b954c634d5924798549379/atmungsraum/quest/native-source.patch)
is the baseline for the following release evidence. Its cumulative patch
reconstructs source tree `df5d27ebdb8f91a1d717cf29ead5d6af32c8ac0d`, matching
the reviewed repair source `f5ea725cb04a4faabc6a424ee98d109f23d2303f`.
Patch SHA256 is `0b8f3fafcf901b74531f0f58635a7e054f6340f84fc5a15ce3b82715e3656b7f`
(2,208,323 bytes). The candidate starts fixed Controller State without volume
calibration and retains the selected manual breathing state through temporary
input loss. It also binds the selected900-second Haupt-Runde plan consistently
and reports original Arm failures without replaying uncertain native work.
Review repairs retain manual Polar selection through technical clock/source
retirement, require durable helper cancellation before intentional closure,
and expire delayed main-thread install handoffs. Software qualification is
separate from the pending physical acceptance.
Native host tests pass896 cases with six intentional ignores; actual signed25-r2
passes1114 package checks. Wearer acceptance remains a separate gate. Meta-installed25 has been observed
on the lab Quest with matching published APK bytes; later candidates require
their own package and attended update evidence.
Applying the exact binary patch to an isolated index at the
pinned upstream base reproduces that tree;
product-build status and prior native identities are in the APK log. The verified build uses compile SDK 36, targets SDK 34 and selects
automatic install location for Meta distribution.
The current Developer build selects the generic opt-in particle adapter with a downstream private
Showcase profile through the existing three-custom-input route. The bundled
command above remains the base experiment composition; it does not reproduce
those private particle inputs. Keep the full profile/shader bodies private,
retain the existing signer, and qualify the actual APK before Meta upload.
The script stages the [shared loop JSON](../audio/box-loop.json), its four phase
cues, and the retained scenario library and calibration menu prompts. It
resolves the [bundled Experimenter inputs](build-assets/README.md) and verifies
the scenario asset inventory and ARM64 loader in the APK, then prints the APK
location and SHA-256. The output and its build receipts are under
`rusty-quest/target/aq2/<RunName>/`. Use a fresh run name for a
second build or add `-ResolveOnly` to check the feature lock without compiling.
Use `-MainStudyAudio` with the authored-audio source candidate to include
the recorded main and guided Probe programmes, the calibration instruction
and Freie Erkundung. The media reader verifies the whole package against the
bound catalog. The selected Quest gateway plays the calibration instruction
before the first Polar calibration and exposes cancellation on the phone.
The candidate also advertises the verified Freie Erkundung track and uses
its media positions and natural end during participant recording. The
guided Probe-Runde plays the verified instruction before the timed run and
uses the Quest media clock for the five minutes. The authored Haupt-Runde
uses one verified FLAC for the 900-second recording and plays the original
19-second outro after a completed save. The authored source still needs
signed headset qualification.

Power-on launch is a separate, explicit `-BootAutostart` selection. It adds the
existing `ui.experiment_boot_autostart` feature and verifies its resolved lock,
receiver, metadata and permissions before building. Assemblies without the
switch retain the opt-out default. The receiver requires the app's overlay
permission and a genuine, new Android boot count; it launches this APK through
MAIN/LAUNCHER once per boot. Selecting the feature does not grant permission
or prove that Quest delivered the app to the wearer.

The selected same-APK kiosk watchdog observes its own lifecycle and OpenXR
presentation. It does not require enabling the separate Accessibility adapter.
Verify effective service, app focus and frame progression, then test physical
Meta-menu recovery and the operator exit separately. Follow the
[boot and Meta-update qualification contract](../for-ai/BOOT-KIOSK-UPDATE.md);
neither a host launch nor an ADB installation proves automatic Meta delivery.

**Meta distribution limitation, observed 2026-11-16:** the dashboard rejects
`android.permission.SYSTEM_ALERT_WINDOW` as unsupported, including for the
private Developer-channel upload. Therefore this boot-enabled composition is
currently a local-development candidate, not a Meta-distributable release.
Removing the permission alone breaks the receiver's admission contract. The
user requires both update-test versions through Meta; resolve the boot route
before claiming combined distribution/autoboot support. The approved alternative
is the [one-time boot helper](boot-helper/README.md), with a Meta-compatible
main APK that leaves `-BootAutostart` unset. The helper source/package is
qualified locally; actual Meta baseline adoption and genuine boot/menu/update
testing remain separate acceptance gates.

Assembly binds `RUSTY_QUEST_NATIVE_RENDERER_BOX_LOOP_SHA256` to the exact
packaged `audio/box-loop.json`, selects
`rusty.quest.experiment.breathing_setup.enabled=true`, and includes the audio
owners in the locked Java source closure. Other native products retain their
unselected defaults.

The current repair parks Automatic at minimum expansion with prepared, silent
audio. Only accepted explicit calibration or Controller selection starts the
shared player. Calibration evaluates the full 32-second window, masking bounded
bad intervals before its coverage/fit decision. The Polar owner automatically
establishes and recovers both required streams while respecting explicit Stop.
Original raw timestamps remain separate from the source-bound timing estimate;
see the [setup and timing contract](../for-ai/SHARED-BREATHING-SETUP.md).
The next local APK must pass the recorded source/package gates and fresh wearer
qualification before any Meta upload.

Code 7 assumes the participant follows the cues, learns their measured tilt
mapping and leaves satisfaction to testing the resulting sphere. It removes
ideal-waveform compliance checks while retaining technical/coverage/range guards.
The app also selects `input.controllers_only`, removes hand/simultaneous features
and declarations, and seals runtime hand interaction off. The original mesh
renderer accepts exactly one locked input mode while preserving its renderer,
background and panel requirements. Other native products keep their defaults.

The Developer selection also enables `experiment_phone_pairing_enabled` and
`experiment_panel_initially_closed`. The native visibility policy is compiled
into the first frame, so the app starts in VR with its panel closed. A verified
new-phone request opens the matching-code Allow/Reject prompt; it has no text
entry. Triple-B still opens the regular app panel. The optional
`experiment_headset_setup_enabled` machine adapter stays disabled.

Bluetooth exposes read-only app-installation discovery. On the same Wi-Fi,
the phone and headset each confirm the code. Saved phones have independent
keys and individual removal; one phone can hold experiment control. The phone
must explicitly claim control after connecting. See the
[pairing contract](../for-ai/PHONE-PAIRING.md) for the protocol and acceptance
limits. No generated QR, laptop or USB is part of this onboarding flow.

Before success, assembly executes the production Java packaged-closure
validator against the signed APK's actual assets. This catches the selected
900-second profile being rejected by a legacy duration check. It is a host
probe of packaged bytes; rendered startup and wearer tests remain separate.

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

A rehearsal assembly includes the matching-code phone protocol by default.
Add `-PhonePairing` to start the rehearsal panel closed, as in Developer.
The rehearsal package and signer remain separate. Follow
[phone pairing](../for-ai/PHONE-PAIRING.md) for the attended setup.

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

For the optional Meta startup updater, select `-MetaPlatformUpdate` and supply
`-MetaPlatformAppId`, `-MetaPlatformSdkRoot`, `-MetaPlatformHeadersSha256`,
`-MetaPlatformLoaderSha256`, `-MetaPlatformArchiveSha256` and
`-MetaRequiredVersionManifestUrl https://viscereality.org/atmungsraum/quest-release.json`
explicitly. Add `-MetaUpdateHelper` for the separately installed same-signer
helper2 fallback; it selects only the fixed signature-protected handoff
permission and protocol metadata. No overlay or boot permission enters the main
APK. The
native packager verifies the externally supplied official Platform SDK207,
actual JNI/DEX/manifest closure and16KiB library alignment. SDK binaries and
signing inputs remain external. Without this option there is no SDK selection.
Meta-distributed experiment builds omit `-BootAutostart`; the separate
[boot helper](boot-helper/README.md) retains its one-time setup.

The APK owns update checks, download and install admission for each native
preparation cycle. A new cycle automatically checks once; an explicit check uses
the same owner and rejects unsafe or stale boundaries. The canonical required
manifest and Meta's offered version remain distinct from actual installed
identity. An unavailable check with no known higher floor permits the installed
version with a warning; a known higher floor or uncertain dispatched install
keeps preparation blocked. The phone displays native progress and original
receipts. Helper2 provides a bounded same-signer relaunch fallback after an
admitted installation; SDK or helper acknowledgement is not installed-version
proof. Actual Meta delivery and physical repeat-cycle/relaunch qualification
remain pending for code19. See the
[startup/update contract](../for-ai/BOOT-KIOSK-UPDATE.md) for qualification and
the current evidence limits.

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
