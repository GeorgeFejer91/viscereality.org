# Verification

## Autonomous device validation

This is the standing verification procedure for every applicable app, APK,
PWA, UI or data collection change. Agents own the cycle from installing the
improvement through confirming the last affected acceptance criterion.

1. At the start of each phase and at sign-off, resolve the current Git
   revision, Pages deployment, live PWA release, signed APK and actual
   installed versions on the selected lab Quest and Pixel. For a Meta update
   check, also verify the channel offer and the prescribed baseline and
   successor. If either app changes during verification, rerun the cases it
   affects with the new pair of versions.
2. Build or obtain the signed Quest APK and install or update it on the lab
   Quest. Confirm package, version, signing identity, installed SHA-256 and
   effective foreground and runtime readiness. On the lab Pixel, refresh the
   idle standalone PWA from the exact live deployment or install it when
   missing. Confirm the active release and loaded assets from the standalone
   app. Check both physical devices on every cycle. Install each new artifact
   on its target and verify the other device is running the latest applicable
   counterpart. Preserve saved pairing and recordings. Use the prescribed
   baseline and the real PWA-driven Meta route for the official transition,
   then verify the installed successor.
3. Use the Meta Quest workflow and serial-scoped ADB for supported headset
   navigation, launch, installs and readback. Follow the
   [physical device development workflow](WORKFLOW.md#physical-device-development)
   for continuous Quest wake coverage, RustDesk MCP as the default Pixel
   driver, and authorized ADB recovery. Try the existing authorized
   remote recovery procedures before requesting assistance. Reserve wearer
   steps for platform prompts that require attendance, physical controls,
   tracking and worn Polar H10 contact. Continue independent checks while
   an attended step is pending.
4. Run the applicable stress and recovery cases in the active checklist.
   Start them autonomously as soon as the installed build pair is ready.
   Before each case, identify the build pair, input driver, repetitions or
   duration, injected fault, expected state and recovery criterion. Use
   the checklist's thresholds where specified. Exercise repeated and rapid
   inputs, disconnect and rejoin, degraded or interrupted network and sensor
   links, and restart or update boundaries where relevant. Record attempts,
   failures, recovery time and final state so the next agent can repeat
   the case. For collection, verify the clean full-duration runs,
   deliberate partial run, recovery, and matching export bytes across all
   required destinations. Check the rendered UI against the layout and
   accessibility rules, and match phone actions to native receipts or
   state changes. Use fixtures to reproduce failures and require physical
   evidence for the hardware cases.
5. Drive the lab loop without waiting for a separate refresh, install or
   retest request. Diagnose and repair each confirmed failure. Verify and
   publish each completed improvement through the release workflow, install it on the
   affected lab devices, and repeat the original case with its original
   input driver plus the affected stress and recovery checks. If lab access
   fails, try the documented recovery, continue independent cases and resume
   device work as soon as the route is available. Carry any pending case into
   the next run with its exact build pair, last verified state and next
   executable action. Record results and publication in
   [the active checklist](PROJECT.md). Keep the goal open until every
   mandatory acceptance criterion passes.

### Independent headset and PWA installations

The lab Quest 2/Pixel 10 pair establishes initial physical evidence. The
portability gate additionally requires independent compatible devices of
the same classes. Bind every case to the latest applicable canonical APK,
PWA, companion, OS and browser versions. Use the same signed headset APK
bytes on both headsets; device identity and membership are runtime state.
Record supported capabilities and leave untested platforms unqualified.

| Case | Required result |
| --- | --- |
| Fresh phone/PWA A → headset A | Bluetooth selection and both confirmation buttons establish the intended authenticated Wi-Fi/WebRTC session. Verify no code entry, hotspot or copied pairing storage. Match native build identity and a Recenter receipt. |
| Fresh phone/PWA B → headset A | The same flow creates independent revocable membership; the native active-controller lease remains authoritative. |
| Phone/PWA A → headset B, then back to A | Selection addresses the intended installation. Other headset identities, memberships and recordings stay intact. |
| Phone/PWA B → headset B | The same release and production setup work without a device-specific build or lab fixture. |
| Restart and route changes | Both launch orders, app/browser restart and interrupted Wi-Fi recover the selected saved pairing. Test direct routing and verified TURN relay on isolated/shared Wi-Fi. |
| Rejected or unavailable peer | Wrong/revoked identity and missing required permissions yield truthful actionable state and no native control. |

Require fresh frames, signed native status and actual command receipts for
successful physical joins. Observe the selected ICE route and retain the
existing disconnect/replay/lease checks. Two profiles on one phone can
exercise storage isolation but cannot qualify a second physical phone.
Fresh-PWA first join must use the production trust setup. Keep the independent
Bluetooth selection and confirmation evidence with that installation's result.
Keep the deferred worn-H10
and guarded baseline/update acceptance separate from this preparation work.

### Button-only WebRTC pairing host verification — 2026-11-19

An empty PWA stopped at the same-Wi-Fi instruction when the selected
Bluetooth service advertised WebRTC and no LAN gateway. The repair uses
the selected native pairing owner over an expiring WebRTC room. Six host
interop cases exercised production Java and browser cryptography with an
inert carrier, including matching digits, both button orders, authenticated
status and saved reconnect, rejection, expiry and callback cancellation.
The mounted production UI ran eight cases, including the two approval
orders, zero entry fields, legacy LAN fallback, rejection/cancellation,
ticket survival after lost status, headset switching and revoked membership.
The previously selected headset's LAN address was removed under the new
installation. The native frame admission and room-lifetime suite passed.
ICE/TURN, real Android Bluetooth, actual headset approval and native Recenter
receipts remain physical checks for the signed bench and canonical release.

### Physical Pixel navigation stress — 2026-11-18

The standalone Pixel PWA loaded `20261118-saved-quest-service` with the
current gateway, pairing and data modules. On publication `13e0e04`, exact
Git bytes matched all six checked live paths, including the three changed
documents, release policy, HTML and app entry. Pages run `44372630949`
succeeded. The installed Meta Developer APK still matched fixed27-r2,
`6f14707053fc9c07b611485deebfe8554a71b405892e8678f2c7f4bb2d8e398b`,
code 27 with Meta installer attribution. Its process was retained.

RustDesk MCP opened Settings from a fresh physical frame, then completed
ten Settings/Back cycles. Every tap was followed by a saved RustDesk frame
and a read-only CDP assertion of the visible page. All twenty transitions
showed the expected heading, current release, ready measurement, saved
pairing present and zero horizontal/vertical document overflow at
411 × 833 CSS pixels. The original Connect page was restored and the
RustDesk viewer input lock was released. The private receipt is
`canonical29-current-pwa-navigation-stress.json` with the twenty frames.

The app stayed disconnected. The browser's saved grant reference still
has no matching visible Bluetooth grant. This navigation check leaves
authenticated reconnect, native action receipts, worn H10, full recordings
and update/recovery acceptance open. The existing standard-companion
in-place signing and matching-enrollment requirements still apply.

### Canonical Developer build inputs — 2026-11-18

The held code 28 successor is also assembled from the recorded fixed27 tree
and verified recovered inputs. Its production v3 signature, ARM64/16 KiB
alignment and actual 900-second packaged session closure pass. Its provider
and audio assets, runtime configuration, original particle/profile/shader
bytes, OpenXR loader and both Meta native libraries match the baseline.
Complete DEX disassembly comparison changes only the provider-manifest digest
and its string-table references. The different Polar source-closure hashes
match the original after converting the checkout's LF to CRLF; generated
session identity binds the new provider manifest. The signed APK and a local
verification receipt are retained privately. See the
[held code 28 record](../quest/apks/README.md#held-developer-28-update-test-candidate).
This establishes package preparation. Wearer acceptance and genuine Meta
delivery remain pending; code 27 remains the installed/released baseline.

An independent lab-controller check selected the exact held APK digest
`111961aeeacf7bc46719ff81c3b9b585950f79570059d21762a64954aabf8295`.
Developer v3 signing, code/version identity, ARM64/16 KiB alignment and the
actual-APK 900-second session closure passed. The archive entry inventory
is unchanged: 87 assets match the baseline byte for byte, and only the four
declared build/feature/settings/panel metadata assets differ. The embedded
Manifold runtime configuration, OpenXR loader, both Meta native libraries
and provider content inventory match the baseline. The private receipt is
`held28-independent-package-and-input-verification.json`. These are static
package/input checks. Live save/update/relaunch and recording recovery
remain unverified; this check installed or uploaded no APK.

The canonical Developer candidate now builds from a verified reconstruction
of the signed baseline inputs. The original visual profile was recovered
from the ARM64 library and matches its locked SHA-256
`6fcc1a35e8bfd5c1243d25e6cb350900501ad2ad84e5a457dac242170956b21a`.
Disassembly binds the four embedded particle arrays to explicit pointers and
lengths. The recovered auxiliary array matches the baseline's compiled
coupling SHA-256 `6f8328b0b484a33db6bb6c1e238c33817733b05e94d0b927c24d348cec1efd96`.
The current particle shader sources compiled with the baseline parameters
produce all three exact SPIR-V blobs in the signed baseline.

The reconstructed input spec and provider use the original values and asset
bytes with local paths. Fresh resolution has identical selected and denied
features, Android manifest, provider content inventory and runtime values,
apart from the two generated activation binding hashes. Those hashes bind
the relocated input paths and provider manifest. This reconstructs effective
inputs and records a new spec hash. The original source app-spec bytes remain
unavailable.

SDK 207 intake now has the user's license approval. The official archive,
header inventory and ARM64 loader match the supplied pins. The native
transport test with Windows-targeted LLVM, production Java/DEX/manifest
checks and negative selection/tamper controls passed. The full canonical
assembly passed the production Developer certificate check and the actual
signed session closure. Independent inspection verified ARM64, 16 KiB
alignment, all ten WebRTC assets, SDK loader bytes, original particle arrays,
visual profile, compiled particle shaders, five provider assets and OpenXR
loader in [Developer code 29](../quest/apks/README.md#canonical-developer-29-candidate).
Its recorded Meta runtime qualification is still pending. The authenticated
Meta dashboard continues to show code 27 and build `1465199570012388`.

Independent lab-controller qualification selected the exact published
candidate digest `7cacbe9eff822d6a95e0b92e375b192a02ed76ef6df522e6b51fa811adc374d8`.
Its retained Developer v3 signature, ARM64/16 KiB alignment and embedded
source identity passed. Direct signed-APK comparison confirmed all 74
original audio entries, five provider assets, five embedded particle inputs
and three compiled shaders against fixed27-r2. The SDK native/DEX closure
digests and seven selected WebRTC runtime/protocol files matched. The
actual-APK session closure and locked 900-second duration passed again.
Private receipts are `canonical-developer-29-independent-verified.json`
and `canonical-developer-29-independent-content-verification.json`.
Bind installation and Meta actions to the exact candidate digest above.

The earlier input investigation follows for provenance.

The retained fixed27-r2 APK matched the installed Meta code 27 by
SHA-256. Its v3 certificate matched the canonical assembler's Developer
signer. The packaged Meta Platform SDK ARM64 loader matched the supplied
SDK 207 loader hash. The official SDK archive and header inventory
still need verification after the build owner accepts Meta's license.
[Meta Platform SDK 207.0](https://developers.meta.com/vr/downloads/package/oculus-platform-sdk/207.0/).

The signed feature lock identified five linked provider assets.
Each APK entry matched its recorded source digest and size. Those
five assets and the exact ARM64 OpenXR loader were staged outside Git
with a private receipt. The packager verifies OpenXR loader byte
identity from input to APK. Four provider assets are also available
with identical bytes in the bundled build inputs. The experiment
session profile and OpenXR loader differ from the bundled versions.

None of the inspected local app specs or provider manifests matched the
signed baseline hashes. The recovered files cover the provider asset
bytes and OpenXR loader. Continue with the retained Experimenter app
spec, base feature lock, provider manifest and private particle source.
The supplied Developer keystore arrived through the user's QR-linked
transfer. The 3,618-byte ZIP decrypted successfully and its single
3,530-byte keystore entry was extracted into private staging. Its
certificate matched signed fixed27-r2 and the canonical Developer
signer. The existing build credentials opened the private key entry,
and an RSA signing and verification challenge passed. The transfer
key, keystore and verification receipts remain outside Git.

Finish the official SDK intake and retained build profile verification
before release assembly. All eight assembly checks passed on the
pinned native tree, and the updater's 11-source selection and tamper
checks passed.

### Listener-aware Bluetooth discovery — 2026-11-18

Bench 35 reported `listener-unavailable` while the Meta baseline process
was retained. The production gateway host fixture reproduced the same
bind state and failed because Bluetooth metadata still listed LAN
addresses. Address publication now follows the gateway's successful
listener bind. The occupied-port regressions verify that the address list
is empty while the trusted WebRTC offer is present. The existing retry
restores the addresses under the same gateway epoch after it binds.

The complete gateway host suite, 59 focused phone and WebRTC checks
including the compiled Java peer, and all eight assembly checks passed.
The public overlay applied to fixed27 in an isolated Git index reproduced
native tree `e7eedd854fd26ab91cb02a0ef56f4eaa25f914e2`. The
[signed bench 36 APK](../quest/apks/README.md#webrtc-bench-with-listener-aware-bluetooth-discovery)
passed v3 signature, ARM64, 16 KiB alignment and session closure. An
exact-serial ADB update retained the bench UID and grants and the installed
APK matched the signed SHA-256.

The same physical startup check advanced native frames from 162 to 314
with the Meta baseline retained. Bench discovery reported advertising,
with the listener unavailable. The bench's run-owned process exited and
the original Meta code 27 process returned to Prepare with its original
APK digest. The Pixel resumed `20261118-recordings-offline-landscape`
with its saved pairing. The control trial that would release the running
baseline was rejected by automated command approval before dispatch.

The physical Bluetooth retest used an uncached read of the discovery
characteristic on the Quest while bench 36 was in Prepare and reported
`listener-unavailable`. A 10.5-second Windows scan found one device
advertising the project service UUID. Both service instances returned
successful reads. The gateway identity matched Meta code 27 to the
228-byte snapshot with one LAN endpoint. The bench instance supplied
a 304-byte snapshot with zero LAN endpoints and a WebRTC offer with
the expected field lengths. The probe recorded only the route counts,
offer shape, and baseline identity match. The Windows radio returned to
Off. The original Meta process returned to Prepare and the Pixel resumed
the current standalone PWA with its saved pairing. The test used the
[Windows GATT read API](https://learn.microsoft.com/en-us/windows/apps/develop/devices-sensors/gatt-client).

The outstanding bench gates are same-owner physical rebind and
authenticated standalone phone pairing with native receipts. The guarded
Meta and worn Polar gates remain separate.

### Pixel companion discovery comparison — 2026-11-18

The lab Meta Developer APK matched code 27 and the signed Quest bench
matched code 36 before launch. The Pixel's installed standalone PWA
loaded `20261118-saved-quest-service`, which matched the live deployment.
Bench 36 reached Prepare with advancing native frames. An uncached
Windows read of one physical Quest found both Bluetooth service
instances and confirmed the bench advertised a WebRTC offer while its
LAN listener was unavailable. The first Windows attempt saw the
advertisements without receiving the payloads. The repeat read both
payloads successfully.

Android socket ownership bound the standard code 1 companion to loopback
8767 and the signed commissioned bench code 4 companion to 8768. The lab
APK matched the retained signed artifact by SHA-256. An earlier launch
check queried the standard port for the lab companion and restored the
standard service. The final comparison used each service's own port and
both owners stayed present for the reads.

Standard code 1 returned the exact Meta code 27 identity with one LAN
endpoint after 2.55 seconds. All 27 polls for the bench identity stayed
at searching through 22.58 seconds. The commissioning request closed
without a response. Bench companion code 4 returned the same Meta
identity after 1.72 seconds and the exact bench 36 identity after
3.48 seconds. Bench 36 returned a WebRTC offer with zero LAN endpoints.
The lab commissioning endpoint returned the identity corresponding to
that same bench installation. Five host cases for the current production
scanner passed. The historical standard scanner uses one GATT service
instance. The newer scanner walks the instances until the requested
installation matches.

The diagnostic used read-only companion requests and verified the same
Meta process back in Prepare, the original PWA back in the foreground,
and the prior Windows radio and phone listener states restored. The
production companion still needs a release signed with its retained
certificate. Its available update has a different signer. Verify
authenticated startup, WebRTC route selection and original native
receipts with the matching enrollment and current production companion.

Preparation rechecked the public certificate of the local Android debug
keystore. It is `51a7052801899e071b728ae7e4dbd1fba0507f0f8c6ffdb7c6754fc1eb65eedb`,
matching the available update's signer. The installed standard companion
uses `5d6502013bba4f61acfb82a008bda7420bcff8c8920311bd34d07e1a618cb0e1`.
The original companion key remains a prerequisite for an in-place release.
No replacement or enrollment change was attempted. The user deferred the
physical acceptance test; preparation and the held APK sequence remain ready.

### Remembered Bluetooth service selection — 2026-11-18

The lab Bluetooth read returned both Atmungsraum service instances under
one BLE device. Meta code 27 returned first. The PWA called the
single-service Web Bluetooth operation even for a saved installation.
A regression using the measured ordering failed when the first
instance belonged to a different installation and the saved Quest
supplied the valid relay-only payload in the next instance.

The remembered-device route now checks each matching primary service
until the discovery data matches the saved installation. A missing match
is rejected and releases the GATT connection. The normal chooser and the
gateway's existing signed admission rules still apply. The regression
checks both the expected and absent installation. All 52 focused Node
checks and five mounted PWA startup groups passed. All browser modules
load the same gateway version, and the connected GUI snapshot waits for
the actual Next and Polar readiness states before measuring. The
installed Pixel's Chrome exposes the multiple-service API. Its saved
record contains a browser grant reference, and Chrome returned no
matching device for that reference. Authenticated reconnect for the lab
installation still needs a matching enrollment.
[Web Bluetooth service lookup](https://webbluetoothcg.github.io/web-bluetooth/#dom-bluetoothremotegattserver-getprimaryservices).

The exact
[Pages run 44353740380](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44353740380)
deployed merge `3cdea7b` after the 421-state GUI gate. Eighteen live paths
matched Git. The installed Pixel loaded the current entry, operator
module, and gateway version 19. A cold standalone launch was visible after
1.23 seconds and still unauthenticated at 30.36 seconds. The exact Meta
gateway returned a different installation ID from the Pixel's saved
pairing. The phone retained its enrollment and had no matching Chrome
grant. RustDesk MCP opened Settings and returned to Connect, with both
transitions confirmed in fresh frames.

### Signed WebRTC Bench 35 startup — 2026-11-18

The preflight matched the installed APK hashes for the signed bench 35 and
Meta code 27 on the awake lab Quest 2. The bounded Windows watchdog had
more than five minutes of hold remaining. The Pixel was on the current
`20261118-recordings-offline-landscape` release. It went to Android Home
for the headset switch so its saved pairing stayed with its existing
installation. The headset used the exact MAIN and LAUNCHER component.

The bench reached the foreground. Two typed product panel receipts
reported experimenter Prepare, a native submitted frame, valid tracking,
an enabled membership and discovery advertising. The submitted frame
advanced from 223 to 313. The bench reported `listener-unavailable`
while the baseline process was retained. The test sent no phone pairing
or experiment request. The run-owned bench process was then stopped.
The same Meta code 27 process returned to Prepare with a fresh submitted
frame and a listening LAN gateway. Its installed digest still matched.
The Pixel returned to standalone on the same release with its saved
enrollment present and no connected Quest. Private evidence includes the
preflight and original native receipts. Physical WebRTC reconnection and
worn Polar collection on this exact revision remain open.

### Meta and standalone startup recheck — 2026-11-18

The authenticated [Meta Developer channel](https://developers.meta.com/vr/manage/applications/1459506690581676/release-channels/)
still offered `0.2.26-dev` (code 27), build `1465199570012388`. The exact
installed APK matched signed fixed27-r2 by SHA-256 and reported
`com.oculus.ocms` as installer. The live policy agreed with Meta and the
device. A fresh read of both installed APK archives found the WebRTC
runtime and release protocol in signed bench 36. The Meta code 27 archive
contained the LAN gateway.

[Pages run 44355388552](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44355388552)
deployed `0f212c0409adf47c3cf7b68bc23b5ae8626b4c21`. Eighteen live paths
matched Git. On the Pixel, three exact-launcher cold starts produced new
standalone documents on `20261118-saved-quest-service` with visible Connect
after 1.34, 1.36 and 1.38 seconds. All three kept searching for the saved
Quest with no dialog and no authenticated session at 30 seconds. The
phone's saved installation did not match the Meta app's live gateway
identity. The headset remained in the Developer app and its gateway
answered on a temporary serial-scoped ADB forward. The installed phone
companion closed the commissioning request before returning any profile
information.

All 43 focused Node checks and five mounted startup groups passed. The
official automatic startup gate failed on the current installed release.
The next physical run needs the signed unified Developer build and a
compatible companion commissioned for that installation. Retest first
join and saved reconnect in both launch orders with native receipts.
The held Meta successor remains in the existing release sequence.

### Meta Developer and Pixel startup — 2026-11-18

The [Meta Developer channel](https://developers.meta.com/vr/manage/applications/1459506690581676/release-channels/)
served `0.2.26-dev` (code 27). Its complete
[build 1465199570012388](https://developers.meta.com/vr/manage/applications/1459506690581676/builds/1465199570012388/)
targeted Quest 2. ADB confirmed Meta installer `com.oculus.ocms` and the
installed base APK matched the signed fixed27-r2 SHA-256. The live
version policy also matched. The exact
[Pages run 44337383494](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44337383494)
matched eight paths at `31b3822`. The installed Pixel loaded the matching
standalone release `20261118-recordings-offline-landscape`.
The mounted APK policy check passed 15 cases across 18 rendered states.

Three serial-scoped cold launches each created a fresh standalone document.
The latest PWA became visible after 1.25–1.45 seconds and continued
searching for its saved Quest for the full 30-second observation each time.
No attempt reached authenticated status. The saved enrollment persisted
and belongs to another installation. The signed app's typed ADB status
reported Prepare, discovery advertising, and its trusted LAN listener
running without an enabled enrolled controller. A current phone-side
TCP attempt failed on the shared Wi-Fi subnet. The same listener answered
on an owned USB forward that was then removed. The exact APK contains no
bundled WebRTC runtime. The Pixel remained on Connect, with RustDesk Input
working and Android display settings restored.

The official startup gate remains open. Test the qualified unified Developer
APK with the matching signed companion, commissioned phone, and current PWA.
Exercise both launch orders, cold first enrollment and saved reconnect on
the supported network matrix, and match a native receipt on each success.
Use the prescribed Meta release sequence. Earlier bench receipts remain
recorded under the [connection milestone](PROJECT.md#connection-milestone--fast-web-pairing-across-isolated-wi-fi).

### Haupt-Runde questionnaire — 2026-11-18

The mounted Chromium tests replay the existing live recording and final
signed-descriptor paths with synthetic data. They check that the form waits
for verified local custody, opens for a completed main, and stays closed
during saving, after a partial run and after a practice run. They verify
all five controls mounted together, scroll and focus at reduced viewport
height, required selections, rejection of untouched sliders, offline reload
and pairing scope, optional access from recordings, and the numeric answers and
scale metadata in the downloaded JSON. The
mounted Drive case authorizes a fixture account, confirms the CSV first,
then uploads the questionnaire automatically under a different file ID in
the same folder. An ambiguous questionnaire response is reconciled without
reserving another ID, and the uploaded bytes match the downloaded file.

The questionnaire uses the production Pretext and viewport gate, including
320 px portrait, short landscape, 200% text and the text-spacing override.
At 200% text and 320 × 428 or 640 × 320, bringing the missing fieldset into
view could leave its focused input below the scrollable form. The native
scroll call now brings the focused input into view as well. The mounted
cases verify an untouched first slider, an unanswered Likert item and a
later missing slider, with the target fully visible and the document
stationary at both sizes.
The separate mounted archive recovery and PWA startup suites pass. The Node
suite passes 334 checks with 12 native-fixture checks skipped and zero
failures. Full lab acceptance is tracked in [PROJECT.md](PROJECT.md#haupt-runde-follow-up-questions--2026-11-18).

### Short Recordings layout — 2026-11-18

At 200% text and 320 × 428 or 640 × 320, the combined controls overflowed
after a questionnaire save. The short layout presents the Quest recording,
download, sync and storage controls in separate views of the same screen.
The mounted Chromium checks cover every view and the failed-sync status.
They verify zero document and panel scroll, fitted labels, 48 px targets,
focus across resize, keyboard entry to the recording selector, both
downloads, Drive dialog close and return from browser archives. The
explicit retry still recovers from a failed archive list. There are no
native session or setup mutations from these views.

The installed Pixel showed the footer clipped at 594 × 272 CSS px. A
regression at that size with 200% CSS text reproduced the scroll. The
landscape layout now starts its two-column header at 568 px, allocates more
footer width to the longer action, and places the sync status beside Retry.
The compact browser archive link retains its full label and fits on two
lines. The mounted cases verify all four views, the failed-sync message,
retry and browser archive return at the new size.

On exact Pages deployment `79d2a49`, the physical Pixel still had 37 px
of document scroll while seeking its saved Quest. The hidden Drive
control left a grid item that pushed the browser archive link to a
second row. The new regression runs the ordinary saved-Quest retry flow
with an unreachable endpoint and checks all four views at 594 × 272,
with default and 200% CSS text, with and without a local recording.
The compact layout removes the empty Drive
item, places the recording contents beside the discovery message, and
uses a short discovery message on Recordings. The Connect page retains
the full Bluetooth status. The fixture checks the footer and performs
Back without a Quest command.

[PR 126](https://github.com/GeorgeFejer91/viscereality.org/pull/126) passed
421 mounted GUI states, five PWA startup groups and 47 targeted controller
and recording tests. The exact
[Pages run 44335925396](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44335925396)
deployed `4c822ea`. Eight live paths matched the merge. ADB relaunched
the installed Pixel in standalone mode and the entry, stylesheet and
operator module reported `20261118-recordings-offline-landscape`.
The saved pairing stayed present. On the disconnected Pixel at 594 × 272
CSS px, RustDesk MCP selected Archive, Files, Sync and Storage. The
archive selector, download control, browser archive link, and both
footer controls were 48 px high and completely visible. Document and
panel scroll were zero in every selection. A RustDesk tap on Back
returned to Connect. Android display size, rotation and font scale were
restored. Device images and measurements remain in the private run folder.

### Quest target signaling startup recovery — 2026-11-18

The production target previously ended its startup after a rejected signaling
attempt. On signed bench 31, the ordinary physical Pixel first acquired
authenticated status. One injected rejection on the bundled Quest WebView
then left the valid offer unannounced for the entire 12-second readback and
the Pixel could not rejoin during that window.

The packaged target now tries again after two seconds while the same native
offer is valid. The retry is retired on page exit or offer expiry. Four
lifecycle tests use the production target and transport with a controlled
SDK response. The initial retry assertion failed before the fix. The full
342-case Node suite, fresh Java host checks, 350-state smartphone GUI gate
and signed APK session closure pass.

Exact installed bench 32 reached focused OpenXR frames on Quest 2. The same
injected rejection led to two total startup attempts, one successful
signaling connection and one announcement on the same valid offer by 3.1
seconds. Both pages and the prior power settings were restored. During this
run the physical Pixel was still searching for the bench over Bluetooth
before opening WebRTC. The lab checklist tracks rediscovery and the complete
phone-to-headset retry separately.

### Multi-instance Bluetooth rediscovery — 2026-11-18

The host regression runs the production scan callback. Its first case
reproduced the failure when the first matching service belonged to another
installation. All five cases pass after the repair, including both Android
characteristic callback forms, skipping a Direct-mode service, ignoring a
failed or malformed read, and returning the first eligible match. The
managed proof host test and both signed Android builds also pass.

On the physical Pixel, the same-signer bench companion update retained
the app UID and Bluetooth grants. With the Quest Developer and bench
packages registered together, auto discovery found the expected
installation with no chooser. The original WebView startup rejection was
repeated on bench APK 32. The same offer was announced in 2.9 seconds, and
the installed PWA then rejoined with a relay path and authenticated status.
A separate RustDesk Recenter run with an isolated lease journal confirmed
the original native receipt. The standard companion's installed certificate
differs from the available lab signer, so that package's in-place update
remains pending.

### Hotel Wi-Fi timing and recovery on the repaired scanner — 2026-11-18

Run 30 alternating first joins with an empty temporary PWA pairing slot
and 30 saved rejoins on the installed Pixel. The Android lab companion
uses a provisioned identity. Each case has 23 seconds to reach authenticated
actionable controls, with the original pairing and lease journals isolated
in the verification document. Retain the installed browser profile.
The first batch's empty-slot cases were ineligible because the test
companion was missing its lab commissioning overlay. The commissioned
bench build restored that overlay before the 60-trial run.

| Path | Passed | Median to controls | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| First join | 30/30 | 1.92 s | 16.56 s | 16.91 s |
| Saved rejoin | 30/30 | 1.77 s | 4.58 s | 4.61 s |

Both devices used the public hotel Wi-Fi. The Quest reported two service
registrations. The installed PWA used default ICE selection and every
successful trial selected a TURN candidate. Code and chooser counts were
zero. The exact lab combination was Quest bench 32, Pixel bench companion
4 and PWA release 20261118-connection-error-status. The official standard
companion requires its retained signing key for the in-place update.

The paired hardware also passed a separate relay-only check with zero LAN
attempts and 1.2-second delays on outbound phone signaling. The empty slot
reached authenticated status in 10.05 seconds. With the transport closed
and signaling held for ten seconds, connected state cleared in the first
observation and stayed disconnected during the fault. The same enrollment
returned 6.08 seconds after signaling resumed. RustDesk confirmed the
original native Recenter receipt on the recovered route. Both fixtures were
removed and the routine PWA and headset power settings were read back.

### Commissioning snapshot expiry — 2026-11-18

Capture the installed PWA's local commissioning HTTP statuses and the timing
and version of data-channel frames. Keep request bodies, pairing material
and offer secrets private. Two hotel Wi-Fi first-join traces reached an open
channel around 1.4 seconds after loading. The companion replied 403 on the
first authorization request and the native proof exchange began only after
rediscovery on a second carrier.

The physical control held the first ready Bluetooth snapshot for 12.25
seconds, past the companion's 12-second cache lifetime. The original PWA
received 403 and opened two carriers for the same native offer. The second
authorization began 14.02 seconds after the first denial.

The production broker now rereads Bluetooth discovery after an authorization
403 using the existing commissioning deadline. It retries the exact proof
request once when the installation, both epochs, room and room secret still
match. The three host regressions cover searching then ready, rejecting a
changed rendezvous, and deadline cancellation. Enrollment still requires
native proof and authenticated status.

[Exact Pages run 44265405035](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44265405035)
and the 350-state GUI gate passed. The exact deployed PWA was read back on
the installed Pixel. With the same expired-snapshot fixture, reauthorization
started 3.30 seconds after the first 403 on one carrier. The same owner and
offer were confirmed, followed by native proof and actionable controls.
The recovery removed its fixture and restored the phone document.

All 60 subsequent hotel Wi-Fi cases passed. First joins had median 1.79
seconds, p95 6.61 seconds, and maximum 6.86 seconds. Saved rejoins had
median 1.53 seconds, p95 2.91 seconds, and maximum 4.59 seconds. Four first
joins received 403 and recovered on the first carrier. All 60 selected
TURN. The native app and the prior headset power policy were restored.

### Official code 27 preparation readiness — 2026-11-18

The official Developer package was Meta-installed fixed27-r2, code 27,
with exact APK SHA-256 `6f14707053fc9c07b611485deebfe8554a71b405892e8678f2c7f4bb2d8e398b`.
The standalone Pixel loaded release `20261118-participant-keyboard-layout`
from [Pages run 44275522298](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44275522298).
Public release policy listed minimum and latest code 27.

A serial-scoped launch using the declared `MAIN`, `LAUNCHER` and Meta VR
categories started the official process. The bench activity retained the
foreground throughout the 85-second observation. Adding an Android Home
dispatch before the same launcher intent brought the Developer app forward.
The official renderer reported `FOCUSED` and advancing submitted frames.
Subsequent bounded wakes again exposed the official foreground.

The package's read-only ProductPanelOperator receipt reported the
experimenter Prepare page, a hidden operator panel, and `listener-unavailable`
with `trusted_lan=true`. The installed Pixel had loopback permission granted.
An owned Quest ADB forward and Pixel USB reverse did not establish signed
status, and the separate host-side gateway probe could not read the endpoint
in the same run. An Android Home intent followed by a warm official launch
again produced focus and submitted frames. The scoped port 8766 gateway
probe still received no response.
The earlier anonymous gateway response was observed while the bench held
the foreground. The official observer sent status requests only. The Meta
offer and preparation action remain unobserved. The two owned USB mappings
and the temporary wake policy were released.

An isolated cold launch with the bench process stopped restored the official
listener. The exact installed Pixel PWA compared its saved record against
the reply and found a different device and serial, with the legacy identity
kind. The official port replied under its app-installation identity. Its
operator readback reported no enabled enrolled controller. Establish
authorized membership for that installation and require signed status for
the live Meta offer and preparation eligibility.

### Legacy pairing recovery — 2026-11-18

The installed Pixel had a legacy record for a different Quest installation.
The startup regression reproduced a similar legacy record suppressing an
available commissioned profile. PWA release
`20261118-legacy-commissioning-recovery` tries the saved enrollment first.
If a legacy reconnect fails, it can consult the local companion and reuse the
existing commissioning proof path. It writes the new pairing only after
the companion and Quest authenticate the current installation and signed
native status arrives. The browser keeps the legacy record during the attempt.

Browser startup checks cover the verified replacement, a rejected signed
reply, a pending receipt or unanswered stored questionnaire that keeps the
original owner, and a newer saved selection arriving during authentication.
Existing saved installation trust keeps its reconnect path. The separate
physical success case still requires an authorized companion profile and an
isolated legacy record on the Pixel.

The exact live release passed 378 GUI states and 346 web checks against
the integrated questionnaire release. The mounted archive recovery suite
also passed. On the actual Pixel, the exact standalone release kept its
ordinary saved pairing and passed RustDesk Settings, Quest Control details
and return to Connect. The host rejected setup of the isolated physical
commissioning fixture by policy.

### Quest gateway listener recovery — 2026-11-18

A temporary ADB reverse mapping occupied Quest port 8766 during an isolated
official code 27 launch. This reproduced `listener-unavailable`. Removing
the mapping left the original process without a gateway response after 12
seconds. An isolated cold relaunch restored the official Prepare listener.

The production gateway host regression failed on the missing retry. The
repair queues one two-second retry after a bind failure while the native
owner and LAN permission remain current. Native retirement cancels the
queued effect. The regression now passes with the shared Remote gateway
suite (89 checks), discovery suite (119) and phone-pairing UI bridge (25).

Signed [bench APK 33](../quest/apks/README.md#webrtc-bench-with-lan-listener-recovery)
matched the installed bytes and retained its signer, UID and grants. The
same port fault exercised the signed APK while OpenXR was focused and
submitted frames. After release, the listener returned in 0.93 seconds under
the same process and panel owner. The installed Pixel PWA authenticated the
new APK code and IDLE native status using the recovered gateway over an owned
USB test route. A subsequent ordinary WebRTC saved rejoin reached controls
in 7.80 seconds, and RustDesk Recenter matched its original native receipt.
The browser's ordinary saved record, temporary ADB mappings and headset
power policy were restored. The controller trust and Meta behavior of the
official APK remain separate release gates.

### Phone TURN/TLS and invalid controller requests — 2026-11-18

The physical devices used standalone PWA `20261118-participant-keyboard-layout`
from exact [Pages run 44272922844](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44272922844),
bench companion 4, and signed, foreground Quest bench APK 32, with the pinned
VDO.Ninja SDK 1.5.5. The Quest reported focused, advancing submitted frames.
The installed official Developer APK separately matched Meta code 27.

The temporary phone fixture first restricted its ICE candidates to the SDK's
TURN TCP/TLS endpoints. The first join and wrong-tag cases timed out before
opening a channel. The altered ciphertext reached a real TURN/TLS route and
the channel closed without control. Valid saved trust then reopened control
over TURN/TLS and RustDesk Recenter matched its original native receipt.

After a default-ICE warm-up, a temporary empty browser pairing slot and a
saved profile reached actionable native status on TURN/TLS in 2.19 and 4.25
seconds. The empty slot obtained its existing device authorization via the
commissioning exchange. Neither case showed a chooser or pairing prompt.
The Pixel reported a local relay with `relayProtocol=tls`. An independent
read-only WebView observer on the Quest saw its selected peer use a local
server-reflexive UDP candidate with a remote relay candidate during the
saved rejoin. The restriction applied on the phone and the Quest retained
its normal ICE configuration.

In the same repeat, default ICE delivered one request with a modified member
tag and two with modified authenticated ciphertext. All three implicated
channels closed. The phone received no authenticated reply and exposed no
actionable controls during either 17-second negative case. The subsequent
valid saved reconnect returned the original native Recenter receipt from a
RustDesk tap. The temporary phone profile and Quest WebView forward were
removed and the prior headset power policy was restored.

Run the end-to-end UDP-block case with a verified network filter and the
LAN comparison on a verified common route. Keep the physical revoked-member
case open until an independently revocable bench identity is available.

## Smartphone GUI release gate

Run `python -B for-ai/scripts/check-gui.py` from `atmungsraum` while editing.
Use `--ref HEAD` to verify the exact committed tree. The tracked pre-push
hook runs the check for the candidate branch revisions and exits nonzero
when a candidate fails. From the repository root enable the hook with
`git config --local core.hooksPath atmungsraum/for-ai/hooks`.

Use the real production controller with its synthetic Quest transport.
Exercise connected and disconnected setup, editing and validation, active
and recovery states, dialogs and expanded controls. Cover 320 × 640 and
390 × 844 portrait, 844 × 390 landscape, 1440 × 900 desktop, 200% text,
user text-spacing overrides, both sides of layout breakpoints and focused
editing with reduced height. Verify positive checks and each independent
failure popup, with keyboard activation and focus restoration.

Check the [UI rules](UI-LAYOUT.md#smartphone-gui-rules) against the painted
DOM and real Pretext measurements. Assert viewport and control bounds,
stretch anchors, tracks and growing group gaps, full text and zero scroll
range for the document and nested regions. Treat overlap, clipping,
inaccessible controls or missing evidence as failures. Confirm safe
areas, the keyboard and the installed PWA on the supported physical phone
when emulation cannot exercise them. Record the candidate SHA, browser or
device, viewport, measured states and results.

Follow the [RustDeskMCP phone validation protocol](RUSTDESK-PHONE-TESTING.md)
for installed-phone input, capture and native receipt evidence.

### Connection guidance after relay and LAN failure — 2026-11-18

With signaling held and an actual native LAN address offered, the physical
Pixel reached Bluetooth readiness but its browser could not reach the LAN
address. The original fallback copied `Failed to fetch` into the operator
status. The production startup fixture reproduces the two route failures
using saved trust, with no chooser. It verifies the existing German
connection guidance, a passing text measurement and horizontal bounds at
320 px and 200% text. The cached-release, idle refresh, Bluetooth and
successful LAN fallback fixtures also pass.

Merge 8e57ffe passed [exact Pages run 44250589158](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44250589158),
six live file comparisons and the 350-state GUI gate. The installed Pixel
loaded standalone release 20261118-connection-error-status with saved
pairing retained. On the signed Quest bench code 31, focused with advancing
submitted frames, the physical retest observed a failed request to the
offered LAN endpoint and displayed the German guidance about 4.6 seconds
into the run. The status fit the phone width. With signaling restored, the
Pixel reached authenticated bridge status and selected TURN after 6.0
seconds without a chooser or new enrollment. The ordinary PWA and prior
headset power policy were restored.

### Temporary status loss after Save — 2026-11-18

The mounted 320 × 568, 200% text regression loses the status read
immediately after the original Save receipt has been confirmed. The
previous page replaced the saved state with an unconfirmed-update message.
The repaired page retains the intent on the same connection. The fixture
observes one Save, no update while the writer and audio checks are pending,
and one check-and-update request after the fresh park observation. Related
cases cover an uncertain Save, lease and owner changes, and the native
automatic attempt. All 27 rendered update cases use the shipped UI and
real Pretext. The full smartphone GUI gate passes 350 states. All 338
Node tests pass with the native JVM fixtures enabled and zero skips.
Startup and idle release refresh checks pass.

The production coordinator, duplicate recovery, service, helper monitor,
queued dispatch, queued exit, main handoff and reusable SDK host checks
pass. These compile the current Java owners with modeled SDK and Android
effects. The physical Save→Meta update→relaunch gate follows the attended
recording plan in [the active checklist](PROJECT.md).

### Combined narrow screen and enlarged text — 2026-11-18

The 320 × 568 fixture at 200% text reproduced overflow in the package
and build display, unavailable release policy, Live ECG and pending
receipt. The app exposes each exact APK identity field in its own
selector option. The dialog uses the available safe width and keeps the
receipt selector above its contents. Updated copy fits the existing
recovery panel while retaining the operation and no-replay instructions.
The narrow stress run exercised all four recording phases with the
ECG dialog, both APK identity views, and the original pending receipt
after local navigation. The Chrome loopback denial regression checks
readable remediation and clears it when the browser grant is restored.

### APK detail measurement — 2026-11-18

The smartphone GUI gate on the previous release caught an unmeasured
APK identity panel at 390 px. The mounted regression reproduced it while
ResizeObserver callbacks were delayed. An actual dialog visibility change now
schedules the existing Pretext pass without waiting for a resize or moving
focus. The regression passes with usable geometry and the existing selector
focused. The release gate passes 339 states and the complete operator
walkthrough. All 35 compact Update renders pass. The installed-phone retest
remains open.

### Smartphone UI rule port — 2026-11-17

The installed pre-push hook tested the archived `faff9b4` candidate and
passed 338 states across eight Chromium viewport and text cases. These covered
the single toolbar row, 48 px controls, the 389/390 px breakpoint, enlarged
text, text-spacing overrides, reduced height during editing, Pretext and
painted bounds, stretch geometry, and zero document or panel scrolling.
Independent diagnostic popups returned keyboard focus to their triggers.
Selecting from 60 synthetic historical records downloaded the expected CSV.

The production controller walkthrough also passed with synthetic Quest
transport. The JavaScript suite reported 316 passed and nine skipped, and the
mounted update suite passed 35 states, including the 320 px and 200% text case.
USB and wireless ADB listed no devices. The installed PWA's physical keyboard
and safe-area checks remain open.

## Attended Meta26 request and owner-read contention — 2026-11-17

The repaired source4edf4b89/tree358a0c91 passes independent spec/quality review,
907 native tests/6 ignored,183 coordinator,108 duplicate,64 service and12
queued-exit checks within all14 updater suites. Real held projection/setup locks
are observed Unavailable while exact wrong IDs and observed retirement still
reject. Consumed SDK messages, post-ARM and post-journal stages resume once,
without another poll/download/install/exit effect while held. Android arm64
cargo check,110-source production javac and actual Java/native audio join pass.
The2,368,442byte source patch reconstructs that exact tree. Fixed27-r2 is built;
its independent signed-artifact qualification passes1138 checks/63 negatives.
Meta1465199570012388 is current with malware/security passed. Actual installation
and27-to28 acceptance remain pending. Source PR58/merge2d59e16 reached exact
Pages44183709720 success and10 live files match Git.

Policy26 release PR57/merge5df115b deployed in exact Pages44178155493. All9
checked live files match Git. Live browser fixtures verify required25 has an
enabled Update action,26 skips that required step, and running25 retains its
explicit Save/Update action without an automatic stop. This is synthetic
transport, not a device installation.

The actual Pixel6a loaded20261117-preparation-audio-recovery; green control,
IDLE/rec:false and an enabled Update button are confirmed. PR56/merge5f3b375
reached Pages44169545352 success,17 live-file matches and26 live browser cases.
Repaired26-r3 passes1135 signed-package checks/39 negatives and is published as
Meta1465122190020126 with security/malware passed. No helper update.

Two original PWA requests were accepted. The first still returned latest25;
the later same-process request1312caf4c1629771 returned26 and began downloading,
then failed native-owner-retired at262,144/8,246,427bytes. No install or helper
handoff occurred. The source treats transient unreadable registry/session/setup
locks as retirement; old logs cannot identify the precise failing predicate.
Conservative typed-observation recovery and real contention regressions are in
progress for27-r2. Actual25 remains the last installed readback. A third planned
phone request did not dispatch because USB disconnected. This is unresolved
hardware/update acceptance, not a successful end-to-end update test.

## Preparation and save before PWA updates — 2026-11-17

Final source follow-up20261117-preparation-audio-recovery passes324 full Node
cases/zero skips,26 mounted preparation renders and26 compact Update states.
The exact narration-park status shows guide-release waiting and conditional
manual app reopen/reconnect recovery; a real native status change returns the
same attempt to ordinary Meta checking. Scoped re-review passes with no
remaining finding. Native2f41df0a/tree70e03235 passes907/6ignored plus the actual
Java/audio/writer/outro joins. The one first-run submitted-frame timing failure,
isolated pass and full serial pass are retained in private evidence. The exact
2,350,099byte cumulative patch reconstructs the reviewed native tree. Signed
26-r3 packaging is complete; independent artifact qualification is still pending.

The merged PWA release20261117-preparation-update passes323 Node tests with
zero failures/skips,25 mounted preparation cases,26 compact Update states,10
current-session/offline-archive renders and the authored operator flow. Exact
calibration instruction identity, native end-preparation admission and the
non-recording preparation owner allow that narration to end for an update.
An unresolved actual guided Start remains protected with a specific wait message
until its original receipt confirms; no unrelated pending command is overridden.
Original Save confirmation precedes writer cleanup, a fresh actual audio park
and update admission. Foreign owners/revisions and uncertain Save outcomes
cannot continue into installation. Scoped independent review and all14 changed
runtime cache identities pass. Rendered checks use synthetic enrolled transport;
they do not prove a Meta installation or wearer acceptance. Signed APK and live
publication evidence is separate. PR55/mergeae6d3e5 reached exact Pages44165786315
success;25 live files match Git and the deployed HTTP runtime passes all25
preparation/save/update cases with synthetic enrolled transport. Installed-phone
refresh remains pending because the USB Pixel disconnected; no actual update
installation is inferred from these checks.

## Review repairs: current-session controls and offline archives — 2026-11-17

PWA20261117-review-repairs passes309 Node tests with zero skips after current-main
integration and cache-version changes. Independent review reruns53 model cases and the mounted real
PWA recovery/archive suite. Ten rendered states/19 viewport tiles pass actual
Pretext/DOM checks. Narrow/200% text uses readable document reflow where needed.
An old preparation receipt is preserved while a fresh current-session lease can
Pause/Resume/Finish; stale ownership/revision, unknown experiment commands and
native recovery still block. Offline committed ZIPs retain their original hash,
exclude incomplete prefixes and foreign identities, survive APK changes, and
cannot reappear after a delayed read following identity forgetting/switching.
No headset or phone was operated. Installed-phone refresh and physical
acceptance remain pending. PR50/merge701e8dd deployed successfully in exact
Pages44135534810;12 public files match Git. Live Chromium verifies current-session
controls/journal preservation and the original-hash offline archive with synthetic
transport. This does not establish installed-phone refresh.

## Review repairs: Polar continuity and terminal/update paths — 2026-11-17

Nativef5ea725c/tree df5d27e passes896 serial host tests, zero failures and6
intentional ignores; boundary scan and scoped/integration reviews pass. Actual
Runtime peers reproduce clock-gap/spontaneous-fault fallback before the fix and
retain Polar selection with unready evidence afterward. Real driver tests cover
all3 Polar methods and all3 diagnosed phases. Explicit recovery and owner/input/
configuration fences remain independently tested.

All14 updater suites pass. Actual helper policy, common terminal dispatcher,
generated writer/ACK continuation and separate cancellation worker pass focused
regressions including old cancellation/new ARM and same-process relaunch. Actual
110-source Android34/36 javac/D8, legacy/new handoff ABI and queued-exit checks
pass. Signed25-r2 passes1114 package checks, including39 repaired DEX instruction
paths and6 flow guards. Immutable24 and seven damaged instruction variants fail
the new guard. Source, signer, SDK207/helper3, controller-only permissions,
private audio/visuals and900-second session closure are independently bound.
Meta publication is in progress; no installation/physical acceptance is claimed.

## Review repair APK and held probe — 2026-11-17

Nativef5ea725c/tree df5d27e passes896 serial host tests, zero failures and6
intentional ignores; boundary scan, scoped reviews and integrated review pass.
Actual Runtime peers reproduce technical retirement before the fix and retain
Polar selection with unavailable evidence afterward. Real driver regressions
cover all3 Polar methods and all3 diagnosed phases, including spontaneous audio
owner faults. Explicit operator recovery and owner/configuration fences remain.

All14 updater suites pass. Actual helper policy, common terminal dispatcher,
generated writer/cancel-ACK continuation and separate cancellation worker pass
race regressions. The110-source Android34/36 javac/D8 checks pass; legacy/new
handoff and queued-exit ABI remain closed. Signed25-r2 passes1114 checks,
including39 repaired DEX method paths/6 compiled control/data-flow guards.
Immutable24 and seven damaged instruction variants fail the new guard. Same
signer, SDK207/helper3, controller-only permissions, private visuals/audio and
900-second configuration are verified. Meta build1464859736713038 is current;
malware and detailed security pass. No physical installation is claimed.

Version-only26 passes1229 signed-package checks with unchanged machine code,
DEX and payloads apart from explicit closed generated identities. It remains
unpublished until an actual repaired25 recording is RUNNING/rec:true and the
coordinating agent releases it for the deferred attended update test.

Release policy PR52/merge9e35559 reached exact Pages44138654869 success and11
checked public files match Git. The live browser accepts25 without an extra
update step and routes24 through the required update step. Current-session
Pause/Resume/Finish preserve the old setup journal and offline ZIP bytes still
match the original CSV hash. These browser checks use synthetic transport; they
do not establish installed-phone refresh or actual Quest installation.

## PWA startup update and Quest reconnect — 2026-11-17

`python -B tools/test-pwa-startup.py` runs the served HTML and production
browser gateway in Chromium with synthetic Bluetooth and signed Wi-Fi
responses. The new case initially observed the old address before the BLE
read. With the fix it checks launch update before BLE, the exact saved ID
with a nearby decoy, the changed address and persisted result, Android
companion discovery, cached Wi-Fi after BLE/update failure, radio recovery
without a tap, and no discovery on the CSV recovery route. The Bluetooth
prompt fits the measured 320×568 layout. Physical phone and Quest evidence
remains separate.

## Haupt-Runde preparation and original Arm failure — 2026-11-17

Passive inspection of the real installed phone confirms release20261117-auto-control,
native-confirmed green control, ready H10 streams, and the original Arm receipt
pending for minutes. Fresh native observations remain IDLE/g0/r0, pending:true,
rec:false. Neither Start nor Arm was replayed. The inspected status HTTP replies
succeed; CDP missing-body messages are observer lookup failures, not evidence of
a rejected connection. No recording-running gate was established.

The actual signed22 package contains two different Haupt-Runde plans: the shared
loop and the legacy authored scenario. The selected gateway supplies the former,
while generated participant validation previously selected the latter and threw
before JNI. The original shell failure route dropped ordinary Arm correlation,
leaving its pending receipt retained indefinitely.

Actual generated Java methods plus signed22 fixtures reproduce the rejection on
the immutable previous builder and pass28 checks after the repair. Missing,
mutated, malformed and extra-field plans still reject; legacy/nonselected routing
is retained. Validation is separated from JNI, so an exception after a possible
effect remains unknown. Correlated failure tests pass64 remote controls and134
real sparse codec/coordinator/Shell joins, including exact local-intent refusal,
durable checkpoint protection, captured headless epoch0, retired local UI, and
same-ID observation after a downgraded uncertain failure. Terminal27 and all four
Java host suites pass. Independent review closes the three findings. The PWA's
unchanged297-case suite also passes with zero skips.

Source d8998233/f24bd09a and the exact reconstructed public patch are qualified
software evidence. Code23-r1 is revoked as a Meta upload target. Repaired23-r2
passes1073 actual signed-package checks; Meta build1464364696762542 is current
and malware/detailed security passed. Installed23 exact APK SHA256, Meta installer/
initiator and new focused renderer process are observed. Root explicitly restarted
idle22 to fetch the repaired baseline, with no host install or subsequent23 launch.
SDK install dispatch and the helper fallback request are captured. No new reboot
or recording/update-during-session acceptance is inferred from that preparation.

Version-only24-r1 passed1071 packaging checks, but its matching DEX also retains
the subsequently proven30000ms emitted-request bug. It is revoked as an upload
target. The actual23 Arm is rejected cleanly with native IDLE/g0/r1, rec:false.
New qualification must exercise compiled Java requests against the900000ms
selected native parser, not only asset hashes and closure validation. Repair
source [PR44](https://github.com/GeorgeFejer91/viscereality.org/pull/44) merged30150da
as01e16db62987cb812810c0326e16c6609435f5cf; exact
[Pages44082868232](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44082868232)
succeeded and all11 checked public files match Git.

## Automatic phone control — 2026-11-17

The PWA requests a free native controller lease after authenticated connection.
Only fresh native ownership makes the toolbar control icon green. Hidden/stale
status, occupied ownership, manual release and uncertain command outcomes retain
separate states. Definite free-lease non-grants retry at five-second spacing for
the first three attempts and then every thirty seconds, including recording;
each attempt rechecks native status. Unknown requests retain their original
ID/owner/revision and are never replayed. Native admission is unchanged.

The final full Node suite passes297/297 with zero skips using the selected
native and Java integration. Nine policy cases and the real mounted UI cover
automatic acquisition, native-confirmed green, stale/hidden removal, occupied
leases, manual release, unknown fencing, reconnect and continuing fourth/fifth
requests, including during recording. Independent review passes. The135-state
rendered layout matrix passes, including320px reflow, enlarged text and stable
navigation. Input24, Meta27, pairing7 and control/location18 rendered checks,
startup, operator flow and four selection-race cases also pass. These fixtures
exercise real app code with synthetic transport; physical Quest/phone acceptance
is not inferred. Evidence is retained privately under the2026-11-17 auto-control
validation and visualization directories. Release marker:20261117-auto-control.

Publication: [PR41](https://github.com/GeorgeFejer91/viscereality.org/pull/41)
mergedb65ec14 ascabadb70f1b2c68d630e680472b432453d308d1a. Exact
[Pages44072213706](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44072213706)
succeeded and all17 checked public files matched Git. A public-runtime Chromium
check confirmed automatic free-lease acquisition, green native-confirmed control,
fixed/adaptive Polar and Controller selection with no page errors. Only the Quest
transport was synthetic. Installed-phone refresh is tracked in PROJECT.md.

## Native manual-state continuity candidate — 2026-11-17

Real Quest22 markers showed repeated Waiting/setup-input-not-ready resets to zero.
The source path also made fixed Controller State wait for120 accepted movement
frames from volume calibration (at least11.9 seconds, longer with holds/rejection).
The candidate starts phase classification directly and separates retained visual
diagnosis from fresh input readiness. Missing Controller/Polar data continues
inhale/exhale/hold at bounded frame rates. Owner/input changes retire the old
phase and confirmation namespace; activation/configuration rejection still resets.

Red-to-green regressions cover Waiting-to-zero and a new owner starting through
Automatic with reused numeric Polar lineage. Final native default host suite
passes888, zero failures, five intentional ignores, serially. Independent review
closes both lineage findings. The inherited parallel host suite has shared
TEST_SELECTED fixture pollution (also reproduced with the prior test binary);
the serial suite is the complete passing default gate. Showcase production check
passes; its host-test build has pre-existing ignored-test references to a module
configured out by showcase, so no showcase-test success is claimed.

Native453af6f4/treee1029a45 is clean and the1981101-byte cumulative source patch
reconstructs that exact tree in an isolated index. Actual selected Android release
compilation and signed-APK900-second session-closure checks pass. Independent
signed-package inspection passes920 checks, including signer, API24–27 signature,
16KiB ZIP alignment, exact capsule source and copied bytes. All private/audio
payloads and Java DEX/SDK/helper closure match22; only intended Rust and generated
identity/bindings differ. Meta distribution and wearer acceptance remain pending.

Source publication: [PR42](https://github.com/GeorgeFejer91/viscereality.org/pull/42)
mergedb0bbff0 as2385101f5ffa5bfcee37fc59bfc46d5b9bde62d9. Exact
[Pages44074293155](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44074293155)
succeeded; all11 checked public files match Git, including the full native patch.
The actual Meta22 baseline is independently reconfirmed by exact installed APK
SHA256, version and com.oculus.ocms installer/initiator. Wi-Fi ADB targets the
correct Quest2; code23 remains unpublished pending the user's running-session gate.

## Attended selection race and flicker — 2026-11-17

Real Pixel6a/Quest2 inspection confirms the phone loaded the published entry,
with native22 selected and IDLE/no recording. Authenticated status repeatedly
reported setup.ready:false/input_phase:unknown between healthy hold samples;
H10 ECG/motion and source/audio clocks remained ready. A method command received
the exact explicit rejected/accepted:false/stale-revision response.

The bounded selector re-reads status, pins the original actor/audio/attempt/input
and selected sensor, and sends at most3 newly identified commands, only after
that exact bound non-acceptance. Accepted/pending/uncertain results never replay.
Four real-app race fixtures cover stale→success, bounded failure, lost response
and changed intent. Additional checks retain local review through transient
ready:false while Start stays disabled and no session command is sent.

An actual idle-phone candidate test used only local runtime-module interception;
Quest transport, authentication and native decisions stayed real. Fixed Polar,
adaptive Polar and Controller were each natively confirmed. Adaptive needed
exactly two explicit stale rejections before the third ID was confirmed. The
same accepted ID was observed again by receipt lookup; it was not resubmitted.
No calibration, Arm or recording. Routine completion was hidden; the connected
phone reported a stable review instruction. The final CSS regression explicitly
compares enabled/disabled computed colours, border and opacity. The routine
detail no longer duplicates that instruction. Final native admission is unchanged.

Independent review passes54 domain/selection tests and all4 real-UI race cases.
The final full288-case suite passes with no failures or skips, including the
selected-sensor binding regression and selected native/Java integration. Re-run receipts are retained privately
under S:/Work/private/atmungsraum/validation/2026-11-17-pwa-input-layout/.
Current release marker:20261117-selection-retry-2. Publication and installed
phone readback are tracked in PROJECT.md.

Publication: [PR39](https://github.com/GeorgeFejer91/viscereality.org/pull/39)
merged4b73139 as0e06fa38d88ff1ee27a15b4708577d49aeff094d. Exact
[Pages44060980318](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44060980318)
succeeded; all15 checked live responses matched Git. Fresh Chromium loaded
20261117-selection-retry-2 from the public runtime with only synthetic Quest
transport substituted. Healthy completion stayed hidden, the first method
switch used three distinct IDs after two known stale rejections, and further
adaptive and Controller selections confirmed with no page errors. Pixel6a was
not listed by ADB at publication, so final physical public-runtime reload is
pending. No new native APK was built or installed for this PWA increment.

Controller source inspection: core.select_controller has no Polar-calibration
precondition. Java setupCommand handles Controller as a real first-loop action,
like calibration; native effective output still requires non-parked audio,
clock_valid/playing and a controller assessment no older than250ms. A direct
fresh-launch controller-only hardware check remains pending; do not conflate
selection acceptance with effective radius driving.


## Input confirmation and update step — 2026-11-17

User reports: method-switch status flashing, Controller remaining gold regardless
of the selection, version comparison always visible, neutral connection icon
and upward-offset ECG baseline. The reproduced presentation causes were the
ready-only hiding rule on operator-input-status and the controller-only enabled
border rule. Confirmed method and all selected controls now derive from the
unchanged validated native projection; readiness continues to gate actions.

Checks completed before publication:

- Full Node test suite passes with ATMUNGSRAUM_QUEST_REPO and the selected
  pairing Java classpath; the 34 focused layout/ECG cases also pass.
- test-input-selection.py reproduces the old hidden confirmed-method failure,
  then passes method switching, single mutation per choice, no document reload,
  stable status visibility across readiness drops, saved-link green status,
  required/current APK routing, and 24 actual Pretext renders of four input
  methods at 390px, 320px and 320px/200% text.
- Full --scope layout capture passes 135 rendered states across 390px, 320px,
  320px/200% text, short landscape and desktop, including native recovery.
- test-operator-layout.py passes actual app navigation, native scenario Arm,
  receipt recovery, participant drafts, session controls, Drive and history.
- test-meta-update.py passes 27 native state/receipt/update renders, preserving
  native admission, progress, open diagnostic focus and scroll at 320px/200%.
- test-meta-handoff.py passes six disconnect/reload/reconnect renders.
- test-pwa-startup.py passes cached launch, idle resume, guarded navigation,
  URL/storage retention and error/timeout fallback.
- Independent review passes product changes and the offline-allowed/required
  native update gate and focus probe. Its old layout-capture Settings routes
  were corrected for the new update page. Its automatic-update Back-loop
  finding was fixed and independently retested in the actual app; manual
  updates retain the prior preparation page.
- The requested Atmungsmessmethode wording and short purpose explanation are
  covered by the final rendered input matrix and broader layout capture.

Publication: [PR37](https://github.com/GeorgeFejer91/viscereality.org/pull/37)
merged implementation7b79208 as dbb29ee2093756c209aa54fc78744948ed5893c4.
Exact [Pages44052231564](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44052231564)
succeeded. All16 affected live responses matched Git. A fresh Chromium page
loaded the public runtime (only Quest/pairing transport replaced by the bounded
synthetic fixture), mounted release20261117-input-status, displayed the German
method wording and green link, skipped the current-APK page and switched fixed
Polar, dynamic Polar and Controller with one command each and zero page errors.
Installed phone/physical Quest confirmation remains an attended follow-up.

Evidence stays private under
S:/Work/private/atmungsraum/validation/2026-11-17-pwa-input-layout/.
This increment changes PWA presentation only. Real phone/Quest acceptance of
this release remains pending; no new physical observation is claimed.


## Actual automatic boot/update acceptance — 2026-11-17

Quest2 boot12→13 is bound to a changed kernel boot ID and serial. Helper launched
the actual Meta21 baseline without a host app launch. Live trace captures own
DownloadSuccess at02:25:57.714, a subsequently issued progress response at
02:25:57.815, SDK install dispatch at02:25:57.830, actual Meta22 replacement at
02:26:01 and helper fallback request at02:26:02.893. New PID6850 renders advancing
FOCUSED frames; user confirms stimulus and PWA22 without a manual update.
Installed22 exactly matches e0de22c7…/65,686,662bytes, Meta installer/initiator and
UID10159 retained. The final Android activity launcher is Quest shell; helper
dispatch is directly captured, without asserting it was the sole winning request.
No host main APK install or post-boot launch. Normal SDK lane passes. Queued
duplicate lane and the disconnected-phone notice have software coverage and
remain separate physical observations. Do not describe all wider app goals as
complete. The collaborator setup runbook describes this evidence and its limits.

Prior PR35/aa3882e passed exact Pages43992621746 and10 byte-matching live responses.
The final22 policy and handoff follow the same exact-publication checks, with
post-merge receipts retained in the owning validation run.

Meta21 publication: exact qualified21 artifact is current in Developer build
1463458816853130, Quest2/13+, Complete tests, malware passed and detailed security
All tests passed. The user-requested new chat recovered browser access; its
private receipt includes the app/channel/size/digest and saved screenshots.
No Meta21 installed-byte proof yet; actual headset remains20, cold-launched only
to request the Meta baseline. No main21/22 host install or host22 launch.

Public source PR34 is deployed through preservinga5376d4, exact successful
Pages43989652647 with27 byte-matching affected/core responses. Its earlier
811af1e Pages job was superseded/cancelled. This policy increment raises the
known required/latest floor to the now-available21; its focused comparison
checks cover20 blocked and21 compatible. Exact deployment is pending.

Monitor3 signed-package checkpoint: native1cd5ee3/treea9dc76d, source patch
SHA256c8756d19…/1,939,646bytes reconstructs the exact tree in an isolated index.
Code21-r2 passes169 independent actual-artifact checks; its strict source-closure
guard remains and now includes the previously missing SelfKiosk owner. The full
web suite passes280 with zero skips using selected native resolution and freshly
compiled real Java pairing interoperability. Helper3 independently passes50
Android boundary,27/170/43 policy and26 builder checks plus Android34 compilation.
Signed helper SHA256b98de54c… is installed with exact byte readback, retained
UID10000 and overlay Allowed. Initial setup reports target Ready/monitor Inactive.
No main21 or22 host install, new Meta delivery or monitor hardware success is
claimed. Publication and exact live-file verification of this source increment
remain pending; actual Meta21 baseline and21-to22 qualification remain open.

## Disconnected Meta update instructions — 2026-11-17

The read-only handoff model has11 tests; the combined Meta/model/control/gateway
suite passes34, and the web suite passes271 with nine native-assembly integration
cases skipped without their selected checkout. The actual mounted PWA passes six
disconnect/reload/reconnect states at390px,320px and320px/200% text, including
real Pretext measurement, readable document reflow and unchanged mutation count.
Fresh actual current APK21 retires the fixture hint; the fixture does not prove
a hardware installation. The existing27 Meta display/reflow states also pass.
Private evidence is retained under the active boot/kiosk/update validation run.
Independent review passes. PR33 merge c10d605 passed exact Pages43986515449;
all10 affected live responses match Git, and live Edge mounted the controller
with entry marker20261117-meta-handoff. Installed-phone refresh is pending:
the Pixel USB serial and its own DevTools forward are no longer available.

PR32 result/diagnostic release a037b2e passed exact Pages43984840599 and all18
affected live response byte comparisons. Main20 is Meta-installed but stopped;
new monitor3/21-to22 hardware qualification remains open.

Latest2026-11-17 checkpoint: Meta20 actually replaces19 after the idle app is
closed, with exact63f3d72f… bytes, Meta attribution and retained UID. No20
automatic relaunch or own SDK installation is proven. Its earlier manual check
initially returned offered19 despite published20, then offered20; StartDownload
failed and scoped OCMS logs rejected a duplicate same-package StartDelayed
request. A new cooperative queued-update handoff is software-reviewed for21;
final selected product build and physical21-to22 qualification remain pending.

The PWA result clarification, strict optional SDK error provenance, recovery
wording, aligned module graph and required/latest20 policy pass269 full web
checks with zero skips,27 focused checks,27 actual Pretext/reflow states and
independent review. The new accepted-message regression was red before the
correction. New SDK/queued states stay blocked and never claim a duplicate is
installed. Publication and installed-phone refresh are pending this increment.
The previous PR31/Pages43974010304 and all19 affected live-file comparisons pass.

The local browser H10 recorder has software review and a repaired asynchronous
device/wake-lock ownership race. Real H10 stream starts are acknowledged, but
the physical link closes; connection and recording acceptance remain open.
Its isolated private worktree/diagnostics own that follow-up, not this public
PWA release. Earlier checkpoints below retain their dated evidence limits.

Current repeatable-updater checkpoint: PWA v2/manual journal/action integration
passes266/266 zero-skip full web tests,21 focused Meta checks,15 real Pretext/
320px/200% render states and independent review. Native cycle/park/gateway software
and whole integration pass independent review. Helper2 source/policy/compiled
manifest checks pass; actual signed helper2 is host-updated on the Quest with
exact hash and retained UID/overlay. Actual installed main18 matches Meta bytes
and retains UID; it arrived through ordinary replacement and was host-launched.
Full SDK install/relaunch acceptance remains open. Candidate19's initial product
packaging exposed a Java lambda/Android bootclasspath compatibility failure.
Explicit Java callbacks correct it without changing updater policy; the actual
signed code19-r2 assembly and session closure now pass at native62ddb621.
Independent final artifact inspection passes: actual signer/SDK207/all-DEX,
helper protocol2/signature permission,16KiB owned SDK/JNI and ZIP closure,
all15 private inputs and79 original audio/guidance/session assets. Meta19 is
published as1463329980199347 in Developer with retained Quest2/13+ and passing
malware/security checks. Actual installation and physical acceptance remain
pending. Successor20 is also built and independently qualified; it stays
unpublished until the actual19 baseline is confirmed.
BootEntry12/110 and actual Android API compile also pass. Concurrent production
UI changes are preserved. PR30 ordinary merge7855177 and exact Pages43970922434
succeed;34 affected/preserved files match live Git bytes. Installed-phone refresh
is pending because the Pixel6a is no longer present in the ADB inventory.
See PROJECT.md and BOOT-KIOSK-UPDATE.md for current tasks and hardware limits.
Older checkpoint paragraphs below are dated evidence, not the current release.


Latest hardware readback (20:15): Meta has subsequently installed16. Its exact
APK digest matches the published16, installer/initiator are Meta, and Android
records lastUpdateTime19:43:44. The headset is asleep at readback. This establishes
a delayed Meta15-to16 transition after the earlier bounded negative boot window;
it does not establish immediate update-on-autoboot or precise continuously
observed installer timing. Post-update boot/kiosk remains pending. The user-
approved explicit in-app startup updater continues, with SDK-enabled17/18 next.

## Operator UI compatibility — 2026-11-16

The user subsequently confirmed that the legacy scenario and condition controls
are intentionally omitted from the smartphone UI. The Chromium fixture verifies
the three supported choices against mixed, main-only, legacy-only and missing
catalogs, plus a Quest with only track B ready. The PWA sends the selected
visible scenario with condition A. Missing admission produces a diagnostic and
zero session commands. The 135-state operator layout matrix and cached startup
release test pass from the `c14d67f` source baseline with this change. The Node
suite passes 244 checks and skips eight native build and Java prerequisites.

The original browser regression checks reproduced blocked Start for a legacy-only
advertised scenario, a missing scenario catalog, and a practice track with only
condition B ready. Recovery could hide the Bluetooth button and prevent a trusted
phone from reclaiming a free lease to read an old receipt. The recordings Drive
dialog closed during polling, and the visible Back link from `#daten` relaunched
the app and lost the patient draft.

`py -3.12 -B tools/test-operator-layout.py` uses the actual PWA and synthetic
Quest responses. It verifies older scenario IDs and condition-only Arm, mixed
catalogs with Haupt-Runde fixed to A, reconnect and local receipt acknowledgement
without replay, offline access to earlier browser exports, history and draft
preservation, Drive form values across polls, and visible live-recording status.
The existing runtime gates still require native receipt confirmation, current
preparation and current sensor evidence as applicable. No participant data or real
Drive authentication is used by these tests.

The Node suite passed 245 checks and skipped six that require native assembly
and Java pairing prerequisites.
Scenario JSON/CSV consistency, context and browser launch/update gates passed.
Chromium passed 135 operator states across five viewport/text settings, seven
pairing states, 18 control and Polar location states, 13 receipt recovery states,
and 12 Meta update states. Physical Quest, H10, installed phone and Google Drive
qualification still needs an attended test.

The 42-state operator walkthrough also reproduced a delayed native dialog close
event reclaiming focus after the operator moved to Bluetooth or the next session
button. The install and Drive dialogs preserve the new focus. Ordinary close still
returns focus to the opener. The walkthrough includes the saved Quest recording
and historical browser CSV download through their actual links.
Connection-page navigation enables after the operator module mounts so a slow
startup release check cannot present actions before their handlers are ready.

On the three-choice controller release, real-app Chromium checks reproduced
two more defects: Back from input stayed on input when Quest skipped the
participant form, and Back from earlier browser recordings returned to the
pre-run URL after native completion. Both checks now pass. The second drives
Arm → Start → Pause → Resume → Finish, with Start pending at the native cycle
boundary, and checks the exact commands, resulting screens and elapsed clock.
Returning to Recordings keeps the participant draft. Both checks verify visible
heading focus; navigation sends no additional commands. No participant data or
real accounts were used.

The APK comparison traced Bluetooth enrollment and lease, APK and kiosk status,
Polar/ECG, breathing setup and renderer data, scenario Arm and run receipts,
and recording export from the phone controller to the native contracts. The
code 18 source patch matched the SHA-256 recorded by the APK release. Code 17
and 18 expose the shared main scenario and private Meta update status. Their
session snapshot reports elapsed time but no total-duration field, so the phone
show elapsed time alone. Input diagnostics also render the older APK's raw
breathing value when no renderer field is available. The custom text spacing
test checks that the actual value stays visible and uses DOM measurement.
Independent Chromium review verified replay fencing, a new input observation,
modern renderer precedence, and suppression of the raw caption during recovery.
The Node suite passed 244 checks, with eight native build or Java integration
checks skipped on this host. Focused browser checks covered the 135-state layout
matrix, enrollment and controller handover, explicit and automatic Polar
recovery, startup cache updates, APK identity and Meta status, and audio recovery.

## Current boot/update checkpoint — 2026-11-16

Actual Meta-installed15 is confirmed by exact APK digest and Meta installer/
initiator, with retained UID. The separate same-signer boot helper is installed,
overlay Allowed and enabled by the user. Genuine reboot automatically launches15:
trusted native cold admission, first submitted frame, OpenXR focus and advancing
frames are observed. The wearer confirms single-menu recovery and protected
three-departure escape; native terminal/task-removal/retired-owner logs corroborate
escape. Helper27policy checks, signed package inspection and independent review pass.

Meta16 was published with the Quest off. Processing completes, malware passes
and security details say all tests passed. The next genuine boot still launches15.
A69-sample window over about208seconds observes no16; available version readbacks
are15 with validated Wi-Fi and charging;
later closed/asleep readbacks remain15. This is a bounded negative result, not
proof Meta updating never works or that kiosk causes it. No host install, host
app launch or manual Update established the test result. Both actual experiment
versions must continue through Meta distribution. Phone control stays with the
user, Location stays ON; normal main updates do not require helper updates.

The user now approves an official in-app startup check/download/install/relaunch.
Use the same [active contract](BOOT-KIOSK-UPDATE.md) extension tasks5–9. Native
Platform SDK207 has a16KiB-aligned ARM64 loader; implementation is in progress,
not built or hardware-qualified. Preserve signer, data/trust, private visuals,
controller-only policy and audio/calibration math. A shared native maintenance
fence must protect calibration, preparation and recording. SDK acknowledgement
is not installed-version evidence. Overall update acceptance remains open.

Verified split source: PR27 merge `4bae41fa3c0d4d30a10996e4e2c569419eefab38`,
exact Pages43921087088 success and18 public files byte-match Git. Public policy
now advertises available16/minimum11; installed15 is still compatible. The new
hardware-evidence/policy increment and SDK implementation await publication.
Historical checkpoint notes below describe their observation time.

## PWA startup updates — 2026-11-16

`python -B tools/test-pwa-startup.py` first failed on a cached launch that
stayed on the old release. With the checker, Chromium made one check when
current, switched to the published release from cached HTML, and kept the
query options, `#daten` and origin storage. HTTP 503 and a delayed newer
response both left the loaded version usable. No page errors were observed.
The real-app operator/history fixture, Node suite (six native prerequisites
skipped), context, syntax and Git whitespace checks passed.

[Pages run 43938500111](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43938500111)
deployed `726204db55b6defa01b47b3ad1cfe990e933a884`. Both entry URLs and nine
public files matched Git. Live Chromium confirmed a current launch made one
successful check without reloading. A seeded stale entry made one check and
switched to the live release with its original route. The lab Pixel 10 was
unavailable for a physical standalone check.

## Scenario selection — 2026-11-16

Chromium exercised the three choices with real radio navigation across the
Vorbereitung and Hauptstudie sections. The production fixture gateway received
`probe-anleitung`, `probe-5`, and `haupt-15` on Arm. The relevant controller,
connection, recording-sync and naming unit tests passed 38/38. The selector
fit at 320 × 568, 390 × 844, 844 × 390 and 1440 × 900. At 200% text it
used readable document reflow. Browser tests used simulated equipment.


## Playground header and automatic PWA refresh — 2026-11-16

Production picks up the playground presentation through `72a1920`. The
existing Quest catalog, calibration evidence, timer, receipts and recording
authority remain in use. The browser regression first failed to find the
shared header, then verified keyboard entry and return focus, one aligned row,
offline recordings, native command guards and the Drive dialog below the row.
The cached-launch regression also first failed on idle resume; it now checks
cached launch, idle return, the settings guard, 503/timeout fallback, and
preservation of the launch URL and browser storage.

The 320px screenshot also exposed duration labels outside the tile borders.
The rendered catalog test now checks the contents against their own tile
bounds. More space for the captions, smaller illustrations on narrow phones,
and a text-driven stacked layout keep the full labels visible. The original
layout and control checks were repeated after the correction.

Chromium passed the production controller walkthrough, 135 layout states
across 390px, 320px, 200% text, landscape and desktop, and 18 control and
Polar states. Settings and enlarged text can use readable document reflow.
The Node suite passed 244 with eight native/Java prerequisites skipped.
Scenario JSON/CSV, context, and Git whitespace checks passed. These are
browser and synthetic gateway checks; physical installed-device acceptance
remains open.

## Authored Quest audio package — 2026-11-17

The authored-audio branch stages seven scenarios and 96 files.
`assemble-apk.ps1 -BundledAssets -MainStudyAudio -ResolveOnly` bound the
catalog SHA-256 into the resolved feature lock. All 96 staged files matched
their recorded lengths and digests. The native patch reconstructed the
candidate source tree from the pinned upstream Git base.

The Quest JVM catalog reader accepted both the authored and legacy packages.
It wrote the original stereo Freie Erkundung bytes and rejected altered
exploration, calibration, Probe instruction, and Haupt-Runde outro media.
The Android media adapter compiled against the Android API. Node passed
302 web and audio tests including Java pairing interoperability. These
checks cover staging, media input and resolution. They do not exercise the
selected Quest session owner, audio output or a signed headset install.

## Polar calibration narration — 2026-11-17

The authored-audio candidate uses the verified calibration clip for the
first Polar setup. The selected Quest gateway holds the original request
pending during playback. On completion it rechecks the controller, Quest
owner, setup revision and Polar eligibility before asking the native
breathing loop to start calibration. Cancellation settles that original
request before a cycle starts. The legacy package follows its existing path.

The 320px Chromium check woke the real phone UI with the synthetic Quest
transport, kept narration pending for 35 seconds, verified the measured
cancel label and cancelled by keyboard. It confirmed zero calibration
attempts and one cancellation addressed to the original request. The
selected Java gateway passed 37 boundary checks including one-time
completion and cancellation on stale owner, failed playback and timeout.
The Android media sources compiled against API 30. The native patch
reconstructed from the pinned base and the feature lock resolved all 96
authored audio assets. Signed installation and headset playback remain open.

## Freie Erkundung recording duration and export — 2026-11-17

For the authored recording at 44,100 Hz, the 19,715,793 frames define a
447,070,136,055 ns completion threshold and the instruction ends at frame
960,768. The same limits govern the recovered plan and the live and final
CSV clocks. Rust passed 79 recording tests, including rejection of a changed
sample rate or frame count and a recovery round trip. The phone passed seven
data checks, including download of the dated exploration CSV with its two
original Polar streams. The PWA startup and reconnect check also passed.
Headset playback and round selection remain open.

## Freie Erkundung phone and Quest playback — 2026-11-17

The enrolled gateway advertises Freie Erkundung after the Quest verifies its
packaged media and participant recording plan. The media backend plays the
standalone track, pauses the setup loop and reports media position. Rust uses
that run's generation and captured position for progress and saves after the
natural-end event. Progress during pause does not change the recording clock.

101 Rust session tests passed with two ignored, including the authored
position, pause, resume, natural completion and final CSV integration case.
The Java harness passed 39 gateway checks and the Wi-Fi clock projection and
media backend checks. The Android media sources compiled against API 30.
Chromium passed the real phone flow with a synthetic Quest transport and
verified the choice in five viewport and text configurations including 200%
text. 306 web tests passed. Phone-pairing assembly resolved all 96 audio
assets in the feature lock. The authored candidate still needs a signed
build and physical Quest and Android PWA qualification.

## Guided Probe-Runde phone and Quest playback — 2026-11-17

The verified Quest package binds the 2,002,176-frame instruction at 44.1 kHz
separately from the 300-second guided practice. The phone reports the
instruction while the original start receipt is pending and the timed
progress stays at zero. On natural completion the gateway checks the
app owner, controller, update cycle, armed session and Polar readiness
before the official start. The Quest media position drives the timed run
and the Rust recorder saves at natural completion.

Chromium exercised the real phone flow over a synthetic Quest transport,
including 45.4 seconds of pending instruction without retransmission and
then the timed clock. All 306 web tests and 102 selected Rust session tests
passed, with two Rust tests ignored. Java passed 51 gateway boundary checks,
the media backend and the staged authored catalog. Phone-pairing assembly
resolved all 96 assets. Signed installation and physical playback remain
open.

## Authored Haupt-Runde playback and outro — 2026-11-17

The verified 39,690,000-frame FLAC includes both controls, instructions,
guided cues and music with its original fades. Its decoded PCM hash
matches the first 900 seconds of the 919-second review render. The separate
19-second outro retains the original WAV and is outside the recording clock.
The verified participant plan selects this source while existing packages
retain the continuous shared loop.

The Rust worker finalizes the recording on natural completion and authorizes
the outro only after a complete save. A late end while held remains
incomplete and does not authorize the outro. All 104 selected Rust session
tests passed with two ignored. Java passed 53 gateway checks, catalog
tampering checks, media backend checks and the generated outro cancellation
and duplicate-delivery checks. All 306 web tests passed. Phone-pairing
assembly resolved the 97 bound audio assets. Chromium rendered 140 layout
states and passed the controller and startup walkthroughs. Signed
installation, physical audio output and the installed Pixel app remain open.

The Android ARM64 source check also passed. The original site deployed
`118fa73` in [Pages run 44150919969](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44150919969).
Fourteen served resources matched the Git bytes. Chromium exercised the
deployed application at 320 × 640 and 1440 × 900 with a synthetic Quest
transport. The active signed release policy still points to Code 25.
Neither the lab Quest 2 nor the Pixel 10 appeared in the USB or wireless
ADB inventory, so device qualification remains pending.

## Quest-backed session preview — 2026-11-17

Code 24 source reconstruction verified the authenticated main-only catalog,
start frame and active clock; its Wi-Fi snapshot omits the total duration.
The real operator UI fixture covers that shape, missing totals for other
builds, pause and resume, painted particle size, out-of-order renderer data,
recording boundaries, and connection during pause. The focused browser
check passed seven viewport and typography cases; the broader screenshot
matrix rendered 140 states. Startup and reconnect checks also passed.

The web/domain tests and Java pairing interoperability passed 301 cases;
native assembly resolution passed another six in a plain-file checkout
using the reconstructed Code 24 source. Independent review found no
remaining release blockers for this PWA change. The browser fixtures use
synthetic transport; these checks do not qualify an installed Pixel or round
audio in a headset. The separate authored main, probe and exploration
playback work still needs a native audio path and APK qualification.

## Playground UI port — 2026-11-16

Ported the presentation from the playground through `60994c4` onto the
production controller. The production gateway, enrollment, native models,
audio, scenario IDs, export writer and Google Drive controller are reused.

The Node suite passed 230 tests with no failures. Six native assembly and Java
pairing interoperability tests require the patched native build environment.
Chromium ran the 45-state operator walkthrough and the 135-state layout suite
with the production JavaScript and fixture gateway. The focused operator test
also exercised Drive form loading and focus return, the red Polar diagnostic,
progress with a native duration, and history without native commands.

Standard 320 × 640 and 390 × 844 input states fit the viewport. The layout audit checked
320px portrait, 200% text, short landscape and desktop for unclipped measured
text, control geometry and complete native actions. Some very short,
enlarged-text and expanded settings states use document reflow under the
[layout contract](UI-LAYOUT.md#smartphone-gui-rules). This is browser evidence
with simulated equipment. Physical keyboard, hardware and OAuth authorization
qualification remains open.

The supplementary phone-pairing (7 states), control/Polar location (18 states),
and editor/caret/scroll (3 states and six live polls each) suites passed.
The 200% Drive popup at 320 × 640 retains complete text without internal
scroll; the surrounding setup, form and input screens still use document
reflow at that text size.

The official main/root Pages deployment
[43930833015](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43930833015)
succeeded for `4d5d3b3ee56e369a03911cfa859d297a20d4aa64`. All 14 changed
public file responses matched that commit exactly. The actual public URL
loaded the controller, four connection checks and measured labels, kept
Weiter disabled on an unpaired profile, and opened and closed the install
dialog with focus return at 320 × 640, 390 × 844, 844 × 390 and 1440 × 900.

## Meta-compatible main builds and one-time helper — 2026-11-16

After the user approved a separate boot helper, main15/16 were built from the
unchanged native source/private inputs/signer with the internal boot feature
opted out. Actual signed manifests omit overlay/boot permission, receiver,
action and metadata; the own-app kiosk manifest/unique DEX closure remains.
Both pass actual signed session closure, package/version/non-debuggable SDK,
89asset inventory, signer, v2/v3 signature and16KiB alignment checks.
Full13→15 resolved/compiled-manifest comparison confirms only the removed boot
selection plus version/run-derived identity. Full15↔16 normalized resolution
matches;90/98ZIP entries, all payload/audio and classes.dex match exactly.

The separate boot helper passes27meaningful policy checks after an observed
pre-implementation failure, SDK36 Java compile, PowerShell parse and signed APK
manifest/DEX/signature/alignment/hash checks. Independent task review reports
no actionable source or package findings. Root reran27policy checks and actual
signature verification. Only boot/overlay permissions and a fixed target query
are declared; setup opt-in and effective permission are required. Current
same-signer MAIN/LAUNCHER is resolved at boot with durable at-most-once admission.
No helper network, installation/update, arbitrary target or kiosk authority.

Actual Meta15 upload to the existing private Quest2 Developer channel succeeds.
Channel test status is Complete; malware requirement passes and vulnerability
details state “All tests passed.” Private screenshots preserve those outcomes.
This is not headset installation or runtime acceptance. Meta15 adoption is
pending user library action; helper is not installed/granted; no reboot has
occurred. Local16 stays unpublished pending the Meta baseline's boot/kiosk test.
Policy latest15/minimum11 reflects availability without inventing incompatibility.

## Autoboot recipe and local candidates — 2026-11-16

Explicit `-BootAutostart` opts into the existing native boot feature. Default
assembly remains opted out. Real resolver tests pass3/3 with zero skips; boot
gate passes12 admission plus104 receiver/observation/lifecycle controls; the
same-APK kiosk static/Java gate passes. Independent source/spec review reports
no actionable findings. Native source remains the clean code11 revision.

Actual Developer12 and13 APKs pass session closure, package/version/non-debuggable
manifest, controller-only selection, boot metadata/action/two permissions,
non-exported receiver and unique boot DEX definition checks. Original signer,
v2/v3 signatures and16KiB ZIP alignment pass;89assets remain. The private build
receipts and public APK record bind their exact bytes. No permission grant,
installation, reboot, wearer foreground/menu acceptance or Meta delivery can
be inferred from these gates. Code11 is still installed at this checkpoint.

The current authenticated Meta dashboard confirms the existing Developer
channel is private, holds code3 with one member, and does not require latest
build at launch. No account setting or channel build was changed. Candidate13
upload was authorized for the test. The user's subsequent correction requires
both baseline and successor to arrive through Meta, superseding the initial
local12→Meta13 acceptance sequence.

After explicit idle/USB readiness, local12 installation is confirmed by Package
Manager, exact remote APK digest/version and retained UID. This is not Meta
delivery. Attempted code13 upload then failed Meta's APK validation because
`android.permission.SYSTEM_ALERT_WINDOW` is unsupported. The current boot owner
requires that permission; no new channel build was accepted. Private screenshot
and install receipts retain the direct observations. No permission setup,
reboot or true boot/kiosk/update acceptance occurred. Code14 was cancelled
before signing; partial/unsigned artifacts are not a qualified release.

## Sequential operator layout — 2026-11-16

PWA-only presentation change: persistent connection/participant/signals/input/
calibration/tracking/session/running/recordings/settings/recovery views follow
the actual root site's code primitives. The native APK11, domain state owners,
audio,900-second main, recording bytes and all explicit commands are unchanged.
Local navigation never starts, calibrates, stops, claims or clears a journal.

Task review found and repaired legacy browser-Back visibility, contextual
unknown/failed APK status, active/recovery access to existing diagnostics and
release, and running ECG. Independent scoped re-review passes (68focusedtests
plus actual pending-journal/recovery/settings checks). Root full phone gate:
**235/235 pass, zero skips**, using the existing real Java/native prerequisites.
Real-app route regression passes: browser Back, local settings/recordings
return, persistent participant fields, zero setup/session navigation mutations,
running ECG, and active settings retaining the original clock/play/finish/release.
Independent whole-change re-review passes after history-return restoration,
visible native paused/finalizing/saving feedback and conditional screen flags
repairing a genuine telemetry scroll clamp. The original three-size/six-poll
edit/caret/scroll regression passes. A fixture review found the inherited fixed
recenter target obscuring scrolled fields; it now occupies document header
space. Independent320px/plain+200% checks pass at15offsets;28enlarged-text
screens/125actual viewport tiles pass region/corner/hit-target checks.
Strict desktop pending-settings checks additionally caught nested APK/kiosk
overlap: hidden-panel height locking is now prevented and actual aside/details
bounds participate in reflow measurement. Independent reproduction plus desktop
journal/scroll and320px/200% editor/caret regressions pass without commands.

The final combined capture command passes288states (140layout +148original),
70regression groups and636actual viewport tiles. Strict configured viewport,
painted text/Pretext fallback, nested panels, regions,48px targets, corner
intersections and click hit checks pass; no app errors, occlusions or missed
targets. Preview/calibration each read44statuses/four seconds, ordinary pages4;
ECG remains4, pending receipts4with original ID/no replay, max in-flight1 and
hidden reads0. Native actual50%/touching geometry and all mode/acceptance/new-
session/history cases pass. Reduced-motion quiet-window checks freeze only
interpolation and separately verify fresh native delivery advances the trace.
All independent reviews pass. PR24 ordinary-merges reviewed5519104 at4a351ecd;
their full trees match. Exact Pages43896368744 succeeds. All34 affected checked
live files match Git; public390px/1440px page/module loads, settings/return,
install-help interaction, cold legacy recordings and no-error/overflow checks
pass. Entry/style `operator-layout-r1`, UI45, resolver1 are live.
Installed-phone qualification remains pending
because the phone is unavailable; deferred new-mode wearer/cadence acceptance
is not implied by these software checks. See PROJECT.md for the active state.

## Physical yellow-display correction — 2026-11-16

Read-only standalone Pixel inspection proves UI43/setup8/display4 and current
CSS are loaded. Blue126px/gold122px at equal progress is the intended2px/side
inset. The phone is not running a stale PWA. Its actual18s capture has1079rAF
frames/89status responses, median54msHTTP, rendererage6ms and zero stale frames,
but repeated150–220ms interior-ramp pauses. The source is the original composite
replanning/oldest-support expiry under uneven replies, not absent native data.

The corrective candidate uses a bounded32-point adjacent GPU source-time history
with280ms presentation playout. Original absolute500ms endpoint expiries remain;
no desired-phase/actual-future prediction, freshness renewal or native authority
change is added. Preview-only reads are100ms/singleflight. Ordinary1Hz work,
source/reset/reduced-motion/visibility fences and physical percent are retained.
An anonymous89-triple timing fixture is independently regenerated from the
private capture. Focused25tests and full223/223 zero-skip gate pass; corrective
spec/quality review passes. The old5Hz exact replay improves34.51%stopped/290ms
maximum to7.90%/140ms; the10Hz captured-age model is1.34%/80ms. These are replay/
model results. Fresh supported brackets interpolate continuously; unsupported
gaps clamp/freeze and prevent a claim of universally continuous live motion.
Actual same-phone100ms confirmation is required after publication/user refresh.
No new APK/native, sensor, audio, classification or recording change is needed.
New mode wearer tests are postponed at the user's request.

The corrective release merged in PR22 at
`5b602970a37f14a4659dd5a9232ca3a42736287c`; exact Pages43877648195 succeeds.
All27 live Git-blob comparisons and public390px/1440px load/policy/install-help,
error/overflow checks pass. Remote context passes. NativeAPK11 is unchanged.
The user reports phone unavailable before the new physical capture. Its owned
CDP forward and phone/port leases are released; no observer remains. No actual
same-phone fix acceptance is inferred from the replay, model or public renders.


## Actual display and selectable Polar state modes / code 11 — 2026-11-16

Native `b1ba2c2b750dbff70488e351c4233b7cd62b73d5` and the combined phone
candidate pass separate task and final integration review. Root phone gate is
218/218 with zero skips; fresh actual Java JCE/state checks pass 369. Native
serial `--lib` gate passes 856, five intentional ignores, with isolated actual
GPU publication separately passing. Reviewer independently reran four preset
cases. Actual Java/native owner tests cover six scenarios; authenticated
HTTP/gateway/Java/native tests pass 45 assertions. Selected production Android36
Java compiles. The earlier non-lib host diagnostic's baseline cfg failure is
retained, not counted as a pass; actual selected Android APK compilation passes.

RED/GREEN covers missing modes/capability, gateway closed parsing, parked PCM
ownership and readiness retirement between command validation/application.
Preset modes retain actual accepted source/clock/quality checks, same confirmed
phase integration and truthful preset provenance. Both traverse the next-cycle
main boundary through exactly 900 seconds in host tests, without inferred
continuous volume. No physiological/wearer acceptance follows from these checks.

Display20 tests, 25 exact stroke-geometry cases, 26 actual Pretext states and
seven scroll/focus/cadence regressions pass. Synthetic browser timing confirms
preview200ms status coalescing, unchanged ordinary work, no hidden polling and
original-ID receipt reconciliation. Both renderer endpoints retain500ms expiry;
source/replay/reduced-motion/stale barriers remain. Native per-request status
and actual GPU publisher tests confirm no one-second producer cache. Physical
LAN/installed-phone cadence remains to measure.

The cumulative patch reconstructs tree `9a482a67f80342cb612b50112996488fac422c3a`.
The signed code-11 APK is 65,170,222 bytes, SHA-256
`71b4fc48ba2a4950e5d9b86204bd1426c1b1f2c80819e36430040daeb32a3b3d`.
Its source capsule, packaged identity, retained certificate, manifest, 89 assets,
16KiB alignment and actual signed session closure pass. Build identifier is
`770823055efe4f31d29af29b6f71d747e6bcfaa9f07b3ddf3c71b4b93f9107ec`.
Controller-only/private particle selection stays unchanged; boot receiver absent.
PR20 merged at `e9e8048d39843865a6db95697e0ecadba2524e72`; exact Pages run
43848286245 succeeds. All27 live files match Git; new cache URLs, release policy,
390px/1440px renders, no-error/overflow and install-help checks pass. Source/remote
context and native boundary gates pass.

After explicit no-recording confirmation, code11 is installed over Wi-Fi ADB.
Refreshed Package Manager reports0.2.10-dev/11, same UID10159; installed APK
SHA-256 exactly matches the inspected build. Current canonical serial/boot is
verified, Location3 remains unchanged and shared ADB is not restarted. Trusted
cold launch is accepted and Android focus belongs to the app. Controller-only
true, live hand/simultaneous false and initial driver0 are observed. Automatic
H10 preferred-candidate admission is recorded; no sensor quality/wearer pass is
inferred. The user confirms0.2.10-dev/code11 after refreshing/reconnecting the
phone and taking control. Installed-phone resource capture/cadence and physical
mode/display acceptance are still pending. No agent calibration/recording or
Meta upload occurred. The finite observer/owned child and leases are released.

## Operator controls and APK identity / code 10 — 2026-11-16

Native `695ef0d4a8c2a482b6ab3a6d37b21dee64f94770` and phone increments
through `4a473c9d13a79bc54d8b606d60c3b72fd1cc3c50` pass independent task
and final integration review. Final full phone gate passes 203 tests with zero
skips, including native assembly and actual Java pairing interoperability;
fresh JCE/state host checks pass 369. Native passes 846 library tests under its
documented `--test-threads=1` runner, five intentionally ignored; the isolated
actual renderer/global-producer case is separately run and passes. Earlier
parallel diagnostic failures are retained, not counted as passes.

Review corrections have real RED/GREEN proof: actual GPU preparation publication
had wrongly depended on a recording-only clock, now uses an independent validated
Android CLOCK_BOOTTIME observation; and a clean disconnect before connection now
rejects the original attempt receipt before independent retry can succeed.
Recording/PCM authority, learner and radius integration remain unchanged.
Real producer/gateway/auth, explicit Connect, kiosk/epoch/CAS, cold-start and
API36 production compilation gates pass. The new private APK-identity gate passes
52 signed-status assertions, including caching, unknown metadata and guest/relay
stripping. Rendered narrow/200% state, stale/fresh telemetry and targeted
scroll/focus checks pass. Kiosk guidance follows actual foreground-service
blockers; it does not generically require Accessibility.

The public patch exactly reconstructs tree
`ec3ad8be7eabe2b8878f58ce7cfc50e41ff78856`. Fresh selected Android compilation
and actual signed-APK session closure pass. Code 10 is 65,260,335 bytes with
SHA-256 `5bde14b95ed0afed49932fda84bc0a53271f2d65d0e735747de85dfa41580d93`;
its retained signer, manifest, 89 assets and alignment pass. The packaged build
identifier `7dfad0b71d545bf126f027de2e977c9688e28c9a1e18b3f2ab940247b0ae0114`
matches its source capsule. This is a packaging fingerprint, not the binary hash.
Native Android ABI compilation is verified. Exact installed APK bytes match;
UID 10159 is retained. Trusted MAIN/LAUNCHER startup is accepted, controller-only
is true, live-hand and simultaneous tracking false, and driver input is zero.
Without a phone Connect action the app discovers/connects H10 and receives both
ACC and ECG stream-start acknowledgements. Continuous sample quality and absent-
sensor retry are not yet physically qualified. A kiosk service record exists;
effective protection remains unproven by authenticated observation. No boot
receiver is registered. Location remains ON. The user ended testing and is
shutting down the headset; the finite startup observer/owned child and exact
Quest lease are released. Publication/live URLs, installed-phone refresh and
remaining wearer checks are pending. No Meta upload.

The secondary account normally merged PR 18 at
`f79800de0836cc2d8b37c4f75d2592df697b92ec`; exact Pages run 43838565290 succeeds.
All 27 checked live files match Git blobs, including new cache URLs, release
policy and native patch. Live 390px/1440px renders pass with ready Pretext, no
horizontal overflow, script errors or failed app requests, and install help
open/close. Screenshots are inspected. Browser plugin is unavailable; the
existing regular Playwright runtime is used. The private smoke script's initial
response-map assertion preceded the asynchronous release-policy response; it
now waits for the real response and both viewport checks pass without a product
change. The source/remote context gate passes. Installed-phone readback is
requested from the user, who retains phone control.

For resumed testing, Wi-Fi ADB returns the same canonical Quest-2 serial and
current boot identity through an authenticated shell. Code 10 remains installed.
The shared ADB daemon is unchanged and setup leases released. This is transport
confirmation, not worn OpenXR/calibration or effective kiosk qualification.

## Participant acceptance and follow-up scope — 2026-11-15

After the post-reboot code-9 calibration and native calibrated-Polar readback,
the participant reports that calibration works well enough. This closes that
subjective fit assessment; it does not qualify repeated recovery, Controller
fallback or a complete main recording. The next authorized increment repairs
automatic discovery/recovery, clarifies input-mode and reset UI, adds an actual
renderer-radius overlay and exposes effective kiosk status/settings. Source and
render verification of that increment are pending in the active PROJECT checklist.

## Explicit audio retry recovery / code 9 — 2026-11-15

Native `9afdcf0ef2cf05569e7ccc67c31f711f0e2fdd87` and PWA
`0ce7eda855fe5f30f86322bc0680cfaf5e93a6a9` pass independent spec/quality review.
Native842 passes/four ignored; the intentionally ignored stdio peer is exercised
through five Java-owner/native groups. Production HTTP parser/auth/gateway through
the real Java owner and Runtime passes 31 assertions with modeled Android/socket
boundaries. Broad gateway regressions, actual API36 owner and joint gateway
compilation, boundary checks and exact patch/tree reconstruction pass. The
final Java diagnostic fix changes no Rust bytes; the final native proof is reused
for that unchanged closure. Its new Java/native/gateway gates pass independently.

The phone suite passes 182 tests with zero skips, including fresh gateway/audio
fixture classes. Twelve recovery and thirteen receipt-fence rendered states pass
with actual Pretext, including 320px/200% text. Existing scroll/edit stability
passes three cases × six live polling rounds. A lost admission keeps its original
ID and uses typed read-only reconciliation; generation change alone cannot
acknowledge away uncertainty. Native capability controls unhealthy retries.

Review/regression corrections cover original ID/proven recovery target, real
gateway field stripping, unpublished old target generations, no late playback
after timeout/retirement and diagnostic generation provenance. The former cached
Polar projection test's 25ms assumption is replaced with a bounded wait for actual
worker publication, retaining its live and final assertions. Clock thresholds,
PCM, volume, learner, main duration and recording authority remain unchanged.

Fresh signed code-9 assembly/session closure, 89 assets, manifest, signer and
alignment pass. Installed bytes match; UID, runtime grants and persistent identity
are retained. Actual cold startup reports controller-only true, hand extension
false, simultaneous false, panel closed, gateway ready and input zero. Phase rates
remain 0.25 per second. Location stays ON; Wi-Fi ADB retained; startup observer and
lease cleanup pass. The combined phone/source record merged in
[PR 17](https://github.com/GeorgeFejer91/viscereality.org/pull/17) at
`73d9d896ecb3d44da304b4c9b52009ebc1dc7a8b`; exact
[Pages run 43756359679](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43756359679)
succeeded. All 24 checked public files match Git, including versioned phone
modules and the cumulative patch. Live 390px/1440px checks confirm ready Pretext,
no horizontal overflow, script errors or failed app requests. The remote context
gate passes. Installed-phone refresh and fresh wearer testing remain pending.
The initial physical stall trigger remained unproven at that release checkpoint.

Fresh code-9 wearer testing subsequently failed before either cycle at frame
zero: raw AudioTrack head stayed zero while real polls/dispatch and PLAYING
continued. Actual player replacement through generation 6 also made no progress;
typed recovery receipts correctly ended `audio-recovery-timeout`. Controller
selection with the observed menu closed also failed. The installed standalone
phone's new versioned resources were verified; first H10 connect became uncertain,
then a checked second request established both ECG and ACC. The user reported
the cold-connect waiting/failure recovery interface was unclear.

A private isolated silent static/primed-stream/focus comparison likewise failed;
the unchanged Horizon output counter and zero consumed queued frames support a
shared playback issue, without proving universal audio failure. User-reported
working system sounds and silent-probe/foreground limitations remain explicit.
Probe cleanup passed. After attended reboot, Wi-Fi ADB and Location ON were
verified, and the output counter advances again. Post-boot Controller audio and
fresh Polar calibration were pending at that restart checkpoint. The user then
independently connected Polar and started calibration, reporting success.
Native readback confirms ready calibrated Polar with detected inhale/exhale/hold;
observed unavailable/unconfirmed hold intervals retain their radius value.
Participant tracking satisfaction, repeated-fit/fallback and full-main acceptance
remain open. No product source change, replacement APK, main run or Meta upload
occurred during this diagnosis. Reboot recovery does not establish the original
intermittent playback trigger.

## Retained calibrated Polar state — 2026-11-15

Native `304be3ccf20e960a2a4bac7dad8ea9ca9bce17b1` passes 833 library tests,
three ignored, and all 164 phone tests pass. The real Runtime/driver replay first
reproduces a loss-value snap at 211ms, then verifies continuous per-render
integration across 180/225ms packet delivery. Inhale/exhale/hold show equal-time
parity at 72/90/120 Hz; measured tilt cannot assign radius directly.

Review found and corrected two cache edges with RED/GREEN regressions: a valid
Hold/Exhale followed by an invalid packet tail before any render, and pre-barrier
cache reuse after rate/slot/activation changes. New valid post-barrier evidence
still survives an invalid tail. A separate monotonic confirmation revision and
lineage-bound fence preserve both rules. An inert unavailable-cursor mutation
also fails before restoration and passes afterward. Runtime29, driver13,
format, boundary and focused static checks pass. Independent spec/quality review
is clear. No source change followed the final full library pass.

The cumulative patch reconstructs tree `2b702ebeda4b69faadc97c3716f9db956d759a1f`
exactly from the pinned upstream. The pre-existing non-test host cfg failure and
broader exact-token static-gate mismatch remain documented baseline limits;
they are not claimed as passing. Signed APK assembly/inspection, installation,
publication and fresh participant acceptance are separate gates. Code 8 assembly,
actual signed-APK session closure, 89 assets, manifest, certificate and ZIP checks
now pass. Installed bytes match exactly; UID, runtime grants and persistent app
identity are retained. Fresh cold startup captures controller-only true, hand
extension false, simultaneous false, panel closed, gateway ready and input zero.
Configured phase rates remain 0.25 per second. Location stays ON and Wi-Fi ADB is
retained; initial observer/lease cleanup passes. The source record merged in
[PR 15](https://github.com/GeorgeFejer91/viscereality.org/pull/15) at
`b735f196b90ebe4d7dcccf0c446baf6886105849`; exact
[Pages run 43745201300](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43745201300)
succeeded, all 24 live files match Git and the remote context gate passes.
Fresh wearer Ready was then received, with the user controlling the phone.
Both H10 streams became ready after a checked reconnection. The actual attempt
failed `clock_stalled` at frame 270,950 (6.143991 seconds), with 150 accepted points
and four motion masks. Twenty-three failed snapshots span 20.976 seconds at that
same frame while playing remains advertised. Recorded retries are rejected
`clock-stale`. This does not test calibrated Polar rendering: no fit completed.

Source/evidence diagnosis confirms the native core would accept plausible forward
clock observations after failure; the failed attempt itself stays terminal. The
Java retry path never reconstructs an already-started active player, so an
unhealthy transport blocks retries indefinitely. Original sink/head/publication
stall cause is not established. AudioService zero stream volume and the silent
hold interval are qualified leads only. The source-clock-ready boolean is a
retained sensor anchor and does not attest ongoing Java clock publication.
Bounded raw-clock diagnostics and explicit pre-main recovery are the next repair.
The app is stopped and owned observers/forward/leases released, Location kept ON.
New calibration requires fresh Ready. Tracking, retry/fallback and the full main
run remain unqualified. No Meta upload.

## Participant-led calibration and controller-only code 7 — 2026-11-15

Three new waveform cases first failed the ideal-template check, then passed
after its removal: nonlinear breathing, changing holds and a different second
cycle. No recurrence/compliance gate replaced it. Core50, focused72 and native824
checks pass, three ignored. Existing clock/source, missing/invalid-data, coverage
and range guards remain. Independent production/test/document review is clear.

Controller-only verification uses actual default/selected resolver outputs,
negative controls and compiled production input/extension selectors. Both actual
Developer and rehearsal assembly tests pass. A real build then exposed an older
original-renderer simultaneous-input restriction. The narrow fix admits exactly
one locked mode and preserves renderer/background/panel and all other constraints;
32 cases, legacy page checks and the actual non-resolve builder path pass.
The successful fresh build retains the normal locked preflight.

The signed code-7 APK passes source-tree reconstruction, 89-asset/session closure,
manifest, signer and ZIP checks. Installed bytes and retained identity/grants
match. Actual running flags are controller-only true, live hand tracking false,
simultaneous input false and app panel false. HTTP gateway readiness is confirmed.
The unworn capture is black; physical rendering, participant satisfaction,
retry/fallback and the full main experiment remain unqualified. Location stays
enabled as requested; no Meta upload. See the [exact APK record](../quest/apks/README.md).

The source record merged in [PR 14](https://github.com/GeorgeFejer91/viscereality.org/pull/14)
at `2fbf5306cb8876bce52c8b87a70e838ca41c7c03`. Exact
[Pages run 43737821830](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43737821830)
succeeded; all 24 checked live files match Git. After fresh wearer readiness,
the actual USB Pixel resource inventory verifies the new brand CSS/app and
current setup modules after refresh. The attended attempt then completed both
cycles (1,411,200 frames, 800/800 accepted points, zero quality masks) and selected
Polar tilt with audio generation unchanged. The wearer confirmed the stimulus
after dismissing a Quest prompt, but rejected its visibly jumping radius.
Sparse private telemetry contains 32 not-ready observations among 255 snapshots
after successful calibration, with source and audio generation unchanged. The
code-7 Waiting branch reset the integrator to its loss value; the code-8 repair
is verified above and wearer acceptance remains open. The code-7 app and owned
observers were stopped, its phone forward and leases released, Location retained ON. Fresh
wearer Ready is required before another calibration; no main run or Meta upload.

The real trained Runtime-to-driver regression reproduces the reset with 200 Hz
samples in 36-sample batches at 180ms cadence plus 45ms jitter: at render 19
(211ms), a readiness lapse resets integrated progress from 0.30000013 to the
test loss scalar 0.2. The user's required correction retains the last valid
breath state, rather than freezing radius. The repaired replay advances at the
configured rate on all 117 render frames, including unavailable intervals,
without reset or packet catch-up. Barrier review and build subsequently passed
in code 8; fresh wearer acceptance remains open.

## Core-site PWA styling — 2026-11-15

The core site's rendered reference was inspected in IAB and retained privately.
The PWA matches its black/white Arial typography, centered bold headings, white
VR figure, hairline borders, blue primary actions and restrained gold accents.
Measured white-on-blue contrast is 6.43:1, hover 4.98:1; gold carries no text.

The final corrected sweep passes 132 screens and 30 regressions with zero app
errors, horizontal/painted overflow, missing measurements, undersized targets or
internal-panel overflow. Checks include 320px reflow, 200% text, WCAG spacing,
focus/disabled controls, participant entry, ECG, reduced motion and preserved
scroll/edit state. Independent review and direct reference/screenshot comparison
are clear. No brand/title Pretext no-fit remains. This is rendered software
evidence; physical installed-phone verification remains separate.

The style increment merged in [PR 13](https://github.com/GeorgeFejer91/viscereality.org/pull/13)
at `0a3451656351d00391b6ca1d9d55bd2805b1d960`. Exact
[Pages run 43736737421](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43736737421)
succeeded. All 24 live files match Git, and actual public-URL phone/desktop checks
pass with no app errors. The installed Pixel was subsequently refreshed and its
exact brand/app/setup resource versions verified during the code-7 test.

## Parked preparation and full-window repair — 2026-11-15

Attended code 6 now proves progressing, full-duration collection: 1,411,200 frames
and 795 accepted training points, with zero quality masks. Both actual streams
were ready with automatic recovery active; the bounded diagnostic identifies raw
type-1 200 Hz ACC. Parked native readback was frame zero, input zero and stopped
audio. The final result was `poor_guided_motion`, after coverage/range checks;
the user confirms following audible cues. This is a failure of the ideal-template
acceptance policy, not a passed personal calibration. The latest user ruling
removes waveform compliance checks and leaves satisfaction to participant testing.
Controller-only composition is also newly required. Both repairs are in progress.

The code-6 source record merged in
[PR 12](https://github.com/GeorgeFejer91/viscereality.org/pull/12) at
`e3b4573cc7fb3107912b3959c14660c78ef762de`; exact
[Pages run 43733535771](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43733535771)
succeeded. All 24 live file checks match Git and the remote context gate passes.
The physical phone was refreshed and showed new controls/warnings; a separate
module-version query failed after USB disconnection. No current claim of exact
phone resource-version readback or successful wearer fit is made. The preview
was stopped and run-owned observers/leases released. Location remains ON at the
user's explicit request. Meta remains on hold.

The attended code-5 test reproduced immediate `reordered_sample` and
`excessive_motion` failures with zero completed frames. Focused probes show that
callback-backdated batches can reorder despite continuous sensor timestamps,
and a 5 ms derivative can classify small direction noise as excessive movement.
These are software reproductions, not a measurement of the worn sensor's exact
frame format. Original raw acquisition records remain untouched.

The current phone increment passes all 164 Node checks with zero skips,
including actual Java/WebCrypto pairing and real native assembly resolvers.
The 56 focused display/setup/transport checks pass. The full rendered fixture
sweep passed 132 screenshots and 30 regression records with no app errors,
including actual Pretext, 320px, 200% text and WCAG text spacing. Independent
review's remaining counter-bound finding was corrected: all five optional
quality counts are integer 0–800, each with a negative test above the bound.

Java audio owner/backend checks and actual API-36 compilation pass. A new
regression first reproduced a monitor acquisition bypassing the command timeout;
the repair enters the bounded request before taking the owner monitor. Tests
cover parked initialization, no lifecycle first-start, accepted explicit start,
retry without replay, queued cancellation and admitted uncertain outcomes.
Independent scoped re-review found no remaining audio issue.

The phone increment merged in [PR 11](https://github.com/GeorgeFejer91/viscereality.org/pull/11)
as `578d330863ce0d3726417e5ee0c14d01df636393`. Its exact
[Pages run 43729861453](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43729861453)
succeeded; all 23 live file checks match Git, and public-URL phone/desktop smoke
checks report zero app errors. The remote context gate passes.

The focused native setup suite passes 69 checks, including the full packet /
clock-pair / jitter / noise / 32-second fit path. Independent reviews identified
and closed slot-counter bounds and a pre-first-packet warmup bypass. Diagnostics
now count bounded 25 Hz slots; absent metadata cannot imply a ready clock, and
explicit compatibility evidence must retain source, receipt and measured-pair
freshness. Selected Android acquisition remains packet-only.

Required-stream recovery passes 47 production controls, source scope 24 and the
bounded diagnostic 15. The full authenticated gateway suite passes, including
new enrolled-relay and guest exclusions. Review also caught and closed stale
pre-Stop data reviving a stopped stream. All 14 actual API-36 compiler source
hashes were rechecked. Original codecs, timestamps and recording bodies remain
unchanged.

The full native suite passes 821 tests with three ignored and no failures. Three
selected private-profile checks and the packaged-default reset regression pass.
The source is frozen at `f31156d7832f71b7caeff9ad005c311ea77c0c8e`, tree
`c095a8df92480b6da4c7810ed3332bf3998667c2`; independent reviews are clear. The
cumulative patch reconstructs that exact tree from the unchanged pinned upstream.
The [code-6 APK](../quest/apks/README.md#025-dev--parked-startup-and-full-window-quality-2026-11-15)
passes actual signed-package closure, manifest, signer, 89-asset and ZIP-alignment
checks. Installed byte hash, version and retained UID/grants match. The persistent
HTTP app identity is unchanged and ready; native frames advance with effective
app-panel visibility false. Meta system Settings remains over VR from the known
Location handoff, with system Location still OFF. Actual wearer silence/minimum,
installed-phone refresh, ACC format and fit/retry/fallback remain pending. No
calibration was started and no Meta upload occurred. Source publication is the
next software gate. Do not interpret fixture progress as actual calibration.

## Stable controls and smooth live displays — 2026-11-15

The exact unmodified baseline reproduces a scroll clamp from 300px to 105px
on each telemetry update, with unchanged navigation origin, form nodes, focus,
caret and draft values. The repair keeps live text updates in the current layout.
Chromium passed 112 rendered screens and 17 regression records with zero app
errors: repeated poll/edit/scroll checks, nine explicit missing-stream recovery
flows, and live display cases across phone, desktop, 320px and 200% text layouts.
The harness uses DPR 2, the actual locked Pretext runtime and production UI.
The final Node suite passes all 156 checks with zero skips, including actual
Java/WebCrypto pairing and both real native assembly resolvers. The focused
re-review passes 48 presentation/transport tests. Context and diff checks pass.

[PR 9](https://github.com/GeorgeFejer91/viscereality.org/pull/9) merged at
`e68452f6467a7b2a838f5f29165e6d382cb38eec`; the exact
[Pages run 43722083291](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43722083291)
succeeded. All 23 live runtime/source/document requests matched committed Git
blobs, including every changed cache-versioned import. Actual public-URL browser
smoke checks passed at 390×844 and 1440×900 with no app errors or failed app
requests and with real Pretext measurement. The context remote gate passed.
Physical phone refresh remains pending while the user controls that device.

Live checks prove breathing movement between native observations, fill bounded
by confirmed progress, pause, delayed/replayed status expiry, reduced-motion
expiry without animation, and background suspension/recovery. They also require
the actual SVG to be visible, moving from buffered received ECG samples, with
gaps preserved. A transient status outage cannot revive duplicate ECG; newer
data recovers it. No animation frame sends a control command or adds polling.

Independent source review found display freshness and SVG reflection defects;
these were corrected and the focused re-review found no remaining P1/P2 issue.
Actual wearer calibration, installed-phone refresh and a full 900-second run
are separate hardware gates and have not been claimed by rendered fixtures.

The combined [local code-5 APK](../quest/apks/README.md#024-dev--combined-showcase-particle-update-2026-11-15)
uses native `6cd05d6a` over the buffered-calibration source. The cumulative patch
reconstructs its exact tree. Native tests passed 812 with three ignored; actual
profile/reset checks and 71 permission diagnostics controls pass. Signature,
16KiB ZIP alignment, manifest, library and actual-APK closure checks pass.
Installed bytes match, the app panel is closed during visible VR startup, and
the gateway is ready. No Meta upload or worn visual/performance acceptance is
claimed. Pre-launch log capture distinguishes working startup dispatch from
the still-failing attended Location-settings handoff; post-hoc logs can lose
that interval amid renderer output. System Location was restored to OFF.

## Startup permissions and buffered calibration — 2026-11-15

The local code-4 repair passed 131 Node tests with zero skips, including actual
Java/WebCrypto pairing and both five-source assembly resolver checks. Native
testing passed 809 with three ignored and zero failures, plus 63 focused setup
tests. Regression cases reproduce idle revision churn and valid buffered-data
rejection before the fixes, then prove accepted commands, bounded historical
alignment, final-window evidence, pause/source barriers and genuine loss failure.

The permission coordinator passed 31 lifecycle checks and compiled negative
controls, with 45 approval UI and 24 Location projection checks. Actual current
Java compiled against API 36. Independent reviews found no blocking issue in
the frozen permission and timing commits. Chromium passed 18 targeted control/
Location states, including 320px and 200% text. A negative rendered control
reproduces the stale post-Recenter status; the repaired release instead prompts
for control. Location OFF blocks scanning, while connected streams stay usable.

The private code-4 APK, exact source tree and patch are recorded in the
[APK log](../quest/apks/README.md#023-dev--local-permission-and-timing-repair-2026-11-15).
Signatures, manifest, 74 catalog/audio files, ARM64 loader and actual-APK closure
passed. Installed bytes matched, followed by actual closed-panel VR startup
and a ready gateway. Runtime permissions had been pre-granted for this bench
test under explicit user authorization; fresh dialog handling is not claimed.
Actual callback timing, wearer fit and the full 900-second trial remain open.

[PR 7](https://github.com/GeorgeFejer91/viscereality.org/pull/7) merged at
`c41c651482eec6781789c1f767c5a2073d9060f5`; its exact
[Pages run 43714495396](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43714495396)
succeeded. All 20 checked live requests matched committed Git blob bytes,
including app startup, the actual versioned imports, source patch and updated
timing/permission notes. The remote context gate passed. Code 4 was not uploaded
to Meta; local wearer acceptance remains pending.

## Phone pairing and closed startup — 2026-11-15

The selected Developer source uses read-only Bluetooth discovery, explicit
app-installation identity when Android cannot supply hardware serial, and
two-sided matching-code buttons. Its initial panel visibility is compiled
closed; startup and idle polling cannot open approval. Only a verified
commitment opening and proof create the wearer prompt. Multiple independent
memberships share one active control lease. Reconnection does not claim it;
the phone offers explicit Claim/Release and waits while another phone owns it.

All 130 Node checks passed with zero skips, including both real assembly
resolvers and complete WebCrypto-to-production-Java begin/reveal/confirm/status/
decrypt flows in both approval orders. The focused crypto gate passed 369
Java checks. Gateway qualification includes 43 selected v2 integration checks,
34 actual legacy-BLE admission checks, 31 registry and 95 discovery checks,
alongside legacy, deferred-effect, recording and protocol regressions.
Selected public BLE mutations are rejected at admission and queued dispatch;
internal gateway receipts remain available.

Native UI passed 45 controls, including cold startup with no pending request,
phone cancellation, expiry, replacement owners and prompt-owned visibility.
Current production sources compiled against Android API 36; source-hash joins,
strict selection, manifest/schema defaults and the native shell policy passed.
Independent source review closed the initial nonce-encoding, adaptive-code and
legacy-BLE authority findings. This is a bounded review, not a formal protocol
proof. The unshipped v1 pairing handshake has no fallback.

Chromium rendered 78 states with 48 reflow checks, then 12 focused states after
the occupied-controller Claim button was disabled. There were zero app errors,
text/panel/document overflow failures, missing Pretext measurements or controls
below 48px. Checks cover real button confirmation, cancellation/rejection focus,
approved-key retention after a lost join, saved reconnection without claim,
handover in the breathing preview, 320px, enlarged text and long labels.
Private galleries and machine receipts remain in ignored output.

The cumulative public patch reconstructs native tree
`8bbc4937ab0fd9ab97d0d6c751dc624260ec0dbe` exactly from the pinned upstream.
Its SHA-256 is
`5059f3c5ac0ba9c20a7d7075ba007cc1a94a99b4a6a4777cd3a3475dc36df9c5`.
APK qualification, exact installed bytes, rendered headset approval, actual
phone discovery/reconnection and wearer calibration are separate gates in
the [active checklist](PROJECT.md#quest-2-calibration-acceptance--2026-11-15).

[PR 5](https://github.com/GeorgeFejer91/viscereality.org/pull/5) merged at
`90f2c759a0991a6b926a7bcaacc520b4c77e8d28`; its exact
[Pages run 43706732014](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43706732014)
succeeded. All 18 live runtime/source/document requests matched their committed
Git blob bytes, including the actual cache-versioned imports. The remote
context gate passed. Actual Quest 2 startup rendered VR with its panel closed.
The refreshed Pixel found the headset and the user confirmed matching codes
on both devices. This user-observed physical flow is separate from the
synthetic capture suite; saved reconnect, handover/revocation and wearer
calibration remain open. H10 setup exposed missing Bluetooth Scan permission
without a working startup/native request path. The user explicitly authorized
pre-granting that permission on this installed APK to finish testing; readback
confirmed the grant. Automatic attended startup permission handling remains a
required fix and fresh-install gate. Meta code 3 passed its malware/security scans and the owner
is joined to the private Developer channel; remote replacement remains untested.

## Shared breathing setup release — 2026-11-14

The selected developer app replaces the earlier simulated phone preview and
embedded timed preparation with one native audio-frame schedule. Automatic is
the startup input. Preparation loops 4–4–4–4 indefinitely; calibration starts
on an inhale boundary and requires two complete cycles of valid Polar tilt
evidence. Success activates the fitted model, retry returns to Automatic, and
Controller State supplies the fixed-orientation fallback. The selected input
continues through the next full cycle boundary into exactly 900 seconds of main
playback. Nonzero guidance bias is unsupported in this selected path.

The final native source passed 798 host tests with three ignored, plus the
36-test standalone core suite. All 116 Node checks passed, including both real
APK resolver assembly checks. Chromium rendered 59 states with the real phone
controller and synthetic native-shaped status: native frame-driven progress,
retry, failure, ready, Controller fallback, frozen frames, reload/re-enrollment,
old APK compatibility, Bluetooth rediscovery and playback controls. Thirty-four
reflow checks included 320px, 200% text, text spacing and keyboard focus. There
were no app errors, actual overflow, missing measurements or undersized targets.
These fixtures do not establish physical sensor or audio alignment quality.

Independent reviews covered the native runtime, Java audio owner, enrolled
gateway, phone UI and packaging. Regression checks exercise source-clock
mapping, rejected and retired sensor evidence, retry model lineage, command
timeouts, H10/frame loss, resumed Controller readiness, exact recording cutoff
and incomplete-run finalization. Compilation checks use the actual Android SDK
boot class path. The signed private developer APK matches the exact source tree
and patch in the [release record](../quest/apks/README.md#020-dev--separate-developer-build-2026-11-14).
Its 74 audio/catalog files, ARM64 OpenXR loader, five JNI exports, closed
manifest fields and v2/v3 signatures passed readback. The private Meta Developer
channel accepted code 1 and passed malware and security scans. Owner invite
consent is pending in the separate distribution chat; the store is unpublished.

[PR 3](https://github.com/GeorgeFejer91/viscereality.org/pull/3) preserved the
source history and merged at `efaed6c13b17305ccd1e9c6ed7a5227330f9e179`.
Its exact [Pages deployment 43647565252](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43647565252)
succeeded. Eleven live files matched committed Git blob bytes by SHA-256:
HTML, the five versioned JavaScript/CSS requests, `audio/box-loop.json`,
`quest/native-source.patch`, and the experiment app/provider/session inputs.
The loop digest is `b50ce8853b9d097b904cc90043e580a75ce17662e1ff2c985bb08b7df45a9ece`;
the native patch digest is `fa5905c0a1f923989d2544d34bef01bb7b5b44acd2e1277610567284b458317d`.
The context checker passed with `-RequireRemote` at the merged revision.

The user explicitly requested finishing the build without hardware testing.
No headset was installed, launched or changed during this feature release.
Worn-Polar fit/retry quality, actual audio/visual phase alignment, Controller
fallback on a wearer, a full 900-second Quest 2 recording, installed Pixel
refresh and remote version-to-version replacement remain pending. Earlier
hardware observations below describe their named older releases.

## Reconnect retry and lab bench sweep — 2026-11-14

The Chromium screenshot suite uses the real controller and a synthetic
gateway. Its new regression puts an unreachable address in the saved pairing,
then reports companion states Bluetooth off, searching and ready. Before the
fix the searching status did not appear within the two-second retry budget.
With the fix the UI observes each transition, joins the advertised endpoint,
and reaches participant setup with only one attempt at the stale address.
The journey also issues Arm, Start, Pause, Resume and Finish through the
rendered controls and checks each operation appears exactly once.
The full 43-screen run, 91 Node tests, six scenario pairs, Python asset test,
and project context check passed.
The participant form reflowed without horizontal overflow across five
portrait, landscape and desktop sizes, including enlarged text. After reload,
the fixture CSV download matched its expected filename and all 57 bytes.
The silent offline diagnostic rendered all six audio timelines at 44.1 and
48 kHz with zero sample error, including the exact 900-second Haupt-Runde.

On physical hardware, the existing enrollment rejoined in 599, 273, 263,
279 and 321 ms on five standalone Pixel reloads, from initiating navigation
to the visible participant control. This timing includes the browser driver
and is distinct from command latency. Quest 2 APK 0.1.29 passed signature and
installed SHA-256 checks. Sampled idle OpenXR scorecards reported 71.7–72.2
frames/s, 1.18–1.73 ms of submitted-frame host CPU time, and zero stale
frames. Fresh physical ECG, native command latency, and recording delivery
were gated on an attended run; the Quest reported its panel open and ECG
unavailable. See [the active checklist](PROJECT.md#start-to-finish-qa-and-reconnect-latency--2026-11-14).
On the same Wi-Fi, 20 signed requests for idle Quest status had an 80 ms
median and 128 ms p95 round trip. The installed Pixel also accepted a
synthetic participant ID and retained the scenario gate without live Polar.
After the exact Pages deployment in PROJECT.md, the new PWA runtime passed
five standalone reloads on the Pixel. Physical playback response times
remain separate from these status and navigation timings.

## Guided phone breathing preview — 2026-11-14

The phone prompts inhale, waits four seconds, then prompts exhale and
waits four seconds before showing the green Continue action. Browser clock
checks verified that neither phase can be skipped or completed early.
The next enabled control receives keyboard focus when the operator stays
in the preview. A Polar readiness interruption resets an unfinished
preview. The synthetic Quest received zero commands from the breath
controls. Scenario selection and playback controls still work afterward.
Portrait, landscape, desktop, 200% text, expanded text spacing and both
motion preferences passed. An independent real-time Chromium check
observed the grow and shrink transitions. All 88 Node tests, six scenario
pairs and the project context check passed.

## Phone breathing preview — 2026-11-14

Mobile Chromium with synthetic Quest state verified the preview gate,
keyboard and focus order, reduced-motion circle changes and Continue. An
unavailable default A track still allowed the operator to choose B. The
fixture checked that breath buttons issued zero Quest commands and that
scenario selection, Start, Finish and participant editing still worked.
Recenter kept the preview visible and disabled Continue while awaiting its
receipt. A separate paused-rejoin fixture kept Resume and Finish available.
A 320 × 568 minimum, 390 × 844 portrait, 568 × 320 landscape and
1440 × 900 desktop fit the measured controls without document overflow.
The 200% text and expanded spacing checks retained full text and controls.
All 88 Node tests, six scenario pairs and the context check passed.

## Full Haupt-Runde on Quest 2 and Pixel — 2026-11-13

Quest 2 ran signed APK 0.1.27 in Remote. The installed Pixel 10 PWA used
release `20261113-polar-only`. The operator entered a synthetic ID, selected
Szenario A and `haupt-15`, and used the phone for Start, Pause, Resume and
Finish. Native events confirmed official Start, audio start and audio end,
the 900-second completion threshold, and finalization. The first manual
Pause held the phone clock at 1:17 while ECG stayed live. The phone observed
Pause and Resume in about 1.3 s and 1.7 s respectively, including status
polling. Sampled Quest OpenXR status stayed focused with frames advancing
at roughly 72 per second.

The live run used `polar-acc` as the applied breath-tracking source. Quest
telemetry showed valid inhale, hold and exhale classifications from the
live sensor. The renderer applied radii across its configured 1.0–2.0 m
range. In one aligned live window, 3,527 valid Polar assessment and
renderer-radius observations within 14 ms had a Pearson correlation of
0.74 between normalized breath volume and applied radius. This describes
the observed software response during the bench test.

The run included nine pause/resume pairs and about 4 min 26 s of pauses.
One was the deliberate phone control test. Signal freshness interruptions
accounted for the other holds. At 13:21 a confirmed Resume was followed
by another native hold before the PWA observed Running. The phone remained
waiting for confirmation. After checking the signed Resume receipt and
the latest paused state for the same generation, the operator cleared
that one resolved tab receipt and reloaded the PWA. The Quest recording
stayed active. The phone rejoined it, and a new Resume reached Running.
The full authored audio ended and the operator finished the same run after
19 min 32 s of elapsed recording time.

Quest finalized with `saved:true`, `completed:true` and zero writer drops.
All six retained stream files matched the lengths in the native receipt
and totaled 368,644,569 bytes. The single v2 CSV was 93,987,682 bytes. A
streaming readback independently reproduced its SHA-256 and parsed 542,144
data rows. The CSV included 148,026 ECG readings, 228,976 three-axis
accelerometer readings, 80,973 breath classifications, 84,168 applied
radius records and one completed terminal record. The opening instruction,
calibration, second instruction, guided breathing and outro were present.

The Pixel automatically pulled the single CSV while Quest was idle. The
verified copy appeared locally about nine minutes after finalization and
the Drive confirmation followed within a minute. The Quest source, the
Pixel's committed CSV, and the private Drive receipt shared the same
SHA-256. Independent Drive inventory showed one 93,987,682-byte private
CSV. After downloading it through Android Drive, the finished file in
Android Downloads independently matched the Quest hash.

## Saved Quest connection — 2026-11-13

All 87 Node tests, six scenario JSON/CSV pairs, UI and context checks passed.
Chrome exercised one-time setup and cold entry at 390×844, 320×568 with
enlarged text and spacing, 844×390, and 1280×900. It rejoined after a synthetic
headset restart and changed address using its saved Bluetooth grant. It
rejected a different device with the same serial before sending an
authenticated request. Explicit disconnect stopped retries for the open
visit. Reconnect sent status requests without replaying a scene command.

USB readback found Quest 2 rehearsal 0.1.20 and the Pixel 10 WebAPK in
standalone mode. The installed Chrome 134 exposes `requestDevice()` and
lacks `getDevices()`. The browser can automatically try the saved LAN
address. Rediscovery after an address change can require the Bluetooth
chooser. Trusted enrollment and a physical cold restart remain open.

GitHub Pages run [43422623818](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43422623818)
deployed `a313621`. Eleven public files matched the release bytes, and all
four Chromium cases passed again on the published URL. The physical Pixel
WebAPK loaded the new entry with only the connection button before setup.
The native LAN gateway did not respond to the bench probe.

## Quest 2 audio owner cleanup — 2026-11-13

On a physical Quest 2, an isolated instrumentation process used the production
Java audio owner and the installed rehearsal assets. Each of the six scenarios
prepared and emitted Start, progress, Pause, Resume and Stop receipts, using
both condition tracks. The first run failed to close the owner. On Android,
the scheduled executor can discard queued cleanup during shutdown when delayed
tasks are disabled. The fix cancels the pending position sample explicitly and
lets the executor finish queued cleanup.

The corrected hardware run passed all six scenarios, closed the audio owner
while a Stop receipt was still being delivered, and released its singleton.
It passed again using the assets of signed release 0.1.20 after USB install.
The installed APK digest matched and its DEX contained the cleanup fix.
The Java audio, phase controls and cleanup barrier suites passed. This test
called the native audio owner directly. VR rendering and the installed Pixel
PWA remain unverified.

## Standalone Quest 2 media player check — 2026-11-13

A temporary Android instrumentation package opened the installed signed
0.1.19 APK on a physical Quest 2. It used the production catalog verifier
and WAV writer to prepare all six authored timelines, then played them with
Android MediaPlayer. The six durations matched the catalog at 68, 300, 600,
900, 900 and 1,200 seconds. All six showed advancing position, a held position
during pause, continued playback after resume, a completion callback and
stop. This sweep sought near the end of each timeline to check completion.
The two packaged condition MP3s passed start, pause, resume, loop and stop.
The 68-second instruction timeline also played from start to its completion
callback without a seek, including a pause and resume.

The main 900-second timeline ran without seeking across calibration. The
player first reported 26,029 and 41,016 ms at the two boundaries. The test
muted player output and verified decoding and position using packaged
placeholders. The headset was locked throughout. In-app controls, audio
output, Wi-Fi handoff, H10, and paired Pixel and headset captures remain open.
The temporary instrumentation package was removed after testing.

## Quest scenario player position — 2026-11-12

Quest 0.1.19 samples the Android player's position once per second while an
authored scenario advances. The audio owner fences the samples by generation
and operation, and Android logs include the position and monotonic timestamp.
Use these receipts to verify the 26–41 second calibration window and
pause/resume on Quest 2.

The Java audio owner test first failed for missing position readback, then
passed with the change. It verified sampling on the audio owner, a stable
paused position and progress under the resumed operation. The phase audio and
cleanup barrier checks also passed. All 85 Node tests passed. Six Chromium
layouts checked the VR-Probe Bluetooth reminder and signed controller flow
with 300 control checks, zero undersized targets and zero page errors against
the synthetic headset.

The Quest 2 USB install of the signed APK passed capsule and signing-lineage
checks. The installed digest and all 67 packaged audio files matched. The
headset lock screen held the app launch. On-device prompt, playback and
calibration evidence require an unlocked Quest 2 and the installed Pixel PWA.

## Recovery after an interrupted participant run — 2026-11-12

Quest 0.1.18 recovers an officially started run from its last verified
checkpoint. The recovered CSV keeps the run index, filename and UUID, records
the outcome as aborted, and uses only checkpointed observations. The export
binds the checkpoint and the durable stream digests. Tests confirmed that a
tail, an uncommitted CSV and the original files remain intact. Legacy
participant manifests without phase offsets retain the clocks and samples and
report an unknown phase.

The 72 Rust recording tests verified recovery, idempotence, export range
bounds, checkpoint and stream tampering, and the absence of a CSV for an Arm
without official Start. The experiment-session suite passed 92 tests when run
serially because its fixtures share a simulated Polar owner. One test was
ignored. All 85 Node tests passed. Five Chromium layouts checked 380 controls
and recovered-file sync on idle join against a synthetic signed gateway. The
download link met the 48 px touch-target check across the matrix.

The signed APK passed capsule, manifest, signing-lineage, recovery-code and
asset readback checks. The Quest 2 USB upgrade and installed digest matched.
Physical recovery after a Quest process stop, H10 and headset playback, and
paired phone and headset captures remain open.

## Quest participant recording — 2026-11-12

APK 0.1.17 integrates the enrolled AES-GCM Arm request, verified authored
scenario and fresh H10 gate. The native writer reserves a participant attempt
and UUID durably, then produces the original CSV when the run ends. It uses
the native ECG observations, media-start and pause events, and authored phase
offsets. The terminal receipt binds the CSV filename, size, SHA-256 and UUID.
The exporter pins that file alongside the existing eight recording artifacts.

The 69 Rust recording tests covered concurrent attempt reservation, native
writer finalization, signal gaps, calibration phase boundaries, pause timing,
filename validation and export integrity. The Java host gateway checks
covered enrollment, encryption, tampering, fresh ECG and stale generation.
Five Chromium layouts exercised the five participant scenarios against a
synthetic signed gateway, with 270 controls checked, repeated taps,
pause/resume, reconnect and verified CSV downloads.

The signed build passed the capsule, same-signer, package and installed digest
checks. Its 67 authored audio entries matched the staged files. The 20-minute
diagnostic remains in the audio package. Participant recording offers the
five study scenarios. The Pixel remained in Settings and Quest 2 remained
locked. The physical H10 run, playback timing and paired screenshots remain
open. Native checkpoint recovery retains abrupt interruption evidence. The
participant CSV still needs recovery for abrupt process termination.

## Startup Bluetooth prompt — 2026-11-12

The Quest APK invokes Android’s Bluetooth enable prompt when the radio is off
and the app has permission to check it. The host suite passed 72 discovery
checks, including the off and on states, permission first, and no duplicate
prompt after returning from the dialog. VR-Probe shows its reminder on
startup. All six Chromium layouts passed with synthetic Bluetooth, including
portrait, landscape, enlarged text and the hotspot handoff. Signed APK 0.1.16
passed capsule, same-signer and byte-exact USB installation checks. Its
manifest and DEX contain the stock Quest launcher lifecycle hook and Android
Bluetooth enable request. The actual
Pixel browser prompt and headset confirmation remain open.

## Pixel hotspot provisioning over Bluetooth — 2026-11-12

The enrolled PWA verifies the selected Quest serial and encrypts the hotspot
credentials to its controller key, current owner epochs and a one-use Bluetooth
challenge. The native app requests the Wi-Fi join through Android and returns
the address assigned to that request. The browser verifies the same Quest on
the signed LAN gateway before showing controls.

The Node suite passed 85 tests. The Java gateway suite covered credential
decryption, wrong-peer and stale-owner rejection, fragment order, bounded GATT
readback and the existing discovery and LAN controls. Six rendered controller
cases used a synthetic headset, including three that started without a Wi-Fi
address and set up a hotspot. Tests covered portrait, landscape, enlarged and
expanded text, phone controls and reconnect.

Signed APK 0.1.14 passed run-capsule, signing-lineage and manifest checks, and
all 67 scenario audio entries matched the staged file hashes. The USB upgrade
and installed APK digest matched. The Pixel and Quest 2 still need to complete
the first actual hotspot join, H10 and media checks, and matching device captures.

## Enrolled phone scene recenter — 2026-11-12

The native mailbox tests passed confirmation after a visible frame, stale
owner and mode rejection, expiration, cancellation and duplicate suppression.
The Java gateway host suite verified the enrolled LAN route, generation and
presentation checks, pending and confirmed receipts, and denial for guests and
relay. Six Chromium layouts passed with a synthetic signed Quest, including
keyboard activation, rapid taps during a run and recovery from a lost reply.
The full Node suite passed 81 tests. The UI and six scenario consistency checks
passed.

Signed APK `0.1.13` passed the exact run-capsule check, Quest 2 and scene
recenter manifest checks, same-signer upgrade, and byte-exact installed APK
readback. The physical Quest was still at its lock screen. The current
screenshots use emulated phone viewports and a synthetic gateway.

## Native scenario selection and audio lifecycle — 2026-11-12

An enrolled VR-Probe controller chooses a timeline independently of the
existing VR condition. Host tests verified that the signed gateway rejects
guest, relay, stale-generation and unknown-scenario Arm requests. A valid
request dispatches once and preserves its original receipt. The native audio
owner's host tests exercised preparation, Start, Pause, Resume, Finish and
release of both the condition and timeline players.

The native timeline renderer produced the exact frame count for all six
scenarios and verified silent PCM. A replacement cue with two nonzero samples
and refreshed metadata reproduced both samples and preserved all 39,690,000
main-round frames. Android decoder output and the native 26–41 s measurement
window still need physical verification.

The 80 Node tests passed. Six Chromium cases used each offered timeline once
across portrait, landscape, 200% text and expanded spacing. The signed
synthetic gateway fixture checked 192 rendered controls, repeated Start
taps, rotation mid-run, a frozen paused clock, Resume, lost Finish replies,
byte-exact download and rejoin. Six separate cases passed with the legacy
gateway and the scenario selector hidden. These results use emulated phone
viewports and synthetic Quest state.

The scenario controller release passed
[Pages run 43294713370](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43294713370).
All 12 changed public paths matched Git bytes. The deployed main-round
selection also passed in Chrome with the synthetic signed gateway, Local
Network Access granted to the isolated test profile and 32 rendered controls
checked. The complete native gateway and control-mode host suites passed.

## Signed Quest 2 assembly and USB installation — 2026-11-12

The [Atmungsraum assembly](../quest/README.md) resolved the shared
Experimenter composition against the pinned Rusty Quest source. The signed
development APK `0.1.12` passed Android signature and exact run-capsule
verification, the Quest 2 manifest check and ARM64 native library readback.
All 67 packaged scenario entries matched their hashes, including the catalog,
six timelines and 53 silent WAVs. A native Java host test rendered all six
timelines at the exact sample counts, confirmed zero PCM in the placeholders
and rejected corrupt cue data and the wrong catalog digest. The UI checks
and six JSON/CSV consistency checks passed.

The APK has two synthetic condition MP3s for the shared experiment player.
Its asset inventory contains no video. The scenario startup and playback
owners are packaged. Participant Arm and the participant CSV writer still
require integration and physical validation.

The host gateway suite passed signed request and receipt checks, the existing
export owner fences, and enrolled AES-GCM for ECG and recording responses.
The tests checked the browser key derivation and request binding, a full 32 KiB
chunk, denial for guests and relay, and no export I/O after rejected admission.
The Android owner uses live H10 sample timestamps in a bounded preview window.
The real H10 path and physical browser transfer remain open.

USB installed the APK on the physical Quest 2. Android reported the expected
package and version, resolved its launcher and returned the signed build's
exact installed APK digest. The Quest lock screen remained active at
readback. The authorized Pixel 10 was still in Settings and had no installed
Atmungsraum WebAPK. Chrome reported version 134. Device launch, paired screenshots, H10 and phone control
remain open. The older Viscereality builds are available for rollback.

## Quest scenario asset staging — 2026-11-12

The Atmungsraum staging tool produced a hash-bound catalog of 66 source files,
including all 53 numbered WAVs and all six authored timelines and manifests.
The fixture verified exact byte readback, the main-round 26–41 s measurement
window, rejection before output on corruption, and acceptance of a replacement
cue with updated hashes and unchanged slot length. The full Node suite passed
79 tests.

## Quest 2 preflight and controller stress — 2026-11-12

Verified against the Atmungsraum source from `main` at `05faa1a`, with the ECG
visibility fix applied. All 78 Node tests, the UI and context checks, the six
scenario JSON/CSV pairs, and the Python scenario asset checks passed. The
placeholder integrity check covered 53 WAVs and 890 scheduled events. FFmpeg
decoded the browser background track and all 20 published voice samples.
Chromium rendered all five menu timelines and the 20-minute engine diagnostic
at 44.1 and 48 kHz with zero sample error. The long diagnostic passed when run
on its own with a larger deadline.

Chromium passed 210 rendered layout cases across 14 viewports and three text
settings, plus the five browser demo play/pause/resume/stop paths. The layout
checks covered 319–1440 CSS pixels, short landscape, doubled root text and
text-spacing overrides. Install and saved-file dialogs restored focus. The
rendered controls checked for size, horizontal clipping and label overflow
passed. Six signed controller fixture cases passed manual or emulated BLE join,
Polar setup, fresh ECG, encrypted participant Arm, both conditions, Start,
rotation, Pause with a frozen clock, Resume, Finish, recording sync, exact CSV
download and disconnect/rejoin. The fixture sent repeated Start taps and
recovered two lost Finish replies without duplicate dispatch. The Node tests
also cover storage, protocol and Drive fault injection.

The signed fixture exposed an SVG that remained hidden with fresh samples.
The controller now changes the SVG's `hidden` attribute. A rendered regression
checks that fresh samples show a waveform and stale samples hide it. Synthetic
screenshots and receipts are kept locally under
`output/playwright/quest2-20261112/`.

Release `f4d47aa` passed
[Pages run 43275951484](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43275951484).
All nine changed live paths matched Git. Chrome 157 replayed the signed
portrait controller fixture on the public URL with Local Network Access granted
in the isolated test browser. The gateway, participant and samples were synthetic.

After USB approval, ADB identified the physical Quest 2 on Android 14.
Its package inventory had no matching Atmungsraum rehearsal APK `0.1.9`.
The Pixel 10 appeared in file transfer with no authorized ADB transport.
Paired device screenshots, physical controller and H10 validation, headset
video/audio playback, and the APK asset inventory and scheduler are open.
The available native three-phase mode covers a different cue set.
See the [active checklist](PROJECT.md#quest-2-control-media-and-ui-acceptance--2026-11-12).

## Automatic Drive delivery fixture — 2026-11-12

The 40-bundle Drive stress test simulates eight lost upload acknowledgements and
receipt-store restarts, then verifies one object per bundle and exact SHA-256.
Additional tests cover a 6 MiB upload, wrong hash, a foreign upload URL, a
shared destination folder and quota recovery using the originally reserved ID.

The local Chromium fixture drives signed Quest actions through native-file
readback, durable save and a simulated Google Drive API. Four bundles per layout
arrive across 390×844, 568×320 and 320×568 at 200% text. It exercises a wrong
Google account, corrupt ACC, a second tab holding the upload lock, lost upload
acknowledgement, restart and offline catch-up, reauthorization after HTTP 401,
and a finalized partial recording. The existing local CSV download remains
byte-exact. The fixture supplies the Google sign-in and Drive responses.
Real provider behavior, the installed Pixel PWA and actual Quest/H10 streams
require the attended acceptance pass in [QUEST-PHONE-SYNC.md](QUEST-PHONE-SYNC.md).

The exact release `d171e39` passed
[Pages run 43123800186](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43123800186).
All 13 changed public paths matched Git, and all three browser fixture cases
passed against the public URL, including migration of existing local storage.
The Drive API's real upload endpoint accepted the Pages origin in an OPTIONS
preflight. No authenticated real Drive upload was part of this verification.

## Quest sensor bundle for Drive delivery — 2026-11-12

The new unit regression checks the participant CSV and exact original ECG and
ACC bytes after multi-chunk retrieval, then corrupts a later ACC chunk and
requires rejection. All 72 Node tests, UI checks and the context checker passed.

The local Chromium fixture ran signed, encrypted Quest control and recording
transfer at 390×844, 568×320 and 320×568 with 200% text. It exercised Arm, Start,
Pause, Resume, Finish, rejection and retry of a corrupt ACC chunk, browser
restart, a new native epoch, catch-up of a run completed with the phone absent,
and a second run. Each case reopened the IndexedDB ZIP after browser restart,
matched both original stream hashes and downloaded the original participant
CSV. The sensor inputs and startup credentials were synthetic. The installed
desktop Chrome channel exited on launch; the fixture used bundled Chromium.
Actual Quest/Polar samples, Pixel standalone launch and real Drive delivery
still require separate evidence.

## Browser session background music — 2026-11-08

FFmpeg decoded the entire stereo MP3 without errors. Chrome reports 425.284
seconds, and the measured mean level matches the WAV at −33.6 dBFS.
The MP3 is 7,651,988 bytes with SHA-256
`6c83316b326b105465cb8c4b434a926f789d9aaf3014543ec477d5e6e0e546fa`.

Chrome on the 390 × 844 browser demo requested music only after Starten,
resumed at the paused position, and played from the beginning after reaching
EOF. End and Polar loss stopped and released the element. A fresh session
started a fresh track. Quest entry made no browser music request. The fixture
used a range-capable local server and synthetic startup credentials.

The exact Pages deployment for `afe19f2` reported success. All ten changed
live files matched Git, both Atmungsraum entry URLs returned the committed
page, and the MP3 responded to a 1,024-byte range request with HTTP 206.
The same Chrome fixture passed on the live URL with zero page errors.
[Pages run 42516486897](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42516486897)

All 71 Node tests passed, including natural completion, a cancelled start,
and a rejected music play request. UI, context, all six scenario JSON/CSV
pairs and Python scenario asset checks passed. Physical phone and headset
listening remain separate checks.

## Required startup unlock and Pages-only delivery — 2026-11-08

Chromium checks the startup password on the default, CSV recovery and Quest
routes. Wrong passwords and Escape keep the interface locked. Correct unlock
clears the field and restores focus to the active heading. Silent restoration
uses a fixture for the browser credential API. Five layouts plus 320px
reflow at 200% text and enlarged spacing passed. The two-action folder fixture
still passes eight simulated uploads and seven intercepted Discord requests.
The production envelope has no relay and the folder makes no provider request.

The previous live folder release also ran a Chromium fixture through the
local upload adapter against real Wormhole/Backblaze. Two ciphertext pieces
restored the exact 9,800,146-byte synthetic ZIP. Discord was intercepted.
Wormhole's live manifest and service worker confirm its system share-sheet
handoff to a same-origin tab with no return of the generated URL.

## Encrypted participant Arm — 2026-11-08

The Java gateway suite and enrolled LAN harness verified authenticated
decryption, guest and relay denial, stale-generation and absent-ECG
rejection, once-only dispatch, and a receipt without the participant ID.
Chrome fixtures reused the participant form, checked the normalized ID and
disabled Start before it was set, decrypted the bounded private Arm on the
fixture server, verified the original participant CSV, and repeated after a
browser and simulated Quest restart. Phone portrait, short landscape and
200% text reflow passed locally and against Pages run
[42491480387](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42491480387).
All nine changed live files matched the browser release. After integrating
the concurrent folder-sync release, 71 Node tests and the UI and scenario
checks passed.

Signed Quest APK `0.1.9` passed the exact run-capsule check and same-signer
upgrade. USB Package Manager and installed SHA-256 matched the built APK.
The headset was still asleep and both Nearby devices permissions were
ungranted at readback. Live H10, Quest CSV, and physical Android phone
verification remain open.

## Startup unlock and two-action folder — 2026-11-08

Chrome's synthetic browser fixture checks initial focus, incorrect password,
opt-in browser storage, silent restoration after reload, cancellation and
recovery, and repeated sends without another prompt. Both send actions
decrypt to the expected ZIP contents. Rejected uploads and Discord posts
leave the receipt ledger pending. The fixture exercises concurrent tabs,
five desktop/portrait/landscape layouts and 200% text and spacing at 320px.
The separate local Node adapter test sends two encrypted chunks to real
Wormhole/Backblaze and restores the exact 9,800,146-byte synthetic ZIP. Discord
posts in the browser fixture are intercepted. Hosted activation remains
open in [PROJECT.md](PROJECT.md#one-startup-unlock-for-folder-sync--2026-11-08).

## Participant CSV writer and idle pull — 2026-11-08

The Quest writer and export host checks passed 66 focused tests. Synthetic
native records verified reservation across reopened and abandoned attempts,
Unicode participant filenames, actual ECG values and sensor timestamps,
blank-voltage gap rows, and a hash-bound descriptor over the finalized CSV.
The Java LAN gateway harness passed 32 checks, including the participant
filename and traversal denial. Neither check installs or runs the Android APK.

The Chrome fixture ran the encrypted idle sync at 390×844, 568×320, and
320×568 with 200% text and custom spacing. It rejected a corrupt participant
CSV, then downloaded only the verified CSV bytes with the Quest filename.
It reopened that exact file after a full Chrome restart and continued to
reconcile legacy ZIPs and runs finished while the browser was absent. The 69
Node tests and UI checks also passed. The Quest was still asleep at USB
preflight. The operator form, encrypted Arm, and real H10 and phone acceptance
remain open.

## Idle Quest sync — 2026-11-08

The signed encrypted Chrome fixture joined a Quest with an existing finalized
recording, automatically saved it in IndexedDB, and ran Arm → Start → Pause →
Resume → Finish. Its gateway rejected recording traffic outside idle. It then
served the two saved runs across separate pages, corrupted one transfer to
exercise hash rejection, and verified the retried ZIP and its original bytes.
After a browser reload and rejoin the archive selector offered both stored
runs without reading their bytes again from Quest. The downloaded ZIP matched
the native test bytes. All three layouts passed (390×844, 568×320, and
320×568 at 200% text with text-spacing overrides). The full 69 Node tests, UI
checks and six scenario consistency checks passed.

The follow-up restart fixture used a persistent Chrome profile and fully
closed the browser. It rejoined the same simulated Quest under new gateway
and native epochs, reopened both saved ZIPs without reading their bytes
from Quest, completed a second rehearsal and synced only that new archive.
All three layouts passed against both the local and deployed app. Runtime
commit `9398177` deployed successfully in
[Pages run 42487461920](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42487461920).
The live entry and five changed JS/CSS assets matched the exact commit bytes.

This fixture used synthetic data. The USB-connected Quest was asleep at
device preflight. Physical H10, participant CSV, and Android phone storage
verification remain open.

## Remote-only Quest data path — 2026-11-08

The new composition passed feature resolution for the original native host,
Remote-only startup and recording export. The native host tests covered
Remote-only control admission, guest denial for private data, and the maximum
encrypted recording chunk. The browser tests verified the encrypted read,
stale and reordered ECG rejection, gap rendering and corrupt recording bytes.
Chrome fixture runs at phone, short landscape and enlarged text sizes used
the actual UI to select condition B, receive changing ECG, Start, Pause,
Resume and Finish. Each run downloaded and checked the original ZIP data,
then reloaded, rejoined and retrieved the same native run again. All three
cases repeated on the deployed website.

The signed 0.1.7 rehearsal includes the ARM64 OpenXR loader and original
native host. Quest 3 installed the exact APK bytes and reported Remote from
its mode owner without operator selection. Desktop Chrome used the live
Atmungsraum page to join the identified Quest and received a valid encrypted
ECG preview reply over Wi-Fi. Android's Nearby devices prompt was active,
so the reply contained no live H10 samples. Actual sensor acquisition, native
scenario completion, phone hotspot discovery and phone storage remain open.

## Browser controls for Quest-owned Polar setup — 2026-11-07

The signed LAN gateway fixture verified missing Android permission, keyboard
activation, exactly one scan/select/connect/start-ECG sequence and native
readiness in Chrome. It then ran the synthetic Arm → Start → Pause → Resume
→ Finish route. The same five Chrome viewport/text settings passed with a
gateway that omitted Polar setup, including 320 CSS px, short landscape,
desktop and 200% text with custom spacing. Both fixtures measured visible
text and checked control heights, overflow, page errors and demo asset
isolation. Unit tests cover invalid setup grants, ambiguous sensor discovery,
duplicate actions, lost receipts and foreground exit. All 65 Node tests and
six scenario consistency checks passed.

The separate synthetic save sweep downloaded and reopened five exact session
CSVs and verified automatic completion of the 68-second Anleitung. The live
121-case Chrome layout sweep and the renderer-crash and quota recovery
fixtures passed for `2aeb0f1`. Delaying the Quest controller import reproduced
a visible join form before listeners were attached. The form now appears only
after its controls are ready. Quest 3
reported both Bluetooth runtime permission grants for the installed rehearsal.
Physical browser-to-Quest control and live H10 ECG require the attended Remote
gate and a separate device run.

## UI and CSV saving sweep — 2026-11-07

The sweep began on Pages run `42406261765` (`53a7633`). Chrome checked 121
layouts spanning patient entry, ECG, round selection, all five playback-ready
screens, playing and paused states, and saved CSVs. The populated CSV list
overflowed at 320 CSS px with 200% text and custom spacing. The filename now
wraps within the grid. The local matrix repeated with zero layout failures
or page errors. A separate probe passed with Pretext unavailable.

Using only synthetic inputs, Chrome downloaded and reopened matching CSV bytes
for all five session choices. The sweep checked scenario hashes, sample order,
pause-aware timing, separate run identities, the saved-files count and dialog
focus return. A full 68-second Anleitung completed and saved automatically.
A forced renderer crash recovered one aborted CSV from the browser checkpoint
with ECG requests blocked. A simulated IndexedDB quota failure showed the
warning and preserved both immediate and manual downloads.

The live browser fixtures passed the 30-second Polar loss rule, 115 PWA
control cases, five signed Quest UI cases, and encrypted sharing on five
layouts. The sharing fixture checked archive bytes, failures and retries,
new/all selection, reload and cross-tab receipts. It intercepted eight
synthetic uploads and seven notification attempts. Brave checked install
help and installed-state UI in isolated profiles. The 61 Node tests,
UI/context checks, six scenario pairs and 20 sample audio profiles passed.
Chrome also verified exclusive voice-sample playback and exact MP3 and JSON
downloads. Offline rendering matched the expected PCM for six scenarios at
44.1 and 48 kHz.

Pages run [42408436234](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42408436234)
deployed `4309c6f` from `main`. Both entry URLs returned HTTP 200 and all four
changed public files matched the commit. The live Chrome 156 matrix repeated
all 121 layouts with zero failures or page errors, and the live renderer
crash and quota fixtures passed.

Physical Android/Quest acceptance and external delivery are separate checks.
The browser probes and synthetic artifacts remain in ignored `output/playwright/`.

## Brave desktop install and taskbar pins — 2026-11-07

Brave 157 parsed the live manifest and installed the app into an isolated
browser profile, creating the Atmungsraum ICO and manifest icon resources. The
profile still shared the Windows taskbar. After uninstall, a taskbar pin retained
the same app ID and referenced its removed ICO in the QA profile. The native
Windows unpin action removed that exact link. A subsequent shortcut scan
returned zero Atmungsraum links. Future native installation QA requires an
isolated OS account or VM.

The original manifest returned an empty related-app list in a regular tab
after installation. A self relation in the manifest returned the exact installed
app ID and manifest URL in Brave. The regression for suppression across a
fresh tab initially failed and now passes. All 54 Node tests and the UI and
context checks pass. The safe Brave probe uses the observed positive response
as a fixture to verify hiding after reload. A fresh Brave profile verifies
manifest parsing, installability, icon decoding and the manual help at
1280×800, 390×844 and 320×568 with 200% text in the smallest case. All three
dialogs open at the top, have no horizontal overflow and restore focus on
Escape. The probe invoked zero native install actions and produced zero page
errors. A fresh taskbar pin in the regular Brave profile is a manual check.
Receipts and the probe remain in ignored `output/playwright/`.

Pages run `42403873448` deployed `6528b47`. The live file hashes matched the
commit, and both entry URLs resolved to `/atmungsraum/`. The live Brave probe
repeated the layout and reload checks with zero page errors or native installs.
A final scan returned zero Atmungsraum taskbar shortcuts.

## Quest rehearsal browser and native preflight

The rehearsal is entered through **VR-Probe** on the patient screen or
`/atmungsraum/?mode=quest#quest`. The browser requires a selected Quest
serial and a packaging fingerprint from the isolated rehearsal build.
Use its exact Remote mode and a trusted LAN. The phone uses the signed
experiment gateway, and the rehearsal mode does not initialize browser
ECG, audio playback or participant exports.

The 61-test Node suite includes the BLE identity read and cleanup, private gateway
and serial admission, the signed status and expected build, the complete
Arm/Start/Pause/Resume/Finish sequence, duplicate input, foreground cancellation
during signing, and unknown results.
Chrome passed manual-address, measurement and signed-fixture lifecycle checks
at 320×568, 390×844, 568×320 and 1280×720, plus 320 CSS px with 200% text and
text-spacing overrides. These checks exercise the real browser UI with a
fixture. Phone LAN permission, physical BLE and wearer presentation require
the installed-Android acceptance run. The exact
[Pages run for 994c0b8](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42405189848)
completed successfully. All nine changed files matched their live bytes and
the five signed-fixture Chrome cases passed again from the live origin.

The native gateway host suite passed production Remote discovery, enrollment
and receipt checks. The private build passed its exact feature lock, synthetic
asset provider, selected Java isolation and v3 signature verification. The
signed APK manifest contains the isolated package and exact site origin.
Quest 3 installed package/version readback matched. Older broad native tests
still flag their historical panel inventory and source assertions; the focused
gateway tests and the actual selected APK closure passed. Wearer presentation,
physical Android control and Showcase visual equivalence remain open.

## One checkout receipt per confirmed send — 2026-11-07

Both targeted tests failed before the separate receipt reader and writer were
added. All 52 Node tests and the UI and context checks pass. The local browser
fixture observes six checkout receipts across eight fixture uploads and seven
fixture posts. It verifies that failures leave no receipt, resends append a
receipt, earlier receipts remain intact, legacy history still suppresses
duplicates, and an open second tab updates after a confirmed send.
The same fixture passed against the published Pages release with five
layouts and no page errors. It used synthetic data and simulated provider
responses.

## Folder delivery and JSON receipts — 2026-11-07

The receipt and URL-only payload tests failed first, then passed. All 52 Node
tests and UI/context checks pass. The Chromium fixture checks five layouts
and 200% text reflow, verifies progress/focus, decrypts the uploaded ZIP to
check exact CSV selection and bytes, and validates the complete URL-only Discord
payload. It covers wrong passwords, aborted prompts, rejected upload/post,
reload, forced resends, a concurrent-tab lock, a file added during upload, a
revised CSV under the same session UUID and a file saved in another tab.
The local run observed seven fixture uploads, six fixture posts and no page
errors. No participant data or real Discord message was sent. The actual
Render deployment remains a separate operator step. See the ignored
`output/playwright/check-file-delivery.mjs` fixture and PROJECT.md release state.
The same browser suite passes on published Pages and verifies that the app
asks the credential manager to save only after an opted-in successful send.
The public ciphertext matches the configured destination after local
decryption; the check does not send a Discord message.

## Render Free upload adapter — 2026-11-07

The targeted tests failed first with the adapter/host support absent, then passed.
Real Node HTTP checks cover health, preflight, incorrect origin/key, `/notify`
blocked, unapproved storage host, over-limit body and exact 8 MiB forwarding.
Browser client tests cover a startup page, successful readiness and cancellation.
All 51 Node tests and UI/context checks pass. Five viewport browser fixture
checks (including enlarged text) exercise Render wake-up, cancellation, upload
failure, QR creation and the direct Discord prompt. Review found no blockers.

The local Node adapter made real calls to Wormhole and Backblaze. A synthetic
9,800,146-byte ZIP uploaded in two encrypted pieces and decrypted on download
byte-for-byte (SHA-256 `2ee9b24329c496fb94e8291b099b2df966fe86211f4c6f00375ad9b81ebd8eb2`).
No participant data or Discord test messages were sent. The actual Render
platform and its idle wake-up still require a user-owned deployment. See
`output/playwright/render-upload-local.json` and `render-sharing-results.json`
(ignored); release and deployment state are in PROJECT.md.

## Larger phone controls and install action — 2026-11-07

The two top surfaces/artwork double the base size at widths through 480px or
coarse-pointer widths through 1024px (112px versus the 56px folder on ordinary
phones). Their equal sizes, safe-area anchors and existing semantics remain.
Content reserves their enlarged clearance. A text install action sits beside
the folder in portrait; short landscape uses a named install icon above it,
outside the central content column. An actual ResizeObserver reserves the dock
height when text enlarges/wraps. Every visible new label uses the existing real
Pretext/DOM measurement owner; the label's box has room for its safety margin.

Four new Node checks first fail with the missing module, then pass. They cover
user-triggered prompting, one-use events/double-click exclusion, dismissal,
prompt rejection, retryable manual help, install consent versus actual
installation, standalone hiding, Escape propagation and focus restoration.
All 32 tracked Node tests, UI-state/context and staged whitespace checks pass.
The release excludes concurrent unfinished automated-sharing code and the
private phone attachment. No participant-data writer or audio asset changes.

Chrome passes 113 rendered cases across 320×568, 360×740, 390×844, 432×814,
480×640, 481×640, 568×320, 844×390, 844×480, 844×481 and 768×1024, using the
staged entry/CSS and released app/data modules. Patient, ECG, choices, all five
player routes and data recovery have no horizontal overflow/clipped labels or
unmeasured visible text. Bounded cases have no page scrolling or content/corner
overlap. Small/enlarged no-fit states use readable document reflow; keyboard
scrolling exposes each patient field/action clear of fixed controls.

The help dialog passes four phone sizes, native keyboard dismissal and focus
return. 200% type/WCAG-style spacing pass portrait and short landscape; a long
install label reflows at 320px. Closing help or saved files with Escape during
real local playback keeps the session active. A simulated browser install
event verifies click-only prompting, double-click exclusion, dismissal and
later help; `appinstalled` hides only the install action, leaving the folder.
These simulations test UI lifecycle, not an actual phone installation receipt.
Zero page errors occur. Phone screenshot inspected. Ignored evidence:
`output/playwright/check-pwa-controls.mjs`, `pwa-controls-results.json` and
`pwa-controls-phone.png`. Exact deployment/live evidence follows in PROJECT.md.

An additional opening-position regression first fails: at320×568 with200%
text/spacing, native autofocus on the bottom close button scrolls help to1832px
and hides its beginning. Close now sits in a measured header above instructions;
the same case opens at scrollTop0. All113 cases re-pass with new initial-scroll
assertions. Three desktop smoke sizes retain the original corner sizing and
usable help. The regression lives at `output/playwright/check-install-scroll.mjs`.
The containing sharing release passes all45 Node tests. No new installer logic
or participant-data behavior is changed by this header correction.

Live release `5ff34d3f5ceaed7e15ab306d6df32b5d6db84e16` completes exact
Pages run 42310953670 and deployment API status `succeed`. All12 affected paths
match Git bytes. The public origin passes all113 phone cases, including help
opening at scrollTop0 with enlarged text, plus three desktop smoke sizes and
error-free manifest/icon/installability checks. An isolated Chrome desktop
profile installs and launches the actual live app in standalone mode with that
user display preference explicitly set. No page errors occur. Evidence stays
ignored in `pwa-live-5ff34d3.json`, `pwa-controls-live-results.json`,
`pwa-controls-live-phone.png` and `pwa-controls-installed-live-results.json`.
Physical-phone installation/relaunch and assistive technology remain untested.

A stricter viewport-centering regression then fails at390×844: the two-top-icon
clearance moves the bounded round menu down27.99px. Equal vertical reserves,
using the maximum of top size + top safe inset and dock height + bottom safe
inset, restore its center to -0.008px. The47/34px unequal-inset probe first fails
at +6.49px, then independently passes at -0.008px. All115 phone cases re-pass with centered ECG/round/option/player
targets required whenever bounded. Enlarged/no-fit states retain readable
document reflow. This correction changes CSS and cache tokens only; its checks
exclude concurrent Discord-password work. Ignored regression:
`output/playwright/check-pwa-centering.mjs`; matrix receipt:
`output/playwright/pwa-controls-centered-results.json`.

The centered runtime `0320d1c1231396caf458a089ae7e60115b781aab` completes exact
Pages run 42313937741 with backend `succeed`. All12 affected paths match the
published Git bytes; the public origin passes all115 cases including simulated
unequal insets and zero opening scroll with enlarged text. No page errors.
Ignored live evidence: `pwa-live-0320d1c.json`,
`pwa-controls-centered-live-results.json` and `pwa-controls-centered-live-phone.png`.
No manifest, installation logic or participant-data changes accompany this
centering correction. Physical phone acceptance remains distinct from this QA.

The current [BeforeInstallPromptEvent API](https://developer.mozilla.org/en-US/docs/Web/API/BeforeInstallPromptEvent)
and [official installation-prompt guide](https://web.dev/learn/pwa/installation-prompt/)
were read. This API is not universal and each supplied prompt is single-use;
unsupported browsers therefore retain explicit manual instructions. Physical
Android/iPhone installation/relaunch acceptance remains open.

## Smartphone PWA installation — 2026-11-06

The primary delivery target is an installable smartphone PWA. The entry page
links a local manifest with stable `/atmungsraum/` ID/start/scope, standalone
display, black theme/background and the existing Viscereality logo. Native
browser installation is the home-screen launch path. The app remains online;
no service worker, offline audio cache or background session behavior is added.

The four opaque PNGs derive from `Images/logo.png` at Git blob
`bcd94ca74f6305eb7cd2e8a9a8b4a66cf99f6caa`. Ordinary 192/512px and Apple 180px
variants contain the unchanged aspect ratio on black at 86% maximum height.
The separate 512px maskable icon uses 66%; a pixel audit confirms every nonblack
pixel is inside the central 80%-diameter safe circle. Source and rendered icon
were visually inspected; no logo redesign or image-generation service is used.

All 28 tracked Node tests, UI-state/context and staged whitespace checks pass.
Chrome 156 (Windows, isolated persistent profile) parses the manifest without
errors, reports `installabilityErrors: []`, decodes each declared icon, and
keeps the entry screen readable at 390×844, 320×568, 844×390 and 568×320. No
horizontal overflow, clipped patient labels or page errors occur. Browser QA
serves the exact staged entry HTML and released JavaScript/CSS/data module,
excluding concurrent unfinished sharing changes. Independent review finds no
actionable issues.

CDP installation/launch opens the canonical URL and initial `#patient` route.
With the browser user display preference explicitly set to standalone (the
equivalent of Open as app/window), actual `(display-mode: standalone)` is true.
CDP's initial default preference opened a browser tab; the test records that
distinction rather than claiming a physical-phone default was exercised. The
test uninstalls its temporary app. Android Chrome/iPhone Safari installation,
relaunch, separate installed-context storage, safe areas, lock/background and
real H10/headset qualification remain open. Existing session-abort and browser
storage rules are unchanged.

Ignored local evidence: `output/playwright/build-pwa-icons.mjs`, `check-pwa.mjs`,
`pwa-local-results.json`, `pwa-local-phone.png` and temporary browser profiles.
Publication evidence belongs to PROJECT.md.

Live verification: release `02217feda1df9d99de830e50867ca32ce94b271c` completes
exact Pages run 42260860687; the deployment API reports `succeed`. All 13
changed public files match Git bytes, with `application/manifest+json` for the
manifest and `image/png` for icons. Slash/no-slash URLs resolve to the canonical
app and return 200. Chrome repeats error-free parsing/installability, icon
decoding, four phone viewport checks and actual standalone launch on the live
origin, with its user window preference explicitly set. The final target-phone
installation/relaunch question is pending; desktop results do not close it.
Ignored live receipts: `pwa-live-02217fe.json`, `pwa-live-results.json` and
`pwa-live-phone.png`. No runtime file changes accompany this evidence update.

Primary platform references read: Chrome's [install-criteria update](https://developer.chrome.com/blog/update-install-criteria)
permits menu installation without a service-worker fetch handler; the current
[MDN installability guide](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Guides/Making_PWAs_installable)
confirms required manifest fields and that service workers are optional for
installation. [WebKit's Safari 26 release notes](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/#every-site-can-be-a-web-app-on-ios-and-ipados)
describe home-screen web apps and the user's Open as Web App choice.
This checks the common contrary assumption that installation requires adding an
empty worker; offline functionality is a separate feature. Browser support and
device installation are distinct from manifest validity.

## Polar signal-loss timeout — 2026-11-06

Two new regressions first fail against the previous runtime: a delayed first
poll retries rather than expiring, and ECG recovery after the deadline revives
the run. They pass after the shared-state fix, along with an exactly-30-second
recovery boundary case. Review identified a direct-ECG deep-link start that
missed the grace period; its regression also fails before arming recovery in
validated `start()`, then passes. All 28 Node tests, the UI-state checker, context checker
and whitespace checks pass. The existing browser-controller regression now
verifies timeout navigation, inactive state, aborted receipt and disconnection
of every scheduled audio cue. Participant-data writers and audio assets are
unchanged.

Native Chrome passes eight signal-loss cases: all five scenario routes using
a delayed first poll, a late recovery event before the next timer callback,
short recovery preserving the same run, and real elapsed loss while paused.
The paused case returns to `#h10` after 30,509 ms even though the audio clock is
frozen. Every expired case saves an aborted CSV/receipt, clears the scenario,
retains patient details and moves focus into the Polar gate. All five lost-signal
scenario cases also verify searching/selecting again; one starts by valid-ECG
deep link before Continue. No page errors occur. The existing 137-case responsive
matrix also passes, including 320 CSS px, portrait/landscape, enlarged text,
spacing and long labels, using real Pretext/DOM checks.

Ignored evidence: `output/playwright/check-polar-timeout.mjs`,
`polar-timeout-results.json`, `polar-timeout-polar.png`, and `polar-layout-*`.
These are browser-demo checks; physical Polar H10, headset and phone/browser
qualification remain open. Publication evidence belongs to PROJECT.md.

Live verification: containing release `c2356be` completed Pages run 42254269547
successfully; the exact deployment API reports `succeed`. All 12 changed public
files match Git bytes and both entry URLs return 200. Native Chrome passes all
eight signal-loss cases on the public origin, including a real paused loss of
30,676 ms, with zero page errors. Initial live harness attempts timed out at
30 seconds during full-page/audio loading; using DOM readiness and a 90-second
audio-preparation allowance completes the checks without changing runtime.
The 30-second signal-loss deadline is unchanged. Evidence remains ignored in
`polar-timeout-live-results.json`, `polar-timeout-live-polar.png` and
`polar-release-c2356be.json`. The generic Node asset fetch hit a connection
timeout; the exact-byte check completed with Python's standard HTTPS client.

## Password-gated direct Discord delivery — 2026-11-07

The user removes Cloudflare from the Discord notification path and requests an
encrypted webhook on GitHub Pages, a masked send-time password prompt and
browser password-manager saving/autofill. Optional automatic Wormhole upload
retains its separately configured relay; the app posts links straight to Discord.

Node regressions fail before the direct sender/envelope module exists, then pass.
49 tests cover exact-host webhook URL validation, no redirects/cookies/referrer,
provider acceptance, authenticated AES-256-GCM round trip, fresh salt/IV,
PBKDF2-SHA256600k, wrong password, tampering, envelope bounds and abort-before-send.
A private helper generates the144-bit password and verifies the actual encrypted
file without printing credentials. Commit only ciphertext. Correct passwords
can recover the webhook; old copies remain usable until webhook rotation.

Native Chrome fixture checks cover masked auto/manual prompts, Escape/Cancel,
wrong password, Discord failure/retry, QR/ZIP retention, stale abort, focus,
keyboard and five viewport sizes plus200% text/spacing. Review's parent-focus
issue and an Escape close-event clearing race are reproduced red then fixed.
The browser credential API fixture observes no save on cancel, wrong password
or failed delivery, and one save on successful explicit opt-in. Native browser
settings/Save/autofill on a physical target device remain user-side behavior.
No site storage receives the password or unlocked webhook. Existing five upload
and seven manual QR cases pass; no manual QR external traffic when forwarding is
off. UI/context/whitespace checks pass. Screenshot inspected without secrets.

A real Chrome run at `https://viscereality.org/atmungsraum/` with local source
routed there before publication loads the actual encrypted file, prompts for
the password and makes zero Discord requests on incorrect entry. Correct entry
makes one real browser POST to Discord (no proxy/fulfil or CORS bypass), gets200
and reads back the matching synthetic keyed link. Network origins are exactly
the site and Discord, with Worker settings empty; no Cloudflare request occurs.
No password is sent or left in the prompt, and no participant data is used.
Ignored receipts: discord-password-results.json, discord-password-phone.png and
discord-password-local-source-network.json. Exact Pages run42362312447 for `3d861b1` succeeds, including byte equality
for all18 changed files and both public entry URLs. The real published page
repeats wrong-password/no-send, direct Discord POST, confirmed readback and
password clearing with no Worker requests. The official Wormhole page downloads
the9,437,363-byte synthetic archive and its SHA-256 matches the prior configured
upload. Five live prompt layouts and200% reflow pass. Chrome's actual native
credential-store call fulfills for the synthetic opted-in case; browser consent
and saved-password selection remain user-controlled. Ignored live receipts:
discord-password-live-network.json, discord-password-native-api-results.json,
sharing-release-3d861b1.json. The operator password and webhook remain private.

## Automatic Wormhole upload and Discord — 2026-11-06/07

The user-selected route now supports browser ZIP encryption/upload, local QR
and automatic notification to a privately configured fixed Discord webhook.
Saved-record/CSV identity and the ZIP writer remain unchanged. See the active
PROJECT checklist and sharing/README.md for installation/account acceptance.

All45 Node tests in the current worktree pass (including four separate PWA
installation tests); UI-state/context and whitespace checks pass.
TDD observations: missing upload client/routes fail before implementation;
the real Backblaze upload path's two opaque suffix segments produce a failing
regression before the allowlist correction. Two-pass encryption tests upload a
9 MiB synthetic ZIP in two pieces, independently decrypt all bytes, verify
metadata/piece hashes, restrict contacted hosts and require final completion
acknowledgement. Worker tests cover origin/auth, bounded bodies, fixed recipient,
unsafe links/storage endpoints, token isolation, provider failures and redirects.
A Cloudflare deployment initially failed before network I/O because workerd
rejects `redirect:error`. Official [Workers Request documentation](https://developers.cloudflare.com/workers/runtime-apis/request/)
confirms `manual` is required. Worker-only fetches now use manual and reject
3xx; browser fetches keep error mode. Regression assertions fail before this
change and pass after it. Independent review finds no remaining blocker.

Native Chrome passes five automatic-sharing layouts (320×568,390×844,568×320,
844×390,1920×1080), four forwarding layouts, seven manual QR layouts and ten
existing file-dialog cases. Checks cover missing settings/no request, confirmed
upload/QR, notification failure then send-only retry, retained share after
close/reopen, unchanged-archive reuse, new-record invalidation, failed/cancelled
upload, downloadable fallback, missing/delayed QR module, independent QR decode,
clipboard denial, enlarged text/spacing, Pretext, focus and no horizontal dialog
overflow. Review found Cancel hid the focused button; browser regression was
observed red then green after focus returned to the upload button. One earlier
forwarding layout run and two initial loading checks were rerun after concurrent
PWA edits settled; all affected checks pass. No page errors or external QR
requests occur. Screenshots/fixtures remain ignored under output/playwright/.

Real network evidence uses synthetic data only:

- Production client + Worker handler running locally uploads a160-byte test ZIP
  to actual Wormhole/Backblaze; independent download/decryption and official
  Wormhole UI download both match its ZIP bytes. The authorized Discord test
  message is acknowledged and read back with the full keyed URL.
- Native Chrome at the canonical site origin (local source routed for this
  prepublication check) reaches the actually deployed Cloudflare Worker,
  uploads two encrypted pieces and receives Wormhole completion200 and Discord
  acceptance200. The exact message is read back and labelled as a synthetic
  Cloudflare browser test. Independent download/decryption restores all9,437,363
  ZIP bytes with SHA-256
  `808a69e028c3d770bdda050982a035cdf04c0143df56f16908f3e8df11097a6f`.
  No participant data, webhook/access keys or keyed links enter public evidence.

Ignored receipts include `wormhole-network-verification.json`,
`automatic-sharing-results.json`, `link-forwarding-results.json`,
`wormhole-results.json`, `files-auto-regression.json` and
`cloudflare-browser-local-source.json`. Private receipts/configuration/claim
links stay local. The Worker is on a temporary Cloudflare account; successful
runtime verification does not establish durable ownership. The user must
complete the claim within its window, or log in and redeploy. Physical phones,
Safari and large real study archives remain unqualified. Runtime release `0d0e170` completes exact Pages run42309917241 and reports
`succeed` from the deployment API. All32 changed public files match Git bytes;
JavaScript MIME types and both entry URLs pass. Native Chrome repeats the full
real upload/Discord/readback/independent-decryption check from published Pages
source with the same9,437,363-byte synthetic ZIP/SHA-256 above. The live origin
also passes five upload, four forwarding, seven manual-QR and ten files-dialog
cases, with no page errors. Ignored receipts: sharing-release-0d0e170.json,
cloudflare-browser-live.json and files-auto-live.json. This qualifies the current
hosted workflow, while the account claim/operator and physical-phone acceptance
remain open. Publication/checkpoint state is owned by PROJECT.

## Wormhole ZIP handoff and local QR — 2026-11-06

The new folder-dialog action packages the existing saved CSVs and guides manual
upload on Wormhole, followed by full-link paste and local QR generation. It does
not implement automatic upload. Actual `/api/room` preflight with site Origin,
POST and content-type returned405 without CORS permission; the current Pages
site has no backend. Official [security](https://wormhole.app/security),
[FAQ](https://wormhole.app/faq) and [roadmap](https://wormhole.app/roadmap)
were read. No participant files or test ZIPs were uploaded to Wormhole.

Both new Node regressions were observed failing before implementation, then
passing. All24 Node tests, UI-state/context checks and whitespace checks pass.
The ZIP from the new handoff download independently extracts all three synthetic
fixture CSVs byte-for-byte. A separate QA-only jsQR1.4.0 decoder reads the actual
rendered SVG back to the complete expected URL, including its22-character key.
The pinned MIT QR encoder is local and lazy; native Chrome's initial check
caught Python's `.mjs` text/plain MIME mapping, fixed by serving the unchanged
module as `.js`. No remote QR renderer or proxy is used.

Native headless Chrome passes seven QR layouts (320×568,390×844,568×320,
844×390,768×1024,1280×720,1920×1080): no horizontal dialog overflow, clipped
labels, unmeasured visible text or controls below48px. Existing file-dialog
regression passes10 cases including60 exports, native sharing/cancel/failure,
200% text/spacing, long names, reload and playback continuing across dialog use.
New QR checks cover enlarged text/spacing at320px, independent decoding, invalid
links, edit/refresh clearing, clipboard success/denial, empty-state disabling,
Escape/focus restoration, missing-encoder fallback and delayed import completing
after dismissal. QR generation sends zero external requests and reports no
page errors. Phone screenshot inspected. Evidence stays ignored under
`output/playwright/check-wormhole.mjs`, `wormhole-results.json`,
`wormhole-phone.png`, `files-wormhole-regression.json` and synthetic ZIPs.
Physical phones/Safari, real Wormhole upload/download and full automatic sharing
are not qualified by these desktop checks. Publication evidence follows in PROJECT.

Live release: containing commit `e5f58b49de54502fe96564ecfce45008a35fe964`
completed [Pages run42247056025](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42247056025)
successfully. All six affected runtime/license files match Git SHA-256 bytes,
with JavaScript MIME types; slash/no-slash URLs return200. Native Chrome repeats
the seven QR cases (including independent decode, missing module and delayed
completion) and ten files-dialog cases on the public origin with zero page
errors/external QR requests. Only synthetic browser-local records are used;
there is no actual Wormhole upload. Ignored receipts are
`wormhole-live-assets.json`, `wormhole-live-results.json` and
`files-wormhole-live.json` under `output/playwright/`.

Blind-spot check: Wormhole's manifest/service worker support installed-app
Web Share Target file receipt. The existing native share action can hand off
to available user-selected targets, but this is not a portable automatic
upload/link-return API for the static site. See the [official browser contract](https://developer.chrome.com/docs/capabilities/web-apis/web-share-target).

The evidence-only `9a91534` deployment timed out in `updating_pages`; two
failed-job retries immediately cancelled, even after explicit API cancellation
confirmed the failed deployment was cancelled. Runtime release `e5f58b4` remains
live and verified. Recovery uses a fresh documentation-only commit/build; no
runtime assets, participant data or unrelated media are changed.

## Numbered per-cue CSV sidecars — 2026-11-06

Each of the 53 existing scenario WAV/TXT/JSON groups now has a same-basename CSV.
The first-use numerical prefix and onset timestamp remain unchanged. Each CSV
lists every occurrence of that cue, accounting for all 890 authored events
across the five menu scenarios and the separate 20-minute diagnostic. No audio,
runtime JavaScript, protocol JSON or playlist bytes changed in this increment.

The new Python regression first failed on all 53 missing CSVs, then passed after
generation. `python -m unittest discover -s tests -p test_scenario_assets.py`
independently parses each CSV with the standard CSV reader and checks prefixes,
onsets, every global event order, frame/second times, cycle/block annotations,
matching TXT words and explicit audio/script status against the source plans.
`python tools/build-scenario-placeholders.py --check`, all 22 Node tests,
six scenario JSON/CSV pairs, UI state/context and whitespace checks pass.
Rendered UI/audio reruns are unnecessary for these data/documentation sidecars;
runtime, audio and scheduled events are unchanged. Publication checks follow.
Independent review found no actionable issues and separately audited all 890
events, confirming each is represented exactly once across the 53 cue CSVs.

Publication: `a216168e92b22f09fef1fb6453fc5da3156d7ba9`, Pages deployment API
status `succeed`. Actions `42243976605` attempt 2 was subsequently cancelled
when the concurrent successor `e11a926` started; backend success and actual
live bytes independently establish publication. All 67 checked public files
match local bytes: 53 per-cue CSVs, six generated scenario README TXT files,
six protocol JSONs, the scenario index and app HTML. Local hash evidence is
in ignored `output/scenario-csv-live-a216168.json` in the active checkout.
The active relocated checkout passes the Python regression and package check;
its scenario library was opened in Explorer. This follow-up changes evidence
only; preserve unrelated concurrent source and publication notes.

## Scenario-owned silent placeholders — 2026-11-06

The current UI uses53 independent silent WAV/JSON/TXT cue groups across six
scenario folders, numbered in first-occurrence order, with manifests, asset and
timeline CSVs and playlists. Draft intended wording is labelled separately from
empty actual spoken text. Original library recordings remain unchanged; the
older speech evidence below describes those retained originals. Main is still
exactly900seconds, calibration26–41seconds, with future accelerometer value null.

Observed the missing-registry test fail before implementation; the native
browser also exposed input-buffer detachment during Web Audio decode. Reproduced
that detachment in the mock and observed the regression fail, then fixed it.
All22 Node tests, six scenario pairs,53 generated placeholder groups, UI/context
and whitespace checks pass. Twelve native Chrome offline renders at44.1/48kHz
have exact lengths and zero sample error. The68-second export is entirely silent.
Five real playback routes pass Start/Pause/Resume/End, actual26/41-second main
transitions, calibration hook, scenario receipt hashes,900-second CSV metadata,
replay and missing-WAV retry. Seven unique silent durations are fetched/decoded,
with scenario-specific buffers verified by a different same-ID recording test.
All18 root/main responsive, enlarged and long-text cases pass; phone screenshot
inspected. No page errors. Read-only independent review found no issues and
independently checked53 pairings, six manifest hashes, cue order and playlists.
Final narration still requires wording approval; no new speech was synthesized.

Publication recovery: feature6f097cf is pushed, but Pages run42241115947
attempt1 timed out in updating_pages after10minutes and failed-job attempt2
was cancelled immediately. Build passed; artifact size2,146,337,782bytes
produces the existing1GB warning. The affected live manifest still returns404.
Trigger a fresh evidence commit/build, then verify its exact deployment and UI.

Resolved: eafec8a836b3a92851ed6b8816806ada103a4d06 deployed in run42243359695
attempt2 after explicitly cancelling the stale6f097cf Pages deployment lock.
All265 changed public files match their committed bytes and both entry URLs
return200. Five live browser routes pass actual26/41-second transitions,
pause/resume, CSV metadata, receipt hashes, replay and missing-WAV retry;
18 enlarged/localized text cases pass with no page errors. The concurrent
a216168 CSV-sidecar increment preserves runtime and audio; retain its owner.

## Single 15-minute main sequence — 2026-11-06

Current acceptance supersedes the earlier two-option/48-second-composite local
checks. The main path is direct and contains exactly 26 + 15 + 40 + (50 × 16) +
19 = 900 seconds, with no baseline/rest/recovery event. Calibration is an explicit
silent asset at 26–41 seconds; Instruction 1 contains the German two-breath
instructions. Instruction 2 reuses instruction.wav. Outro contains seven quiet
seconds plus the original 12-second ending PCM, unchanged. All original nine
WAVs remain intact. Asset classes/transcripts are in the manifest and
main-asset-classes.csv; adjacent JSON sidecars retain hashes/provenance.

All three Instruction 1 phrases passed Secret Tunnel's native Whisper check
with zero word edits; no speech was stretched. Original receipts and the unused
earlier composite remain ignored/private. Repacking uses the verified MP3 cache
and rejects corruption. Respyra source comparison and exact German narration
are in audio/calibration.md. H10 accelerometer integration is future work;
calibration value/unit/method remain null and status not-integrated.

The new single-main-choice regression was observed failing before implementation.
All 21 Node tests, six scenario JSON/CSV pairs, UI state and context checks pass.
Tests cover exact class boundaries/totals, all fifty complete cycles, pause/resume,
no retrigger, receipt and advance-event calibration timing, and unchanged outro
source samples. Native Chrome silently rendered all six scenarios at 44.1/48 kHz (12 renders),
with exact totals and zero maximum sample error. The revised main has 900 seconds,
50 complete cycles, no baseline/rest/recovery, and the proper class order.
All 137 responsive/Pretext cases and 18 main/root text cases (320×568,568×320 and
1920×1080, ordinary/200% plus spacing/long labels) pass. Actual browser checks
covered all five menu playback routes, direct main entry with no duration screen,
calibration transition at 26 s and transition to Instruction 2 at 41 s, pause/resume
inside calibration, one fresh window on replay, advance scheduling event with
unset value, exact saved scenario hashes and participant CSV phases/duration.
All 12 shared WAVs and six JSONs preload once; missing-calibration failure/retry
works. No page errors occurred. Screenshots were inspected. Independent final
review found no actionable runtime issue and confirmed original nine WAV bytes.
Current ignored evidence is output/playwright/calibration-final-* and the
scenario-offline JSON reports. Feature publication passed: commit 23b5f2821eb5d0fd13bdc71af7075268fd3b2e30,
Pages run 42225450192, successful/built for the exact commit. All 54 changed public
files match committed bytes, both entry URLs return 200, and all three removed
main10 asset/review URLs return 404. Live Chrome repeated all five flow checks and
18 main/root text cases with no errors, including real 26/41-second calibration
boundaries, clock freeze/resume, timing event/null measurement, CSV phases and
900-second budget, replay and missing-asset retry. Local/remote main match the
feature SHA at this evidence write. The closing documentation follows the same
publication rule; runtime acceptance is complete.
Physical audio timing, listening acceptance and hardware calibration remain open.

## Responsive proportions — 2026-11-06

The patient composition is centered and capped at 560px high; its equal-column
form is at most 640px wide, uses the same gutters as the gender grid, and caps
the flexible group region at 384px. All corners share a responsive 56–112px
size and proportional artwork. Default body type is 16–18px. Enlarged/no-fit
states retain document reflow and full labels; existing Pretext owns measurement.

All 19 Node tests, seven scenario pairs, UI-state and context checks pass.
`check-proportions.mjs` passes 149 native Chrome cases with no page errors:
320×568 through 1920×1080, portrait/short landscape, breakpoint sides, all menus
and six ready players, exact corner bounds/fixed positions, centering, 200%
text/spacing, keyboard/focus access, long ID/label, locale and resize recovery.
The ten-case folder suite also passes individual/ZIP downloads, share boundary,
modal focus, reload and active playback. The first folder run lost its shared
HTTP server; fresh suites against a dedicated server pass. Phone/desktop PNGs
were visually inspected. All QA artifacts remain ignored in `output/playwright/`
(`check-proportions.mjs`, `proportions-results.json`, `proportions-files.json`).
Physical phones, Safari/WebViews and assistive technology remain unqualified.
Two missing-module/Intl cases retain readable full-label reflow and successful
form navigation. Independent review finds no actionable issues.

Layout commit `f2d683c74bf1c32d56f36f183d27b31dc1938829` is published within
`b164c805c1df1b43f565d3ee69aeffda53eb7307`. Its exact Pages run
[42213877838](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42213877838)
succeeded; the first layout run was automatically superseded by this concurrent
documentation update. All seven changed public files match Git bytes and both
entry URLs return 200. All 149 live layout cases and ten live folder cases pass
with no page errors. Receipts are `proportions-live-results.json`,
`proportions-files-live.json` and `live-release-assets-b164c80.json` in the same
ignored directory. Main/origin match `b164c80` with a clean tree before this
closing evidence update. Publish/check this documentation-only follow-up under
the standing rule; report its exact final Git/deployment state at handoff.

## Polar artwork and frame glow — 2026-11-06

The separate dot is removed. CSS uses the unchanged PNG as a luminance mask;
the original strap artwork and outer frame share state color and cycle duration.
Connected pulses green over four seconds, lost pulses red over two seconds,
waiting is neutral, and reduced motion retains steady state-colored glows.
Accessible status text and the connection/reconnect state owner are preserved.

`check-belt-glow.mjs` passes 24 native Chrome cases: waiting/connected/lost at
320×568, 390×844, 568×320, 844×390, 768×1024, 1280×720 and 1920×1080, plus
reduced-motion loss/recovery and restoring motion. It checks no dot, luminance
mask/source decode, exact symbol/corner sizes, both animations and matching
durations/colors, changing symbol opacity/filter and frame shadow, folder access,
and status text. No page errors occur. Peak green/red screenshots were inspected.
UI-state/context and whitespace checks pass; focused review has no actionable
findings. Receipts and crops remain ignored in `output/playwright/belt-glow-*`.
No bitmap regeneration, logging, playback or layout changes are involved.
Native phones/WebViews remain outside this Chrome evidence.

Release `7e07b9c6b0abd9c8d3e22d54abd339bec130e297` is pushed and live. Pages run
[42213036555](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42213036555)
succeeded; all six published files match Git bytes and both entry URLs return 200.
All 24 live glow checks pass with no page errors. Receipts are
`belt-glow-live-results.json` and `live-release-assets-7e07b9c.json` in the same
ignored directory. The subsequent independent proportions commit `f2d683c`
retains the glow while superseding the original dimensions; its owner tracks
that layout release. Publish/check this closing documentation-only evidence
under the standing rule, preserving the separate task's state.

## Threefold corner enlargement — 2026-11-06

All three fixed surfaces measure exactly 144×144 CSS px. Recenter/folder artwork
is three times its previous dimensions. Polar embeds the unmodified dissertation
`PolarH10.png`, Git blob `71fef4c5acd3cb20122be459675963d476d5bc89`;
the recovered original is raster, not a newly claimed vector asset.

All 19 Node tests, seven scenario pairs, UI/context and whitespace checks pass.
`check-large-corners.mjs` passes 85 native Chrome cases across 320×568, 390×844,
568×320, 844×390, 768×1024, 1280×720 and 1920×1080: patient, ECG, round/options,
all six players and legacy recovery; exact fixed corner bounds, loaded images,
full-label Pretext/DOM checks, bounded centering, inert recenter, keyboard folder
open/close/focus, signal loss/recovery, reduced motion, 200% text/spacing and focus
access. `check-files.mjs` passes ten cases plus downloads/reload/active playback.
Neither suite reports page errors; review has no actionable findings.

The larger fixed surfaces change the supported no-scroll boundary. Patient,
round/options and player screens at 320×568 use document reflow; landscape patient
forms and 568×320 practice choices also reflow. Other tested ordinary cases fit
bounded geometry. Short landscape reserves a central column; >=1280px desktop
retains the centered content lane. The former 161-case suite's blanket bounded
assertion is not applicable to these revised minimums. Physical devices and
screen readers remain unqualified. Screenshots were visually inspected; local
receipts/scripts remain ignored in `output/playwright/large-corners-results.json`
and `corners-files-local-results.json`.

Release `0a3c99a83301c3650d79dca1f88a52389494b2c8` is pushed and live. Pages run
[42208617785](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42208617785)
succeeded. All eight published files match Git bytes, including the exact Polar
image; both entry URLs return 200. The live 85-case corner matrix and ten-case
folder suite both pass without page errors, including download/reload/playback
behavior. Main/origin match with a clean tree before this closing evidence update.
Live receipts are `large-corners-live-results.json`, `corners-files-live-results.json`
and `live-release-assets-0a3c99a.json` in the same ignored directory. Publish/check
the closing documentation-only evidence under the standing rule.

The closing-evidence run `42209351688` for `463c842` timed out while syncing files
and its failed-job retry was immediately cancelled. Build remained successful;
the site artifact is 2,143,894,180 bytes and Pages warned of a 1 GB allowance.
The prior UI deployment remains live and verified. Recovery uses a fresh
documentation commit/run rather than modifying runtime files or unrelated media;
confirm its exact result before claiming that the evidence is live.

## Folder-only data entry — 2026-11-06

Both CSV-Dateien footer links and their obsolete visibility update are removed.
The existing folder dialog remains the visible entry; the legacy recovery URL
and shared link nodes remain compatible. All 19 Node tests, seven scenario pairs,
UI-state/context, syntax and whitespace checks pass. Focused review has no findings.

`check-files.mjs` passes ten Chrome layout cases plus individual/ZIP downloads,
keyboard/focus, share-boundary and 61-file reload/playback checks with no errors.
The first full stretch check exposed the remaining patient link's 180px singleton
budget at 768×1024. A scoped 240px budget fixes that full-label no-fit without
changing other footers; all 161 cases then pass with no errors. The final desktop
screenshot was visually checked: only the centered patient return and corner
folder remain. Receipts are ignored under `output/playwright/` in
`folder-only-local-results.json` and `stretch-results.json`. Native phone sharing
and real hardware remain outside this desktop evidence.

Release `0895f63213792e8a5994712b137620f1dd91ef70` is pushed and live. Pages run
[42203147921](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42203147921)
completed successfully. All nine changed public files match committed bytes;
both entry URLs return 200. The live folder suite passes all ten cases and its
download/reload/playback sequence with no page errors. Local/remote main match
with a clean tree before this closing documentation-only update. Receipts are
`folder-only-live-results.json` and `live-release-assets-0895f63.json` in the same
ignored output directory. Publish/check the closing evidence under the standing rule.

## Folder and standing-workflow release acceptance — 2026-11-05

For the user's explicit all-changes push, refreshed origin/main and local HEAD
both start at `7233468`. Reviewed the complete intended source/assets/tests/notes
and new root/subproject standing Superpowers/publication guidance. Fresh checks
pass all 19 Node tests, seven scenario pairs, UI state, context, JavaScript syntax,
SVG XML and Git whitespace. Cold opening 60 retained files uses exactly two
full-store reads. Ten native Chrome popup layout cases and its download/share
boundary/cancel/error/reload/playback checks pass; the full 161-case stretch
matrix also passes. Both browser suites report zero page errors. Private/local
participant, voice and diagnostic artifacts remain ignored. The following release
record will identify the exact pushed commit, Pages result and live verification.

Release `a52d56a875cc6c30f149f1c4831e098ac12c8711` is pushed to main, with
local/remote SHA equality and a clean tree before the closing evidence commit.
GitHub Pages run `42079217920` completed build and deployment successfully.
`check-live-release.mjs a52d56a875cc6c30f149f1c4831e098ac12c8711` verified all
20 public changed files byte-for-byte against Git and HTTP 200 at both entry
URLs. `check-files.mjs https://viscereality.org/atmungsraum/ files-live-results`
passed all ten layout cases and the full synthetic download/share-boundary/
recovery/playback sequence against the live site, with zero page errors.
The native app share boundary remains simulated; no email or message was sent.
The exact-commit Pages run is
<https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42079217920>.
Receipts are ignored in `output/playwright/files-live-results.json` and
`live-release-assets-a52d56a.json`. Standing instructions were reviewed with no
material contradictions; the subsequent evidence commit changes documentation
only and preserves the verified runtime bytes.

## Saved-files popup and ZIP/share — 2026-11-05

`npm test` passes all 19 tests. The ZIP check reads the actual Blob with Python's
independent standard-library zipfile reader, verifies CRCs and exact CSV bytes,
and extracts Unicode filenames and a same-name collision without data loss.
ZIP32 uses stored entries, with no new runtime dependency; the interoperability
test uses the existing Python interpreter. UI state, seven scenario pairs,
context and Git whitespace checks pass.

`node output/playwright/check-files.mjs` passes ten native headless Chrome layout
cases with zero page errors: empty popup; 320×568, 390×844, 568×320, 844×390,
768×1024, 1280×720 and 1920×1080; 200% root text with WCAG-style spacing; long
German headings and unbroken filenames. Actual Pretext measurement is present,
labels fit horizontally, and controls retain 48px targets. The list scrolls by
design; short/enlarged cases allow whole-dialog vertical scrolling. Phone and
desktop screenshots were visually inspected. The fixed folder has safe-area
offsets and centered footers leave room beside it.

Functional checks use 60 synthetic saved CSV fixtures in an isolated browser:
empty state, individual/ZIP downloads, ordering, unsupported-share fallback,
supported sharing payload and active user gesture, cancellation/rejection,
native-dialog keyboard containment/focus restoration, all menus and unchanged
hash. Opening/dismissing the popup during actual browser audio leaves playback
running with advancing time. Ending that run adds a 61st CSV; all 61 reappear
after reload before patient/sensor entry. Only the platform share API was
simulated; no destination app opened and no message was sent. Physical phones,
email, WhatsApp, Safari/WebViews and screen readers remain unqualified.

The existing `check-stretch.mjs` passes all 161 rendered/playback/reflow cases
with zero page errors after the footer reservation change. Receipts, executable,
synthetic ZIP and screenshots remain ignored under `output/playwright/`:
`check-files.mjs`, `files-results.json`, `files-fixture.zip`, `files-*.png`.
No participant data or generated diagnostic enters public source. The original
implementation-only check did not claim publication; the release acceptance
entry above now records the verified live release.

Focused code review caught a cold-open hydration notification loop. Notifications
now originate only when a run finishes, rather than once per hydrated record.
The ignored `check-files-cold.mjs` regression reproduced 62 full export-store
reads for 60 files before the fix, then exactly two reads with all 60 files and
ZIP readiness after it. The reviewer revalidated the fix with no remaining
actionable findings; all 19 Node tests and the popup functional suite were rerun.

## Stretch-accordion acceptance — 2026-11-05

The main UI now follows the explicit settings in SKILLS.md: equal outer anchors,
dimension-responsive gaps/type, stretched patient/CSV groups, narrow stacked
footer actions and short-landscape arrangements. Practice choices become four
columns; round/main cards use wider rectangles where squares would force wrapping.
The existing Pretext 0.0.9 runtime checks inner text boxes and painted DOM bounds.
Fixed actions check natural width, line count and line height; multiline text
checks widest line and height. User word spacing is marked DOM-verified because
it is outside Pretext's model. No user enlargement is reduced to satisfy a box.

`node output/playwright/check-stretch.mjs` passed 161 local native headless Chrome
cases with zero page errors: 320×568, 320×1080, 390×844, 768×568, 768×1024,
1280×720, 1920×568, 1920×1080, 568×320, 844×390, 480×640, 481×640, 844×480
and 844×481 CSS px across patient, Polar, rounds, both option menus, empty/filled
CSV recovery and ready players. All six real browser audio routes passed
Start/Pause/Resume/End in phone portrait and short landscape. Bounded panels had
no page/group scrolling, clipped labels or targets below 48px; central controls
stayed within 2px of the viewport midpoint and footers retained their anchor.

At fixed 768px width, patient group separation grew from 25.08px at 568px height
to 145.44px at 1024px height. Column separation grew from 12.8px to 30.72px with
width, reaching the 40px ceiling on desktop. Preferred body type grew from 16px
to 20px when both dimensions allowed it; wide/short type stayed 16px. Portrait
forms at 320×568 fit bounded geometry. Short landscape forms and filled CSV lists
that exceed the middle budget use deliberate vertical document reflow, with no
inner scrollbar or horizontal overflow. They are explicit supported-size
exceptions, not strict no-scroll passes.

Additional cases passed 200% root text with WCAG-style spacing, long German round,
status/action labels, an unbroken identifier, locale invalidation and recovery
after text/viewport changes. Word-spacing cases use rendered bounds, not a
Pretext accuracy claim. Two separate fallbacks verified blocked Pretext modules
and missing Intl.Segmenter: both marked measurement unavailable and retained a
usable form in readable document flow. Reflow explicitly schedules a second
measurement even when a wrapped label's box dimensions do not change; hidden
status nodes are excluded from geometry checks. Patient Tab/radio-arrow flow
and derived-ID updates passed. Phone, desktop and short-landscape screenshots
were inspected; executable, JSON receipt and PNGs stay ignored under
`output/playwright/check-stretch.mjs`, `stretch-results.json` and `stretch-*.png`.

Fresh checks also passed all 19 current Node tests, seven JSON/CSV pairs,
UI-state/context, app syntax and Git whitespace. Concurrent CSV-folder/dialog
work is preserved and owns its separate overlay acceptance. No package upgrade,
audio protocol or participant logging change was introduced by this layout task.
These are desktop Chrome results; physical phones, browser zoom, Safari/WebViews,
screen readers, real H10/headset behavior and live final-refinement deployment
were not qualified by this check. RTK was unavailable; native bounded commands
were used. See PROJECT.md and the publication task for observed Git/release state.

## Patient/ECG and recenter release checks — 2026-11-05

For the user's all-changes publication request, refreshed remote main matched
starting HEAD `ac39ec4`. Reviewed all 26 changed/new files: runtime/source,
tests, instructions, native recenter SVG, synthetic NeuroKit JSON/binary and
the parent subproject contract. Local participant exports and diagnostic/voice
artifacts remain ignored. No hosting settings or authored audio were changed.

Fresh release checks pass all 18 Node tests, seven scenario pairs, UI state,
context and whitespace. The added background-stream regression records at
source indices 4 and 20 around an idle interval, retaining the same epoch and
sample array; each export contains only its own recorded interval. Existing
pause checks verify ongoing ECG sample changes with frozen audio time. This
proves the current demo's acquisition/recording separation, not real H10 input.

Native Chrome reran all 25 patient/integration cases, actual >30-second loss,
natural 68-second completion, six CSV routes and four fallback cases without
page errors. All 50 recenter cases passed on the final tree. An initial
concurrent recenter run timed out waiting for Pretext readiness; a diagnostic
rerun passed without a product-code change or page errors. Exact deployed
commit, live asset integrity and browser evidence will be recorded here after
the Pages build. The Pages connector returned an internal error; the existing
authenticated GitHub CLI/API confirmed the main/root Pages source and prior
`ac39ec4` build, and is the deployment-verification fallback.

Initial release `e2b6e32` pushed successfully with remote/local SHA equal.
While its Pages build ran, additional `app.js`, `index.html` and `style.css`
stretch edits arrived. They improve DOM/text fit checks, locale/font/content
remeasurement, dynamic-viewport sizing, patient/CSV vertical distribution,
column gaps and the readable reflow path. Reviewed their complete diff and
reran all 50 Chrome cases against this updated working tree: all passed, no
page errors, clipping or horizontal overflow. Extended browser evidence records
two successive runs at source positions 2313–2436 and 2607 onward, proving the
background source advanced through the intervening menu/idle time. Every CSV
ECG value matched the shared binary; 102 paused ECG rows retained one audio
time while ECG values changed. This late layout work is included in the same
authorized publication, preserving the existing central controls and recenter
placeholder contract.

Pages run `42059584563` succeeded for `e2b6e3230df84870109ac9dc3932c53b891736b4`;
the Pages API reported that exact commit built. All 25 public changed files,
including the ECG binary and documentation, matched committed SHA-256 bytes.
Both slash/no-slash entry URLs returned HTTP 200. Live Chrome passed 50 cases
with no page errors. The first recording started at source sample 5300 and
ended at 5424; the next started at 5605. Every exported ECG value matched the
published binary, and 102 paused ECG rows retained frozen audio time while
ECG values changed. The desktop screenshot was reviewed. Live scripts/receipts
and synthetic downloads stay ignored as `live-release-assets.json` and
`recenter-live-*` under `output/playwright/`.

The final native Chrome stretch matrix passed all 156 cases with no page
errors. It checks independent width/height budgets and breakpoint sides,
patient spacing growth, column/type growth, preserved center/footer anchors,
every screen and all six playback/pause/resume routes, short landscape,
enlarged text, long content/recovery and unavailable measurement. Screenshots
and the detailed receipt remain ignored under `stretch-*`. The final source
also uses rectangular round cards in short landscape and four practice columns
where necessary to retain bounded panels. Publication of this follow-up still
requires exact-commit deployment evidence.

Follow-up source/evidence commit `72334681b5e447f7298aacf58b255fc5543b99b6`
pushed with local/remote SHA equal. A fresh check of that committed layout passed
all 156 Chrome cases. Pages run `42060990832` completed build and artifact upload,
but deployment attempt 1 returned GitHub HTTP 500 while creating the deployment.
The GitHub connector successfully requested rerun of failed jobs only; successful
build/report jobs and the artifact are reused. A Pages API `built` response alone
does not establish that the deploy job succeeded. Later CSV/ZIP and Pretext guard
edits remain local under their separate active tasks, outside this release snapshot.

Deployment attempt 2 also returned HTTP 500; attempt 3 retries only the failed
job after a delay. The artifact-size warning is a separate standing concern:
this upload is 2,142,307,246 bytes versus 2,142,305,761 bytes for the successful
preceding release, so the warning alone does not establish this failure's cause.
[GitHub's limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)
state a 1 GB published-site limit. The official
[GitHub status page](https://www.githubstatus.com/) reported Pages operational
and no incident on 5 November at the diagnostic check. Do not describe these
repository-specific failures as a confirmed platform-wide outage or remove
existing site media without a separate reviewed plan.

Attempt 3 succeeded: run `42060990832` reports success for exact commit
`72334681b5e447f7298aacf58b255fc5543b99b6`, and the Pages API reports it built.
All 25 public files changed since `ac39ec4` match their committed SHA-256 bytes;
both entry URL forms return 200. A final sequential live Chrome pass verifies
all 50 recenter/layout/no-op cases with real Pretext and no page errors. Two
recordings start at source positions 7395 and 7689 (first ends at 7523); every
ECG value matches the published binary. The 103 paused ECG samples retain one
audio position while ECG values change. The final desktop screenshot was reviewed.
Receipts/screenshots/synthetic CSVs stay ignored as
`live-release-assets-7233468.json` and `recenter-live-final-*` in `output/playwright/`.

Initial concurrent final browser runs hit the 30-second gate timeout while
the ECG binary and a Pretext module were still downloading. No JS or validation
error occurred; an independent uncached binary fetch took 13.93 seconds versus
1.32 seconds for a query-qualified fetch, with identical bytes. The sequential
rerun allowed a 90-second network timeout and passed without changing released
code. This establishes functionality, not a loading-speed budget or physical
phone qualification. The local final-source stretch matrix passed 156 cases;
the wider live matrix stopped at that same initial download timeout, while the
final 50-case live matrix completed successfully.

At final Git observation local and remote main equal `7233468`. Newer CSV/ZIP
work, an extra Pretext capability guard and these closing notes remain local
in the shared checkout and are outside the verified release snapshot. They
must not be reported as deployed by this task.

A separate fresh live navigation also reached Polar found, connected and the
round menu. `live-rounds-fresh.png` was reviewed with the heading, both footer
actions and both corner controls visibly rendered; their DOM names, colors,
bounds and hit targets were confirmed. An earlier automated capture omitted
the header/footer paint despite visible measured bounds, so use this stable
fresh capture for the final visual evidence rather than inferring pixels from
geometry alone. The only console warning in that diagnostic was the site's
unrelated favicon 404; the functional final 50-case run had no page errors.

## Patient entry, NeuroKit mock and participant CSVs — 2026-11-05

All 17 Node tests pass, including existing audio/scenario regressions and new
identity/filename validation, pause-aware sample export, missing-signal gaps,
delayed-stop boundary, signed pre-start waiting, checkpoint reconstruction and
shipped mock-source integrity. UI-state, seven scenario JSON/CSV pairs, context,
JS syntax and Git whitespace checks pass. The actual Python generator produced
460,800 finite float32 samples at 256 Hz using NeuroKit2 0.2.13. Binary SHA-256:
`ebe8407430f905f53118ce66c3858815d4de9383bad2a8ef984689ecf8dc6187`.
No runtime speech, audio samples or authored scenario timing changed.

Native Chrome passed 25 rendered/integration cases: patient entry at 320×568,
390×844, 768×1024, 1280×720, 1920×1080, 568×320, 844×390, 480×640 and
481×640 CSS px; 320px with 200% root text, long German labels and WCAG-style
text spacing; round selection; playing/paused states for all six scenarios;
and the CSV recovery view. Normal supported portrait/desktop panels stayed
bounded; compressed/enlarged text used readable document reflow with no inner
scrolling, clipped labels or horizontal overflow. Actual locked Pretext reported
ready/fit. All checked controls retained at least 48px targets and playback
remained at the viewport midpoint. Phone, desktop and enlarged screenshots were
reviewed; local valid-input screenshots were also captured.

All six session routes downloaded the expected `Fej_35_M_probe1..4.csv` and
`Fej_35_M_main1..2.csv`, with common identity/schema, correct scenario, actual
source values and status. Audio elapsed froze during pause while mock ECG
continued. Reload retained completed exports and counters. A real uninterrupted
loss stayed in the session through 29 seconds, then returned to Polar beyond
30 seconds, retained patient details and exported `main3` as aborted with blank
ECG values during the gap. A full real-time 68-second Anleitung naturally
completed and exported `probe5` with `completed` status. Reduced-motion Chrome
used no indicator animation. There were no page errors.

Additional Chrome checks passed native form validation, keyboard Tab/radio-arrow
order with correct M/F/O/N ID updates, unchanged typing focus on sensor updates,
JavaScript-disabled submission protection, missing ECG asset blocking/retry,
unavailable Pretext reflow and interrupted-run recovery with network ECG blocked.
The independent read-only reviewer reproduced and verified fixes for delayed
counter reservations losing signal marks/extending aborted tails, stale failed
attempts aborting a newer run, and pre-start pauses losing signed audio lead.
The reviewer verified IndexedDB checkpoint recovery without network source
access and that another tab cannot recover a recording while its Web Lock is
held; teardown released the lock. No P1/P2 findings remained in that review.

Detailed receipts/scripts remain local under `output/playwright/`:
`check-patient-data.mjs`, `patient-data-results.json`,
`check-patient-fallbacks.mjs`, `patient-fallback-results.json` and the screenshots.
RTK was unavailable at its documented location, so bounded native commands were
used. These are local desktop Chrome checks, not physical-phone/Safari testing,
H10 acquisition, acoustic synchronization, study-data qualification or live
publication. Recovery ends at the last durable checkpoint and can omit an
uncommitted tail. See DATA-LOGGING.md for the complete logging contract.

## Persistent VR recenter placeholder — 2026-11-05

Native headless Chrome passed 50 rendered cases using the local HTTP-served
page. Patient entry covered 320×568, 390×844, 768×1024, 1280×720, 1920×1080,
568×320 and 844×390 CSS px. Round selection, both option menus, CSV recovery,
Anleitung and 15-minute main ready screens covered six phone/desktop/landscape
sizes; additional checks covered Polar discovery/ECG, playing, paused, 320px
with 200% text plus text-spacing overrides, and scrolling the enlarged page.
The button stayed fixed, visible and topmost with a 48×48px target. Heading text
cleared the control. No horizontal overflow, clipped measured labels or page
errors occurred; actual Pretext was ready and the labels fitted. Bounded
playback retained its viewport midpoint. Reflow remains available where needed.

Pointer clicks and Enter/Space activation left route, UI state and audio state
unchanged on patient entry, Polar, every menu/recovery/ready screen, playing,
paused and scrolled reflow. Keyboard focus remained visible. The control exposes
its German name and unavailable description; it has no handler, transport event
or command. Native SVG parsed as XML with no embedded raster. The phone patient,
paused-player and desktop screenshots were reviewed. The executable, 50-case
receipt and screenshots stay ignored under `output/playwright/check-recenter.mjs`,
`recenter-results.json` and `recenter-*.png`.

UI-state, context and Git whitespace checks passed. This task adds no JavaScript
behavior, dependency or participant logger. RTK was absent at its configured
path and on PATH, so bounded native shell reads were used. Physical phone safe
areas, Safari/WebViews, screen-reader output and actual headset recentering were
not qualified by this desktop Chrome check. No commit/push/deployment was
performed; existing patient/logging work was preserved in the shared checkout.

## Combined release and nested skill baseline — 2026-11-05

The subproject entry chain now routes `AGENTS.md` → `for-ai/README.md` →
`for-ai/SKILLS.md`, using the installed For-AI scaffold's full shared baseline
and existing menu-specific rules. Review confirmed RTK, GSD Core, Superpowers,
Ponytail, Pretext, accessibility, performance, MobX and conditional native/remote
routes, one state owner and Git/publication safeguards. The unavailable older
remote skill identifier is explicitly mapped to the installed BRSP skill.
No product dependency or native stack selection was added.

Fresh prepublication checks passed all 13 Node audio tests, seven scenario
JSON/CSV pairs, UI state, context and whitespace. Native Chrome passed the
13-case round-layout check, path-only SVG alpha checks, 24 player-layout cases,
and all six scenario Start/Pause/Resume/End routes with exact JSON receipt
hashes, nine shared WAV fetches, missing-JSON failure and successful retry.
No browser errors occurred. The existing ignored executables and receipts in
`output/playwright/` own detailed evidence; prior native offline-render evidence
is recorded in the scenario section below.

Release `5a0e3a93e6bccf70c1d30091aaa66e72b1d7869e` was pushed to `main`,
with local and remote HEAD equal. Pages run `42048199569` succeeded; the Pages
API reported that exact commit `built`. On the public site all 26 checked text
assets matched the committed content: HTML/CSS/app and audio modules, the shared
manifest, two SVGs, the AGENTS/README/SKILLS entry chain and seven JSON/CSV pairs.
Both slash/no-slash URLs resolved successfully. Live Chrome verified both round
branches at 900×553 and 320×568 CSS px with real Pretext, no page overflow and
no page errors; the desktop screenshot was inspected. A separate live browser
pass exercised every one of the six scenario Start/Pause/Resume/End paths,
exact JSON receipt hashes, nine shared WAV fetches and missing-JSON failure/retry.
All passed with no page errors. Receipts/screenshots remain ignored under
`round-infographics-live*` and `scenario-player-live*` in `output/playwright/`.
Hardware and listening acceptance remain separate open checks.

## Scenario audio protocol — 2026-11-05

The six menu choices and engine-only 20-minute diagnostic each have JSON/CSV
timelines in `audio/scenarios/`. Playback, native offline diagnostics and WAV
verification exports now consume saved JSON. The nine shared library WAVs and
approved timing remain unchanged; a regression proves all seven saved plans
equal their former calculated plans, with explicit null assets for silence.

All 13 Node audio tests passed, including proof that an edited JSON asset is
actually scheduled, receipt hashes match loaded JSON, invalid assets/timing/
metadata/library revisions are rejected, and missing/invalid files prevent any
fallback playback. Seven JSON/CSV consistency checks, the UI-state check,
context checker and Git whitespace check passed. A failing LF-byte regression
caught Windows CRLF changing the pinned manifest hash; scoped `.gitattributes`
now fixes LF for the selected manifest and scenario JSON/CSV. Its LF hash is
`1a4fe0f2546d5850865bb2aab8c737cff05244ff40286044189e7a3cd53fb275`,
identical to the published Git blob. Audio sample bytes were not changed.

Native Chrome silently rendered all seven scenarios at both 44.1 and 48 kHz
(14 renders). Every output sample matched the selected clips/silence with zero
maximum error; the exact 68/300/600/900/1200-second totals and silent 300-second
main baselines held. The verification exporter also produced the exact 68-second
Anleitung from saved JSON. All 886 participant text-score ranges, German scripts
and clip references matched the scenario events; their source hashes were refreshed.

Real browser checks passed Start/Pause/Resume/End for all six menu routes,
correct scenario/JSON hashes in receipts and one preload of exactly nine WAVs
and seven JSONs. A missing scenario blocked playback with retry feedback;
restoring it allowed preparation to recover. No page errors occurred. The phone
playback screenshot was inspected. Local executable checks, native render
receipts and screenshots are ignored under `output/playwright/scenario-*` and
`check-scenario-*.mjs`. These checks establish local browser scheduling, not
physical acoustic/video synchronization, headset/H10 behavior, listening
acceptance or deployment. No push/publication was performed for this change.

## Playback buttons and progress — 2026-11-05

The controller displays elapsed/total time and native progress from the same
Web Audio clock as playback. Its extended regression first failed against the
old countdown, then all eight audio tests passed with exact 600/900-second main
totals, a paused 12-second position, the 68-second Anleitung total, completion,
replay reset, signal-recovery gating and cancelled-unlock receipt protection.
The UI-state check, context checker, JS syntax and Git whitespace checks passed.

Native headless Chrome passed 24 rendered cases. Ready, playing and paused
states at 320×568, 390×844, 768×1024, 1280×720, 1920×1080, 568×320 and
844×390 CSS px stayed bounded without document/internal scrolling, clipped
labels, overlap or page errors. Both actions retained at least 48px targets;
the primary button stayed within 2px of the viewport midpoint. Real Pretext
reported ready/fit. Short landscape groups phase and time above the bar; loading
or failure status occupies the upper spare track rather than shifting playback.

The remaining cases verified 320px with 200% root text, combined long German
labels/text-spacing/200% text, and unavailable Pretext. They retained readable
document reflow without horizontal overflow or clipping. Keyboard Space starts,
pauses and resumes; Escape stops. Native progress advanced during playback,
remained exactly frozen while paused and advanced again on resume. Progress is
read-only and is not a seek control. Screenshot review covered phone, desktop
and enlarged-text layouts. The local runnable check, JSON receipts and PNGs
are ignored under `output/playwright/check-player.cjs`, `player-results.json`
and `player-*.png`. This is Chrome/local verification; physical phone browsers,
Safari, H10/headset execution and live deployment were not checked by this task.

## Round-button SVG infographics — 2026-11-05

The user simplified the initial diagrams to practice lungs with one repeat
arrow and a research/data-collection clipboard with a three-point data chart.
Two new built-in image generations were converted with VTracer to two and three
SVG paths, respectively. Runtime SVGs are 8,907 and 12,982 bytes, with the same
240×160 view box and two-color palette. They contain no bitmap, background
rectangle, script or external reference; transparent outer corners and internal
gaps were checked in rasterized alpha. The page loads the SVGs with revised
image cache tokens, preserving visible German labels and link names. Source
PNGs and exact prompts/conversion settings live in `assets/rounds/`.

The 13-case rendered/keyboard/fallback check below was repeated successfully
for the simplified revision. It is now published and live-verified by the
combined release above; `a79275f` and Pages run `42044006553` refer to the
earlier artwork.

Native headless Chrome passed 13 rendered cases: 320×568, 390×844, 768×1024,
900×553, 1280×720, 1920×1080, 568×320, 844×390, 480×480 and 481×480 CSS px,
plus 320px cases with 200% text, user text-spacing overrides and long German
labels. Images decoded, actual Pretext reported ready/fit, labels and images
did not overlap, cards had no internal overflow, and ordinary viewports had
no page scrolling. Enlarged text used readable document reflow. Phone and
desktop screenshots were visually inspected.

Both links retained exactly their visible accessible names, visible keyboard
focus and Enter activation into the expected four/two choices. Blocking Pretext
retained readable choices in document flow. No page errors or failed runtime
requests occurred. Local receipts, screenshots and the runnable check are in
ignored `output/playwright/round-infographics-*` and
`output/playwright/check-round-infographics.mjs`. Physical phones and Safari
were not tested. All eight audio tests, UI-state, context and Git whitespace
checks also passed. RTK was unavailable at the documented Windows path, so
bounded native commands supplied the verification output.

## Centered main UI — 2026-11-05

Before correction, native Chromium measurements showed left-aligned single
footer actions, an off-center sensor-selection button and primary controls
shifted vertically by the unequal header/footer and playback-feedback rows.
The shared shell now reserves equal measured edge tracks, and the play body
reserves equal feedback tracks around its primary control. Headings, single
footer actions, the sensor selection, ECG and choice groups share the centerline.
The shell follows the dynamic viewport, with its existing small-viewport fallback.

Headless Chrome passed 79 rendered cases at 320×568, 390×844, 768×1024,
1280×720, 1920×1080, 568×320 and 844×390 CSS px. Sensor selection, ECG,
both choice grids and idle/playing/paused buttons matched the viewport midpoint
within two CSS px in bounded layouts; single footer actions matched its
horizontal midpoint. All visible labels had real Pretext fit results, with no
clipped text, page overflow or page errors. Phone/desktop screenshots were
visually reviewed. The later 18-case pass covered six routes at 320 CSS px
with 200% text, text-spacing overrides and readable measurement fallback.
Keyboard checks passed sensor selection, Continue, both menu levels,
Start/Space, pause/resume and Escape. All eight audio tests, UI-state/context
checks, JavaScript syntax and Git whitespace checks passed. Local evidence is
under ignored `output/playwright/`; physical phones, Safari and headset behavior
are not established by these Chromium viewport checks.

Public verification: alignment commit `e4f76d9` was carried into `a79275f` after
the first Pages run was superseded by the concurrent asset update. Pages run
`42044006553` succeeded for `a79275f`. Live `index.html`, `style.css` and `app.js`
returned HTTP 200 and matched the alignment commit byte-for-byte. The same
79-case Chromium matrix then passed at <https://viscereality.org/atmungsraum/>,
with no page errors, overflow or text-fit failures. Local geometry error was
zero horizontally and at most 0.0078125 CSS px vertically for bounded primary
controls. Keep concurrent scenario/progress work separate from this evidence;
later changes require fresh checks against their own committed deployment.

## Audio protocol text review — 2026-10-27

`audio-review/` inventories all eight checked-in audio preparation/export tools
plus runtime sequencing/controller/selection and retained manifests. It includes
a participant walkthrough, an explicitly separate audition comparison, and full
text timelines for all six menu choices plus the engine-only 20-minute option.
The timelines were generated from production `createProtocol()` and the runtime
manifest, recording SHA-256 for that manifest, engine, selection and controller.

Fresh checks passed all nine runtime WAV byte counts/hashes; seven timelines
had contiguous events, expected total duration/cycle/rest counts and exactly
two breath variants per third complete cycle. Written source-sample entries
matched the production event counts. Totals were 68/300/600/900 seconds for
Anleitung/practice, 600/900 for main rounds and 1200 for the engine-only option.
An independent reread checked every saved event range, exact German quote,
cycle/block annotation and source hash against the current production plan;
all passed. All eight audio tool scripts were present in the inventory.
All eight Node audio tests, UI-state check, context check and Git whitespace
check passed. Product code and audio assets were unchanged, so rendered UI and
native offline audio checks were not repeated for this documentation task.

The opening promises counting in each phase, while saved holds are uncounted;
this is identified for review without changing speech. Main rounds begin with
300 seconds of silence. Anleitung is one replayable 68-second example, not an
endlessly looping sound sequence. The newer Halten! audition is separate from
the runtime's Und halten. Its documented generated package and source cache
are absent locally; no regeneration, API call, listening assessment, hardware
measurement or fresh public-site verification was performed.
## Voice sample subpage — 2026-10-28

`samples/` contains twenty fully decoded MP3s (397,476 bytes total), each bound
to its saved profile and the exact requested German phrase. The original MP3
hashes and timing are preserved. `node for-ai/scripts/check-samples.mjs` checks
the twenty unique names, both 0–9 series, LANG valence ordering, all source
links/downloads, byte hashes, and truthful transcript checks without private
profile identifiers or machine paths. Native raw Whisper scores are preserved;
digits 2/3/4 and German zwei/drei/vier have a separately documented equivalence
check. Later single-take samples use independent German ASR on the final decoded
MP3, without fabricating a native checked-take score.

Headless, audio-muted Chromium passed seven ordinary viewport checks:
320×568, 390×844, 568×320, 760×800, 761×800, 1280×800, and 1920×1080.
At 320 px, 200% text, WCAG text-spacing overrides, and a long German profile
label passed without horizontal page overflow or clipped measured text.
Pretext measured all 76 text regions with Arial parity. Enlarged text can
enter explicit readable reflow; no font shrinking masks a failure.
All twenty MP3s loaded real positive duration metadata. Silent actual playback
confirmed starting a second sample pauses the first. Missing Pretext and
JavaScript-disabled pages retain all twenty native players and downloads.
No ordinary JavaScript errors occurred. Physical-phone and perceptual listening
checks were not performed. The existing UI-state and context checks also passed;
the session scheduler and its audio assets were unchanged.

Public deployment of feature commit `1f3f5db` was verified on 2026-10-28.
GitHub Pages run `40730446610` succeeded and the Pages API reported `built`.
`/atmungsraum/samples/`, its catalog, and all twenty MP3 URLs returned HTTP 200;
all public audio hashes matched the generated originals. Live 390×844 Chromium
reported twenty players/downloads, ready Pretext measurement and no horizontal
overflow. Real keyboard Space play/pause and the MP3 download filename passed
with browser audio muted. The context checker with `-RequireRemote` passed.

## Browser prototype completion — 2026-10-13

The completion pass preserved the latest approved menu, baseline, playback-control,
and audio-asset decisions. The UI-state checker, all eight audio tests, context
checker, JavaScript syntax checks, Python syntax checks for both pending listening
helpers, and Git whitespace check passed.

The existing native-browser baseline check passed 41 rendered cases covering
main choices, both final screens, playback, pause/resume, 320 CSS px, phone
landscape, desktop, 200% text, text-spacing overrides, and long German text.
The current phone screenshot was visually inspected. Native OfflineAudioContext
checks passed both 600/900-second main sessions at 44.1/48 kHz: every baseline
sample was silent and all scheduled audio samples matched with zero error.

Release `d6af2f2` was pushed to `main`; GitHub Pages run `38460690617`
completed successfully for that exact commit. The live
<https://viscereality.org/atmungsraum/> was checked on 2026-10-13 at 20:16 UTC:
the six HTML/CSS/JavaScript entry files matched the checkout, the no-slash URL
redirected correctly, and browser checks passed the demo gate, both round
choices, four practice choices, instruction playback, and both main sessions'
baseline, pause/resume, stop, and timing receipts. No page errors or failed
site requests occurred. The remote context gate passed and the source tree was
clean after the release. Local QA output is in ignored `output/playwright/`.

The existing runtime samples remain selected; local listening revisions are not
promoted by this release. Real H10 acquisition, headset transport, and VR-event
execution are explicitly outside this verified browser prototype. No hardware
or acoustic-output verification is implied.

## Observed connected count phrases — 2026-10-12

Two complete ElevenLabs phrases were generated with the selected website voice,
request settings and character alignment. Independent PCM comparison proved that
every output word retains its native sample length and waveform, apart from gain
and short boundary fades: interior reconstruction residual was below 0.6 PCM
units and all speech tempo factors are one. Source takes are 4.24/4.48 seconds;
pause adjustment yields exact four-second phase clips. Actual word starts are
0/1.056/2/3 seconds for inhale and 0/1.164/2/3 for exhale. They are recorded exactly
in JSON rather than claimed as uniformly spaced counts. All 13 WAV/JSON pairs,
frame/hash/headroom checks and component/cycle reconstruction pass. The quiet
Halten! clips are byte-identical to the preceding set; breath gain remains fixed.
Each full cycle is exactly 16 seconds. Runtime assets remain unchanged.
Unprompted Whisper-small review recognized all ten requested words in the cycle.

## Observed quiet Halten! listening revision — 2026-10-12

One new cached ElevenLabs take used literal Halten! and the selected website
voice/settings. Both hold roles reuse the same 0.7527-second prompt PCM, followed
by 3.2473 seconds of exact silence. All 13 WAV/JSON pairs passed independent
frame/hash/headroom checks. Holds have one word slot and speech/quiet segments;
the full voice cycle has ten requested words. Unprompted Whisper-small recognized
those ten words with no hold counts. Ten component clips remain four seconds and
three cycles remain 16 seconds, with sample-identical sums/concatenation. Inhale,
exhale, voice/mix and breath samples are byte-identical to the preceding set.
Runtime assets remain unchanged; the previous listening package is preserved.

## Observed constant-breath/Halten listening revision — 2026-10-12

Thirteen adjacent WAV/JSON pairs passed independent frame/hash/metadata checks:
ten four-second clips and three 16-second cycles. Both hold files use exactly
the same cached German Halten/count PCM, with separate role metadata. Transcripts
contain no Pause. Voice counts still descend; voice + breath equals mix at every
sample and cycles concatenate exactly. Against the retimed source recordings,
the breath stems have a constant scalar throughout the interior: fitted gains
vary by less than 0.01% across seconds and reconstruction residual is under one
PCM unit. Only the 10 ms boundaries fade. Average component power remains matched
within 0.0001 dB; all audio has at least 3 dB sample headroom. Runtime assets remain
unchanged, and no new synthesis/API call was needed.

## Observed minimal UI — 2026-10-12

The final UI replaces three playback buttons with one Starten/Pause/Fortsetzen
button and folds Beenden into the existing back action during playback. Round
icons/descriptions, repeated final-plan text, ready/playback messages, and their
unused wrappers/styles are removed. Search retry is shown only for recovery.

All eight audio tests, the UI-state check, context/source check, JavaScript syntax
checks, and Git whitespace check pass. The controller check proves the same
button pauses/resumes the original schedule, leaves Pause usable during signal
recovery, gates Fortsetzen on current ECG, aborts via Beenden, and preserves the
receipt guard when cancellation interrupts startup.

Chrome rendered round/practice choices without page or element overflow at
320×568, 390×844, 768×1025, 1280×720, 1920×1080, 568×320, and 844×390 CSS px.
The main choices also passed at 356×631, 434×938, 853×1138, 1422×800,
2133×1200, 631×356, and 938×434 CSS px. Actual CSS sizes were read from the DOM
because this Chrome profile uses 90% zoom. Pretext reported ready and full labels
fitting. At 320×568 with 200% root text, the main choices use deliberate vertical
document reflow with no horizontal overflow; the final Starten/Pause/Fortsetzen
screen remains bounded without overflow. Button focus persists through the
playback label changes. Physical phone/headset checks remain pending.

The final screen also passed at 1280×720, 1920×1080, and 568×320 CSS px,
including paused playback in landscape. Longer German choice labels and WCAG
text-spacing overrides at 320×568 wrap/reflow without clipped text or horizontal
overflow. Beenden was observed returning to Haupt-Runde with sessionActive=false.
These are rendered DOM/text bounds checks; browser screenshot capture timed out,
so this pass includes no saved pixel capture.

## Observed main-round normal-breathing baseline — 2026-10-12

All eight Node audio tests, the six-route UI-state checker, context/source
checker, JavaScript syntax checks and Git whitespace check passed. Main choices
are 10/15 minutes total, with five normal-breathing minutes followed by the
existing five-/ten-minute guided sequence. Tests prove that no source starts
before t0 + 300 seconds, the welcome begins at that boundary, the first cycle
starts at t0 + 340 seconds, and pause/resume preserves this same clock/plan.
Receipts distinguish total, baseline and guided duration.

Isolated headless Chromium checked the main menu and both ready/playing leaves
at 320×568, 390×844, 768×1024, 1280×720, 1920×1080, 568×320 and 844×390 CSS px.
All ordinary page/text bounds fitted, without horizontal/vertical page overflow
or an inner scrollbar. Both leaves displayed Normal atmen, had exact 600/900-second
audio spans, paused/resumed the baseline countdown, and saved the correct receipt.
At 320 CSS px, 200% root text, text-spacing overrides and a long German hint
passed full-text/no-horizontal-overflow checks using deliberate document reflow.
Restricting the two-choice width rule to bounded layout fixed the enlarged-text
overflow. Real phone/headset tests and physical acoustic timing remain pending.

Native OfflineAudioContext renders through `tests/offline-audio.html?round=haupt`
passed for both totals at 44.1 and 48 kHz: exact 600/900-second lengths, every
baseline sample zero, and all later samples matching the shifted reusable
assets with zero maximum sample error. These checks use the production scheduler.

## Observed selected ElevenLabs listening set — 2026-10-12

The user rejected the OmniVoice set and selected the existing website's British
Breathing Instructor. Saved profile name/ID and Eleven v4 German TTS support
were confirmed through the API. Only Halten was newly synthesized; website
inhale/exhale/counts and cached Pause were reused. Thirteen adjacent WAV/JSON
pairs passed independent format/frame/hash/transcript-slot checks: ten exact
four-second clips and three exact 16-second component cycles. Voice fades and
breath rises; powers agree within 0.0001 dB during inhale/exhale; all samples
have at least 3 dB headroom; component sums and concatenation match exactly.
Unprompted Whisper-small recognized all sixteen words. Runtime assets remain
unchanged. The rejected 26-pair listening set is preserved in the local archive.

## Observed Start/Pause controls — 2026-10-12

All seven audio tests, the UI-state checker, context/source checker with the
remote gate, JavaScript syntax checks, and Git whitespace check passed. The new
engine test proves that pause/resume retains the original sources and audio-clock
anchors, prevents a second start, records transitions, and leaves a cancelled
resume aborted. Controller coverage checks button visibility and enabled states.

Native browser playback showed the main start button followed by a smaller
Start/Pause row. Pause held the phase and countdown across repeated size changes;
Start resumed with identical start/end anchors and a moving countdown. Stop
removed the smaller controls and enabled a fresh main start. No document overflow
was observed at 320×568, 390×844, 768×1024, 1280×720, 1920×1080, 568×320, and
844×390 CSS px. After giving the landscape countdown five character widths,
Pretext reported every visible final-stage label fitting at 320×568 and 568×320.
At 200% root text and 320×568, all labels fitted in deliberate document reflow,
with no horizontal overflow or inner scroll container. Physical headset/VR-event
pause synchronization remains unimplemented.

## Observed listening asset packaging — 2026-10-12

The earlier package, now in `audio-assets/archived-omnivoice/`, contains 26
same-folder WAV/JSON pairs across
female/male sets: 20 four-second phase/component clips and six 16-second voice,
breath and mix cycles. Independent reading checked formats, exact frames,
transcripts/slot counts, authored timing, matching hashes, voice + breath = mix
at every sample, and exact cycle concatenation including silent breath holds.
Generated files are ignored; runtime assets remain unchanged. Folder opening
and subsequent user listening are separate from these sample checks.

## Observed counted hold variants — 2026-10-12

Two restrained voices now have voice-only and voice-and-breath auditions with
counted Einatmen/Halten/Ausatmen/Pause. Independent finished-PCM reading checked
all sixteen four-second phase WAVs (176,400 frames at 44.1 kHz) and four 16-second
montages (705,600 frames). Sample peaks have at least 3 dB headroom, boundaries
and beat tails are quiet, voice beats descend, breath beats ascend, inhale/exhale
component powers differ by less than 0.0001 dB, holds match between mix versions,
and montages concatenate their elements exactly. ZIP contents match the local
files and pass integrity checking. Runtime asset hashes still match the manifest.
Source Pause takes passed the native speech check; saved Halten/counts were reused.
Final unprompted Whisper-small review recognized all sixteen requested words for
both cycles. Preserving 30 ms before Pause's speech threshold corrected the male
cycle's initial recognition error. Listening judgment remains a separate check.
Human acceptance and clinical release remain pending.

## Observed voice-to-breath prototype — 2026-10-12

Two restrained profiles produced standalone Halten in one take each with zero
source word edits. Four phase files per voice and 16-second montages were rendered
locally. Independent reading of the finished PCM confirmed exact 176,400-frame
phases, 705,600-frame montages, correct hashes, at least 3 dB sample-peak headroom,
descending voice and ascending breath per-second RMS, component power differences
below 0.0001 dB, quiet hold tails, and sample-identical montage concatenation.
The current browser manifest and asset hashes remain unchanged. Human mix/voice
acceptance is pending; no runtime or clinical deployment is claimed.

## Observed calm German voice sourcing — 2026-10-12

The original LANG XLS workbook was read with xlrd in an isolated ignored tools
directory. All 240 recordings were screened against their auditory norms; none
met the provisional joint positive-valence ≥6/arousal ≤3 threshold after the
explicitly inferred valence-polarity reversal. Raw scores and source hashes are
retained. EmoDB's neutral speaker 13 and 15 references use confidence and
naturalness ≥0.8; numeric calm/high-valence norms are not claimed for them.

Secret Tunnel MCP calls returned internal errors; the installed app's documented
local CLI reported model/ASR ready, created two profiles, and generated identical
German welcomes. Both passed its Whisper word check in one take with zero edits.
Strict MP3 decoding passed. Matched-level review WAVs and full receipts remain
ignored/local. Model CC-BY-NC terms and unestablished speaker-clone consent mean
these checks do not establish unrestricted clinical/commercial release rights.
The user reviewed both and found them too robotic; calm/warm acceptance failed.
Positive reference auditions from female speaker 8 and male speaker 3 also
completed in one take with zero recognized word edits and strict MP3 decode
success. Review WAVs match -24 dBFS RMS and preserve native model timing. They
were rejected by the user as too energetic. Two restrained profiles then
generated welcome and connected counted-cycle speech. Four separate four-second
phase files per candidate and a 16-second review montage passed frame/hash checks;
unprompted secondary ASR recognized the entire requested sequence. Their listening
acceptance remains pending. Publisher metadata lists
CC-BY-4.0 while archive metadata lists CC0: preserve both and attribution.
See `audio/voice-sourcing.md`.

## Current soft-v2 audio checks — 2026-10-12

The current nine-asset manifest uses a 40-second welcome, four separate four-second
phases with Einatmen/Ausatmen plus counts and Und halten plus quiet, a 12-second ending, a 20-second
rest, and two four-second breath-overlay variants. Exact total 5/10/15/20-minute
protocols have 15/33/50/68 whole cycles, 0/1/2/3 rests, and 8/0/8/0 seconds recovery.
All six Node audio tests, the UI-state checker, the context checker, and Git
whitespace check pass after the quiet-hold update. The extra controller regression
test proves that cancelling a new audio unlock cannot save the preceding run's
receipt under the new scenario.

Native offline renders were repeated after final Einatmen/Ausatmen/Und halten regeneration:
every output sample matched at 44.1 kHz for all four totals and at 48 kHz for five
minutes, with zero maximum sample error and exact total frame counts.
Actual browser Play/Stop was verified for all four durations, with 300/600/900/1200
second audio-clock spans. Stop/Escape/back navigation and 20-minute double-click
protection were observed. A complete 68-second instruction example reached its
native completed/replay state. Playing phase/count/time and real Pretext fit were
observed at 320×568 CSS px with no document overflow. The current regenerated
sample set was loaded and started/stopped again after reloading. See
[PLAYBACK-AUDIT.md](PLAYBACK-AUDIT.md). Full target-device,
clinical listening, physical onset/synchronization, and final UI resize checks
remain pending. Historical v1 evidence below refers to the preserved v1 set.

## Context and source

From `atmungsraum/`, run:

```powershell
./for-ai/scripts/check-context.ps1 -ProjectRoot .
```

The current checker verifies the guidance, AI document set, four practice/two
main choices, black background, local artwork/runtime assets, and locked Pretext
version. It does not verify an H10 connection, ECG, or stretch-layout behavior.
After pushing the validated change to `main`, add `-RequireRemote` to verify that
the local repository HEAD matches the remote publishing branch.

## H10 gate and ECG acceptance

When implemented, verify the first screen is the H10 gate, including direct links
to session branches. Exercise no device, connecting, verified connection, missing
ECG data, and disconnect/data interruption. Session actions must not appear ready
before connection verification, and stale data must not appear live.

Use a real Polar H10 to verify the waveform comes from its ECG samples and is
displayed in a centered box. Test the chosen connection path on each supported
smartphone/browser. Mock data may check rendering and state transitions but cannot
qualify the sensor connection or live ECG. Record those hardware checks as pending
until observed.

## Stretch-accordion menu acceptance

Verify every menu function, including the gate and ECG view, against the installed
`uncodixfy-pretext` Pretext and accordion-stretch references. Check phone portrait
and landscape, desktop, 320 CSS px reflow, 200% text/zoom, long/localized labels,
user text-spacing overrides, focus order, visible focus, and touch controls.

Test minimum width with tall height, minimum height with wide width, both minimum,
expanded dimensions, and both sides of every breakpoint. Check header/footer
anchors, ordered control groups, centered ECG bounds, non-overlap, and growth of
group/cell spacing as space increases. Inspect panel, status, and document bounds:
the bounded menu has no internal scrolling, and a full-window panel must not make
the page scroll. Hidden overflow is not evidence of fit.

Verify real Pretext measurements with loaded matching fonts, readable no-fit and
unavailable fallbacks, and rendered DOM parity. Record the tested size/browser
range and unresolved failures; do not infer a pass from package presence.

## Observed four-choice UI tree verification — 2026-10-12

The later minimal UI replaces the anatomical images with sprout/wave symbols,
removes global context and the ECG return button, and adds measured one-sentence
exercise descriptions. Its exercise choice passed bounded page/text checks at
all seven sizes below. Narrow portrait tiles grow taller to fit the descriptions.
At 320 CSS px and 200% text, the final copy reflows without horizontal overflow
or clipped text. The new compact recording gate passed at 320×568, 390×844,
1280×720, and 568×320; refresh hides/clears ECG and changes Continue to red and
disabled, while recording makes it green and enabled.

The pure state check proves retries every five seconds, preserving the current
flow through exactly 30 seconds, expiry strictly after 30 seconds, and recovery
cancelling retries/deadline. A browser fixture also verified temporary loss stays
on the exercise screen, emits a retry, and returns to ECG only after an injected
expired deadline; Continue is then disabled and the old trace absent. These are
emulated signal-state tests; physical reconnection remains unverified. Runtime
phase/time/stop controls are shown only during playback.

The browser-local tree was checked in the Codex in-app Chromium browser at
320×568, 390×844, 768×1024, 1280×720, 1920×1080, 568×320, and 844×390 CSS px.
The connected ECG gate, two round choices, both four-box menus, and the final
play screen passed page/text bounds checks at these ordinary sizes. Header and
footer anchors remained visible; panels had no internal scrollbar or page
scroll. Both round choices stayed square and visible together.

All eight choices routed to the matching final play screen; Anleitung carries
no duration. Browser back/forward and heading focus passed. Connecting, waiting
for ECG, synthetic ECG reception, missing ECG, disconnect, and recovery were
exercised. Missing ECG/disconnect cleared the trace and set Continue disabled.
The Node self-check verifies all valid leaves, cold deep-link gating, invalid
routes, signal loss, reset, and duplicate-start protection. Run it with:

```powershell
node for-ai/scripts/check-ui.mjs
```

At 320 CSS px, long German labels and the WCAG text-spacing override passed
rendered text/page-width checks. At 200% text enlargement, readable document
reflow with one-column choices preserves full labels without horizontal overflow
or an inner scrollbar; vertical page scrolling is deliberate outside the bounded
normal-size matrix. Pretext predictions use loaded Arial and remain marked ready;
text-spacing parity is assessed in the DOM because word spacing is not modeled.
A missing Pretext module switches to measurement=unavailable and readable flow
while retaining the decision tree. No-JavaScript behavior retains the initial
unverified gate. Real phone browsers and physical H10/headset tests are pending.

A concurrent audio task connected the final play action to browser audio during
this UI pass. The initial final screen hides runtime phase/time/stop controls
until playback starts; preserve the audio task's engine and controller. The
above UI checks establish navigation/layout, not acoustic timing or real headset
execution. See the audio-specific verification below for that separate scope.

## Earlier radio-menu verification — 2026-10-12

Windows desktop Chrome rendered the start at 320×568, 390×844, 768×1024,
1280×720, 1920×1080, 568×320, and 844×390 CSS px. Both choices remained square
and visible together with no horizontal page overflow or clipped text. Probe
branch text bounds were checked at 320×568, 390×844, 1280×720, and 568×320;
the main branch and every leaf choice were exercised at 390×844.

At 320 CSS px, 200% text enlargement, text-spacing overrides, and a longer
German start label passed rendered text-bound/page-width checks. Both branches
also passed the 200% text page-width check. Text enlargement and short landscape
branch views use vertical document flow where needed; full bounded/no-page-scroll
stretch acceptance remains pending. No inner scroll container or hidden overflow
is used to conceal text.

Round selection, every probe/main radio option, confirmation output, Zurück,
browser history, heading focus, keyboard radio selection, JavaScript-disabled
navigation/selection, and the Pretext-unavailable fallback passed. Runtime
measurement reached `ready` using the locally shipped package; ordinary checks
reported no JavaScript errors. Reduced-motion preference was also exercised.
Source/context and Git whitespace checks passed. No publication or H10/headset
hardware checks are part of this round-menu validation.

## Observed audio cue verification — 2026-10-12

All five `assets/audio/breathwork/de/*.mp3` files fully decoded with FFmpeg.
FFprobe confirmed MP3 audio at 44.1 kHz with positive durations; the combined
duration is 13.885 seconds and the combined file size is 227,587 bytes. Every
file matched its recorded byte count and SHA-256 hash. Measured peak levels
ranged from -2.5 to -1.8 dBFS; each file contained non-silent audio.

The exact German texts and generation settings are recorded in the adjacent
`cues.json`. Pronunciation, accent, and subjective voice quality have not been
listened to and remain pending. No playback scheduling, UI integration,
publication, or headset audio check is part of this asset verification.

## Observed reusable box-breathing verification — 2026-10-12

Both supplied YouTube reference tracks were downloaded as AAC/M4A and decoded
with FFmpeg strict error checking. Native German/English captions were extracted
and analyzed locally; these caption timings are not acoustic onset measurements.

All six reusable WAVs have exactly 176,400 mono 16-bit PCM frames at 44.1 kHz,
four seconds each. Hashes and byte counts matched the asset manifest; each file
contains speech with silent boundaries and sample peaks below full scale.
Five Node tests passed: exact complete-cycle protocols/remainders, actual sample
integrity, common-clock scheduling, cancellation/interrupted contexts/stale
callbacks, and corrupt-asset rejection.

The production scheduler was rendered with native Chromium OfflineAudioContext.
Every output sample matched the expected repeated blocks with zero maximum
difference for 5/10/15 minutes at 44.1 kHz and 5 minutes at 48 kHz. Full PCM
verification exports also matched their sample-indexed protocols, including
silent recovery, at exactly 300, 600, and 900 seconds. Runtime playback reuses
the six samples; concatenated exports are local verification artifacts only.

Local Whisper transcription recognized the four phase prompts and ending;
the compact instruction's short words were joined by ASR. Human listening remains
pending. No realtime phone/headset acoustic timing, H10 synchronization, host
receipt persistence, menu integration, or publication is claimed by these checks.

## Deployment

Prepublication check on 2026-10-12: the UI-state check, all six audio tests,
context/source checker, and Git whitespace check passed. GitHub Pages source was
confirmed as the existing `main` branch, root `/`, with `viscereality.org`.
The user authorized publishing this navigation prototype and browser audio.
Physical H10/headset acquisition and the VR event sequence remain pending.

- Verify the GitHub Pages build reports `built` for the published commit, without
  a build error.
- Request both `/atmungsraum` and `/atmungsraum/`; a redirect from the first to the
  second is normal for a directory index.
- Confirm the final response is HTTP 200 and contains the expected page source,
  then inspect the public page in a browser.
- Verify the deployed `atmungsraum/for-ai/README.md` is reachable.

Report failed or pending checks explicitly; do not infer deployment from a push.

## Observed local verification — 2026-10-09

The context/source checker and Git whitespace check passed. Browser checks at
1280 × 720 and 320 × 568 CSS px showed an empty body with zero child elements,
`rgb(0, 0, 0)` as the root background, and no horizontal or vertical overflow.
Both rendered views were visually solid black. Deployment must still be checked
against the published commit using the procedure above.

## Compact Update presentation — 2026-11-17

Real mounted UI with inert transport passes independent review:297 Node tests
without skips,135 layout states,26 compact Update renders,27 Meta cases and18
conditional-APK renders/15cases. The original strict320px/200% scroll/focus checks
remain intact. Current status and check action are concise; full identity and
provider diagnostics remain in collapsed Details. Evidence is retained privately
in the2026-11-17-update-minimal-review validation folder. Runtime marker:
20261117-update-simple. PR45 merged36d9216 as eedbdf72; exact Pages44091657428
succeeded and all20 checked public files match Git. A fresh public-runtime
Chromium probe confirms compact current status, closed Details, automatic
control and three method switches with no page errors (transport synthetic).
Installed-phone refresh remains pending because Pixel6a is absent from USB.

## Current Polar health and battery review candidate — 2026-11-17

Native0c82ff/tree2444f55f exports current-generation link, fresh ECG/ACC,
HR-service contact flags and valid0..100 battery values. Enrolled LAN receives
power/health in idle and active phases; guest, relay and compact BLE strip the
new fields including nested health. Quest/controller reads use public Android
APIs with five-second caching and explicit unknown values. Polar battery uses
the existing serialized GATT owner with exact callback/generation checks; a lost
read callback retires only its original transport through existing recovery.
No admission, classifier, recording or automatic battery-abort changes.

Independent host gates pass: Polar47, serialized battery22, authenticated
power/privacy89, full WiFi/BLE projection and real Android34 compile. Actual
signed24-r2 passes1106 package/source/signature/SDK/private-asset checks,54
actual Java codec/trusted-closure controls, actual signed DEX data paths and the
explicit Java-to-native parser contract (one pass, zero ignored). The APK is
65,694,854bytes, SHA2567b8709dd669aa4a33eb689f52456b9bf3525133e7bbf16491514703a1225105c.
The2139506-byte cumulative source patch reconstructs exact tree2444f55f.

PWA health is independent of old initial readiness and pi.f. It expires current
link/ECG/ACC/contact observations after five seconds and Polar battery after
120seconds, with owner/generation/witness aging. Battery review is local and
bound to participant/preparation; absent controller battery does not block.
Existing active/recovery controls and native Arm/Start gates are preserved.
Independent review passes22 targeted unit tests and six actual Chromium/Pretext
battery renders at390/320/320px with200% text. The full Node suite passes305/305
with zero skips. The full actual-Pretext layout matrix passes140 states, plus
operator flow, selection races,24 input renders and26 compact Update renders.
Source/cache graph review has no findings. [PR46](https://github.com/GeorgeFejer91/viscereality.org/pull/46)
implementation and [PR47](https://github.com/GeorgeFejer91/viscereality.org/pull/47)
policy merged as166f483386d70e1aa56b53d3d88349bf00f2d468. Exact
[Pages44097236831](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44097236831)
succeeded;21 live-file comparisons pass. PR46's earlier run was superseded and
cancelled; the successful final deployment includes both increments. The public
runtime probe exits0 with the correct cache graph, floor/latest24 and all focused
health/battery routes/expiry checks. Only transport is synthetic; this is not
new hardware acceptance. Meta24 build1464488216750190 is current and malware/
detailed security passed, Require Latest remains unchecked.

Actual controller battery availability, off-body contact behavior and installed
phone refresh require hardware acceptance. Pixel6a is absent from USB at the
current readback. The active-session update test remains open; no recording has
started from the rejected23 Arm. Version25 remains unpublished until an actual
repaired24 RUNNING/rec:true observation. Its version-only private package passes
1100 checks with102 ZIP entries; SHA256
a8a2de290680fcfa76820839ae848c8886406dfd6b5ae559599dd5a3e78b208b.

The user explicitly deferred attended testing. Actual headset readback remains
Meta23;24 was not sideloaded or host-launched. Final documentation publication
retains the above unchanged runtime proof. Phone refresh, actual controller
battery availability, H10 skin-contact behavior and running-session update remain
open in PROJECT.md.

Meta24 publication: build1464488216750190 is current on the private Developer
channel, malware passed and detailed security All tests passed. Quest2/13+ and
Require Latest unchecked are retained. This is provider availability evidence,
not actual headset installation. Public policy advances to24 only now. The held
version-only25 passes1100 independent signed-package checks, unchanged source,
DEX/SDK/helper/private payloads; it remains unpublished until real24 recording.
