# Project

### Active: reconcile latest source and pending updater repair — 2026-11-19

The user requests synchronization of George's current source and reconciliation
of the four unfinished updater files before attended testing. Reuse the lab
validation goal below. Preserve Meta code 27, helper 3, held successor 28 and
the original dirty native checkout while qualifying this source increment.

- [x] Sync the clean website worktree to main `c0569ab`; preserve the previous branch.
- [x] Reconstruct Developer 30 from fixed27-r2 plus `webrtc-native.patch` and
  match exact native tree `da94a0c2c7b1acb60fa734fb9ca5cb4c0857c485`.
- [x] Reproduce the ambiguous duplicate-download target failure against that
  source with the preserved regression before changing production code.
- [x] Port the pending coordinator, duplicate/service tests and documentation;
  targeted checks pass 162 duplicate and 63 native-cycle cases.
- [x] Complete updater, WebRTC compatibility and PWA regression checks; review.
  All16 updater suites, actual production javac,11-source closure and gateway/
  pairing controls pass. All361 web/assembly tests pass with zero skips and
  real WebCrypto-to-Java fixtures. Independent review has no remaining findings.
- [x] Commit and publish a separate repair overlay with exact reconstruction
  evidence; retain the immutable Developer 30 and held 28 records.
- [x] Prepare the qualified source and signed Developer31 candidate for the
  next attended hardware check. Signature/alignment, unchanged74 audio and
  three loaders,10 WebRTC assets and actual900-second signed session pass.
  No new APK installation, Meta publication or policy increase is established
  by the source reconciliation itself.

Reconciled native commit `a7289a89d0ab092a0d5a9a89a6ef67f7cb36ff57`, tree
`310ee8870fce0de0108cc05891abdd309d976b09`. The separate
[`updater-reconciliation.patch`](../quest/updater-reconciliation.patch) applies
after the unchanged WebRTC overlay. Its23,933bytes/SHA256
`91d9ff51790246392eec5f03c98d9fbfb7123a99e0549defc5bd0bcbed5c2302`
reconstruct that exact tree using an isolated Git index. Its only production
delta removes unsafe adoption of the offered version from targetless SDK207
duplicate progress; bounded cancellation, fresh version and owned completion
are required with helper3 as well. No helper binary changed. Source PR151 merged
as `e54de249`; exact Pages44525274968 succeeded and the live patch hash matches.
The signed [Developer31 candidate and this-PC bench38](../quest/apks/README.md)
are built locally.31 SHA256
`3f318be948cd3cee12570f56ed898d939b38b30fc5c14dfd452d0d98dd9589c6`.
No31 install or Meta publication is claimed. Private evidence is retained outside
Git. The original native checkout still holds all four unfinished files unchanged.

- [x] Complete the attended bench38 pairing/native-receipt check.
  Pixel6a loaded the current standalone release and controlled native27 at
  fresh `IDLE/rec:false/pending:false`. Bench38 installed with matching bytes
  and reached its actual Android first-run permission prompt. After a true
  process cold start, saved membership rejoined and recenter received a native
  confirmation over authenticated LAN. No experiment or Meta update has started.
- [ ] After the startup follow-up, restore the known Meta27 launch and the
  operator's saved connection. Preserve the existing27-to-held28 acceptance gate.

Attended follow-up: the two-sided826276 code exchange saved enrollment. The
first permission flow exceeded the kiosk's60-second prompt suppression;
three own-surface departures selected terminal exit while startup permission
delivery was still pending. There was no crash. A warm Activity relaunch kept
a retired Polar runtime; a true bench-process cold start recovered it. The
phone automatically rejoined the saved bench membership, fresh native38 status
confirmed `IDLE/rec:false/pending:false`, and the real recenter control reported
native confirmation. The observed route was authenticated LAN HTTP, not a
qualified WebRTC relay. Keep first-run and warm-relaunch issues open.

Bounded next repair, inside this same validation goal:

- [x] Add a RED permission/kiosk regression: an actual pending permission or
  Location prompt for the exact current native Activity stays protected beyond
 60seconds; another, retired or finishing Activity cannot hold that protection.
- [x] Expose the startup owner's actual pending-prompt observation and use it
  in the own-app kiosk observer before departure/escape counting. Preserve the
  existing bounded prompt API for other callers, default-off compositions and
  normal triple-escape once the actual prompt finishes. No timer proves completion.
- [x] Run startup-permission, kiosk and actual production compilation
  checks; independent review.94 lifecycle,42 actual observer/watchdog and14
  optional-class controls pass, including120seconds and subsequent ordinary
  escape. Kiosk policy/static and production javac pass. A fresh updater-suite
  check found an older generated kiosk stub missing the new observer method;
  the test-only correction restores all16 passing updater suites.
- [x] Build signed Developer32 and bench39 packages from the reviewed source;
  actual900-second session closure, v3 signatures,16KiB alignment, source
  capsules,74 unchanged audio entries,10 WebRTC assets and retained loaders pass.
- [ ] Commit/publish the separate startup repair overlay. No Meta publication yet.
- [ ] Leave a real bench39 startup permission prompt pending beyond60seconds,
  then verify actual foreground recovery after the wearer's response.

Startup native commit `fe22d7d402d2edfa912412777720685c571b4f5b`, tree
`a99d4eb89b05b170e89a10bfb45f11044807ecbb`, follows the updater repair.
Its [`startup-permission-kiosk.patch`](../quest/startup-permission-kiosk.patch)
is 30,378 bytes/SHA256
`d432a1435e0c114e45f856cd927b097733a78ed79450ea42f321706561caf70b`.
Local Developer32/bench39 packaging is complete. Exact package records are in
the [APK log](../quest/apks/README.md); neither is Meta-published. No helper,
required-version or PWA-runtime change is implied. Preserve held28 and installed27.

The separately applicable [host fixture correction](../quest/startup-kiosk-host-fixture.patch)
is 1,287 bytes/SHA256
`e81ec8b79788cec8767a8dc13d0eb10383db0b48ab20120ad7d2ec0867541321`.
It reconstructs native `30cde1d1aff280a9a6d6f79576df774e506616bc`, tree
`678e7fe27274353344feaab9992196967d0e2eb6`; production source/docs remain
byte-identical to the signed `fe22d7d4` builds. All16 Meta suites pass; the
install-handoff12, startup observation42+14, existing kiosk and terminal pending-arm27
controls pass. Root independently reran the actual12-case handoff harness and
confirmed no production delta. The prior patch still reconstructs the exact
immutable signed-build tree.

### Active: full lab validation and UI repairs — 2026-11-18

The Codex validation goal uses this checklist. Repeat affected checks on the
latest applicable release, and fix and retest any confirmed UI failure against
the current GUI rules and original input driver.

For the hotel network blocker and the exact restart checks, see the
[hotel Wi-Fi handoff](HOTEL-WIFI-HANDOFF.md).

The user's 2026-11-18 portability requirement applies to every independent
installation of the Atmungsraum PWA and other compatible headsets and phones
of the same classes as the lab devices. The user selects the intended headset
through Bluetooth; that connection supplies its installation identity and
bootstrap information for authenticated Wi-Fi/WebRTC correspondence.
Subsequent saved connections reconnect automatically. Initial platform
permissions remain part of setup. Fresh pairing uses the displayed digits
and two confirmation buttons, with no code entry or copying, hotspot,
lab-specific credentials or per-device APK. One canonical signed headset APK
serves compatible headsets; each installation has its own identity and trust.
Quest 2/Pixel 10 is the initial physical acceptance pair. General compatibility
requires the [independent-installation matrix](VERIFICATION.md#independent-headset-and-pwa-installations).

Current repair: first pairing through Bluetooth-selected WebRTC. The user's
latest clarification permits confirmation buttons and requires no manual
code entry. Preserve the existing proof-bound v2 pairing and both real device
confirmations; saved reconnect remains automatic. The mounted production UI
reproduces a failure when Bluetooth supplies a valid RTC offer with no LAN
endpoint: the fresh-phone path stops before opening WebRTC.

- [x] Reproduce the RTC-only fresh-join failure and run the current real
  browser/Java pairing controls (10 tests, zero skips).
- [x] Add red tests and a bounded native WebRTC pairing frame, using the
  existing `ExperimentPhonePairing` owner without experiment dispatch or
  approval bypass. Publish the selected pairing offer before the first member.
- [x] Route explicit fresh-phone Bluetooth selection through that exchange
  in `quest-webrtc.js` and `quest-ui.js`; preserve LAN fallback, saved target
  identity, cancellation and membership persistence. Test service ordering.
- [x] Run actual Java/browser interop, native gateway controls, the full web
  suite and GUI gate; review the change before publishing the source overlay.
  Passed six native/browser pairing cases (including member status and
  reconnect), eight mounted pairing UI cases, 361 web/assembly tests with
  zero skips, and all 421 smartphone GUI states. Review regressions covered
  carrier cleanup and stale LAN addresses. The overlay reconstructs native
  tree `da94a0c2c7b1acb60fa734fb9ca5cb4c0857c485` from fixed27-r2.
- [x] Publish `20261119-bluetooth-buttons`. Merge `efe4c3e` passed the
  [exact Pages run](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44424045321).
  All 24 changed live files matched Git bytes, and the remote context gate
  passed.
- [x] Reproduce a missed idle PWA release check in the real app when its
  five-minute timer overlaps saved-headset discovery. Queue the due check
  for the next idle Connect window before another search. A Chromium
  regression verifies one update and exact saved membership retention.
- [x] Publish `20261119-idle-refresh`. Merge `759beb7` passed the
  [exact Pages deployment](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44427512855),
  all six live file comparisons, and the remote context gate. The target
  Chromium startup suite, eight pairing UI cases, 361 web and assembly
  checks, and 421 smartphone GUI states passed.
- [x] Verify that the installed Pixel has refreshed to
  `20261119-idle-refresh`. The first standalone readback after deployment
  still showed `20261118-saved-quest-service` and its saved pairing.
  On 2026-11-19 the RustDesk viewer dismissed the idle PWA task in Android
  Recents. A RustDesk MCP tap on its launcher icon opened Chrome's
  `ERR_ADDRESS_UNREACHABLE` page, and its Reload button returned the same
  error. The phone's IPv4 connection to the live host on port 443 failed
  with `No route to host`; the workstation received HTTP 200 with the
  current release. Rejoining the same hotel Wi-Fi exposed the Pixel sign-in
  prompt. The operator approved completing it through RustDesk. The Pixel then
  reached the site and signaling host on TCP 443, and the existing standalone
  app loaded the current entry and connection UI module.
- [x] Verify automatic refresh and reinstall on the Pixel over the restored
  hotel Wi-Fi. Bringing the existing Chrome tab forward through RustDesk
  advanced that tab from `20261118-idle-release` to `20261119-idle-refresh`
  without a Reload tap. RustDesk removed the old Android WebAPK and installed
  Atmungsraum again through Chrome. A launcher tap opened the new WebAPK in
  standalone mode. DevTools read back entry `20261119-idle-refresh`,
  `quest-ui.js?v=20261119-bluetooth-buttons`, and unchanged saved pairing.
  [Recovery and evidence](HOTEL-WIFI-HANDOFF.md#pixel-recovery-and-reinstall).
- [x] Build and inspect signed bench 37 and the selected canonical Developer
  candidate 30. Match the bench 37 installed SHA-256 and retained UID. Preserve
  the installed Meta 27 bytes and process and the staged code 28 update.
  See the [signed APK records](../quest/apks/README.md).
- [ ] Pair on the latest installed PWA and bench 37 using RustDesk and
  serial-scoped ADB where supported. A bounded ADB wake brought the Quest
  up briefly, then it returned to sleep. Resume with an awake headset and
  the operator available for the real headset confirmation. Match a native
  status and Recenter receipt, then test automatic reconnect in both startup
  orders and measure the ICE route. Cross-device and worn-H10 evidence remains open.
  The initial lab readback marked the Quest's active Wi-Fi captive, with
  signaling TCP 443 failing on both devices. The Pixel recovered after its
  own hotel sign-in. Recheck the Quest's portal state and both devices'
  application egress before attempting pairing or attributing a relay
  failure. The Pixel's saved pairing remains available for the reconnect case.

Use the Meta Quest workflow and exact-serial ADB to inspect, launch and
navigate the Quest autonomously where the platform provides a supported
route. Prefer RustDesk MCP for the Pixel UI and use authorized ADB recovery
or navigation when needed. Confirm each result from the next frame.
Check and restore the existing authorized remote controls
when capture or input drops. Limit wearer assistance to physical actions,
protected headset prompts, real Touch/tracking, and Polar contact.
Autonomously build, install and verify each in-scope improvement, exercise
the relevant stress and recovery cases, and repair and rerun against the
latest applicable versions until the acceptance criteria pass.
Follow [the standing device validation rules](VERIFICATION.md#autonomous-device-validation).

- [x] Confirm the lab Quest APK matches Meta fixed27-r2 and Boot Helper 3 is
  installed. Match the live smartphone-rules PWA and policy bytes to Git.
- [x] Authenticate RustDeskMCP to the lab Pixel from the operator-owned
  permanent credential file. Receive a fresh Android frame. Record the
  [reusable procedure](RUSTDESK-PHONE-TESTING.md) without the credential.
- [x] Launch the installed PWA through RustDeskMCP and open Settings with a
  frame-confirmed tap. The Android RustDesk Settings tap also confirms input.
- [x] Confirm the latest deployed PWA loads in standalone mode on the lab
  Pixel. Verify RustDesk Settings/Back, Bluetooth chooser, and diagnostic
  open/close with physical phone frames.
- [x] On signed WebRTC Bench code 31, the enrolled Pixel automatically rejoined
  the same Quest 2 without a code or chooser. RustDesk MCP verified the measured
  APK Details build identity and the original native Recenter receipt.
- [x] Reproduce the missing APK identity measurement at 390 px, schedule its
  measurement on the visibility change and pass the GUI gate and original case.
  See [verification](VERIFICATION.md#apk-detail-measurement--2026-11-18).
- [x] Bundle the [connectivity and release protocol](../quest/webrtc/release-protocol.md)
  in the signed WebRTC build. Verify exact signed and installed bytes for
  [bench code 35](../quest/apks/README.md#webrtc-bench-with-bundled-connectivity-and-release-protocol).
- [x] Verify the exact signed bench 35 starts and reaches native Prepare on
  the lab Quest 2. Both native receipts reported valid tracking and the frame
  advanced from 223 to 313. Restore the same Meta code 27 process and the
  Pixel standalone PWA with its saved pairing still present.
  [Startup evidence](VERIFICATION.md#signed-webrtc-bench-35-startup--2026-11-18).
- [x] Correct false LAN endpoints in Bluetooth discovery while the gateway
  has a contended listener. The host regression reproduced the advertised
  endpoint before the fix and now verifies an empty LAN list, an available
  trusted WebRTC offer, and restoration of LAN addresses after the bind.
  Signed [bench code 36](../quest/apks/README.md#webrtc-bench-with-listener-aware-bluetooth-discovery)
  matches installed bytes and reached Prepare with advancing native frames.
  The same Meta code 27 process and the Pixel's original pairing were restored.
  [Verification](VERIFICATION.md#listener-aware-bluetooth-discovery--2026-11-18).
- [x] Read the actual Bluetooth discovery payloads from the lab Quest while
  signed bench 36 was in Prepare. One Bluetooth device returned both service
  instances. The known Meta baseline supplied one LAN endpoint. The bench
  supplied zero LAN endpoints and a well-formed WebRTC offer. Windows radio
  state, the original code 27 process, and the Pixel PWA were restored.
  Authenticated phone pairing and physical listener recovery remain open.
- [x] Repair the PWA remembered-Bluetooth path for the two service
  instances observed on the lab Quest. It now checks matching service
  instances against the saved installation ID, so a preceding foreign
  service cannot hide the saved Quest. The mounted startup and focused
  gateway tests passed.
- [x] Load the published `20261118-saved-quest-service` PWA on the lab
  Pixel. [Pages run 44353740380](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44353740380)
  deployed merge `3cdea7b`. Eighteen live files matched Git. The GUI gate passed
  all 421 states. RustDesk opened Settings and returned to Connect. A
  cold launch retained saved pairing and reached Connect in 1.23 seconds.
  It stayed unauthenticated during the 30-second observation. The saved
  enrollment still points to a different installation and Chrome exposed
  no matching grant. Qualify authenticated reconnect with the approved
  phone and headset enrollment.
- [x] Recheck startup on the current Meta Developer release and installed
  PWA. The dashboard, live release policy and exact Quest APK agree on
  `0.2.26-dev` (code 27). The standalone Pixel PWA matches the current
  Pages deployment. Three cold starts opened Connect in 1.34–1.38 seconds.
  None authenticated during the 30-second observations. Signed bench 36
  has the WebRTC runtime and Meta code 27 lacks it. The saved Pixel
  enrollment identifies a different Quest installation.
  [Startup evidence](VERIFICATION.md#meta-and-standalone-startup-recheck--2026-11-18).
- [x] Compare both installed Android companions against the same live Quest
  Bluetooth advertisements. Standard code 1 owns loopback 8767 and found
  Meta code 27 in 2.55 seconds. It still searched for bench 36 after
  22.58 seconds. Signed bench companion code 4 owns lab loopback 8768 and
  found bench 36 in 3.48 seconds with the commissioned identity, zero LAN
  endpoints and a WebRTC offer.
  [Physical scanner evidence](VERIFICATION.md#pixel-companion-discovery-comparison--2026-11-18).
- [ ] Update the standard Android companion in place with the retained
  signer and current scanner. Replay the real PWA startup with the
  matching authorized enrollment, and match native status and a receipt.
- [x] Recover the five exact provider assets and the ARM64 OpenXR loader
  from signed fixed27-r2 into private release staging. Match every file
  against the signed APK and its recorded input bytes. Confirm the Meta
  SDK loader hash and Developer signer on fixed27-r2.
  [Build input evidence](VERIFICATION.md#canonical-developer-build-inputs--2026-11-18).
- [x] Retrieve the supplied Developer keystore from the QR-linked transfer.
  The certificate matches signed Meta fixed27-r2, and a local RSA
  signing and verification challenge passed. The keystore and transfer
  receipts are in private staging outside Git.
  [Signing evidence](VERIFICATION.md#canonical-developer-build-inputs--2026-11-18).
- [x] Reconstruct the private Experimenter inputs from signed fixed27-r2.
  Recover the exact visual profile and four particle arrays, and verify all
  three compiled shaders against the signed native library. The fresh
  resolver preserves feature selection, Android manifest, runtime settings
  and the five-asset content inventory. Verify the licensed Meta SDK 207
  archive, headers and ARM64 loader.
- [x] Build the previous Developer code 29 / `0.2.28-dev` candidate with the
  retained production signer, then-current WebRTC overlay, SDK 207 updater and bundled
  release protocol. Verify the signature, ARM64 alignment, signed session
  closure, original visual bytes and all ten WebRTC assets. See the
  [candidate record](../quest/apks/README.md#canonical-developer-29-candidate).
  Independent lab-controller inspection matched that exact candidate digest,
  all 74 original audio entries, five provider assets, five particle inputs,
  three compiled shaders, SDK closure and packaged 900-second session.
- [x] Build the held Developer code 28 / `0.2.27-dev` successor from the
  recorded fixed27 source tree and recovered baseline inputs. Verify the
  production signature, ARM64 alignment, packaged session closure, original
  assets and native particle/shader bytes. Full DEX disassembly differs only
  in the relocated provider-manifest digest and string-table references.
  [Held candidate](../quest/apks/README.md#held-developer-28-update-test-candidate).
  Keep it unpublished until the recorded baseline/update hold is cleared.
  Independent lab-controller inspection confirmed its exact digest,
  Developer v3 signature, alignment and actual 900-second session closure.
  Eighty-seven assets, including the embedded runtime configuration, and
  the original OpenXR/Meta libraries match the baseline. Physical update and
  cold-start recovery remain deferred and unverified.
- [x] Repeat ten Settings/Back cycles through RustDesk MCP on the current
  standalone Pixel PWA. All twenty transitions returned the expected page,
  ready measurement and zero page overflow at 411 × 833 CSS pixels. Saved
  pairing remained present. The app remained disconnected. See the
  [physical navigation evidence](VERIFICATION.md#physical-pixel-navigation-stress--2026-11-18).
- [ ] Verify automatic startup in both launch orders on the qualified
  unified Developer APK and a Pixel commissioned for that installation.
  Use the retained Developer signer and approved Meta Platform SDK 207
  from the release build environment. Follow the held Meta update
  sequence and match a native receipt on both Wi-Fi routes.
- [ ] Qualify fresh Atmungsraum PWA installations with other compatible
  headsets and smartphones. Bluetooth selection must establish the intended
  authenticated Wi-Fi/WebRTC session with button confirmations and no manual
  code entry or copied lab pairing state. Test independent identities, saved reconnect, headset
  switching, protected control leases and rejected wrong/revoked peers.
  The current commissioned-Pixel bench result does not complete this gate.
- [ ] Qualify physical Touch, Polar contact and tracking, one clean 900-second
  baseline run and one fault recovery/partial run. Verify native, phone and
  private Drive custody with matching file hashes.
- [ ] Coordinate the held successor after the actual lab baseline reports
  recording. Verify preparation check, PWA-driven save/update, trusted
  relaunch, and a second clean 900-second run with equivalent custody.
- [ ] Qualify one canonical Quest Developer APK with automatic WebRTC
  pairing, Polar recording and the Meta updater together. Record a single
  deliverable version, source tree, signing certificate and APK digest.
  Follow the approved Meta release sequence and verify installed bytes.

The [Meta APK synchronization procedure](WORKFLOW.md#quest-apk-changes-and-releases)
is part of every qualified Developer release. The authenticated Viscereality
dashboard shows private Developer serving `0.2.26-dev` (code 27), build
`1465199570012388`, with Quest 2 verified in the build list. This matches the
live version policy. The next Meta change is the staged successor after the
app reports recording on the lab baseline and the release hold is cleared.

The signed fixed27-r2 APK does not contain the WebRTC bridge or assets.
Signed bench 35 embeds the Bluetooth, Wi-Fi, TURN and WebRTC instructions and
the required qualification and Meta sync steps. Its WebRTC runtime and DEX
match bench 34 byte for byte. Eight assembly checks, 26 browser and native
interop checks, the production gateway host suite, and five companion scan
checks passed. The canonical assembly rejects a foreign Meta app ID and
verifies the staged protocol bytes. Canonical Developer code 30 supersedes
29 with the Bluetooth-selected WebRTC pairing repair. It uses the recovered
private inputs and retained Developer key.
Continue through the recorded code 27-to-28 update sequence and qualify the
unified candidate on the lab Quest, Pixel and companion before assigning its
Meta release. The physical and Meta publication checks remain open.

Both signed successor APKs are now prepared in private release staging.
Code 28 preserves the baseline transport feature set for the genuine Meta
update test; code 30 carries the unified WebRTC runtime and bundled protocol.
The user explicitly chose **Continue preparation, test later** on 2026-11-18.
Keep physical acceptance deferred and continue independent preparation.
Before the next physical session, confirm fresh wearer readiness with the
Quest, Touch controllers and Polar H10, then resume the clean recorded code 27
baseline and held update sequence. No new calibration or collection was
started. Do not replace the Meta baseline with an ADB-installed successor.

Updating the standard Android companion already installed on the lab Pixel
in place still requires that installation's original signing key.
Its installed certificate is
`5d6502013bba4f61acfb82a008bda7420bcff8c8920311bd34d07e1a618cb0e1`.
The available local Android debug key has certificate
`51a7052801899e071b728ae7e4dbd1fba0507f0f8c6ffdb7c6754fc1eb65eedb`.
It cannot sign an in-place update to that installation. Prepare the current
scanner with the original companion key, then verify its installed bytes,
retained UID/grants and matching Quest enrollment for unified acceptance.
This is a prerequisite for that legacy installation's update. General
compatibility must use a reusable production build and supported enrollment for
each new phone/PWA installation. It must not depend on the lab Pixel's key,
saved browser storage or a synthetic commissioned fixture.

The signed code 29 input/build record is published through
[PR 138](https://github.com/GeorgeFejer91/viscereality.org/pull/138), merge
`13e0e042dae0d7fde92510b708fc6c99e73a87e9`, with successful
[Pages run 44372630949](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44372630949)
and exact live comparisons of all three changed documents.
The held code 28 evidence was published through
[PR 140](https://github.com/GeorgeFejer91/viscereality.org/pull/140), merge
`a145a05d2cf318d44bff07a8a8614c7730cca59e`. Its exact
[Pages run 44376330876](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44376330876)
succeeded and all three live documents matched Git. The preparation handoff
uses branch `codex/atmungsraum-test-deferral` and
[PR 141](https://github.com/GeorgeFejer91/viscereality.org/pull/141).
The context checker and whitespace review pass. Follow the handoff's exact
merge deployment and compare the live notes before ending; the runtime
release policy remains at the verified Meta baseline.

[PR 113](https://github.com/GeorgeFejer91/viscereality.org/pull/113) makes the
autonomous install, stress, repair and retest loop part of the instructions
agents read at project entry. The exact
[Pages run](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44309823231)
passed and all three changed documents matched the live Markdown bytes.
The smartphone GUI gate passed 399 states. Fourteen targeted WebRTC and
recording sync checks passed, with one compiled Java gateway fixture skipped.

The signed WebRTC Bench now includes the durable AudioTrack completion
receipt. All 909 executed native host tests passed, including the 900-second
fixed and adaptive Polar scheduler and the finalization round trip. Six
native tests were intentionally ignored. All 14 evidence verifier tests
passed, including missing and wrong frame receipts. The retained synthetic
native partial export still verifies. All 14 Drive fixture checks passed,
including upload interruption and resume from the acknowledged byte. The lab
Quest accepted the same-signer Bench code 34 update with matching installed
bytes, retained UID and grants. The normal launcher reached Prepare with a
native submitted frame. The bench was stopped and Meta Developer code 27 was
restored to Prepare with a native submitted frame. Pixel input has been
restored. Verify the standalone release, then run code-free pairing and
the worn Polar and export checks. See the
[APK record](../quest/apks/README.md#webrtc-bench-with-durable-audiotrack-completion-evidence).

The PWA checks for updates every five minutes on the visible, idle Connect
screen. [PR 116](https://github.com/GeorgeFejer91/viscereality.org/pull/116)
and [Pages run 44316222303](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44316222303)
published `20261118-idle-release`. Eleven live files matched Git. The
mounted regression preserved Settings and origin storage, and live Chromium
checked the current release without reloading. RustDesk Input was already
enabled when the lab Pixel was rechecked. RustDesk Settings and Back worked.
Serial-scoped ADB backgrounded the old app and relaunched its exact launcher.
The installed standalone app loaded `20261118-idle-release` with its saved
pairing intact. ADB Settings and Back also passed with fresh RustDesk frames.
The available standard Bluetooth companion
0.2.1 APK and the installed 0.1.0 APK have different certificates. An
in-place upgrade requires a build signed with the installed certificate.

The Meta-installed fixed27-r2 APK reached OpenXR focus on the Prepare page.
After an isolated cold launch its trusted LAN gateway replied over the
temporary USB route. The Pixel's ordinary saved pairing is a legacy record
for a different installation, and the official gateway reports no enabled
enrolled controller. The Pixel stopped on the identity check. The official
Prepare runtime and listener were restored after the bench test.
The current standalone PWA still holds a legacy enrollment for a different
installation. Both devices now report the same Wi-Fi subnet. The Pixel's
direct TCP check to the Quest gateway failed, and the exact Quest gateway
answered over the temporary USB loopback route.

- [x] Recheck Meta Developer and the installed PWA together. Meta showed
  complete Quest 2 build `1465199570012388`, `0.2.26-dev` (code 27).
  The Meta-installed APK matched the signed SHA-256 and live policy.
  Exact Pages run `44337383494` at `31b3822` matched eight paths. The
  installed Pixel ran `20261118-recordings-offline-landscape`.
- [ ] Pass automatic startup on the current Developer APK and installed
  PWA. Three cold Pixel launches created fresh standalone documents.
  None authenticated within 30 seconds while code 27 was awake,
  advertising and listening. The Pixel kept a saved enrollment for a
  different installation. Code 27 has no bundled WebRTC runtime, and
  the phone could not reach its LAN gateway on the shared Wi-Fi subnet.
  Continue with the commissioned, unified Developer build and the
  [connection milestone](#connection-milestone--fast-web-pairing-across-isolated-wi-fi).
  Repeat the startup gate on the approved Meta release and installed
  counterpart. The existing guarded successor sequence still applies.

On the exact Meta-installed code 27, the signed app's typed ADB operator
showed the panel with a native submitted-frame receipt and confirmed six
Controls and Condition navigation transitions. The same native owner and
process returned to Prepare with the panel closed. Physical Touch and
wearer input still need their own acceptance pass.

A controlled port reservation reproduced `listener-unavailable` on official
code 27. Releasing the port left that process unavailable for the full
12-second observation. The composed native fix passed the same fault on
signed bench code 33, authenticating the Pixel after the same owner rebound
within 0.93 seconds. A RustDesk Recenter matched its native receipt over
ordinary WebRTC. See [listener recovery](VERIFICATION.md#quest-gateway-listener-recovery--2026-11-18).
Next, establish authorized membership for the official installation and read
the signed Meta offer before any preparation action. The held successor,
canonical signing and wearer gates remain open.

The PWA previously skipped the commissioned profile whenever a saved legacy
record existed. The recovered path tries the saved pairing first, then allows
the companion's authorized installation to replace an unreachable legacy
record only after native authentication. Pending receipts and a newer saved
selection retain their original owner. See [legacy pairing recovery](VERIFICATION.md#legacy-pairing-recovery--2026-11-18).

The [exact Pages deployment](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44287710879)
served the recovery release and matching PWA assets. The installed Pixel
passed frame-confirmed Settings, Quest Control details and return to Connect
through RustDesk MCP. The ordinary saved pairing was present at sign-off.
The physical automatic replacement case remains open.

After the installed Pixel refreshed to `20261118-dataset-sync`, RustDesk MCP
opened Recordings, changed the selected saved entry and returned to Connect.
Eight repeated Recordings-to-Connect cycles passed. The heading in each of
the 16 RustDesk frames matched the expected screen.
The mounted archive recovery and PWA startup checks passed on that release.

Serial-scoped ADB verified the installed Meta code 27 and lab bench code 33
against their signed APK digests. Quest File Manager woke the Meta baseline.
The native read-only receipt showed Prepare, Bluetooth enabled and the
trusted LAN listener available. OpenXR focus and frame progression were
observed. No fresh Polar connection or PMD start marker appeared in the
observation window. The next physical step is to verify worn H10 contact and
fresh ECG and ACC before collection. The official phone membership and Meta
successor gates above remain open.

The PWA repair merge `b7784bf` passed exact
[Pages run 44197683228](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44197683228).
Eight live files match Git. The delayed-resize APK Details regression passes
using 42 live assets and synthetic Quest transport. The physical Pixel loaded
`20261118-dialog-measurement` in standalone mode.

The reconstructed fixed27 native tree matches the release. All 325 Node tests
pass with zero skips, the GUI gate passes 339 states, and all 35 compact Update
renders pass. The native suite passes 907 tests with six fixture-dependent
ignores using a test scratch directory outside the checkout. The ARM64 check,
actual installed APK signature/alignment, cold session-profile closure, and
all 14 updater host suites pass. The Windows JNI fixture used MSVC. Current
Polar acquisition, battery, health and recovery checks pass. The Location
projection, required stream scope and startup fixtures now include the
current health projection, automatic discovery and battery retirement
dependencies.

The lab Quest launched and logged actual Polar connect, PMD readiness and
ACC and ECG stream starts. The Pixel retained saved membership and saw
Quest 2 in the real Chrome chooser. The public SSID placed the devices on
different subnets and blocked phone access to the Quest gateway. The
connection milestone below covers automatic recovery on that topology.
After reconnect, verify fresh ECG, ACC and contact before recording.
The separate attended update route remains its own hardware gate.

On the docked Pixel 10 (Android 17, RustDesk 1.5.3), RustDeskMCP first reached
the phone without an active capture. With capture active again, it returned
a frame and sent working taps. A fresh USB connection authenticated with the
existing permanent password and returned a frame without an approval tap.
Stay awake while charging was already enabled. The primary session was left
connected and the temporary verification forward was removed. Follow the
[docked reconnect procedure](RUSTDESK-PHONE-TESTING.md#reconnect-while-the-phone-is-docked)
when capture ends. Lock and reboot recovery still need an attended check.

#### Haupt-Runde follow-up questions — 2026-11-18

- Five required German placeholder questions appear together in the smartphone
  PWA after the verified final v4 CSV for a live-staged Haupt-Runde. This form
  may scroll vertically and uses three five-point Likert items and two 0–100
  sliders. Submitted files use integer answers and include the scale bounds.
  A completed locally imported main recording can also open the questionnaire
  from Aufnahmen. Responses use a separate private JSON file with the same
  participant ID, session UUID and run stem. It syncs automatically to the
  existing private Drive folder when the phone is authorized and Quest is
  idle. See [the data contract](DATA-LOGGING.md#haupt-runde-questionnaire).
- [x] Publish PWA release 20261118-post-session-questions in
  [PR 102](https://github.com/GeorgeFejer91/viscereality.org/pull/102).
  The exact [Pages run 44285296799](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44285296799)
  for merge 08284a0 passed and nine public files match Git. The exact merge
  passed the 378-state smartphone GUI gate, including independent mocked
  Drive delivery and recovery. The installed Pixel 10 loaded this release
  with the matching controller module and stylesheet in standalone mode.
- [x] Publish the single-page Likert and slider release in
  [PR 109](https://github.com/GeorgeFejer91/viscereality.org/pull/109).
  The exact [Pages run 44296820487](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44296820487)
  deployed merge 65336fc. Seven public assets match Git. The 378-state gate
  and a browser run on the public assets confirmed scrolling, numeric JSON
  and mocked Drive reconciliation.
- [x] Restore RustDesk Input on the docked Pixel. Android reported the service
  bound, and RustDesk frames confirmed working Settings and Back taps.
  A later check found the service off. It was restored, both taps worked
  again, and the service stayed bound after the validation forward was
  removed. The lab Quest remained awake in the developer APK under the
  watchdog.
- [x] Read back the phone's 18 uploaded payloads and their private Drive
  objects. Every phone payload matched its SHA-256 receipt and each remote
  ID, filename, byte length and private parent matched. Two independent
  Drive downloads matched the phone's recorded SHA-256, including a
  completed Haupt-Runde CSV. All 11 CSVs
  contained ECG, accelerometer and radius samples under one run identity.
  The six completed schema 4 Haupt-Runde exports each logged an ECG interruption.
  The phone had no submitted questionnaire in this audit.
- [x] Audit observation coverage in the six retained schema 4 Haupt-Runde
  CSVs directly on the Pixel. All 118 ECG and 30 accelerometer interruption
  markers overlapped gaps of more than one second between recorded sample
  observations. All six recordings also included playback pauses. One
  recording had 13 ECG observation gaps and a maximum spacing of 16 ms
  between adjacent sensor timestamps. Collect original PMD batches with
  their receipt and sensor clocks during the next worn run to distinguish
  delivery delay, timestamp projection and missing samples. The retained
  exports establish historical gaps. A fresh native capture is still needed
  for the uninterrupted acceptance gate.
- [x] Accept the native terminal completion status for the questionnaire.
  Three of the six completed schema 4 main runs ended with a projected
  audio clock below 900 seconds. The recording evidence check still
  verifies the native completion timing independently. The mounted Chromium
  gate passed 399 states, including the projected clock regression,
  byte-identical questionnaire delivery to a Drive fixture and recovery after
  a lost response. The Node suite passed 340 tests with 12 native-fixture
  skips. All 14 recording evidence tests passed.
- [x] Publish `20261118-questionnaire-completion` in
  [PR 115](https://github.com/GeorgeFejer91/viscereality.org/pull/115).
  The exact [Pages run 44314751038](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44314751038)
  deployed merge `739f4ca`. Eight live files matched the merge. Six
  questionnaire states on the public assets covered completed, aborted
  and practice runs, offline reload, and byte-identical Drive fixture
  delivery and recovery after a lost response.
- [x] Refresh the installed Pixel to the current PWA and verify its loaded
  questionnaire module. The Android launcher relaunch loaded
  `20261118-idle-release` with the questionnaire completion module in
  standalone mode. The phone is idle on Connect and still needs commissioning
  for this exact lab Quest.
- [x] Verify Recordings fits at 320 × 428 portrait and 640 × 320 landscape
  with 200% text in the mounted PWA. The short views retain selection,
  downloads, retry, Drive, browser archives and 48 px targets. See
  [verification](VERIFICATION.md#short-recordings-layout--2026-11-18).
- [x] Drive all four short Recordings views and the browser archive return
  with RustDesk MCP on the installed Pixel at 320 × 373 CSS px. Android text
  scale was 2.0 and Chrome reported a 16 px CSS root. The views showed no
  document or panel scroll, and the archive and view selectors were 48 px.
- [x] Reproduce the clipped Recordings footer at 594 × 272 CSS px on the
  physical Pixel. The mounted 200% regression also failed before the fix.
  Use two columns at the measured landscape width, keep the sync status
  beside Retry and allocate the longer footer action more space. All 21
  targeted mounted states passed, including failed sync and recovery.
- [x] Repeat the original disconnected Pixel case after the first fix.
  Exact Pages run 44331741215 at `79d2a49` matched all nine affected live
  paths. The deployed Recordings layout still had 37 px of document scroll
  with the Bluetooth discovery status and no Drive control. Restored the
  device display settings and reproduced the disconnected state in mounted
  Chromium. The corrected layout passes 16 combinations of view, text size,
  and availability of a locally saved recording.
- [x] Publish and retest the disconnected layout correction
  [PR 126](https://github.com/GeorgeFejer91/viscereality.org/pull/126).
  The exact [Pages run 44335925396](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44335925396)
  deployed `4c822ea` and eight live paths matched Git. The GUI gate passed
  421 states. Serial-scoped ADB relaunched the idle Pixel into
  `20261118-recordings-offline-landscape` with its saved pairing intact.
  RustDesk MCP opened all four disconnected Recordings views at 594 × 272
  CSS px and returned from Storage to Connect. All four views showed zero
  document and panel scroll, full 48 px footer controls and visible
  content. Android text size, screen size and rotation were restored.
- [x] Retest same-folder sync on the deployed `20261118-idle-release` assets
  from [Pages run 44326627442](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44326627442).
  With synthetic Quest data and a simulated Drive endpoint, the PWA saved
  separate JSON and CSV files with a matching participant ID, session UUID
  and recording stem. Both files had the same private dataset parent.
  The JSON bytes matched the local download, and a lost response plus
  journal rebuild recovered the original file ID. All 24 targeted Node
  checks passed. The current standalone Pixel has 18 verified recording
  deliveries and zero submitted questionnaires. The actual private
  research Drive folder has 11 CSVs and seven ZIPs, with zero questionnaire
  files or dataset subfolders.
- [ ] Finish the paired lab validation above with the questionnaire release.
  After a real completed 900-second run, enter responses on the installed Pixel
  and match the downloaded questionnaire's participant ID, UUID and CSV stem
  to the finalized recording. Check that the separate JSON and CSV have
  the same parent ID in research Drive and that the JSON is byte-identical
  to the phone download. Use the
  [combined evidence check](QUEST-PHONE-SYNC.md#check-one-haupt-runde-evidence-bundle).
- [x] Extend the evidence verifier to check the Quest CSV and questionnaire
  together, including the run identity, private folder, separate IDs, exact
  downloaded bytes and schema 2 UTC metadata. The mounted PWA fixture passed
  offline reload, lost upload response and restoration of the original Drive
  receipts. Real questionnaire custody remains open on the lab pair.

#### Dataset folders and repeatable Quest–Pixel binding — 2026-11-18

- [x] Implement private hardware-pair and per-dataset Drive folders, preserve
  CSV custody and resumable IDs, and save Quest start and TimeAPI.io provenance.
  Node passed 339 tests with 12 native-fixture skips. The Chromium fixture
  verified the new hierarchy, both timestamp sources and a retried questionnaire
  upload. Startup and context checks passed. See
  [the data contract](DATA-LOGGING.md#dataset-folders-and-utc).
- [x] Document the [UTC timestamp protocol](DATA-LOGGING.md#utc-timestamp-protocol)
  for native measurements, phone observations, questionnaire submissions
  and Drive copies. The production PWA origin obtained TimeAPI.io UTC in
  95 ms with the fixed timezone request and no credentials or referrer.
  All 27 targeted Node tests and all 14 recording evidence tests passed,
  along with the project context check. Verify the UTC fields and clock
  provenance again on the next paired lab run and its new Drive files.
- [x] Register the USB-connected lab Quest 2 and Pixel 10 using the
  [USB provisioning command](QUEST-PHONE-SYNC.md#bind-the-exact-usb-hardware-pair).
  Read back the binding from the installed Pixel and obtained a live TimeAPI.io
  response in its browser. The USB Quest uses automatic network UTC. The
  headset briefly woke and then returned to sleep.
- [ ] Authenticate the exact Quest on the installed Pixel. It still showed a
  saved legacy pairing for another Quest. Activate the new binding after
  authenticated, automatic code-free commissioning over WebRTC.
- [ ] Authorize the research Drive account on that Pixel and check two new
  datasets, sibling files, UTC metadata and exact downloaded bytes.

Dataset release `20261118-dataset-sync` merged as `8ca2af6` in PR 105.
[Exact Pages run 44289536519](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44289536519)
succeeded. Eleven live runtime assets match that merge, and all 378 exact
smartphone GUI states passed. The lab Pixel reloaded the release in
standalone mode. Its deployed UTC module received a TimeAPI.io response
in 93 ms, and the USB pair binding survived the reload. Both devices had
automatic Android time enabled. The selected Quest was still asleep, and
the Pixel remained on Connect with the other legacy Quest. The next action
is to confirm headset pairing and authorize the existing Drive account.

At 12:13 CET the approved QFM single-device keep-awake route woke the lab Quest
with a 30-minute proximity hold requested. Fresh Android readbacks showed the
display on for over a minute and Atmungsraum Developer in the foreground.
QFM showed virtual proximity CLOSE; this firmware did not expose the remaining
hold time to QFM. The Pixel retained the USB pair binding and was still on
Connect with its other saved Quest. The official APK still needs authorized
automatic commissioning on the lab pair.

An empty Pixel upload journal previously made the same hardware pair reserve
a second Drive dataset. The PWA now finds the original private folder by
Quest recording ID and verifies its existing CSV and questionnaire before
restoring their delivery receipts. The mounted PWA fixture clears only the
synthetic upload journal, reauthorizes Drive and recovers both original IDs
with zero new reservations. Host tests reject changed bytes, a shared file
and duplicate or incomplete inventory. The physical Drive custody gate
remains open.

PR 107's `20261118-drive-receipt-recovery` release shipped with
[Pages run 44295116093](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44295116093).
Eighteen live files matched Git. The installed Pixel still showed
`20261118-dataset-sync` with saved pairing. RustDesk MCP captured fresh
frames, but Settings taps had no effect on the existing or newly
authenticated session. The latest readback still shows the Pixel's RustDesk
Input disabled, with Quest 2 awake on Meta-installed code 27 and separate
WebRTC Bench code 33. Once input is restored, verify a reversible RustDesk
tap, reopen the installed PWA on the exact deployed release and continue
the pending phone, pairing and Drive cases.

RustDesk MCP opened the Pixel Bluetooth chooser, selected Quest 2 and
pressed Pair. The next frame confirmed Bluetooth discovery and a failed
network connection. Both devices were on the hotel Wi-Fi. A phone-side
TCP probe to the Quest reported no route, while the same Quest gateway
answered via its exact USB connection. Continue with automatic WebRTC
pairing on the hotel Wi-Fi, then authorize the research Drive account. The
[standing development workflow](WORKFLOW.md#physical-device-development)
requires awake coverage throughout device work and RustDesk MCP for all
interactive Pixel actions.

#### Connection milestone — fast web pairing across isolated Wi-Fi

The latest applicable Pixel PWA and signed Quest APK must pair quickly and
automatically across both ordinary and client-isolated Wi-Fi. Opening or
resuming the PWA must connect to the authorized Quest without a code, chooser,
approval tap or hotspot. Test a provisioned phone's first connection with
empty PWA storage separately from saved reconnection.

- [ ] Provision lab device trust before the operator opens the app. The
  Android companion and Quest must authenticate each other using per-device
  proof and a short-lived rendezvous. A provisioned phone can obtain its
  enrollment automatically even with empty browser storage. A saved phone
  can rejoin automatically. A foreign or revoked device must be denied.
  Bluetooth discovery and the VDO room alone grant no authority. Hand the
  transport only dedicated session material.
- [ ] Start pinned, data-only VDO.Ninja WebRTC signaling as the primary
  connection route in the PWA and signed development APK. Let ICE select a
  reachable peer path and use TURN when a relay is needed. Keep the existing
  authenticated LAN route available when Internet signaling or ICE fails.
  Automatic connection attempts must never request, create or switch to a
  hotspot.
- [ ] Use the same native lease, typed requests, operation IDs, receipts and
  recorder across routes. Verify application authentication and protection
  for private payloads. Select one controller route and recover an original
  receipt after switching. Keep pending and stale states visible within the
  existing UI rules.
- [ ] Benchmark the installed PWA and signed APK side by side. Run at least
  30 trials per common-route comparison across repeated batches with
  alternated route order. Log time from Connect or resume to authenticated
  actionable controls, first native receipt and recovered control. Record
  medians, p95, failures and retries, selected ICE pair on both ends, traffic
  and active time. Include cold enrollment, warm rejoin and network
  transitions. Identify every APK, PWA and SDK build in the evidence.
- [ ] Before releasing the primary WebRTC route, bring shared-network
  readiness and action latency within measured LAN baseline variability,
  match or improve successful joins and recoveries, and verify comparable
  device and network cost. Fix and remeasure any missed gate. Repeat these
  measurements after pairing, transport or runtime updates.
- [ ] On physical Quest 2 and Pixel 10, exercise shared Wi-Fi, isolated
  guest Wi-Fi, high latency and loss, blocked UDP with reachable TURN,
  no signaling egress, interruption and return. Verify automatic first join
  and rejoin, a selected TURN route at both ends where relayed, and a matched
  native action receipt. Verify expired rendezvous, wrong peer, revoked trust
  and simultaneous routes cannot take or duplicate control.
- [ ] Complete the H10 and recording gates on the qualified isolated-network
  path. Verify that Quest keeps acquiring during an outage, the Pixel
  recovers the same session, and Quest, Pixel and private Drive exports
  match by hash using a synthetic participant identity.

The 320 px Connect render exposed a visibility-only fit gap. The main-screen
measurement observer now measures changed `hidden` state, including nested
labels, while ignoring idempotent visibility writes and digit updates to live
values. The delayed-resize regression reproduced the missing Connect summary
measurement before the fix and passes after it. The installed Quest 2 base
APK still matches fixed27-r2 by SHA-256.

Reference qualification passed 81 deterministic cases. Two separate Chromium
contexts ran the pinned VDO.Ninja SDK against public signaling and TURN.
Both reported selected relayed ICE candidates and matched an authenticated
reference command at revision 1.

The PWA pairing and gateway now accept a typed request transport with no LAN
address. The existing v2 matching-code ceremony and RQG1 status, owner proof,
lease and receipt rules run through that boundary. An alternate route cannot
request legacy enrollment, accept a forged status or publish a reply after
closing. Host verification recompiled the released fix27-r2 Java pairing
owner and passed 369 checks. A synthetic two-browser TURN probe reached that
Java owner, matched the two approval codes and stored exactly one enrollment
after both confirmations. Both browser peers reported relay candidates.

The PWA now tries WebRTC first when an enrolled Quest advertises a room in
BLE discovery. Without a reachable WebRTC peer, it can try the existing LAN
addresses. Automatic Connect leaves hotspot settings alone. The candidate
Quest bridge limits its WebView to the bundled scripts and VDO signaling.
WebRTC frames contain room-scoped tags and application ciphertext. The
Quest checks that proof before dispatching through the existing RQG1 member,
lease and journal. Untrusted room participants receive no enrollment or
experiment action from this route.

The exact browser module and bundled target page passed direct and forced
TURN probes with the compiled Java gateway and a synthetic owner. Both
peers reported the expected ICE route, the original synthetic receipt ID was
preserved and the decrypted 24,000-byte recording fixture matched exactly.
The native host checks and all 334 Node tests pass with no skips. The PWA
startup and all 340 rendered smartphone GUI states pass. The signed arm64
rehearsal APK passed v3, alignment, locked session, exact runtime asset and
assembly closure checks. [Candidate build and hash](../quest/apks/README.md#webrtc-rehearsal-candidate).
Automatic first enrollment of a provisioned phone with empty PWA storage and
physical Pixel/Quest timing, denial, recovery and recording checks remain open.

The non-Developer WebRTC build now has its own package,
`org.viscereality.atmungsraum.webrtcbench`. Its signed APK was installed
on lab Quest 2 over USB and the installed file matched SHA-256. The Quest
also reported the rehearsal and Meta Developer versions as separate
packages. A synthetic membership is staged in the bench package for the
attended tests. First launch permission, physical relay and the phone
receipt remain separate gates.

Automatic first pairing now starts from Android managed configuration.
A commissioned phone with empty PWA storage gets one authorized Quest
identity from the companion, discovers its current room and proves
possession of a dedicated device key. The companion verifies the Quest's
response before the browser can open the encrypted membership ticket.
An isolated Quest bench APK is installed at code 28 with exact signed
byte readback. The separate Meta Developer release continues at code 27
for the attended update protocol.

The real PWA launch, companion proof implementation, bundled target
page and native Java gateway passed on direct WebRTC and forced TURN
with initially empty browser storage and zero LAN addresses. A forged
Quest proof left storage empty and dispatched zero native status
requests. These host cases used synthetic records. All 335 Node cases,
the startup check, the native gateway host checks and 340 GUI states
pass. The lab Pixel still has discovery companion code 1, and the test
code-2 artifact has a different signer. Physical first pairing needs
a correctly signed companion and the authorized device configuration.
See [the managed phone procedure](../phone-discovery/README.md#managed-first-pairing).

The commissioning release is live from merge `d884853f496a`, Pages run
`44216308810`, with all 27 changed files matching Git. The installed Pixel
PWA reloaded that release, reached measurement readiness and retained its
saved Quest trust. A separately named debug companion is staged on the
Pixel for the synthetic bench run. The installed debug APK matches its
signed build and coexists with the original companion. Its private fixture
contains a synthetic derived bootstrap key for the already staged Quest
bench membership.

The bench companion now runs on the Pixel with Android Nearby devices
granted. RustDesk MCP capture was active while its Android input binding
was off. Re-enabling that binding through Accessibility restored verified
taps. Chrome 157 had granted Local network access but automatically
blocked Apps on device for the site. An exact-origin permission change
through RustDesk restored loopback access. The installed PWA observed
the real Android companion receive GET and POST, return 200 for the
commissioned identity and reject an invalid proof request with 403.

The lab Quest woke under a bounded QFM hold after a serial-scoped ADB
wake lasted briefly. ADB launched the signed bench. Android showed its
Nearby devices prompt in the immersive compositor and the bench waited
for the grant. ADB could bind the prompt to the bench owner and capture
the compositor, but the Allow control was outside the captured view.
After the grant, check a temporary empty pairing profile against real
BLE, authenticated native status and the selected ICE route. Require the
expected membership, zero chooser and code prompts, and restoration of the
ordinary PWA profile. The managed production phone and the attended
recording and update gates remain open.

Canonical Developer assembly now resolves the existing non-debuggable
package, controller-only policy, WebRTC bridge and pinned assets, Meta SDK
updates and the external helper handoff together. Eight focused assembly
tests pass, including stale-version and conflicting boot-options rejection.
The existing Developer signing certificate matches the installed code 27
and helper. Building the canonical artifact requires access to the retained
private profile, SDK inputs and signing environment. See [the release
recipe](../quest/README.md#canonical-quest-developer-apk).

The combined 320 × 568 and 200% text stress case exposed overflow in
the APK identity, release policy, Live ECG and pending receipt dialogs.
Individual package and build views preserve both exact values. The
dialog uses the safe available width, and the receipt selector stays
ahead of its content. Focused browser measurements confirm the body,
footer and content bounds with the same enlarged text settings.
The smartphone GUI gate passes 349 states and the extended layout run
passes 166 rendered states. All 337 Node tests pass with the native
JVM fixtures enabled and zero skips. Direct ICE and forced TURN each
pass the current empty-profile browser test with no chooser, code or
LAN addresses.

The [autonomous verification release](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44223717447)
from `56e4d66a5441` matched all 27 changed live files to Git. The actual
Pixel loaded `20261118-companion-permission` with measurement ready and
saved trust present. RustDesk taps verified Settings, Quest control, close
and return. Re-pulling the two installed Quest APKs confirmed their exact
signed bytes. The bounded Quest hold ended with prior stay-on and
proximity state observed again. The Quest is asleep and the bench grant
and attended run remain open.

Native verification commit `53766ae3b59b08239f5227f46e333d053c3dd08a` repairs three
host fixtures. It extracts the actual health projection and tests that
existing connected ACC and ECG stay fresh when Location blocks discovery
(36 assertions), that GATT retirement invalidates a pending battery read
and the latest private ACC frame (18 assertions), and that startup
permission and ownership gates survive Direct/Remote selection.
Both required-stream admission fixtures pass 12 assertions each. All
eight focused Polar host suites pass. The signed APK runtime sources
match the earlier checkout. [Native verification recipe](../quest/README.md#webrtc-candidate)
The [Polar verification deploy](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44225449427)
from `885c096316ab` matched all four changed files to Git.

#### WebRTC under delayed signaling — 2026-11-18

The browser proof with a synthetic Quest owner and real VDO.Ninja signaling
reproduced the old five-second negotiation cutoff. With 800 ms added to
each outgoing signaling message, first pairing over forced TURN timed out
before any native exchange. A saved pairing also tried its advertised
unreachable LAN address before later joining.

A fresh pairing or a saved pairing without LAN candidates now has up to
15 seconds to reach the data channel. With saved trust and an advertised
LAN address, the five-second fallback still applies until VDO.Ninja
selects a peer. Once a peer is negotiating, the 15-second deadline lets
its ICE and TURN exchange continue.

With the new deadlines, an empty profile joined without a chooser or code
after 7.3 seconds on direct ICE and 8.8 seconds over forced TURN with
1,200 ms added to outgoing signaling. The saved profile on forced TURN
joined in 5.6 seconds at 800 ms delay without attempting the blocked LAN
address. A separate saved profile with no advertised WebRTC peer in the
signaling room tried its LAN address after 5.4 seconds. These are
individual synthetic browser observations. The physical network matrix
and benchmark remain open.

The GUI gate twice caught the focused participant form extending beyond
the reduced viewport. Holding resize events reproduced an 844 px
document inside a 591 px viewport. The main layout and dialogs now cap
the measured height at the current CSS viewport. The regression checks
zero scroll range, full geometry, and retained focus, value and caret
with resize event delivery held. The actual Pixel keyboard remains
part of the physical UI matrix.

The [Pages deployment](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44227436820)
from `abae263905c6` matched all eight changed files to Git. The installed
Pixel loaded `20261118-webrtc-signal-budget` in standalone mode with saved
trust and measurement ready. Frame-confirmed RustDesk MCP taps opened
Settings and returned to Connect. The lab Quest still matched the signed
Meta Developer code 27 and isolated WebRTC bench code 28 bytes.

#### WebRTC recording transfer stress — 2026-11-18

The current PWA recording modules, compiled Java gateway and all nine
WebRTC assets from the signed bench APK exercised a 74,967,884-byte
synthetic export in two Windows Chromium contexts. On direct ICE and
forced TURN, the peer closed during an in-progress batch. IndexedDB
retained exactly 1,048,576 committed bytes and the new connection retried
from that offset. Each run completed 2,273 live range requests and 16
final range requests, preserved the original receipt ID and matched the
source SHA-256.

| Selected ICE at both peers | Rejoin | Remaining live prefix | Range p95 | Concurrent status and receipt p95 |
| --- | ---: | ---: | ---: | ---: |
| Host | 0.87 s | 55.77 s | 26.9 ms | 55.1 ms over 36 pairs |
| TURN relay | 1.43 s | 157.15 s | 80.4 ms | 148 ms over 97 pairs |

These are single runs over each route with synthetic bytes and an
injected peer close. Thirteen focused recording and simulated Drive
checks passed, including repeated lost upload responses with one file
ID per run. The physical network matrix and Quest-to-Pixel-to-private
Drive custody for a new H10 recording remain open.

The lab bench UID now has confirmed Bluetooth scan, connect and advertise
grants from serial-scoped Android developer commands. The Quest woke
under a bounded hold. The prior permission panel stayed visible above the
Developer scene after the grants and a bench relaunch attempt. The bench
was absent from the foreground and process readbacks, so the physical
first-pairing probe stayed stopped. The hold expired with the previous
stay-on setting and the Quest asleep. At the next attended window, clear
the old headset panel and open WebRTC Bench. Resume the empty profile
check with the installed Pixel, then physical reconnect and receipt checks.

#### Controller State CSV v4 — 2026-11-18

The v4 projection dropped Controller State observations already present in
the native breath stream. The regression reproduced the missing row. The
repair writes the unbiased classification for both recognized input sources
after official Start. The native manifest records the projection used by
the run so the interrupted-run replay keeps the previously published
prefix. Tests cover current replay, older v4 and v3 recovery, and a
backdated pause with already committed bytes.

All 80 focused recording tests and 908 native tests passed, with six
fixture-dependent native tests ignored. The cumulative patch reconstructs
tree `877d3a7c045a867c1ceef3d6d3ab5bc7db2e963b` from fixed27-r2.
The [signed bench code 29](../quest/apks/README.md#webrtc-bench-with-controller-state-csv-v4)
is installed with exact byte readback. The Meta Developer code 27 APK hash
was verified after replacement. The installed Pixel still loads
`20261118-webrtc-signal-budget` with saved trust.

The stale permission panel survived replacement. Serial-scoped Android task
inspection identified a dedicated root containing only the bench's request,
and closing that root cleared it. The next ordinary VR launch started the
bench and exposed two ungranted Location permissions in its selected startup
sequence. The lab grant and cleanup were limited to the bench and its
permission task. All six startup grants now read back, and neither permission
panel remains. Horizon kept the Developer scene in front when the bench
process started. The bounded hold expired and the prior stay-on setting
was retained. The next action is to open WebRTC Bench from Unknown Sources,
then resume physical first pairing and receipt checks. In the recording
test, compare Controller State's raw breath source and unbiased phase with
the CSV v4 and custody hashes.

#### Shared-scheduler Polar recording and evidence check — 2026-11-18

The shared scheduler supplied Java-clock Polar breath assessments to a
recorder that expected OpenXR time. A native regression reproduced a
queue rejection. The repair accepts the shared-scheduler clock when the
run's validated tracking configuration binds Polar ACC to Java time.
The same rule applies during queue admission, writing and recovery.
The regression now saves, recovers and exports an unbiased phase with
the original clocks and zero writer drops. All 909 native tests passed
with six fixture-dependent tests ignored.

The [recording checker](QUEST-PHONE-SYNC.md#check-one-haupt-runde-evidence-bundle)
compares native ECG, ACC, radius and breath rows and signal events with
the CSV. It checks the native hashes, checkpoint receipt, three CSV
copies and one reserved Drive object. Thirteen CLI fixture tests pass. The
retained synthetic partial artifact from the native writer still passes.
A Drive inventory marked incomplete cannot prove uniqueness. The checker
verifies native signal-event UTC and supports requiring an uninterrupted
run with breathing observations and continuous ECG and ACC receipts.
The focused regressions reject missing Polar coverage even when the three
CSV hashes agree and no interruption marker was written.
The checker replays the native playback transitions against the CSV and
checks their UTC, elapsed, pause and phase fields through the terminal row.
Fixtures reject missing or fabricated transitions and altered clocks even
when all three CSV copies and their hashes agree.
Matching hashes and completed flags could previously admit a bundle with a
missing or out-of-range native completion event. The checker now binds the
durable completion event to the native start, terminal and UTC clock. It
reports the start-to-completion time alongside the projected audio duration.
A fresh native build passes all 81 recording tests. A synthetic finalized
partial run from the native writer and export adapter passes with local CSV
copies and simulated Drive metadata. The physical run still requires the
actual AudioTrack and collection receipts.

[Signed WebRTC bench code 30](../quest/apks/README.md#webrtc-bench-with-shared-scheduler-polar-recording-clocks)
passed exact installed-byte readback and retained the bench UID and all
six startup grants. The Meta Developer APK digest matched code 27.
The installed Pixel PWA loaded the current signaling release. The
provisioned Kiosk catalog selected code 30 and acknowledged a normal
launch. ADB found the bench process with the Developer scene still in
front. The Kiosk catalog did not reach the foreground, and its original
selection and guard were read back after restoration. Open WebRTC Bench
in the headset, then resume the physical pairing, recordings and custody
checks.

#### Physical automatic commissioning over WebRTC — 2026-11-18

The lab Pixel reached its provisioned Android companion and scanned while
the signed bench code 30 was foreground. The Java Bluetooth validator
rejected its own v2 discovery snapshot when the WebRTC owner appended the
optional offer, so advertising was never reached for that snapshot. The
production callback regression reproduced the rejection and now passes with
119 checks. The gateway host suite also passes.

The [signed bench code 31](../quest/apks/README.md#webrtc-bench-with-bluetooth-offer-admission)
matches installed bytes on Quest 2 with the app UID and six grants retained.
The normal Android launcher brought it to the foreground, and fresh OpenXR
readback showed focused, advancing submitted frames. The installed Pixel
PWA discovered the headset and authenticated native status in 19 seconds,
using the real Android companion and a temporarily isolated pairing slot
that started empty. No chooser or code appeared. The phone selected a
local TURN relay and remote server-reflexive candidate at 26 ms round-trip
time. Saved membership brought the installed Pixel back to authenticated
control in 4.4 and 4.6 seconds without a chooser, code or enrollment request.
A RustDesk MCP Recenter tap showed the pending state, then confirmed the
original native operation ID. RustDesk also opened APK Details and selected
Build. The full reported identity matched the signed APK asset and passed
the installed phone text measurement. The ordinary phone document and prior
Quest power policy were restored. The wearer runs, physical network stress
matrix and official Meta transition remain open.

#### Physical relay-only connection and recovery — 2026-11-18

The physical Pixel and signed Quest bench code 31 completed first pairing
in 10.0 seconds with zero LAN endpoints exposed to the installed PWA,
relay-only ICE on the phone and 1.2 seconds of delay per outbound signaling
message. Saved reconnect under those constraints first reached native
status in 8.2 seconds. Closing that transport and withholding phone
signaling for 10 seconds cleared connected status within 0.7 seconds.
It returned automatically 6.4 seconds after signaling was restored.
The Quest WebView and Pixel reported the matching relay candidate pair.
A RustDesk Recenter tap confirmed its original native receipt after recovery.

This controlled fault injection ran on the existing shared Wi-Fi subnet.
The actual isolated guest Wi-Fi and the physical timing matrix remain
unqualified. When signaling was withheld and the signed Quest offered a
LAN endpoint, the physical browser reported `ERR_ADDRESS_UNREACHABLE`
with local-network permission granted. The installed interface exposed the
raw fetch error. The PWA now keeps the connection error type through
fallback and renders the established German recovery guidance. The installed
Pixel loaded the exact deployed release and reproduced the corrected status
with passing text fit. After signaling was restored, the phone returned to
authenticated bridge status over TURN in 6.0 seconds with saved pairing.
See [connection guidance verification](VERIFICATION.md#connection-guidance-after-relay-and-lan-failure--2026-11-18).

#### Public hotel Wi-Fi pairing stress — 2026-11-18

Both lab devices reported the public hotel Wi-Fi on a shared IPv4 subnet.
The installed PWA saw its offered LAN address fail with local-network
permission granted. Venue client-isolation policy is unconfirmed.
The standalone PWA loaded release 20261118-connection-error-status.
The signed Quest bench code 31 was focused and submitted advancing frames.

The first default-ICE batch alternated three empty temporary pairing slots
and three saved reopens, with 23 seconds to reach authenticated actionable
controls per trial. Bluetooth rediscovery reached ready in every trial.
One first join (6.4 seconds) and one saved rejoin (1.5 seconds) passed and
selected TURN. Four attempts timed out without an observed selected ICE pair.
The follow-up alternated three default-ICE and three relay-only trials.
All six reached actionable controls over WebRTC with no chooser or code.
Default ICE took 1.8–5.5 seconds and relay-only took 2.1 seconds per trial.
Every selected path used TURN. A later default-ICE saved rejoin reached
authenticated bridge status in 20.9 seconds. RustDesk MCP then confirmed
the original native Recenter receipt. The normal Pixel document and prior
Quest power settings were restored.

[Pages run 44251972245](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44251972245)
passed for this executable release and the verification procedure. The
short batches leave reliability open. Capture signaling and room admission
during separate starts after Quest sleep to explain the timed-out attempts,
then repeat the full timing matrix. Still run the baseline and fault
collection cases with the wearer and the official Meta transition.

#### Quest target startup recovery — 2026-11-18

The physical fault reproduced a dropped target startup on bench 31 after
a rejected signaling attempt. Signed bench 32 now reconnects its signaling
and announces the same valid room by 3.1 seconds on the focused Quest 2.
The new APK matches installed bytes and retains the app UID and six
startup grants. Meta Developer 27 was also read back by its official package
and digest. The Pixel retest stopped at Bluetooth searching before WebRTC
while both lab APKs were advertising. Next, verify bench rediscovery with
both packages present, rerun the full phone recovery with the latest
installed PWA, and continue the hotel timing matrix.
See [verification](VERIFICATION.md#quest-target-signaling-startup-recovery--2026-11-18).

#### Bluetooth rediscovery with both Quest packages active — 2026-11-18

The lab headset registered the same discovery service twice. The previous
Pixel companion read only the first matching service UUID and kept
searching when it was owned by the other app. Companion 0.2.1 reads the
registered instances in sequence and accepts only the enrolled installation
in Remote mode.

The signed bench companion update retained the Pixel app UID and Nearby
devices grants. The latest installed PWA found bench APK 32 with both
headset packages present. The original injected target startup failure
now recovered authenticated phone status over TURN. In a separate physical
test with an isolated lease journal, a RustDesk MCP Recenter tap returned
its original native receipt. The normal phone document and prior headset
power settings were restored. The stock companion update requires the
retained signer for the lab Pixel's installed standard companion.
See [scanner verification](VERIFICATION.md#multi-instance-bluetooth-rediscovery--2026-11-18).

The restored lab commissioning profile and signed bench companion code 4
passed 30 first joins and 30 saved rejoins on the public hotel Wi-Fi with
the same focused Quest bench code 32. Every trial reached actionable
controls through TURN using the installed PWA, without a pairing code or
chooser. Median readiness was 1.92 seconds for first join and 1.77 seconds
for saved rejoin. The corresponding p95 was 16.56 and 4.58 seconds.

With LAN excluded from discovery and phone signaling delayed by 1.2
seconds per outbound message, first pairing passed in 10.05 seconds.
A 10-second signaling interruption cleared the connected state and the
same saved pairing regained native status 6.08 seconds after signaling
returned. RustDesk MCP matched the native Recenter receipt after recovery.
The normal phone and Quest power state were restored.

Continue with the common-route LAN comparison, distinct client-isolation
and network-wide blocked-UDP cases, revoked trust and simultaneous routes,
worn H10 recording and export custody, and the official Meta transition.

#### TURN/TLS and invalid controller requests — 2026-11-18

The current standalone PWA and focused Quest bench 32 paired through a
TURN/TLS-only phone ICE fixture. Cold browser enrollment took 2.19 seconds
and saved reconnection took 4.25 seconds in the controlled repeat. The
headset independently observed the phone relay candidate. A wrong member
tag and altered ciphertext reached real WebRTC channels. The channels
closed without authenticated status or actionable controls. The final
valid reconnection returned the original RustDesk Recenter receipt.
Two attempts in the earlier batch did not reach a data channel. See the
[route and trust evidence](VERIFICATION.md#phone-turntls-and-invalid-controller-requests--2026-11-18).

#### Commissioning after Bluetooth snapshot expiry — 2026-11-18

Two traced first joins opened their first WebRTC data channels, then the
phone's commissioning request returned HTTP 403 before any proof frame
reached Quest. The PWA waited for its next connection attempt. A fresh
Bluetooth scan and a second WebRTC carrier brought both cases to control
in about 16.4–16.6 seconds.

A controlled repeat aged the first ready snapshot while keeping the
native offer in place. The same local rejection reproduced and the
original PWA created a second carrier. Release 20261118-commissioning-snapshot-retry
refreshes discovery after the local denial and retries once on the same
carrier if the full owner and offer match. See
[verification](VERIFICATION.md#commissioning-snapshot-expiry--2026-11-18).
The exact deployed Pixel PWA passed the expiry fixture on Quest bench 32
with commissioned companion 4. It reopened authorization after 3.30 seconds
using the original WebRTC carrier. All 60 default-ICE trials reached
actionable controls. Four natural 403 cases recovered on their existing
channels. First-join p95 fell from 16.56 to 6.61 seconds across the two lab
batches. The H10 custody and official Meta transition gates remain open.

#### Physical participant keyboard layout — 2026-11-18

RustDesk opened the participant step and submitted its empty form. The
browser kept the step open and focused the required surname input. On the
411 px Pixel viewport, the visible height fell to 428.6 px when the keyboard
opened. The measured footer extended to y=459 px and the document grew to
467 px. The app entered geometry reflow, leaving its action buttons partly
behind the keyboard with every field label still marked as fit. After
dismissing the keyboard, a RustDesk Recenter tap returned its original
native receipt.

The production GUI fixture reproduced the empty form at 411 × 428 px with
436 px of document height. The compact participant layout reduces the gaps
around the form and status messages at 480 px of height or less. The focused
regression checks zero document scroll, visible 48 px footer targets, text
fit, and retained native invalid focus when the viewport expands again.

Release `20261118-participant-keyboard-layout` passed 351 GUI states and 345
web and bridge checks. The standalone Pixel loaded the exact
[Pages deployment](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44271093318).
Three RustDesk MCP taps on the empty form opened and dismissed Gboard against
Quest bench APK 32. In each cycle the 411 × 428.6 px viewport contained the
428 px document, and both 48 px action targets ended at y=419 px. All four
labels fit with both status messages visible. Native validation kept focus
on the surname, and Recenter matched its original Quest receipt after the
keyboard closed. The ordinary PWA document and Quest idle power policy were
restored. Exact installed APK reads matched bench 32 and official Meta 27.

Continue with the network fault and trust cases listed above, the worn H10
collection and export checks, and the official Meta transition.

#### Temporary status loss after Save — 2026-11-18

The mounted PWA dropped an explicit Save→Update intent when one native
status read failed after the confirmed Save receipt. The updated flow keeps
the saved state during a missing read on the same connection. It waits for
fresh native status and revalidates owner, lease, writer and audio park
before issuing the single update request. The regression reproduced the
failure at 320 × 568 with 200% text and now passes. The
[update verification](VERIFICATION.md#temporary-status-loss-after-save--2026-11-18)
records the software gates for `20261118-save-update-status`. Merge
`6d78a89` passed [exact Pages run 44238768542](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44238768542)
and six live file comparisons. The installed Pixel loaded the expected
standalone release and retained saved pairing. Six RustDesk MCP
Settings-to-Connect cycles passed, including three with rapid repeated
Settings input. The final frame showed Connect, and the Quest still reported
asleep over ADB. The attended Meta transition remains open.

### PWA update copy simplification — 2026-11-17

The actual Pixel repeated the same update requirement in four Connect notices
and repeated it on Update beside an enabled action. The compact-update release
uses one shared status/action projection. Versions and raw diagnostics stay in
Details. A published version not yet offered by the headset SDK says “Meta bietet
das benötigte Update noch nicht an.” with “Erneut prüfen”; an available required
update offers “Jetzt aktualisieren”. Never tell the operator to wait for
preparation while the action is enabled. Preserve independent native errors,
recording/save controls and unresolved original receipts. Routine successful
version checks remain hidden on ordinary pages. Independent specification/quality review passes. All325 full Node tests pass
with zero skips;163 focused cases and35 compact real-DOM renders pass, including
320px/200% text and stable focus/scroll, plus26 preparation/27 Meta cases. All
eight changed runtime assets and their import parents have fresh cache URLs.
Release marker:20261117-compact-update. Policy27 PR59/mergea4c46d4 reached exact
Pages44186834653 success;9 live files match and live policy27 probes pass
(synthetic transport). Compact PR61/merge704328c reached exact Pages44190653853
success. All16 affected live files match Git and35 public-runtime compact renders
pass (synthetic Quest transport, no native mutations). Local branch matches that
merge and is clean before this evidence follow-up. The lab PC also synced this
release and passed all 35 compact renders. The connected lab Pixel 10 WebAPK
loaded `20261117-compact-update` in standalone mode on the Quest connection
screen. Bluetooth radios and relevant permissions were enabled on both lab
devices. The lab Quest 2 was asleep with Developer closed during discovery;
wake it, clear prompts, open Atmungsraum Developer, and retry the in-app picker.
Attended
Quest1WMHHA46S32126 now has exact27 and active PWA control; the interrupted
25→26→27 route required the Meta Update button.27→28 unattended acceptance stays open.

### Active: PWA update from every preparation state — 2026-11-17

Attended continuation: the actual PWA check4e3deb141b1ae8c0 was accepted as
attempt4876935929144199935. Meta now advertised27, but StartAppDownload returned
DUPLICATE_REQUEST and reported the cached8,246,427byte download complete. Meta
installed exact26-r3 first and the old process closed. Helper3 stayed active,
truthfully observing26 while waiting for target27. The wearer saw the Meta store
Update button and confirms pressing it. Meta then installed exact27-r2; its
SHA256 matches6f14707053fc9c07b611485deebfe8554a71b405892e8678f2c7f4bb2d8e398b.
Helper3 requested fallback and the actual new process became focused; the
reloaded installed Pixel6a PWA reports27/updated/installed-package-confirms-update,
session_allowed:true and active phone control. No host install, app launch,
stop or reboot was used. This confirms installation and helper relaunch, but
fails unattended25→27 acceptance. Preserve the cached older-download/Meta-store
confirmation issue for a clean27→28 test; do not generalize it into a requirement
that every update needs a button press. The earlier21→22 unattended result stands.
The first passive observer stopped when pidof correctly returned exit1 while
the app was absent; its evidence is retained. The successor read-only observer
accepts that transition and confirms exact27/focus. No timer proves installation.

Attended extension: same-process PWA request1312caf4c1629771 found Meta26 and
started its8,246,427byte download, then ended update_required/native-owner-retired
after262,144bytes. No installation/helper handoff occurred. The old app does not
log the exact failed predicate; source inspection identifies ordinary nonblocking
registry/session/setup lock contention being conflated with definite retirement.
25 and26 share this hazard. Preserve the accepted attempt on UNAVAILABLE, pause
SDK effects until a fresh CURRENT observation, and retire only observed ownership
changes. Actual install grants remain mandatory; never block render locks or
interpret missing observations as valid.

- [x] Regress contention and implement/review/test typed CURRENT/UNAVAILABLE/RETIRED
  across prepared op9, polled replies, final install grant and helper handoff.
- [x] Publish qualified fixed27-r2/run046 through Meta.
- [x] Retain separate lab-sync evidence of exact Meta-installed fixed27-r2.
- [x] Confirm exact fixed27 on the attended Quest and reconnect the PWA.
  Manual Meta Update confirmation was required; not an unattended-update pass.
- [x] Sync the published PWA and native source patch to the lab PC, retain the
  verified signed APK locally, and confirm the installed Pixel 10 PWA release.
- [ ] Hold version-only28 until a fixed27 session is RUNNING/rec:true and root
  releases the attended update test.
- [ ] Test preparation/check and running-session save/update to28 on hardware.

Source4edf4b89/tree358a0c91 passes independent specification/quality review.
All907 Rust tests pass with6 intentional ignores, all14 updater suites pass
(183 coordinator/108 duplicate/64 service/12 queued-exit), plus110-source javac,
Android arm64 typecheck and the actual Java/native audio join. The locked-read
regression is RED as Retired and GREEN as Unavailable; this reproduces the
hazard, not the unlogged physical predicate. The worker retains the exact reply
and nested continuation; UNKNOWN cannot grant install or repeat helper ARM.
The old boolean JNI remains fail-closed; private nativeObservedFence(IJJJ)I
provides the typed worker result. Helper3 IPC has not changed.

The2,368,442byte cumulative patch reconstructs358a0c91 (SHA256
5c6e49967f7150fbb1e30390b92ed02aa81906e646e52d5b25dcc986ff070723).
Fixed27-r2/run046 passes1138 signed-package checks and63 negatives. Meta
1465199570012388 is current and malware/security passed. PR59 mergea4c46d4
set required/latest27; Pages44186834653 succeeded and root verified9 live files.
Concurrent lab-sync PR60 separately records a Meta-installed fixed27-r2 matching
the exact signed digest, helper3 and the Pixel10 standalone PWA release. Preserve
that lab observation; its update trigger and immersive relaunch are unverified.

The ongoing attended device in this worktree is Quest2 serial1WMHHA46S32126 over
Wi-Fi192.168.2.48:5555 with Pixel6a28251JEGR21026. Before the latest continuation,
readback was25/PID29345 and the previous request offered26/required27. After USB
reconnection, the continuation above established exact27 and actual PWA control.
Keep this evidence distinct from the separate lab-sync observation.
Version-only28 stays private and held until an actual fixed27 session is
RUNNING/rec:true and root releases the attended test.

Prior PR57/merge5df115b reached
exact Pages44178155493 success;9 live files match and live policy26 probes verify
25 requires the enabled update action,26 continues, and an actual-shaped running
25 retains explicit finish/save/update without automatic stop (synthetic transport).

Source PR58/merge2d59e16 reached exact Pages44183709720 success and10 live
files match, including the reconstructed patch. The user confirms fixed27 then
test28. Old version-only27-r1/run045 is revoked,
unpublished and lacks this fix; preserve its stopped qualification evidence.
Helper3 is unchanged: the observed failure precedes its handoff. No ADB app
stop/start/reboot workaround. A planned third request was not dispatched while USB was absent. After
reconnection, the PWA-only check815086825825642834 completed with installed25,
available26, required27 / required-update-not-yet-on-channel. No27 download or
installation began. The portal independently shows a single effective Quest2
Developer target27, Complete; a propagation/cache explanation remains unproven.

The user resumes attended testing and authorizes operating the USB Pixel. Quest2
is already on Wi-Fi ADB. Actual Meta23 was IDLE/g0/r1, rec:false, pending:false,
but its breathing preparation was playing and native update can_check:false /
busy_reason:session-active. The phone's current PWA hid the update action behind
plain Weiter, then displayed a disabled Jetzt aktualisieren without its primary
blocker. Its Polar question mark was missing legacy23 health telemetry, despite
live ECG/ACC. These are actual attended observations, not synthetic fixtures.

Root used one ADB idle23 force-stop/resolved launcher reopen to bootstrap25;
this bypassed the PWA trigger and MUST NOT count as a successful PWA update test.
The SDK downloaded/installed25 through Meta; helper3 requested fallback, and25
reopened with exact published SHA256226b90cedc56b7b44a8cdba885260ca280c59fce381341a9d7f5677a3e027951.
No host APK installation or host25 launch. The user explicitly challenges ADB
app relaunch as a workaround: subsequent app/update actions must use PWA, with
ADB limited to diagnosis/readback. No new host app stop/start/reboot is authorized
as a substitute for the PWA update flow.

After reconnecting the actual Pixel,25 reports fresh ECG/ACC/contact detected;
the Polar toolbar is green, automatic input is parked/silent, and no recording
runs. A real PWA Erneut prüfen click is accepted and settles current25, same
correlated attempt, cycle1/revision4→5. This proves a PWA-triggered check while
parked, not a PWA-triggered installation after active preparation.

Accepted user behavior: update must remain accessible outside actual recording.
End calibration/input testing safely before checking/installing. During actual
recording, offer an explicit end-session/save/update action, wait for confirmed
writer completion and actual audio parking, then check/install. Never silently
stop recording for a background check, erase an uncertain journal, force unknown
writer state, install while recording, or use a timer as proof of park/save.

Implementation plan (this document remains the single lifecycle owner):

- [x] Capture actual phone/Quest baseline and the parked25 PWA manual check.
- [x] Native owner: regress active preparation causing permanent can_check false;
  add a fenced preparation-end/actual-park route before update admission. Reuse
  current runtime/writer/audio/helper owners and correlate all requests/results.
  Preserve unknown-command/writer and irreversible-install barriers.
- [x] PWA owner: expose a clear update action in sequential setup, show any real
  blocker alongside it; distinguish preparation from a recording. Compose
  explicit Stop/Save→native park→update with original operation receipts and
  reconnect/actor/generation fences. Improve missing-health legacy copy while
  retaining no false strap-contact confirmation from ECG alone.
- [ ] Independent scoped/integration review, native and real-DOM regressions,
  exact source-patch reconstruction, signed APK qualification and publication.
- [ ] Attended PWA-only installation, calibration/testing→update, running-session
  save→update and helper-relaunch acceptance. Keep actual wearer gates explicit.

Ruling: user instruction supplies design/execution authority; no repeat approval
menu. Use existing calibration worktrees and this plan, not another planning
ledger. SAIA task-router is unavailable; deterministic local work and the
selected Superpowers subagent workflow apply. One native owner and one PWA owner
coordinate the wire contract before integration; no duplicate concurrent edits.
Current Meta baseline is25, parked and eligible for a PWA-only bootstrap.
Old version-only26-r1 is revoked as an upload target and stays unpublished.
Repaired26-r3 is the planned combined baseline; a new held27 follows its qualification.
The replacement repaired baseline is distinct from a version-only test probe.

Owned interface: private enrolled-LAN capability meta_update_preparation:1 and
truthful native meta_update.can_end_preparation. New closed operation
end-preparation-and-check uses the existing v/id/op/cycle/revision request;
accepted receipt binds EXACT request cycle+1 as its target and retains one
attempt through measured audio parking and SDK checking. PWA follows that exact
target cycle/attempt without retiring its journal merely because the cycle moved.
Original same-ID requests reconcile without replay. Native actual recording and
ARMED/writer work cannot use this preparation-only route; existing Finish/save
must first confirm its original receipt. Fresh ownership/revision is checked
again before the subsequent update, and an already running automatic post-save
check is followed rather than duplicated.

Early independent review requires regressions for concurrent duplicate IDs,
pre-reset paused/frame0 clock reuse, and an owner/cycle retiring between native
preparation admission and Java readback. No accepted receipt may silently adopt
an unrelated later cycle. These are implementation gates, not device evidence.

Before concurrent integration, the updater implementation passed scoped and
whole-change review,319 Node tests/zero skips,23 mounted preparation renders,
and902 native tests with6 intentional ignores plus actual Java/audio/writer joins.
Native source b90e2f94/tree65718c6d is committed. Intermediate26-r2 built from it
is unpublished and revoked as an upload target: current main additionally landed
authored instructions/Probe/Freie Erkundung/main/outro source. Preserve those
changes while merging the updater repair; do not overwrite the public authored
native source patch or silently enable new audio. MainStudyAudio remains an
explicit opt-in; the update candidate retains the already qualified loop/audio
payload and900-second main. Revalidate the combined source and interfaces, then
build26-r3 under a new run name. Root owns final private source binding and release.

Merged PWA qualification:323 Node cases pass with zero skips;25 mounted
preparation/update renders include exact calibration narration and pending guided
Start, plus26 compact Update states,10 current-session/offline archive renders
and the authored operator flow. Scoped independent review passes and all14
changed runtime cache identities match. Calibration narration is cancelable only
with exact native instruction identity and explicit preparation-end admission.
An unresolved guided Start retains its original journal; Save waits for that
original confirmation. No device acceptance is inferred from these fixtures.
Native integration is reviewed and committed at2f41df0a/tree70e03235. The
cumulative source patch reconstructs that exact tree (2,350,099bytes/SHA256
bfeaaad65be614c1889ab3adb721fca4a3094ce371b6afbd265644c786936304).
All907 native cases pass with6 intentional ignores after an isolated and full
serial rerun of one existing submitted-frame freshness failure. Actual Java
gateway76/service56,110-source production compilation, source-closure11,
900-second/participant-plan bindings and Java/native audio joins pass. Natural
optional outro retains decoder admission until actual release; explicit accepted
preparation updates retire it without allowing stale receipts to cancel a newer
instruction. Default73 assets/900-second loop remain selected. Signed26-r3/run044
passes1135 signed-package checks and39 rejecting negatives. Meta build
1465122190020126 is current; malware/security passed. Actual installation is
pending after the observed owner-guard failure; package qualification is not
hardware acceptance.

The final PWA follow-up20261117-preparation-audio-recovery passes324 full Node
cases with zero skips,26 preparation and26 compact Update renders, and scoped
re-review. Narration-release waiting has specific conditional manual app
close/reopen and phone-reconnect guidance. No timer, mutation, journal or native
admission change is used for that copy. PR56/merge5f3b375 and exact Pages44169545352 succeeded;17 public files match
Git and26 live HTTP cases pass. The actual Pixel6a loaded that release with
active control and an enabled update button before the two requests.
The PWA is published through [PR55](https://github.com/GeorgeFejer91/viscereality.org/pull/55),
mergeae6d3e5, exact [Pages44165786315](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44165786315)
success. All25 checked public files match Git; the live HTTP runtime passes all25
preparation/save/update cases with synthetic enrolled transport. Installed Pixel
refresh remains pending: the previously connected phone is no longer visible
over USB and the user has been asked to reconnect it. Actual Quest remains25.


Private evidence: validation/2026-11-17-code25-attended contains native update/
helper observation, installed checksum, real phone DOM/screenshots, native status
and the accepted PWA25 manual-check receipt. No phone trust/data was cleared.

### Approved review repairs — 2026-11-17

The user approves implementing all five findings in
[the review](REVIEW-2026-11-17.md), which is the accepted repair specification.
Reuse the isolated web/native calibration worktrees and this lifecycle owner.
Hardware is unavailable: no headset/phone operation or attended-test claim.

Implementation plan:

- [x] R1, native Rust: add failing core/runtime/driver regressions for calibrated,
  fixed and adaptive Polar through clock/source retirement; retain manual
  selection and last visual phase while retiring readiness/model/confirmations.
  Preserve explicit owner/input/reset and initial Automatic behavior.
- [x] R2/R4, PWA: reproduce uncertain preparation journal against a fresh active
  recording; independently fence Pause/Finish without replay or journal erasure.
  Expose committed local archives with saved verified identity while disconnected;
  native synchronization remains connected, enrolled and idle-only.
- [x] R3/R5, Android: reproduce protected escape with armed helper watch and late
  handoff; cancel the watch through the common intentional terminal path before
  closure, preserving queued-update handoff. Expire delayed main-thread work.
- [x] Review each scoped repair, then independently review the integrated branch.
  Run meaningful host/native/real-DOM regressions and applicable package checks.
- [x] Publish verified PWA increments through ordinary PR, exact Pages and public
  runtime checks. Update cumulative native source/evidence, build a repaired APK
  baseline and keep its update-test successor unpublished until an actual run.

All tasks consume the existing native/status/journal contracts. R2 and R4 share
quest-ui.js and have one owner. R1 and R3/R5 touch separate native owners; source
freezes and commits are coordinated before APK packaging. No authority or
recording-admission relaxation is permitted. The independent audio-authoring
increment6074c11 is retained without adding its pending native integration.

PWA repair qualification:309/309 Node tests, zero skips, after integration with current main;
independent53-case review plus mounted production regression cases and10 rendered
states/19 viewport tiles pass. Pause/Resume/Finish preserve the original setup
journal and fresh native lease/revision/transport fences. Offline committed ZIP
downloads retain original bytes and verified device/install identity across APK
changes; identity switching/forgetting invalidates delayed reads and URLs.
Release marker20261117-review-repairs is live through
[PR50](https://github.com/GeorgeFejer91/viscereality.org/pull/50), merge701e8dd,
[Pages44135534810](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44135534810)
success. All12 checked public files match Git. A live Chromium probe verifies
Pause/Resume/Finish with the preserved setup journal and an original-hash offline
archive using synthetic transport. Installed-phone refresh remains deferred.
Native f5ea725c/tree df5d27e passes896 tests with6 intentional ignores and the
boundary scan; scoped and integration reviews pass. Signed25-r2 passes1114 package checks, including39 repaired DEX method paths/6
compiled guards and unchanged audio/private visuals/900-second closure. Meta
publication completed as build1464859736713038 in the private Developer channel;
malware/detailed security passed, Quest2/13+ and Require Latest unchecked retained.
Public minimum/latest is25. Source/evidence PR51 merged0a7fb24;
[Pages44137540961](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44137540961)
succeeded and all11 checked files match Git, including the reconstructable patch.
Final policy PR52 merged9e35559;
[Pages44138654869](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44138654869)
succeeded and all11 checked files match Git. Live Chromium confirms24 requires
the sequential update step,25 continues without that extra screen, and both
current-session journal controls and original-hash offline archives still work.
No new headset observation is claimed.

Ruling: existing publication authorization applies to verified repairs; no repeat
approval is needed. The old held version-only25 remains unpublished and is not
the repaired baseline. Repaired baseline25-r2 replaces it. Version-only26 passes
1229 signed-package checks and remains unpublished until actual25 is recording
RUNNING/rec:true and explicitly released for the attended update test.

Next attended action: refresh the installed Pixel PWA, verify Meta-installed25 by
exact APK/version/installer readback, inspect battery/contact freshness, verify
all input modes and tracking-loss continuity, then start Haupt-Runde. Do not
publish26 until the actual25 session has advancing timer and rec:true. Verify
no update interrupts that run and monitor the post-save update/relaunch. Overall
hardware/session/update acceptance remains open. No headset/phone command was
issued during these repairs. Native source is committed atf5ea725c; web runtime,
source patch and policy are synchronized through main9e35559. Final handoff
evidence is synchronized as documentation without changing runtime or policy.
All build leases are released. No unresolved software-review finding remains;
the deferred physical acceptance above keeps the overall project goal open.

### PWA startup update and Quest reconnect — 2026-11-17

- [x] Check the published PWA before the saved Quest connection starts. A new
  release loads first; an unavailable update check falls back to the loaded app.
- [x] Find the saved headset over browser Bluetooth or the Android companion
  first and authenticate the advertised Wi-Fi bridge. If discovery is
  unavailable or still searching, try the previously authenticated address.
- [x] Explain when Bluetooth needs attention and retry after it is enabled.
  Browser permission for a first or revoked grant still requires a tap.
- [x] Verify stale-release replacement, matching grant selection, a changed
  Wi-Fi address, companion-first discovery, Wi-Fi fallback, disabled Bluetooth
  recovery and the CSV-only route in Chromium.
- [ ] Repeat on the installed Pixel with the actual Quest's current and
  changed Wi-Fi addresses. No Android device was connected during this
  increment. Leave the separate wearer and recording checklist open.

The runtime commit e3a6969 reached GitHub main and deployed in
[Pages run 44131085006](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44131085006)
with deployment API status succeed. The entry, versioned app/controller
and stylesheet, and five affected notes matched the Git bytes. Live
Chromium passed changed-address discovery and Bluetooth-off recovery with
synthetic transport. Local Node tests: 297 passed, 9 skipped; the focused
layout, native-companion status fixture and remote context gate passed.
Next: reopen the installed standalone Pixel app at the deployed release and
repeat both connection cases with the actual headset.

### Unattended comprehensive review — 2026-11-17

The user is unavailable for device tests and requests a comprehensive review.
The [review report](REVIEW-2026-11-17.md) records four reproduced P2 issues and
one lower-priority P3 race: Polar clock/source retirement selects Automatic;
an old setup journal blocks current-session Pause/Finish; protected kiosk
escape leaves a pending boot watch; local archives require a live Quest to
download; and a timed-out handoff can apply a late kiosk side effect.

- [x] Independent PWA, breathing/audio and updater/helper reviews, plus sensor,
  recording and integration review. All five reproductions rerun independently.
- [x] Fresh PWA305/zero skips and native888/zero failures (6 intentional ignores),
  plus targeted health, recording and helper regression checks pass.
- [ ] Repair the findings with targeted regressions; prioritize retained Polar
  state and current-session controls before the next attended trial.
- [ ] Resume actual code24/contact/battery and running-session update acceptance
  after review disposition. Hardware remains deferred at the user's request.

This increment changes review/evidence only. PWA runtime, native0c82ff and Meta
channels are unchanged; code25 remains unpublished. The report's reviewed web
baseline is ab294b7, with the independently landed audio6074c11 additionally
reviewed for source timing, staging hashes and full rendered length. It remains
future authoring awaiting native integration. No further defect was verified.
Review documentation is being synchronized through the
existing GitHub/Pages workflow. Keep the overall goal open.

### Current attended test and preparation improvements — 2026-11-17

The Meta-installed23-r2 baseline is verified by exact APK hash, Meta installer
and focused renderer. The user confirms Controller responds promptly before
Polar calibration. A subsequent Haupt-Runde Arm is cleanly rejected: native
IDLE/g0/r1, rec:false. No experiment recording has started.

- [x] Repair selected/legacy plan selection and preserve original pre-JNI Arm
  failure identity (native d8998233/f24bd09a; public PR44, Pages44082868232).
- [x] Identify the remaining emitted-request mismatch: both signed23 codecs
  emit30000ms while the selected main parser requires900000ms. The existing
  signed24-r1 has identical DEX and is revoked as an upload target.
- [x] Implement closure-derived duration in both codecs and exact pre-JNI
  validation. Host Java/D8 and actual Java-to-native parser qualification pass;
  independent review and signed24-r2 qualification pass:1106 checks including
  actual emitted Java requests, signed DEX and the unchanged native parser.
- [x] Simplify Update to one primary actionable status, one check button and
  collapsed optional Details. Preserve native admission/recovery instructions.
  Independent review passes297 Node tests,135 layout states,26 compact Update
  renders and strict conditional-APK scroll/focus qualification.
- [x] Publish compact Update through PR45 as eedbdf72, exact Pages44091657428
  success and20 live-file comparisons. Installed-phone refresh remains pending:
  Pixel6a is no longer present on USB at the current readback.
- [x] Implement ongoing Polar green from current link/stream/contact health.
  Skin contact is explicit detected/not-detected/unsupported/stale/unknown;
  off-body ECG/ACC alone must not claim the strap is worn.
- [x] Implement sequential battery review before actual Start: Quest, optional
  controller and Polar. Missing/unavailable controller battery is informational,
  not a new admission requirement. No automatic abort/pause for battery values.
- [x] Qualify repaired24-r2 source and signed assets; reviewed native0c82ff/tree2444f55f.
  Code24-r1 stays revoked. Repaired24 is current on private Meta Developer
  build1464488216750190; malware and detailed security passed. Actual24
  installation and physical health/battery acceptance remain pending.
- [x] Publish Meta24 and health/battery PWA. PR46 implementation and PR47 policy
  merge166f483; exact Pages44097236831 success,21 public-file comparisons and
  public-runtime health/battery/cache/policy24 probe pass (transport synthetic).
- [ ] Refresh the installed phone, update actual Quest23→24 through Meta and
  verify current-device battery values and off-body/contact loss. The user
  defers attended testing until later; do not start recording or publish25.
- [ ] Establish actual RUNNING/rec:true on repaired24 before publishing held25;
  verify recording/saving continuity and safe deferred update/helper relaunch.

The overall goal remains open. Health/battery runtime20261117-device-health and
minimum/latest24 are live through PR47/Pages44097236831, with exact public
bytes and behavior checked. The final evidence increment follows unchanged
runtime proof. The held25 probe passes1100 signed-package checks; do not publish
before real24 RUNNING/rec:true. The user defers attended testing until later.
Last Quest readback remains Meta23; no host installation/restart for24. Location
and Wi-Fi ADB are retained. Phone refresh, actual readings/contact acceptance
and running-session update remain open. Native0c82ff and public source patch
are committed; the website branch is synchronized through origin/main166f483
before this final evidence increment. No worker source remains unfinished.

### Input continuity, automatic phone control and active-session updates — 2026-11-17

The wearer confirms Controller can drive the sphere before Polar calibration,
but reports about20seconds before motion and two abrupt minimum-radius changes,
possibly during tracking loss. The late device log shows repeated
setup-input-not-ready resets; it does not retain the initial startup interval.

- [x] Trace fixed Controller State startup to the unrelated120-frame volume
  calibration and implement a selected-experiment phase-only route.
- [x] Implement last-diagnosed-state visual continuation through temporary
  Controller/Polar loss. Keep native data freshness and admission gates.
- [x] Add regressions for phase-only startup, stale/reordered poses, state
  continuation, recovery, activation rejection and old-owner confirmation fences.
- [x] Complete native review and the final888-case host suite (five intentional
  ignores). Native453af6f4/treee1029a45 is committed; the public cumulative
  patch reconstructs that exact tree from its pinned upstream base.
- [x] Qualify signed code23: actual selected Android release build,900-second
  signed-APK session closure, same signer, unchanged SDK/helper/DEX and private
  payloads, and16KiB ZIP alignment. Final independent package receipt passes920
  checks. Source/evidence b0bbff0 merged through PR42 as2385101; exact
  Pages44074293155 succeeded and all11 checked public files match Git.
- [ ] Complete attended fresh-launch Controller/tracking-loss checks. No new
  APK has been installed yet; Meta delivery remains held for the running test.
- [x] Implement default native-confirmed phone control acquisition, continuing
  retries for definite free-lease non-grants, occupied/unknown barriers and
  manual-release pause. Toolbar green requires fresh native ownership.
- [x] Publish automatic-control PWA b65ec14 through PR41, mergecabadb7. Exact
  Pages44072213706 succeeded; all17 checked public files match Git. The public
  runtime automatically claims a free synthetic native lease, confirms green
  and switches all three methods with no page errors. Physical phone refresh
  and confirmed automatic acquisition are requested at the attended idle point.
- [ ] Publish the improved Meta version in coordination with an actual running
  session. Verify new availability does not interrupt recording or saving;
  observe the deferred update and helper relaunch afterward. Establish actual
  installed versions and lifecycle evidence, not an SDK acknowledgement alone.

The user is available and the Quest2 is verified over Wi-Fi ADB. Preserve the
current Meta22 baseline for the actual running-session test; code23 publication
is held until package qualification and the recording-running gate. The upload
chat confirmed Meta22 remains current and private, with Require Latest unchecked.

### Attended method-switch follow-up — 2026-11-17

The Pixel6a and Quest2 reproduced issues beyond the initial synthetic render:
healthy Meta completion still occupied the shared notice; sample-level ready
changes hid review instructions, changed button colours and erased local review;
and a real method request was rejected with stale-revision while native r changed
several times per second. This does not establish a broken calibration model.

- [x] Hide routine current/updated completion outside the explicit update page.
  Keep actionable warnings, errors and update-page focus/geometry.
- [x] Retain one review instruction and constant confirmation-button appearance.
  Native readiness still disables the action. Preserve local operator approval
  only for the same actor/audio/attempt/method/selected-sensor and reviewable
  testing state. Ready:false disables Start without bouncing out of Session.
- [x] Re-observe before each method request. Retry only a bound explicit
  rejected/accepted:false/stale-revision result, maximum3 new IDs, with unchanged
  intent and fresh action gates. Never replay accepted, pending or unknown work.
- [x] Verify actual fixed Polar, adaptive Polar and Controller selection on the
  connected Quest22. Adaptive received two explicit stale rejections and then a
  confirmed third request. No calibration, Arm or recording was started.
- [x] Verify unit/real-UI regressions, native gates and independent review.
- [x] Publish public20261117-selection-retry-2 through PR39. Implementation
  4b73139 merged as0e06fa3, exact Pages44060980318 succeeded, and all15 checked
  live responses matched Git. A fresh public-runtime Chromium check confirmed
  hidden routine completion, two explicit stale rejections followed by one
  confirmed switch, further adaptive/Controller switches and no page errors.
- [ ] Reload the idle physical phone from the public origin and repeat status
  checks. Pixel6a was absent from ADB after deployment, so this remains open;
  earlier candidate tests retained real Quest authentication/native decisions.
- [ ] At an attended fresh launch, test Controller before any Polar calibration.
  Source selection has no Polar-model prerequisite; effective output requires
  an unparked running audio clock and controller assessment fresh within250ms.
  A cold-start transport/tracking issue that calibration incidentally unblocks
  remains a separate hypothesis, not an established dependency.


### PWA input confirmation and conditional update step — 2026-11-17

- [x] Keep the confirmed input visible above the tracking circles; readiness
  polls gate actions without hiding/revealing the method/status block.
- [x] Name the choice Atmungsmessmethode and explain what it controls.
- [x] Use the same gold active outline for Controller, fixed/adaptive Polar
  STATE and personally calibrated Polar. Keep the active Controller choice
  present; a requested method remains separate until native confirmation.
- [x] Move APK comparison/update controls to a conditional preparation page.
  Current APKs skip it. Settings retains an optional update-page entry. Native
  admission remains authoritative, including offline use with a warning.
- [x] Mark the authenticated Quest link green after saved Wi-Fi rejoin, with
  transport wording that does not claim Bluetooth is carrying live commands.
- [x] Center the live ECG display on the median baseline; raw microvolts,
  timestamps, gaps, expiry, recordings and replay fences remain unchanged.
- [x] Verify the full Node suite with the selected native/Java integration,
  real-app input selection, update/handoff, layout and startup regressions.
  Independent review findings on update Back routing and stale capture-fixture
  routes were corrected and retested. See [verification](VERIFICATION.md#input-confirmation-and-update-step--2026-11-17).
- [x] Publish through the ordinary secondary-account PR37. Implementation
  7b79208 merged as dbb29ee; exact Pages44052231564 succeeded and all16 changed
  live responses matched Git. The live app mounted release20261117-input-status
  and switched fixed Polar, dynamic Polar and Controller through synthetic
  native receipts without page errors. Branch remains
  codex/atmungsraum-calibration-web-20261114; no native/APK edits.
- [ ] Refresh the installed phone at the next attended idle point and confirm
  switching methods against the actual Quest. Rendered checks use synthetic
  transport and do not establish a new hardware result.


Earlier hardware readback (20:15): Meta has subsequently installed16. Its exact
APK digest matches the published16, installer/initiator are Meta, and Android
records lastUpdateTime19:43:44. The headset is asleep at readback. This establishes
a delayed Meta15-to16 transition after the earlier bounded negative boot window;
it does not establish immediate update-on-autoboot or precise continuously
observed installer timing. Post-update boot/kiosk remains pending. The user-
approved explicit in-app startup updater continues, with SDK-enabled17/18 next.


## Active milestone: smartphone PWA and Quest 2 rehearsal

### Windows rehearsal phone pairing — 2026-11-16

- [x] Identify the missing pairing capability in the standard local code 18
  rehearsal package and qualify the opt-in `-PhonePairing` assembly profile.
- [x] Install and confirm the paired rehearsal package on the authorized Quest.
- [ ] Open the installed Pixel PWA, match the code on both devices, and claim
  control. Verify Polar readiness, Start/Pause/Resume/Finish, and recording recovery.

The Windows MSI now sees both devices authorized over USB. QFM confirmed the
installed Quest package, code `19`, rehearsal signer, and exact APK bytes.
Android resolved and started the rehearsal activity. The first-run Nearby
devices prompt awaits the wearer. The installed Pixel WebAPK and headset are
on the same Wi-Fi. The MSI's bounded QFM watchdog reports the headset awake
with stay-on and virtual proximity closed. Its local heartbeat runs every
15 seconds. Quest power readback confirmed a refreshed 30-minute proximity hold
within the existing watchdog deadline. The installed script now schedules a
renewal on its first eligible poll after future starts. The Pixel closed the
RustDesk session before a usable frame was available, and the temporary USB
forward was released. The operator can open the installed PWA directly on the
Pixel, or authorize a new RustDesk session.

Local rehearsal code `19` (`0.2.17-local-pairing`) uses the verified code 18
native tree and existing Windows rehearsal signer. The signed manifest and DEX
include matching-code pairing. The six assembly checks, 369 native pairing
assertions, nine browser-to-Java checks, 16 KiB alignment, and the packaged
900-second session check passed. The 55 focused browser tests for pairing,
connection UI, rehearsal commands, scenario protocol and recording sync also
passed. The private build capsule retains the APK digest and exact input lock.
On 2026-11-17 Android reported all three rehearsal Bluetooth permissions granted
and the rehearsal activity resumed. Verify an actual rendered rehearsal frame,
then match the phone and headset codes before claiming control and starting the
test.

### Operator UI compatibility audit — 2026-11-16

- [x] Apply the user's clarification that only Anleitung, five-minute Probe-Runde
  and fifteen-minute Haupt-Runde belong in the smartphone PWA. Use condition A
  and preserve the original native scenario and condition commands.
- [x] Trace the sequential screens to the existing Quest pairing, control,
  Polar and preparation, scenario Arm and playback, and recording/Drive paths.
- [x] Restore receipt access after reconnect and the recordings and Drive
  recovery routes. Keep the native freshness and recording gates.
- [x] Keep Back working when Quest skips participant entry and preserve
  the Recordings history entry after a native run finishes.
- [x] Restore the raw breathing value in Input diagnostics on APKs that report
  raw input but do not supply the newer renderer radius.
- [x] Verify both PWA increments on GitHub Pages and smoke-test the public app.
- [ ] Repeat the operator flow on the installed phone with Quest, a worn H10
  and the authorized Drive account at the next attended idle session.

The audit started from `fa22ff4` on `origin/main`, deployed by Pages run
[43939877200](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43939877200).
Compatibility release `dd80712` reached `origin/main` and passed the exact
[Pages deployment](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43947472089).
All 14 changed paths matched Git, and the live startup, offline recordings,
history and install dialog passed in Chromium at four viewport sizes.
The focus and startup follow-up `d0f049b` matched `origin/main` and passed
[Pages run 43948845817](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43948845817).
All eight changed live paths matched Git. Live Chromium repeated the four
viewport checks, including moving focus to Bluetooth before the dialog's
close event. A browser fixture also completed Drive authorization with fake
OAuth and verified the selected account and private folder readback.
The reduced picker release `4ccec3d` is live from
[Pages run 43956101913](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43956101913).
All 12 changed paths matched Git. The public PWA passed the synthetic Quest
catalog and exact Arm payload checks, plus startup and recovery in four
Chromium viewports. This PC's ADB inventory listed no device for the installed
lab Pixel refresh. Verify the installed phone release when it is available idle.
The follow-up reproduced and fixed two navigation failures and the missing raw
breathing value without changing Quest commands or polling. Runtime commit
`502d111` matched `origin/main` and passed
[Pages run 43959145457](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43959145457).
All nine changed paths plus both APK update assets matched Git. Chromium
startup and recovery passed at four sizes, and all three fixes passed against
the public assets with synthetic Quest transport. The APK comparison verified
both code 17 and 18 source-patch digests. Neither local device inventory found
the lab phone or headset. At the next attended idle session, refresh the
installed phone and verify pairing, Polar and input, Start/Pause/Resume/Finish,
and a synthetic recording through private Drive.
See [compatibility verification](VERIFICATION.md#operator-ui-compatibility--2026-11-16).
The native shared-scheduler catalog offers only `haupt-15`. The PWA needs
the corresponding native capability fields to show determinate session progress
or in-app Meta update progress. The earlier patient-first browser mode was
retired with the move to Quest-only operation. Its CSVs remain available at
`#daten`.

### Automatic PWA startup updates — 2026-11-16

- [x] Check the published release at every fresh launch before starting the
  Quest controller. Switch once if the release differs, and continue with
  the loaded version if the check fails. See the [release workflow](WORKFLOW.md#smartphone-pwa).
- [x] Verify a cached two-release launch, error and timeout fallback, and
  existing controller behavior in Chromium. See [verification](VERIFICATION.md#pwa-startup-updates--2026-11-16).
- [x] Verify the exact Pages deployment and the public startup behavior (PR28/Pages43939877200,21file matches and live ready/no-error check).
- [x] Verify `726204d` in [Pages run 43938500111](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43938500111).
  Nine public files matched Git; both entry URLs and the live current/stale
  Chromium launches passed.
- [x] Verify the installed app on the lab Pixel 10. On 2026-11-17 the WebAPK
  launched Chrome's standalone Atmungsraum document and loaded its app module
  with `20261117-meta-handoff`, matching the current public entry response.
- [ ] Verify live pairing and Polar H10 through that installed PWA. The Pixel's
  lock screen covered the standalone document. The Quest resumed the separate
  rehearsal app while the Developer code22 activity was in the background.
  Unlock the Pixel, open Atmungsraum Developer on the headset, then verify the
  phone's control lease and live ECG and motion from the worn H10.

Started from clean sparse publication main at `b3a6a59`, matching
`origin/main`. The primary checkout has pre-existing index changes and
cloud-backed files; it was left untouched. The verified runtime commit
`726204d` matched remote main, and the remote context gate passed.

### Main study audio — 2026-11-16

- [x] Prepare two control WAVs, one instructions WAV, four reusable breathing
  WAVs and `end.wav`. The [cue library](../assets/audio/scenarios/haupt-15/cue-placeholders/README.md)
  owns the 81-second opening, 50 four-phase cycles, and 19-second closing.
  All eight files contain silence. Their timeline totals 900 seconds and
  serves cue preparation. The active native loop owns Quest playback.
- [x] Catalogue 24 attachments as 23 unique MP3/TXT pairs. The
  [study library](../assets/audio/main-study/de/README.md) records transcripts,
  phase pools, quiet listening breaks and the review/staging commands.
- [x] Stage the playground-approved programme. Both controls are exactly
  5,615,640 samples. With the instruction they fill 5:00. Guided breathing
  fills 10:00, with 37 full cycles and a final inhale and exhale. The outro
  is a separate 19-second asset after the 15-minute timer. Verified all
  84 catalog file identities, all rendered narration and phase PCM against
  the playground, and the 919-second review WAV.
- [x] Stage the exact calibration narration and Freie Erkundung MP3 in the
  authored build. The Quest reader verifies the encoded clips, the guided
  Probe instruction and the Haupt-Runde outro against the catalog. See the
  [authored package checks](VERIFICATION.md#authored-quest-audio-package--2026-11-17).
- [x] Connect the first Polar calibration instruction to the selected Quest
  setup gateway and phone controls in the authored-audio candidate. Start
  calibration after playback completes and allow the operator to cancel the
  original request from the phone.
- [x] Preserve the authored Freie Erkundung duration and phase boundary in
  Quest recording, recovery and CSV export. Verify the corresponding phone
  export with both original Polar streams.
- [x] Connect Freie Erkundung's verified media to the enrolled phone chooser
  and native participant session. Use the headset media position for
  progress, preserve pause and resume, and save once at natural completion.
- [x] Connect the five-minute guided Probe-Runde in the authored candidate.
  Play the verified German instruction before the official timed start,
  keep the phone on the pending instruction, and derive the run's progress
  and completion from the Quest media.
- [x] Bind the authored Haupt-Runde programme and separate post-session
  outro to the Quest playback owner. Preserve preparation, recording
  provenance and the 900-second completion boundary.
- [ ] Qualify a signed authored build on a physical Quest 2 with the
  installed Pixel 10 PWA, including headset audio, Polar input and export.

The authored Probe-Runde package stages the supplied 45.4-second German
instruction separately from the 300-second plan. The timed plan uses the
study phase library and ends on an exhale. The authored candidate exposes
the guided Probe-Runde and Freie Erkundung through the Quest's verified
scenario list. The authored Haupt-Runde binds one lossless 900-second
programme to the recording clock and plays the separate 19-second outro
after saving a complete run. Device acceptance remains open.

### Separation of preparation and main study — 2026-11-16

- [x] Expose Anleitung, the five-minute Probe-Runde and the 15-minute
  Haupt-Runde. Give Haupt-Runde its own Hauptstudie section. See the
  [scenario selection verification](VERIFICATION.md#scenario-selection--2026-11-16).

### Operator UI port from Atmungsraum UI — 2026-11-16

- [x] Port the connection checks/setup, aligned participant form, large tracking
  confirmation, and session tiles from the playground onto production main.
- [x] Verify the controller flow, recovery, recordings, and rendered desktop/phone
  layouts with production browser fixtures. See [verification](VERIFICATION.md#playground-ui-port--2026-11-16).
- [x] Verify `4d5d3b3` in [Pages run 43930833015](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43930833015).
  All 14 changed public files matched Git; live setup and installation tested
  in four Chromium viewports. The primary checkout's existing index state was
  left alone; production was published from the sparse sync checkout.
- [x] Bring the current playground header and illustrated tiles to production.
  The four diagnostics, recenter, Einstellungen and Aufnahmen share the top
  row. The three Quest-supported options stay separated into Vorbereitung and
  Hauptstudie. Enlarged labels can reflow without clipping.
- [x] Check for updates on each fresh web/PWA launch and on return to the idle
  connection screen. The installed app keeps its ID and browser storage.
  Active views defer refresh. See the [startup workflow](WORKFLOW.md#smartphone-pwa).
- [x] Verify production release `a9d4096` on [Pages run 43962255351](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43962255351).
  The six public entry, runtime and manifest responses matched Git. Live
  Chromium passed the toolbar, focus and historical-recordings flow at
  320 × 640, 390 × 844, 844 × 390 and 1440 × 900. The exact previous release
  bootstrapped into the live update while preserving its URL and storage.
  ADB and Mobile MCP listed no local phone; standalone Pixel acceptance is open.
- [x] Port the phone's session particle preview and timed progress. Scope
  renderer samples to the current run. Code 24 supplies its active clock and
  confirmed main start frame but no duration field; use its fixed 900-second
  boundary only for the authenticated shared-loop main-only catalog.
  See [session preview verification](VERIFICATION.md#quest-backed-session-preview--2026-11-17).
- [x] Port the playground's single-viewport rules and automatic release gate
  to the production PWA. The gate qualified 338 rendered Chromium states
  against commit `faff9b4`. See [smartphone rule verification](VERIFICATION.md#smartphone-ui-rule-port--2026-11-17).
- [ ] Qualify the installed-phone keyboard, actual safe areas, Quest pairing,
  live Polar data and Google authorization on physical hardware.

Scope: production HTML/CSS, Pretext integration, and controller presentation;
keep native state, command admission, recording/Drive, audio, enrollment, and
the existing utility and recovery routes.

### Active update extension — 2026-11-16

Today's boot/update subset passes,2026-11-17: Meta22 build1463496130182732 is
private Developer current, Quest2/13+, malware/security passed. Actual boot12→13
autolaunched21; own DownloadSuccess and a subsequent fresh progress preceded SDK
install. Exact22 bytes were Meta-installed at02:26:01, retained UID10159, and the
helper fallback request at02:26:02.893 preceded the new focused22 process. The
Quest shell handles the final activity launch. User confirmed automatic stimulus,
PWA22 and no manual update trigger. No host main install or post-boot app launch.
Code21 baseline had also been exactly Meta-installed and launched by helper3.
Required/latest floor advances to22. Full package receipts and live boot trace
remain private. The duplicate/queued lane and physical phone handoff notice are
not separately qualified by this normal SDK run. Broader calibration/PWA fine
tuning remains open; the user ends testing for today.

Prior source/policy PR35/aa3882e passed exact Pages43992621746 and10 affected/core
byte comparisons, preserving concurrent PROJECT work. The final22 policy and
collaborator handoff use the same standing exact-publication workflow. The
collaborator receives a signed helper3 bundle and repository-backed setup
runbook; the signing key and private visual inputs remain private.

MSI installation, 2026-11-17: the selected Quest 2 already had the exact
Developer22 APK installed by Meta, matching the current published floor. QFM
installed the bundled, v3-signed Boot Helper code3 as a new package. Independent
device readback matched both APK digests, the shared signing identity and the
helper's `SetupActivity` launcher. This installation left the helper's overlay
permission at its Android default. Launch-after-restart setup and a real reboot
check on this headset remain open.

Meta21 is published,2026-11-17: a user-requested fresh chat recovered browser
access and uploaded the exact qualified artifact. Developer build1463458816853130
is current for Quest2/13+, tests Complete, malware passed and detailed security
All tests passed. Existing private access, artwork/testers and Require Latest
unchecked are preserved. Public required/latest policy advances to21. Installed
headset is still20; root cold-launched20 to prepare its SDK fetch and requested
one attended manual check. Actual21 Meta baseline and subsequent22 update test
remain open. No main21/22 host install or host22 launch has occurred.

Source PR34/811af1e was retained in descendanta5376d4. Its own Pages job was
superseded; exact descendant Pages43989652647 succeeded and all27 affected/core
responses matched Git. The code21 policy/publication checkpoint awaits its own
deployment verification. Pixel refresh remains user-operated with USB absent.

Monitor3 build checkpoint,2026-11-17: independently reviewed native source
1cd5ee3/treea9dc76d produces signed code21-r2, qualified169 actual-package checks.
The cumulative public patch reconstructs that exact tree from the pinned base.
The actual source-inventory failure on the first21 attempt is corrected and
covered by the strict resolver/packager/builder regression. All280 web/assembly/
actual-Java interoperability checks pass with zero skips. Helper3 passes50
Android boundary,27 boot,170 authority,43 monitor and26 builder checks plus
actual Android34 compilation and independent review. Its signed APK is installed
on the authorized Quest with exact byte readback, same UID10000 and overlay
Allowed; no main app was launched. The initial helper UI reports the installed
target Ready and monitor Inactive. Code22 is built but held unpublished; actual
Meta21 baseline and21-to22 monitor/install/relaunch/focus remain open. No main21
or22 host installation is permitted as distribution proof.

PWA handoff follow-up,2026-11-17: the operator now retains a bounded read-only
update notice across connection loss and document reload. Confirmed native
handoff is distinguished from an interrupted check. The notice explicitly says
phone progress is unavailable, asks the operator to inspect the headset and
reconnect once Atmungsraum reopens. No percent, ETA, timer completion or local
admission is added. Fresh same-Quest actual APK/native readiness retires the
hint. Eleven model tests,34 focused checks,271 web checks and six actual mounted
PWA disconnect/reload/reconnect renders at390px/320px/200% pass; the nine native
assembly integration cases require their separate selected checkout. Existing
27 Meta display/reflow renders also pass. Independent review passes. PR33 merge
c10d605 passed exact Pages43986515449 and all10 affected live responses match
Git; live Edge loaded the new entry marker and mounted controller successfully.
The Pixel is no longer present on USB, so installed-phone refresh remains open.
Actual new monitor3 hardware qualification remains pending.

Previous result/diagnostic increment PR32 merged a037b2e and exact Pages
43984840599 succeeded; all18 affected live responses match Git.

Current checkpoint,2026-11-17: Meta20 is actually installed with exact signed
bytes, retained UID and Meta installer attribution, at00:33:21 after root closed
idle19. It has not been host-launched, and no process is running at readback.
Earlier19 host launch reached advancing focused frames; repeated manual checks
ran in the same process. Meta20's offer initially lagged in the SDK, then
reported20. Actual download requests failed; scoped OCMS logs reject the
same-package StartDelayed request because one already exists. This ordinary
Meta replacement is not SDK install or automatic relaunch proof. The user
accepts closing the app for updates. Native21 adds a selected helper handoff and
safe cooperative exit for queued updates; independent software review passes,
with final package checks/builds and hardware qualification pending. Do not
adopt an unidentified existing download as completed.

Quest is verified on Wi-Fi ADB with the same boot. Its idle19 app was explicitly
stopped to release the H10 for the PC browser recorder. The phone is
USB-connected and user authorization permits controls. The local recorder's
null-device connection race is repaired and independently reviewed. Actual Edge
receives successful ECG/ACC start acknowledgments, then a Bluetooth link close;
physical connection acceptance remains open. The participant confirms a worn,
moistened strap. The recorder is committed in its own private worktree; no
actual recording was started by the agent.

PR31/Pages43974010304 succeeds with19 live-file matches. The new PWA clarification,
required/latest20 policy, bounded SDK diagnostics and calibration comparison are
local and unpublished;269 zero-skip full web checks,27 focused checks and27
Pretext/reflow renders pass, with independent review. Finish publication,
exact Pages/live checks and installed-phone refresh before claiming that release.

User requests manual check/update and fresh verification for each new native
session. On unavailable service/internet, permit installed build with warning;
known required update or installation uncertainty still blocks. Extend the
existing helper for bounded same-signer post-update launch and startup replacement
race; add public-repo branded Developer listing artwork. Follow the dated
[amendment and tasks10–16](BOOT-KIOSK-UPDATE.md#user-amendment-repeatable-checks-session-freshness-and-launch-fallback--2026-11-16).

Actual Meta18 installation is confirmed by exact hash/Meta attribution/retained
UID, following ordinary background replacement during boot12. It did not
relaunch automatically. Root subsequently host-launched18; this is not SDK
install/relaunch proof. Progress correction7b442a41 passes152/3/12 independent
checks, with real Android closure checked; new package/hardware proof remains
pending. PR29 ordinary mergec14d67f/Pages43953141067 succeeds; all16 affected/
preserved live files match and public PWA startup reaches ready. Phone's actual
native status verified boot10 channel17 and boot11 channel18 with progress error.

- [x] Diagnose actual Meta channel delay and download progress failure separately.
- [x] Review/test bounded progress correction and retain numeric diagnostics.
- [x] Repeatable native checks/policy/cycle grant and authenticated action implemented and independently reviewed; selected19 package/hardware qualification pending.
- [x] PWA v2/manual controls, original receipts and native status gates;266 zero-skip web checks,15 rendered states and independent review pass. PR30 ordinary merge7855177/Pages43970922434 succeeds;34 live files match. Installed-phone refresh remains pending; the connected Pixel6a is no longer listed by ADB.
- [x] Bounded helper fallback source/review and signed helper2 package pass; actual helper2 bytes/UID/overlay retained after host update. Protocol2 main/physical acceptance pending.
- [x] Adapted Developer landscape/square/portrait/hero covers and original icon saved and reload-verified in Meta draft metadata. Private distribution retained. Actual headset-library propagation remains pending. Public assets and prompts are in `assets/store/`.
- [x] Correct actual19 product javac lambda/Android bootclasspath compatibility. Signed19-r2 assembly/session closure and independent actual SDK/helper2/DEX/manifest/signer/private-input/audio qualification pass at native62ddb621; public cumulative patch reconstructs tree7806ebcf exactly.
- [ ] Establish Meta-installed19 baseline, publish higher20 and prove manual/in-process SDK install, automatic relaunch/focus and repeat-session freshness. Code20 assembly and independent actual package inspection pass. No main19/20 USB install.

Code19-r2 is65,682,565bytes, SHA256
`bc74a2bedc7cfc67c3ac4bf2cdbc57293f440361f870b97de73d729bd1604cb2`.
All15 private profile inputs and79 original audio/guidance/session assets match18;
new public audio-review material is not activated. Current publication checkpoint
does not establish physical updater/helper2 acceptance. The late answer to an
old reboot question is excluded from new-test evidence; no additional reboot
has occurred since boot12.
Meta19 build1463329980199347 is published in the private Developer channel;
Quest2 and13+ are retained, malware passes and detailed security says all tests
passed. Closing idle18 lets Meta perform the baseline update; it is not yet
confirmed installed. Successor20 stays unpublished until that readback.

### Code18 successor checkpoint — 2026-11-16

The final BLE correction is reviewed and packaged at native `f5946e19058e347f1dc69fde85b7265814c8ed9d`,
tree `7307b9c5a15e053fcb0aa211cc8a6afdfe2296ec`. New cumulative patch SHA256 `d787082a4bf358af4cfc4bf17cede57aebb9bb43766c4571bd53d5f7c5736d71` reconstructs exactly;
code17's original source remains in Git at `0fa9046`. Final18-r3 SHA256 is
`1463fa802a5439548bfda055b662ca483c86bf4cd59e03e4f75cc224c8aacb14`,
65,662,083bytes. Only the intended BLE class changes; all79 audio/session assets,
private inputs and SDK transport match17. Full product qualification passes.
Meta18 build1463187006880311 is available in Developer with retained13+ age
group and malware/security passes. The headset was off before publication.
Genuine boot10 launches17, reaches FOCUSED and calls Meta GetVersion; the wearer
confirms automatic stimulus. Download/install acceptance remains pending.
No manual/host18 installation.

### Autoboot, kiosk and explicit Meta startup updates — 2026-11-16

Follow [BOOT-KIOSK-UPDATE.md](BOOT-KIOSK-UPDATE.md). Overall hardware acceptance
remains open. Meta15 was installed through Meta with retained UID. The enabled,
same-signer boot helper passed genuine reboot, submitted-frame/focus progression,
physical menu recovery and protected escape. Meta16 later arrived automatically
at19:43:44 while idle; exact APK digest and Meta attribution match. The initial
208-second power-on window had not observed16, so immediate autoboot updating
is not claimed.

- [x] Implement official SDK207 check/download/install/relaunch before preparation.
- [x] Shared native maintenance admission protects calibration, direct/local
  Arm/Start/Resume, delayed main entry and recording. Safe failure permits setup;
  dispatched/external/persisted install uncertainty stays blocked across warm
  activity restarts. Kiosk hands off before installation and stays suppressed
  until a fresh process. Retrying a safe failure requires a Quest restart.
- [x] Independent native task and whole reviews, coordinator75checks,
  service/policy12checks, binding3cases, native admission5tests and actual
  Android BootEntry12/110checks pass. Native suite before the final bounded
  handoff passes862with five existing ignores. Existing broad panel/standalone
  audio-closure blockers are retained explicitly; they are not passing claims.
- [x] Build/sign/inspect final17/0.2.16-dev and18/0.2.17-dev. Actual Java/ARM64,
  session closure, SDK/DEX/manifest/hash closure, retained signer,16KiB alignment
  and independent private-input/pair comparisons pass. Both are65,662,083bytes.
  No SDK-enabled main APK is installed through USB.
- [x] Publish17 only to the existing private Quest2 Developer channel; Meta
  malware passes and security details say all tests passed. Same13+ age group,
  one-member channel and access policy retained.18 is now published separately.
- [x] Actual Meta17 baseline installation confirmed by exact digest, Meta
  installer/initiator and retained UID10159. Explicit trusted cold launch reaches
  FOCUSED; SDK initialization and GetVersion request are observed in vendor logs.
- [ ] Phone trust/result acceptance: compact BLE snapshots currently report
  budget errors with the strap off. A valid oversized optional Polar retry
  summary reproduces the error in host tests; exact installed payload is not
  captured. Code18 will include the bounded optional-summary fix.
- [x] Rebuild and independently qualify final18-r3 with the bounded BLE fix;
  all79 audio/guidance/session assets and private input files match17.
- [x] User confirms17 headset completely off, USB connected, before18 upload.
- [x] Confirm Meta18 availability and malware/security passes before power-on.
- [x] Genuine17 autoboot reaches FOCUSED and calls GetVersion; wearer sees stimulus.
- [ ] Complete the attended download/install/relaunch test.
- [ ] With17 installed and the Quest powered off, publish18, then prove the
  explicit in-app download/install/new-version/relaunch/focus/kiosk path.
- [ ] Publish the final SDK source/package evidence and verify exact Pages/live URLs.

Code17 native commit `b4a2d418abaccb3c317a820b88b2477bef798b93`, tree
`db0980b0cc6d6cbc399cde2102e9a77c110cb223`; cumulative patch SHA256
`db00bc88563f62f88cfe582b16f50eba88108d9203ec30d89ed1cf27f9ab3a72`
reconstructs that exact tree. SDK binaries, keys and private visual bodies remain
external. All15 original private input files match;85 original payload assets
are byte-identical. Four provenance/binding metadata assets change and two SDK
notices are added. Pair comparison normalizes only declared provenance/bindings;
native binary identity is not claimed. See [APK record](../quest/apks/README.md).

The readonly PWA progress/native gates are live in PR28 merge
`fa22ff4ba839173a474e886c9929d89332d02130`; Pages43939877200 succeeds,
21files byte-match Git, live startup is ready with no errors. Latest production
visual/session-section and PWA startup-refresh changes are preserved. Full web
249/249zero-skip, independent integration103/103,12update renders at320px/200%,
18release-policy renders, navigation/scroll/draft/focus and cadence checks pass.
Phone control remains with the user; Location stays ON. The user is ready for
the attended test and has been told to keep the Quest on until explicit power-off.

### Operator layout improvement — 2026-11-16

The user requests clearer information hierarchy, shorter sequential pages and
closer styling to the actual Viscereality website code, without new features.
The [layout contract](UI-LAYOUT.md) and [bounded plan](UI-LAYOUT-PLAN.md) own
this implementation inside the existing milestone. Actual root HTML/CSS and
the existing operator state/DOM were audited. Independent preflight passes.

- [x] Audit root-site primitives and existing screen/state/action ownership.
- [x] Specify focused preparation/run/recordings/settings pages with purely
  local navigation and preserved native guards, drafts and recovery.
- [x] Implement persistent sequential screens and source-matched styling.
- [x] Qualify rendered flows, actual Pretext, accessibility, edit/scroll
  stability and complete phone interoperability suite; independent review.
- [x] Publish exact reviewed increment and verify Pages/live assets/flows.
- [ ] Installed-phone layout acceptance when the phone is available again.

Phone is unavailable; no device operation is part of this layout increment.
APK11 and the deferred wearer/new-mode/physical-cadence checks remain unchanged.
Overall hardware acceptance stays open.

Reviewed software candidate: full235/235 phone tests pass with zero skips and
the real Java/native prerequisites. The final combined real-app capture suite
passes288states (140new layout +148original),70regression groups and636actual
viewport tiles with no app errors, control occlusions or failed hit tests.
All five viewport/text configurations pass, including320px/200% text. Actual
native renderer50%/touching inset, native two-cycle progress, original pending
receipts, explicit modes, acceptance invalidation, next-session identity,
preview-only100ms/singleflight reads and six-poll editor/scroll stability pass.
Task, whole-change and fixture reviews pass after the recorded corrections.
Entry/style are `operator-layout-r1`, UI45, local resolver1; setup8/display5 and
APK policy11 are unchanged. No phone/device
operation, native/APK rebuild, audio change or Meta upload occurred.

Publication verified: [PR24](https://github.com/GeorgeFejer91/viscereality.org/pull/24)
ordinary-merges reviewed source `55191045607255305b81a3023654a3af4be704d8` at
`4a351ecd3916ed50221499b81cac2a4d6e69a8c2`, with identical tree
`35f2820500a9e8de7cdd4b69bb65f1008fa0f785`. Exact
[Pages43896368744](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43896368744)
succeeds. All34 live-file comparisons match Git; public390px/1440px loading,
new module/cache graph, local settings/return, install help, cold legacy route,
no-error and width checks pass. The runtime is live. This evidence follow-up
changes notes and the context checker, which now recognizes the additional
screen class/attribute order and requires the local layout module/contract.
Its remote/context gate passes; physical phone refresh/acceptance stays pending.

### Reusable ambient background music — 2026-11-16

The request is for ambient background music during a 10–15-minute session.
The first choice is Kevin MacLeod's [Wisps of Whorls](https://incompetech.com/music/royalty-free/index.html?Search=Search&isrc=USUAN1200082),
a 14:16 electric-piano and synth track. [New Direction](https://incompetech.com/music/royalty-free/index.html?Search=Search&isrc=USUAN1100677)
is an 11:20 synth alternative. The composer lists both tracks as
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and both original
MP3 download endpoints returned HTTP 200. Credit the title, composer, source
and license in the distribution, and disclose any edits.

The options are ready for listening and selection. The next step if a track
is selected for participant use is to audition it under the breathing cues
on the target device and record the selected file and playback level.
The smartphone and Quest rehearsal checklist remains active. The preceding
audio review was committed to main as `1d88415`.

### Physical display correction — 2026-11-16

The user postpones fixed/adaptive mode testing and reports that yellow radius
and percentage still step even under Automatic. Keep the phone user-controlled;
USB access is read-only. No calibration, new-mode switch or recording is
requested during this diagnosis. The initial software-only display acceptance
is reopened for this observed behavior.

- [x] Install code11 while confirmed idle, verify exact hash/version/retained UID,
  accepted MAIN/LAUNCHER and controller-only startup; user confirms phone version.
- [x] Verify the actual standalone Pixel loaded UI43/setup8/display4 and current
  CSS. Blue126px/yellow122px at equal progress confirms the intended2px-per-side
  inset; this is subtle by design, not an overlapping-ring/cache failure.
- [x] Capture18s of real read-only network/rAF evidence:89 statuses,1079 frames,
  median request54ms, native renderer age6ms, no stale frames. Repeated150–220ms
  actual-radius plateaus occur while the phase moves. Early/late responses
  truncate chained composite blends at their oldest endpoint expiry.
- [x] RED/GREEN against captured jitter; bounded adjacent observed-sample playout
  with a finite reserve, original500ms endpoint expiry and no future estimate.
- [x] Independent corrective review and full223/223 zero-skip phone gate.
- [x] Publish PWA-only fix and exact Pages/live plus public phone/desktop checks.
- [ ] Repeat read-only physical cadence check after phone availability/user refresh.
  APK11 is unchanged; user defers new-mode testing.

Corrective source checkpoint: 25 focused display tests pass, including the
anonymous captured-timing fixture and exact continuous interpolation where a
fresh observed bracket exists. The old 5Hz trace replay improves interior ramp
stops from34.51%/290ms maximum to7.90%/140ms; a 10Hz captured-age model gives
1.34%/80ms. These are offline/model results, not a physical fix acceptance.
Preview reads are now100ms, with44status reads/four seconds, ordinary4ECG and
4receipts, at most one in flight, and no hidden/outside-preview acceleration.
Source-time playout is bounded280ms; strict500ms endpoint freshness remains.
Rare unsupported gaps still clamp/freeze. Corrective spec/quality review and full223/223 zero-skip gate pass.
The correction is published; same-phone confirmation remains pending. APK11
stays unchanged.

Publication verified: [PR22](https://github.com/GeorgeFejer91/viscereality.org/pull/22)
merged at `5b602970a37f14a4659dd5a9232ca3a42736287c`; exact
[Pages43877648195](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43877648195)
succeeded. All27 live files match Git, and public390px/1440px loading, policy,
install-help interaction, no-error/overflow and remote-context gates pass. Entry
is `actual-playout-r2`, UI44 and display5; CSS and native code11 are unchanged.
Concurrent origin documentation d92cb2e was retained by ordinary merge.

The user says the phone is no longer available. Actual new-100ms physical
smoothness is NOT yet confirmed. Read-only observers ended, the owned9226 CDP
forward is removed/absent, and phone/port leases are released. Do not request
further phone interaction or mode/calibration readiness until the user resumes.
When available, user refreshes the PWA and parent compares the same read-only
moving-ramp timing/perception; preserve the user's control. Overall milestone
stays open; no new mode/calibration/recording was initiated.

Private evidence: `2026-11-16-yellow-diagnosis`. The display implementer owns
live-breath-display.js and focused tests; parent owns phone observation, cache,
evidence, remote reconciliation/publication and cleanup. Preserve concurrent
origin/main documentation commit d92cb2e during reconciliation. New-mode and
full-main physical acceptance stay open for later.


### Smooth actual-radius display and selectable Polar state modes — 2026-11-16

User requests smooth yellow actual-circle/percentage updates, an inner yellow
stroke touching the blue stroke at equal progress, and PWA selection of the
existing pretrained fixed-axis and adaptively adjusting tilt-state modes.
Continue under standing authorized implementation/publication; retain the
participant-calibrated mode, Controller fallback, Automatic parked default,
single audio clock and 900-second main. No additional approval loop.

- [x] Reproduce actual-display cadence/freshness at the real polling interval;
  repair bounded observed-sample blending and shared circle/percentage animation.
  Do not invent future radius data or extend actual-sample freshness silently.
- [x] Give equal-progress strokes exact touching geometry across responsive
  sizes, while percentages retain the same physical 0–100% meaning.
- [x] Add authenticated native setup selection for pretrained fixed and adaptive
  tilt classifiers, with typed receipts, readiness/source-generation fences,
  recording provenance and unchanged audio/retained-state integration.
- [x] Expose confirmed selection and capability-aware mode controls in the PWA;
  show participant calibration only where applicable. Old APKs cannot claim
  the new choices. Preserve scroll/focus and actual Pretext measurement.
- [x] Focused RED/GREEN, task spec/quality and combined integration review;
  native/phone host gates and narrow/200% rendered checks.
- [x] Bind cumulative source, package/inspect local code 11 and publish exact
  Pages/live with public phone/desktop checks.
- [x] Install while idle, verify exact APK bytes/version/UID and trusted launch.
- [ ] Qualify affected device behavior after fresh wearer Ready; user keeps
  phone control. No Meta upload.

Task ownership: display implementation owns live-breath-display.js, display
CSS and display tests; native implementation owns selected native adapter,
setup/core/gateway and host tests; phone-mode implementation owns setup/UI/HTML
and mode tests. Parent owns shared evidence, resource coordination, source
binding, release policy/cache bumps, packaging, publication and final checks.
Use this checklist as the single plan; private task/review evidence is under
the 2026-11-16 display-and-modes validation folder. No competing GSD plan.

Software progress: display and phone task reviews pass independently. All 218
phone tests pass with zero skips and fresh actual Java pairing interoperability
(369 JCE/state checks). Observed-cadence blending expires both endpoints within
500ms, snaps after stale gaps, and shares actual progress with the integer
percentage. Preview-only 200ms status reads coalesce without accelerating the
ordinary receipt/ECG/export/connection work. Twenty display tests, 25 stroke
geometry cases and 26 actual Pretext states/seven stability regressions pass.
Native mode selection, shared-loop continuity, recording provenance and full-main
behavior pass task and final integration review. Native serial library gate is
856 passes/five intentional ignores; isolated actual renderer publication passes,
as do six Java/native peer scenarios, 45 actual HTTP/auth/gateway assertions and
selected API36 compilation. The earlier non-library host cfg diagnostic fails
on a preexisting Android-only bridge and is not counted as a pass. Native source
`b1ba2c2b750dbff70488e351c4233b7cd62b73d5` is clean; the public cumulative patch
reconstructs tree `9a482a67f80342cb612b50112996488fac422c3a`, SHA-256
`6b1180654ba1fadda6bb88a2bbf677c3c2d0c2b3dca1ca8ae4a5ee2289c150ef`.
Selected code-11 Android compilation and signed-APK inspection pass with the
unchanged private particle inputs and retained signer. See the APK log for exact
bytes/source/build identity. Build and installation leases are released. The
current software is published and code11 is installed; actual phone/LAN timing
and wearer acceptance remain open. No calibration was started by the agent.

Publication verified: [PR 20](https://github.com/GeorgeFejer91/viscereality.org/pull/20)
merged through the secondary account at
`e9e8048d39843865a6db95697e0ecadba2524e72`; exact
[Pages run 43848286245](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43848286245)
succeeded. All 27 checked live files match Git, including the code-11 release
policy and new cache versions. Live390px/1440px rendering, actual policy fetch,
install-help interaction and source/remote context gates pass. The live policy
requires code11/0.2.10-dev; new preset choices also require explicit capability.

Installed checkpoint, 2026-11-16: the user confirmed no recording and requested
installation/launch. Exact code11 APK bytes match the inspected SHA-256; Package
Manager reports0.2.10-dev/code11, UID10159 retained from code10. The Wi-Fi route
was freshly bound to the same canonical Quest/current boot without restarting
shared ADB. Normal MAIN/LAUNCHER cold start is accepted; the target activity owns
Android focus. Startup reports controller-only true, live hand extension and
simultaneous tracking false. Initial driver value is zero; no agent calibration
or recording command was sent. Automatic H10 preferred-candidate admission is
observed; stream quality/readiness needs the phone/wearer check. Location staysON.
The bounded observer and its ADB child are stopped; all owned leases released.

The user refreshed/reconnected the phone, took control and confirmed
“0.2.10-dev / code11” on its version display. This is user-reported installed-phone
readback, not independent resource capture or physical cadence qualification.
Fresh wearer Ready is requested before testing the fixed/dynamic modes, smooth
yellow percentage/circle, exact stroke contact, personal-fit and Controller
fallback continuity. The user retains phone control. Full-main physical
acceptance and effective kiosk qualification remain open. No Meta upload.

Local sync verification, 2026-11-16: the clean sparse checkout matched
`b01cb3e277f99e63bea1dce6e83f0b77d04cbee6` (PR 21) and
[Pages run 43849311195](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43849311195).
Fifteen live files matched Git. The 390px and 1440px live views passed Pretext,
overflow, install-help focus and code-11 policy checks. The remote context
check and all six scenario pairs passed. The local Node run passed 213 tests
with five native Java interoperability cases skipped because the harness
was unavailable on this PC. The original checkout has an inaccessible Git
index and config. The synced sparse checkout uses separate local Git metadata.


### Publication resumed through secondary account — 2026-11-16

The user explicitly requests merging PR 18 through the secondary Codex account.
That account completed the ordinary two-parent merge at
`f79800de0836cc2d8b37c4f75d2592df697b92ec`, preserving reviewed head
`bd0af7db9063533987e1b83b137086d2a5afe87c`. The earlier access blocker below
is historical and resolved for this release. Exact Pages run 43838565290 and all
27 checked live files pass, with public phone/desktop renders. Installed-phone refresh and
remaining device acceptance are separate and still open; user operates the phone.

Quest 2 Wi-Fi ADB was restored after today's reboot and verified against the
same canonical device and current boot over a shell connection. Code 10 remains
installed. No shared daemon restart or app launch was performed. Its setup leases
were released; retain the requested network transport and Location ON. Obtain
fresh wearer Ready before any calibration or recording.

### Automatic connection and clear operator controls — 2026-11-15

The participant accepts the current calibration as good enough. Preserve its
learner, retained breath-state integration and single PCM clock. The next user
request covers five existing flows; continue under the standing implementation
and publication authorization without an additional approval loop.

- [x] Trace/repair cold-start H10 discovery and sustained bounded scan/connect
  retries in the native owner; retain explicit Stop, permission and multi-sensor
  selection barriers. Establish/retry both required streams after connection.
- [x] Replace the obscure idle acknowledgement with visible, state-specific
  failure/recovery guidance. Distinguish local uncertainty reconciliation from
  an actual headset/app reset; never replay unknown mutations automatically.
- [x] Display the confirmed active input and pending switch explicitly. In
  Controller mode hide Polar calibration progress/failure controls; provide an
  explicit route back to Polar setup/calibration while the shared loop continues.
- [x] Overlay labelled, differently styled blue desired/gold actual radius
  indicators using fresh Quest-applied renderer progress. Trace existing fields
  before choosing the narrow additive telemetry contract; never substitute the
  learned sensor coordinate or desired phase for actual renderer output.
- [x] Trace effective soft-kiosk activation and expose truthful status, missing
  prerequisites and supported controls in PWA settings. Preserve platform prompt
  and permission boundaries; do not claim a requested guard is effective.
- [x] Add authenticated installed APK package/version/build identity, a public
  per-package required-version policy and honest unknown/update indicators.
- [x] Run RED/GREEN regression, independent spec/quality review, narrow/200%
  Pretext and stable-scroll checks, source reconstruction, signed local APK
  build and inspection. No Meta upload.
- [x] Install/read back code 10 while idle and verify trusted cold launch,
  controller-only input and automatic H10 connection/ACC+ECG stream starts.
- [x] Publish the matching PWA through exact Pages/live and public phone/
  desktop checks.
- [ ] Verify refreshed installed-phone resources. Qualify remaining hardware
  flows after fresh wearer Ready; full-main acceptance remains separately open.

Implementation order: native bootstrap/retry/kiosk and actual-radius contract;
phone receipt/mode/dual-display/settings; integration/review; package/publication
and attended acceptance. Native and phone changes have separate file ownership,
with this project state as the single plan/evidence owner. PR 18 is now merged; see the current publication checkpoint above.

Software checkpoint, 2026-11-16: all requested source changes pass independent
task and final cross-boundary review. Final phone gate is 203/203, zero skips;
native 846 passes with the existing serial-runner requirement and five intentional
ignores, plus the isolated real renderer/global-producer case passing separately.
Actual source callback/receipt and non-recording GPU-publication RED/GREEN, API36
and gateway/auth/identity gates pass. The local same-key code-10 APK is built and
inspected; see the [code-10 record](../quest/apks/README.md#029-dev--operator-controls-and-installed-build-identity-2026-11-16).
Native `695ef0d4a8c2a482b6ab3a6d37b21dee64f94770` reconstructs tree
`ec3ad8be7eabe2b8878f58ce7cfc50e41ff78856`; selected Android compilation and
packaged build identity match it. The prior cold-start failure is traced to the
root's malformed bare-component test launch; guarded startup remains unchanged.
Current code 9 and new code 10 do not register the optional boot receiver, so
true power-on autostart is not enabled or qualified. Controller-only tracking
remains selected. APK update policy reports compatibility; it does not install
updates or guarantee Meta update timing before launch.

Installed checkpoint, 2026-11-16: the user confirmed no recording and authorized
installation/launch. Exact installed bytes match the code-10 hash, UID 10159 is
retained and trusted MAIN/LAUNCHER admission succeeds. Runtime reports controller
only, no live hand extension, no simultaneous tracking and driver value zero.
Without a phone Connect action, the app finds the H10, connects and receives ACC
and ECG stream-start acknowledgements. This establishes automatic startup, not
signal quality or sustained absent-sensor retry acceptance. The kiosk service
record exists; effective protection still requires authenticated observation.
Location remains ON. The optional boot receiver remains absent.

The user ended today's testing and is shutting down the Quest. No new calibration
or recording was started on code 10. The finite startup observer has stopped its
owned ADB child and the exact Quest lease is released. The user continues to own
phone operation. Resume with fresh wearer Ready; check APK identity/current
policy, actual-radius display, confirmed Controller UI and effective kiosk, then
remaining retry/fallback/full-main acceptance. No Meta upload.

Publication verified, 2026-11-16: the secondary Codex account successfully
performed the ordinary two-parent merge of
[PR 18](https://github.com/GeorgeFejer91/viscereality.org/pull/18), exact reviewed
head `bd0af7db9063533987e1b83b137086d2a5afe87c`, at merge commit
`f79800de0836cc2d8b37c4f75d2592df697b92ec`. Exact
[Pages run 43838565290](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43838565290)
succeeded. All 27 checked live files match Git blobs, including entry/cache
versions, the APK release policy, kiosk/version modules and cumulative source.
Live 390px phone and 1440px desktop checks pass with ready Pretext, no horizontal
overflow, no script or failed-app requests, and working install-help open/close.
The source/remote context gate passes. The initial private smoke assertion read
the async policy-response map too early; waiting for its actual response fixes
the test, with no product change. Browser plugin is unavailable; existing regular
Playwright is the recorded validation route. Public screenshots are inspected.
No permissions/protections, selected model, private payload or Meta upload changed.

The user returned for testing. Exact Quest-2 Wi-Fi ADB is restored and verified
against the same canonical serial and current boot over the network shell;
code 10 remains installed. Shared ADB was not restarted. Transport/headset setup
leases are released. This verifies transport, not immersive readiness or an
experiment. The user controls the phone; a refreshed installed-phone version
readback is requested. Next verify APK 0.2.9-dev/code 10, automatic H10 readiness,
actual-radius overlay, confirmed Controller UI and effective kiosk; obtain fresh
wearer Ready before calibration. Sustained retry/fallback and full-main acceptance
remain open. The overall milestone stays open.

### Explicit audio retry recovery — 2026-11-15

- [x] Capture the code-8 failed attempt and distinguish the proven retry dead
  end from the unproven initial player/head/publication stall trigger.
- [x] Implement/review native admission, fresh track/generation, actual progress
  and post-claim input gates, bounded diagnostics and typed gateway receipts.
- [x] Implement/review phone pending/uncertain receipt reconciliation, native
  eligibility and clock-health presentation with stable DOM/Pretext.
- [x] Verify native842/four ignored, real Java/native and HTTP/auth integration,
  phone182/zero skips, rendering/scroll, exact source tree and API36 compilation.
- [x] Build, inspect and install same-key local Developer 0.2.8-dev/code 9.
- [x] Publish the combined phone/source/record increment; verify exact Pages
  and live URLs.
- [x] Refresh the installed phone app and verify the new resources while the
  user retains control of that device.
- [ ] Obtain fresh wearer Ready; test audible/advancing loop across holds and
  wraps, calibration/tracking/retry/Controller fallback and the full main run.

Native `9afdcf0ef2cf05569e7ccc67c31f711f0e2fdd87`, tree
`46593d96e655dd515cd2bfbb39d696eae6eed2b3`, and phone candidate
`0ce7eda855fe5f30f86322bc0680cfaf5e93a6a9` pass independent spec/quality review.
Native authority remains actual audio frames. An explicit unhealthy preparation
retry reserves one idempotent recovery before Java mutates transport; fresh
forward frames and post-claim input precede a new full 32-second window.
Healthy retry continuity, strict clock guards, main/recording and retired-owner
fences remain. The original stall cause remains unproven.

The [code-9 record](../quest/apks/README.md#028-dev--explicit-audio-clock-recovery-and-diagnostics-2026-11-15)
binds the inspected APK. Installed bytes, UID, grants and persistent identity
match. Cold start confirms controllers only, no live hand extension, no
simultaneous mode, panel closed, input zero and a ready gateway. Location is ON;
Wi-Fi ADB is retained. Startup observers and owned leases are released. The user
controls the phone and removed the Polar after code 8; no new calibration is
authorized until fresh Ready. No Meta upload. Overall acceptance remains open.

The combined release merged in [PR 17](https://github.com/GeorgeFejer91/viscereality.org/pull/17)
at `73d9d896ecb3d44da304b4c9b52009ebc1dc7a8b`; exact
[Pages run 43756359679](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43756359679)
succeeded. All 24 checked live files match that Git revision, including the
versioned phone modules and cumulative native patch. Public phone/desktop checks
at 390/1440px report ready Pretext, no horizontal overflow, script errors or
failed app requests. The remote context gate passes. That runtime release is
synchronized to GitHub and live; the APK stays local. Installed-phone refresh and
fresh wearer testing are the next actions, with original stall causation open.

The documentation-only evidence follow-up is pushed in
[PR 18](https://github.com/GeorgeFejer91/viscereality.org/pull/18), still open at
handoff. The authorized secondary-account merge is blocked by repeated selected
model capacity errors; the primary GitHub integration returns 403 for merge.
No permission or model change was requested. Resume its normal two-parent merge,
then exact Pages/live checks when that account is available. This blocks only
publication of these follow-up notes; PR 17's runtime is already deployed and
verified. Owned resources were released at that earlier handoff.

The user subsequently supplied fresh code-9 wearer Ready and retained phone
control. Read-only inspection confirms the installed standalone PWA loaded
the new app, UI39, gateway11, rehearsal25 and setup6; a separate Chrome tab
still has the older release. The first scan/select completed, but its connect
request ended `outcome_unknown` with H10 disconnected. The user reported that
the grey controls and missing timely failure explanation made progress unclear.
After checked idle acknowledgement, a second scan/select/connect succeeded;
ECG and ACC became ready, with native automatic acquisition enabled. No
calibration had started at that checkpoint; audio was parked at frame zero.
Resolve the cold first-connect failure and its unclear recovery UI as an open
issue.

The subsequent code-9 calibration failed `clock_stalled` at frame zero. Real
polling/dispatch continued, but Android reported PLAYING with raw head zero,
no timestamp and no consumed frames. Explicit retries replaced actual players
through generation 6 and returned typed `audio-recovery-timeout`; Controller
selection also failed after the user closed an observed menu. No full window
or calibrated tracking was reached. This is physical recovery failure, not an
accepted code-9 calibration. The user was told the Polar could be removed.

A separate local diagnostic APK compared identical silent PCM in primed STREAM
and STATIC modes, with audio focus granted before/after creation; neither
advanced. Horizon service counters stayed fixed across captures despite queued
active sources and full, unmuted media volume. This supports a shared playback
problem, with silent-probe/foreground limitations; the user reports system
sounds work, so a universal audio failure is not established. The probe was
removed, its private evidence retained, and its build lease released. Product
source and installed code-9 bytes were unchanged.

An attended Quest-2 reboot completed. Wi-Fi ADB was restored and verified against
the same serial and new boot identity without restarting the shared host ADB
daemon; Location remains ON. Pixel observation ended when its USB was removed,
and its port/device leases were released. Fresh post-boot wearer Ready was
received for audio only, with Polar off. Atmungsraum reopened parked/silent and
the headset output counter now advances. User-operated Controller audio check
was pending at that checkpoint; fresh Polar Ready was required before another
calibration. No Meta upload or main experiment occurred.

The user then connected Polar and started calibration themselves, reporting
that it worked well after reboot. Actual native readback now selects a ready
calibrated Polar lineage and measured inhale/exhale/hold states. Hold values
remain unchanged through observed unavailable/unconfirmed samples, consistent
with the retained-state repair. Tracking satisfaction is being requested;
this is not yet qualification of repeated fits, Controller fallback or a full
900-second main run. No replacement APK or speculative audio-adapter change
was needed for that successful post-boot calibration. The intermittent playback
trigger and cold-connect UI issue remain open. The finite observer ended with
its ADB child stopped; all owned device, port and build leases are released,
with no remaining forwards. The app remains running for user-operated tracking
assessment; Location is ON and Wi-Fi ADB is verified. PR 18's documentation merge
remains blocked by the secondary account's selected-model capacity error.

### Retained Polar breath state — 2026-11-15

- [x] Reproduce the radius reset through actual calibration, batched 200 Hz
  samples, jittered receipts and frame-time driver updates.
- [x] Retain the latest sensor-confirmed inhale/exhale/hold through unavailable
  or invalid samples; preserve actual source/audio/model/configuration barriers.
- [x] Review the source and verify native833 (three ignored), phone164 and exact
  cumulative patch/tree reconstruction.
- [x] Build, inspect and install same-key local Developer 0.2.7-dev/code 8.
- [x] Publish the source/record and verify exact Pages deployment and live bytes.
- [ ] Diagnose the code-8 audio-clock stall and repair explicit retry recovery;
  retain strict frame authority and require a new full calibration attempt.
- [ ] Qualify smooth tracking after fresh wearer Ready, retry, Controller
  fallback and the full 900-second main run.

Reviewed native `304be3ccf20e960a2a4bac7dad8ea9ca9bce17b1` reconstructs tree
`2b702ebeda4b69faadc97c3716f9db956d759a1f`. Two scoped review corrections
preserve a newly confirmed state before an invalid packet tail, and fence old
confirmations at driver configuration/activation retirement. Both are cleared;
spec and code-quality review pass. The new confirmation cursor remains monotonic
across classifier re-prime; unavailable metadata cannot authorize a phase or
mark input ready. The existing integrator/rates, classifier, learner, raw-data
freshness and recording admission remain unchanged. Fresh APK assembly and actual
session closure, manifest, 89 assets, signer and alignment checks pass. The
[code-8 record](../quest/apks/README.md#027-dev--retain-confirmed-polar-breath-state-2026-11-15)
binds the artifact. Exact installed bytes match; UID, grants and persistent
identity are retained. Cold startup confirms controllers only, no hand extension,
no simultaneous mode, app panel closed and gateway ready. Observed input is zero;
configured inhale/exhale rates are 0.25 per second. Location remains ON and Wi-Fi
ADB is retained. Initial startup observers and owned leases were released.
The source/record merged in [PR 15](https://github.com/GeorgeFejer91/viscereality.org/pull/15)
at `b735f196b90ebe4d7dcccf0c446baf6886105849`; exact
[Pages run 43745201300](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43745201300)
succeeded. All 24 checked live files match Git and the remote context gate passes.
The user supplied fresh code-8 Ready and explicitly controlled the phone. The
first H10 link request became uncertain; observed Bluetooth was disconnected,
so the user acknowledged it and began one new scan. Native then connected H10
and automatically established both ACC and ECG. The attended calibration advanced
to 270,950 frames (6.143991 audio seconds), accepted 150 training points and masked
four motion points, then failed `clock_stalled`. Status kept advertising playing
but the frame remained fixed for at least 20.976 seconds. Actual retry requests
were rejected `clock-stale`. Calibrated Polar tracking was never reached.

Read-only Java/native investigations establish a retry dead end: native requires
a healthy clock before admission, while the already-started active Java owner
does not reprepare/restart on retry. Native clock ingress itself is not latched;
real advancing observations would restore clock health while leaving the failed
attempt failed. The initial trigger is unproven without raw poll/head/dispatch
history. A zero-volume stream and a silent PCM hold are leads, not demonstrated
causes. The Java player/contract are unchanged from code 7's successful attempt.

The user removed the Polar and requested diagnosis. The app is stopped; native
observer, phone helper/forward and all owned leases are released. Location stays
ON and Wi-Fi ADB is retained. Fresh Ready is required before another calibration.
The next bounded repair is explicit pre-main recovery into a fresh authoritative
audio generation plus bounded clock diagnostics; normal healthy retries keep the
same loop. The source design is accepted and implementation is under verification.
Native admits one idempotent recovery reservation before Java replaces transport;
fresh forward frames and post-claim input must precede a new complete calibration.
Main/recording and retired-owner boundaries remain. The real HTTP gateway now
preserves typed native pending/terminal receipts, with current-controller and
route/authentication checks. Diagnostics separate intent, polling, dispatch and
actual head advancement; the original stall trigger remains unproven.

The local PWA candidate is `0ce7eda855fe5f30f86322bc0680cfaf5e93a6a9` (unpublished).
Its 182 tests, 12 recovery renders, 13 receipt-fence renders and 3×6 scroll/edit
checks pass. A scoped review correction preserves the original ID after a lost
admission response: same-actor audio-generation changes offer read-only recheck;
only proven actor retirement permits acknowledgement. The verified code-9
source/build/install above supersedes that implementation checkpoint. Combined
publication and fresh wearer qualification remain open. The overall milestone
remains open; no Meta upload has occurred.

### Participant-led calibration and controller-only app — 2026-11-15

- [x] Remove ideal-waveform compliance rejection without adding a recurrence
  gate. Assume cue adherence and learn the participant's measured tilt mapping.
- [x] Retain usable-data, coverage, source/clock and finite/range guards; disclose
  the existing 0.02–0.50 measured-span engineering limitation.
- [x] Select controller-only app capabilities and verify the actual locked
  manifest plus running input/hand-extension flags.
- [x] Build and install same-key Developer 0.2.6-dev/code 7 with exact installed
  byte-hash readback and retained UID/grants/persistent identity.
- [x] Publish the final code-7 source record and verify exact Pages/live bytes.
- [x] Run a fresh attended personal calibration: both complete cycles collected,
  800/800 training points accepted, automatic switch to calibrated Polar tilt.
- [x] Correct the source of abrupt radius changes in the reviewed native path;
  retain the last valid breath state through unavailable/invalid samples and
  continue its frame-time ramp until valid data establishes a new state.
- [ ] Repeat participant tracking acceptance, then test retry/Controller fallback/main.

The [code-7 record](../quest/apks/README.md#026-dev--participant-led-calibration-and-controllers-only-2026-11-15)
binds native `50eaa1ac55eb9161879c48916da82894cfaff9af`, tree
`0d6e505055a77611b2f46a4b50d50d32ce8bc95d`. Core50, setup72 and native824 checks
pass with three ignored; controller-only resolver/compiler and 32 panel checks,
actual non-resolve builder regression and independent reviews pass. The first
build exposed a legacy simultaneous-only guard; the reviewed fix preserves all
other original-renderer checks. Fresh assembly, package/source proof and install
succeed. Actual startup confirms controllers only, no live hand extension, no
simultaneous mode and a closed app panel. The gateway is ready. The initial
unworn black capture did not prove rendering; the wearer later dismissed a
Quest prompt and confirmed the actual stimulus.

Location remains ON by explicit user instruction; requested Wi-Fi ADB remains
available. No Meta upload has started. The overall goal is open.

[PR 14](https://github.com/GeorgeFejer91/viscereality.org/pull/14) merged as
`2fbf5306cb8876bce52c8b87a70e838ca41c7c03`; exact
[Pages run 43737821830](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43737821830)
succeeded. All 24 checked live files match Git and the remote context gate passes.
The user then supplied fresh code-7 readiness and USB Pixel access. The physical
phone was refreshed; exact new brand CSS/app and setup-module resource URLs were
verified on that phone. The attended code-7 attempt completed all 1,411,200 frames
with 800/800 accepted training points and no quality masks, then selected Polar
tilt without restarting the audio generation. Participant satisfaction failed:
the sphere visibly jumped rather than following the expected smooth state ramp.
Code 7's transient Waiting path reset the radius; sparse private readback
contains 32 not-ready observations after successful calibration. The investigation
led to the code-8 repair above. The code-7 app was stopped, its phone helper/
forward and owned leases released, and Location retained ON.
The wearer may remove the strap; require fresh Ready for another calibration.
The user explicitly corrected the proposed radius freeze: hold the last valid
breath state instead. Inhale/exhale must keep changing the radius through sample
gaps; hold must remain still. Genuine lifecycle barriers still clear the state.
The continuity regression is reproduced and its frame-time repair passes the
native library suite (829 tests, three ignored) and phone suite (164 tests).
Independent review found a same-packet edge: a newly confirmed state can be
erased by a later masked sample before rendering. The fix must retain the latest
confirmed sensor classification separately from current sample/classifier
availability. Fix round 1 is implemented at native
`d9b5f05082835bcb8c0605bb5758f8e02271999e`, with 831 native library passes and
three ignored. Real Runtime regressions cover both Hold and Exhale confirmed
before a masked batch tail. Scoped re-review accepts that repair but found a
new cache-retirement edge: config/activation reset can reload a pre-barrier
confirmed state from unavailable Runtime output. Fix round 2 is frozen at
`304be3ccf20e960a2a4bac7dad8ea9ca9bce17b1`: monotonic confirmation revisions
and a driver fence retire pre-barrier confirmations while admitting newer valid
ones, including valid data followed by an invalid batch tail. Runtime29 and the
full native library suite (833 passed, three ignored) pass. The cumulative patch
reconstructs tree `2b702ebeda4b69faadc97c3716f9db956d759a1f`. Scoped re-review
is clear; build/install and wearer acceptance are tracked above.

### Core-site visual identity — 2026-11-15

The user requests that the PWA visually match the core Viscereality website.
The prepared style uses its white VR figure, bold centered Arial headings,
black/white surfaces, thin frames, blue actions and gold accents. Product controls,
real Pretext, scroll/edit stability and percentage display isolation are retained.
The final rendered sweep passes 132 screens and 30 regression records with zero
app errors, overflow, unmeasured labels or undersized targets. Brand no-fit was
corrected and independent visual/source review is clear. The style is published
and the physical Pixel's release resources were verified during the code-7 test.

[PR 13](https://github.com/GeorgeFejer91/viscereality.org/pull/13) merged as
`0a3451656351d00391b6ca1d9d55bd2805b1d960`; exact
[Pages run 43736737421](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43736737421)
succeeded. All 24 checked live files match Git. Public-URL phone/desktop smoke
checks pass with zero app errors. Installed-phone refresh and exact resource
version readback subsequently passed during the attended code-7 test.

### Parked startup and full-window calibration repair — 2026-11-15

Latest correction: assume guided breathing during calibration, learn the person's
tilt movement and let their post-calibration test decide satisfaction. Remove
ideal-waveform compliance rejection without adding a recurrence test. The app
must also be controller-only to prevent hand/controller auto-switch interruptions.
These source changes are implemented in the code-7 increment above.

The attended code-5 test did not collect calibration data. User attempts failed
as `reordered_sample`; one assistant-operated retry, explicitly authorized by
the user's “try yourself”, failed immediately as `excessive_motion`. Neither an
accepted command nor Automatic audio proves collection: completed frames never
advanced. The user may remove the strap while this repair is built; require a
fresh Ready before another physical calibration.

The latest accepted behavior is Automatic selected but silent and held at minimum
expansion on load. Only an explicit calibration or Controller selection starts
the unlimited native loop. Calibration assesses the entire two-cycle window,
masking bounded noisy intervals and fitting only after sufficient usable phase
coverage; a short movement does not fail the attempt immediately. Real source
retirement and an actual audio pause during collection still invalidate it.
Connected Polar acquisition automatically establishes and recovers both required
streams. The phone percentage shares the circle's bounded display behavior.

- [x] Reproduce callback-derived host reordering and noise-sensitive motion
  rejection in focused probes without changing original raw records.
- [x] Verify parked Java audio startup and explicit first-start behavior with
  host owner/backend checks and actual Android API-36 compilation.
- [x] Review and verify source-bound estimated alignment, whole-window quality,
  automatic stream recovery and the matching phone controls.
- [x] Build, verify and install the next same-key local Developer APK.
- [ ] Publish its verified public source record through exact Pages/live checks.
- [ ] Resume a fresh attended test, inspect the actual ACC frame format, confirm
  advancing calibration progress, then qualify fit, retry and Controller fallback.
- [ ] Replace ideal-waveform rejection with participant-decided calibration
  acceptance, retaining technical/data-quality guards and usable signal limits.
- [ ] Select controller-only app input and verify the locked manifest/runtime.

Code 6 is now installed with exact byte-hash readback and retained app UID,
grants and persistent identity. System Location remains OFF and this task's
build/headset leases and startup observer were released.
Wi-Fi ADB remains enabled as requested. The user has authorized assistant phone
operation for this diagnosis; the earlier read-only preference is superseded
within that scope. Meta upload remains on hold. The overall milestone stays open.

The phone increment was published separately in
[PR 11](https://github.com/GeorgeFejer91/viscereality.org/pull/11), normal merge
`578d330863ce0d3726417e5ee0c14d01df636393`. The exact
[Pages run 43729861453](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43729861453)
succeeded. All 23 checked live files match committed Git bytes and actual
public-URL phone/desktop smoke checks pass. The installed Pixel is currently
disconnected from USB; its release refresh remains pending with the next
attended headset test. Native timing and acquisition are a separate increment.

The native repair is frozen at `f31156d7832f71b7caeff9ad005c311ea77c0c8e`,
tree `c095a8df92480b6da4c7810ed3332bf3998667c2`. The cumulative patch reconstructs
that exact tree from the pinned upstream. Focused setup tests pass 69; the full
native suite passes 821 with three ignored and no failures. The three selected
profile checks and packaged-default reset check pass with the retained private
visual profile. Independent source reviews are clear. Developer 0.2.5-dev/code 6
passes actual signed-APK closure, manifest, signer, asset and ZIP checks; its
[APK record](../quest/apks/README.md#025-dev--parked-startup-and-full-window-quality-2026-11-15)
binds the artifact and source. Native frames advance with the app panel closed
and a ready gateway. The existing Location handoff leaves Meta system Settings
over VR; fresh wearer silence/minimum and calibration acceptance remain open.

The next attended code-6 attempt actually collected all 1,411,200 frames. Both
H10 streams became ready with automatic recovery active; actual ACC is raw type 1
at 200 Hz. Native parked readback was frame zero, stopped audio and input zero.
Calibration accepted 795/800 training points with no motion/invalid/timing masks,
then failed `poor_guided_motion`. The user confirms following audible cues.
Sparse private readback shows tilt movement in both cycles; the strict ideal
fit is the remaining rejection, so participant-fit acceptance is not passed.
Controller selection was accepted but usable live tracking remains unqualified.
The app was force-stopped for the repair, observers/leases released, and a new
wearer Ready is required before another attempt. The user explicitly requests
leaving Quest Location ON; current mode is 3. Do not restore the earlier OFF state.

[PR 12](https://github.com/GeorgeFejer91/viscereality.org/pull/12) published the
code-6 source record as `e3b4573cc7fb3107912b3959c14660c78ef762de`. Exact
[Pages run 43733535771](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43733535771)
succeeded and all 24 checked live files match Git, including timing provenance.
The phone was actually refreshed and displayed the new parked/quality UI; exact
resource-version readback was interrupted by USB disconnection at test cleanup.

### Stable phone controls and Showcase visuals — 2026-11-15

The user requested one combined update while away: retain scroll and focused
edits during live polling, clarify partial Polar readiness, reuse the Showcase
PWA's stable controls and the website's visual language, smooth the breathing
and ECG displays, and integrate the other chat's Showcase particle/coupling
port. Preparation and the main experiment keep their existing native authority.

- [x] Reproduce the scroll reset: telemetry briefly changed reflow to bounded,
  clamping scroll from 300px to 105px without navigation or a focus change.
- [x] Keep layout and form nodes stable during telemetry; preserve focused edits,
  caret and scroll through repeated status updates.
- [x] Name missing ECG or motion data and offer an explicit action to start
  only the missing streams. Bluetooth connection alone is not readiness.
- [x] Add bounded native-clock breathing presentation and buffered raw ECG
  scrolling, with stale, delayed, replay, pause and background recovery checks.
- [x] Integrate the generic particle adapter on top of the buffered-calibration
  source; stage full private Showcase coupling with experiment tracers and
  the separate pole/retention indicator disabled.
- [x] Publish the verified phone/source update and the updated local APK record.
- [ ] Resume attended Quest/Polar validation after a fresh ready response.

[PR 9](https://github.com/GeorgeFejer91/viscereality.org/pull/9) merged as
`e68452f6467a7b2a838f5f29165e6d382cb38eec`. The exact
[Pages run 43722083291](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43722083291)
succeeded. All 23 checked live runtime/source/document requests matched Git
blobs, including the new cache-versioned presentation modules. Real public-URL
browser smoke checks passed on phone and desktop viewports with no app errors.
The final Node suite passed 156 with zero skips; 112 rendered screens passed.
Developer 0.2.4-dev/code 5 is installed locally and remains withheld from Meta.
User-operated phone refresh and wearer testing are the next actions.

The user operated the phone: control claim, Polar connection and the synthetic
participant form succeeded, and both ECG and motion were received. Calibration
has never started collecting. The user then stepped away; no stale Ready answer
authorizes starting it. Phone interaction remains user-controlled. The temporary
Quest Location toggle was restored to OFF, USB observation was removed and
the corresponding leases were released; Wi-Fi ADB stays available as requested.

The code-4 cold restart did not display Location settings despite complete app
grants. This remains an actual-device blocker; a resolved Android intent and
passing host tests do not prove the settings surface appeared. Code 5 adds
bounded diagnostics. A pre-launch live capture proves the closed-panel callback
and zero missing runtime grants, followed by a Location intent that returns
through Meta's relay with Location still disabled. Earlier post-hoc logs had
already lost the startup interval. Do not claim that startup Location recovery
or fresh-install permission handling has passed.

The main experiment, wearer fit/retry/fallback, physical phone refresh, visual
parity/performance and Meta version-to-version delivery remain open. Keep the
overall milestone open and withhold iterative APKs from Meta until local tests.

### Quest 2 calibration acceptance — 2026-11-15

Current local repair: Developer 0.2.3-dev/code 4 is installed from native
`ff36ec42`, tree `922b085585263bf68aebbd91d3011217aaf09d11`, with exact byte
readback and visible closed-panel VR startup. Keep this test APK local until
the relevant headset checks pass. Calibration has not started; the local
update is ready for a fresh wearer test.

[PR 7](https://github.com/GeorgeFejer91/viscereality.org/pull/7) merged as
`c41c651482eec6781789c1f767c5a2073d9060f5` and deployed successfully in
[Pages run 43714495396](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43714495396).
All 20 checked live runtime/source/document requests matched committed Git
blobs, including the updated cache-versioned imports and native patch. The
remote context gate passed. The APK remains local; Meta still has code 3.

Direct inspection of the USB-connected Pixel 6a showed the phone connected
but waiting for control. Claiming the free lease made the Polar controls
visible. The installed code also required fine/coarse location grants and
enabled system Location services, beyond Bluetooth Scan. The user authorized
the missing app grants for this test. Location was temporarily enabled for
scanning and must be restored afterward. After an explicit acknowledgement of
an uncertain setup result, both ECG and ACC were received. Synthetic participant
ID `Tst_0_N` labels the bench test.

Two fresh breathing-loop Starts were then rejected as `stale-revision`.
Unconsumed idle Automatic ACC batches changed a diagnostic revision eight to
ten times per second. Native `cda6e861` removes that churn while retaining
source-owner CAS barriers. A further probe found that valid buffered history
could fail latest-clock-only alignment. `ff36ec42` adds bounded actual audio
history and final-window evidence; see the
[timing contract](SHARED-BREATHING-SETUP.md#buffered-polar-timing--2026-11-15).

Native `f77e98d3` unifies all three startup permission requesters independently
of panel visibility. Normal attended dialogs request the declared permission
union, followed when needed by one attended Location settings surface. The app
never toggles system Location itself. The phone now distinguishes Location OFF
from missing app grants and preserves already-connected stream controls.

Checks pass: 809 native tests, three ignored and zero failures; 63 focused setup
tests; 131 Node checks with zero skips; 31 permission lifecycle controls,
45 approval UI controls, actual API-36 compilation and 18 targeted rendered
control/Location states. Bounded independent reviews found no remaining blocker.
Actual H10 callback timing, wearer fit/retry/fallback, the full trial and fresh
permission-dialog acceptance remain separate hardware gates.

The user resumed hardware testing with Quest 2, a worn H10 and a Touch
controller. Wi-Fi ADB was requested and verified against the same device and
boot identity before releasing USB transport. Private device evidence stays
outside Git.

The installed Developer 0.2.0-dev/code 1 matched its published SHA-256 but
failed its first cold start with `experiment-session-runtime-threshold-invalid`.
The Java packaged-closure verifier still required a 30-second runtime while
the selected shared-loop profile correctly requested 900 seconds. Native
commit `252f8120` makes this check follow the strict boolean selection in the
feature lock: selected is exactly 900 seconds; legacy remains 30 seconds.
The outer legacy condition records and all asset/hash checks are preserved.
A host regression reproduced the failure before the fix and passed afterward,
including actual signed-APK assets. Assembly now requires that probe before
reporting success. This probe executes host-compiled production Java against
APK assets; physical startup remains a separate required check.

Code 2 installed with exact byte/hash readback and reached foreground OpenXR
rendering at approximately 72 frames/s. All 116 Node checks and the Java shell/
closure policy suite passed. Its fresh release package then exposed a separate
first-use gap: Android denied hardware-serial access, there was no provisioned
serial, and discovery correctly refused to advertise incomplete metadata.
The old lab setup had already been provisioned. Bluetooth permissions alone
cannot populate identity or trust a new phone/package pairing.

The approved permanent onboarding uses read-only discovery to any nearby phone; saved phones
authenticate, while new phones request pairing. Phone and headset show the
same code and each confirms with a button; no headset keyboard. Retain
multiple distinct phone memberships, individual revocation, and one active
experiment controller with explicit handover. Discoverability is not authority
to control or read recordings. General phone enrollment must not require a
laptop, USB or a generated QR. The next Developer APK selects this route and
closed startup; its optional shell/DUMP machine bootstrap stays unselected.
The headset acceptance panel opens only for a real, verified phone pairing
request, never at app startup. Allow, Reject, cancellation and expiry dismiss
the prompt-owned opening; manual triple-B panel access remains available.
Normal Android permission and Bluetooth power dialogs remain separate.
See the exact [pairing and control contract](PHONE-PAIRING.md).

- [x] Implement independent persistent memberships, revocation and one active
  controller, with selected read-only discovery and closed startup.
- [x] Implement actual wearer code comparison and Allow/Reject controls.
- [x] Resolve the reviewed commitment/reveal protocol and wire-encoding
  defects; prove a complete WebCrypto-to-production-Java pairing flow.
- [x] Complete rendered phone onboarding and control-handover checks.
- [x] Build and qualify a same-key Developer update, publish the source/PWA,
  and verify exact Pages/live bytes.
- [ ] Request all required Bluetooth permissions on app load independently of
  closed panel visibility; qualify fresh-install permission handling.
- [ ] Verify physical discovery, two-sided button pairing, saved reconnection,
  wearer calibration/retry/fallback, main transition and the full trial.

Developer 0.2.2-dev/code 3 is software-qualified and installed with exact byte
readback. Actual Quest 2 cold startup rendered VR with the panel closed, and
the gateway reported selected v2 discovery with an explicitly unavailable
hardware serial. See the [APK record](../quest/apks/README.md). Real phone
pairing and wearer acceptance remain pending. The initial unshipped protocol's adaptive
SAS-code selection and nonce-encoding defects are repaired in native
`0511b82d`. The full WebCrypto-to-production-Java exchange passes both approval
orders and altered-transcript checks; the focused gate reports 369 Java and
nine browser checks. Native `5f457bcb` closes the legacy public BLE mutation
bypass. An independent source review found no remaining blocker in those
bounded repairs. All 130 Node checks pass with zero skips, including actual
Java interoperability and both real assembly resolvers. Chromium passed 78
screens plus 12 focused final control screens, with actual Pretext and DOM
checks. These are software fixtures; real headset/phone acceptance remains open.

[PR 5](https://github.com/GeorgeFejer91/viscereality.org/pull/5) merged as
`90f2c759a0991a6b926a7bcaacc520b4c77e8d28` and deployed successfully in
[Pages run 43706732014](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43706732014).
All 18 checked live runtime/source/release files matched committed Git blobs,
including every changed versioned JavaScript/CSS request and the native patch.
The remote context gate passed. On the refreshed Pixel, the user found the
Quest, saw matching codes on phone and headset and confirmed both buttons.
This is user-observed physical first pairing; saved reconnection and subsequent
wearer calibration remain open. The next step exposed a startup-permission
defect: the phone reported missing Nearby devices permission, but no native
permission-request option appeared. Only Bluetooth Scan was missing. At the
user's explicit request, it was pre-granted on this installed test APK and
read back as granted so testing could continue. This is a bench exception;
future fresh installs must request required permissions automatically on load
using the normal attended platform flow while keeping the app panel closed.

Meta accepted code 3 as private Developer build `1460570757141936`; malware
and security scans passed. The owner's existing invitation is now joined;
no invitation action remains. The Store is unpublished. This exact headset
update used ADB, so Meta version-to-version delivery remains untested.
The user now requires local headset testing before Meta submission; keep
iterative test APKs local and upload a qualified release afterward.

### Shared scheduler and guided tilt setup — 2026-11-14

The user approved Automatic on app load, one shared native audio/phase clock,
an unlimited 4–4–4–4 preparation loop, exactly two full cycles for measured
Polar tilt calibration, native progress in the phone UI, automatic successful
tilt activation, repeat calibration and Controller State fixed-orientation
fallback. Start main seamlessly at the next cycle boundary, preserving input.
The main run remains exactly 15 minutes. This supersedes the illustrative phone
preview and old embedded 15-second calibration. See the approved design and
implementation checklist in [SHARED-BREATHING-SETUP.md](SHARED-BREATHING-SETUP.md).

- [x] Confirm behavior and 900-second main duration with the user.
- [x] Reconstruct released native tree `cd2d5a5837d2c642a70ba6b1cf7256f14ebd7969`
  from the published patch against upstream `99352620`.
- [x] Implement the native schedule/model core: 36 standalone tests cover full
  cycles, valid fit, failed evidence, source retirement, retry and fallback.
- [x] Integrate native input/capture and exact 900-second finalization; mapped
  source clocks, rejected evidence, retry lineage and pause/resume are reviewed.
- [x] Integrate audio, enrolled commands and native-driven experimenter UI.
- [x] Build and verify the separate non-debuggable developer APK, its source
  tree, signer, 74 audio files, ARM64 libraries and five shared-loop JNI exports.
- [x] Publish the verified source/PWA increment and verify the exact Pages run.
- [ ] Complete Quest 2/worn-Polar, installed Pixel and remote update acceptance
  (deferred on November 14; Quest 2 wearer testing resumed on November 15).

The independent Meta developer-distribution chat owns account/channel setup.
Feature work remains in this owner. Developer build 0.2.0-dev (code 1) was
accepted in the private Meta Developer channel; malware and security scans
passed. Owner enrollment still awaits the account owner's invite consent in
that chat. The store remains unpublished. Its exact
source/signature/hash are in the [APK log](../quest/apks/README.md). The developer
manifest targets SDK 34 with install location auto; rehearsal defaults stay
unchanged. Device acceptance remains pending; the resumed test is tracked above.

Source [PR 3](https://github.com/GeorgeFejer91/viscereality.org/pull/3) merged
as `efaed6c13b17305ccd1e9c6ed7a5227330f9e179` and deployed successfully in
[Pages run 43647565252](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43647565252).
All eleven checked live runtime/source files matched their committed Git bytes,
including the five versioned JavaScript/CSS requests, HTML, canonical loop,
native patch and three packaging/provider inputs. The remote context gate
passed. See [verification](VERIFICATION.md#shared-breathing-setup-release--2026-11-14).

The merged browser controller passed all 116 Node checks, including both real
assembly checks, and the native suite passed 798 with three ignored. Chromium
rendered 59 synthetic states including
native setup, reload recovery, Bluetooth rediscovery, 320px, enlarged text,
spacing and keyboard focus. The fixtures advance only supplied native frames.
Private screenshots and check receipts stay in ignored output. Worn-Polar/Quest
2 validation remains required. The concurrent 0.1.30 prompt and
reconnect releases were merged; their source and public assets are preserved.

### Quest calibration menu voice cues — 2026-11-14

The Quest **Direct → Developer settings → Breath → Calibration** menu plays
“Atme bitte jetzt ganz ein.” after an accepted Start, then “Atme bitte ganz aus.”
four seconds later if the same calibration is still active. Cancel, Reset,
switching configuration, closing the menu or backgrounding it stops playback.
The phone preview and the Haupt-Runde timeline retain their existing behavior.

Both German scripts were explicitly approved. Secret Tunnel v6 generated the
two MP3s from the EmoDB Atmungsraum 3 voice; its Whisper check reported zero word
edits for both. See the [prompt inventory](../assets/audio/calibration-menu/manifest.json)
and the [signed Quest release](../quest/apks/README.md#0130--quest-2-calibration-menu).
APK 0.1.30 is signed by the same lab certificate as 0.1.29; both bundled
MP3s and the unchanged scenario bytes passed archive readback. Release
`b360e99` reached `origin/main` and deployed in
[Pages run 43636115452](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43636115452).
The public APK, both MP3s, and both transcripts matched the release bytes by
SHA-256.
Next: listen through an attended Quest 2 calibration with active sensor data,
including Cancel before the exhale cue. Keep the overall rehearsal goal open.

### Start-to-finish QA and reconnect latency — 2026-11-14

From the clean sparse publication checkout, the Chromium review covered 43
screens across startup, setup, all five scenario choices, playback, recording
recovery and error states. The installed Pixel 10 PWA joined the already
enrolled Quest on five consecutive reloads in 263–599 ms from navigation to
the participant control. These are bench reload measurements. The Quest 2
installed APK matched the signed 0.1.29 release, and the renderer's idle
scorecards sampled 71.7–72.2 frames/s with no stale frames.

The reconnect regression reproduced an eight-second wait after the companion
reported that Bluetooth was off or scanning. The PWA now checks companion
progress on its existing one-second tick and probes the failed saved address
at most once every eight seconds while attempting recovery. It still tries
that address first during normal startup and authenticates the same Quest
before saving any replacement address. The rendered regression verifies
Bluetooth off → scanning → Wi-Fi recovery without a tap or chooser.

Reconnect fix `c469b92` reached GitHub main and shipped in the superseding
[`b360e99` Pages deployment](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43636115452).
The live HTML, versioned modules, manifest and QA notes matched Git bytes.
The Pixel loaded the new runtime in standalone mode and rejoined five times:
892 ms on the first updated load, then 260–285 ms on four warm reloads.
The installed headset still ran 0.1.29. The newly published 0.1.30 artifact
passed local signature and package-content checks; its attended run is open.

The physical headset still reported an open panel and no live H10 streams
during this sweep. After a wearer returns to the scene with the strap active,
run the short guided scenario through Start, Pause, Resume, Finish and CSV
retrieval, measuring the command-to-visible-state timings. Repeat the address
change with the normal Chrome loopback permission; Chrome blocked its prompt
while the Pixel's RustDesk overlay was visible. Maestro still failed to find
the USB Pixel while the shared ADB server listed an offline emulator. The
physical PWA observations used Chrome DevTools.

### Smartphone screenshot review — 2026-11-14

The [phone screenshot protocol](PHONE-SCREENSHOTS.md) builds a numbered
review pack from the app checkout. The first run rendered 44 PNGs against
runtime revision `4b74033`, including all five participant scenario
selections, both breath button glow peaks, and error and recovery states.
The gallery and CSV use the same screen numbers. The Wormhole download
matched the local archive by SHA-256. Keep review archives and transfer
links out of Git. Use the screen numbers to track feedback on the
operator flow. Physical headset and phone acceptance remain separate.

### Automatic Quest connection — 2026-11-14

The operator's connection flow should require only turning on phone Bluetooth
if it is off. Once Atmungsraum opens, the phone discovers the Quest, retrieves
the connection information over Bluetooth, establishes the Wi-Fi control link,
and proceeds to preparation. Controller ID, Quest key, hotspot details and
a save action are absent from the operator flow.

After the first verified connection, remember the same headset. On later
launches and after a connection loss, restore its Wi-Fi control link without a
tap or chooser. If Bluetooth is needed and switched off, show a status asking
the operator to turn it on, then resume automatically. An existing authenticated
Wi-Fi connection can continue with Bluetooth off.

- [x] Save the Quest serial, stable device identity, signed build fingerprint,
  Bluetooth device ID, last working gateway address, trusted enrollment and
  any provisioned hotspot settings in the phone's browser profile after a
  verified join. The installed lab Pixel reconnected using its saved address.
- [x] Verify recovery on the installed Pixel after interrupting the live Quest
  gateway requests. The app detected loss and automatically restored the
  same authenticated connection and private recording controls when requests
  were allowed again. A later physical Quest Wi-Fi disable/enable test also
  passed with the installed Pixel. The headset received its previous Wi-Fi
  address, and the phone restored the same authenticated Quest without a tap.
  The check left Quest Wi-Fi enabled and retained the phone's saved pairing.
- [x] Verify a reload with an unusable saved gateway and Bluetooth off. The
  installed Pixel displayed the Bluetooth-off message, then recovered the
  same Quest 2 and persisted its advertised gateway with no chooser or app
  tap after Bluetooth was turned on. The test substituted an unreachable
  address in the phone's saved pairing and temporarily granted Chrome
  loopback permission through DevTools. Both test overrides were restored.
- [ ] Repeat after changing the headset's actual Wi-Fi address. Confirm
  recovery with a durable Chrome permission and with another Quest nearby.

The Android discovery companion supplies the saved Quest's advertised address
when Chrome has no Bluetooth device object. It reports the phone radio state.
The browser contract tests reject mismatched identities and public addresses.
All 91 Node tests passed. The companion and PWA integration deployed in
[Pages run 43634783135](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43634783135)
at `5cf2605`. The versioned public runtime matched Git, and the installed Pixel
loaded it in standalone mode. Chrome's normal one-time loopback permission
approval is pending because the active RustDesk overlay blocked the tap.
See [Android discovery setup](../phone-discovery/README.md).

The PWA now offers one **Mit Bluetooth verbinden** action and no manual
connection fields. It reads the selected Quest's identity and endpoints and
opens the authenticated Wi-Fi connection with the saved enrollment. The
2026-11-17 startup increment checks for the PWA release first and attempts
Bluetooth rediscovery before the saved endpoint. The Quest still owns trusted
enrollment.
APK 0.1.29 advertises identity and addresses. Its hotspot characteristic
requires existing enrollment and credentials for a running phone hotspot.

- [x] Remove the connection form and repair button. Use the same Bluetooth
  action for discovery and recovery, including refreshing the signed build
  for the same enrolled Quest. The local Chromium review captured 43 screens
  and verified keyboard connection, status focus, and the missing-enrollment
  state. All 89 Node tests, six scenario pairs and the context check passed.
  Runtime commit `0fc97c2` deployed in
  [Pages run 43631515034](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43631515034).
  All four live runtime files matched Git. The enrolled lab Pixel 10 loaded
  the release in standalone mode and reconnected on launch. After an idle
  disconnect, the Bluetooth action opened Chrome's chooser and selection
  rejoined Quest 2 APK 0.1.29 over Wi-Fi with participant and private
  recording controls available. No app form appeared. Chrome displayed the
  headset as an unnamed device, and the headset reported its panel open.
  The test used an existing enrollment and a shared Wi-Fi network.
  The connection button fit at five sizes from 320px mobile to desktop,
  including landscape and a user text-spacing override.
- [ ] Implement automatic discovery and hotspot management in an Android
  phone component. Chrome's first Web Bluetooth request requires a gesture
  and the browser's device chooser. Android requires the Nearby devices
  runtime grant. Validate grant and hotspot behavior on the lab Pixel.
- [ ] Implement a trusted Bluetooth handoff on the Quest and phone. Reuse
  the enrolled LAN gateway for control and recording access. Establish trust
  before exchanging keys or hotspot credentials. Keep discovery advertisements
  and unprivileged reads free of secrets. Verify the Quest can rejoin the
  managed network after initial OS authorization.
- [ ] Connect the controller UI directly to preparation after authenticated
  Wi-Fi join. Verify a fresh provisioned Pixel and Quest 2, Bluetooth off/on,
  no shared Wi-Fi, process restart, transient transport loss, and a second
  nearby Quest. Confirm no manual setup screen appears, and verify control
  and private recording access on the intended Quest.

Platform references: [Chrome Web Bluetooth](https://developer.chrome.com/docs/capabilities/bluetooth),
[Android Bluetooth permissions](https://developer.android.com/develop/connectivity/bluetooth/bt-permissions),
[Android local-only hotspot](https://developer.android.com/reference/android/net/wifi/WifiManager#startLocalOnlyHotspot(android.net.wifi.WifiManager.LocalOnlyHotspotCallback,android.os.Handler)),
and [Quest-side Android Wi-Fi requests](https://developer.android.com/reference/android/net/wifi/WifiNetworkSpecifier.Builder).

### Smartphone UI decision tree — 2026-11-14

This tree describes the current release. German labels match the [smartphone controller](https://viscereality.org/atmungsraum/).
**Atemvorschau** is a dedicated phone screen between setup and scenario
selection. The Quest displays the VR scene and runs playback.

```text
Open Atmungsraum → Quest-Steuerung
├─ Provisioned phone → automatic reconnect
│  ├─ Connected → preparation
│  └─ Unavailable → Mit Bluetooth verbinden → choose Quest in Chrome
│     → authenticated Wi-Fi connection → preparation
├─ Phone without a trusted enrollment → authorization status
└─ Quest connected → preparation
   ├─ Headset panel open → close it to return to the VR scene
   ├─ Teilnehmer-ID eingeben → surname prefix, age, gender
   │  └─ Teilnehmer-ID übernehmen
   ├─ Polar H10 verbinden → wait for ECG + accelerometer data
   │  └─ Zero or multiple straps found → adjust straps and try again
   └─ Atemvorschau → once the setup is ready
      ├─ Einatmen → example circle grows over four seconds
      ├─ Ausatmen → example circle shrinks over four seconds
      └─ Green Weiter zur Auswahl → Quest-Steuerung
         └─ Choose VR-Bedingung → Bedingung A or Bedingung B
            └─ Choose Ablauf for either condition
               ├─ Anleitung                → guided example (1:08)
               ├─ Probe · 5 Min.            → five-minute practice
               ├─ Probe · 10 Min.           → ten-minute practice
               ├─ Probe · 15 Min.           → fifteen-minute practice
               ├─ Haupt-Runde · 15 Min.     → fifteen-minute main run
               └─ Box-Atmung · 20 Min.      → diagnostic, when offered
```

**Starten** prepares and starts the chosen run when the Quest is ready,
the participant ID is confirmed and both sensor streams are available.
The run view shows the Quest clock and the live ECG where available.
**Pause** and **Fortsetzen** change playback. **Beenden** goes through
finalizing and saving before returning to preparation. A Polar hold can
pause the run and keep **Fortsetzen** disabled until data is fresh.

The phone lists the timelines advertised by the enrolled Quest. Older Quest
connections can show **Szenario A/B** by itself. The participant-enabled
Quest filters out **Box-Atmung · 20 Min.** because the diagnostic has no
participant recording plan.

Other phone views and actions

- **App installieren** opens the browser install prompt or install help when
  available in the browser.
- When the Quest supports them, **VR zentrieren** centers the view and
  **Quest geprüft** clears an uncertain command after the operator checks
  the idle headset. **Verbindung trennen** returns to connection setup.
- **Gespeicherte Quest-Aufnahmen** offers a recording picker and
  **Aufnahme herunterladen**. **Jetzt abgleichen** retries idle sync.
  **Google Drive** expands the separate authorization form.
- The direct `#daten` route opens **Frühere Browser-Aufnahmen** with links
  to historical CSVs when present and **← Zurück** to the controller.

The circle uses a simulated radius. The next breath button pulses twice
and stays highlighted. Reduced motion uses a static highlight and a
discrete size change. The green **Weiter zur Auswahl** button appears after
both four-second preview phases. An interrupted sequence starts again at
**Einatmen**. The preview also resets after editing participant details or
explicitly disconnecting.
A run already in progress remains controllable when the phone reconnects.
Define the audio cues and calibration protocol before connecting the preview
to either. See [preview verification](VERIFICATION.md#phone-breathing-preview--2026-11-14).

The initial preview commit `720d017` reached `origin/main` and deployed in
[Pages run 43609888532](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43609888532).
The four public runtime files matched their Git bytes. All six viewport
and text-layout checks passed against the deployed app with synthetic
Quest state. The installed lab Pixel 10 reloaded in standalone mode,
loaded the three versioned release assets and kept the scenario picker
hidden before setup. A physical tap-through remains pending a Quest that
reports ready Polar streams. The Quest reported Polar setup unavailable
during this readback.

Guided preview commit `1ae9930` reached `origin/main` and deployed in
[Pages run 43612442755](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43612442755).
All four runtime files matched Git. Seven browser checks passed against the
live release with synthetic Quest state, including the phase timers, glow,
reduced motion, interrupted setup, and green continuation. The lab Pixel 10
loaded the new assets in standalone mode. The physical tap-through remains
pending ready Polar streams. Next, define the audio and calibration protocol.

### Quest APK release record — 2026-11-14

The [Quest APK log](../quest/apks/README.md) records the signed 0.1.29
artifact, the source tree verified against the public patch, the native
changes since 0.1.28, and the Quest 2 and Pixel acceptance evidence.
The [APK workflow](WORKFLOW.md#quest-apk-changes-and-releases) requires
an Atmungsraum commit and release record for every changed APK.
The release includes the full 0.1.29 APK and the matching Experimenter
audio, particle payload, shaders, configuration and ARM64 libraries in
the repository. The [public build recipe](../quest/README.md#build-from-the-public-repository)
pins the native source and sibling dependencies and resolves the bundled
inputs without a private checkout. External developers use their own
locally generated development signature.
The 2026-11-14 clean build used fresh public clones and reconstructed native
tree `cd2d5a5837d2c642a70ba6b1cf7256f14ebd7969`. Android SDK tools verified
the independently signed APK. All 67 scenario files, five provider assets,
and two ARM64 libraries matched the lab APK byte-for-byte.

### Live recording sync to Google Drive

Goal: transfer verified recording increments from Quest 2 to the installed
Pixel and upload them to the existing private research Drive during a full
15-minute Polar H10 run. Quest retains local acquisition, recording, audio,
and rendering authority. The final Drive object must be the same dated
participant CSV, with the same bytes and SHA-256 as Quest and Pixel.

- [x] Establish the current boundary. The live phone waveform is a rolling
  two-second ECG preview. The complete CSV and Drive upload currently use the
  idle-only finalized recording path. The 15-minute trial exported 93,987,682
  bytes, and post-session transfer and delivery took about nine minutes.
- [x] Record detected ECG and ACC interruption and recovery as stream-specific
  research events. Retain timestamps and reason when known. Distinguish these
  events from a manual playback pause and recorder queue loss. Preserve the
  original sample timestamps and support historical CSVs.
  - [x] Add CSV v3 signal events for both streams and preserve the v2 reader.
    The 76 focused native recording tests cover checkpoint recovery and
    finalization. The 84 PWA tests include both CSV versions, and all 72
    gateway checks passed. Signed Quest APK 0.1.28 from `84bcf7fd` was
    installed and its SHA-256 matched the build. The installed Pixel rejoined
    over its enrolled gateway and loaded the v3 reader from `d4e2dcc`,
    deployed in [Pages run 43480098804](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43480098804).
  - [x] Verify deliberate ECG and ACC interruptions on Quest 2 with a live H10.
    The 2026-11-14 CSV records a distinct interruption/resumption pair for
    each deliberate stop and separate playback transitions.
- [x] Expose durable recording increments under the enrolled private gateway.
  Prove the transferred byte prefix remains valid through pause, delayed
  samples and finalization. Use bounded sync work that yields to controls and
  keeps transfer failures off the acquisition and rendering path.
  - [x] Native source `bdfbf8d6` adds the append-only CSV v4 projection and
    current-run LAN queries. The projection runs on a separate Quest worker.
    Host tests proved prefix equality through backdated Pause, late
    samples, finalization and checkpoint recovery. The 78 native recording
    tests and 86 Android gateway checks passed, including guest/relay denial,
    committed-range bounds, and active-run replacement while reading.
    Same-signer APK 0.1.29 was installed and read back byte-for-byte on Quest 2.
    [Pages run 43485062024](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43485062024)
    deployed `1843da3`. The installed Pixel loaded that PWA release, rejoined
    Quest over the enrolled gateway, saw the new capability, and confirmed
    that a live read is refused while idle.
  - [x] Verify the live prefix feed and its sustained rate on Quest 2.
    The 2026-11-14 sync-enabled trial persisted checkpoint ranges on Pixel.
    Drive acknowledged 74,711,040 of 74,967,884 bytes before Finish. In the
    uninterrupted comparison window, the observed prefix-to-Drive delay had
    a 5.35 s p95 across 69 checkpoints.
    Idle physical transport check on APK 0.1.29 and the installed Pixel read
    160 signed, encrypted 32 KiB ranges of the earlier finalized trial.
    The 5 MiB transfer took 24.957 s (205.2 KiB/s). With parallel status
    polling, 23 signed status requests had a 135 ms p95 response time.
    The prior 93,987,682-byte trial averaged 102 KiB/s over its 900 active
    seconds. The idle read benchmark does not include a running CSV writer
    or Drive uploads.
- [x] Buffer verified increments on the Pixel and upload them with one resumable
  Drive session for the final CSV. Confirm acknowledged offsets, handle expired
  authorization or upload sessions, and use the existing final export to repair
  missing ranges. Finalization must verify the file identity and hash.
  - [x] Verify that the installed Pixel can upload an unknown-length prefix to
    the private research Drive, read its acknowledged byte range, and finalize
    the same file ID with the expected size and SHA-256. Implement bounded
    resumable requests and persisted offsets in the PWA transport, including
    for complete CSVs already staged on the phone.
    [Pages run 43478317772](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43478317772)
    deployed `640e5e7`. The refreshed installed Pixel loaded the deployed
    module and recovered a deliberately lost Drive acknowledgement at byte
    1,048,576. It finalized the 1,310,725-byte synthetic file under the same ID,
    verified its SHA-256 and moved it to trash.
  - [x] Connect the uploader to durable Pixel staging and the signed Quest
    feed. The 85 browser tests cover range hashes, resume offsets, and repair
    from the final manifest. The final signed SHA-256 is checked before the
    final Drive request. The Pixel keeps one Drive file ID for the run.
    A synthetic IndexedDB check on the installed Pixel verified offset resume
    after reopening the store, cross-chunk reads, overlap rejection and
    atomic replacement of staged bytes with the final archive.
    A separate installed-Pixel fixture interrupted both transports. Pixel
    resumed from byte 131,072 after reopening IndexedDB. Drive reconciled
    a lost acknowledgement and accepted 786,432 bytes before Finish, then
    verified the 1,310,725-byte file under the same ID. The fixture used
    synthetic Quest ranges and the real private Drive, then trashed its file.
  - [x] Drain a backlog through the last published checkpoint in bounded
    batches. Reuse that descriptor until caught up. Release to controls
    between batches and stop the drain on cancellation. All 87 PWA tests
    pass, including resumed offset and batch-bound checks.
  - [x] Recover when staged bytes are missing or fail the signed final hash.
    Download a fresh copy from Quest and restart any suspect resumable session
    under the existing Drive file ID before finalization. All 88 PWA tests pass.
    [Pages run 43489844527](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43489844527)
    deployed `0720992`. The refreshed Pixel opened a new upload session under
    the same ID after a bad synthetic prefix. Drive verified the corrected
    262,149-byte file and the fixture trashed it.
  - [x] Verify durable recovery on the installed Pixel through a real Wi-Fi
    interruption and WebAPK reload. After reauthorization, staging resumed
    at byte 524,289 and Drive resumed at byte 524,288 within the original
    upload session. Drive verified the 1,310,725-byte result under the same
    file ID. The test used synthetic recording ranges while Quest was idle.
    See [device recovery evidence](QUEST-PHONE-SYNC.md#pixel-wi-fi-and-reload-recovery).
  - [x] Verify installed Pixel staging and Drive receipt during a physical run.
    The 2026-11-14 sync-enabled trial finished after all 900 audio seconds
    with zero writer drops. Quest, Pixel and one private Drive CSV matched
    by byte count and SHA-256. Drive preserved its original file ID through
    the faults and finalization. See the
    [physical sync trial](QUEST-PHONE-SYNC.md#sync-enabled-physical-trial-on-2026-11-14).
- [x] Run comparable sync-disabled and sync-enabled 15-minute physical trials.
  Measure Quest frame timing, Polar freshness, control acknowledgements,
  writer loss, and time from a Quest checkpoint to Drive acknowledgement.
  Exercise LAN loss, a Drive failure, and a Pixel reload. Confirm catch-up,
  explicit interruption markers, and one private Drive object per run.
  Both runs completed the 900-second audio and saved with zero writer drops.
  The [paired comparison](QUEST-PHONE-SYNC.md#paired-physical-trial-results-on-2026-11-14)
  records frame cadence, Polar freshness and the signed phone controls.
  The sync-enabled run recovered from all three planned transport faults
  and one additional Chrome renderer exit on the Pixel. The baseline used
  live sync off and delivered its verified CSV via the post-session path.
  The local Quest, installed Pixel and private Drive copies matched for
  both runs, with one Drive object per run. The comparison was published
  in [Pages run 43593889030](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43593889030).
  The browser exit recurred in a repeat and without continuous DevTools
  observation after recovery. Android Chrome reported a bad Blob reference
  in both repeat exits. The PWA now stages owned byte arrays and reads
  previously stored Blob ranges directly. The Pixel passed a mixed-format
  IndexedDB fixture. The updated release from
  [Pages run 43595036518](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43595036518)
  resumed an ongoing run across formats and finalized it under the same
  Drive file ID. The [full run on the updated PWA](QUEST-PHONE-SYNC.md#full-trial-on-the-updated-pwa)
  repeated the transport and sensor faults without a Chrome renderer exit.
  Quest saved with zero writer drops. Pixel and Drive verified the same
  67,400,882-byte CSV and kept one private object under the original ID.
  The paired runs verified directional Polar biofeedback, and the updated
  run retained similar frame cadence, fresh raw streams and working phone
  controls. The updated run and a short sync-disabled probe each encountered
  native motion rejection. A subsequent full session started on the same
  PWA and demonstrated valid directional changes during live upload.
  See [fresh Polar response](QUEST-PHONE-SYNC.md#fresh-polar-response-on-the-updated-pwa).
  The completed 69,906,729-byte CSV matched across Quest, Pixel and
  Drive. Its 70,280 radius observations ranged from 1.00 to 2.00 m.
  The live biofeedback evidence was published in
  [Pages run 43598957102](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43598957102).
  The [repeat comparison](QUEST-PHONE-SYNC.md#repeat-comparison-on-the-updated-pwa)
  on the same PWA release measured identical submitted-frame cadence,
  a 0.10 ms increase in p95 frame submission CPU time, fresh raw Polar
  input and zero writer drops during live sync. The paired phone actions
  took up to 1.26 seconds with live sync. The repeat baseline's Polar
  classification remained unknown, so directional response in both modes
  is verified by the earlier pair. After fault recovery, Drive acknowledged
  live checkpoints at 3.20 seconds median and 4.94 seconds p95.
  [Pages run 43602269910](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43602269910)
  published the measured comparison. The same-release baseline also
  delivered one private 77,448,407-byte CSV after Finish. Its name, size
  and SHA-256 matched the Quest and Pixel copies.
  Follow the [paired measurement procedure](QUEST-PHONE-SYNC.md#paired-physical-trial-measurements).
- [x] Reproduce the earlier phone lock after Resume immediately followed by a
  new Polar hold. Clear the pending control on a confirmed signed receipt
  and a fresh same-generation paused state with the Polar latch set. The
  86-test PWA suite includes this case. Retest it during the physical probe.

### Polar H10 recording and single participant CSV — 2026-11-13

- [x] Open the Quest controller by default and retire browser-local recording.
  Preserve read-only download of historical browser files at `#daten`.
- [x] Verify native CSV v2 against ECG, accelerometer, applied renderer radius,
  and Polar breath-classifier fixtures with independent timestamps. Reuse the
  checkpoint writer for interrupted attempts. The 73 focused Rust recording
  tests passed on native commit `535cea3b`.
- [x] Verify that the controller fetches only the hash-bound v2 CSV for
  participant download and Drive delivery. Older Quest bundles retain their
  existing reader. All 88 Node tests passed, along with four Chromium
  viewport/zoom and historical download checks.
- [x] Build, install and launch the same-signer Quest APK `0.1.26` from
  native commit `535cea3b`. Installed SHA-256 matched the signed ARM64 APK,
  and the renderer produced advancing OpenXR frames.
- [x] Publish `0f22030` in [Pages run 43457129533](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43457129533).
  Nine live files matched Git. Live Chromium and the refreshed standalone
  Pixel PWA opened the new controller with no mock ECG request.
- [x] Request and confirm both live H10 streams in the phone setup flow.
  Require fresh ECG and accelerometer samples before participant Arm.
- [x] Reauthorize Bluetooth pairing for build 0.1.26 on the installed Pixel
  and verify the enrolled signed LAN connection. The live H10 v2 test reached
  official Start, paused, resumed and saved with zero writer drops. Quest,
  Pixel and the private Drive folder agreed on the dated CSV name and exact
  7,167,072-byte SHA-256. The CSV had 11,457 ECG rows, 17,673 three-axis
  accelerometer rows, 6,355 applied-radius rows with 3,377 distinct radii,
  6,334 breath rows and one terminal row. The participant, scenario and
  attempt matched across all 41,820 data rows.
  A readback against the Quest originals matched every ECG and accelerometer
  sample, all 6,355 renderer frames to native float precision, and all
  classified phases. The 88.526-second run paused and ended manually, with
  radii from 1.0 to 2.0 metres. Six observation timestamps stepped backward
  by up to 351 microseconds. Use the clock fields when ordering rows.
- [x] Interrupt the v2 CSV media PUT before Drive receives it. The installed
  PWA retained the verified local file and unconfirmed receipt. It
  automatically retried the reserved ID after transport recovered. Independent
  inventory showed one private CSV and the PWA confirmed the matching hash.
  See [CSV v2](DATA-LOGGING.md#participant-csv-v2).

- [x] Remove the retired browser recorder, NeuroKit generator and 1.84 MB
  mock trace, demo audio controller, and demo-only styling and checks. Preserve
  read-only downloads of completed browser CSVs, Quest transfer, and offline
  audio verification. All 80 Node tests, six scenario pairs and the context
  check passed. Chromium verified Quest entry, install focus and byte-exact
  legacy downloads at four viewport/zoom settings. A fresh recovery route
  leaves the retired database absent.


### Windows Quest 2 keep-awake task — 2026-11-13

The lab Windows account starts the **Quest 2 Keep Awake** Task Scheduler
task at sign-in. It checks the authorized Quest 2 over ADB about every eight
seconds and renews a ten-minute proximity hold when missing or after eight
minutes. The task retries after a lost connection. It needs a running
Windows session and an authorized ADB connection.

After a headset reboot, ADB showed the previous device watchdog had stopped
and the display had returned to standby. The Windows task restored the
display and proximity hold. Verification observed a scheduled repair of a
short test hold, followed by an awake poll that left the fresh hold in place.
To restore normal wear detection, disable and stop the Windows task first,
then follow [the Quest awake runbook](QUEST-AWAKE-TESTING.md). The physical
phone and Quest app rehearsal checks remain open.

Protocol update, 2026-11-18. The installed `$keep-quest-awake` skill and the
[Quest awake runbook](QUEST-AWAKE-TESTING.md#windows-periodic-watchdog-for-the-lab-quest-2)
now cover the scheduled task and the separately documented MSI/QFM variant.
This Windows account's task, installed-app, and process inventory showed neither
watchdog. For a renewed bench hold, check the configured lab host for the
matching watchdog and confirm a fresh heartbeat plus Quest power/proximity
readback. App launch and Pixel discovery still need their own checks.

### Pixel PWA release refresh — 2026-11-13

After every Atmungsraum PWA deployment, refresh the installed app on the lab
Pixel 10 and verify the release there. [WORKFLOW.md](WORKFLOW.md#smartphone-pwa)
owns the standing procedure.

Verified the current installation via USB and reloaded the live VR-Probe
route in standalone mode. The four loaded stylesheet and module paths
matched the runtime bytes on `main` at `64994f2`. The lab Pixel kept its
existing Chrome profile. No reinstall or site data reset was needed.

### Pixel RustDesk remote input — 2026-11-13nn- [x] Verify live Pixel 10 frames and a Windows RustDesk click that switchesn  the phone from Settings to Connection. The session uses the existingn  Pixel USB ADB forward to a Windows loopback listener.n- [x] Resolve missing remote input. Android showed the RustDesk Input servicen  enabled but unbound while a Mobile CLI UiAutomation helper held the Pixel.n  Stopping the Pixel helper restored the binding and RustDesk reported inputn  enabled. Verify with Android service readback and a RustDesk click. Avoidn  running an accessibility UI inspector during the control check.n- [ ] Verify the attended handoff from a second laptop. The Pixel stayed awaken  on USB power with screen capture active and the Windows viewer open. An  direct connection by phone ID failed in the test. The second laptop cann  connect to the Windows host with local approval and use its Pixel viewer.nnThe separate RustDeskMCP fork and the Quest 2 input checks have their ownnacceptance items. No experiment code changed in this verification.nn### One saved Pixel–Quest 2 connection — 2026-11-13

- [x] Learn the Quest serial and LAN address over Bluetooth and the build
  fingerprint from signed status. Save one authenticated pairing and any
  hotspot settings in the phone's browser profile.
- [x] Open the paired phone directly into VR-Probe. Try the saved address
  first, then use the previously granted Bluetooth device to rediscover the
  address and provision the saved hotspot when needed.
- [x] Verify setup, cold entry, a changed address and headset epoch, rejection
  of a different headset, and explicit disconnect in Chromium. All 87 Node
  tests, six scenario pairs, UI and context checks passed.
- [ ] Complete trusted enrollment and power-cycle both physical devices.
  The installed Pixel Chrome 134 exposes the Bluetooth chooser. Its
  `getDevices()` API is unavailable, so automatic use of the saved address
  and changed-address recovery need separate physical checks.
  See [verification](VERIFICATION.md#saved-quest-connection--2026-11-13).

Runtime commit `a313621` reached `origin/main` and deployed in
[Pages run 43422623818](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43422623818).
All 11 affected public files matched Git. Four Chromium layouts repeated
the pairing fixture against the live site. The installed Pixel loaded the
new entry in standalone mode. The Quest LAN gateway did not respond to
the bench probe. Next, approve trusted controller enrollment on Quest,
complete first setup on the Pixel, then power-cycle both and verify a new
signed status observation without retyping settings.

### Password-free startup — 2026-11-13

The start page, VR-Probe and local CSV recovery open without the shared
password. The two folder send actions unlock the encrypted delivery
configuration on demand and reuse it for the current tab. Quest enrollment
still governs access to the headset and its recordings.

Local Chromium passed desktop, phone portrait and landscape, and 320px
entry checks. It downloaded an exact synthetic CSV, checked send-time
password retry, cancellation and saved credential lookup, and confirmed the
Pages-only upload limit without a request to Discord or Wormhole. Escape
stays within the delivery dialog instead of invoking the playback shortcut.
The Node and context checks passed.

Runtime commits `e48c3ef` and `ec7c736` reached GitHub main from the clean
`codex/atmungsraum-open-startup` sparse checkout. The containing commit
`e013fc6` passed [Pages run 43418701040](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43418701040).
Both public home URLs and four versioned runtime assets matched the commit
bytes. The live Chromium matrix repeated the startup, CSV download, and
synthetic send-time credential checks. The primary checkout retains unrelated
index changes and unreadable cloud-backed files. Continue the existing Quest 2
and Pixel hardware checklist below.

### RustDesk on Quest 2 — 2026-11-13

- [x] Sideload the official [RustDesk 1.5.3 ARM64 release](https://github.com/rustdesk/rustdesk/releases/tag/1.5.3).
  USB ADB installed `com.carriez.flutter_hbb` (version code 2071) on the lab
  Quest 2 running Android 14 (API 34, build `UP1A.231005.007.A1`). The installed
  APK matched the release SHA-256 and Android resolved `.MainActivity`.
- [x] Open RustDesk as a 2D panel. The initial launch failed because the APK
  was registered for the owner and the headset had switched to a secondary
  profile. Register the installed APK for the current Android user with
  `cmd package install-existing --user <user> com.carriez.flutter_hbb`.
  The user-scoped cold launch returned `Status: ok` and RustDesk gained focus.
- [ ] Verify a Quest-to-PC session using a Touch controller on a test window.
- [x] Approve **Share screen → Start service**, the first-run **I Agree**
  confirmation and the Horizon OS capture prompt. The wearer approved capture.
  Android reported an active RustDesk MediaProjection session.
- [x] Enable **RustDesk Input** for the active Quest profile. The default
  Accessibility Settings intent opened Horizon's General settings. ADB
  opened the Android Accessibility panel by targeting the installed Settings
  package with `am start --user <user> -a android.settings.ACCESSIBILITY_SETTINGS -p com.android.settings`.
  Under **Downloaded apps → RustDesk Input → Use RustDesk Input**, approve
  **Allow**. Android listed the service as bound with gesture capability, and
  RustDesk logged `onServiceConnected` and `input:true`. UIAutomator inspection
  temporarily disconnected the service before it rebound, so use Android
  service and RustDesk log readback for the final state.
- [x] Verify changing Quest video in Windows RustDeskMCP over USB. The fresh
  Horizon OS prompt accepted ADB keyboard focus on **Share** and Enter. Android
  then reported a RustDesk projection and virtual display. Windows showed
  changing compositor frames and the Quest Browser panel.
- [ ] Verify one harmless RustDesk click that visibly changes a 2D panel.
  Verify deliberate disconnect and stop sharing.

The Windows RustDeskMCP public-ID connection initially required sign-in.
After sign-in the retry failed to reach the relay/rendezvous server.
Enable [Direct IP access](https://rustdesk.com/docs/en/self-host/client-configuration/advanced-settings/#direct-server)
in RustDesk under **Settings → Share screen**. On this headset, ADB keyboard
navigation reached the toggle where panel-coordinate taps lost focus. Android
then accepted TCP connections on port 21118. A serial-scoped USB
`adb forward tcp:0 tcp:21118` created a Windows loopback-only listener.
RustDeskMCP established a direct TCP connection over USB. The Quest showed
an incoming control request whose peer ID matched the Windows client's
`--get-id` readback. Accepting the request in RustDesk authenticated the
session. Windows displayed **Connected, waiting for image**. At session start
the earlier MediaProjection token stopped and Horizon OS presented a fresh
capture prompt. ADB confirmed that Android rejected the previous token.
Bringing the RustDesk task forward, focusing the visible **Share** button,
and pressing Enter through ADB granted the fresh token. A coordinate tap
did not establish capture.

An idle Mobile CLI UiAutomation helper on the Quest suppressed Android's
RustDesk accessibility binding while the permission stayed enabled. Stopping
the Quest-only helper when it had no client restored the binding. The helper
reappeared during another check. Verify the bound service immediately before
the RustDesk click test and avoid UI hierarchy automation during that test.
On the 2D Browser test, clicks on **Create tab** and the banner dismiss
button produced RustDesk `doDispatchGesture` log entries without the expected
panel change. The Android display-0 Browser input region was
`[0,640][427,907]`, while one click on the visible **Create tab** button
was dispatched at `3902,34`. The RustDesk virtual display was 5496 × 960.
The [Android capture code](https://github.com/rustdesk/rustdesk/blob/1.5.3/flutter/android/app/src/main/kotlin/com/carriez/flutter_hbb/MainService.kt)
uses default-display metrics and the
[input code](https://github.com/rustdesk/rustdesk/blob/1.5.3/flutter/android/app/src/main/kotlin/com/carriez/flutter_hbb/InputService.kt)
rescales the pointer and dispatches a gesture. The measured mismatch supports
testing coordinate translation into a focused panel's input region.
The current test proves delivery to the RustDesk input service. A panel
action still requires verification.

A later check found no active capture, no prior USB forwarding entry and an
Atmungsraum rehearsal in the foreground. The active owner profile did not
have RustDesk Input enabled. Resume on a free headset, verify the active
profile and input binding, then reconnect and approve capture. After the
panel test, disconnect, stop sharing, disable the temporary Direct IP
setting and remove this test's USB forward.

The Windows RustDeskMCP release `v1.4.6-mcp.1` was running. Its loopback MCP
endpoint returned HTTP 503 while disabled. For an attended session the same
client supports [the normal `--connect` route](https://github.com/YaoxinCS/RustdeskMCP/blob/v1.4.6-mcp.1/src/core_main.rs#L740),
with the wearer accepting the connection on Quest. Use the installed standard
Windows RustDesk `1.5.0+68` as the target for the reverse test. Its CLI returned
a local peer ID without needing to display credentials.
[RustDesk's Android guide](https://rustdesk.com/docs/en/client/android/#request-permissions-and-start-services)
documents separate capture consent and input permission. [Meta's MediaProjection guide](https://developers.meta.com/vr/documentation/native/native-media-projection/)
describes Horizon OS compositor capture. [The earlier Quest 2 sideload report](https://github.com/rustdesk/rustdesk/discussions/3047)
leaves mouse input unresolved. Keep peer IDs, credentials and captured frames local.
The installation note at `920adb0` passed [Pages run 43413359983](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43413359983) and the live project notes and app checks.
The launch evidence at `b7b55c6` was verified live in descendant `2ea1ea7`
after [Pages run 43414690094](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43414690094).
The accessibility evidence at `f17e8ba` was verified live in descendant
`e013fc6` after [Pages run 43418701040](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43418701040).
The USB transport note at `081bd3a` passed
[Pages run 43420808589](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43420808589) and the live notes and app checks.

### RustDesk MCP client and skill — 2026-11-13

Installed the portable Windows x64 RustDeskMCP `v1.4.6-mcp.1` release alongside
standard RustDesk. Verified the archive SHA-256, the Start menu shortcut and
the running client window. The executable and DLL are unsigned. The running
client bound only to loopback and returned HTTP 503 with MCP disabled by
default. No RustDesk MCP session with the Pixel was run.

Codex discovered `rustdesk-mcp-operator`. The [project copy and launch
route](SKILLS.md#rustdesk-remote-operation) retain upstream provenance, license
and notice. GitHub main received `4689849` from the clean publication checkout.
[Pages run 43412221700](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43412221700)
succeeded for that commit. The route, skill, workflow reference, license,
notice, skill metadata and Atmungsraum home passed live checks.

The primary checkout still has unrelated Git index changes. The active phone
and Quest checklist remains open. For RustDesk qualification, verify remote
frame capture and one non-destructive input action on the Pixel.

### Codex Android phone testing — 2026-11-13

The [Android phone testing route](SKILLS.md#android-phone-testing) nests the
official Google and Mobile MCP skills and the local Maestro wrapper. Codex
reported all three skills enabled. Mobile MCP 1.0.17 initialized over local
stdio, exposed 34 tools, and found the USB-connected Pixel 10 on Android 16.
Mobile MCP and Android CLI 1.1.16919734 each captured a screenshot over USB.
Play Protect blocked Android CLI's instrumentation server, so its UI hierarchy
check stopped at helper installation. Maestro 2.12.1 ran on Windows, but its
[offline-device discovery bug](https://github.com/mobile-dev-inc/Maestro/issues/3706)
hid the Pixel while the shared ADB server listed an offline emulator. No
Maestro flow ran. Physical Pixel installation and standalone launch passed
on 2026-11-13. The headset/H10 and phone controller flow remain open.

Continue with the existing Quest 2 checklist and use an explicit device ID
for each phone command. Retry Maestro when the offline emulator is absent
or a release fixes device discovery.

GitHub main received `041f360` from an isolated sparse checkout because the
primary checkout's cloud-backed Git metadata was unavailable. Exact
[Pages run 43398518216](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43398518216)
succeeded. The rendered skill route, three linked skill pages, two license
files, project notes and Atmungsraum URL passed live checks.

### Quest 2 control, media and UI acceptance — 2026-11-12

The [VR UI testing protocol](VR-UI-TESTING.md) documents Meta XR Operator
against Meta's standalone, Unity, Simulator and Quest guides. The standalone
layer is the candidate for the native rehearsal. Compatibility with the
Atmungsraum build and Quest 2 firmware is pending. To qualify it, build the
isolated test APK, read a focused session with advancing frames, and verify
one input and its visible result. Continue the physical phone and Quest 2
acceptance below using the regular rehearsal build.
The For-AI context check and the protocol's local links and anchors passed.

GitHub main received protocol commit `8f21107`. Its exact Pages deployment
reported success. All five changed live Markdown files matched the commit,
and the rendered protocol, four entry links and Atmungsraum home passed.
A newer main commit superseded the [Pages workflow run](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43427274203)
after the protocol was live. Publication used a clean sparse checkout because
the primary checkout's cloud-backed Git metadata could not be mapped.

**Meta Quest 2 is the required headset for Atmungsraum development and release
acceptance.** Use a physical Quest 2 and the installed Android PWA. Earlier Quest 3
checks remain historical evidence. Keep Quest as the authority for scene, media
clock, sensor acquisition and recordings. The phone sends operator requests and
shows the corresponding Quest state and receipts.

- [x] Build and install the signed Quest 2 rehearsal APK from the shared
  Experimenter source. Verify Quest 2 targeting, ARM64/OpenXR loader and the
  installed APK digest. See the [assembly recipe](../quest/README.md).
- [ ] Launch on Quest 2 and confirm a rendered frame, Remote startup and scene
  readiness. Keep build fingerprints, enrollment, logs and device captures in
  private evidence.
- [ ] Exercise every headset-facing PWA action from the installed phone. Cover
  Bluetooth hotspot setup from a disconnected Quest 2, Bluetooth discovery on
  existing Wi-Fi, saved LAN reconnect, enrollment, Polar setup and live ECG, condition
  selection, Arm, Start, Pause, Resume, Finish, recovery, disconnect/rejoin and
  recording download. Match one original native receipt and state transition to
  each command. Verify the visible VR recenter control on Quest 2 and track any
  missing action or scenario mapping as an implementation gap.
- [x] Stage the exact authored audio library for the native build. Run
  `node quest/stage-audio.mjs output/quest-audio-package` with a new output path.
  The catalog binds six timelines, six manifests, the shared protocol and all
  53 WAVs. The regression verifies byte identity, a rejected corrupt asset and
  a valid replacement cue using the existing timing.
- [x] Package the authored [scenario JSON](../audio/scenarios/README.md) and
  [numbered audio assets](../assets/audio/scenarios/). The APK readback matched
  all 67 files, including the catalog. The native audio worker verifies cue
  identities, hashes, frame counts and schedule continuity before readiness.
- [x] Connect enrolled VR-Probe scenario selection to the native playback
  owner. Validate the generation-bound Arm request, both players' lifecycle,
  all six timeline frame counts and the rendered phone selector.
- [ ] Verify an offline cold launch and scenario playback on Quest 2. Retain
  the matching JSON/CSV/TXT sidecars with the library. Keep the 15-minute main
  round and verify the 26–41 s measurement window against native telemetry.
  Final audio replaces the numbered WAVs and refreshes their asset metadata
  and hashes. Preserve the wording approval gate.
- [ ] Inventory the APK's videos and audio by scenario and interaction. On Quest 2,
  verify the expected video and cue begin, advance, pause, resume and stop with
  native active time. Check first/last frames, loops and transitions where
  applicable, and verify terminal cleanup, repeated runs and absent/corrupt assets.
  Capture phone and Quest 2 screenshots at matched transitions. Use playback
  receipts and a listening or audio capture check for sound and cue timing. Verify
  the current silent placeholders as silence and use approved test media for
  audible output checks where needed.
- [ ] Run the existing controller, recording and audio suites and the rendered UI
  matrix. Stress portrait/landscape and rotation mid-run, narrow and short viewports,
  200% text, keyboard/focus, touch targets, slow media, rapid and duplicate taps,
  background/foreground, network and H10 loss, stale replies, app restarts, quota
  and concurrent sync. Check clipping, reachability, responsive controls, correct
  receipts, and recovery without duplicate commands or recordings. Record physical
  results and clearly identify synthetic fixture coverage in
  [VERIFICATION.md](VERIFICATION.md).

The browser and asset checks are recorded in
[VERIFICATION.md](VERIFICATION.md#quest-2-preflight-and-controller-stress--2026-11-12).
They exposed a hidden SVG waveform in VR-Probe. The browser now toggles the
SVG's visibility attribute and clears stale samples.

The web release `f4d47aa` passed
[Pages run 43275951484](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43275951484).
All nine changed public paths matched Git. A Chrome 157 smoke test on the live
URL passed with a signed synthetic gateway and a scoped test permission.

Both physical devices are authorized over USB. The signed `0.1.20`
development rehearsal is installed on Quest 2, and the installed APK hash
matches the signed build. The headset lock screen is the launch blocker.
The audio owner logs the actual Android player position for
[scenario timing verification](VERIFICATION.md#quest-scenario-player-position--2026-11-12).
All six authored timelines and both condition tracks passed a standalone
[Android media test on Quest 2](VERIFICATION.md#standalone-quest-2-media-player-check--2026-11-13).
The production Java audio owner passed all six scenarios and
[shutdown on Quest 2](VERIFICATION.md#quest-2-audio-owner-cleanup--2026-11-13)
in a separate instrumentation process.
The rehearsal app's launch and phone-driven playback still need device checks.
The Pixel 10 was updated from Chrome 134 to 157 through Google Play and
the installed PWA relaunched in standalone mode. Grant Atmungsraum Local
Network Access if prompted during the physical Quest join.
[Chrome 142 introduced this permission](https://developer.chrome.com/release-notes/142#local-network-access-restrictions).
At startup the Quest APK asks Android to show its turn-on confirmation when
Bluetooth is off. The phone shows a reminder when VR-Probe opens. Chrome’s
Bluetooth picker checks the phone’s power state after a tap.
The phone can provision the Pixel hotspot to the enrolled headset over
Bluetooth. Enable the hotspot on the Pixel, enter its credentials in VR-Probe
and confirm the Wi-Fi join in the Quest when Android prompts. Verify the first
join, retry after denial, rejoin after connection loss and the signed LAN handoff
on the physical Pixel and Quest 2.
The shared native Experimenter source has two condition playback tracks and
a procedural VR scene. The current rehearsal inventory contains no video files. The
Atmungsraum build adds encrypted ECG preview and recording replies for the
enrolled LAN controller. Host checks passed. Enrolled Participant Arm binds the participant to the
verified scenario and reserves a native attempt. Quest writes the CSV from
the durable ECG and event streams and serves it over the authenticated export
route. After an abrupt process stop, the writer uses the last validated
checkpoint to publish an aborted CSV and the durable sensor prefixes under
the same recording ID. Physical validation remains open. The enrolled phone can
request recentering of the app scene. Quest confirms it after a focused,
visible OpenXR submission. The physical recenter check is open. Verify the
replacement on Quest 2 and the installed phone before retiring superseded
Viscereality packages.

Quest 2 power preflight on 2026-11-12 confirmed that the built-in display
stayed on past the 15-second off-head sleep timer with a virtual proximity
hold. The device-side watchdog heartbeat advanced without repairs during
the observation. Follow [the awake runbook](QUEST-AWAKE-TESTING.md) to
stop the watchdog and restore the prior policy. After USB debugging was
authorized, the plugged-in Pixel 10 reported its display still on after
two minutes without input. Its screen timeout was extended and Android
reported stay-on active. Clear the Quest lock screen before the physical
rehearsal and repeat the existing app acceptance checks.

### Fast and reliable Bluetooth pairing on Pixel and Quest 2 — 2026-11-13

Minimize the time from opening VR-Probe or choosing a headset to the first
authenticated Quest status. Keep the enrolled device identity and native
gateway checks intact. Use Bluetooth for discovery and hotspot setup, then
hand off to the signed Wi-Fi gateway for control and recording sync.

- [x] Establish the first-pairing blocker on the physical devices. Pixel Chrome
  can scan. The signed Quest 2 rehearsal accepted trusted controller enrollment
  and opened its LAN listener, but its discovery status stayed
  `inactive / remote-required`. The 0.1.20 manifest omits the original native
  panel host required to select Remote control. Time this as a failed attempt.
- [x] Resolve the Quest assembly with the original native panel and command
  gateway selected. The 0.1.20 manifest failed the Remote-host check. The
  corrected manifest declares the native product Activity and both required
  metadata flags. The assembly now rejects a build that omits either flag.
- [x] Build and install 0.1.21 on the active Quest 2 profile. The APK matched
  the installed signing lineage, Android returned the same APK digest, and
  the assembly verified all 67 scenario files and the ARM64 OpenXR loader.
- [x] Launch signed 0.1.21 on the Quest 2 owner profile. Package Manager
  resolved the native product Activity. USB granted and read back the four
  Nearby devices permissions. The app-private hardware serial matched the
  verified headset. The native panel accepted trusted controller enrollment
  and reported a listening LAN gateway. The earlier secondary profile was
  absent from Android's user list.
- [x] Document the pre-fix 0.1.21 result. The headset repeatedly slept
  off-head. The launcher briefly gained focus and reported
  `inactive / remote-required`. The Pixel chooser found no compatible device.
- [x] Make Remote the rehearsal default in signed 0.1.22. A focused Quest 2
  submitted frames and advertised Bluetooth after launch without a mode
  selection. Pixel Chrome listed the headset and the installed PWA completed
  its signed Wi-Fi join after Local Network Access was granted. RustDesk's
  floating overlay had blocked the Chrome prompt. The wearer-facing panel was
  closed using the app's typed operator action and confirmed by native frames.
- [x] Measure the saved-address fast path on the installed Pixel against
  Quest 2 build 0.1.23. Twelve authenticated joins succeeded with zero chooser
  launches. From the saved Wi-Fi address to the first signed status, median
  latency was 137 ms and p95 was 300 ms. These samples exclude app launch,
  Bluetooth discovery, and human interaction.
- [ ] Measure first pairing, warm reconnect, cold restart, a changed Wi-Fi
  address and hotspot fallback on the same physical devices. Record discovery,
  GATT, hotspot join and signed LAN verification separately. Report median,
  p95, completion rate, chooser count and retries per route. Measure operator
  response to permission or chooser prompts separately from software latency.
- [ ] Reduce measured waits and repeat work. Preserve the saved-address fast
  path, the single GATT handoff, and rejection of stale or different headset
  identity. Keep scanning bounded and stop it after selection or cancellation.
- [ ] Repeat the same physical workload after tuning. Confirm that
  cancellation, radio loss, permission denial and reconnect never replay
  session commands or duplicate recordings.

The research account completed Google Cloud onboarding and the separate policy
acceptance. Google Cloud verified the Atmungsraum PWA web client in the
dedicated Drive Sync project, with the production origin, the research account
as the sole test user and the `drive.file` scope. The phone still needs to
authorize that scope after a signed Quest join.

The assembly change and this performance checklist were published in
`b53e856`. [Pages run 43428481356](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43428481356)
succeeded. The live assembly script matched Git, and the build guide,
project checklist and Atmungsraum page passed live checks.

### Quest 2, Pixel and Google Drive collection — 2026-11-13

After the installed Android controller saves a finalized Quest recording, it
must deliver the original Polar H10 ECG and accelerometer (ACC) streams to a
private folder in the designated research Google Drive. Reconcile finalized
recordings in idle before or after a breathing session. Qualify all three
stages with the [operator procedure and stress test](QUEST-PHONE-SYNC.md).
Keep account backup receipts, sign-in details and participant data outside
this repository.

- [x] Verify ADB access and check for the official Google Drive app. Android
  Package Manager reports it installed on the Pixel 10.
- [x] Install the Atmungsraum PWA in Android Chrome and verify standalone
  launcher relaunch on the Pixel. After the Google Play update, Chrome
  157.0.8103.50 opened the installed WebAPK and loaded the live VR-Probe
  pairing route.
- [x] Grant Chrome Nearby devices on the Pixel. The Web Bluetooth chooser
  changed from a permission error to an active scan after Android reported
  both Bluetooth scan and connect granted.
- [x] Register the signed 0.1.20 rehearsal on the active Quest 2 profile,
  grant Nearby devices access for the rehearsal and verify OpenXR frame
  submission after launch.
- [x] Pair the worn H10 with the Quest 2 on 0.1.21. The Polar runtime requires
  coarse and fine location permissions in addition to Nearby devices. After
  granting those app permissions, the native status was fresh and connected,
  and the ACC sample counter advanced across reads. ECG was still inactive.
- [x] Verify a signed join from the installed Pixel and fresh Polar H10 ECG
  and ACC on Quest 2. The phone rendered the live ECG and reported Quest
  ready. The native sensor counters advanced for both streams.
- [x] Install signed 0.1.23. Its app-lifetime audio worker keeps the scenario
  catalog it verified across repeated preparation. The Quest reported a
  verified catalog, the Pixel showed the participant form and scenario
  choices, and an enrolled Arm prepared the authored timeline. The saved
  phone pairing correctly rejected the changed APK fingerprint until the
  operator re-enrolled it.
- [x] Enable participant session controls when the native status requires
  fresh Polar input. On the physical Arm, the old PWA disabled Start even
  though the signed status reported fresh input. The controller now gates
  Start and Resume on freshness and keeps Pause and Finish available during
  signal loss. The targeted regression and existing Quest tests passed.
- [x] Reproduce the full phone control sequence against 0.1.23. Start, Pause,
  Resume and Finish were confirmed and Quest saved both native sensor
  streams. The attempt did not produce a participant CSV. The Pixel sent an
  encrypted participant Arm, but the running panel used a separate JSON
  encoder that omitted the participant plan. The unusually long armed wait
  also put the legacy run above the 128 MiB native export cap. Repeat with
  the fixed encoder and start immediately after Arm.
- [x] Start, Pause, Resume and Finish the 68-second instruction scenario from
  the installed Pixel on Quest 2 build 0.1.24. The signed Quest Arm bound the
  participant to the native writer. A temporary ECG interruption paused
  progress and the session resumed after the live stream returned. The
  terminal receipt saved the participant CSV and both sensor streams with
  zero dropped writer records. The instruction scenario records a
  stopped-early disposition when finished manually.
- [x] Match Quest CSV, ECG and ACC SHA-256 hashes to the Pixel PWA's retained
  ZIP. The Pixel downloaded the CSV into Android Downloads. Its SHA-256
  matched the Quest CSV and it contained 12,506 lines.
- [x] Reopen the downloaded participant CSV in Android Files. The interface
  displayed its filename, Downloads path and size. Shell readback verified
  the original bytes. The Pixel has no CSV viewer to render its rows.
- [x] Preserve the exact native ECG and ACC bytes alongside the existing
  participant CSV. Signed synthetic transfers and durable browser restart
  readback matched both stream hashes in three browser layouts. Physical
  readback subsequently matched both streams across all three bench runs.
- [x] Implement foreground delivery from the committed recording store. Keep
  pending uploads and Drive IDs durable and verify folder, owner, size and
  SHA-256 before saving a delivery receipt.
- [x] Configure the Atmungsraum PWA web OAuth client for
  `https://viscereality.org` in the Drive API project. Verify the designated
  account in the testing audience and the per-file Drive scope.
- [x] Authorize Drive in the installed PWA. The first consent returned without
  a usable token. A second account selection returned the scoped token. The
  PWA verified the account and folder, and a separate Drive metadata query
  confirmed the private destination **Atmungsraum – Quest-Aufnahmen**.
- [x] Verify real automatic Drive upload and readback of a finalized participant
  bundle. The Pixel committed the 22,301,743-byte ZIP in IndexedDB, uploaded
  it while idle and stored a verified receipt for the same SHA-256. Separate
  Drive metadata confirmed one ZIP of the same size in the private folder.
  The archive and delivery receipt survived a Quest app restart and a Pixel
  PWA reload. The Pixel rejoined via its saved address and did not reupload it.
- [x] Repeat the same condition across a Quest process restart and Pixel PWA
  reload, then verify catch-up after a third run finishes with the PWA
  backgrounded. The private Drive folder held exactly three verified ZIPs,
  one per finalized run. The third ZIP downloaded in Android Drive matched
  the Pixel bundle and the three source files on Quest byte for byte.
- [x] Run the targeted automated sync and fault-injection tests. Eight passed,
  including 40 simulated Drive bundles through lost responses and restarts.
- [x] Complete the five-minute `probe-5` scenario on Quest 2 with live H10.
  Quest saved a completed receipt with zero writer drops. Signal pauses
  required restoring ECG and ACC before continuing. The full internal
  recording exceeded the old export limit. The CSV, ECG and ACC totaled
  131,155,067 bytes.
- [x] Update new participant exports to use the ID, UTC Start date, authored
  scenario and run number. Apply the stem to the Quest and demo CSV, Pixel ZIP,
  exported ECG, ACC and manifest and Drive ZIP. Continue reading legacy files.
- [x] Install and verify signed Quest 2 build 0.1.25. Verify the live installed
  Pixel PWA release in standalone mode after Pages deployed commit `28dec06`.
  The native and Pixel fixtures passed 72 Rust, 72 gateway, and 88 Node
  checks, with UI, scenario pairs and project context passing separately.
- [x] Verify the new dated filenames from Quest through Drive. Quest saved a
  new partial `probe-5` test with zero drops. Its first CSV row matched the
  participant ID, UTC Start date, scenario and attempt encoded in its name.
  The Pixel retained the CSV and the identically named ZIP on the session
  stem. The CSV, ECG and ACC ZIP entries matched their Quest SHA-256 values,
  and Drive independently listed the private 2,694,638-byte ZIP. The
  export manifest used the same stem.
- [x] Reconcile the completed five-minute run with the larger native validation
  bound. The Pixel durably saved a 131,158,666-byte ZIP. A PWA reload interrupted
  delivery after the queue reserved a Drive ID. Account renewal restarted
  delivery using that same ID. Independent folder inventory found exactly one
  private object for the completed run. The Drive receipt and Pixel ZIP
  SHA-256 matched, as did the CSV, ECG and ACC entries against Quest.
  The resumed upload completed in about four minutes.
- [x] Align the Quest export limit with the v2 single-CSV payload. Keep the
  128 MiB combined limit for older CSV and separate sensor files. All 74
  focused native recording tests passed, including the v2 size regression.
  The same-signer 0.1.27 APK matched the installed SHA-256 and submitted
  OpenXR frames in Remote. The installed PWA loaded the exact live
  `20261113-polar-streams` release after
  [Pages run 43460541864](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43460541864)
  deployed `19be1ac`. The fresh Bluetooth join reached the signed gateway.
  Cold H10 setup initially returned an unknown ECG start outcome while ACC
  was live. An operator started ECG using the headset's existing command
  and confirmed both streams before Start. The same signed 0.1.27 run saved
  a 1,400,076-byte dated v2 CSV with zero writer drops. Its 8,187 data rows
  included ECG, all three ACC axes, breath and radius. The installed Pixel
  and Drive confirmed the Quest SHA-256, and independent inventory found
  one private object for the run. An independent download through the
  Android Drive app matched the Quest SHA-256 in Android Downloads.
- [x] Complete the authored 15-minute main round on Quest 2 build 0.1.27
  from the installed Pixel. Start, Pause, Resume and Finish reached Quest.
  Live Polar accelerometer tracking drove the applied Viscereality radius.
  The native receipt recorded completion and zero writer drops. Quest
  retained all six streams and the 93,987,682-byte v2 CSV.
  See [full Haupt-Runde verification](VERIFICATION.md#full-haupt-runde-on-quest-2-and-pixel--2026-11-13).
- [x] Verify the full main-round CSV in the Pixel and private Drive. The
  installed PWA pulled the 93,987,682-byte file in about nine minutes
  after Quest finalized it and automatically delivered it as a single CSV.
  The Quest, Pixel and verified Drive receipt agreed on the SHA-256.
  Independent folder inventory found one private copy. Android Drive's
  completed download matched the same SHA-256.
- [ ] Reproduce and fix the phone's pending Resume state when a confirmed
  Resume is immediately followed by another native Polar hold. Keep the
  command journal and the no-replay rule for uncertain outcomes.
- [ ] Qualify physical hotspot, changed-address and repeat pairing timings
  in [QUEST-PHONE-SYNC.md](QUEST-PHONE-SYNC.md). Lost successful Drive replies,
  quotas, shared-folder permission and corrupt chunks have automated coverage.

Both devices were reachable over USB. The Pixel WebAPK reloaded the live
password-free release in standalone mode and opened VR-Probe. The installed
Quest APK matched the signed 0.1.20 build and submitted OpenXR frames at
about 72 fps. The active Quest profile had no enrolled controller and the LAN
endpoint did not respond to the USB forward probe. Google Cloud redirected
to sign-in and the research Drive had no matching destination folder. Next,
enroll the Pixel from the headset and authorize Drive with a Pages-origin
OAuth client. Real H10 capture, Pixel transfer and Drive delivery remain open.

The wearer reported unresponsive Touch controllers. ADB cleared the Meta
report and Nearby devices overlays using their mapped virtual displays.
The rehearsal regained OpenXR focus and submitted frames, but reported no
active Touch interaction profile or physical input. The USB-launched
experimenter panel also showed that the new Android profile lacked the
hardware serial. The signed test APK has no debug identity provider.
The verified ADB and Android serial was added to the active profile's
app-private identity preferences with the test process stopped. Readback
matched and the signed APK relaunched. This was lab provisioning of the
test APK. After the restart, Horizon Shell blocked experimenter-panel
placement while reporting a reprojected OS dialog. Enrollment remains open.
First confirm Touch pointer and trigger in Quest Home, then reopen the
experimenter panel and verify the signed LAN join before recording.

The Pixel and Quest responded over the same Wi-Fi. Chrome could scan but
did not list the Quest while the separate RustDesk screen-sharing prompt
had headset focus. Leave that prompt to its operator. After focus returns,
verify the Quest's Remote discovery, enrollment and signed Wi-Fi join.
Google Cloud still needs interactive sign-in and the research Drive had no
matching destination folder.

All 85 Node tests, the UI state check and all six scenario pairs passed.
Documentation release `e013fc6` passed
[Pages run 43418701040](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43418701040).
All five updated documentation pages showed the new content and the live
Atmungsraum HTML matched the committed blob. The publication checkout was
based on current GitHub main to preserve unrelated primary checkout changes.

Preflight: all 71 existing Node tests passed. ADB confirmed the Pixel; no Quest
USB transport was present. The prior browser path saved the participant CSV
alone. The sensor bundle increment verifies both original streams and preserves
the existing CSV download. All 72 Node tests and the UI/context checks passed.
Confirm both streams contain real sample rows before accepting a hardware run.
At entry, `main` and `origin/main` matched `bcd5da3`; its Pages workflow recorded
a startup failure before any job ran. The sensor bundle release `40f44c7`
passed [Pages run 43121754755](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43121754755).
All nine changed public paths matched the released Git blobs.

Drive runtime release `d171e39` passed
[Pages run 43123800186](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43123800186).
The exact deployment succeeded and all 13 changed live paths matched Git.
All 78 Node tests and the UI/context checks passed. The deployed Chromium
fixture repeated all three layouts, including upgrade from existing browser
storage. The remaining collection checks are the real OAuth setup and attended
Quest/H10/Drive run.

The account export checkpoint `0a72dde` matched local `main` and GitHub's
`main` on 2026-11-12. [Pages run 43175488204](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43175488204)
deployed that commit, and the published project notes matched its Git blob.
Google accepted a one-time export of the selectable products. The local
inventory verified its copies of Drive items, including portable snapshots
of two shared Forms with response archives. Full archive verification
remains necessary before a separate account reset.

### Participant CSV and idle synchronization — 2026-11-08

Protocol and acceptance contract: [QUEST-PHONE-SYNC.md](QUEST-PHONE-SYNC.md).

The Quest remains the recording authority. The operator enters participant
details on the phone and sees the ID from `data-logging.js`. The
enrolled-only encrypted Arm payload binds that identity to the native
attempt before samples are written. Quest reserves the run index and UUID
durably and finalize the participant-named CSV on Quest using DATA-LOGGING.md.
Keep the native original streams available for recovery and provenance. Legacy
rehearsals have no participant ID and retain their original recording identity.

- [x] Use the existing encrypted RQG1 list/describe/read route to reconcile
  all exportable finalized rehearsals with the enrolled browser's durable
  local store. Transfer only in observed idle and let Start preempt transfer.
  Retry on foreground idle and after rejoin. Check all hashes before committing.
  Existing archives can be selected and downloaded individually.
- [x] Verify synthetic Chrome join, ECG, run controls, idle-only sync of an
  earlier and a newly finalized run, corrupt-transfer recovery, durable reload,
  and exact ZIP download. See VERIFICATION.md.
- [x] Close and restart Chrome with the same profile, rejoin the same Quest
  identity with a new simulated native epoch, and finish a second rehearsal.
  Confirm all prior copies remain and only the new run's bytes transfer.
- [x] Route the existing participant form into VR-Probe and pass its
  normalized identity in a versioned encrypted, generation-bound Arm.
  Reject missing identity for the participant recording mode. Confirm that
  identity is frozen by Arm; completed files retain their original IDs.
- [x] Extend the native writer and export allowlist for participant CSVs,
  including finalized partial runs. Host tests verify durable run allocation,
  a native CSV with sensor timestamps and gaps, and hash-bound export. The
  browser fixture verifies CSV-only transfer with the legacy ZIP fallback.
- [x] Resolve, build and install the participant-capable rehearsal. Version
  `0.1.9` is the same-signer upgrade of the isolated rehearsal, with the
  ARM64 OpenXR loader. Installed APK SHA-256 matches the signed build.
- [ ] Launch the installed APK and verify file publication and retrieval
  from the real Quest and phone.
- [ ] With a consenting wearer and synthetic ID, verify fresh real H10 ECG,
  run and abort receipts, participant metadata and CSV bytes on Quest, then
  repeat the complete control and sync path on the installed Android phone.
  Use the same ID for two same-condition runs across full Quest and PWA app
  restarts. Check distinct UUIDs and filenames with increasing run indexes,
  finish once with the sync controller absent, rejoin, and verify one durable
  copy per run and an actual Files download. The USB-connected Quest was
  asleep after the verified install. Wake to Quest Home and clear system
  prompts, then approve Nearby devices for the installed rehearsal. Both
  Bluetooth permissions are still ungranted. Run the prepared private
  desktop H10/CSV smoke before physical Android phone qualification.

Desktop launch preflight on 2026-11-08 confirmed the installed `0.1.9` launcher
over USB. The physical display was off. Live VR-Probe opened in Brave and
showed the startup unlock dialog. At preflight, `main`, `origin/main`, and the
successful Pages build matched `a808a44`. Wake to Quest Home and unlock the
controller before launching and verifying the native scene and gateway join.

The chosen phone target is the browser's origin-scoped IndexedDB, with
explicit download for the phone filesystem. Sync runs while an enrolled
controller page is visible and connected. A closed or suspended PWA
catches up after reopening and rejoining. Keep Quest copies as the source of
truth. The subsequent [Drive delivery step](QUEST-PHONE-SYNC.md#google-drive-delivery-acceptance)
uses the verified copy after the Quest-to-Pixel transfer.

The browser Arm increment `2eaf0b5` passed
[Pages run 42491480387](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42491480387).
Nine changed live files matched the commit bytes. All three participant
Arm and restart fixture layouts passed on the deployed URL. Native source,
signing receipts, installed package identity and the device test runner stay
in their private owner lanes. For the next attended hardware test, follow
[Keeping Quest awake during testing](QUEST-AWAKE-TESTING.md) before launch and
record separate transport, display, foreground, and target frame evidence.
The runbook links and PowerShell syntax passed local checks. The QFM and Fleet
examples were checked against pinned provider sources. Physical qualification
of the hold and restoration remains part of the attended rehearsal.
Runbook commit `ed7b9bb` reached `origin/main` and deployed in
[Pages run 42493012092](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42493012092).
All four live Markdown files matched the commit bytes. The rendered runbook,
its three entry links, and the Atmungsraum page returned HTTP 200.
The Codex profile now has the user-scoped `$keep-quest-awake` skill. Skill
Creator validation and the installed Codex skill inventory passed. A dry run
covered protected lock recovery, watchdog lifetime, and prior policy restoration.
Use the skill for the next attended rehearsal preflight.

### Remote-only controller and native data — 2026-11-08

- [x] Keep the core Experimenter flow: service-filtered Bluetooth discovery,
  verified Wi-Fi join, manual address fallback, native Polar setup, native
  condition selection, session commands and original receipts. Bluetooth
  supplies the Quest address. The phone hotspot and the Quest's Wi-Fi
  connection are set in their operating systems.
- [x] Select the original native host and Remote-only startup in the isolated
  rehearsal build. Retain a build switch to restore local control. Keep the
  native tuned payload, two rehearsal conditions and original recording
  writer in their owning build lane.
- [x] Render an enrolled Quest's bounded live Polar H10 waveform with sample
  timestamps, microvolts, gaps and expiry. The browser does not store raw
  preview samples. The NeuroKit demo remains a separate simulation.
- [x] Bind completed-run retrieval to the observed Quest recording ID. After
  Quest returns to idle, verify the eight native files and offer the original
  bytes in a ZIP. Preserve the observed ID for a manual retry after rejoining
  the same Quest. Use the authenticated, encrypted enrolled route for both
  waveform and recording. No upload or browser data writer participates.
- [x] Install the signed Remote-only rehearsal on Quest 3. Package Manager and
  the installed APK digest matched version 0.1.7. The running owner reported
  Remote at startup. Desktop Chrome joined the actual gateway using the
  enrolled controller and authenticated both status and an encrypted preview
  response. The Android permission prompt interrupted immersive presentation.
- [ ] Verify the signed package on Quest 2 with a worn H10. Run a native
  rehearsal, confirm the changing browser waveform and reopen the resulting
  phone ZIP. Verify Bluetooth discovery, Android Local Network Access, hotspot
  reachability and actual storage on the phone. Retain physical evidence in
  the native release lane.

Browser release `bc4b445` deployed in
[Pages run 42425988651](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42425988651).
All 13 changed public files matched the release. The 67 Node tests, UI and
context checks, six scenario consistency checks and three live encrypted
browser fixture layouts passed. Next, approve Nearby devices in Quest, return
to the scene, and connect the worn H10 from the browser. The real preview
response observed so far contained no live samples.

The [Experimenter controller contract](https://github.com/MesmerPrism/MesmerPrism.github.io/blob/d64257af156dfddd4e1ede01a007c032a2816a2c/viscereality-control/README.md)
is the source for discovery and remote-control parity. The standalone monitor
for breathing, advanced relay/network setup, bias editing, and Showcase
tuning are separate surfaces. Study scenarios still require their native
composition and timing qualification. The current practice/verification
conditions use the two packaged Quest rehearsal tracks.

Atmungsraum is exclusively the external operator interface on PC and smartphone
in every integrated run. It sends bounded operator requests and displays
authenticated Quest state and receipts. The wearer sees only the Viscereality
scenario in Quest. The Polar H10 connects directly to Quest over Bluetooth.
Quest owns acquisition, transitions, stimulus, cue timing and recording. The
standalone synthetic demo is a separate test path. Test the controller on this
PC first. The physical acceptance gate remains the installed Android Chrome PWA
over trusted Wi-Fi.

- [x] Record the approved handoff, product boundary and ordered acceptance
  checklist. Keep private pins, payloads, signing and device receipts with the
  native product owner. Preserve Showcase as the visual and rollback baseline.
- [x] Resolve, sign and install an isolated Quest experiment APK with the existing
  session owner, Remote VR preparation, one fixed synthetic condition and
  the exact `https://viscereality.org` origin. Use a separate package, retain
  native Remote consent, and prepare it before handing the headset to the wearer.
- [x] Expose explicit phone rehearsal entry with Experimenter BLE discovery
  and a manual LAN gateway fallback. Verify the intended device, build and live
  owner before enabling actions. Keep the browser demo available separately.
- [ ] Verify Arm → Start → Pause → Resume → Finish from the installed phone.
  Match original receipts and native state, keep an unknown result visible,
  and prove reconnect never repeats a command. Verify the native clock pauses
  and that the scene is the only app content shown to the wearer. Use synthetic
  data without demographics, browser audio or browser recording.
- [ ] Qualify headset-owned H10, actual timestamped ECG with units and gaps,
  an encrypted route for study data, and completed/aborted native recordings.
  Join data by opaque run identity and use the shared naming contract for native
  exports.
  Abort on continuous ECG loss beyond 30 seconds, including during pause.
- [ ] Qualify the native 900-second main round and 26–41 second calibration
  window, followed by the practice scenarios and unlimited Anleitung. Qualify
  ACC measurement, final narration, study protocol and iPhone separately.

The browser exposes **VR-Probe** and requires the Quest serial and exact
rehearsal build fingerprint. Website release `994c0b8` passed the exact Pages
deployment and live Chrome fixture checks. The isolated APK passed signing,
manifest, Java source and Quest 3 installation checks. The build and private
install receipts remain with the native product owner. See VERIFICATION.md.

The current browser sweep covered all five demo session choices, real CSV
downloads, saved-file reload, interrupted-run recovery, storage failure and
fixture sharing. It also corrected filename wrapping on the saved CSV screen
at 320 CSS px with 200% text. Runtime commit `4309c6f` deployed from `main` in
Pages run `42408436234`. The live 121-case layout matrix and crash/quota checks
passed. See the UI and CSV saving sweep in VERIFICATION.md.

The [MesmerPrism device workflow](SKILLS.md#quest-device-operations) is
installed in Codex and verified at public source revision `39a4641`. Its
playbook resolver verified the local source. File Manager confirmed an
authorized USB Quest 3 and observed its display go from off to on. File
Manager's power status does not report the foreground activity. A bounded,
read-only serial-scoped ADB check then found Quest Home resumed alongside
the Browser panel, with no lock or Guardian activity in the observed state.

Current device smoke test: serial-scoped ADB launched the isolated Viscereality
rehearsal on Quest 3. The original APK stalled at loading because the ARM64
OpenXR loader was absent. The locked rebuild explicitly packages that loader.
The same-signer update passed installed-version readback and a Quest screenshot
shows the rendered scene. The next gate is native Remote selection followed by
Start and End from the actual PC browser controls. Keep Quest and browser
screenshots, native command receipts and build inputs private. Use synthetic
recording inputs for this rehearsal.

The previous installed rehearsal reports both Bluetooth runtime permissions
granted. The Remote-only successor selects the required native host and starts
in Remote. A fresh package may still require Android Bluetooth permission and
one operator enrollment to unlock private data.

### Quest-owned H10 with browser control

The sensor link is **Polar H10 → Quest over BLE**. The control link is
**Atmungsraum in Chrome → Quest experiment gateway**. Use desktop Chrome for
the current round, then the smartphone browser as the operator surface.
Keep full ECG on Quest until the enrolled export is requested. Use a
consenting test wearer and a synthetic identifier.

- [x] Run the current browser UI and save checks on PC Chrome. The 61 Node
  tests, 121 layouts, five demo session CSVs, crash/quota checks and five
  signed VR-Probe fixture cases passed. These exercised synthetic data.
- [x] Verify the installed rehearsal APK includes Quest-owned Polar H10
  acquisition and the experiment command gateway. The installed rehearsal
  package reports Bluetooth scan and connect grants. It omits the recording
  export capability.
- [x] Put Polar setup in the external controller's **VR-Probe** screen. After
  joining Quest in Remote mode, **Polar H10 verbinden** asks Quest to scan,
  select the only detected strap, connect and start ECG using the
  [native Polar setup](https://github.com/MesmerPrism/rusty-quest/blob/99352620fd79ec5cdcee71b47c2ba4056241af16/docs/REMOTE_POLAR_SETUP.md)
  contract. Several candidates require operator disambiguation by leaving
  only the intended strap active. The button uses the existing authenticated
  command and original receipt flow. A disconnect or background transition
  never continues the setup sequence. The headset handles Android permission
  approval, Bluetooth and acquisition. The browser shows only readiness.
- [ ] Install the Remote-only rehearsal APK and open its VR scene.
  Join from PC Chrome and click **Polar H10 verbinden** with the test wearer
  wearing the strap. Verify fresh native ECG and keep device receipts and
  screenshots in the private evidence lane. This is the next unfinished action.
- [ ] Connect desktop Chrome to the identified Quest gateway. Verify Starten,
  Pause, Fortsetzen and Beenden from original native receipts. Verify that
  native ECG continues during pause and continuous signal loss beyond 30
  seconds aborts the run. For waveform display, require an approved encrypted
  route for bounded native samples with timestamps, units, freshness and gaps.
- [ ] Verify that Quest produces completed and partial recordings using the
  shared filename and identity contract. The controller can request export and
  show the native result once transfer is qualified. Then repeat controller
  acceptance on Android Chrome, including Local Network Access.

The browser demo still uses NeuroKit replay. The enrolled Quest route now
accepts a bounded native preview and completed recording in addition to
the readiness flag. Qualify both on hardware before study use.
Host verification covers 65 Node tests, six scenario consistency checks, ten
signed Chrome fixture cases across supported viewport/text sizes, and the
synthetic demo save sweep. See [VERIFICATION.md](VERIFICATION.md).
Pages run [42420247829](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42420247829)
deployed the Polar controls in `2aeb0f1` from `main`. A slow-load regression
found during live verification is addressed by withholding the Quest
connection form until its handlers are ready. The physical Quest gateway
was still unreachable at the last PC check. Confirm Remote and scene
readiness before sending a device command.

## Primary goal: smartphone progressive web app — 2026-11-06

### Brave desktop install and taskbar pins — 2026-11-07

The user reports repeated additions and blank taskbar icons in desktop Brave.
Brave 157 registers the stable Atmungsraum app and creates its ICO resource.
One orphaned Windows pin pointed to the ICO removed by uninstalling a temporary
QA profile's app. Its app ID and QA path were verified before the native Unpin
action. A follow-up scan found zero Atmungsraum taskbar shortcuts.

- [x] Reproduce the missing installed state on a regular browser tab. The live
  manifest returned an empty related-app list after an actual Brave install.
  Adding the manifest self relation returned the same canonical app ID and URL.
- [x] Check the browser's installed-app state on load and after accepted prompt
  consent. Keep the Viscereality manifest icons and canonical identity. Add
  Brave's native app list to the desktop install help.
- [x] Verify 54 Node tests, UI/context checks and the Brave help layout at three
  viewport sizes, including 200% text. Verify install hiding across reload with
  a fixture matching Brave's observed native result. Gate native install QA
  behind a separate OS account or VM and check the host taskbar after testing.
  Independent review confirms focus returns to the folder on a late match.
- [x] Publish the desktop fix and verify the Pages release and live Brave UI.
  Runtime commit `6528b47` deployed in Pages run `42403873448`. Both entry
  URLs resolved to `/atmungsraum/`, all changed files matched the commit, and
  the safe Brave probe passed on the live site with zero native installs.

Entry was `f3a4751` on `main`, synchronized with the remote. Evidence is under
ignored `output/playwright/brave-pwa-*` and `brave-install-ui-*`. The branded
taskbar result in the user's regular Brave profile remains a manual check.

### Smartphone controls and install action — 2026-11-07

The user's phone screenshot requests at least twice-sized top controls; their
follow-up requests an install button beside the bottom-right folder. Double
both top surfaces and their artwork on narrow/coarse-pointer phone layouts,
preserving their equal size, safe-area anchors and existing meanings. Keep the
folder at its current size. The install action uses a browser-provided prompt
on a user click, with a native help dialog for browsers without that API.
Short landscape uses an accessibly named install icon above the folder, keeping
the central controls clear. Reuse existing Pretext and native dialog semantics.
The screenshot shows a browser tab; it does not prove installed-app relaunch.

- [x] Add prompt lifecycle/failure tests, then implement the install action and
  double-sized top phone controls, preserving unrelated sharing work.
- [x] Pass 113 rendered phone/screen cases, native help/keyboard/focus checks,
  200% text/spacing, long install labels and playback-preserving dismissal.
- [x] Pass all 32 tracked tests, UI/context and whitespace checks; independent
  review finds no actionable issues in the staged product diff.
- [x] Publish this scoped increment and verify exact Pages deployment/live code
  and rendered controls.

Entry: `main` at `ad3d432`, equal to fetched remote. The prior physical-phone
installation/relaunch acceptance remains open; this is concrete UI feedback,
not that acceptance. Ignored evidence: `output/playwright/pwa-controls-*`.
Initial publication: `b16a49a` was pushed, then concurrent sharing release
`0d0e170` superseded its cancelled Pages run. Containing run 42309917241 succeeds,
deployment API status is `succeed`, and all 12 PWA-control paths match that
published commit. The integrated suite passes all 45 tests. Concurrent
documentation release `41c3a76` preserves this increment.

Final enlarged-text review reproduces an install-help autofocus bug: the old
bottom close button scrolls the 320×568 dialog to 1832px, hiding its start.
Moving close into a measured header makes initial scrollTop zero; a failing
browser assertion now passes and all 113 phone cases re-pass, including the
new opening-position check. Desktop smoke checks pass at 768×1024,1280×720 and
1920×1080. Independent review repeats the zero-scroll check and finds no
actionable issues. This follow-up changes the dialog/header and cache tokens only.
Final runtime release `5ff34d3f5ceaed7e15ab306d6df32b5d6db84e16` completes
[Pages run 42310953670](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42310953670)
successfully, with deployment API status `succeed`. All 12 PWA-control paths
match its Git bytes and both app entry URLs return 200. Native Chrome repeats
all 113 phone cases on the public origin, including initial help scrollTop0,
200% text/spacing, prompt lifecycle and playback-preserving dismissal; three
desktop smoke checks pass. Manifest/icon checks and a real isolated desktop
installation/standalone launch also pass, with the browser user display
preference explicitly set. No page errors occur. Ignored receipts:
`pwa-live-5ff34d3.json`, `pwa-controls-live-results.json`,
`pwa-controls-live-phone.png` and `pwa-controls-installed-live-results.json`.
`main` equals remote at this release before the evidence-only follow-up. The
increment is complete; the overall PWA goal remains open for physical-phone
installation/relaunch acceptance. Next: collect that phone/browser result.

Centering follow-up: a direct geometry regression shows the enlarged top-only
clearance shifts the bounded round menu down27.99px. The shared vertical reserve
now uses the maximum of top size + top safe inset and dock height + bottom safe
inset on both edges. A separate unequal-inset case first shows +6.49px and then
also returns -0.008px after including the insets in that maximum.
The same check returns -0.008px, preserving the standing viewport-midpoint
contract. All115 cases pass with added strict center checks for bounded ECG,
round/option and playback targets. No install/session/data logic changes.
Independent review confirms the unequal-inset correction with no remaining
actionable issue. Documentation deployment77e0e72 timed out in syncing_files;
its backend is deployment_cancelled. The fresh correction includes those notes.
- [x] Publish this isolated CSS/cache correction and verify its exact deployment,
  live bytes and centered phone layouts. Preserve concurrent Discord-password work.
Centered release `0320d1c1231396caf458a089ae7e60115b781aab` completes
[Pages run 42313937741](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42313937741)
with deployment API status `succeed`. All12 control/evidence paths match Git
bytes and both entry URLs return200. Live Chrome repeats all115 cases including
bounded-target centering, simulated unequal insets and enlarged-text help, with
zero page errors. Ignored receipts: `pwa-live-0320d1c.json` and
`pwa-controls-centered-live-results.json`. `main` equals remote at this release
before the evidence-only follow-up. Unrelated Discord-password edits and the
private attachment are excluded. Next: the existing physical-phone
installation/relaunch acceptance; no further phone-control changes are pending.

The UI's primary delivery target is a progressive web app (PWA) for smartphones.
Installing it adds a Viscereality-logo home-screen icon that opens the live
`https://viscereality.org/atmungsraum/` app in standalone mode. Keep its existing
purpose as the phone control surface for the corresponding VR headset UI.
Phone touch interaction, portrait/landscape, safe areas and installed-app
lifecycle take priority; desktop browser access remains supported.

This increment supplies install metadata and icons, using the existing static
HTML/plain JavaScript stack. It remains online: no service-worker cache,
offline session promise, native widget extension or app-store wrapper is needed
for the requested home-screen launch. Keep participant storage and session
visibility/abort rules intact. Physical Android/iPhone installation and real
headset/H10 qualification remain acceptance work for the overall product goal.

- [x] Add a scoped, stable manifest and iPhone metadata; derive ordinary,
  maskable and Apple icons from the existing Viscereality `Images/logo.png`.
- [x] Verify browser manifest parsing/installability, icon decoding and launch
  URL, smartphone rendering and existing runtime checks; review the exact diff.
- [x] Publish only this increment, follow its exact Pages deployment and verify
  live manifest/icons and app behavior. Preserve concurrent sharing work.
- [x] Install and launch from the Viscereality icon on the target Pixel 10
  running Android 16. Chrome installed the canonical Atmungsraum web app.
  Android resolved the installed WebAPK launcher and the foreground WebAPK
  activity opened without the browser address bar. iPhone remains untested.

Entry: canonical `main` at `5ae2d5f`, equal to fetched `origin/main`. Sharing
changes are already unfinished in the worktree and must not be swept into this
release. RTK is unavailable; use bounded native output.

Local evidence: all 28 tracked Node tests, UI/context and staged whitespace
checks pass. Chrome 156 parses the manifest with no errors and returns no
installability errors; all icons decode at their declared sizes. Four phone
viewports (320/390 portrait and 568/844 landscape widths) have no horizontal
overflow or clipped patient labels. An isolated browser profile installs and
launches `/atmungsraum/#patient` with actual standalone media mode after the
browser's user display preference is set to standalone (Open as app/window).
CDP's default install initially opened a browser tab; this test does not prove
physical-phone default installation behavior. Icons are opaque; maskable logo
pixels fit the safe circle. Independent review finds no actionable issues.
See VERIFICATION.md and ignored `output/playwright/pwa-local-results.json`.
Release `02217feda1df9d99de830e50867ca32ce94b271c` is pushed on `main` and
was equal to `origin/main` after publication. Exact Pages run
[42260860687](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42260860687)
completed successfully; the deployment API reports `succeed`. All 13 changed
public files match Git bytes; PNG/manifest MIME types are correct, and slash /
no-slash entry URLs return 200 at the canonical slash URL. Native Chrome repeats
the installability, icon decoding, four phone-size and standalone launch checks
on the public origin with no page errors. Its explicit user display preference
is recorded. Ignored receipts: `pwa-live-02217fe.json`, `pwa-live-results.json`
and `pwa-live-phone.png` under `output/playwright/`.

The earlier physical-device blocker cleared on 2026-11-13. Chrome's **Add to
home screen → Install** flow created an Atmungsraum launcher icon on the
Pixel 10. Opening the icon launched the installed web app in standalone mode
and showed the shared-password prompt. Mobile MCP supplied the phone input.
The standard RustDesk USB session showed live frames but reported keyboard
permission off. The separate RustDeskMCP endpoint was disabled. Next, use the
installed app for the existing attended Quest 2 and H10 checks.

Documentation release `e2953d960bc61277b08fff81329f1b0be214985a` completes
exact Pages run 42306629457 with deployment API status `succeed`. Both changed
public notes match Git bytes and both app entry URLs return 200. Read-only Git
checks confirm `main` equals remote at that commit before this handoff update.
Ignored receipt: `output/playwright/pwa-live-e2953d9.json`.
Overall H10/headset integration remains a separate open product goal. This
evidence follow-up changes documentation only and follows the same Pages rule.
Unrelated unfinished sharing edits remain in `.gitignore`, `app.js`,
`data-files.js`, the sharing section of `index.html`/this state, `style.css` and
new link/Worker/test files; they are preserved outside the PWA commits.
2026-11-11 phone preflight: Windows detected a physical Pixel 10 over USB and
`adb devices` initially reported `unauthorized`. At preflight, `main` and
`origin/main` matched `1a80572`, and its exact
[Pages run](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42516879788)
succeeded. The live entry linked the standalone manifest, the manifest matched
the checkout, and all three manifest icons decoded at their declared sizes.
USB debugging was subsequently authorized. A serial-scoped ADB shell verified
the Pixel 10 running Android 16. None of its three installed Chrome WebAPKs
declared the canonical Atmungsraum URL. Next: open the live URL in Android
Chrome, install, and verify standalone relaunch from the Android launcher.

## Browser session background music — 2026-11-08

- [x] Package a browser-ready copy of the supplied `ji3ji3` track with an empty
  spoken-word TXT. Load on Starten, loop during the browser demo, and follow
  pause, resume, completion, navigation and sensor timeout. Quest retains its
  native audio ownership.
- [x] Verify the full MP3 decode, unchanged scenario schedules, controller
  lifecycle and rendered browser playback. See [VERIFICATION.md](VERIFICATION.md#browser-session-background-music--2026-11-08).
- [x] Verify the exact Pages release, the deployed MP3 bytes and the live browser
  playback path.

Browser release `afe19f2` reached `origin/main` and deployed successfully.
All ten changed files matched the deployed bytes, and Chrome confirmed the
live loop and session controls. [Pages run 42516486897](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42516486897)

Next for device qualification, repeat Starten, pause/resume, the loop boundary
and session exit on an installed phone. The original WAV and the untracked local
attachments stay outside Git.

## One startup unlock for folder sync — 2026-11-08

The startup behavior in this record was superseded on 2026-11-13. See
[password-free startup](#password-free-startup--2026-11-13).

Use only **Ungesendete Daten senden** and **Alle Daten senden** for the
browser's saved session CSVs. The shared password now unlocks the entire app
interface at startup, including the Quest and individual download routes.
Access stays in tab memory. A supported browser can save and silently restore
the password. GitHub Pages still serves static assets publicly.

The startup password accepts a nonempty single line without a length limit.
The version 1 envelope was re-encrypted with the operator's chosen password.
Local checks verified the original Discord destination, rejection of the prior
password, and browser unlock with the new password and a synthetic password
longer than 1,024 characters. The startup fixture passed on the default,
CSV recovery, and VR-Probe routes.

Release `aa5d4c8` passed [Pages run 42498919176](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42498919176).
All seven published files matched the commit. Live Chrome accepted the chosen
password on all three entry routes and rejected the prior password on VR-Probe.

- [x] Remove the service setup controls and reuse the unlocked session for both
  sends. Preserve the exact file selection, encrypted ZIP and checkout receipts.
- [x] Support the version 2 encrypted envelope with the existing Discord webhook
  and central upload relay credentials. Discord still receives the keyed URL
  directly from the browser.
- [x] Check synthetic startup, saved credential restore, both send actions, file
  bytes, acknowledgement, errors, focus, reflow and concurrent tabs. Restore a
  two-chunk synthetic ZIP from live Wormhole via the local Render adapter.
- [x] Require the startup password before opening any app route. Verify
  wrong-password retry, Escape, saved credential restore, and focus.
- [ ] Enable automatic upload under the GitHub Pages-only constraint. Direct
  browser upload from the phone carries the Pages site's origin. Wormhole
  returned HTTP 405 to the upload preflight and omitted cross-origin
  permission. Wormhole's manifest registers an installed share target that
  can receive a ZIP after a user selection. Its same-origin handoff provides
  no result URL to Atmungsraum. Keep the public version 1 envelope and leave
  automatic sends unavailable until a supported browser-only handoff exists
  or the hosting constraint changes. Sites registration remained private
  and unpublished. See [the delivery boundary](../sharing/README.md).

The public send button also passed an end-to-end Chromium fixture against a
local relay and real Wormhole/Backblaze. The uploaded 9,800,146-byte synthetic
ZIP restored byte-for-byte across two encrypted pieces. The fixture checked
the Discord payload and simulated acknowledgement. It made no Discord post.

Required startup unlock is in `7974cff` and live through containing commit
`e38dd8af9c5a92e1f192854fc4258be31079511c`.
[Pages run 42493720473](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42493720473)
and the deployment API succeeded. All 14 changed public files matched Git.
Live Chromium passed the three entry routes, saved credential fixture,
two-button send fixture and actual encrypted envelope check. Mobile Chromium
emulation confirmed that a request from the phone layout still carries the
Atmungsraum origin and the browser rejects the direct Wormhole request.

Browser release `6ff82b293ef6ff40ddbeb56e8d06caf430858f02` is on main.
The commit-specific Pages deployment reported `succeed` and all 15 changed
public files matched Git. The [Actions run](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42491168761)
reports cancelled. Main advanced with a concurrent Quest change that includes
this release. The deployed Chrome fixture passed with eight simulated uploads and
seven intercepted Discord posts. The actual public version 1 envelope unlocks
with the existing password. With that envelope the folder reports the missing
relay and makes zero provider requests. The untracked local attachments stay
outside Git.

## Folder delivery: new files or all files — 2026-11-07

The workflow above supersedes the setup and password timing in this historical
verification record.

The folder now sends saved CSVs to Wormhole and posts the URL alone directly
to Discord. This supersedes the earlier QR, manual sharing and page-memory
receipt flows below. Individual CSV downloads remain at `#daten`.

- [x] Show two send choices, pending count, progress and confirmed delivery.
- [x] Track exact file revisions in browser-local JSON after Discord accepts
  each post; support a deliberate resend of all files and prevent parallel
  sends across tabs. Preserve data writers and recovery.
- [x] Decrypt the public webhook ciphertext on-device before any upload;
  preserve optional browser password-manager saving after delivery.
- [x] Verify the browser with synthetic CSVs, rejected uploads/posts, exact
  archive bytes, exact URL payload, cancellation, reload and concurrent tabs.
- [x] Verify the exact Pages deployment and the public folder flow.
- [ ] Verify hosted Render with its actual URL after the operator deploys the
  Free service. Use the ignored private setup page and see sharing/README.md.

The checkout receipt update gives each acknowledged send its own browser-local
JSON record. **Neue Dateien senden** combines those receipts with the earlier
history. The revised 52-test suite and synthetic browser fixture pass locally,
including legacy receipt compatibility, retries, repeat sends and live counts
across tabs. Main and origin/main matched checkout release
`e3eac238a89cc30a15c7fc497eb243b20d64cb74`, and the exact
[Pages deployment](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42401759135)
succeeded. All nine changed public files match Git and both entry URLs work.
The live browser fixture confirms six separate receipts across eight simulated
uploads and seven simulated Discord requests. Hosted Render verification
remains the next step when the operator has its URL.

Main and origin/main matched the initial folder feature release
`d02cf262d9a530d286650bb11dfdecdaba63fb7e`.
[Pages run 42400503595](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42400503595)
and the commit-specific deployment API report success. Both live entry URLs
work; all 21 changed public files match Git and all three removed QR assets
return 404. The public-browser fixture passes five layouts and seven uploads
with six fixture Discord posts, no page errors and no real data sent. The
published encrypted envelope still decrypts to the configured destination.
Next: obtain the signed-in Render deployment URL and verify its real upload
with a synthetic archive. This evidence follow-up changes only documentation.

## Render Free automatic upload — 2026-11-07

The user selects a different free host and wants to keep automatic ZIP upload,
QR generation and the direct password-gated Discord send. Render Free removes
the Cloudflare account requirement for the upload route. Preserve the manual
fallback and existing host support.

- [x] Add an upload-only Node adapter and Render Free Blueprint. Scope the relay
  to the production site origin, one private key and existing bounded provider
  routes. Render receives no Discord credential.
- [x] Accept Render URLs and wake the service with a credential-free readiness
  check. Wait up to two minutes, with cancellation and no retry of mutations.
- [x] Verify 51 Node tests, five browser fixture layouts and live Wormhole
  upload/download via the local Node adapter. The 9,800,146-byte synthetic ZIP
  round-trips in two encrypted pieces, first piece exactly 8 MiB. No Discord
  test message or participant data was sent for this host change.
- [x] Prepare the ignored private Render setup page and access key; record the
  signed-in Free deployment steps in sharing/README.md. Independent review
  found no blocking issues.
- [x] Publish and verify the exact Pages release.
- [ ] Deploy in the user's Render account and verify the actual `onrender.com`
  service with a synthetic archive. The public Pages code and a local
  live-provider test are not proof of hosted Render operation.

Release `d5e80b649c90fdf8801bb7a3807c90e21d94c90f` synchronized on main and
deployed by [Pages run 42373102567](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42373102567).
The Pages API reports `succeed`; all 16 changed files match Git bytes, both
Atmungsraum entry URLs respond, and all five live browser fixture cases pass.
No Render credential or connected Render tool was available. Next: use the
private setup page, confirm the Free service in Render and obtain its URL.
Hosted verification: `node output/playwright/check-render-network.mjs` with the
service URL as its argument. This does not send Discord messages. See
VERIFICATION.md and ignored `render-upload-local.json`,
`sharing-release-d5e80b6.json` and `render-sharing-results.json` receipts. This
evidence follow-up changes only documentation; verify its Pages publication.

## Password prompt and encrypted direct Discord — verified, 2026-11-07

The user explicitly replaces the Cloudflare notification hop: host the webhook
only as an encrypted file on GitHub Pages, request a password immediately before
sending, decrypt on the device and send directly to Discord. The message contains
the complete Wormhole URL/key. Preserve ZIP/download/native share/QR and the
optional existing automatic Wormhole upload. Cloudflare ownership is no longer
required for Discord; its earlier blocker below applies only to optional upload.

- [x] TDD AES-256-GCM/PBKDF2-SHA256600k envelope encryption/decryption, fresh
  salt/IV, bounds, wrong-password/tamper rejection and direct acknowledgement.
- [x] Generate a strong144-bit password privately; publish only encrypted JSON.
  The ignored setup page provides the password. No raw webhook/password in Git.
- [x] Add masked send-time prompt; Cancel/Escape/wrong password sends nothing.
  Clear input on close, keep focus inside the parent modal, cancel stale sends
  and retain QR/downloads. No password/unlocked webhook app persistence.
- [x] Enable optional native password-manager saving/autofill after accepted
  Discord delivery. Cancel, wrong password and rejected delivery do not request
  saving. The operator still confirms each send.
- [x] Real Chrome at canonical origin/local source: wrong password yields zero
  Discord requests; correct password posts directly and message readback matches
  the synthetic share. Network origins are only the site and Discord, with empty
  Worker settings. No relay or mocked Discord response is used.
- [x] Finish review/regressions, publish and verify exact Pages/live behavior.

49 Node tests, UI/context/whitespace, five prompt/reflow, five automatic upload
and seven manual QR browser cases pass. Password-manager call tests use a
synthetic browser API fixture; native user Save/autofill depends on settings.
Independent review finds one focus-return defect, reproduced red then fixed;
no encryption/secret-handling blocker. See VERIFICATION.md for final evidence.
No new hosting account is needed for this delivery route. Never claim the
webhook is unrecoverable after unlocking: public ciphertext permits offline
password guessing, and a password holder can extract the decrypted webhook.
Changing the password alone cannot revoke such access; rotate the webhook.
Release `3d861b18070d74f68a3f172d37429003b15d3224` is synchronized on main.
Exact [Pages run42362312447](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42362312447)
and deployment API both succeed. All18 changed files match public Git bytes and
both entry URLs return200. Native Chrome from the actual published page makes
zero Discord requests for a wrong password, then sends the correct share URL
straight to Discord and verifies message readback. The actual Wormhole page
restores the9,437,363-byte synthetic ZIP with SHA-256
`808a69e028c3d770bdda050982a035cdf04c0143df56f16908f3e8df11097a6f`.
Network origins are only the site and Discord for sending, with no Worker
credentials. All five live prompt/reflow cases pass. An isolated real Chrome
credential-store request fulfills after successful opt-in; cancel, wrong
password and rejection do not request saving. Native Save/autofill is still
controlled by the user's browser and physical phone acceptance remains separate.
No participant data or plaintext credentials are published. Ignored receipts:
sharing-release-3d861b1.json, discord-password-live-network.json and
discord-password-native-api-results.json under output/playwright/.

The requested implementation and exact user setup instructions are complete.
Next: use the private setup page's password, select browser saving if desired,
and confirm the browser's own Save prompt. The original automatic Wormhole
upload integration was previously verified with a configured relay; its durable
hosting setup remains optional/user-controlled as recorded below. The current
Discord delivery does not depend on that ownership. Keep unrelated attachments
untracked. This evidence follow-up changes notes only; verify its publication.

## Automatic Wormhole upload — optional relay ownership checkpoint

The user supplies a Discord webhook and explicitly requests automatic forwarding
of Wormhole URLs. The share button packages every saved browser export into the
existing ZIP, encrypts/uploads it to Wormhole, retains its link/local QR and
forwards the complete keyed URL to that fixed destination. Keep manual upload,
native sharing and downloads. Recording/identity/filenames are unchanged.

- [x] TDD bounded authenticated Worker notification and restricted upload routes;
  fixed destination, private secrets, exact site Origin and no provider-error logs.
- [x] Implement interoperable browser encryption/upload using official pinned MIT
  crypto, two passes over an immutable ZIP and bounded 8 MiB ciphertext pieces.
  Acknowledge final service completion before showing or forwarding a share.
- [x] Integrate accessible progress/cancel/configuration/retry. Preserve confirmed
  link/QR in page memory for unchanged data; Discord retry does not reupload.
  Clear dialog credentials and stale results; retain all manual/download paths.
- [x] Real synthetic Wormhole/Backblaze upload, independent decrypted ZIP equality,
  official Wormhole browser download and authorized Discord message/readback.
- [x] Deploy Cloudflare Worker with private destination/access-key secrets to a
  temporary preview account; provide ignored private claim/connection setup page.
- [x] Deployed-Worker and public-Pages browser E2E plus independent review pass.
- [x] Publish verified source/docs; exact Pages deployment and live checks pass.
- [ ] User must complete Cloudflare account claim/login for durable hosting and
  connect their operator browser. Physical phone behavior remains unqualified.

Entry baseline was `5ae2d5f`; concurrent PWA commits through `b16a49a` are
preserved. Current worker preview was deployed at approximately 22:54 UTC;
claim must complete within its 60-minute Cloudflare window. Claim URL and
operator access key are in the ignored local `sharing/.dev.vars.setup.html`;
webhook/access secrets are in ignored `.dev.vars`, not public source. The user
has been given the private setup page and the claim-completion question is pending.
No connected permanent Cloudflare credential was present. Successful preview
execution is not proof of permanent hosting. The user claim-completion question
is still pending. Next: complete account claim/login, then connect the operator's
browser using the private setup page. If the preview expires, follow the two
login/deploy commands in sharing/README.md and use the returned Worker URL.
Keep this goal open until durable ownership and operator setup are confirmed.

Runtime release `0d0e1706c40460f7d512a3dde9acc2cac33a3eb3` is pushed on main.
Exact [Pages run42309917241](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42309917241)
and deployment API both succeed. All32 changed public files match Git bytes,
JavaScript MIME types are correct and both entry URLs return200. Native Chrome
from the actual published page uploads a9,437,363-byte synthetic ZIP through the
deployed Worker, receives Wormhole/Discord acknowledgements, reads back the
message and independently downloads/decrypts the exact ZIP bytes. All five
upload, four forwarding, seven manual-QR and ten files-dialog live cases pass
with no page errors. No participant data is used. All45 Node tests, UI/context,
whitespace and staged-secret checks pass. Independent review's Cancel focus and
the deployed runtime's redirect-mode findings are fixed and verified.

The containing release `5ff34d3f5ceaed7e15ab306d6df32b5d6db84e16` completes
exact Pages run42310953670 with deployment API status `succeed`. All32 affected
public files match Git bytes, both entry URLs return200, and the public sharing
UI repeats all five automatic-upload cases with no page errors. This also
publishes the final setup/evidence notes from `41c3a76`; the later PWA help
changes are preserved. Ignored receipts: sharing-release-0d0e170.json,
sharing-release-5ff34d3.json, cloudflare-browser-live.json and the live browser
result files in output/playwright/.

Ownership blocker revalidated on2026-11-07: Wrangler still reports unauthenticated
and the user's claim-completion question has no answer. The preview answers
allowed-origin preflight204; that establishes current reachability only, not
permanent account ownership. Browser automation ended because it could not
verify the current browser URL, so it cannot supply account-claim evidence.
The same required user sign-in/claim remains after three consecutive goal
turns. All implementation, publication, real-provider verification and setup
instructions are complete; no further autonomous code changes resolve this
account requirement. After the user's direct-Discord decision above, this account requirement applies
only to optional automatic Wormhole upload; it does not block Discord delivery.
If the preview expired, run the documented login/deploy commands with the
existing ignored private secrets, update the operator Worker URL, then verify
the configured hosted flow. Never regenerate previews repeatedly in place of
durable ownership. Overall headset/H10/phone acceptance goals remain separate.

Observed main/origin synchronization before this checkpoint is `eb26983`;
tracked files are clean and unrelated local attachments remain untracked.
This checkpoint changes only sharing state; follow its exact Pages deployment
and public-note check before handoff.

The first real network test runs the production Worker handler locally with
real outbound providers. It sends only a harmless 160-byte synthetic ZIP, restores
it byte-for-byte (SHA-256 `b5410588aad3b0c2fb1026215de366b2d602d8aae8753de8f788ef9798f6df40`),
then posts one clearly labelled synthetic test link to the supplied Discord
webhook and verifies its message readback. A separate Chrome test downloads
the exact same ZIP through the actual Wormhole page. All participant data stays
untouched. Independent review finds one Cancel focus defect, reproduced red
then fixed; no encryption/authentication/data-integrity findings. Test/deployment
evidence and remaining limitations are recorded in VERIFICATION.md. A real
native Chrome test through the deployed Cloudflare Worker now uploads a
9 MiB ZIP in two ciphertext pieces, obtains Wormhole/Discord acknowledgements, reads back
the Discord message and independently decrypts the byte-identical ZIP. The
Worker-specific redirect-mode defect was reproduced and fixed before this pass.
No permanent ownership is inferred from the preview deployment.

The earlier manual-only choice below is historical and superseded for this goal.
FormBee's hosted-service closure (July 2026, official repository) rules it out.
Notification alternatives were researched, but Discord is now the chosen route.

## Polar signal-loss scenario abort — 2026-11-06

Outcome: more than 30 seconds of continuous missing ECG aborts the scenario and
returns to Polar selection (`#h10`), retaining patient details. Shorter loss,
including exactly 30 seconds, preserves the run. Pause does not pause this
elapsed-time deadline. Expiry does not depend on reconnect retries; recovery
arriving after the deadline must not revive the scenario. Clear the selection,
stop all scheduled audio and save the existing aborted receipt/CSV.

- [x] Reproduce delayed-first-poll and late-recovery failures before fixing them.
- [x] Enforce expiry in the shared state owner and route late recovery to Polar;
  keep the existing audio/data cleanup and align module cache versions.
- [x] Pass all 28 Node tests, UI/context and whitespace checks; verify every
  scheduled cue is disconnected on timeout. Eight browser checks pass across
  all five scenario routes, delayed polls, late/short recovery and a real paused
  loss of 30,509 ms. Aborted CSV/receipt, focus return, retained patient details
  and Polar reselection pass. All 137 responsive/Pretext cases pass.
- [x] Independent review found a valid-ECG deep-link start did not arm recovery;
  reproduce that failure, then arm the grace period in validated `start()`.
- [x] Commit/push and verify exact Pages deployment,
  live source bytes and the affected browser flow.

Entry: canonical `main` was clean at `9a91534`. Concurrent documentation commit
`289896c` arrived during this task and is preserved; its Pages run 42251796609
completed successfully. RTK is unavailable at its configured path; bounded native output
is used. Detailed browser evidence remains ignored under `output/playwright/`.
No sensor/headset hardware is connected by this change. Overall real-sensor,
headset, physical-phone and wording acceptance goals remain open.

Release: implementation `6da9539555015087ad3e884b63ebbf85942f4858` is on
GitHub. Concurrent documentation commit `c2356bee24acfd71c607538ea3087c418c565a99`
superseded its first Pages run (42252774069, cancelled). The containing release
completed [Pages run 42254269547](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42254269547)
successfully and its exact Pages deployment reports `succeed`. All 12 affected
public files match Git SHA-256 bytes and both page URLs return 200. The public
origin passes all eight signal-loss browser cases with zero page errors,
including a 30,676 ms real loss while paused, aborted CSV/receipt, focus return,
patient retention, all five routes and a valid-ECG direct start.

Observed `main`/`origin/main` synchronized at `c2356be`, clean before this
evidence-only update. Runtime is verified live; this closing note requires its
own prompt publication/deployment and published-byte check under the standing
rule. Next product action remains real H10/headset integration and pending
wording acceptance; those overall goals are open. This signal-loss increment
is complete. Keep participant exports and detailed QA receipts ignored.

## Collected-data ZIP and Wormhole QR sharing — 2026-11-06

Requested outcome: package all saved browser exports, upload to Wormhole and
share its full download URL as a QR code. Reuse `data-files.js` and its ZIP
writer; do not change participant data, recording, identity or retention.
Wormhole `/api/room` currently answers cross-origin upload preflight with405
and no CORS permission. This static Pages app has no upload backend. The user
explicitly chose the Wormhole handoff on 2026-11-06: automate ZIP creation, upload
on Wormhole, then show its pasted complete link as a locally generated QR.
That accepted scope is implemented; automatic service upload is not required.

- [x] Inspect existing ZIP/storage flow and verify Wormhole's official security,
  FAQ/roadmap and the actual API preflight. No participant files uploaded.
- [x] Add a local-only QR generator with strict complete Wormhole-link validation;
  test key preservation and reject unrelated/unsafe/incomplete URLs.
- [x] Integrate a clearly labelled ZIP → Wormhole → QR flow in the existing
  accessible file dialog, preserving downloads and native sharing.
- [x] Verify rendered layouts, focus, failure/recovery and QR decoding. All24
  Node tests, UI/context, seven QR layouts and ten existing files-dialog cases
  pass; see VERIFICATION.md. Independent review found no actionable issues
  and separately passed the data-file regressions and whitespace check.
- [x] Review, commit/push and check exact Pages deployment/live assets.
- [x] User selected the implemented Wormhole handoff; no backend or alternative
  provider is needed for this completed sharing task.

Entry: canonical checkout on `main`, synchronized at `a216168`; audio sidecar
publication is separate work. RTK is unavailable at the configured path; use
bounded native output. Keep the overall sensor/headset and wording goals open.
Release: implementation `e11a926531e72f6ef6de804777c0463f4c1dd2b9` is on
GitHub. Concurrent audio evidence commits superseded its initial Pages run;
the containing `e5f58b49de54502fe96564ecfce45008a35fe964` release completed
successfully in [Pages run42247056025](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42247056025).
Six live runtime/license files match Git SHA-256 exactly, both URL variants
return200, and the live seven QR/layout cases plus ten existing files-dialog
cases pass. Independent QR decoding preserves the full link/key; zero external
requests occur during QR creation. No participant data was uploaded.

Observed `main`/`origin/main` synchronized at `e5f58b4`, clean before this
evidence-only update. This note's publication keeps runtime bytes unchanged;
verify its exact deployment and published note before handoff. The user has
now selected the implemented handoff, completing this sharing task. No upload
backend/provider decision remains. Participant exports stay local until the
user shares them. Overall hardware, physical-phone and wording goals stay open.

Closing-note publication recovery: `9a91534`, Pages run42248772267, built
successfully, but deployment attempt1 timed out in `updating_pages` after ten
minutes. Attempts2/3 immediately reported cancellation, including after explicit
API cancellation of only that task's failed deployment and confirmation of
`deployment_cancelled`. The artifact is2,146,393,046bytes and carries the
existing1GB warning; this does not diagnose the timeout. The verified `e5f58b4`
QR feature stayed live. Recovery commit `289896c` subsequently deployed
successfully in [Pages run42251796609](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42251796609);
all eight runtime/license/context files matched Git bytes, both entry URLs
returned200 and the remote context gate passed. This choice-only update starts
from clean synchronized `main` at `6da9539` and preserves the separate Polar
increment. Publish and verify the selected-route notes; no runtime change is
needed for this user confirmation.

## Matching numbered audio / CSV / TXT groups — 2026-11-06

The current request requires scenario-owned audio folders and matching numerical
prefixes for each WAV, CSV and TXT. Reuse `assets/audio/scenarios/haupt-15/`
and the four `probe-*` folders; `box-20/` remains an engine-only diagnostic.
The latest remote already implements independent numbered WAV/JSON/TXT cues.
Complete that structure with one same-basename CSV per cue, listing every
occurrence in the authored timeline. Keep first-use numbering and first-onset
timestamps, with the complete playlist/timeline specifying repeated cycles.
No wording, audio bytes, runtime paths or protocol timings need to change.

- [x] Reconcile with the newer scenario-placeholder release; preserve the older
  local draft snapshot in Git stash `codex/pre-scenario-folder-sync-2026-11-06`.
- [x] Baseline: 22 Node tests, six JSON/CSV pairs and 53 packages pass.
- [x] New sidecar regression fails on all 53 missing per-cue CSV files.
- [x] Extend the existing packager, generate matching CSVs, document the order,
  and verify each occurrence, transcript, status and frame time.
- [x] Review, commit/push, verify exact Pages deployment and live sidecars,
  then open the current scenario library for the user.

Entry: main synchronized to `eafec8a`; clean after integrating the newer remote.
Previous Pages run `42241115947` timed out; recovery run `42243359695` is queued.
Current evidence: all 53 matching sidecars cover the 890 authored events; the
Python regression, generator check, all 22 Node tests, six scenario JSON/CSV
pairs, UI state/context and whitespace checks pass. WAVs, authoritative JSON,
playlists and runtime source are unchanged. Independent review found no issues
and separately confirmed that the 53 CSVs partition all 890 events exactly once.
Release evidence: sidecars committed/pushed as
`a216168e92b22f09fef1fb6453fc5da3156d7ba9`. Its exact Pages deployment API
reports `succeed`; Actions run `42243976605` attempt 2 was cancelled after the
backend published successfully when the concurrent `e11a926` release started.
All 67 affected/public context files match local bytes: 53 cue CSVs, six
scenario README files, six protocol JSONs, the scenario index and app HTML.
The active D: checkout has the verified sidecars and its scenario library was
opened in Explorer. Existing recordings and silent-placeholder status are
unchanged. The older draft snapshot remains in the original checkout's Git
stash. Concurrent unfinished publication notes and unrelated QR work are
preserved; they are not part of this sidecar change.

This asset-organization increment is complete. Next product action: user
approves or edits the draft wording before final speech generation. Overall
real sensor/headset/listening acceptance goals remain open. This evidence-only
follow-up must also finish its exact Pages deployment/live checks.

## Scenario folders and working silent-placeholder UI — 2026-11-06

The latest user requests authorize silent WAV placeholders now so the UI works;
final narration and wording remain pending approval. This supersedes the prior
text-only hold on runtime/publication, without authorizing new synthesized speech.
Each of six scenarios owns its independent numbered unique cues in
`assets/audio/scenarios/<scenario>/`: 53 WAVs, matching JSON/TXT, asset CSV,
timeline CSV, manifest and playlist. Numbers encode first-use order and the
filename includes first onset; repeated cycles reuse the unique cue.
TXT is the intended draft script, while `spoken_text` is empty and metadata
explicitly says `silent-placeholder` / `awaiting-wording-approval` (or no-speech).
All original shared recordings remain byte-for-byte unchanged.

The UI loads these scenario assets, caching identical PCM by hash within one
context. The saved scenario JSON remains runtime authority and pins the shared
protocol manifest. Main goes directly to its sole 15-minute sequence; all slots
still total 900 seconds, with preparation26s, measurement15s at26–41s,
Instruction2 40s, fifty16-second cycles and Outro19s. Participant labels read
“Vorbereitung” and “Zwei tiefe Atemzüge”. Timing events remain available; the
future H10 accelerometer measurement remains not-integrated with null value.
Receipts record the scenario hash and `audio_asset_mode: silent-placeholders`.

- [x] Copy the complete sparse checkout, Git metadata, drafts and ignored assets
  to the user-selected secondary drive; verify all 1,489 files (97,320,133 bytes)
  by SHA-256 and recreate the internal dependency junction at its new target.
  Serve and edit the relocated copy. The original was left for manual cleanup
  after automatic review blocked recursive removal; Explorer selects it.
- [x] Prepare 43 individual original-library/sample TXT drafts and a review index;
  Instruction1 proposes “Wir beginnen mit zwei ruhigen, tiefen Atemzügen.”
- [x] Generate independent scenario placeholders and connect runtime/CSV/export
  to those assets; preserve original speech and all protocol timings.
- [x] Observe missing-registry regression fail, and reproduce native decode input
  detachment in a test before fixing it. Pass all22 Node tests, six scenario
  CSV pairs,53 placeholder packages, UI/context checks and12 native offline
  renders at44.1/48kHz with zero sample error. Verify silent preview export.
- [x] Pass five actual browser playback routes,26/41-second transitions,
  pause/resume, replay, CSV export and missing-audio retry;18 enlarged/localized
  text cases pass, screenshot inspected, no page errors. Independent review
  found no issues and audited all53 asset groups/manifests/playlists.
- [x] Commit/push and verify exact Pages deployment and affected live assets/UI:
  feature6f097cf is included in eafec8a, successful Pages run42243359695
  attempt2. All265 changed public files match Git bytes; both entry URLs return
  200. Five live playback routes,26/41-second transitions, pause/resume/export,
  replay/retry and18 enlarged/localized text cases pass without page errors.
- [ ] User approves draft wording before final speech creation/replacement;
  after approval, align exact TXT/JSON/CSV/recordings and reverify timing.

Entry: main/origin at `990172adeaadf3bfcdf1a1a78c79badcef18363b`;
previous exact Pages release `42226744863`. The verified placeholder change is pushed as
`6f097cf91d52036da2781c12578e2d0ecacfbb4f`, with a clean main/origin. Pages
run `42241115947` built successfully, but attempt1 timed out in updating_pages
after10minutes; failed-job attempt2 was cancelled immediately. Its existing
site artifact is2,146,337,782bytes with the standard1GB warning; that warning
alone does not establish the cause. Recovery run42243359695 initially failed
because the timed-out6f097cf deployment still held the Pages lock. Explicitly
cancelled only that stale deployment through the Pages API, verified
deployment_cancelled, then reran the eafec8a failed job. Attempt2 succeeded;
live file and browser checks above pass. The concurrent a216168 increment adds
matching per-cue CSVs and preserves runtime/audio; its publication has its own
top-level checklist. Preserve it and its active deployment. The local server
serves the relocated checkout; machine-specific paths and migration hashes
stay outside published source. This placeholder UI increment is complete.
Final narration awaits wording approval. Overall real sensor/headset,
physical-phone and human listening acceptance remain open.

## Main-round calibration and one 15-minute scenario — 2026-11-06

The latest user instructions supersede the earlier two-main-option/silent-baseline
contract: Haupt-Runde goes directly to one final 15-minute Start screen. There is
no main duration menu or five-minute baseline. All assets together total exactly
900 seconds: Instruction 1 (26 s), Calibration (15 s), Instruction 2 (40 s), fifty
complete 16-second inhale/pause/exhale/pause cycles (800 s), and Outro (19 s,
including seven seconds of quiet settling before the unchanged ending speech).
No additional rest cue interrupts the main cycles. Practice remains unchanged.

The shared manifest names eight main asset classes; scenario JSON drives playback
and its CSV contains exact times, asset paths and transcript_de. Instruction 1
uses the user-requested Secret Tunnel existing-instructor clone. The original
nine WAVs remain unchanged. Calibration is a separate silent measurement window
at source frames 1,146,600–1,808,100 (26–41 s). Receipts and the advance
`atmungsraum:calibration-scheduled` event expose its audio-clock start/end.
The later breathing input will be the ECG sensor's accelerometer. Value, unit
and method remain unset with status not-integrated; no physiological result is
fabricated from synthetic ECG. See audio/calibration.md.

- [x] Verify Respyra source; distinguish its comfortable natural-range protocol
  from our requested two-breath adaptation and original German narration.
- [x] Package reusable Instruction 1 / silent Calibration / Outro assets and
  transcript JSON/CSV, using cached checked Secret Tunnel speech without retiming.
- [x] Replace main duration choice with direct 15-minute entry, remove haupt-10,
  author six scenario JSON/CSV pairs, and add explicit calibration clock metadata.
- [x] Observe the changed main-choice regression fail, then pass all 21 Node tests,
  six scenario pairs, UI state and context checks. Preserve pause/resume/identity.
- [x] Verify current native offline renders, responsive/Pretext layout, actual
  browser calibration transitions/CSV and timing hook, then final independent review.
- [x] Commit/push and verify exact Pages deployment and affected live assets/flow.

Entry: clean canonical main at 64341d9; continuing publication authority applies.
RTK is unavailable, so bounded native commands are used. No prior feature push
occurred before the user's later corrections. The earlier two-option/48-second
composite implementation was locally verified but is superseded, not a release.
Current verification: 21 Node tests; six scenario pairs; UI/context; full PCM
decode/integrity; 12 native Chrome offline renders at 44.1/48 kHz with zero sample
error; 137 responsive/Pretext cases and 18 main/root enlarged/localized text cases;
five actual playback routes, advance timing hook, CSV phases/duration, calibration
entry/exit at 26/41 s, pause/resume/replay and missing-audio retry. Independent
review found no runtime issues and confirmed original nine WAVs unchanged.
Publication verified: feature commit 23b5f2821eb5d0fd13bdc71af7075268fd3b2e30,
Pages run 42225450192 completed successfully; Pages API reports built for that
exact commit. All 54 changed public files match Git bytes and both entry URLs
return 200. Removed main10 JSON/CSV/review URLs return 404. Live Chrome passes all
five playback routes and 18 enlarged/localized text cases, including direct main,
actual 26/41-second calibration transitions, pause/resume, CSV phases/duration,
replay and missing-asset recovery, with no page errors. Live screenshot inspected.
At this evidence write main/origin both match 23b5f28 and source is clean; these
closing notes are the only pending changes. Publish/check this documentation
increment under the standing rule. No runtime blocker remains. Next product
work is real H10 accelerometer/ECG and headset integration, physical-phone
qualification and human listening acceptance; those overall goals remain open.

## Responsive proportions — 2026-11-06

Rebalance the patient screen and permanent controls for smartphone and desktop.
Use a centered patient composition capped at 560px high, a 640px-wide form,
aligned equal columns and shared column gutters. Form groups still distribute
space within a 384px ceiling. Body type tops out at 18px; user enlargement keeps
readable document reflow. The three equal corner surfaces now scale from 56px
to 112px with both viewport dimensions; this supersedes fixed 144px sizing.
Preserve the concurrently completed belt-artwork glow and all interaction/data
owners. No participant writer or audio behavior changes belong to this task.

- [x] Apply scoped CSS geometry and refresh the entry cache tokens.
- [x] Verify 149 responsive/Pretext/text/focus cases, ten folder/download cases,
  two unavailable-measurement fallbacks, all 19 Node tests, seven scenario pairs
  and UI/context checks. Independent review has no actionable findings.
- [x] Commit/push and verify the exact Pages deployment and live assets:
  layout commit `f2d683c74bf1c32d56f36f183d27b31dc1938829` is included in
  `b164c805c1df1b43f565d3ee69aeffda53eb7307`, successful Pages run `42213877838`.
  All seven changed public files match Git bytes and both entry URLs return 200.
  All 149 live layout cases and ten live folder cases pass with no page errors.

Entry: canonical main at `a792da7`, with existing belt-glow changes preserved;
their owner committed them separately as `7e07b9c` during this task. RTK is absent;
bounded native commands are used. A concurrent documentation update superseded
the first Pages run; its successful descendant above contains the exact layout.
At live verification main/origin both match `b164c80`, with a clean working tree
before this documentation-only closing evidence. This UI increment is complete
with no blocker; publish/check the closing evidence under the standing rule.
Next overall acceptance actions remain real H10/headset, physical-phone sharing
and listening checks; those product goals stay open.

## Coherent Polar belt and frame glow — 2026-11-06

Remove the separate status dot. The original belt artwork and its frame
pulse together: green with ECG (four-second cycle), red after signal loss
(two-second cycle), neutral while waiting, and steady with reduced motion.
The PNG is used as a CSS luminance mask so the glow follows the belt artwork
rather than its opaque rectangular background; source bytes remain unchanged.

- [x] Replace the dot and implement synchronized symbol/frame glow using the
  existing state owner, accessible status text and shared color/cycle tokens.
- [x] Verify 24 Chrome state/size/reduced-motion cases, actual pulse variation,
  source-mask loading and folder open/close; inspect green/red screenshots.
  UI-state/context and whitespace checks pass; review has no actionable findings.
- [x] Commit/push and verify the exact Pages deployment and affected live behavior:
  `7e07b9c6b0abd9c8d3e22d54abd339bec130e297`, successful Pages run `42213036555`.
  All six published files match Git bytes; both entry URLs return 200. All 24
  live glow cases pass without page errors, including both cyclic border states.

Entry: clean main/origin at `a792da7`. The glow release is complete with no blocker.
During live verification the independent proportions owner advanced main/origin
to `f2d683c`, preserving the glow and updating corner dimensions under its own
task. Preserve that task's publication state. This closing evidence changes
documentation only; publish/check it under the standing rule. Overall real
H10/headset, physical-phone sharing and listening acceptance remain open.

## Larger corners and recovered Polar artwork — 2026-11-06

The three permanent corner surfaces are 144×144 CSS px: three times the previous
48px recenter/folder dimensions. Their SVG artwork grows threefold too. The Polar
status keeps its accessible status semantics and green/lost/reduced-motion states.
It embeds the original dissertation Polar H10 belt illustration. The recovered
source is PNG, not SVG: Dissertation-scaffold-overleaf, Study 4/figures/PolarH10.png,
Git blob `71fef4c5acd3cb20122be459675963d476d5bc89`; shipped bytes match exactly.
No standalone belt SVG was located in the searched publication/dissertation trees.

- [x] Enlarge all three surfaces and reuse the existing belt illustration.
- [x] Reserve corner space and verify readable portrait/landscape/zoom layouts,
  fixed positions, all menus, signal states and folder/playback behavior.
  All 19 Node tests, seven scenario pairs, UI/context/whitespace, 85 new rendered
  cases and ten folder cases pass. Review has no actionable findings.
- [x] Commit/push and confirm the exact Pages deployment and affected live assets:
  `0a3c99a83301c3650d79dca1f88a52389494b2c8`, successful Pages run `42208617785`.
  All eight published files match Git bytes; both entry URLs return 200. The 85
  live corner cases and ten live folder cases pass with no page errors.

Entry: clean main/origin at `c962c20`. At live verification, local/remote main
match `0a3c99a` with a clean tree before this closing documentation-only update.
The requested increment is complete with no blocker; publish/check the closing
evidence under the standing rule. See VERIFICATION.md for the revised reflow
boundary. Next product acceptance actions remain real H10/headset,
physical-phone sharing and listening checks; those goals stay open.

Closing-evidence publication recovery: `463c842`, Pages run `42209351688`, built
successfully but attempt 1 timed out in `syncing_files` after ten minutes; the
failed-job-only retry was immediately reported cancelled. The artifact warning
is 2,143,894,180 bytes against a stated 1 GB allowance; this is the whole existing
site, not a diagnosis of the timeout. The verified `0a3c99a` UI stays live.
Publish this updated evidence as a fresh commit/run, then verify its exact status
and public bytes. No runtime or unrelated media changes are part of recovery.
Resolved: `a792da7` deployed successfully in run `42211204098`; all eight changed
public files match Git bytes and both entry URLs return 200. Local/remote main
matched with a clean tree at handoff.

## Folder-only data entry — 2026-11-06

Remove both CSV-Dateien footer links. The permanent bottom-right folder is the
sole visible entry to collected data, including retained exports after reload.
Keep the legacy recovery URL and the shared download links used by the dialog.

- [x] Remove the redundant links and their obsolete visibility update.
- [x] Verify saved exports, folder/download/reload behavior, responsive layout
  and existing checks; review the complete diff.
- [x] Commit/push and confirm exact Pages deployment and affected live URLs:
  `0895f63213792e8a5994712b137620f1dd91ef70`, Pages run `42203147921` succeeded.
  All nine published files match Git bytes; both entry URLs return 200.
  Ten live folder cases plus download/reload/playback checks pass, no page errors.

Local evidence: all 19 Node tests, seven scenario pairs, UI/context/syntax and
whitespace checks pass. Chrome passes ten folder cases plus downloads/reload/
playback and 161 stretch cases, with no page errors. The remaining patient link
has a 240px width budget after its old 180px singleton width caused no-fit at
768×1024. Focused review has no findings; see VERIFICATION.md.

Entry: clean canonical main at `ba7efdb`, matching refreshed origin/main.
The concurrent instruction-evidence owner advanced main/origin to `7bbd231`
without including these edits. At live verification, main/origin both point to
`0895f63` and the tree is clean before this documentation-only evidence update.
This task is complete with no blocker; publish/check this closing evidence
under the standing rule. The next product acceptance actions remain real
H10/headset, physical-phone sharing and listening checks; those goals stay open.

## Prompt live publication is standard — 2026-11-06

The user requires future agents to push completed verified changes to the live
site as soon as ready. The root standing rule now explicitly covers prompt
incremental publication, no repeat permission request, exact Pages tracking,
affected live-URL verification and docs-only updates. Root/local entrypoints,
skill indexes and WORKFLOW.md route to that single authority.

- [x] Refresh origin/main: clean local/remote main both `3f62dec` at entry.
  Its Pages run `42080146068` is confirmed successful.
- [x] Update the existing root/subproject instructions and decision owner.
- [x] Run scoped context and Git whitespace checks; review the eight-file
  instruction-only diff. No runtime regression suite is needed for prose changes.
- [x] Push `ba7efdbfe225f1cd2d3d88cc5a5398069dd2269e`; Pages run `42202202945`
  succeeded. All eight public changed files match Git bytes and both site entry
  URLs return 200. Context/remote equality checks pass.

This instruction-only release is verified live. No blocker remains for the rule.
Observed main and origin/main both `ba7efdb` before this closing evidence update;
the separate folder-only task has since added unfinished changes in shared files.
Preserve those changes and stage only this task's evidence. Push this evidence
follow-up and verify its deployment before handoff. The overall H10/headset/
physical-phone/listening goals remain open; their acceptance is the next product
action. Live receipts remain ignored in `output/playwright/live-release-assets-ba7efdb.json`.

## Publish folder/sharing and standing workflow instruction — 2026-11-05

The user explicitly requests pushing everything currently completed and making
Superpowers plus GitHub synchronization a standing instruction for future agents.
Root constraints own that continuing authorization; root/local AGENTS.md, skill
routes and WORKFLOW.md make it discoverable. This release includes the folder,
ZIP/sharing feature, final Pretext refinements and all current project evidence.
Keep participant data, voice caches and ignored QA output local.

- [x] Fetch remote main and review all current source/assets/tests/notes;
  starting local and remote HEAD both `7233468`.
- [x] Record the standing workflow/publication instruction in the existing
  root and subproject control planes.
- [x] Run fresh complete tests, context/scenario checks and rendered acceptance:
  19 Node tests, seven scenario pairs, UI/context/syntax/whitespace, cold-open
  regression, ten popup cases and all 161 stretch cases pass with no page errors.
- [x] Commit/push all 20 reviewed intended files: `a52d56a875cc6c30f149f1c4831e098ac12c8711`;
  local main and remote main match, with a clean tree before this evidence update.
- [x] Verify Pages run `42079217920` succeeded for that exact commit; all 20
  published files match Git bytes and both entry URLs return 200. All ten live
  popup cases plus download/share-boundary/reload/playback checks pass, no errors.
  This closing evidence accompanies a final documentation-only commit/push.

The folder release and standing instruction are complete. No release blocker
remains. The overall real H10/headset, physical-phone share sheet and listening
acceptance goals remain open; those are the next product acceptance actions.
Git observation immediately before this evidence update: main and origin/main
both `a52d56a`, no modified/untracked source. This evidence-only follow-up must
also be pushed, with final SHA/synchronization reported at handoff. The standing
authorization is durable and requires no repeat publication permission.

## Saved-files folder and sharing — 2026-11-05

The user requests a permanent bottom-right folder button with an authored SVG,
a scrollable popup listing downloaded session data, and all-file ZIP/email/app
sharing. The native dialog lists locally retained CSV exports, including records
recovered after reload, without navigating or stopping audio. It cannot enumerate
the operating system's Downloads folder. ZIP download and the device share sheet
use one archive of every saved CSV from this browser. Email/WhatsApp destinations
depend on installed apps and browser file-share support; unsupported browsers
retain ZIP and individual CSV downloads.

- [x] Add the fixed safe-area-aware folder SVG/button and accessible modal list.
- [x] Reuse the shared writer's retained exports; preserve CSV bytes and names,
  with session UUID directories only for colliding filenames.
- [x] Verify ZIP interoperability, sharing/cancel/error fallback, keyboard focus,
  reload recovery, active playback, responsive/enlarged layouts and regressions.

Evidence: 19 Node tests including independent ZIP extraction/CRC/Unicode checks;
10 new Chrome overlay layout cases plus functional sharing/download and 61-record
recovery checks; the existing 161-case stretch matrix; UI/context and seven
scenario-pair checks all pass. See VERIFICATION.md. The platform share API alone
was simulated; no email/WhatsApp message was sent.
Focused review is complete: a hydration-triggered read loop was fixed, and the
cold-open regression now lists 60 files with two full export-store reads.

Publication is now verified by the folder release above: `a52d56a` is pushed and
live, including the final Pretext refinements and preceding release notes.
Native Chrome passed the ten overlay cases and functional file/download/reload/
playback sequence against the public URL, with zero page errors. The next feature
acceptance action is real native sharing on supported phones.
The overall real H10/headset and listening acceptance goals remain open.

## Publish patient/ECG and recenter changes — 2026-11-05

The user explicitly requests pushing all current changes to the live site.
Include patient entry, continuous background mock ECG, session CSV recording,
the persistent recenter placeholder, tests and their For-AI notes. Keep local
participant exports, diagnostic output and voice/source caches ignored.
Clarification: Start records a segment of an independently running ECG stream;
it must never restart the source. Audio pauses leave ECG recording running.
DATA-LOGGING.md owns this contract; real H10 acquisition is still pending.

- [x] Refresh remote main and review all changed/new paths; starting HEAD and
  origin/main both `ac39ec4`.
- [x] Fresh local UI/context/scenario checks, 50 recenter layout/no-op cases,
  25 patient/integration cases and four fallback cases pass. Full integration
  includes actual >30-second signal loss and natural 68-second completion.
- [x] Verify the explicit background-stream regression with the complete suite:
  all 18 Node tests pass; separate recordings start at advancing source indices
  and exclude intervening idle time without changing the shared source epoch.
- [x] Review the three additional app/HTML/CSS stretch-layout edits that arrived
  during deployment. Their 50-case Chrome check passes with no page errors;
  two real browser recordings retain exact source values and advance across idle
  time, including 102 ECG samples recorded while audio time is paused.
- [x] Run the final stretch acceptance matrix: all 156 native Chrome cases pass
  with no page errors, including growth in both dimensions, short landscape,
  all six playback routes, reflow/spacing and missing-Pretext fallback.
- [x] Commit all reviewed changes, push main and verify the exact Pages build:
  `72334681b5e447f7298aacf58b255fc5543b99b6`, run `42060990832` attempt 3 succeeded.
- [x] Verify deployed assets and browser behavior: all 25 public changed files
  match committed bytes, both entry URLs return 200, and 50 final live Chrome
  cases pass with exact continuous-source CSV values and no page errors.

The Pages connector returned an internal error; use the existing authenticated
GitHub CLI/API for exact-commit build evidence. No hosting settings change is
needed: Pages is already configured from main, root `/`, on viscereality.org.
Initial release `e2b6e32` was pushed with local and remote SHA equal and a clean
tree at that instant; Pages run `42059584563` succeeded for that exact commit.
All 25 public changed files matched committed bytes, and 50 live Chrome cases
passed, including successive ECG recording segments and audio-pause capture.
The later three layout files are the only additional uncommitted product changes at the
next observation and are included in this same all-changes publication task.
The overall headset/H10 and listening acceptance goals remain open.

Release boundary: follow-up `7233468` contains the reviewed stretch source and
initial live evidence; it is pushed with remote SHA verified. The next shared
working-tree edits (CSV/ZIP export work and an additional Pretext capability
guard) arrived after this verified release snapshot and belong to their active
tasks. Preserve them; do not sweep them into publication while incomplete.
Pages run `42060990832` built successfully but attempt 1 failed creating the
deployment with GitHub HTTP 500. Retry only failed jobs, retaining the successful
build artifact; verify the deployment job and live bytes before declaring it live.

Final observation: main and origin/main both point to `7233468`; the exact
Pages build/deployment succeeded on attempt 3. This publication task is complete.
The final live recordings start at samples 7395 and 7689, with the first ending
at 7523; 103 paused ECG samples retain one audio time while values change.
Newer CSV/ZIP source/assets/tests/notes and the Pretext guard remain uncommitted
in the shared checkout, along with this closing evidence. Preserve that active
work and validate/publish it under its own task. Do not claim the working tree
is clean or that the newer feature is live. See VERIFICATION.md for receipts
and network/deployment recovery. Overall hardware/listening goals stay open.

## Stretch-accordion settings applied — 2026-11-05

Apply the installed `uncodixfy-pretext` accordion-stretch settings to the entire
main menu, including patient entry and CSV recovery. `SKILLS.md` owns the
geometry/typography rules; `VERIFICATION.md` owns the rendered acceptance matrix.
Keep the existing dark static UI, centered controls, patient/audio state owners,
and fixed recenter/signal controls. Stretch means distributing space between
ordered groups, not adding collapsible sections.

- [x] Make the durable layout settings and supported-size policy explicit.
- [x] Apply dimension-responsive group spacing and complete Pretext/no-fit checks.
- [x] Verify independent width/height resizing, anchors, full text, keyboard flow,
  enlarged text and unavailable measurement; record fresh evidence and Git state.

Evidence: 161 native Chrome rendered cases and two measurement-unavailable
fallbacks passed, including all six Start/Pause/Resume/End routes, independent
dimension growth, both responsive breakpoints, 200% text/spacing, long text/IDs,
locale changes and bounded-layout recovery. All 19 current Node tests, seven
scenario pairs, UI/context/syntax and whitespace checks pass. See VERIFICATION.md.
No blocker remains for this layout task. Physical phones/Safari/WebViews, real
H10/headset integration and listening acceptance remain open product checks.

During this task the concurrent publication owner committed the initial guidance
in `e2b6e32` and the core layout in `7233468` on `main`. Final capability/reflow
polish and these evidence notes are local at this observation, alongside separate
CSV-folder/dialog work in shared runtime files. Preserve that work; the publication
task above owns deployment. Next action: integrate the final reviewed refinements
through that release and verify its exact deployed bytes. No push was performed
by this layout task; no live qualification of these final refinements is claimed.

Starting state: shared canonical checkout on `main`, HEAD `ac39ec4`, with existing
uncommitted patient/ECG/logging and recenter work. Preserve those changes in place.
RTK is absent on PATH and at its documented path; use bounded native commands.
The overall real headset/H10 product goal remains open.

## Persistent VR recenter placeholder — 2026-11-05

Keep the icon-only `VR zentrieren` button fixed at the top left across every
screen, playback state and document scroll, including patient entry and CSV
recovery. Respect the device safe area and reserve heading space below it.
The local vector asset `assets/icons/recenter-vr.svg` shows a VR headset with
four arrows pointing inward from above, below, left and right toward the
central view target. It is editable SVG with no raster dependency.

The original release shipped the button as a keyboard-focusable, disabled
placeholder. The current [Quest 2 milestone](#quest-2-control-media-and-ui-acceptance--2026-11-12)
enables it for an enrolled phone and a ready immersive scene. Quest uses the
midpoint of its current OpenXR eye poses to recenter the app scene and returns
the original receipt after a visible submission.

- [x] Add the persistent inert button and four-direction headset SVG.
- [x] Verify visibility, keyboard focus, no-op activation and responsive layout:
  50 native Chrome cases passed with real Pretext and no page errors; UI state,
  context, SVG XML/vector-only and whitespace checks passed. See VERIFICATION.md.

Git observation: canonical shared checkout on `main` at `ac39ec4`; patient,
mock-ECG and participant-logging changes were already uncommitted at entry.
This placeholder remains local alongside that work; no commit, push or deployment
was performed for the original UI task. The publication task above now owns its
release state. The physical headset and H10 checks are tracked in the
current Quest 2 milestone.

## Patient entry, mock ECG and data logging — 2026-11-05

Add a German patient-details screen **before** the Polar/ECG gate. Collect
exactly three surname letters (confirmed by the user), age in whole years, and one
explicit gender choice (male, female, other, prefer not to say). Reuse the
existing state owner, bounded Pretext layout and native accessible controls.
Show the derived participant ID. The corner ambient indicator breathes green
with signal, flashes red on loss (steady under reduced motion), and loss beyond
30 seconds returns to Polar while retaining patient details. Use a seeded
NeuroKit2-generated mock ECG continuously throughout the demo. Export a CSV for
every started run; keep identity consistent in all participant data logs.
[DATA-LOGGING.md](DATA-LOGGING.md) owns the enforced naming/schema contract:
`Fej_35_M_probe1.csv` / `Fej_35_M_main1.csv`, with condition-specific run indices.
Patient form state is in memory; saved CSVs, counters and linked audio receipts
persist in this browser. No participant values enter navigation URLs.

- [x] Gate all exercise routes/start until valid patient details are submitted.
- [x] Add the form before connection and an edit route; preserve reconnects.
- [x] Add the ambient signal state and the actual NeuroKit source.
- [x] Enforce CSV names/schema, persistent run indices and download recovery.
- [x] Verify validation, keyboard flow, rendered phone/desktop/reflow layouts,
  existing audio checks and context; record evidence and Git state here.

Evidence: 17 Node tests, UI-state/context gates, seven scenario pairs and the
25-case Chrome integration passed. Real >30-second loss and natural 68-second
completion produced the expected CSVs. Targeted fallback/review checks passed
keyboard gender/ID updates, no-JS form protection, missing-source retry,
unavailable Pretext, delayed-counter abort/failure ownership, signed lead-time
pause, offline checkpoint recovery and cross-tab live-session exclusion. See
VERIFICATION.md; local scripts/screenshots/receipts are ignored in output/.

This task is implemented locally. At last observation the canonical checkout
is on `main`, HEAD `ac39ec4`, with uncommitted source/docs and concurrent VR
recenter-placeholder changes in shared files. Preserve that independent work;
no push or live deployment check is part of this request. The recorded origin
ref matched the starting HEAD; no fresh remote verification is claimed. Next
product action remains listening acceptance and real H10/headset integration,
including actual acquisition timestamps and a shared study storage owner.
Those overall goals remain open. Recheck fresh Git state before integration.
The subsequent publication task above now owns release of this implementation.

## Combined publication and skill baseline — 2026-11-05

The user explicitly authorizes pushing all current repository changes,
including the playback controls, simplified transparent round SVGs, scenario
audio protocol and remaining verification notes. Update this subproject's
`for-ai/` with the installed For-AI baseline and preserve its existing product
contracts; the parent repository remains the Git/publication owner.

- [x] Nest the full shared baseline in `for-ai/SKILLS.md` and route to it from
  local `AGENTS.md` and `for-ai/README.md`; keep project-specific UI rules.
- [x] Run current context, scenario, audio and rendered-browser checks.
- [x] Commit all intended changes, push `main`, and verify remote SHA and Pages.
- [x] Verify deployed SVGs, scenario files and browser playback/layout.

Starting Git observation: canonical checkout on `main` at `0f5d05a`, three
commits ahead of refreshed `origin/main`; only the existing alignment-publication
notes remained uncommitted. Publication is authorized by this request. Keep the
overall H10/headset/video-integration and listening acceptance goals open.
Fresh checks passed 13 Node audio tests, seven JSON/CSV pairs, UI/context gates,
13 round-layout cases, SVG transparency, 24 player-layout cases and all six
scenario playback/retry routes. No browser errors occurred. See
`VERIFICATION.md`. Release `5a0e3a9` was pushed to `main`; Pages run
`42048199569` succeeded and the Pages API reported that exact commit built.
All 26 checked public files matched the release, including both SVGs, runtime
modules, shared manifest, fourteen JSON/CSV files and the baseline entry chain.
Live Chrome passed phone/desktop round layout and both branches, all six
scenario Start/Pause/Resume/End routes, exact receipt hashes and missing-file
retry recovery, with no page errors. The live desktop screenshot was reviewed.

The push request and nested-baseline task are complete. This final evidence is
saved in the following documentation-only commit; product files remain those
verified at `5a0e3a9`. At live verification, local/remote `main` matched and the
tree was clean. Next product actions: listening acceptance and real headset/H10
transport/video synchronization, with target-device evidence. Those overall
product goals remain open; recheck fresh Git state at the next session.

## Playback controls task — 2026-11-05

The user requests clearer buttons and a video-player-style progress display.
Keep the centered primary Starten/Pause/Fortsetzen control, style Beenden as
a bordered button, and group the phase, progress bar and elapsed/total clocks
below playback. The timeline is visible before starting and remains visible
after stopping or finishing. It follows active Web Audio time and freezes on
pause; replay resets it. The Anleitung example shows its real 68-second audio
duration even though the menu choice has no imposed exploration time limit.
Use a semantic read-only progress element; seeking would alter the authored
breathing protocol and is outside this change.

- [x] Implement audio-clock progress, elapsed/total time and clearer buttons.
- [x] Extend the controller regression to cover reset, pause, exact totals,
  completion and replay; all eight audio tests pass.
- [x] Verify 24 Chrome cases: seven viewports in ready/playing/paused states,
  plus 200% text, long German labels/text spacing, and missing Pretext.
  See `VERIFICATION.md` for evidence and device limits.

The overall headset/H10 product goal and listening review remain open.
Existing round artwork and viewport-centering work are preserved. This task
is committed on `main` under `Add playback buttons and audio-clock progress`;
its predecessor was `a79275f`, matching the locally recorded `origin/main`.
The original playback task left publication to the combined release above,
which now owns its current Git/deployment state. Screenshots and the 24-case
executable check remain ignored in `output/playwright/`. Real phone/Safari,
headset/H10 and listening acceptance remain separate open checks.

## Scenario audio protocol — 2026-11-05

Every final sequence must play predefined assets from the shared `assets/audio/`
library. Each scenario owns a JSON timeline and a matching CSV under
`audio/scenarios/`; JSON is the runtime authority and CSV is its checked,
readable timing sheet. Preserve the chosen runtime samples and existing exact
durations, silent main-round baseline, pause/resume and abort behavior. The
current browser player is the integration target; real video/headset execution
remains pending. No new voice generation or asset selection is authorized here.

- [x] Add all six menu scenarios plus the existing engine-only 20-minute
  diagnostic as JSON/CSV pairs; bind each to the shared library manifest.
- [x] Load and validate those files before playback; retain the single Web Audio
  clock and record scenario identity/hash in scheduling receipts.
- [x] Verify file/CSV consistency, invalid-input rejection, all existing audio
  behavior and native offline renders; update the audio workflow and evidence.

Implementation uses the existing plain JavaScript engine and Node checks with
no new dependencies. Work stays in the shared checkout to preserve the pending
UI/player edits already present in `app.js`, `index.html`, `style.css`,
`audio/session-ui.js` and `tests/box-breathing.test.mjs`. Initial Git observation:
`main` at `a79275f`, matching the locally recorded `origin/main`; no publication
of this change has been observed. Overall headset/H10 acceptance remains open.

Evidence: 13 Node audio tests, seven JSON/CSV consistency checks, fourteen native
offline renders (all seven scenarios at 44.1/48 kHz with zero sample error), all
six real-browser playback routes and missing-JSON/retry behavior passed. All
886 participant text-score ranges/scripts/clips match the new JSONs. Hash-bound
JSON/CSV uses pinned LF bytes so Windows and GitHub Pages agree. See
`VERIFICATION.md`. Concurrent tasks committed the existing player/artwork edits
as `2e58f6f` and `7c8cadc` during this work. This change is prepared as the local
commit `Load final-sequence audio from scenario JSON timelines` on `main`;
no push/merge/deployment is part of this task. Unrelated alignment-publication
notes remain separate working-tree changes. Next product action: listen/review
the chosen samples and implement the actual headset/video transport with a
shared timing anchor; acceptance requires target-device audiovisual evidence.

## Round-button infographic task — 2026-11-05

- [x] Generate two matching transparent infographics and convert them into
  actual SVG paths. Originals, SVGs and exact prompts/settings are retained
  under `assets/rounds/`.
- [x] Embed the SVGs above the existing German round labels, preserving the
  routes, accessible link names and real Pretext measurement.
- [x] Verify 13 rendered cases, both branch routes, keyboard activation and
  readable unavailable-measurement fallback. See `VERIFICATION.md`.

The final simplified practice graphic shows lungs with a repeat arrow. The main
graphic shows a clipboard with a simple three-point data chart, identifying the
research/data-collection role requested by the user. Both SVG backgrounds and
the clipboard interior are transparent. This replaces the first breathing-loop
artwork and supersedes the earlier text-only round-choice rule. The symbols do
not establish implemented research data collection; the overall headset/H10
product goal and pending audio listening review remain open.

Git observation: the shared checkout is on `main`. Concurrent UI alignment work
committed the two image references and their CSS in `e4f76d9` while this task
was tracing/verifying the assets; player/audio edits remain owned by the other
task. The initial assets were published in `a79275f` (Pages run `42044006553`
succeeded). The simplified revision is saved locally under
`Simplify round SVGs for practice and research`; its new PNGs, real path-only
SVGs, prompts and transparent-alpha checks are retained with the existing
13-case rendered QA. It was initially kept local alongside the separate player
and scenario changes. The combined release above now publishes and verifies
this simplified artwork, the player and scenario audio together. Preserve the
remaining product boundaries; the overall headset/H10 goal remains open.

## Current UI alignment task — 2026-11-05

The user requires symmetric, centered controls on the main Atmungsraum UI.
Keep headings and single footer actions on the horizontal midpoint. Center
the sensor-selection button, ECG, choice grids and primary playback button
against the viewport itself, with equal top/bottom space budgets. Starting,
pausing or showing the countdown must not displace the playback button.
Use the dynamic viewport height with the existing small-viewport fallback.
Enlarged text retains the deliberate document-reflow escape hatch.

- [x] Correct the shared shell, sensor action, footer and playback geometry.
- [x] Verify 79 rendered cases at seven ordinary viewport sizes, plus 18
  enlarged-text/spacing/fallback cases and the keyboard interaction sequence.
- [x] Publish and verify the exact UI files and centering on the live site.

Local Chromium checks, UI state, all eight audio tests, context and whitespace
checks passed; screenshots and geometry receipts are ignored in
`output/playwright/`. See `VERIFICATION.md`. Branch is `main`; the refreshed
remote sample-page changes were merged while retaining the earlier local audio
review notes. Publishing the navigation UI is covered by the existing approved
prototype deployment workflow. The overall headset/H10 product goal and
listening acceptance remain open.

Alignment feature `e4f76d9` is published as an ancestor of `a79275f`; Pages run
`42044006553` completed successfully. All three live entry files matched the
feature's committed bytes, and all 79 live Chromium layout cases passed.
Concurrent scenario/progress/image UI work is present in the shared checkout;
preserve it. The alignment request is complete. Any subsequent UI revision must
retain this geometry and rerun the centered-layout checks. The live verification
notes are updated locally alongside that concurrent work.

## Voice sample subpage — 2026-10-28

The user requested publishing the exact phrase “Einatmen … zwei … drei … vier.”
from all twenty newly saved Atmungsraum profiles at `/atmungsraum/samples/`.
This separate listening page uses ten LANG reference variants (two actors) and
ten EmoDB actors. LANG ordering uses constituent-source valence means; EmoDB
neutral-valence estimates are tied and source validation breaks the tie.
Neither source ratings nor exact-word ASR establish the synthesized delivery's
calmness. Samples preserve their original generated bytes and timing.

Scope: the new `samples/` directory, its publication metadata, and these owning
notes. Reuse the existing locked Pretext runtime and static GitHub Pages hosting.
The existing session scheduler, chosen ElevenLabs assets, and H10 UI keep their
current behavior. Original references and full machine-specific receipts stay
private. The user's request explicitly authorizes publishing these new auditions.

- [x] Generate and fully decode twenty German MP3s; verify their exact words.
- [x] Check the static players, downloads, text measurement and responsive reflow.
- [x] Publish to `main` and verify the public sample page and all MP3 URLs.

Publication verified on 2026-10-28: feature commit `1f3f5db` completed GitHub
Pages build/deployment run `40730446610`. The public page and all twenty MP3s
returned HTTP 200; every downloaded MP3 matched its recorded SHA-256 hash.
Live Chromium verified the twenty players, text measurements, keyboard
play/pause and MP3 download. URL: <https://viscereality.org/atmungsraum/samples/>.

- Purpose: provide a smartphone-compatible browser interface that controls actions
  on a VR headset screen through a corresponding Atmungsraum app.
- Intended user: someone operating the VR experience from a smartphone browser.
- Repository: <https://github.com/GeorgeFejer91/viscereality.org> (public).
- Default and publishing branch: `main`.
- Public URL: <https://viscereality.org/atmungsraum/>.
- Entrypoint: `atmungsraum/index.html`, relative to the repository root.
- Current website stack: static HTML, `style.css`, and plain JavaScript; local
  Pretext runtime pinned to `@chenglou/pretext` 0.0.9.
- Hosting: the existing GitHub Pages branch deployment from `main`, root `/`,
  with the existing `viscereality.org` custom domain.

## Current task: audio protocol review — 2026-10-27

- [x] Inventory the checked-in generation, audition, packaging and sequencing
  scripts, separating runtime and local listening revisions. Evidence:
  `audio-review/01-PROTOCOL-INVENTORY.txt`.
- [x] Provide participant-readable German speech and unspoken sound/silence
  notes in a separate review folder. Evidence: `audio-review/02-PARTICIPANT-EXPERIENCE.txt`
  and seven full text timelines, derived from production `createProtocol()`.
- [x] Verify source hashes, all nine runtime WAV hashes/byte counts, timeline
  continuity/totals/overlay counts, eight audio tests, UI state, context and
  whitespace. See `VERIFICATION.md`.
- [ ] Review wording and delivery by listening; decide whether to revise the
  welcome's counting promise, silent main-round opening or hold wording.

The current request is fulfilled as a local text review; the overall product
goal remains open. No speech, scheduler or UI behavior changed. The documented
`reference-audio/` cache and generated ElevenLabs listening package are absent
from this checkout; restoring them is required for audition comparison, not
for reading the current runtime timelines. No fresh listening or hardware
verification is claimed. Next action: collect timestamped review notes using
the template in `audio-review/00-START-HERE.txt` before changing audio assets.

Git observation at task start: canonical repository `GeorgeFejer91/viscereality.org`,
primary checkout on `main`, HEAD `b365411`, clean and aligned with the locally
recorded `origin/main`. Task changes are limited to `audio-review/` and this
subproject's guidance. No remote ref refresh or live deployment check was made;
this review is not a publication request. Recheck Git state on next entry.
Local review work is recorded under the commit subject
`docs(atmungsraum): inventory audio protocols and participant review`.
No push or deployment is part of this task; generated audio remains untouched.

## Intended product contract — 2026-10-12

The user approved this prototype for publication on 2026-10-12 as the main
navigation logic for the UI controlling the VR setup. It remains a prototype.
The bottom-level Start action is intended to play an audio sequence and initiate
a corresponding sequence of VR events. Browser audio already works; the VR event
set, command transport, timing, and synchronization still need to be defined.

The primary target is a smartphone progressive web app, installed from the
browser to the home screen and launched at the live `/atmungsraum/` URL with
the Viscereality logo. It is the control surface for the VR headset app.
The current integration contract places all Atmungsraum controls on the phone.
Quest presents only the Viscereality scenario to the wearer. See the active
Quest rehearsal milestone at the top of this file.

Actions initiated in the phone browser should trigger the corresponding actions
on the VR headset screen. The intended interaction runs from phone browser to
headset app; controls must ultimately affect the headset experience, not only
the browser's local UI.

Live screen or video streaming is outside the rehearsal milestone.

## Patient entry and Polar H10 verification

The first page state collects patient details and shows the generated participant
ID. Weiter then opens the Polar connection gate. The gate precedes the round/session
menu; session actions must not appear ready before details and ECG are verified.

When the H10 is connected, display its live ECG waveform in a box in the middle
of the interface. Use actual ECG samples from that sensor. A selected device,
heart-rate reading, or simulated waveform does not establish a working ECG feed.
Represent connecting, disconnected, and connected-but-no-ECG states accurately;
do not leave a stale trace presented as live after the connection or data stops.

The intended browser checks the headset and receives ECG from it. The H10
acquisition path within the headset app and the transport remain Undecided.
Choose the transport against the actual supported smartphones and browsers. The
verified ECG view enables a Weiter button to the round menu.

## Menu layout contract — 2026-10-12

Use the installed `uncodixfy-pretext` skill and its outside-in stretch-accordion
layout for every UI function in this menu, including the H10 gate and ECG view.
Allocate the outer panel first, retain header and action/footer anchors, preserve
ordered groups and label/control pairs, and distribute extra width and height
between groups and cells. Use shrinkable Grid/Flex tracks with `min-width: 0` and
`min-height: 0`; derive spacing and preferred type from both dimensions.

The bounded menu panel has no internal scrollbar. Reflow controls or revise the
supported minimum when they do not fit. Preserve readable type, touch targets,
keyboard order, visible focus, and user-enlarged text. This stretch geometry does
not require collapsible sections. Use actual locked Pretext measurements for all
bounded labels and status text, with font parity and a readable unavailable/no-fit
fallback. Verify the rendered layout; CSS controls geometry and Pretext measures
text. See [SKILLS.md](SKILLS.md) and [VERIFICATION.md](VERIFICATION.md).

## Round menu and UI prototype — 2026-10-12

Use easy, simple German throughout the interface. The browser prototype implements
this browser-local decision tree:

1. Emulate the headset searching for Polar, finding it, selecting/connecting it,
   then waiting for ECG. One compact status line shows the current step, with
   Polar auswählen available when found. The ECG box appears only during
   recording. Weiter is red/disabled until recording, then green/enabled.
   Erneut suchen appears only when disconnected or waiting for ECG, where retry
   aids recovery. The status and canvas identify the demo.
   Patient entry precedes this step. Every started run follows DATA-LOGGING.md;
   CSV-Dateien provides local download recovery. The corner ambient signal is
   visible across screens without replacing the connection gate.
2. Show "Übung wählen" with Probe-Runde and Haupt-Runde together. Use the two
   generated SVG infographics above HTML labels, soft dark surfaces, muted sage
   and cream, gentle corners, and
   restrained borders. Choices are square where the measured labels fit.
3. Show the clickable boxes for the chosen round:
   - Probe-Runde: Anleitung (guided free exploration, no time limit), 5, 10,
     or 15 Minuten. Anleitung and Frei ausprobieren are the same option.
   - Haupt-Runde: direct entry to one 15-minute sequence. Instruction 1,
     Calibration, Instruction 2, fifty complete breathing cycles and Outro
     together last exactly 900 seconds. There is no main duration menu/baseline.
4. Every box goes directly to a final screen with a **Starten** play
   button. The selected round/option is shown for orientation. Starting is
   browser-local; the separately integrated audio controller owns playback.

There are no intermediate radio confirmations or separate instruction pages.
The initial final screen shows the play action. During playback that same button
becomes Pause, or Fortsetzen while paused, and the back link becomes Beenden.
The phase and elapsed/total progress remain visible in every player state.
Remove redundant descriptors, decoration,
duplicate buttons, and routine ready/playback messages; keep the choice timing
breakdown and actionable loading/completion/error feedback. Real headset session
execution remains pending.
Hash navigation supports Zurück and browser history; cold deep links still pass
through the connection gate. After Weiter there is no ECG return button. Signal
loss after this verified transition keeps the current screen and existing audio
while requesting reconnection every five seconds. More than 30 seconds without
recovery aborts the scenario, stops its audio, clears its selection and returns
to Polar selection; clear the old waveform and lock Weiter. The elapsed-time
deadline also applies while paused and never depends on a prior retry. ECG
recovery after expiry cannot revive the old run. Recovery through exactly
30 seconds cancels the deadline and retry count.
New sessions cannot start without current ECG. JavaScript is required for the demo; without it the
patient form stays disabled with an explanatory message and cannot submit
participant values into a URL.

Runtime assets remain local; Pretext 0.0.9 measures bounded text with the painted
Arial font. The header and footer anchor a bounded panel, and the middle controls
use shrinkable container-sized tracks. At enlarged text/no-fit the page explicitly
reflows into document flow, with no inner scrollbar and no font shrinking.

The actual product should receive verified connection state and live H10 ECG
from the headset. This UI demo implements neither real sensor acquisition nor
browser-to-headset transport. Do not present its synthetic waveform or local
playback as hardware verification or a command delivered to a headset.

The future headset transport can report `atmungsraum:sensor-state` CustomEvents
with `{ state: 'ecg' | 'no-ecg' | 'disconnected' }`. The browser emits
`atmungsraum:reconnect-request` with the retry number. These hooks implement the
UI's recovery behavior; an actual transport and verified ECG sample source are
still required. Do not mistake events or synthetic traces for hardware validation.

## German audio cues — 2026-10-12

Five separate German speech cues are saved in `assets/audio/breathwork/de/`:
Ankommen, Einatmen, Ausatmen, natürlich atmen, and Abschluss. Their exact text,
filenames, generation settings, durations, and SHA-256 hashes live in `cues.json`
beside the MP3 files. ElevenLabs returned the saved voice named
`British Breathing Instructor` (ID `rt9KyyQmwqa1uhHGJkMa`); use this voice for
these Atmungsraum cues. The generated set uses `eleven_multilingual_v2`, a speed
of 0.85, and 44.1 kHz / 128 kbps MP3 output.

These are local audio assets. Playback, breathing-cycle timing, and headset
integration are not implemented by their generation. All files passed full MP3
decode and integrity checks; German pronunciation and subjective voice review
remain pending.

## Reusable box-breathing engine — 2026-10-12

`audio/box-breathing.js` loads twelve reusable German PCM WAV samples from
`assets/audio/box-breathing/de/soft-v2/manifest.json`. The phase samples are exactly
four seconds at 44.1 kHz. Repeat the four separate files `inhale.wav`, `hold_in.wav`,
`exhale.wav`, and `hold_out.wav` in that order. The instruction and ending samples
play once (40 and 12 seconds). The two hold files are independent roles with
identical Und halten text followed by quiet. Inhale/exhale say Einatmen/Ausatmen
with counts two/three/four. Welcome, ending, and rest use du. Delivery is softer
and slower; the exact phase wording follows the user's latest instruction.
Do not generate or concatenate new speech for each cycle.

The user selected exact 5/10/15/20-minute total durations including welcome,
ending, and rests. Complete 16-second cycles form blocks of at most 18 cycles
(288 seconds, the nearest complete-cycle duration below five minutes). A 20-second
natural-breathing rest separates blocks when another full cycle fits. Session
cycle counts are 15/33/50/68, with 0/1/2/3 rests and 8/0/8/0 seconds quiet recovery.
Every third complete cycle overlays both inhale and exhale with the supplied
breath sounds. Use the Web Audio clock and preload
all samples; JavaScript timers and ended callbacks must not trigger phase audio.
Haupt-Runde is the single 900-second class-based scenario described above.
Instruction 1 prepares two comfortable deep breaths; the separate silent
Calibration window runs at t0 + 26 to t0 + 41 audio-clock seconds. Instruction 2
then explains paced breathing, which begins at t0 + 81. Fifty complete cycles
end at t0 + 881; Outro lasts 19 seconds and ends at t0 + 900. Main has no baseline
or between-block rest. Pause freezes this clock and resumes the same nodes;
receipts preserve planned calibration timing with its measurement still unset.
Hidden pages, interrupted audio clocks, failed preparation, or missed start
deadlines invalidate/abort an attempt. Host-side saving of receipts and physical
audio/sensor synchronization remain integration work. The browser controller
saves the last ten participant-linked scheduling receipts locally; the CSV
writer retains each exported run in IndexedDB. These are not acoustic onset
measurements or an integrated study data store. See DATA-LOGGING.md for identity,
clock, mock-source and persistence limits.

The original German/English reference audio and native-caption transcripts are
downloaded under ignored `reference-audio/`. Newly synthesized German samples
use British Breathing Instructor. See [audio/README.md](../audio/README.md) for
the source analysis, scripts, timing contract, API, verification, and limits.
The round menu's final Play button is wired to browser playback through
`audio/session-ui.js`; Stop/Escape/navigation and interrupted clocks abort the run.
The main button changes between Starten, Pause, and Fortsetzen. Pause deliberately
suspends the Web Audio context and freezes the countdown; Fortsetzen resumes the
existing schedule from the same position. Beenden returns to the choice menu and
aborts the run; Escape stops in place. No separate Stop or Start/Pause row is shown.
Timed totals count active audio-clock time, so deliberate pauses extend wall-clock
duration. Pause/resume records are included in the local scheduling receipt.
No H10 or headset behavior is established by the audio checks.

Alternative voice sourcing is local audition work: two neutral, two positive,
and two restrained EmoDB profiles are saved in Secret Tunnel. The user rejected
the neutral takes as robotic and the positive takes as too energetic. Restrained
16-second counted cycles await listening. Archive CC0 and publisher CC-BY-4.0
metadata conflict; preserve attribution. LANG norms contain no low-arousal/high-positive-valence
match at the current thresholds. OmniVoice's pretrained model is CC-BY-NC and
speaker clone consent is not established by corpus metadata. Do not replace or
publish runtime samples as unrestricted clinic voices on this evidence. See
[voice-sourcing.md](../audio/voice-sourcing.md).

## Current implementation

The initial 2026-10-09 page was a blank black placeholder. The working tree now
contains the functional German UI prototype described above, including synthetic
ECG gating, four practice/direct main choice, and final play-button routes. Real H10 data,
headset connection, and headset session execution remain unimplemented. Browser
audio is wired separately through audio/session-ui.js. Keep the black visual
foundation and preserve work in progress.

The AI guidance lives at `atmungsraum/for-ai/`; it is not page content. This
subproject shares the parent Git repository and deployment, with no nested repo.

## Undecided

- Encrypted study transport, physical ECG acquisition, and additional phone browsers.
- Hardware verification of the headset-delivered ECG gate.
- Headset implementation of the selected session start and unlimited Anleitung.
- Physical qualification of the Android Chrome / Quest trusted-LAN rehearsal.
- The VR event sequence initiated by Start, including audio/event timing and
  synchronization between the browser and headset.
