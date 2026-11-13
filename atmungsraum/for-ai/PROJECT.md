# Project

## Active milestone: smartphone PWA and Quest 2 rehearsal

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
Maestro flow ran. The physical PWA flow and headset/H10 acceptance remain open.

Next for phone acceptance, reopen the installed Atmungsraum PWA in standalone
mode on the Pixel and verify its foreground state. Continue with the existing
Quest 2 checklist and use an explicit device ID for each phone command. Retry
Maestro when the offline emulator is absent or a release fixes device discovery.

GitHub main received `041f360` from an isolated sparse checkout because the
primary checkout's cloud-backed Git metadata was unavailable. Exact
[Pages run 43398518216](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43398518216)
succeeded. The rendered skill route, three linked skill pages, two license
files, project notes and Atmungsraum URL passed live checks.

### Quest 2 control, media and UI acceptance — 2026-11-12

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
  existing Wi-Fi, manual LAN join, enrollment, Polar setup and live ECG, condition
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
The Pixel 10 reported Chrome 134. Update Chrome before installing the PWA
and grant Atmungsraum Local Network Access for the Quest connection.
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

### Automatic Google Drive delivery of Quest recordings — 2026-11-12

After the installed Android controller saves a finalized Quest recording, it
must deliver the original Polar H10 ECG and accelerometer (ACC) streams to a
private folder in the designated research Google Drive. The account reset
depends on a verified backup outside that account. Keep account backup receipts,
sign-in details and participant data outside this repository.

- [x] Verify ADB access and check for the official Google Drive app. Android
  Package Manager reports it installed on the Pixel 10.
- [ ] Complete and verify the account backup before clearing account data.
  The operator signed in and Google accepted a one-time export of every
  selectable product. The local Drive inventory has a verified copy of each
  item, including portable snapshots of two shared Forms with response archives.
  Google is preparing the full archive. Download and verify every part before
  clearing account data.
- [ ] Install the Atmungsraum PWA in Android Chrome and verify standalone
  launcher relaunch on the Pixel.
- [x] Preserve the exact native ECG and ACC bytes alongside the existing
  participant CSV. Signed synthetic transfers and durable browser restart
  readback matched both stream hashes in three browser layouts. Physical
  sensor validation is still required.
- [x] Implement foreground delivery from the committed recording store. Keep
  pending uploads and Drive IDs durable and verify folder, owner, size and
  SHA-256 before saving a delivery receipt.
- [ ] Configure a real Google Cloud web OAuth client for the Pages origin and
  authorize the designated account on the Pixel. Verify real Drive readback.
- [ ] Run fault-injection tests, then complete the attended Quest + worn H10 +
  Pixel + real Drive stress pass in [QUEST-PHONE-SYNC.md](QUEST-PHONE-SYNC.md).
  Synthetic fault injection passed in Node and Chromium. The connected Pixel
  was locked and no Quest appeared in the USB device list.

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
storage. The next actions remain the verified account export, Android PWA
install, real OAuth setup, and attended Quest/H10/Drive run.

The account export checkpoint `0a72dde` matched local `main` and GitHub's
`main` on 2026-11-12. [Pages run 43175488204](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43175488204)
deployed that commit, and the published project notes matched its Git blob.

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
truth. No cloud service participates in this flow.

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
- [ ] Obtain installation/relaunch acceptance on the user's target phone:
  record phone/browser, Viscereality icon, canonical app and standalone window.
  Keep any other untested phone platforms explicitly unqualified.

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

The user explicitly sets the finish-PWA goal active in this chat. Automated and
publication checks are complete; the final target-phone installation/relaunch
question is pending. Do not mark the PWA goal complete without that result.
The same acceptance blocker remains on recheck: ADB lists no connected device,
and no connected phone-testing tool is available. The existing phone-test
question has no result. Further desktop checks cannot supply physical-phone
acceptance; resume when the user supplies that result or a phone is available.

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
