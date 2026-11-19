# Autoboot, kiosk and Meta update qualification — 2026-11-16

## Controlled32-to33 test staged — 2026-11-19

Developer33 (`0.2.32-dev`, Meta build1470412299491115) is current/Complete
for Quest2. The production source is identical to32; helper3 is unchanged.
It was published while installed32/PID20269 stayed alive. Neither the headset
Store nor the personally used phone was accessed; no restart/update/launch
command was issued. The public latest-version hint is33, minimum27 unchanged.

On wearer return, observe the PWA's first refreshed/reconnected response before
any Store visit or restart. Test a PWA-triggered check while32 stays running.
If Meta still offers32 despite published33, record that response, then isolate
a real cold process restart without opening the Store. Observe the fresh
startup SDK check and actual package replacement/relaunch separately. A menu
visit or timed wait cannot be substituted for a confirmed new installed code.
If Meta still offers32, leave that mismatch explicit and offer a retry/manual
route; do not enter a restart loop or claim the target was delivered.

The optional PWA [manual guide](MANUAL-UPDATE-HELP.md) is presentation only.
Its authentic screenshot route awaits attended capture. This controlled probe
adds no automatic restart mode, helper change or new update-detection logic.

## Actual startup prompt ownership — 2026-11-19

During this-PC bench38 first launch, pending runtime permission dialogs exceeded
the generic60-second suppression and kiosk counted three own-surface departures,
then admitted its intentional exit. The app did not crash. Warm Activity return
also exposed a retired Polar owner; a real process cold start recovered the
saved membership and native control. Keep that separate warm-lifecycle issue open.

The [startup permission overlay](../quest/README.md#actual-startup-permission-kiosk-protection)
adds an exact current-Activity observation for actual runtime/Location prompts.
The own-app watcher checks it before counting departures. Time cannot end that
protection: actual platform completion, owner retirement or loss of the exact
current owner ends it. Generic prompt timing/default-off behavior and ordinary
escape after completion are preserved. Host regressions replay120seconds and
the after-completion triple escape. Physical acceptance requires leaving a real
prompt open past60seconds, verify no exit, then allow it and verify native
focus/control. No helper binary or Meta update policy changes are required.

Attended bench39 result: the real Android GrantPermissionsActivity remained
top-resumed after99seconds while the same app process and kiosk service stayed
alive. After the wearer allowed the requested access, the stimulus returned
in that same process, native frames advanced and kiosk reported ready-armed.
Bluetooth rediscovery then reused saved membership and the Pixel confirmed
active control with the new build fingerprint. An initial empty scan may have
been a range issue, as reported by the wearer; PC scanning independently saw
the correct discovery service. Specific LAN/TURN transport and a new physical
triple-escape check are not established by this result. The warm-relaunch Polar
issue remains separate. Developer32 was subsequently published to the private
Developer channel as Meta build1470180652847613 with Quest2 target and
processing Complete. The first actual PWA-triggered check still returned
available27 from Meta's SDK.

The later32 delivery passed after a diagnostic restart: two PWA checks while27
was running offered27, despite the headset's Developer channel and Store showing32.
The wearer did not press Update. Following kiosk escape, a host cold launch
restored the baseline, whose startup SDK check detected and installed32.
Package replacement, exact32 bytes and Meta installer/initiator are confirmed.
Helper3 requested fallback and the stimulus returned automatically; SDK relaunch
was also active, so the helper is not proven necessary. The Pixel automatically
reconnected with its saved trust and active control; kiosk is effective.
PWA-only fresh detection remains unqualified. A successor test must leave32
running, avoid Store inspection, then isolate a restart-only refresh if needed.
The installed baseline is now32; held28 remains unpublished.
The normal Meta27 app and Pixel pairing were restored afterward, with fresh
native `IDLE/rec:false/pending:false`, active phone control and kiosk ready-armed.

## Actual boot/update acceptance — 2026-11-17

Meta21 baseline and Meta22 successor have exact installed-byte, signer and Meta
installer evidence. A real boot12→13 automatically launched21, performed its SDK
check/download/install and returned as22 with advancing FOCUSED frames. Helper3
fallback dispatch was captured before the new process; Quest shell handles its
final activity launch. The user triggered no manual update. No host main APK
install or post-boot app launch established this result. The normal SDK lane is
qualified; the queued duplicate lane and physical disconnected-phone notice
remain separate from this run's evidence. See the APK log and VERIFICATION.

For setup on another headset use
[the collaborator runbook](COLLABORATOR-QUEST-SETUP.md) and its supplied signed
helper APK. A diagnostic component-only Android launch intentionally lacks a
trusted updater owner: use the normal MAIN action and sole LAUNCHER category
for a cold baseline launch. Bare `am start -n` may render while reporting no
startup checker. The actual boot helper uses the qualified intent route.

## Queued Meta update amendment — 2026-11-17

The user accepts closing the main app to let Meta apply its queued update.
Observed20 replaces19 only after the idle app is closed; actual bytes/Meta
attribution/UID are verified, without a new launch. The prior SDK StartDownload
attempt fails and OCMS rejects a duplicate StartDelayed request. Existing-job
target and completion are not exposed by the SDK progress API.

The earlier selected-helper route could hand off an unidentified queued payload
while persisting the currently offered version as its target. The attended
25-to-27 attempt exposed why that is unsafe: Meta first installed cached26, and
the user then pressed Update for27. SDK207 progress has no target/job identity;
the same full counters can describe either payload.

The 2026-11-19 [reconciled source](../quest/README.md#reconciled-updater-source--2026-11-19)
therefore uses bounded cancel/requery/new-download recovery even when helper3 is
selected. Require a matched positive cancel ACK, fresh installed/offer/floor
validation, one owned DownloadSuccess and a subsequent fresh progress response
before the existing durable target/native/helper/kiosk/SDK install sequence.
Duplicate progress never arms the helper or closes the app. External installing
states remain uncertainty, never cancel permission. Keep the existing queued
handoff protocol/status compatibility for older builds; it is not selected by
ambiguous duplicate progress in the repair. Active recording and writer guards
are unchanged. The monitor amendment below still governs helper3's observation
and relaunch after an authorized install. This repair is source-qualified;
installed/published27 still contains the earlier behavior until a qualified
successor is delivered.

Optional private v2 status `sdk_error` is null or the closed shape
`{v:1,stage,code,http,install_result}`. Stage is1–6, code is a nonzero signed32-bit
integer, HTTP is null or100–599, and result is one of UNKNOWN, LOW_STORAGE,
NETWORK_ERROR, DUPLICATE_REQUEST, INSTALLER_ERROR, USER_CANCELLED,
AUTHORIZATION_ERROR, SUCCESS. Only a normalized enum and numeric provenance
cross the boundary. Raw error text, tokens and URLs do not. An error containing
SUCCESS never authorizes install or a session. New attempts clear stale error
provenance; unrelated/retired responses cannot replace it. The PWA displays
native state and the distinct handoff reason without granting authority.

### Continuous monitor and disconnected-phone instructions

The user requires a continuing observation of the actual update/app lifecycle.
Helper3 watches actual package identity and trusted started/focused-ready
acknowledgments until completion or explicit cancellation. There is no overall
download expiry or estimated Meta relaunch grace. Package events wake observation;
poll intervals/backoff only pace actual reads. Durable once-per-real-boot launch
consumption and the existing singleTask launcher prevent foreground retry loops.
Reboot resumes a pending target with current-boot proof. Short per-IPC/main-loop
deadlines still reject stale commands; they never establish download completion.
New clients require explicit monitor3 capability before closing. Legacy helper
boot metadata/replies remain compatible with older clients.

The PWA must retain only a presentation notice across connection loss. A native
confirmed handoff may say the headset is updating; a check interrupted before
that evidence must report its last confirmed state and unknown current status.
While the app is closed, no progress or completion may be invented. Tell the
operator to inspect the update in the headset and reconnect the PWA once
Atmungsraum is open. Only fresh same-headset actual APK identity and native
readiness can complete that notice; local storage, elapsed time or dismissal
cannot release session/update authority.

Latest hardware readback (20:15): Meta has subsequently installed16. Its exact
APK digest matches the published16, installer/initiator are Meta, and Android
records lastUpdateTime19:43:44. The headset is asleep at readback. This establishes
a delayed Meta15-to16 transition after the earlier bounded negative boot window;
it does not establish immediate update-on-autoboot or precise continuously
observed installer timing. Post-update boot/kiosk remains pending. The user-
approved explicit in-app startup updater continues, with SDK-enabled17/18 next.


PROJECT.md owns the active lifecycle. The user accepts the latest attended
APK/PWA as sufficient for now and requests power-on launch, menu/foreground
recovery and a genuine Meta-distributed update while autoboot is enabled.
Both the outdated baseline and its successor must be installed through Meta.

## Accepted architecture

Meta rejected the attempted code13 upload because
`android.permission.SYSTEM_ALERT_WINDOW` is unsupported. The existing native
boot receiver requires that permission. The user explicitly approved a separate
one-time USB-installed boot helper, leaving both experiment versions eligible
for Meta distribution.

- Main baseline15 (`0.2.14-dev`) and successor16 (`0.2.15-dev`) retain the
  existing package/signer, controller-only policy, private visual inputs,
  phone pairing, audio/calibration/data behavior and own-app kiosk watchdog.
  Default boot opt-out must omit boot receiver/metadata and the unsupported
  overlay permission from their actual manifests.
- The separate `org.viscereality.atmungsraum.boot` helper uses the existing
  developer signer. Only this helper declares overlay and boot permissions.
  It has no network, installation/update, Bluetooth, sensor, Accessibility,
  device-admin or arbitrary-command authority.
- At each genuine new boot, with effective overlay permission, the helper
  resolves the currently installed `org.viscereality.atmungsraum.developer`
  MAIN/LAUNCHER entry and verifies its signature matches the helper. No APK
  path, version or activity name is pinned. Admission is persisted before an
  at-most-once launch; dispatch is not readiness. Ordinary main-app updates
  therefore do not require helper updates. Package/signer or OS compatibility
  changes may require maintenance and must be qualified.
- The helper does not relaunch after an explicit in-boot app exit, close
  protected OS prompts, download updates or govern experiment state. The main
  app owns foreground recovery, protected escape and recording durability.
  Stock-Quest soft recovery is not tamper-resistant lock-task mode.

## Test order and evidence

1. Inspect actual signed15/16 packages: source/input closure, package/version,
   retained signer, actual manifest/DEX/assets, signature/alignment and exact
   functional-composition comparison. Run focused helper policy tests and
   inspect its signed package separately. No native/profile/cue/math changes.
2. Publish15 only to the existing private Developer channel. Verify Meta
   processing and unchanged channel access/member/launch policy. The user
   installs/updates through the Quest's Meta library/channel. Verify actual
   installed version, signer, installer/provenance, UID and retained state.
   Neither local Dev Hub/ADB installation nor channel entitlement proves delivery.
3. Install/provision the helper once by USB, with effective permission readback.
   Keep unrelated settings/services, original signer, pairing and recordings.
   No headset enrollment, device-owner or account-mode change is authorized.
4. Attended genuine reboot with USB observation. No host app launch, synthetic
   BOOT_COMPLETED, wake bypass or competing launcher may establish success.
   Capture real boot admission, target-owned OpenXR focus/frame progression and
   Android placement. The wearer clears real lock/Guardian/permission prompts.
   Test physical Meta menu/Home recovery and explicit operator exit separately.
5. Only after Meta-delivered15 passes, publish inspected16 to the same channel.
   Preserve15 as the outdated installed baseline. Observe an actual15→16
   transition through Meta without host installation or a manual Update action.
   Bind version, signer, UID/state retention and app-reported identity. Record
   whether delivery occurs while running, during idle, or after another boot.
   Verify post-update helper launch and kiosk again.
6. Do not infer immediate update from upload, entitlement, Store presence or
   reboot. If delivery stalls, inspect active channel/provenance and update
   timing before blaming autoboot or designing an app-owned idle window.
   Never forge installer attribution, uninstall/downgrade, or automatically
   terminate a recording to manufacture a successful result.

## Checkpoint and remaining work

- [x] Explicit native boot recipe option with default opt-out, real resolver
  tests3/3zero skips, boot/kiosk gates and independent source/final reviews.
- [x] Inspect signed local12/13 candidates, exact same signer/functional inputs;
  normalized composition equivalence recorded, not native binary identity.
- [x] After explicit idle/USB readiness, local12 install confirmed by exact
  remote APK digest/version and retained UID. It is only a local smoke-check.
- [x] Record Meta13 unsupported-permission rejection; channel remains code3.
  Code14 was cancelled before signing; no qualified14 release is claimed.
- [x] Inspect Meta-compatible15/16 candidates and signed boot helper;27helper
  policy checks and independent task/package review pass.
- [x] Publish15 to the existing private Quest2 Developer channel; Meta completes
  its checks, malware passes and vulnerability details report all tests passed.
- [x] Establish actual Meta-delivered15 baseline and provision the helper.
- [x] Prove genuine boot, effective kiosk/menu recovery and protected exit.
- [x] Publish16 and prove delayed automatic Meta delivery (exact digest/Meta installer/retained UID at20:15).
- [ ] Qualify post-update boot/kiosk and the new explicit SDK path.
- [x] Publish verified split implementation; PR27/Pages43921087088 and18 live files pass.

Actual Meta15, helper installation/permission/opt-in, genuine boot and physical
menu recovery/escape are now verified. Meta16 is published with completed tests.
The subsequent genuine boot observes no16 across69 samples over about208seconds; later
closed/asleep readbacks also remain15. Automatic delivery and post-update behavior
remain open. PR27/Pages43921087088 and18 public-file comparisons pass. The user
now authorizes the extension below. Phone control stays with the user, Location
stays ON. Account/device IDs, raw logs and APKs remain private.

## Extension: automatic in-app Meta updates — 2026-11-16

The user explicitly requests that the APK check, download and install available
Meta-channel updates at startup, including when launched by the boot helper.
The helper remains unchanged. Current actual baseline15 was installed by Meta
with retained UID; genuine boot and physical menu recovery/escape passed.
Meta16 is published. A bounded real boot observation (69 samples, about208s)
and subsequent closed/asleep readbacks still showed15. A later20:15 readback confirms actual16, lastUpdateTime19:43:44. This proves
delayed automatic delivery after the earlier bounded negative window.

### Binding behavior

- The updater is an explicitly selected optional native app composition. The
  Meta app ID and SDK paths enter through validated build inputs; other apps
  remain inert. Use official Platform SDK207 native C APIs through a small owned
  JNI adapter. SDK archive SHA256 is
  `00c4035dce5cda094eaf0272ad70f2deda389c32bf0d3eacc12135c16d385401`;
  actual ARM64 loader has16KiB PT_LOAD alignment. SDK binaries stay external.
- A genuine trusted cold native launch arms maintenance before preparation can
  begin. Start SDK work only after first submitted frame/current owner proof.
  Do not block XR rendering. Initialize asynchronously, request version, download
  only when the numeric latest code is strictly greater than actual installed
  code, poll progress, then request install-and-relaunch with default options.
  SDK current-code disagreement or malformed values fail closed for updating.
- One shared native admission fence serializes maintenance with actual breathing
  setup, calibration/rehearsal, Arm/Start, delayed experiment entry and recording
  persistence. IDLE alone is insufficient. Direct/local commands retain the same
  fence. No update starts during a session; revalidate before install. No locks
  are held across SDK/JNI callbacks. After durable pending state and native
  install admission, latch a process-local kiosk handoff on the main looper
  before SDK dispatch. Cancel scheduled recovery and departure accounting;
  retain suppression through ambiguous outcomes and warm activity restarts.
  Fresh installed-process startup restores ordinary kiosk initialization. Owner retirement invalidates callbacks.
- Initialization/version have15s limits each; download stalls after45s without
  byte progress or reaches a10min total limit. Before install dispatch, failures
  revoke the attempt's install authority and permit normal installed-version
  setup with an explicit warning. Never interpret a late callback as authority.
  No automatic retry in the same process. A Quest restart provides a fresh trusted
  process for retry; normal app closure may retain its cached process. A warm
  trusted activity relaunch reports a fresh safe warning instead of stale success.
- Persist a minimal pending target/attempt before requesting install. SDK dispatch
  or acknowledgement is not proof of installation. After dispatch, timeout or
  ambiguous outcomes stay blocked with a restart-required explanation; do not
  allow preparation while a late install may still terminate the process.
  Bound repeated same-target installation across failed relaunches. Only fresh
  actual package identity can confirm completion. Do not persist launch trust.
- Preserve strict existing trusted MAIN/sole-LAUNCHER admission. Do not weaken it
  based on an assumed SDK relaunch intent. Qualify actual Meta relaunch; a distinct
  route needs an explicit reviewed one-shot provenance contract if observed.
- Native status provides optional `meta_update` version1: `state`, `reason`,
  `installed_code`, nullable `available_code`, nonnegative `downloaded_bytes`,
  nullable `total_bytes`, and `session_allowed`. States: `waiting`, `checking`,
  `current`, `downloading`, `installing`, `updated`, `failed`, `restart_required`.
  Busy states and restart_required deny preparation. Legacy feature absence
  preserves existing behavior. Malformed advertised status is not success.
- The PWA displays native progress/warnings beside APK identity and before
  preparation; existing controls respect the native gate. Public release JSON
  remains an availability hint, never install authority. Rendering, refresh and
  preview polls never initiate an update. No new remote update commands in this
  increment; retry is a clearly described Quest restart after safe failure.
- Preserve signer, package, controller-only policy, private visuals, breathing
  math/audio/900s recording and stored phone trust. No overlay, installer, admin,
  Accessibility or new account permissions in the experiment APK.

### Implementation and qualification order

5. Implement the native SDK bridge and explicit packaging selection, with exact
   external artifact hashes, SDK/API/ELF/dependency and signed APK closure checks.
   Test asynchronous request IDs, errors and stale callback behavior with a fake
   SDK; compile against actual SDK207 headers/library. Preserve opt-out packages.
6. Implement the native maintenance/session fence, trusted startup orchestration,
   persisted attempt/relaunch bookkeeping and read-only status. Test races,
   timeouts, equal/lower/latest versions, direct bypasses, setup-under-IDLE and
   post-dispatch uncertainty. Independently review tasks5/6 together.
7. Implement the PWA display and native readiness gate, preserving stable DOM,
   scroll, edits, existing journals and100ms preview-only cadence. Test valid,
   absent and malformed status plus all blocking/terminal states and render
   narrow/200% text cases. Review the task and whole increment.
8. Build a new SDK-enabled baseline and higher successor with retained private
   inputs/signer. Publish both through the existing private Meta channel in
   sequence. Establish actual Meta baseline, then genuinely boot it with a newer
   version available; prove download/install/new package identity, data/UID/trust,
   post-update initialization, focus/frame progression, boot-helper and kiosk.
   Never count a host install, channel listing or SDK acknowledgement as success.
9. Update existing PROJECT/evidence and publish each verified source/PWA increment
   through ordinary PR/exact Pages/live checks. Keep physical update acceptance
   open until the Meta-managed transition is witnessed.


## User amendment: repeatable checks, session freshness and launch fallback — 2026-11-16

This supersedes the earlier startup-only/no-remote-command/fresh-process-retry
limits. The user requests manual check/update from the PWA and a fresh check
for each new experiment session. On unavailable internet/Meta, the user selects
**allow the installed version with a warning**. A known newer required build or
uncertain installation still blocks new sessions. Offline cannot discover an
unseen release. Remember the highest known required code across process restarts;
a failed check, lower/stale channel response or phone cannot erase that floor.
The user also authorizes extending the existing separately installed helper for
post-update launch, and adding existing public Viscereality cover/library artwork
to the private Developer listing. No public Store release or new account access.

### Observed boundary and intent

Meta17 was genuinely installed and autobooted. Boot10 GetVersion reported17
although the channel dashboard already showed18. Boot11 reported18; download
progress was rejected by our cross-state monotonic check, and the SDK subsequently
reported download SUCCESS. Exact rejected sample/status was not captured.
Boot12 launched17 before ordinary Meta replacement installed18 at22:00:41;
Meta/UID/exact18 hash match, but no in-app install request or automatic relaunch
was observed. Root explicitly host-launched18 afterward. Do not count that as
SDK relaunch proof. Progress correction7b442a41 scopes monotonic telemetry to
active DownloadQueued/Downloading, retaining strict bounds and independent
Success plus subsequently issued fresh progress. Independent152/3/12 checks
pass; package/hardware qualification for that correction remains pending.

### Contract for implementation

- Extend the **same** native cycle/update owner, not a second updater. A native
  preparation cycle exists before Arm, bound to actual owner/epoch. After actual
  terminal writer finalization and acknowledged parked preparation/audio, advance
  the cycle and require a new verification. PWA navigation, reconnect, pause,
  resume or save request alone cannot advance it. Delayed main/Arm/Start effects
  must retain and revalidate the same cycle grant. Existing calibration math,
  32-second loop and main900s clock remain unchanged.
- SDK initialization is process-owned; checks have distinct random attempts and
  request lineage. Retire an attempt's IDs separately from final process close.
  Admit before invoking an SDK function, discard stale responses, serialize the
  pump, and do not hold native authority locks over JNI/SDK/network calls.
- Check/update requests are allowed only at an actual safe parked boundary.
  Active calibration/tracking/preparation, queued/main/paused/writer/finalization,
  unknown mutation and unresolved install return typed busy/prerequisite; they
  must not secretly cancel preparation or queue a later install. Existing status,
  acquisition, receipt lookup and terminal cleanup remain available.
- Every new cycle automatically starts one native **check-and-update** attempt.
  A manually requested check uses the same path. If newer required code is known,
  sessions remain blocked while download/install/relaunch runs or fails. Only
  fresh actual package identity satisfying the remembered floor clears it.
  Safe unavailable checks with no known higher floor release with warning.
- Native independently fetches the composition-bound canonical HTTPS required
  manifest: `https://viscereality.org/atmungsraum/quest-release.json`. Add explicit
  `meta_required_version_manifest_url` schema/resolver/metadata/source closure;
  it is never supplied by the phone. Bounded fresh GET, no credentials/redirects/
  cached304 fallback, deadline/size limits and strict package/integer validation.
  Public minimum is required floor; latest is a hint. Native Meta latest also
  establishes known update availability. Keep actual installed, policy minimum
  and Meta latest separate. A required floor above Meta's offered code is
  `required-update-not-yet-on-channel`, never current. Retain normal Meta install.
- Persist pending installation before dispatch; exact main-looper kiosk handoff
  and native irreversible admission remain mandatory. No retry during ambiguous
  installation, no remote acknowledgment/unavailable policy clearing phase3.

### Shared wire/API agreement

Native private enrolled LAN status advertises `meta_update_control:1` and
`meta_update.v:2`. Preserve existing v1 read-only support. v2 has existing fields
plus `cycle` and `revision` canonical positive decimal strings, nullable `attempt`
canonical positive decimal string, `required_code` positive integer or null,
`can_check` boolean and bounded canonical `busy_reason`. Existing states plus
`update_required`; session_allowed is native truth. Missing/invalid selected-v2
state must not manufacture a session grant. Meta-v1/legacy exposes no new actions.

One enrolled-current-controller private-LAN action `meta-update`, using the
existing authenticated command journal/owner/controller revision and payload:
`{v:1,id,op:"check-and-update",cycle,revision}`. Read-only original receipt lookup:
`{v:1,id,op:"receipt"}`. IDs use the existing project journal ID format and exact
original canonical body binding. Reply includes v/id/state (accepted, busy,
stale, unsupported, rejected), cycle/revision/attempt and canonical reason.
Same ID cannot start another attempt; lost replies reconcile the original ID.
Automatic native attempts are adopted/read by the PWA rather than duplicated.
No URL, APK, AppID, component, shell or arbitrary SDK method is remote input.

Service bridge to be agreed by native owners before editing shared seams:
Java service owns request/pump/policy/persistence, Rust owns cycle/grant/park/
effect admission. Cycle/revision must be native-authoritative and included in
all manual admission and release. Distinct release-with-grant versus blocked
required-update release; never revive an irreversible install. Do not merely
reset session_seen, Java acquired or JNI Queue.closed. Preserve disabled builds.

PWA Settings offers `Updates prüfen und installieren`, native busy explanation,
progress/error/required-code and original receipt recovery. Auto-cycle preflight
is visible before new participant/preparation. Existing stable DOM/focus/scroll,
Pretext narrow320/200% layout, journals and recordings/Drive remain intact.

### Helper fallback contract

Update the existing helper once, retaining enabled/overlay preferences and signer.
Normal main updates then use it without ADB or another helper update. Only exact
same-signer Atmungsraum launcher may be launched; helper never installs/downloads.
Use a signature-protected exact-main-caller handoff before install, with durable
attempt/expected-newer-code/current-boot and finite expiry. Give Meta an initial
relaunch opportunity; stop on acknowledged new trusted launch/frame and expire
otherwise. At most one fallback launch for the bound attempt/actual target version.
Also cover version replacement during the original unconfirmed boot launch.
Do not create an indefinite foreground watchdog or undo intentional kiosk escape.
Early trusted-launch/frame acknowledgments and explicit exit revoke watcher
state where needed. No broad broadcast/APK URL/arbitrary launch authority.
Exact helper/main protocol and Android34 background/FGS closure must be agreed
and tested; do not assume the existing one-shot BOOT receiver stays alive.

### Bounded implementation tasks (same SDD/lifecycle owner)

10. Complete reviewed progress correction and bounded numeric diagnostics;
    preserve final18 package identity and historical source in Git. **Software
    complete at native7b442a41;152 coordinator/3 metadata/12 kiosk checks pass.**
11. Native Java/JNI reusable attempts + canonical policy reader/explicit builder
    composition + selected offline/known-floor semantics. Red tests before edits;
    stale IDs, repeat checks, no SDK effects on invalid attempt, policy boundaries.
12. Native cycle/grant/admission + safe post-finalization park/ack + authenticated
    journaled gateway. Agree service seam with11. Test queued effects, writer/
    calibration/finalization races, same-PID two sessions and all bypass paths.
13. Same-signer bounded helper fallback and main handshake, including boot update
    race and explicit exit. Tests permission/caller/version/replay/expiry/warm
    launch/ack/no-duplicate. Root owns device install/reboot, no worker effects.
14. PWA action/progress/required-version/session-check UI and tests against frozen
    contract, real Pretext/stable DOM/reflow. Independent cross-boundary review.
15. Build/inspect/sign with retained private inputs/audio/cert, increasing Meta
    versions, actual two-cycle/manual/offline/known-update qualification and real
    Meta install/relaunch/focus. Publish source/PWA via ordinary PR + exact Pages
    and live assets; update installed phone. Keep hardware criteria open until
    actual proof. Existing task8/9 acceptance stays active.
16. Use appropriate existing public artwork for Developer listing assets; inspect
    actual Meta slots, retain private distribution and verify saved result.


Historical helper2 wire freeze for tasks11/13 (the monitor3 amendment above
supersedes its finite duration/grace and capability behavior): declare signature permission
`org.viscereality.atmungsraum.boot.permission.UPDATE_HANDOFF` in helper2 and
provider authority `org.viscereality.atmungsraum.boot.handoff`, exported only with
that permission. Provider also checks Binder UID resolves exact main package and
same current signer; main verifies helper package/signer before calling it.
Use ContentResolver.call, methods `arm-update`, `launch-started`, `launch-ready`,
`cancel-watch`, `status`. Bundle v1; arm has attempt canonical decimal string,
installed_code and target_code positive actual values. Helper owns boot/time/
expiry, clamps duration to a finite internal bound, and synchronously persists
before accepting. arm reply has accepted boolean and canonical reason. Other
ack methods include immutable loaded-process installed_code; compare to actual
PM identity so a replaced old process cannot acknowledge the new binary.
Main trusted cold launch acknowledges started, first submitted focused frame
acknowledges ready; intentional kiosk escape cancels the watch. No raw paths,
components, caller-provided expiry or arbitrary package input. Helper status is
read-only. Both products retain same signing key. Main explicit composition flag
`experiment_meta_update_helper_enabled` selects metadata/protocol2 and custom
uses-permission; selected own SDK install requires positive helper arm ACK.
Main metadata `org.viscereality.atmungsraum.boot_protocol=2` lets helper enable the
bounded boot-update watcher only for compatible main builds; old18 retains
legacy one-shot boot behavior until main19 is Meta-installed. New helper FGS/
notification/receiver closure and actual Android34 eligibility must be qualified.
