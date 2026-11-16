# Autoboot, kiosk and Meta update qualification — 2026-11-16

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
- [ ] Publish16 and prove automatic Meta delivery plus post-update behavior.
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
and subsequent closed/asleep readbacks still showed15. This establishes no
observed automatic15-to16 transition, not a general failure of Meta updating.

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
  are held across SDK/JNI callbacks. Owner retirement invalidates callbacks.
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
