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
- [ ] Establish actual Meta-delivered15 baseline and provision the helper.
- [ ] Prove genuine boot, effective kiosk/menu recovery and protected exit.
- [ ] Publish16 and prove automatic Meta delivery plus post-update behavior.
- [ ] Publish verified split implementation and exact acceptance evidence.

No helper permission grant/install or reboot has yet occurred. Main15 is Meta-
published and16 is local/unpublished. The user thought15 had arrived, but fresh
Package Manager and exact hash readback still show local12 with shell installer
provenance. Actual library/active-channel diagnosis is pending; do not count
this as automatic delivery. Phone control remains with the user; Location stays
ON. Signing files, account identifiers, raw device logs and build artifacts
remain private.

Source-preparation PR26 merged at `2418869c19b0fb1f84723fcf566c225bd9dfcf5b`;
exact Pages run43915823855 succeeded and eight affected/preserved public files
matched the Git blobs. Subsequent split/evidence changes remain in progress.
The overall task stays open until its own hardware/update criteria pass.
