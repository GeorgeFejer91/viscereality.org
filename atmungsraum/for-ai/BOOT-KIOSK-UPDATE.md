# Autoboot, kiosk and Meta update qualification — 2026-11-16

The user accepts the latest attended APK/PWA behavior as sufficient for now;
further UI/calibration tuning remains possible. The next authorized task is
power-on app launch, foreground/menu recovery and actual Meta Developer-channel
N→N+1 delivery while autoboot is selected. PROJECT.md remains the lifecycle owner.

## Composition and test order

1. Retain code11's package, developer signer, controller-only policy, private
   particle bundle, audio/calibration/data behavior and phone pairing. Select
   the existing `ui.experiment_boot_autostart` feature explicitly for baseline
   code12 (`0.2.11-dev`). Preserve default opt-out for other assemblies.
2. Run focused boot/kiosk and actual signed-package checks; install baseline12
   only when idle. Retain UID, grants, saved phones and recordings. Configure
   only its actual prerequisites. This composition selects the app-owned
   self-watchdog, not the separate Accessibility adapter: do not enable that
   unrelated service. Autoboot requires the app's overlay permission. Confirm
   effective state; a requested flag is insufficient.
3. Attended cold reboot with USB observation. No host app launch, synthetic
   BOOT_COMPLETED, wake bypass or competing launcher may establish boot success.
   Capture genuine boot receiver admission and target-owned XR focus/frames.
   Test normal Meta menu/Home recovery separately; protected OS prompts and an
   explicit operator exit must remain usable. Stock-Quest foreground recovery
   must not be described as tamper-resistant lock-task mode.
4. Prepare code13 (`0.2.12-dev`) with the same functional composition and signer.
   Qualify locally before Meta upload. Preserve baseline12 as the outdated
   installed test state; do not ADB-install13 on the target to claim delivery.
   Any local temporary version changes need a concrete data-preserving route.
5. Use existing Atmungsraum Developer application/private Developer channel and
   joined owner membership. No public Store release, new tester invitation,
   new key or account change. Check current channel/build and Meta scans before
   assigning the new build. Retain normal channel launch policy so it cannot
   fabricate an automatic-update result by refusing the old build.
6. Observe an actual installed12→13 transition through Meta, binding package,
   signer, installer, UID, saved state and app-reported build identity. Record
   whether delivery occurs while autobooted/running, during an idle window, or
   after a restart. Do not infer immediate update from upload, entitlement,
   Store presence or reboot. If delivery stalls, diagnose before designing an
   app-owned update window; no automatic session termination.

The current code11 installation was initiated by the Android shell and has no
installer package recorded. That is a local sideload baseline, not evidence of
Meta-managed installation. Before interpreting a failed update as an autoboot
conflict, verify the headset account's entitlement, active Developer channel,
library entry and actual installation provenance. If Meta adoption requires an
explicit library install/update, record that as setup, not automatic delivery.
If an additional baseline channel upload is necessary after local qualification,
retain data and the signer; never forge an installer identity, uninstall or
downgrade to manufacture a Store-managed result. A subsequent higher-version
transition must still be observed without host installation or manual Update.

## Boundaries

User now explicitly authorizes a qualified newer Meta upload for this test,
superseding the prior instruction to keep every iterative build local; local
hardware qualification still precedes submission. No experiments/recordings are
interrupted implicitly. The wearer must be available after a reboot to clear
any real lock/Guardian/permission surface. Keep the phone user-controlled.
Keep signing files, account identifiers, raw device logs and build artifacts
private. Publish verified source/recipe/evidence through the existing workflow.

## Implementation steps

- [x] Add an explicit recipe option selecting the existing boot feature; test
  default/selected resolution and actual manifest/Java closure.
- [x] Build and inspect baseline12 using the unchanged private input bundle.
- [ ] Install/qualify boot and kiosk on the exact attended Quest2.
- [ ] Build/qualify13 and publish it only to the existing private channel.
- [ ] Prove actual Meta delivery and post-update autoboot/kiosk/state retention.
- [ ] Record results, limits and exact source/build/publication checkpoints.

Current checkpoint: source review and real resolver3/3zero-skip tests pass.
Both signed12/13 packages are inspected;90/98ZIP entries match exactly, with
version, run-derived provenance/native binding and signature differences
recorded privately. This is functional-composition equivalence, not native
binary identity. Code11 remains installed. No candidate permission setup,
install, reboot or Meta upload has occurred; attended idle/USB confirmation is
pending. The source/build-preparation increment is ready for normal GitHub and
Pages publication; hardware/update acceptance remains open in PROJECT.md.
