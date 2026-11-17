# Set up another Quest and phone

Validated reference: **2026-11-17, Quest 2, Developer code22 / 0.2.21-dev**.
This is a handoff for an agent working with its headset owner. The owner already
has access to the private Developer distribution through their Meta account.
Detect the actual headset model and phone/browser first. Quest2 is the physical
acceptance target tested here; confirm Meta eligibility for other models.

## Task to give the agent

Set up Atmungsraum Developer on this Quest so every real restart launches the
current installed app, kiosk returns from accidental Meta menus, Meta updates
can install and the app returns afterward. Pair this phone with this headset's
PWA. Preserve existing recordings, settings and trusted phones. Work only on the
explicitly selected headset. Keep experiments and calibration stopped during
installation, restart and updater tests. Record actual results and remaining
limitations; do not treat a successful launch command as proof of a visible app.

## Files and identities

The setup bundle supplied with this handoff contains the **already signed**
`atmungsraum-boot-helper-v3.apk` and checksums. Install that helper once by USB;
the main experiment app must come through Meta. No developer signing key or
private particle assets are needed for this setup.

| Item | Value |
| --- | --- |
| Main app | Atmungsraum Developer |
| Main package | `org.viscereality.atmungsraum.developer` |
| Meta app ID | `1459506690581676` |
| Release channel | Developer, private; ID `1529361979246951` |
| Qualified current version | code22 / `0.2.21-dev`, Meta build `1463496130182732` |
| Helper package/version | `org.viscereality.atmungsraum.boot`, code3 / `3.0` |
| Helper APK SHA256 | `b98de54c504ac284f79c34da2df79ce89a303a44e2ed178c7809f315c443539e` |
| Main and helper signer SHA256 | `372324e933ce701b9e71ff1a24239aeba8589231fa54e298c424e1f1a2cf570e` |
| Main22 APK SHA256 | `e0de22c7f8f54bcec85b49caddea2bab21f66fc55dd98689241aca7a0645c94a` |

Read the [current required/latest version policy](https://viscereality.org/atmungsraum/quest-release.json)
as well: a later release may supersede22. Do not downgrade a newer Meta build.
Verify the helper digest and signer before installation. Rebuilding it with an
unrelated key will fail the deliberate same-signer checks. Source access does
not provide the signing key; use the supplied signed APK. Obtain it from the
maintainer if it is missing. Keep the key private.

## Setup sequence

1. **Identify and connect the headset.** Use its normal Developer Mode and
   Android platform tools. The wearer approves USB debugging on the headset.
   Use `adb devices -l`, choose the exact serial and qualify every command with
   `adb -s SERIAL`. Follow local machine coordination rules. Do not restart a
   shared ADB server or operate another connected Quest.
2. **Install the main app through Meta.** In the normal library, install/open
   Atmungsraum Developer and select its Developer release channel. Verify the
   actual installed package and code, not just the release-channel listing.
   For the current reference that is22; newer qualified versions may be used.
   Do not substitute a local main APK. Do not uninstall or clear app data as a
   routine update step. If kiosk gets in the way of initial setup, close the
   idle main app using its protected escape or a serial-scoped force-stop.
3. **Install or upgrade the supplied helper.** With no experiment running:
   `adb -s SERIAL install -r atmungsraum-boot-helper-v3.apk`. Confirm helper code3
   and the same signer. Retain existing helper data on an upgrade.
4. **Enable its one-time setup.** Open Atmungsraum Boot Helper from Unknown
   Sources, or use its `.SetupActivity`. Allow its overlay permission and tick
   **Enable launch after restart**. Scroll if the checkbox is below the status
   text. Confirm **Overlay permission: Allowed** and the main target is Ready.
   The exact authorized ADB grant is
   `adb -s SERIAL shell appops set org.viscereality.atmungsraum.boot SYSTEM_ALERT_WINDOW allow`.
   Check it with the corresponding `appops get` command. This does not require
   device-owner enrollment, Accessibility or a new kiosk framework.
5. **Check Bluetooth setup.** The main app requests its Nearby devices/Bluetooth
   permissions on load; allow them. Headset Bluetooth and Android Location
   services must be enabled for the tested H10 acquisition path. These are
   persistent settings. The main package uses SCAN, CONNECT and ADVERTISE;
   inspect actual grants if discovery fails. The helper needs none of them.
6. **Pair the phone.** Open [the PWA](https://viscereality.org/atmungsraum/) in a
   supported Web Bluetooth browser; Chrome on Android is the tested route.
   Install it to the home screen if desired. Start the Quest app, choose
   **Mit Bluetooth verbinden**, select the headset, compare the codes and
   confirm on both devices. The headset needs no typing. Its approval panel
   opens for the incoming request. Each headset has its own phone trust;
   sharing a Meta account does not provision that pairing. Several phones may
   be remembered, with one active controller. Tap **Steuerung übernehmen**.
7. **Verify actual status.** In Einstellungen, check the installed APK code and
   Meta status. The initial Automatic input is parked at minimum expansion and
   silent. Explicit preparation/calibration starts the breathing loop. A worn,
   moistened H10 is needed for breathing tests, not for the boot/update test.
   Avoid another client simultaneously using its raw acceleration stream.

## Acceptance checks

- Perform a **real reboot**, keeping USB attached for observation. The wearer
  clears normal Quest startup prompts. Do not manually launch Atmungsraum;
  confirm that the stimulus appears automatically. Bind logs to a changed boot
  ID/count and actual advancing focused frames.
- Open the right controller's Meta menu once and wait. Confirm the app returns.
  Protected escape is **three separate Meta menu openings within five seconds**,
  allowing the stimulus to return between openings. It should exit to Home and
  stay there for the current boot. It does not disable the next boot's setup.
- At a safe idle boundary, the PWA's **Updates prüfen und installieren** asks
  for a fresh Meta check. Automatic checking also occurs on trusted startup and
  fresh preparation cycles. Check native results; a channel listing alone does
  not prove the SDK has offered or installed that version.
- If no newer build exists, a current-version result is correct. Mark actual
  replacement testing pending until a newer Developer build is published.
  Both baseline and successor must be Meta-installed. Do not sideload an older
  main APK to manufacture an update test.
- During an update the main app may close, ending its phone connection. The
  PWA cannot observe live progress while it is closed. Follow its instruction
  to check the headset and reconnect once Atmungsraum opens again. A retained
  notice is presentation only; it cannot authorize a session or finish a task.
- The helper monitors the actual installed target and startup acknowledgments.
  There is **no estimated download completion timer or overall download expiry**.
  Polling/backoff paces observations; short command deadlines reject stale IPC.
  A trusted started acknowledgment is followed by the first-focused-frame
  acknowledgment. Explicit Stop monitor/disable/protected escape cancels it.
- Verify the new installed code, signer, Meta installer, retained UID/data and
  foreground/focus after replacement. Never use a host launch to claim the
  helper's post-update relaunch passed. The helper resolves the current installed
  version; ordinary main updates do not require updating the helper.

Known newer/required versions and uncertain installation block a new session.
An unavailable update check can permit the installed version with a warning
only when no known higher required floor exists. These are native gates.

For diagnostic baseline launches use the real MAIN action and sole LAUNCHER
category, with the actual resolved main component. A component-only
`am start -n ...` intentionally does not create the trusted updater owner, even
if a scene draws. It can cause “Meta-Startprüfung … nicht gemeldet”. A normal
helper autoboot or properly formed cold launcher route is qualified. Inspect
the startup intent if another launch route reports no updater. Wi-Fi ADB may disappear at reboot; USB avoids confusing loss of the
diagnostic transport with an app failure.

## Code and owning documentation

- [Website/PWA repository](https://github.com/GeorgeFejer91/viscereality.org):
  `atmungsraum/for-ai/README.md`, `PROJECT.md`, `WORKFLOW.md`,
  `BOOT-KIOSK-UPDATE.md`, `PHONE-PAIRING.md`, `SHARED-BREATHING-SETUP.md`.
- [Helper source, manifest, tests and builder](https://github.com/GeorgeFejer91/viscereality.org/tree/main/atmungsraum/quest/boot-helper).
  Inspect `WatchPolicy`, `AndroidHost`, `HandoffProvider`, `BootReceiver`,
  `WatchService` and `SetupActivity` for the actual behavior.
- [Native build instructions and cumulative patch](https://github.com/GeorgeFejer91/viscereality.org/tree/main/atmungsraum/quest)
  and [qualified APK log](https://github.com/GeorgeFejer91/viscereality.org/blob/main/atmungsraum/quest/apks/README.md).
- [Rusty Quest base repository](https://github.com/MesmerPrism/rusty-quest).
  The current cumulative `native-source.patch` applies to base
  `99352620fd79ec5cdcee71b47c2ba4056241af16`. For21/22 it reconstructs tree
  `a9dc76d2014f915d7ac7d9756fc581346dad5faa`, source commit label
  `1cd5ee3b69550912b5d942e8ca893a25f1a7f361`. Use the patch instructions;
  do not assume that local commit label is directly fetchable upstream.
  Native owners are `NativeRendererMetaUpdateCoordinator/Service/Sdk/Helper`,
  `NativeRendererMetaPlatformTransport` and `NativeRendererSelfKiosk*` under
  `apps/native-renderer-android/src`. Main helpers require monitor protocol3.
- [Viscereality renderer/tuner repository](https://github.com/MesmerPrism/rusty-gpu-viscereality)
  is a related reference. It is not required to provision this Meta app. The
  private production visual inputs and signing key are not supplied by a normal
  website checkout.

Reference hardware result: boot12→13 automatically launched21, downloaded and
installed Meta22, then reopened22 with advancing focused frames and the same
UID. Installed bytes matched the qualified22 APK. The helper's fallback request
was captured; Quest shell performed the final activity launch. The user did not
manually launch or request the update. This run exercised the normal SDK install
lane. The separate duplicate/queued lane and PWA disconnected notice have
software coverage; do not claim their distinct physical qualification from this
run. Calibration and PWA fine tuning remain separate work.
