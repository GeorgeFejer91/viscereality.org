# Keeping Quest awake during testing

Use this runbook for development headset checks and the attended Atmungsraum
rehearsal. Before touching the headset, follow the
[Quest device operations route](SKILLS.md#quest-device-operations) and its
resolved Meta Quest playbooks. The selected provider owns device changes.

## Set up the bench

1. Connect the headset directly with a known USB data cable. Keep the host
   powered and awake for the test window. Start with a charged headset, leave
   ventilation clear, and check the battery trend. Pause on a thermal warning
   or if the charge will run out before the test ends.
2. Have a wearer authorize the trusted debugging host, wake to Quest Home, and
   clear lock, boundary, tracking, and permission prompts. Developer mode and
   USB debugging must already be set up for the intended headset.
3. Privately record the headset model, canonical identity, current ADB selector,
   OS build, boot identity, intended package/activity, and ADB path/version.
   Bind every device command to the exact selector. A USB serial and a Wi-Fi
   endpoint can refer to the same physical headset. Give one operator control of
   awake policy, installs, launches, input, and capture. Other observers use
   read-only checks. [Android ADB guide](https://developer.android.com/tools/adb)
4. Capture the prior power/proximity settings and agree on the final settings,
   hold duration, and stop time before applying an override. Follow the machine's
   coordination rules for exclusive device access and transport changes.

## Pick one awake method

### QFM for one local headset

Use QuestIonAble File Manager's typed keep-awake command. Confirm that the
installed executable exposes the command and the expected provider. The syntax
below is from QFM Labs Alpha.14. Replace `YOUR_ADB_SELECTOR` with the confirmed
selector and use the installed CLI path. [QFM Alpha.14 release](https://github.com/MesmerPrism/QuestIonAble-File-Manager/releases/tag/v0.5.0-alpha.14)

```powershell
$selector = 'YOUR_ADB_SELECTOR'
questionable-file-manager.exe devices --json
questionable-file-manager.exe device status --serial $selector --json
questionable-file-manager.exe device keep-awake --serial $selector --on --duration-ms 1800000 --confirm-device-settings --json
questionable-file-manager.exe device status --serial $selector --json
```

This example requests a 30-minute proximity hold. Use an explicit duration
between 60,000 and 28,800,000 ms. Require the receipt to confirm stay-on,
awake/display state, virtual proximity, and positive hold time with the expected
duration. QFM applies the hold once. When it expires, check Android stay-on
separately. Restore custom prior values through their owner and read them back.
Use the following command only if normal policy is the agreed final state.
[QFM awake control](https://github.com/MesmerPrism/QuestIonAble-File-Manager/blob/f38f9d245d0669dff0d28d7de514ac38144b3a54/docs/quest-awake-control.md)

```powershell
questionable-file-manager.exe device keep-awake --serial $selector --off --confirm-device-settings --json
```

### Configured Fleet watchdog for long runs

Use the existing Fleet Hub only with an enrolled headset, reviewed target
identity revision, and pinned awake provider. Choose the Windows watchdog or
the temporary device-side helper. Run this example from the configured Fleet
checkout. Replace the target placeholder, review the preview before executing,
and substitute the IDs it returns.

```powershell
cargo run --locked -p fleetctl -- awake-preview start-device-watchdog 1800000 5000 'FLEET_ID@IDENTITY_REVISION'
cargo run --locked -p fleetctl -- awake-execute OPERATION_ID PREVIEW_ID
cargo run --locked -p fleetctl -- awake-get OPERATION_ID
```

The example sets a renewable 30-minute hold and a five-second poll. Assign and
enforce a separate watchdog stop deadline. Use fresh preview/execute operations
with the reviewed target for `status`, `stop-watchdogs`, or `restore-normal`.
`awake-get` retrieves the
stored receipt. `status` reads the device again. `stop-watchdogs` stops repair
and leaves policy in place. `restore-normal` stops repair and applies normal
policy. The device helper is temporary and must be re-established after a reboot
if the test still requires it and the original authorization applies.
[Fleet awake control](https://github.com/MesmerPrism/rusty-fleet/blob/94c12c6f181db4380596ff3736e155b66bc6ffda/docs/QUEST_AWAKE_CONTROL.md)

### Meta VR Developer Hub for manual bench work

Open **Device Manager**, select the headset, and use **Device Actions →
Proximity Sensor** to disable proximity detection. Check the effective state
and displayed duration. Meta documents reactivation ten minutes after MVDH
disconnect, or at the time shown by the tool. Keep the connection active for
the test and restore the prior setting at handoff.
[Manage your device with MVDH](https://developers.meta.com/vr/documentation/unity/ts-mqdh-basic-usage/#turn-off-the-proximity-sensor)

### Other development routes

- On a dedicated headset provisioned for Meta scriptable testing with one test
  user, read the current `disable_autosleep` value with `GET_PROPERTY` on
  `content://com.oculus.rc`. Use the signed-in test account's Store PIN securely
  with `SET_PROPERTY` to change only `disable_autosleep`. Confirm `Success=true`,
  read the property back, and check effective display state. Restore the prior
  value and verify it after testing. The setting persists across reboot. Keep the
  PIN out of shell history, shared logs, and receipts. Leave Guardian and
  security dialogs enabled and skip device wipe/setup commands.
  [Meta scriptable testing](https://developers.meta.com/vr/documentation/native/android/ts-scriptable-testing/)
- For a native app you maintain, `FLAG_KEEP_SCREEN_ON` can keep the display on
  while the relevant activity is visible. Clear the flag when the activity no
  longer needs it. Verify Quest wear state and XR focus independently.
  [Android screen-on guidance](https://developer.android.com/develop/background-work/background-tasks/awake/screen-on)

## Verify readiness before taking evidence

1. **Transport.** Check the selected route is `device` and returns a fresh shell
   response. After a reboot, check `sys.boot_completed` and repeat every higher
   check. [Android ADB states](https://developer.android.com/tools/adb)
2. **Power and blockers.** Confirm awake state, display on, expected proximity
   hold, and no protected lock or boundary prompt. Stay-on is a policy field.
   A `CLOSE` reading alone can reflect a wearer. Check the hold receipt and live
   state together.
3. **Visible UI.** Confirm the intended foreground window/activity and use a
   fresh screenshot or capture for visual assertions. Verify fresh app-owned
   progress or heartbeat. A running PID can outlive a stalled UI.
4. **Immersive rendering.** Check advancing valid Shell vsync before launch or
   placement. After placement, confirm the target owns the focused OpenXR
   session and its frames advance. If testing a media source, check that source
   separately. Use the supported XR input route for controller or hand tests.
   [Quest readiness signals](https://github.com/MesmerPrism/meta-quest-agent-workflow/blob/39a4641d34d806a7604e61dd5dc9e810d86608df/docs/quest-signal-patterns.md)

Check readiness before assertions and after build gaps. Start with a five-second
deadline for a shell echo, a larger finite deadline for dumps, and up to two
read-only retries with backoff. Use a cheap heartbeat every five to ten seconds
when waiting for the next test action. Record last success and repair counts.
Read before repairing actual drift. Keep foreground recovery separate from
awake repair and capture only what the assertion needs.
[Long-running watchdogs](https://github.com/MesmerPrism/meta-quest-agent-workflow/blob/39a4641d34d806a7604e61dd5dc9e810d86608df/docs/long-running-watchdogs.md)

### Serial-scoped ADB fallback

Prefer the provider's status or receipt. For a documented provider gap, use
bounded ADB calls and inspect output and exit status. Keep one known ADB
installation and the shared daemon for ordinary discovery and readback.
Set `$selector` to the verified ADB target before running these examples.

```powershell
adb devices -l
adb -s $selector get-state
adb -s $selector shell echo QUEST_ADB_OK
adb -s $selector shell getprop ro.build.fingerprint
adb -s $selector shell cat /proc/sys/kernel/random/boot_id
adb -s $selector shell getprop sys.boot_completed
adb -s $selector shell settings get global stay_on_while_plugged_in
adb -s $selector shell dumpsys power
adb -s $selector shell dumpsys vrpowermanager
adb -s $selector shell dumpsys window windows
adb -s $selector shell pidof YOUR_TARGET_PACKAGE
```

For an authorized temporary raw intervention, record the provider gap, exact
target, stop condition, and restoration first. The first command applies
Android's plugged-in policy. The second requests wake without toggling power.
Repeat the power and Quest power-manager readbacks after either command.
Restore the recorded setting after the run, including an unset prior value
where applicable. Use the selected provider for firmware-specific proximity
commands.
[Quest mutation and readback contract](https://github.com/MesmerPrism/meta-quest-agent-workflow/blob/39a4641d34d806a7604e61dd5dc9e810d86608df/docs/host-headset-mutation-confirmation.md)

```powershell
adb -s $selector shell svc power stayon true
adb -s $selector shell input keyevent KEYCODE_WAKEUP
```

If using wireless ADB, keep external power and USB recovery available. Verify
the current endpoint and an authenticated shell on the trusted network. Confirm
whether the headset uses TLS pairing or classic TCP on port 5555. Qualify any
boot helper on the exact firmware and expect that reboot may require attended
trust approval. Coordinate reconnect, `tcpip`, pairing, server-port changes, and
daemon recovery outside the heartbeat.
[Wireless ADB qualification](https://github.com/MesmerPrism/meta-quest-agent-workflow/blob/39a4641d34d806a7604e61dd5dc9e810d86608df/docs/boot-wireless-adb-qualification.md)

## Recover and finish

| Observation | Action |
| --- | --- |
| `unauthorized` | Pause UI actions and have the wearer approve the intended host. Repeat the read-only shell check. |
| Missing/offline device or command timeout | Pause assertions, reconcile any command with an unknown outcome, then check cable, port, power and transport with bounded reads. |
| Shell responds but the display is asleep | Check power, proximity and blockers. Repair the awake hold once within its bounds and require sustained awake readback. |
| `SensorLockActivity`, VR lock screen, Guardian or immediate return to sleep | Stop synthetic wake and launch attempts. Have the wearer wake with the physical power button and clear prompts to Quest Home. |
| Wrong foreground or stalled app | Reconcile the last action and check foreground and runtime separately. Take a bounded diagnostic capture and use the app's owner route for recovery. |
| Boot identity changed | Invalidate earlier readiness. Repeat authorization, boot, display, blocker, rendering and app checks. Reapply temporary helpers only within the authorized window. |

Exclude attempts with sleep, invalid vsync, or placement timeout from accepted
UI and performance samples. At handoff, stop run-owned watchers, recorders and
forwards whose lifetime ended, then restore the agreed settings and read them
back. Release run-owned leases. If a hold continues by agreement, record its
expiry, owner and stop procedure.

Keep the run receipt private. Record the provider/version, headset and transport,
OS build and boot identity, app, hold expiry and watchdog deadline, last good
transport/display/foreground/runtime evidence, repairs, restored settings and
outstanding wearer actions. Keep serials, network addresses, PINs, and raw
diagnostics outside this repository.
