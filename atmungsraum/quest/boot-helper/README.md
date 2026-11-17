# Atmungsraum Boot Helper 3

One-time USB/Wi-Fi ADB helper update, same package and developer signer. Existing
checkbox, overlay grant and `boot-helper` preferences are retained. Ordinary
main updates need no helper update. Runtime never requires a PC, installs an
APK, downloads content, controls Meta's queue or reports estimated SDK progress.

## Observed update monitor

An authorized main arms the exact installed code, strictly newer target code,
canonical attempt and real BOOT_COUNT. The helper retains that target across
process/service restarts and real reboots. There is **no download expiry or
Meta launch grace period**. PackageManager code must actually reach or exceed
the target. Package events wake observation; polling backs off from5seconds to
at most60seconds. Those timers only schedule observations. They never establish
download progress, package replacement, launch success or a failure deadline.

When the actual matching package is installed, the helper may issue one normal
MAIN/LAUNCHER request per real boot, durably consumed before dispatch. It
resolves the current enabled/exported component and exact current signer set
again. The actual launcher must be `singleTask`, as is the selected main's
`android.app.NativeActivity`: a simultaneous Meta launch then reuses that task.
No pinned component/version/path, data, selector or caller-controlled target.
Dispatch retains a finite five-second check; a stale dispatch is never retried
within that boot.

Only a trusted immutable loaded-process code equal to actual PackageManager
code can acknowledge startup. For an update it must also satisfy the target.
`launch-started` suppresses helper launch while retaining the monitor;
`launch-ready` from main's first FOCUSED submitted frame completes it. A request
returning or a startup acknowledgment alone does not prove focus/readiness.
Duplicate events, slow first focus, failed dispatch and process death cannot
repeat the helper launch in the same boot. No general foreground watchdog.

Protected intentional kiosk exit, explicit Stop monitor or disabling the helper
cancels durably. Overlay loss blocks launch while retaining read-only package
observation; restoring permission may admit the one still-unused request. A
replacement gap, absent package or wrong signer never launches and never
manufactures completion. Setup and foreground notification show the actual
monitor phase and expose **Stop monitor**. Notification cancellation is an
immutable PendingIntent into the non-exported service; no exported stop command.

After a real reboot, a pending update is rebound to the actual new BOOT_COUNT
and its per-boot dispatch/start consumption resets. The receiver resumes
observation instead of launching the old experiment. Failed durable resume
cannot fall through to an old-app boot launch. Once the target is installed it
may launch once in that new boot. Cancelled/completed watches cannot revive.
The bounded128-entry attempt ledger never evicts replay protection during a
boot; the pending attempt is retained across reboot. Format1 pending watches
migrate without inheriting their former elapsed-time expiry/grace fields.

## Provider contract and compatibility

The fixed provider `org.viscereality.atmungsraum.boot.handoff` requires signature
permission `org.viscereality.atmungsraum.boot.permission.UPDATE_HANDOFF`, Binder
UID resolving exclusively to the exact main package, and exact current APK
signer sets. Another same-signer app or shared UID is rejected. Calls accept
null arg and closed, typed v1 Bundles only:

| Method | Legacy fields besides Integer `v:1` |
| --- | --- |
| `arm-update` | String positive canonical decimal `attempt`; Long `installed_code`, `target_code` |
| `launch-started`, `launch-ready`, `cancel-watch` | Long immutable loaded-process `installed_code` |
| `status` | None |

Every monitor3 request additionally supplies Integer `monitor_protocol:3`, Long
`boot_count` and Long `deadline_elapsed`. The deadline must be unexpired, within
five seconds of helper elapsed time and tied to its current real boot. It is
rechecked after package queries and foreground-service admission. This bounds
individual IPC effects, **not download duration**. Late/failed admission revokes
only a newly created matching watch, never a prior accepted duplicate watch.
The client also rejects late replies and retains its own finite main-dispatch
deadline. No package/component/URL/path/expiry customization is accepted.

Replies contain Integer `v:1`, Integer `protocol`, Boolean `accepted`, String
`reason`, Boolean `watching`, Integer `monitor_protocol:3`, Boolean `monitor:true`.
The last two advertise capability; only accepted+watching confirms active arm.
Monitor3 requests receive protocol3; immutable legacy requests receive protocol2.
Helper metadata retains `org.viscereality.atmungsraum.boot_protocol=2` for old
clients and adds `org.viscereality.atmungsraum.monitor_protocol=3`. New main
metadata uses boot_protocol3, and its queued exit requires explicit helper
monitor3 capability. Helper2 cannot satisfy that requirement. Old protocol2
boot/handshake clients remain usable; main without a supported boot protocol
keeps the original once-per-boot launcher path.

## Setup and limits

Open **Atmungsraum Boot Helper**, enable **Enable launch after restart**, and
use **Open Android overlay settings** for the helper's normal Android grant.
Setup does not launch the target or count as boot proof. No overlay window is
created. This Android34 permission supplies background launch eligibility;
foreground-service status alone supplies no activity-launch authority. A
protected real BOOT_COMPLETED is required for normal boot admission; no
synthetic-broadcast entry exists.

The non-exported specialUse foreground service returns the boot receiver
promptly and uses START_STICKY to reload durable state after process death.
Opening setup can resume a pending monitor. Force-stop, platform service denial,
power loss, or the user stopping an FGS can prevent observation until the helper
is allowed to run again. No extra privilege bypasses those platform limits.
Uninstall/clear-data resets consent/history. Failed cancellation persistence is
reported rather than claimed saved. Actual Quest scheduling, task reuse,
focused-ready acknowledgment and Meta delivery remain hardware acceptance gates.

Permissions remain RECEIVE_BOOT_COMPLETED, SYSTEM_ALERT_WINDOW,
FOREGROUND_SERVICE, FOREGROUND_SERVICE_SPECIAL_USE and the signature provider
permission declaration. Visibility is limited to the exact target. No network,
storage, Bluetooth, sensor, UsageStats, accessibility, admin or root authority.
The helper never starts an old experiment to bypass an update floor.

Android references: [FGS background-start restrictions](https://developer.android.com/develop/background-work/services/fgs/restrictions-bg-start),
[specialUse service type](https://developer.android.com/develop/background-work/services/fgs/service-types#special-use),
[background activity starts](https://developer.android.com/guide/components/activities/background-starts),
[singleTask lifecycle](https://developer.android.com/guide/components/activities/tasks-and-back-stack#TaskLaunchModes).
Raising target SDK beyond34 needs a new platform review.

## Verification and build

```powershell
./atmungsraum/quest/boot-helper/test-policy.ps1 -JavaHome <JDK17>
python ./atmungsraum/quest/boot-helper/tests/test_android_boundary.py
./atmungsraum/quest/boot-helper/tests/test-build-closure.ps1
```

Production pure Java tests cover observed downloads beyond nine hours, actual
replacement, start versus focused-ready, Meta startup races, reboot recovery,
replay/cancel/disabled/signer/singleTask barriers and individual command expiry.
The Android-boundary test executes the actual provider/persistence/service/boot
receiver with inert Android seams. Compile actual sources against Android34
separately. Unit/boundary tests prove no physical launch or Meta install.

Only the coordinated build-slot owner may package. The builder requires
explicit SDK/JDK, existing external key, signed reference APK, fresh absolute
output outside Git and name of a privately set signing-password variable.
It never creates keys, records passwords or overwrites output:

```powershell
./atmungsraum/quest/boot-helper/build.ps1 -AndroidSdk <SDK> -JavaHome <JDK17> `
  -Keystore <existing-private-key> -ReferenceApk <signed-main-apk> `
  -OutputRoot <fresh-private-output> -SigningPasswordEnvironment <variable-name>
```

Uses SDK34/build-tools36.0.0, min29/target34, plain Java without runtime
dependencies or native libraries. Retains signed-manifest permission/provider/
service/capability checks, signature equality, DEX owner closure, alignment and
hash receipts privately. No install, device, account or upload action. Separate
helper Meta approval is not claimed; main APK omits helper overlay authority.
