# Atmungsraum Boot Helper 2

Version2 is a one-time USB helper update using the existing package and signer.
It retains the existing checkbox, overlay grant and `boot-helper` preferences.
The user's task13 amendment adds finite replacement launch assistance.

## Bounded replacement assistance

Only main metadata `org.viscereality.atmungsraum.boot_protocol=2` enables the boot
watch. Old main18 retains legacy one-shot boot behavior. The helper durably
binds an update watch to the current boot, canonical attempt, actual installed
code, expected strictly newer code and helper-owned elapsed-time expiry.
Watches last at most five minutes. Once a valid newer package is observed,
Meta gets20seconds to relaunch first. The helper then allows at most one
fallback total per watch, consumed durably before dispatch. Duplicate package
events, service recreation and failed dispatch cannot repeat it.

A trusted `launch-started` from the original installed code records it without
retiring an unconfirmed boot watch; `launch-ready` retires that watch. A PM-equal
replacement start at or above the bound target also retires it immediately,
so slow first-frame focus cannot cause a second launcher request. For an
armed update watch, started or ready may retire it only when loaded code equals
actual PackageManager code and satisfies the expected target. A replaced old
process cannot acknowledge the new binary. Protected kiosk escape cancels the
watch. Checkbox/permission loss, expiry or reboot stops it. A128-entry per-boot
attempt ledger denies at exhaustion without evicting replay protection; duplicate
exact arm calls do not renew expiry. No general foreground watchdog remains.

The fixed provider `org.viscereality.atmungsraum.boot.handoff` requires signature
permission `org.viscereality.atmungsraum.boot.permission.UPDATE_HANDOFF`, plus
Binder UID resolving exclusively to the exact main package and exact current
APK signer set. Main independently validates helper package/signer/protocol.
Shared UID callers and another same-signer package are rejected.

`ContentResolver.call` uses null arg and closed Bundle fields:

| Method | Fields besides Integer `v:1` |
| --- | --- |
| `arm-update` | String positive canonical decimal `attempt`; Long `installed_code`, `target_code` |
| `launch-started`, `launch-ready`, `cancel-watch` | Long immutable loaded-process `installed_code` |
| `status` | None |

Replies contain Integer `v:1`, Integer `protocol:2`, Boolean `accepted`, String
canonical `reason` and Boolean `watching`. No caller-provided expiry, package,
component, URL or path is accepted. Status is read-only.

A non-exported `specialUse` foreground service owns the finite watch,
low-importance notification, one-second checks and dynamic package receiver.
The BOOT receiver returns promptly. `START_STICKY` recreation reloads the same
durable boot/expiry/consumption; it does not renew the watch. Update handoff waits
for foreground service creation before a positive ACK; service-start failure
revokes the watch and denies handoff. Original boot queues service creation
without blocking its main looper. Force-stop and OS denial remain limitations.

Android34 eligibility uses the existing effective overlay grant for background
service/activity launch; real BOOT_COMPLETED also exempts the foreground-service
start. Foreground status alone is not activity-launch authority. The manifest
declares FOREGROUND_SERVICE, FOREGROUND_SERVICE_SPECIAL_USE and the special-use
subtype; no while-in-use sensor service is involved. POST_NOTIFICATIONS is not
required to start an FGS. Raising target SDK to35+ needs review because the
overlay exemption changes. See [Android FGS restrictions](https://developer.android.com/develop/background-work/services/fgs/restrictions-bg-start),
[service types](https://developer.android.com/develop/background-work/services/fgs/service-types#special-use),
[background activity starts](https://developer.android.com/guide/components/activities/background-starts)
and [notification permission](https://developer.android.com/develop/ui/views/notifications/notification-permission#exemptions).
Actual Quest scheduling and Meta replacement/relaunch remain hardware gates.

The policy tests execute production Java; `tests/test_android_boundary.py`
executes the real provider, persistence adapter and service against inert
Android seams. Compile all sources against actual Android34 separately. These
checks establish no APK/device/Meta installation or foreground success.

One-time, separately installed Android helper for the exact same-signer target
`org.viscereality.atmungsraum.developer`. Normal target updates do not require
helper updates. The helper never installs, downloads or updates another app.

Open **Atmungsraum Boot Helper**, enable **Enable launch after restart**, and
use **Open Android overlay settings** to grant this helper the normal overlay
permission. Both are required. This setup does not launch the target or count
as boot proof. No overlay window is created. The grant supplies Android's
background activity launch exemption; actual Quest behavior requires a real
restart test. Force-stopping the helper suppresses Android boot delivery until
it is opened again.

Only Android's protected `BOOT_COMPLETED` action reaches the non-exported boot
receiver. No custom/synthetic broadcast, alarm or network interface exists.
The bounded protocol2 service above owns its own lifetime. The receiver reads
real `Settings.Global.BOOT_COUNT`
and synchronously commits admission before any launch. A denied or failed
attempt consumes that boot, so exit/force-stop, permission changes, setup
changes and duplicate delivery cannot repeat its original boot launch. The
explicit bounded replacement contract above is the only fallback exception.
Missing/unreadable boot count fails closed. Existing state survives ordinary
helper updates; uninstall/clear-data resets setup and history.

Each boot queries the current enabled, exported `MAIN` / `LAUNCHER` activity of
the fixed package. It rejects other components and mismatched package signing
certificate sets using `SigningInfo.getApkContentsSigners`; dispatch rechecks the
signature and component. The explicit launch Intent contains only the normal
launcher action/category and task flags. No version, activity class or APK
location is pinned. The setup screen shows read-only status. `startActivity`
returning is reported as **requested, unconfirmed**, never successful foreground
or XR admission. Root/device validation supplies that proof separately.

Run the pure Java policy boundary test:

```powershell
./atmungsraum/quest/boot-helper/test-policy.ps1 -JavaHome <JDK17>
python ./atmungsraum/quest/boot-helper/tests/test_android_boundary.py
./atmungsraum/quest/boot-helper/tests/test-build-closure.ps1
```

Build only while holding the coordinated build slot. The builder requires
explicit SDK/JDK, an existing external developer keystore, same-signer reference
APK, a fresh absolute output directory whose parent is outside Git, and the
name of a privately set signing-password environment variable. It never creates
keys, records passwords or overwrites prior output:

```powershell
./atmungsraum/quest/boot-helper/build.ps1 -AndroidSdk <SDK> -JavaHome <JDK17> `
  -Keystore <existing-private-key> -ReferenceApk <signed-main-apk> `
  -OutputRoot <fresh-private-output> -SigningPasswordEnvironment <variable-name>
```

Uses SDK34/build-tools36.0.0, min29/target34, plain Java with no external runtime
dependencies or native libraries. Retains actual signed manifest/badging,
signature, DEX class closure and hash receipts privately. Only
`RECEIVE_BOOT_COMPLETED`, `SYSTEM_ALERT_WINDOW`, `FOREGROUND_SERVICE` and
`FOREGROUND_SERVICE_SPECIAL_USE` are used, plus the signature provider permission;
visibility is
limited to the exact target. No device/account/install or upload actions are
performed by the builder. Meta approval of the separate helper is not claimed;
the main APK must omit overlay authority and be distributed through Meta.
