# Atmungsraum Boot Helper

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
receiver. No custom/synthetic broadcast interface, alarm, service, watchdog or
network interface exists. The receiver reads real `Settings.Global.BOOT_COUNT`
and synchronously commits admission before any launch. A denied or failed
attempt consumes that boot, so exit/force-stop, permission changes, setup
changes and duplicate delivery cannot relaunch the target during that boot.
Missing/unreadable boot count fails closed. Existing state survives ordinary
helper updates; uninstall/clear-data resets setup and history.

Each boot queries the current enabled, exported `MAIN` / `LAUNCHER` activity of
the fixed package. It rejects other components and mismatched package signing
certificates using `PackageManager.checkSignatures`; dispatch rechecks the
signature and component. The explicit launch Intent contains only the normal
launcher action/category and task flags. No version, activity class or APK
location is pinned. The setup screen shows read-only status. `startActivity`
returning is reported as **requested, unconfirmed**, never successful foreground
or XR admission. Root/device validation supplies that proof separately.

Run the pure Java policy boundary test:

```powershell
./atmungsraum/quest/boot-helper/test-policy.ps1 -JavaHome <JDK17>
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

Uses SDK36/build-tools36.0.0, min29/target34, plain Java with no external runtime
dependencies or native libraries. Retains actual signed manifest/badging,
signature, DEX class closure and hash receipts privately. Only
`RECEIVE_BOOT_COMPLETED` and `SYSTEM_ALERT_WINDOW` are declared; visibility is
limited to the exact target. No device/account/install or upload actions are
performed by the builder. Meta approval of the separate helper is not claimed;
the main APK must omit overlay authority and be distributed through Meta.
