# Atmungsraum Bluetooth discovery for Android

This companion supplies the saved Quest's current Wi-Fi address when the
installed Atmungsraum web app cannot reopen its saved browser Bluetooth grant.
The PWA starts discovery automatically after checking for an update; it can
still use the previous Wi-Fi address while discovery is unavailable or
searching. The companion requires Android 12 or later. The PWA authenticates
the Quest using the enrollment already saved in the browser profile.

Install the signed APK and open **Atmungsraum Bluetooth** once. Grant Nearby
devices. The foreground service starts again at boot and only scans on demand.
During setup, allow the site to reach the companion. In Chrome 157 on
Android, open viscereality.org in a regular tab and choose **⋮ → Site
controls → Permissions → Apps on device**. This site permission applies
to the installed PWA. Chrome lists **Local network** separately for
connections to other devices. Close interfering overlays if Chrome
blocks the permission tap.

The service listens only on IPv4 loopback port 8767. Its GET path contains the
saved Quest device identity, and requests require the exact Atmungsraum web
origin. It returns only radio status or the matching headset's public
Bluetooth discovery metadata. Pairing keys, participant data and commands
stay in the existing Wi-Fi gateway. The browser checks the serial, stable
device ID and signed gateway status before adopting a new address.

## Build

Use PowerShell, JDK 17, Android SDK Platform 35 and Build Tools 35.0.0.
Set JAVA_HOME, ANDROID_HOME and ATMUNGSRAUM_DISCOVERY_STOREPASS, then run:

```powershell
./build.ps1 -Output <new-build-directory> -Keystore <signing-keystore>
```

Install with the intended phone's serial and retain the same signer for
updates. This companion supports rediscovery on a reachable Wi-Fi network.
The Quest's existing hotspot provisioning remains a separate path.

## Managed first pairing

Phone discovery version 0.2.0 supports one commissioned Quest through
Android managed app configuration. The Android administrator sets
`atmungsraum_quest_device` to the authorized Quest installation ID and
`atmungsraum_bootstrap_key` to the lowercase hex digest of
`HMAC-SHA256(member_key, "ATMB1|credential|device|controller")`.
The member key is its UTF-8 lowercase hex text. Supply these values
through the authorized device policy and keep them out of logs and
public artifacts. Provision the corresponding membership on the Quest
as part of commissioning. The device and browser permission grants are
part of initial setup.

The proof broker runs on the commissioned Android profile and accepts
the exact Atmungsraum origin on loopback. Device policy owns membership
of that Android profile and the apps installed in it. The companion
returns the commissioned identity and time-bounded proof messages.
The browser receives the controller key only inside a Quest-issued
ticket encrypted to its ephemeral P-256 public key and verified by
the companion. The Bluetooth metadata and WebRTC room carry no
controller key. Remove the Quest membership and managed phone policy
to revoke the phone.

Opening the PWA after commissioning uses this route when saved pairing
is absent. It saves the membership only after a fresh signed native
status. Normal first launch and reconnect need no pairing code, chooser,
headset approval tap or hotspot.

[Chrome's loopback permission guide](https://developer.chrome.com/docs/modern-web-guidance/guides/security/local-network-access)
