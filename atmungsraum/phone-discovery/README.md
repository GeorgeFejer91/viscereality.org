# Atmungsraum Bluetooth discovery for Android

This companion recovers a saved Quest's Wi-Fi address when the installed
Atmungsraum web app can no longer reach it at its previous address. It requires
Android 12 or later. The app still authenticates the Quest using the enrollment
already saved in the browser profile.

Install the signed APK and open **Atmungsraum Bluetooth** once. Grant Nearby
devices. The foreground service starts again at boot and only scans on demand.
At the first loopback request Chrome may ask to let viscereality.org access
services on the phone. Allow this for the companion. Close any floating
overlays if Chrome blocks the permission tap.

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
