# Atmungsraum Quest Developer connectivity and release protocol

This is the operator and agent checklist bundled in a WebRTC-enabled APK at
`assets/atmungsraum-webrtc/release-protocol.md`. The installed PWA at
<https://viscereality.org/atmungsraum/> is the operator interface.

## Phone connection

1. Open Atmungsraum Developer on the Quest 2 and allow the requested Android
   Nearby devices permissions. Open the installed Atmungsraum PWA on the phone.
   For the Android discovery companion, allow Nearby devices and the site's
   Apps on device permission. Allow Local network access when using the direct
   Wi-Fi route. Use the existing managed profile for automatic first pairing.
2. Bluetooth LE v2 advertises the installation ID, gateway and native epochs,
   and the available gateway routes. The discovery service and metadata
   characteristic use `9a7b2001-7d6a-4b7f-9d4a-6f7c0a020101` and
   `9a7b2001-7d6a-4b7f-9d4a-6f7c0a020103`. The companion scans all matching
   instances and serves the selected installation's data through IPv4 loopback
   port `8767` to `https://viscereality.org`. Discovery conveys no experiment
   authority. For manual enrollment, use **Mit Bluetooth verbinden**, compare
   the six-digit code and confirm on both devices.
3. On shared Wi-Fi or the authorized phone hotspot, the PWA may use the native
   LAN HTTP gateway on port `8766` with the same installation and epochs.
   Bluetooth-assisted Wi-Fi setup for an enrolled phone uses the protected
   setup path. The PWA requires fresh authenticated native status before it
   adopts a discovered address or enables controls.
4. For WebRTC, the Quest publishes an expiring offer through Bluetooth after
   target admission. The pinned VDO.Ninja SDK `1.5.5` carries data-only frames.
   ICE selects the viable path, including TURN relay where needed. The native
   gateway authenticates `RQG1-RTC1` requests bound to the installation, both
   epochs and the room. For commissioned first pairing, the `ATMB1` exchange
   proves existing membership with a scoped HMAC and encrypts the ticket to the
   phone's ephemeral P-256 key. The Quest owns member revocation, the active
   controller lease, replay checks and native operation receipts.

Read the [pairing wire contract](https://viscereality.org/atmungsraum/for-ai/PHONE-PAIRING.md)
and the [Android companion setup](https://viscereality.org/atmungsraum/phone-discovery/README.md)
for exact message fields and commissioning steps.

## Agent release requirements

1. Use the current reviewed project sources, native overlay and pinned runtime
   assets. Resolve and assemble with `quest/assemble-apk.ps1 -CanonicalRelease`
   for `org.viscereality.atmungsraum.developer` and Meta app `1459506690581676`.
   Preserve the Developer signer and Quest 2 targeting. Check the active Meta
   build and choose an unused, increasing Android version code in the approved
   release sequence. Verify the signed APK's package, signer, SHA-256, native
   source closure and WebRTC asset bytes, including this file.
2. Verify that exact APK on the lab Quest 2 with the current installed PWA and
   matching signed Android companion. Record the three identities and test
   commissioned first join with an empty test pairing slot, saved reconnect,
   direct Wi-Fi, and relay-only WebRTC. Verify the selected ICE candidate pair
   and actual TURN relay protocol from WebRTC stats. Exercise TURN/TLS with UDP
   blocked by a verified network filter, signaling interruption and recovery,
   stale discovery, and wrong or revoked membership. Confirm fresh native status
   and match an actual command receipt after each successful join. Keep rejected
   peers without control. Restore test fixtures and saved operator state.
3. Run the affected native, phone and hardware checks, including the existing
   Polar, recording and Meta updater acceptance gates for the canonical release.
   Keep bench candidates in the lab. Respect the documented staged-update order.
   Record blockers, failing routes and missing physical evidence explicitly.
4. Synchronize every qualified Developer release to Viscereality's private
   Developer channel `1529361979246951`. Verify Meta processing and the active
   version name, version code, build ID and Quest 2 target. If the exact release
   is already active, record verification. Align the public `quest-release.json`
   latest version after Meta confirms it. Record the signer, digest, build ID,
   device/PWA evidence and delivery outcome in the APK log and project state.
   Verify the installed hash and Meta installer attribution for delivery checks.

Keep keystores, Meta credentials, Wi-Fi credentials, TURN credentials, room
secrets, membership keys and raw participant data outside published artifacts.
The [release workflow](https://viscereality.org/atmungsraum/for-ai/WORKFLOW.md#quest-apk-changes-and-releases)
and [project checklist](https://viscereality.org/atmungsraum/for-ai/PROJECT.md)
track the current release and evidence.
