# Atmungsraum Quest Developer connectivity and release protocol

This is the operator and agent checklist bundled in a WebRTC-enabled APK at
`assets/atmungsraum-webrtc/release-protocol.md`. The installed PWA at
<https://viscereality.org/atmungsraum/> is the operator interface.

## Phone connection

1. Open Atmungsraum Developer on the Quest 2 and allow the requested Android
   Nearby devices permissions. Open the installed Atmungsraum PWA on the phone.
   For the optional Android discovery companion, allow Nearby devices and
   the site's Apps on device permission. Allow Local network access when
   using the direct Wi-Fi route.
2. Bluetooth LE v2 advertises the installation ID, gateway and native epochs,
   and the available gateway routes. The discovery service and metadata
   characteristic use `9a7b2001-7d6a-4b7f-9d4a-6f7c0a020101` and
   `9a7b2001-7d6a-4b7f-9d4a-6f7c0a020103`. The companion scans all matching
   instances and serves the selected installation's data through IPv4 loopback
   port `8767` to `https://viscereality.org`. Discovery conveys no experiment
   authority. On a fresh phone, press **Mit Bluetooth verbinden**, select the
   intended Quest, compare the digits displayed on both devices, and press
   the two confirmation buttons. No typing or copying is required.
3. WebRTC is preferred. The Quest publishes an expiring offer through Bluetooth
   and the pinned VDO.Ninja SDK `1.5.5` carries data-only frames.
   ICE selects the viable path, including TURN relay where needed. The native
   gateway uses a closed v3 frame for the proof-bound first pairing exchange.
   Saved membership reconnects automatically. The gateway authenticates
   `RQG1-RTC1` requests bound to the installation, both epochs and the room.
   For separately commissioned pairing, the `ATMB1` exchange
   proves existing membership with a scoped HMAC and encrypts the ticket to the
   phone's ephemeral P-256 key. The Quest owns member revocation, the active
   controller lease, replay checks and native operation receipts.
4. When WebRTC is unavailable before first pairing starts, shared Wi-Fi can
   use the native LAN HTTP gateway on port `8766` for the same installation
   and epochs. The PWA requires fresh authenticated native status on either
   route before it enables controls. The connection flow requires no hotspot.

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
   Bluetooth-selected first join from an empty installed PWA with both
   button orders and no manual code entry, saved reconnect, direct Wi-Fi,
   and relay-only WebRTC. Verify the selected ICE candidate pair
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
