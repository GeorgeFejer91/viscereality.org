# Hotel Wi-Fi handoff for Quest and Pixel

## Current blocker — 2026-11-19

The installed Pixel PWA cannot reach its public entry page. Both lab devices
also fail to reach the WebRTC signaling host. These failures occur before the
latest PWA and the WebRTC bench can attempt authenticated pairing together.
Resume the [active validation checklist](PROJECT.md) after restoring these
routes.

| Check | Observed result |
| --- | --- |
| Workstation → live Atmungsraum | HTTP 200 and entry marker `20261119-idle-refresh`. The workstation could also open TCP to `wss.vdo.ninja:443`. |
| Pixel 10 → `viscereality.org:443` | IPv4 TCP failed with `No route to host`. A cold launch from the installed icon and a RustDesk MCP Reload tap both showed Chrome's `ERR_ADDRESS_UNREACHABLE`. |
| Pixel 10 → `wss.vdo.ninja:443` | IPv4 TCP failed with `No route to host`, although Android marked the active Wi-Fi network validated. |
| Quest 2 → `wss.vdo.ninja:443` | IPv4 TCP failed with `No route to host` while the Quest was awake. Android marked its active Wi-Fi captive. |

The cause of the hotel's forwarding failure is undetermined. Direct LAN
reachability is a separate check. The old PWA document had a saved pairing
before the cold launch. The launch test left site data alone. Follow
[RustDesk phone testing](RUSTDESK-PHONE-TESTING.md#run-and-evaluate-cases)
for the frame-confirmed launcher and Reload procedure. The private lab evidence
directory `Atmungsraum-Validation/button-bluetooth-wifi-validation` contains
`pixel-cold-launch-egress.json` and the two phone frames. Keep device
identifiers, addresses, credentials, and raw captures outside Git.

## Release and evidence boundaries

- The [live PWA](https://viscereality.org/atmungsraum/) serves
  `20261119-idle-refresh`, including the button-only pairing path and the idle
  refresh repair. [Runtime Pages deployment](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44427512855).
- Meta Developer code 27, `0.2.26-dev`, is the installed baseline for the
  guarded study and update sequence. Use [WebRTC bench 37](../quest/apks/README.md#webrtc-bench-37-with-button-only-pairing)
  for the new pairing test. Its installed digest was checked, and its
  button-only physical pairing and reconnect still need qualification.
  [Developer code 30](../quest/apks/README.md#canonical-developer-30-candidate)
  is the single signed canonical candidate in private staging. Code 28 is
  reserved for the guarded Meta update test.
- The hotel network supported TURN on November 18 in the commissioned lab
  profile with bench 32. Thirty fresh-slot joins and thirty saved rejoins
  passed, and a separate relay-only recovery returned an original native
  Recenter receipt. That run used a provisioned identity. The independent
  installed-PWA test with an empty pairing slot and both real confirmation
  buttons remains open on the current release.
  [Earlier hotel trial evidence](VERIFICATION.md#hotel-wi-fi-timing-and-recovery-on-the-repaired-scanner--2026-11-18).
- The current button-only path passed host browser/Java interop and mounted UI
  tests. The separately tested PWA refresh fix queues a release check when
  saved-headset discovery overlaps its timer. Physical transport on the
  current hotel connection remains unverified.
  [Current software evidence](VERIFICATION.md#button-only-webrtc-pairing-host-verification--2026-11-19).

Bluetooth discovery supplies the selected installation identity and expiring
WebRTC offer. Both clients need reachability to the signaling service and a
viable ICE path. TURN is the relay path when peer-to-peer routing is unusable.
The existing shared-LAN fallback uses the authenticated native gateway on port
`8766`. An authenticated native status response and the controller lease gate
experiment controls on either route. Follow the
[pairing contract](PHONE-PAIRING.md) and
[protocol bundled in the APK](../quest/webrtc/release-protocol.md).

## Resume the physical run

1. Check the current GitHub Pages deployment, live entry marker, installed APK
   hashes and Meta offer. Confirm the Quest is awake using the
   [awake workflow](QUEST-AWAKE-TESTING.md). Clear any attended Wi-Fi sign-in
   and restore usable internet on an approved non-hotspot connection. Track
   the network used for each trial. Hotel Wi-Fi qualification requires a
   physical pairing and reconnect on the hotel network.
2. Use exact lab USB selectors from `adb devices -l` to check the three
   routes below. Record each exit code and compare it with a fresh capture
   of the installed PWA. A successful TCP check is a preflight. Check the
   actual signaling and selected ICE candidate pair in the run.

   ```powershell
   adb -s $PixelSerial shell 'toybox nc -4 -w 4 -q 1 viscereality.org 443 </dev/null'
   adb -s $PixelSerial shell 'toybox nc -4 -w 4 -q 1 wss.vdo.ninja 443 </dev/null'
   adb -s $QuestSerial shell 'toybox nc -4 -w 4 -q 1 wss.vdo.ninja 443 </dev/null'
   ```

3. Use RustDesk MCP to reload the installed Pixel PWA. Verify standalone
   mode and the current loaded entry and connection modules. Preserve the
   existing pairing. Use the scoped test slot from the
   [independent-installation matrix](VERIFICATION.md#independent-headset-and-pwa-installations)
   for the fresh-pairing case.
4. On the isolated WebRTC bench, verify the Bluetooth offer belongs to the
   intended headset installation. Select it, compare the digits displayed on
   both devices, and press both confirmation buttons. No code entry or
   copying is required. Verify authenticated native status and an
   original Recenter receipt. Repeat saved reconnect in both startup orders.
   Record the selected ICE pair and relay protocol for the TURN case. If a
   step fails, report the earliest failed gate and preserve the original
   receipt or error for diagnosis.
5. Continue the real Touch, tracking, worn-H10, recording, recovery, and
   custody checks in the [active project checklist](PROJECT.md). Coordinate
   the [guarded Meta update](BOOT-KIOSK-UPDATE.md#queued-meta-update-amendment--2026-11-17)
   with the release owner after the verified baseline recording gate.
