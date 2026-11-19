# Hotel Wi-Fi handoff for Quest and Pixel

## Hotel Wi-Fi failure — 2026-11-19

During the initial test, the installed Pixel PWA could not reach its public
entry page, and both lab devices failed to reach the WebRTC signaling host.
The Pixel [recovered after hotel sign-in](#pixel-recovery-and-reinstall).
Recheck the Quest's portal state and egress before continuing the
[active validation checklist](PROJECT.md).

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

## Pixel recovery and reinstall

Rejoining the same hotel Wi-Fi made Android show **Sign in to a Wi-Fi
network**. With the operator's approval, the RustDesk session completed the
portal sign-in. Android then marked the network validated, and IPv4 TCP to
both the PWA host and the signaling host on port 443 succeeded from the Pixel.
The existing standalone app loaded the current release after a Reload tap.

Bringing a stale Chrome tab forward then advanced its entry from
`20261118-idle-release` to `20261119-idle-refresh` without a Reload tap. The
RustDesk uninstall and Chrome install prompts produced a new Android WebAPK.
A RustDesk launcher tap opened the new installation in standalone mode with
entry `20261119-idle-refresh` and `quest-ui.js?v=20261119-bluetooth-buttons`.
DevTools confirmed the saved pairing was unchanged. The private lab evidence
directory contains the `pixel-reinstall-*` frames and version readbacks.
The Quest's post-recovery egress and authenticated pairing remain open.

## Current USB validation

Subsequent background-music check: the USB Quest accepted the same-signer
[bench 42](../quest/apks/README.md#background-music-bench-42--2026-11-19)
update with exact installed bytes and the existing bench UID. Developer 27's
APK and process remained available. The Quest still reports no route to the
public PWA and signaling hosts. Playback qualification requires a working
authenticated phone connection.

On November 19, the Quest on this PC matched Meta-installed Developer 27
and signed WebRTC bench 37 by package, version and installed APK hash.
Developer 32 is active on the Meta channel. Use the
[earlier delivery record](PROJECT.md#active-publish-meta32-and-test-the-pwa-update--2026-11-19)
for the prior observed installation of code 32. Verify the installed
package separately on each selected USB target.

The existing watchdog kept this Quest awake with the display on. Android
reported captive Wi-Fi. TCP 443 from the Quest to the PWA and VDO.Ninja failed.
The Pixel reached both public hosts. Its connection to the Quest's listening
LAN gateway on port 8766 failed.

RustDesk verified Settings and Back, the real Android Bluetooth chooser,
selection of the Quest 2, and the on-screen network failure with experiment
controls disabled. A separate launcher start returned to the standing
Bluetooth search on the latest PWA. Chrome DevTools confirmed standalone
mode, `20261119-idle-refresh`, the expected connection UI module and unchanged
saved pairing. Authenticated controls remained unavailable. At `b48a21d`,
the 421-state GUI gate and all 361 web and assembly tests passed with zero
skips using the recorded button-pairing native test tree. The private evidence
directory contains the `pixel-current-validation-*` frames and result logs.

Complete the headset's hotel sign-in and recheck Quest signaling reachability.
Use the regular Meta delivery path and read back the exact installed APK
before qualifying code 32. Continue with fresh pairing, a matched native
receipt, reconnect, and the remaining hardware and recording gates below.

## Release and evidence boundaries

- The [live PWA](https://viscereality.org/atmungsraum/) serves
  `20261119-idle-refresh`, including the button-only pairing path and the idle
  refresh repair. [Runtime Pages deployment](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/44427512855).
- Meta Developer publishes [code 32](../quest/apks/README.md#meta-developer-32-publication--2026-11-19)
  as `0.2.31-dev`. The current USB target retains Meta Developer code 27,
  `0.2.26-dev`, and verified [WebRTC bench 37](../quest/apks/README.md#webrtc-bench-37-with-button-only-pairing).
  Button-only fresh pairing and saved reconnect on the current hotel setup
  still need qualification. Code 28 remains unpublished under the later
  code 32 publication decision.
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
   any further [guarded Meta update](BOOT-KIOSK-UPDATE.md) with the release
   owner and follow the current version and acceptance sequence.
