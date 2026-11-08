# Verification

## Encrypted participant Arm — 2026-11-08

The Java gateway suite and enrolled LAN harness verified authenticated
decryption, guest and relay denial, stale-generation and absent-ECG
rejection, once-only dispatch, and a receipt without the participant ID.
Chrome fixtures reused the participant form, checked the normalized ID and
disabled Start before it was set, decrypted the bounded private Arm on the
fixture server, verified the original participant CSV, and repeated after a
browser and simulated Quest restart. Phone portrait, short landscape and
200% text reflow passed locally and against Pages run
[42491480387](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42491480387).
All nine changed live files matched the browser release. After integrating
the concurrent folder-sync release, 71 Node tests and the UI and scenario
checks passed.

Signed Quest APK `0.1.9` passed the exact run-capsule check and same-signer
upgrade. USB Package Manager and installed SHA-256 matched the built APK.
The headset was still asleep and both Nearby devices permissions were
ungranted at readback. Live H10, Quest CSV, and physical Android phone
verification remain open.

## Startup unlock and two-action folder — 2026-11-08

Chrome's synthetic browser fixture checks initial focus, incorrect password,
opt-in browser storage, silent restoration after reload, cancellation and
recovery, and repeated sends without another prompt. Both send actions
decrypt to the expected ZIP contents. Rejected uploads and Discord posts
leave the receipt ledger pending. The fixture exercises concurrent tabs,
five desktop/portrait/landscape layouts and 200% text and spacing at 320px.
The separate local Node adapter test sends two encrypted chunks to real
Wormhole/Backblaze and restores the exact 9,800,146-byte synthetic ZIP. Discord
posts in the browser fixture are intercepted. Hosted activation remains
open in [PROJECT.md](PROJECT.md#one-startup-unlock-for-folder-sync--2026-11-08).

## Participant CSV writer and idle pull — 2026-11-08

The Quest writer and export host checks passed 66 focused tests. Synthetic
native records verified reservation across reopened and abandoned attempts,
Unicode participant filenames, actual ECG values and sensor timestamps,
blank-voltage gap rows, and a hash-bound descriptor over the finalized CSV.
The Java LAN gateway harness passed 32 checks, including the participant
filename and traversal denial. Neither check installs or runs the Android APK.

The Chrome fixture ran the encrypted idle sync at 390×844, 568×320, and
320×568 with 200% text and custom spacing. It rejected a corrupt participant
CSV, then downloaded only the verified CSV bytes with the Quest filename.
It reopened that exact file after a full Chrome restart and continued to
reconcile legacy ZIPs and runs finished while the browser was absent. The 69
Node tests and UI checks also passed. The Quest was still asleep at USB
preflight. The operator form, encrypted Arm, and real H10 and phone acceptance
remain open.

## Idle Quest sync — 2026-11-08

The signed encrypted Chrome fixture joined a Quest with an existing finalized
recording, automatically saved it in IndexedDB, and ran Arm → Start → Pause →
Resume → Finish. Its gateway rejected recording traffic outside idle. It then
served the two saved runs across separate pages, corrupted one transfer to
exercise hash rejection, and verified the retried ZIP and its original bytes.
After a browser reload and rejoin the archive selector offered both stored
runs without reading their bytes again from Quest. The downloaded ZIP matched
the native test bytes. All three layouts passed (390×844, 568×320, and
320×568 at 200% text with text-spacing overrides). The full 69 Node tests, UI
checks and six scenario consistency checks passed.

The follow-up restart fixture used a persistent Chrome profile and fully
closed the browser. It rejoined the same simulated Quest under new gateway
and native epochs, reopened both saved ZIPs without reading their bytes
from Quest, completed a second rehearsal and synced only that new archive.
All three layouts passed against both the local and deployed app. Runtime
commit `9398177` deployed successfully in
[Pages run 42487461920](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42487461920).
The live entry and five changed JS/CSS assets matched the exact commit bytes.

This fixture used synthetic data. The USB-connected Quest was asleep at
device preflight. Physical H10, participant CSV, and Android phone storage
verification remain open.

## Remote-only Quest data path — 2026-11-08

The new composition passed feature resolution for the original native host,
Remote-only startup and recording export. The native host tests covered
Remote-only control admission, guest denial for private data, and the maximum
encrypted recording chunk. The browser tests verified the encrypted read,
stale and reordered ECG rejection, gap rendering and corrupt recording bytes.
Chrome fixture runs at phone, short landscape and enlarged text sizes used
the actual UI to select condition B, receive changing ECG, Start, Pause,
Resume and Finish. Each run downloaded and checked the original ZIP data,
then reloaded, rejoined and retrieved the same native run again. All three
cases repeated on the deployed website.

The signed 0.1.7 rehearsal includes the ARM64 OpenXR loader and original
native host. Quest 3 installed the exact APK bytes and reported Remote from
its mode owner without operator selection. Desktop Chrome used the live
Atmungsraum page to join the identified Quest and received a valid encrypted
ECG preview reply over Wi-Fi. Android's Nearby devices prompt was active,
so the reply contained no live H10 samples. Actual sensor acquisition, native
scenario completion, phone hotspot discovery and phone storage remain open.

## Browser controls for Quest-owned Polar setup — 2026-11-07

The signed LAN gateway fixture verified missing Android permission, keyboard
activation, exactly one scan/select/connect/start-ECG sequence and native
readiness in Chrome. It then ran the synthetic Arm → Start → Pause → Resume
→ Finish route. The same five Chrome viewport/text settings passed with a
gateway that omitted Polar setup, including 320 CSS px, short landscape,
desktop and 200% text with custom spacing. Both fixtures measured visible
text and checked control heights, overflow, page errors and demo asset
isolation. Unit tests cover invalid setup grants, ambiguous sensor discovery,
duplicate actions, lost receipts and foreground exit. All 65 Node tests and
six scenario consistency checks passed.

The separate synthetic save sweep downloaded and reopened five exact session
CSVs and verified automatic completion of the 68-second Anleitung. The live
121-case Chrome layout sweep and the renderer-crash and quota recovery
fixtures passed for `2aeb0f1`. Delaying the Quest controller import reproduced
a visible join form before listeners were attached. The form now appears only
after its controls are ready. Quest 3
reported both Bluetooth runtime permission grants for the installed rehearsal.
Physical browser-to-Quest control and live H10 ECG require the attended Remote
gate and a separate device run.

## UI and CSV saving sweep — 2026-11-07

The sweep began on Pages run `42406261765` (`53a7633`). Chrome checked 121
layouts spanning patient entry, ECG, round selection, all five playback-ready
screens, playing and paused states, and saved CSVs. The populated CSV list
overflowed at 320 CSS px with 200% text and custom spacing. The filename now
wraps within the grid. The local matrix repeated with zero layout failures
or page errors. A separate probe passed with Pretext unavailable.

Using only synthetic inputs, Chrome downloaded and reopened matching CSV bytes
for all five session choices. The sweep checked scenario hashes, sample order,
pause-aware timing, separate run identities, the saved-files count and dialog
focus return. A full 68-second Anleitung completed and saved automatically.
A forced renderer crash recovered one aborted CSV from the browser checkpoint
with ECG requests blocked. A simulated IndexedDB quota failure showed the
warning and preserved both immediate and manual downloads.

The live browser fixtures passed the 30-second Polar loss rule, 115 PWA
control cases, five signed Quest UI cases, and encrypted sharing on five
layouts. The sharing fixture checked archive bytes, failures and retries,
new/all selection, reload and cross-tab receipts. It intercepted eight
synthetic uploads and seven notification attempts. Brave checked install
help and installed-state UI in isolated profiles. The 61 Node tests,
UI/context checks, six scenario pairs and 20 sample audio profiles passed.
Chrome also verified exclusive voice-sample playback and exact MP3 and JSON
downloads. Offline rendering matched the expected PCM for six scenarios at
44.1 and 48 kHz.

Pages run [42408436234](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42408436234)
deployed `4309c6f` from `main`. Both entry URLs returned HTTP 200 and all four
changed public files matched the commit. The live Chrome 156 matrix repeated
all 121 layouts with zero failures or page errors, and the live renderer
crash and quota fixtures passed.

Physical Android/Quest acceptance and external delivery are separate checks.
The browser probes and synthetic artifacts remain in ignored `output/playwright/`.

## Brave desktop install and taskbar pins — 2026-11-07

Brave 157 parsed the live manifest and installed the app into an isolated
browser profile, creating the Atmungsraum ICO and manifest icon resources. The
profile still shared the Windows taskbar. After uninstall, a taskbar pin retained
the same app ID and referenced its removed ICO in the QA profile. The native
Windows unpin action removed that exact link. A subsequent shortcut scan
returned zero Atmungsraum links. Future native installation QA requires an
isolated OS account or VM.

The original manifest returned an empty related-app list in a regular tab
after installation. A self relation in the manifest returned the exact installed
app ID and manifest URL in Brave. The regression for suppression across a
fresh tab initially failed and now passes. All 54 Node tests and the UI and
context checks pass. The safe Brave probe uses the observed positive response
as a fixture to verify hiding after reload. A fresh Brave profile verifies
manifest parsing, installability, icon decoding and the manual help at
1280×800, 390×844 and 320×568 with 200% text in the smallest case. All three
dialogs open at the top, have no horizontal overflow and restore focus on
Escape. The probe invoked zero native install actions and produced zero page
errors. A fresh taskbar pin in the regular Brave profile is a manual check.
Receipts and the probe remain in ignored `output/playwright/`.

Pages run `42403873448` deployed `6528b47`. The live file hashes matched the
commit, and both entry URLs resolved to `/atmungsraum/`. The live Brave probe
repeated the layout and reload checks with zero page errors or native installs.
A final scan returned zero Atmungsraum taskbar shortcuts.

## Quest rehearsal browser and native preflight

The rehearsal is entered through **VR-Probe** on the patient screen or
`/atmungsraum/?mode=quest#quest`. The browser requires a selected Quest
serial and a packaging fingerprint from the isolated rehearsal build.
Use its exact Remote mode and a trusted LAN. The phone uses the signed
experiment gateway, and the rehearsal mode does not initialize browser
ECG, audio playback or participant exports.

The 61-test Node suite includes the BLE identity read and cleanup, private gateway
and serial admission, the signed status and expected build, the complete
Arm/Start/Pause/Resume/Finish sequence, duplicate input, foreground cancellation
during signing, and unknown results.
Chrome passed manual-address, measurement and signed-fixture lifecycle checks
at 320×568, 390×844, 568×320 and 1280×720, plus 320 CSS px with 200% text and
text-spacing overrides. These checks exercise the real browser UI with a
fixture. Phone LAN permission, physical BLE and wearer presentation require
the installed-Android acceptance run. The exact
[Pages run for 994c0b8](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42405189848)
completed successfully. All nine changed files matched their live bytes and
the five signed-fixture Chrome cases passed again from the live origin.

The native gateway host suite passed production Remote discovery, enrollment
and receipt checks. The private build passed its exact feature lock, synthetic
asset provider, selected Java isolation and v3 signature verification. The
signed APK manifest contains the isolated package and exact site origin.
Quest 3 installed package/version readback matched. Older broad native tests
still flag their historical panel inventory and source assertions; the focused
gateway tests and the actual selected APK closure passed. Wearer presentation,
physical Android control and Showcase visual equivalence remain open.

## One checkout receipt per confirmed send — 2026-11-07

Both targeted tests failed before the separate receipt reader and writer were
added. All 52 Node tests and the UI and context checks pass. The local browser
fixture observes six checkout receipts across eight fixture uploads and seven
fixture posts. It verifies that failures leave no receipt, resends append a
receipt, earlier receipts remain intact, legacy history still suppresses
duplicates, and an open second tab updates after a confirmed send.
The same fixture passed against the published Pages release with five
layouts and no page errors. It used synthetic data and simulated provider
responses.

## Folder delivery and JSON receipts — 2026-11-07

The receipt and URL-only payload tests failed first, then passed. All 52 Node
tests and UI/context checks pass. The Chromium fixture checks five layouts
and 200% text reflow, verifies progress/focus, decrypts the uploaded ZIP to
check exact CSV selection and bytes, and validates the complete URL-only Discord
payload. It covers wrong passwords, aborted prompts, rejected upload/post,
reload, forced resends, a concurrent-tab lock, a file added during upload, a
revised CSV under the same session UUID and a file saved in another tab.
The local run observed seven fixture uploads, six fixture posts and no page
errors. No participant data or real Discord message was sent. The actual
Render deployment remains a separate operator step. See the ignored
`output/playwright/check-file-delivery.mjs` fixture and PROJECT.md release state.
The same browser suite passes on published Pages and verifies that the app
asks the credential manager to save only after an opted-in successful send.
The public ciphertext matches the configured destination after local
decryption; the check does not send a Discord message.

## Render Free upload adapter — 2026-11-07

The targeted tests failed first with the adapter/host support absent, then passed.
Real Node HTTP checks cover health, preflight, incorrect origin/key, `/notify`
blocked, unapproved storage host, over-limit body and exact 8 MiB forwarding.
Browser client tests cover a startup page, successful readiness and cancellation.
All 51 Node tests and UI/context checks pass. Five viewport browser fixture
checks (including enlarged text) exercise Render wake-up, cancellation, upload
failure, QR creation and the direct Discord prompt. Review found no blockers.

The local Node adapter made real calls to Wormhole and Backblaze. A synthetic
9,800,146-byte ZIP uploaded in two encrypted pieces and decrypted on download
byte-for-byte (SHA-256 `2ee9b24329c496fb94e8291b099b2df966fe86211f4c6f00375ad9b81ebd8eb2`).
No participant data or Discord test messages were sent. The actual Render
platform and its idle wake-up still require a user-owned deployment. See
`output/playwright/render-upload-local.json` and `render-sharing-results.json`
(ignored); release and deployment state are in PROJECT.md.

## Larger phone controls and install action — 2026-11-07

The two top surfaces/artwork double the base size at widths through 480px or
coarse-pointer widths through 1024px (112px versus the 56px folder on ordinary
phones). Their equal sizes, safe-area anchors and existing semantics remain.
Content reserves their enlarged clearance. A text install action sits beside
the folder in portrait; short landscape uses a named install icon above it,
outside the central content column. An actual ResizeObserver reserves the dock
height when text enlarges/wraps. Every visible new label uses the existing real
Pretext/DOM measurement owner; the label's box has room for its safety margin.

Four new Node checks first fail with the missing module, then pass. They cover
user-triggered prompting, one-use events/double-click exclusion, dismissal,
prompt rejection, retryable manual help, install consent versus actual
installation, standalone hiding, Escape propagation and focus restoration.
All 32 tracked Node tests, UI-state/context and staged whitespace checks pass.
The release excludes concurrent unfinished automated-sharing code and the
private phone attachment. No participant-data writer or audio asset changes.

Chrome passes 113 rendered cases across 320×568, 360×740, 390×844, 432×814,
480×640, 481×640, 568×320, 844×390, 844×480, 844×481 and 768×1024, using the
staged entry/CSS and released app/data modules. Patient, ECG, choices, all five
player routes and data recovery have no horizontal overflow/clipped labels or
unmeasured visible text. Bounded cases have no page scrolling or content/corner
overlap. Small/enlarged no-fit states use readable document reflow; keyboard
scrolling exposes each patient field/action clear of fixed controls.

The help dialog passes four phone sizes, native keyboard dismissal and focus
return. 200% type/WCAG-style spacing pass portrait and short landscape; a long
install label reflows at 320px. Closing help or saved files with Escape during
real local playback keeps the session active. A simulated browser install
event verifies click-only prompting, double-click exclusion, dismissal and
later help; `appinstalled` hides only the install action, leaving the folder.
These simulations test UI lifecycle, not an actual phone installation receipt.
Zero page errors occur. Phone screenshot inspected. Ignored evidence:
`output/playwright/check-pwa-controls.mjs`, `pwa-controls-results.json` and
`pwa-controls-phone.png`. Exact deployment/live evidence follows in PROJECT.md.

An additional opening-position regression first fails: at320×568 with200%
text/spacing, native autofocus on the bottom close button scrolls help to1832px
and hides its beginning. Close now sits in a measured header above instructions;
the same case opens at scrollTop0. All113 cases re-pass with new initial-scroll
assertions. Three desktop smoke sizes retain the original corner sizing and
usable help. The regression lives at `output/playwright/check-install-scroll.mjs`.
The containing sharing release passes all45 Node tests. No new installer logic
or participant-data behavior is changed by this header correction.

Live release `5ff34d3f5ceaed7e15ab306d6df32b5d6db84e16` completes exact
Pages run 42310953670 and deployment API status `succeed`. All12 affected paths
match Git bytes. The public origin passes all113 phone cases, including help
opening at scrollTop0 with enlarged text, plus three desktop smoke sizes and
error-free manifest/icon/installability checks. An isolated Chrome desktop
profile installs and launches the actual live app in standalone mode with that
user display preference explicitly set. No page errors occur. Evidence stays
ignored in `pwa-live-5ff34d3.json`, `pwa-controls-live-results.json`,
`pwa-controls-live-phone.png` and `pwa-controls-installed-live-results.json`.
Physical-phone installation/relaunch and assistive technology remain untested.

A stricter viewport-centering regression then fails at390×844: the two-top-icon
clearance moves the bounded round menu down27.99px. Equal vertical reserves,
using the maximum of top size + top safe inset and dock height + bottom safe
inset, restore its center to -0.008px. The47/34px unequal-inset probe first fails
at +6.49px, then independently passes at -0.008px. All115 phone cases re-pass with centered ECG/round/option/player
targets required whenever bounded. Enlarged/no-fit states retain readable
document reflow. This correction changes CSS and cache tokens only; its checks
exclude concurrent Discord-password work. Ignored regression:
`output/playwright/check-pwa-centering.mjs`; matrix receipt:
`output/playwright/pwa-controls-centered-results.json`.

The centered runtime `0320d1c1231396caf458a089ae7e60115b781aab` completes exact
Pages run 42313937741 with backend `succeed`. All12 affected paths match the
published Git bytes; the public origin passes all115 cases including simulated
unequal insets and zero opening scroll with enlarged text. No page errors.
Ignored live evidence: `pwa-live-0320d1c.json`,
`pwa-controls-centered-live-results.json` and `pwa-controls-centered-live-phone.png`.
No manifest, installation logic or participant-data changes accompany this
centering correction. Physical phone acceptance remains distinct from this QA.

The current [BeforeInstallPromptEvent API](https://developer.mozilla.org/en-US/docs/Web/API/BeforeInstallPromptEvent)
and [official installation-prompt guide](https://web.dev/learn/pwa/installation-prompt/)
were read. This API is not universal and each supplied prompt is single-use;
unsupported browsers therefore retain explicit manual instructions. Physical
Android/iPhone installation/relaunch acceptance remains open.

## Smartphone PWA installation — 2026-11-06

The primary delivery target is an installable smartphone PWA. The entry page
links a local manifest with stable `/atmungsraum/` ID/start/scope, standalone
display, black theme/background and the existing Viscereality logo. Native
browser installation is the home-screen launch path. The app remains online;
no service worker, offline audio cache or background session behavior is added.

The four opaque PNGs derive from `Images/logo.png` at Git blob
`bcd94ca74f6305eb7cd2e8a9a8b4a66cf99f6caa`. Ordinary 192/512px and Apple 180px
variants contain the unchanged aspect ratio on black at 86% maximum height.
The separate 512px maskable icon uses 66%; a pixel audit confirms every nonblack
pixel is inside the central 80%-diameter safe circle. Source and rendered icon
were visually inspected; no logo redesign or image-generation service is used.

All 28 tracked Node tests, UI-state/context and staged whitespace checks pass.
Chrome 156 (Windows, isolated persistent profile) parses the manifest without
errors, reports `installabilityErrors: []`, decodes each declared icon, and
keeps the entry screen readable at 390×844, 320×568, 844×390 and 568×320. No
horizontal overflow, clipped patient labels or page errors occur. Browser QA
serves the exact staged entry HTML and released JavaScript/CSS/data module,
excluding concurrent unfinished sharing changes. Independent review finds no
actionable issues.

CDP installation/launch opens the canonical URL and initial `#patient` route.
With the browser user display preference explicitly set to standalone (the
equivalent of Open as app/window), actual `(display-mode: standalone)` is true.
CDP's initial default preference opened a browser tab; the test records that
distinction rather than claiming a physical-phone default was exercised. The
test uninstalls its temporary app. Android Chrome/iPhone Safari installation,
relaunch, separate installed-context storage, safe areas, lock/background and
real H10/headset qualification remain open. Existing session-abort and browser
storage rules are unchanged.

Ignored local evidence: `output/playwright/build-pwa-icons.mjs`, `check-pwa.mjs`,
`pwa-local-results.json`, `pwa-local-phone.png` and temporary browser profiles.
Publication evidence belongs to PROJECT.md.

Live verification: release `02217feda1df9d99de830e50867ca32ce94b271c` completes
exact Pages run 42260860687; the deployment API reports `succeed`. All 13
changed public files match Git bytes, with `application/manifest+json` for the
manifest and `image/png` for icons. Slash/no-slash URLs resolve to the canonical
app and return 200. Chrome repeats error-free parsing/installability, icon
decoding, four phone viewport checks and actual standalone launch on the live
origin, with its user window preference explicitly set. The final target-phone
installation/relaunch question is pending; desktop results do not close it.
Ignored live receipts: `pwa-live-02217fe.json`, `pwa-live-results.json` and
`pwa-live-phone.png`. No runtime file changes accompany this evidence update.

Primary platform references read: Chrome's [install-criteria update](https://developer.chrome.com/blog/update-install-criteria)
permits menu installation without a service-worker fetch handler; the current
[MDN installability guide](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Guides/Making_PWAs_installable)
confirms required manifest fields and that service workers are optional for
installation. [WebKit's Safari 26 release notes](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/#every-site-can-be-a-web-app-on-ios-and-ipados)
describe home-screen web apps and the user's Open as Web App choice.
This checks the common contrary assumption that installation requires adding an
empty worker; offline functionality is a separate feature. Browser support and
device installation are distinct from manifest validity.

## Polar signal-loss timeout — 2026-11-06

Two new regressions first fail against the previous runtime: a delayed first
poll retries rather than expiring, and ECG recovery after the deadline revives
the run. They pass after the shared-state fix, along with an exactly-30-second
recovery boundary case. Review identified a direct-ECG deep-link start that
missed the grace period; its regression also fails before arming recovery in
validated `start()`, then passes. All 28 Node tests, the UI-state checker, context checker
and whitespace checks pass. The existing browser-controller regression now
verifies timeout navigation, inactive state, aborted receipt and disconnection
of every scheduled audio cue. Participant-data writers and audio assets are
unchanged.

Native Chrome passes eight signal-loss cases: all five scenario routes using
a delayed first poll, a late recovery event before the next timer callback,
short recovery preserving the same run, and real elapsed loss while paused.
The paused case returns to `#h10` after 30,509 ms even though the audio clock is
frozen. Every expired case saves an aborted CSV/receipt, clears the scenario,
retains patient details and moves focus into the Polar gate. All five lost-signal
scenario cases also verify searching/selecting again; one starts by valid-ECG
deep link before Continue. No page errors occur. The existing 137-case responsive
matrix also passes, including 320 CSS px, portrait/landscape, enlarged text,
spacing and long labels, using real Pretext/DOM checks.

Ignored evidence: `output/playwright/check-polar-timeout.mjs`,
`polar-timeout-results.json`, `polar-timeout-polar.png`, and `polar-layout-*`.
These are browser-demo checks; physical Polar H10, headset and phone/browser
qualification remain open. Publication evidence belongs to PROJECT.md.

Live verification: containing release `c2356be` completed Pages run 42254269547
successfully; the exact deployment API reports `succeed`. All 12 changed public
files match Git bytes and both entry URLs return 200. Native Chrome passes all
eight signal-loss cases on the public origin, including a real paused loss of
30,676 ms, with zero page errors. Initial live harness attempts timed out at
30 seconds during full-page/audio loading; using DOM readiness and a 90-second
audio-preparation allowance completes the checks without changing runtime.
The 30-second signal-loss deadline is unchanged. Evidence remains ignored in
`polar-timeout-live-results.json`, `polar-timeout-live-polar.png` and
`polar-release-c2356be.json`. The generic Node asset fetch hit a connection
timeout; the exact-byte check completed with Python's standard HTTPS client.

## Password-gated direct Discord delivery — 2026-11-07

The user removes Cloudflare from the Discord notification path and requests an
encrypted webhook on GitHub Pages, a masked send-time password prompt and
browser password-manager saving/autofill. Optional automatic Wormhole upload
retains its separately configured relay; the app posts links straight to Discord.

Node regressions fail before the direct sender/envelope module exists, then pass.
49 tests cover exact-host webhook URL validation, no redirects/cookies/referrer,
provider acceptance, authenticated AES-256-GCM round trip, fresh salt/IV,
PBKDF2-SHA256600k, wrong password, tampering, envelope bounds and abort-before-send.
A private helper generates the144-bit password and verifies the actual encrypted
file without printing credentials. Commit only ciphertext. Correct passwords
can recover the webhook; old copies remain usable until webhook rotation.

Native Chrome fixture checks cover masked auto/manual prompts, Escape/Cancel,
wrong password, Discord failure/retry, QR/ZIP retention, stale abort, focus,
keyboard and five viewport sizes plus200% text/spacing. Review's parent-focus
issue and an Escape close-event clearing race are reproduced red then fixed.
The browser credential API fixture observes no save on cancel, wrong password
or failed delivery, and one save on successful explicit opt-in. Native browser
settings/Save/autofill on a physical target device remain user-side behavior.
No site storage receives the password or unlocked webhook. Existing five upload
and seven manual QR cases pass; no manual QR external traffic when forwarding is
off. UI/context/whitespace checks pass. Screenshot inspected without secrets.

A real Chrome run at `https://viscereality.org/atmungsraum/` with local source
routed there before publication loads the actual encrypted file, prompts for
the password and makes zero Discord requests on incorrect entry. Correct entry
makes one real browser POST to Discord (no proxy/fulfil or CORS bypass), gets200
and reads back the matching synthetic keyed link. Network origins are exactly
the site and Discord, with Worker settings empty; no Cloudflare request occurs.
No password is sent or left in the prompt, and no participant data is used.
Ignored receipts: discord-password-results.json, discord-password-phone.png and
discord-password-local-source-network.json. Exact Pages run42362312447 for `3d861b1` succeeds, including byte equality
for all18 changed files and both public entry URLs. The real published page
repeats wrong-password/no-send, direct Discord POST, confirmed readback and
password clearing with no Worker requests. The official Wormhole page downloads
the9,437,363-byte synthetic archive and its SHA-256 matches the prior configured
upload. Five live prompt layouts and200% reflow pass. Chrome's actual native
credential-store call fulfills for the synthetic opted-in case; browser consent
and saved-password selection remain user-controlled. Ignored live receipts:
discord-password-live-network.json, discord-password-native-api-results.json,
sharing-release-3d861b1.json. The operator password and webhook remain private.

## Automatic Wormhole upload and Discord — 2026-11-06/07

The user-selected route now supports browser ZIP encryption/upload, local QR
and automatic notification to a privately configured fixed Discord webhook.
Saved-record/CSV identity and the ZIP writer remain unchanged. See the active
PROJECT checklist and sharing/README.md for installation/account acceptance.

All45 Node tests in the current worktree pass (including four separate PWA
installation tests); UI-state/context and whitespace checks pass.
TDD observations: missing upload client/routes fail before implementation;
the real Backblaze upload path's two opaque suffix segments produce a failing
regression before the allowlist correction. Two-pass encryption tests upload a
9 MiB synthetic ZIP in two pieces, independently decrypt all bytes, verify
metadata/piece hashes, restrict contacted hosts and require final completion
acknowledgement. Worker tests cover origin/auth, bounded bodies, fixed recipient,
unsafe links/storage endpoints, token isolation, provider failures and redirects.
A Cloudflare deployment initially failed before network I/O because workerd
rejects `redirect:error`. Official [Workers Request documentation](https://developers.cloudflare.com/workers/runtime-apis/request/)
confirms `manual` is required. Worker-only fetches now use manual and reject
3xx; browser fetches keep error mode. Regression assertions fail before this
change and pass after it. Independent review finds no remaining blocker.

Native Chrome passes five automatic-sharing layouts (320×568,390×844,568×320,
844×390,1920×1080), four forwarding layouts, seven manual QR layouts and ten
existing file-dialog cases. Checks cover missing settings/no request, confirmed
upload/QR, notification failure then send-only retry, retained share after
close/reopen, unchanged-archive reuse, new-record invalidation, failed/cancelled
upload, downloadable fallback, missing/delayed QR module, independent QR decode,
clipboard denial, enlarged text/spacing, Pretext, focus and no horizontal dialog
overflow. Review found Cancel hid the focused button; browser regression was
observed red then green after focus returned to the upload button. One earlier
forwarding layout run and two initial loading checks were rerun after concurrent
PWA edits settled; all affected checks pass. No page errors or external QR
requests occur. Screenshots/fixtures remain ignored under output/playwright/.

Real network evidence uses synthetic data only:

- Production client + Worker handler running locally uploads a160-byte test ZIP
  to actual Wormhole/Backblaze; independent download/decryption and official
  Wormhole UI download both match its ZIP bytes. The authorized Discord test
  message is acknowledged and read back with the full keyed URL.
- Native Chrome at the canonical site origin (local source routed for this
  prepublication check) reaches the actually deployed Cloudflare Worker,
  uploads two encrypted pieces and receives Wormhole completion200 and Discord
  acceptance200. The exact message is read back and labelled as a synthetic
  Cloudflare browser test. Independent download/decryption restores all9,437,363
  ZIP bytes with SHA-256
  `808a69e028c3d770bdda050982a035cdf04c0143df56f16908f3e8df11097a6f`.
  No participant data, webhook/access keys or keyed links enter public evidence.

Ignored receipts include `wormhole-network-verification.json`,
`automatic-sharing-results.json`, `link-forwarding-results.json`,
`wormhole-results.json`, `files-auto-regression.json` and
`cloudflare-browser-local-source.json`. Private receipts/configuration/claim
links stay local. The Worker is on a temporary Cloudflare account; successful
runtime verification does not establish durable ownership. The user must
complete the claim within its window, or log in and redeploy. Physical phones,
Safari and large real study archives remain unqualified. Runtime release `0d0e170` completes exact Pages run42309917241 and reports
`succeed` from the deployment API. All32 changed public files match Git bytes;
JavaScript MIME types and both entry URLs pass. Native Chrome repeats the full
real upload/Discord/readback/independent-decryption check from published Pages
source with the same9,437,363-byte synthetic ZIP/SHA-256 above. The live origin
also passes five upload, four forwarding, seven manual-QR and ten files-dialog
cases, with no page errors. Ignored receipts: sharing-release-0d0e170.json,
cloudflare-browser-live.json and files-auto-live.json. This qualifies the current
hosted workflow, while the account claim/operator and physical-phone acceptance
remain open. Publication/checkpoint state is owned by PROJECT.

## Wormhole ZIP handoff and local QR — 2026-11-06

The new folder-dialog action packages the existing saved CSVs and guides manual
upload on Wormhole, followed by full-link paste and local QR generation. It does
not implement automatic upload. Actual `/api/room` preflight with site Origin,
POST and content-type returned405 without CORS permission; the current Pages
site has no backend. Official [security](https://wormhole.app/security),
[FAQ](https://wormhole.app/faq) and [roadmap](https://wormhole.app/roadmap)
were read. No participant files or test ZIPs were uploaded to Wormhole.

Both new Node regressions were observed failing before implementation, then
passing. All24 Node tests, UI-state/context checks and whitespace checks pass.
The ZIP from the new handoff download independently extracts all three synthetic
fixture CSVs byte-for-byte. A separate QA-only jsQR1.4.0 decoder reads the actual
rendered SVG back to the complete expected URL, including its22-character key.
The pinned MIT QR encoder is local and lazy; native Chrome's initial check
caught Python's `.mjs` text/plain MIME mapping, fixed by serving the unchanged
module as `.js`. No remote QR renderer or proxy is used.

Native headless Chrome passes seven QR layouts (320×568,390×844,568×320,
844×390,768×1024,1280×720,1920×1080): no horizontal dialog overflow, clipped
labels, unmeasured visible text or controls below48px. Existing file-dialog
regression passes10 cases including60 exports, native sharing/cancel/failure,
200% text/spacing, long names, reload and playback continuing across dialog use.
New QR checks cover enlarged text/spacing at320px, independent decoding, invalid
links, edit/refresh clearing, clipboard success/denial, empty-state disabling,
Escape/focus restoration, missing-encoder fallback and delayed import completing
after dismissal. QR generation sends zero external requests and reports no
page errors. Phone screenshot inspected. Evidence stays ignored under
`output/playwright/check-wormhole.mjs`, `wormhole-results.json`,
`wormhole-phone.png`, `files-wormhole-regression.json` and synthetic ZIPs.
Physical phones/Safari, real Wormhole upload/download and full automatic sharing
are not qualified by these desktop checks. Publication evidence follows in PROJECT.

Live release: containing commit `e5f58b49de54502fe96564ecfce45008a35fe964`
completed [Pages run42247056025](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42247056025)
successfully. All six affected runtime/license files match Git SHA-256 bytes,
with JavaScript MIME types; slash/no-slash URLs return200. Native Chrome repeats
the seven QR cases (including independent decode, missing module and delayed
completion) and ten files-dialog cases on the public origin with zero page
errors/external QR requests. Only synthetic browser-local records are used;
there is no actual Wormhole upload. Ignored receipts are
`wormhole-live-assets.json`, `wormhole-live-results.json` and
`files-wormhole-live.json` under `output/playwright/`.

Blind-spot check: Wormhole's manifest/service worker support installed-app
Web Share Target file receipt. The existing native share action can hand off
to available user-selected targets, but this is not a portable automatic
upload/link-return API for the static site. See the [official browser contract](https://developer.chrome.com/docs/capabilities/web-apis/web-share-target).

The evidence-only `9a91534` deployment timed out in `updating_pages`; two
failed-job retries immediately cancelled, even after explicit API cancellation
confirmed the failed deployment was cancelled. Runtime release `e5f58b4` remains
live and verified. Recovery uses a fresh documentation-only commit/build; no
runtime assets, participant data or unrelated media are changed.

## Numbered per-cue CSV sidecars — 2026-11-06

Each of the 53 existing scenario WAV/TXT/JSON groups now has a same-basename CSV.
The first-use numerical prefix and onset timestamp remain unchanged. Each CSV
lists every occurrence of that cue, accounting for all 890 authored events
across the five menu scenarios and the separate 20-minute diagnostic. No audio,
runtime JavaScript, protocol JSON or playlist bytes changed in this increment.

The new Python regression first failed on all 53 missing CSVs, then passed after
generation. `python -m unittest discover -s tests -p test_scenario_assets.py`
independently parses each CSV with the standard CSV reader and checks prefixes,
onsets, every global event order, frame/second times, cycle/block annotations,
matching TXT words and explicit audio/script status against the source plans.
`python tools/build-scenario-placeholders.py --check`, all 22 Node tests,
six scenario JSON/CSV pairs, UI state/context and whitespace checks pass.
Rendered UI/audio reruns are unnecessary for these data/documentation sidecars;
runtime, audio and scheduled events are unchanged. Publication checks follow.
Independent review found no actionable issues and separately audited all 890
events, confirming each is represented exactly once across the 53 cue CSVs.

Publication: `a216168e92b22f09fef1fb6453fc5da3156d7ba9`, Pages deployment API
status `succeed`. Actions `42243976605` attempt 2 was subsequently cancelled
when the concurrent successor `e11a926` started; backend success and actual
live bytes independently establish publication. All 67 checked public files
match local bytes: 53 per-cue CSVs, six generated scenario README TXT files,
six protocol JSONs, the scenario index and app HTML. Local hash evidence is
in ignored `output/scenario-csv-live-a216168.json` in the active checkout.
The active relocated checkout passes the Python regression and package check;
its scenario library was opened in Explorer. This follow-up changes evidence
only; preserve unrelated concurrent source and publication notes.

## Scenario-owned silent placeholders — 2026-11-06

The current UI uses53 independent silent WAV/JSON/TXT cue groups across six
scenario folders, numbered in first-occurrence order, with manifests, asset and
timeline CSVs and playlists. Draft intended wording is labelled separately from
empty actual spoken text. Original library recordings remain unchanged; the
older speech evidence below describes those retained originals. Main is still
exactly900seconds, calibration26–41seconds, with future accelerometer value null.

Observed the missing-registry test fail before implementation; the native
browser also exposed input-buffer detachment during Web Audio decode. Reproduced
that detachment in the mock and observed the regression fail, then fixed it.
All22 Node tests, six scenario pairs,53 generated placeholder groups, UI/context
and whitespace checks pass. Twelve native Chrome offline renders at44.1/48kHz
have exact lengths and zero sample error. The68-second export is entirely silent.
Five real playback routes pass Start/Pause/Resume/End, actual26/41-second main
transitions, calibration hook, scenario receipt hashes,900-second CSV metadata,
replay and missing-WAV retry. Seven unique silent durations are fetched/decoded,
with scenario-specific buffers verified by a different same-ID recording test.
All18 root/main responsive, enlarged and long-text cases pass; phone screenshot
inspected. No page errors. Read-only independent review found no issues and
independently checked53 pairings, six manifest hashes, cue order and playlists.
Final narration still requires wording approval; no new speech was synthesized.

Publication recovery: feature6f097cf is pushed, but Pages run42241115947
attempt1 timed out in updating_pages after10minutes and failed-job attempt2
was cancelled immediately. Build passed; artifact size2,146,337,782bytes
produces the existing1GB warning. The affected live manifest still returns404.
Trigger a fresh evidence commit/build, then verify its exact deployment and UI.

Resolved: eafec8a836b3a92851ed6b8816806ada103a4d06 deployed in run42243359695
attempt2 after explicitly cancelling the stale6f097cf Pages deployment lock.
All265 changed public files match their committed bytes and both entry URLs
return200. Five live browser routes pass actual26/41-second transitions,
pause/resume, CSV metadata, receipt hashes, replay and missing-WAV retry;
18 enlarged/localized text cases pass with no page errors. The concurrent
a216168 CSV-sidecar increment preserves runtime and audio; retain its owner.

## Single 15-minute main sequence — 2026-11-06

Current acceptance supersedes the earlier two-option/48-second-composite local
checks. The main path is direct and contains exactly 26 + 15 + 40 + (50 × 16) +
19 = 900 seconds, with no baseline/rest/recovery event. Calibration is an explicit
silent asset at 26–41 seconds; Instruction 1 contains the German two-breath
instructions. Instruction 2 reuses instruction.wav. Outro contains seven quiet
seconds plus the original 12-second ending PCM, unchanged. All original nine
WAVs remain intact. Asset classes/transcripts are in the manifest and
main-asset-classes.csv; adjacent JSON sidecars retain hashes/provenance.

All three Instruction 1 phrases passed Secret Tunnel's native Whisper check
with zero word edits; no speech was stretched. Original receipts and the unused
earlier composite remain ignored/private. Repacking uses the verified MP3 cache
and rejects corruption. Respyra source comparison and exact German narration
are in audio/calibration.md. H10 accelerometer integration is future work;
calibration value/unit/method remain null and status not-integrated.

The new single-main-choice regression was observed failing before implementation.
All 21 Node tests, six scenario JSON/CSV pairs, UI state and context checks pass.
Tests cover exact class boundaries/totals, all fifty complete cycles, pause/resume,
no retrigger, receipt and advance-event calibration timing, and unchanged outro
source samples. Native Chrome silently rendered all six scenarios at 44.1/48 kHz (12 renders),
with exact totals and zero maximum sample error. The revised main has 900 seconds,
50 complete cycles, no baseline/rest/recovery, and the proper class order.
All 137 responsive/Pretext cases and 18 main/root text cases (320×568,568×320 and
1920×1080, ordinary/200% plus spacing/long labels) pass. Actual browser checks
covered all five menu playback routes, direct main entry with no duration screen,
calibration transition at 26 s and transition to Instruction 2 at 41 s, pause/resume
inside calibration, one fresh window on replay, advance scheduling event with
unset value, exact saved scenario hashes and participant CSV phases/duration.
All 12 shared WAVs and six JSONs preload once; missing-calibration failure/retry
works. No page errors occurred. Screenshots were inspected. Independent final
review found no actionable runtime issue and confirmed original nine WAV bytes.
Current ignored evidence is output/playwright/calibration-final-* and the
scenario-offline JSON reports. Feature publication passed: commit 23b5f2821eb5d0fd13bdc71af7075268fd3b2e30,
Pages run 42225450192, successful/built for the exact commit. All 54 changed public
files match committed bytes, both entry URLs return 200, and all three removed
main10 asset/review URLs return 404. Live Chrome repeated all five flow checks and
18 main/root text cases with no errors, including real 26/41-second calibration
boundaries, clock freeze/resume, timing event/null measurement, CSV phases and
900-second budget, replay and missing-asset retry. Local/remote main match the
feature SHA at this evidence write. The closing documentation follows the same
publication rule; runtime acceptance is complete.
Physical audio timing, listening acceptance and hardware calibration remain open.

## Responsive proportions — 2026-11-06

The patient composition is centered and capped at 560px high; its equal-column
form is at most 640px wide, uses the same gutters as the gender grid, and caps
the flexible group region at 384px. All corners share a responsive 56–112px
size and proportional artwork. Default body type is 16–18px. Enlarged/no-fit
states retain document reflow and full labels; existing Pretext owns measurement.

All 19 Node tests, seven scenario pairs, UI-state and context checks pass.
`check-proportions.mjs` passes 149 native Chrome cases with no page errors:
320×568 through 1920×1080, portrait/short landscape, breakpoint sides, all menus
and six ready players, exact corner bounds/fixed positions, centering, 200%
text/spacing, keyboard/focus access, long ID/label, locale and resize recovery.
The ten-case folder suite also passes individual/ZIP downloads, share boundary,
modal focus, reload and active playback. The first folder run lost its shared
HTTP server; fresh suites against a dedicated server pass. Phone/desktop PNGs
were visually inspected. All QA artifacts remain ignored in `output/playwright/`
(`check-proportions.mjs`, `proportions-results.json`, `proportions-files.json`).
Physical phones, Safari/WebViews and assistive technology remain unqualified.
Two missing-module/Intl cases retain readable full-label reflow and successful
form navigation. Independent review finds no actionable issues.

Layout commit `f2d683c74bf1c32d56f36f183d27b31dc1938829` is published within
`b164c805c1df1b43f565d3ee69aeffda53eb7307`. Its exact Pages run
[42213877838](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42213877838)
succeeded; the first layout run was automatically superseded by this concurrent
documentation update. All seven changed public files match Git bytes and both
entry URLs return 200. All 149 live layout cases and ten live folder cases pass
with no page errors. Receipts are `proportions-live-results.json`,
`proportions-files-live.json` and `live-release-assets-b164c80.json` in the same
ignored directory. Main/origin match `b164c80` with a clean tree before this
closing evidence update. Publish/check this documentation-only follow-up under
the standing rule; report its exact final Git/deployment state at handoff.

## Polar artwork and frame glow — 2026-11-06

The separate dot is removed. CSS uses the unchanged PNG as a luminance mask;
the original strap artwork and outer frame share state color and cycle duration.
Connected pulses green over four seconds, lost pulses red over two seconds,
waiting is neutral, and reduced motion retains steady state-colored glows.
Accessible status text and the connection/reconnect state owner are preserved.

`check-belt-glow.mjs` passes 24 native Chrome cases: waiting/connected/lost at
320×568, 390×844, 568×320, 844×390, 768×1024, 1280×720 and 1920×1080, plus
reduced-motion loss/recovery and restoring motion. It checks no dot, luminance
mask/source decode, exact symbol/corner sizes, both animations and matching
durations/colors, changing symbol opacity/filter and frame shadow, folder access,
and status text. No page errors occur. Peak green/red screenshots were inspected.
UI-state/context and whitespace checks pass; focused review has no actionable
findings. Receipts and crops remain ignored in `output/playwright/belt-glow-*`.
No bitmap regeneration, logging, playback or layout changes are involved.
Native phones/WebViews remain outside this Chrome evidence.

Release `7e07b9c6b0abd9c8d3e22d54abd339bec130e297` is pushed and live. Pages run
[42213036555](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42213036555)
succeeded; all six published files match Git bytes and both entry URLs return 200.
All 24 live glow checks pass with no page errors. Receipts are
`belt-glow-live-results.json` and `live-release-assets-7e07b9c.json` in the same
ignored directory. The subsequent independent proportions commit `f2d683c`
retains the glow while superseding the original dimensions; its owner tracks
that layout release. Publish/check this closing documentation-only evidence
under the standing rule, preserving the separate task's state.

## Threefold corner enlargement — 2026-11-06

All three fixed surfaces measure exactly 144×144 CSS px. Recenter/folder artwork
is three times its previous dimensions. Polar embeds the unmodified dissertation
`PolarH10.png`, Git blob `71fef4c5acd3cb20122be459675963d476d5bc89`;
the recovered original is raster, not a newly claimed vector asset.

All 19 Node tests, seven scenario pairs, UI/context and whitespace checks pass.
`check-large-corners.mjs` passes 85 native Chrome cases across 320×568, 390×844,
568×320, 844×390, 768×1024, 1280×720 and 1920×1080: patient, ECG, round/options,
all six players and legacy recovery; exact fixed corner bounds, loaded images,
full-label Pretext/DOM checks, bounded centering, inert recenter, keyboard folder
open/close/focus, signal loss/recovery, reduced motion, 200% text/spacing and focus
access. `check-files.mjs` passes ten cases plus downloads/reload/active playback.
Neither suite reports page errors; review has no actionable findings.

The larger fixed surfaces change the supported no-scroll boundary. Patient,
round/options and player screens at 320×568 use document reflow; landscape patient
forms and 568×320 practice choices also reflow. Other tested ordinary cases fit
bounded geometry. Short landscape reserves a central column; >=1280px desktop
retains the centered content lane. The former 161-case suite's blanket bounded
assertion is not applicable to these revised minimums. Physical devices and
screen readers remain unqualified. Screenshots were visually inspected; local
receipts/scripts remain ignored in `output/playwright/large-corners-results.json`
and `corners-files-local-results.json`.

Release `0a3c99a83301c3650d79dca1f88a52389494b2c8` is pushed and live. Pages run
[42208617785](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42208617785)
succeeded. All eight published files match Git bytes, including the exact Polar
image; both entry URLs return 200. The live 85-case corner matrix and ten-case
folder suite both pass without page errors, including download/reload/playback
behavior. Main/origin match with a clean tree before this closing evidence update.
Live receipts are `large-corners-live-results.json`, `corners-files-live-results.json`
and `live-release-assets-0a3c99a.json` in the same ignored directory. Publish/check
the closing documentation-only evidence under the standing rule.

The closing-evidence run `42209351688` for `463c842` timed out while syncing files
and its failed-job retry was immediately cancelled. Build remained successful;
the site artifact is 2,143,894,180 bytes and Pages warned of a 1 GB allowance.
The prior UI deployment remains live and verified. Recovery uses a fresh
documentation commit/run rather than modifying runtime files or unrelated media;
confirm its exact result before claiming that the evidence is live.

## Folder-only data entry — 2026-11-06

Both CSV-Dateien footer links and their obsolete visibility update are removed.
The existing folder dialog remains the visible entry; the legacy recovery URL
and shared link nodes remain compatible. All 19 Node tests, seven scenario pairs,
UI-state/context, syntax and whitespace checks pass. Focused review has no findings.

`check-files.mjs` passes ten Chrome layout cases plus individual/ZIP downloads,
keyboard/focus, share-boundary and 61-file reload/playback checks with no errors.
The first full stretch check exposed the remaining patient link's 180px singleton
budget at 768×1024. A scoped 240px budget fixes that full-label no-fit without
changing other footers; all 161 cases then pass with no errors. The final desktop
screenshot was visually checked: only the centered patient return and corner
folder remain. Receipts are ignored under `output/playwright/` in
`folder-only-local-results.json` and `stretch-results.json`. Native phone sharing
and real hardware remain outside this desktop evidence.

Release `0895f63213792e8a5994712b137620f1dd91ef70` is pushed and live. Pages run
[42203147921](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42203147921)
completed successfully. All nine changed public files match committed bytes;
both entry URLs return 200. The live folder suite passes all ten cases and its
download/reload/playback sequence with no page errors. Local/remote main match
with a clean tree before this closing documentation-only update. Receipts are
`folder-only-live-results.json` and `live-release-assets-0895f63.json` in the same
ignored output directory. Publish/check the closing evidence under the standing rule.

## Folder and standing-workflow release acceptance — 2026-11-05

For the user's explicit all-changes push, refreshed origin/main and local HEAD
both start at `7233468`. Reviewed the complete intended source/assets/tests/notes
and new root/subproject standing Superpowers/publication guidance. Fresh checks
pass all 19 Node tests, seven scenario pairs, UI state, context, JavaScript syntax,
SVG XML and Git whitespace. Cold opening 60 retained files uses exactly two
full-store reads. Ten native Chrome popup layout cases and its download/share
boundary/cancel/error/reload/playback checks pass; the full 161-case stretch
matrix also passes. Both browser suites report zero page errors. Private/local
participant, voice and diagnostic artifacts remain ignored. The following release
record will identify the exact pushed commit, Pages result and live verification.

Release `a52d56a875cc6c30f149f1c4831e098ac12c8711` is pushed to main, with
local/remote SHA equality and a clean tree before the closing evidence commit.
GitHub Pages run `42079217920` completed build and deployment successfully.
`check-live-release.mjs a52d56a875cc6c30f149f1c4831e098ac12c8711` verified all
20 public changed files byte-for-byte against Git and HTTP 200 at both entry
URLs. `check-files.mjs https://viscereality.org/atmungsraum/ files-live-results`
passed all ten layout cases and the full synthetic download/share-boundary/
recovery/playback sequence against the live site, with zero page errors.
The native app share boundary remains simulated; no email or message was sent.
The exact-commit Pages run is
<https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42079217920>.
Receipts are ignored in `output/playwright/files-live-results.json` and
`live-release-assets-a52d56a.json`. Standing instructions were reviewed with no
material contradictions; the subsequent evidence commit changes documentation
only and preserves the verified runtime bytes.

## Saved-files popup and ZIP/share — 2026-11-05

`npm test` passes all 19 tests. The ZIP check reads the actual Blob with Python's
independent standard-library zipfile reader, verifies CRCs and exact CSV bytes,
and extracts Unicode filenames and a same-name collision without data loss.
ZIP32 uses stored entries, with no new runtime dependency; the interoperability
test uses the existing Python interpreter. UI state, seven scenario pairs,
context and Git whitespace checks pass.

`node output/playwright/check-files.mjs` passes ten native headless Chrome layout
cases with zero page errors: empty popup; 320×568, 390×844, 568×320, 844×390,
768×1024, 1280×720 and 1920×1080; 200% root text with WCAG-style spacing; long
German headings and unbroken filenames. Actual Pretext measurement is present,
labels fit horizontally, and controls retain 48px targets. The list scrolls by
design; short/enlarged cases allow whole-dialog vertical scrolling. Phone and
desktop screenshots were visually inspected. The fixed folder has safe-area
offsets and centered footers leave room beside it.

Functional checks use 60 synthetic saved CSV fixtures in an isolated browser:
empty state, individual/ZIP downloads, ordering, unsupported-share fallback,
supported sharing payload and active user gesture, cancellation/rejection,
native-dialog keyboard containment/focus restoration, all menus and unchanged
hash. Opening/dismissing the popup during actual browser audio leaves playback
running with advancing time. Ending that run adds a 61st CSV; all 61 reappear
after reload before patient/sensor entry. Only the platform share API was
simulated; no destination app opened and no message was sent. Physical phones,
email, WhatsApp, Safari/WebViews and screen readers remain unqualified.

The existing `check-stretch.mjs` passes all 161 rendered/playback/reflow cases
with zero page errors after the footer reservation change. Receipts, executable,
synthetic ZIP and screenshots remain ignored under `output/playwright/`:
`check-files.mjs`, `files-results.json`, `files-fixture.zip`, `files-*.png`.
No participant data or generated diagnostic enters public source. The original
implementation-only check did not claim publication; the release acceptance
entry above now records the verified live release.

Focused code review caught a cold-open hydration notification loop. Notifications
now originate only when a run finishes, rather than once per hydrated record.
The ignored `check-files-cold.mjs` regression reproduced 62 full export-store
reads for 60 files before the fix, then exactly two reads with all 60 files and
ZIP readiness after it. The reviewer revalidated the fix with no remaining
actionable findings; all 19 Node tests and the popup functional suite were rerun.

## Stretch-accordion acceptance — 2026-11-05

The main UI now follows the explicit settings in SKILLS.md: equal outer anchors,
dimension-responsive gaps/type, stretched patient/CSV groups, narrow stacked
footer actions and short-landscape arrangements. Practice choices become four
columns; round/main cards use wider rectangles where squares would force wrapping.
The existing Pretext 0.0.9 runtime checks inner text boxes and painted DOM bounds.
Fixed actions check natural width, line count and line height; multiline text
checks widest line and height. User word spacing is marked DOM-verified because
it is outside Pretext's model. No user enlargement is reduced to satisfy a box.

`node output/playwright/check-stretch.mjs` passed 161 local native headless Chrome
cases with zero page errors: 320×568, 320×1080, 390×844, 768×568, 768×1024,
1280×720, 1920×568, 1920×1080, 568×320, 844×390, 480×640, 481×640, 844×480
and 844×481 CSS px across patient, Polar, rounds, both option menus, empty/filled
CSV recovery and ready players. All six real browser audio routes passed
Start/Pause/Resume/End in phone portrait and short landscape. Bounded panels had
no page/group scrolling, clipped labels or targets below 48px; central controls
stayed within 2px of the viewport midpoint and footers retained their anchor.

At fixed 768px width, patient group separation grew from 25.08px at 568px height
to 145.44px at 1024px height. Column separation grew from 12.8px to 30.72px with
width, reaching the 40px ceiling on desktop. Preferred body type grew from 16px
to 20px when both dimensions allowed it; wide/short type stayed 16px. Portrait
forms at 320×568 fit bounded geometry. Short landscape forms and filled CSV lists
that exceed the middle budget use deliberate vertical document reflow, with no
inner scrollbar or horizontal overflow. They are explicit supported-size
exceptions, not strict no-scroll passes.

Additional cases passed 200% root text with WCAG-style spacing, long German round,
status/action labels, an unbroken identifier, locale invalidation and recovery
after text/viewport changes. Word-spacing cases use rendered bounds, not a
Pretext accuracy claim. Two separate fallbacks verified blocked Pretext modules
and missing Intl.Segmenter: both marked measurement unavailable and retained a
usable form in readable document flow. Reflow explicitly schedules a second
measurement even when a wrapped label's box dimensions do not change; hidden
status nodes are excluded from geometry checks. Patient Tab/radio-arrow flow
and derived-ID updates passed. Phone, desktop and short-landscape screenshots
were inspected; executable, JSON receipt and PNGs stay ignored under
`output/playwright/check-stretch.mjs`, `stretch-results.json` and `stretch-*.png`.

Fresh checks also passed all 19 current Node tests, seven JSON/CSV pairs,
UI-state/context, app syntax and Git whitespace. Concurrent CSV-folder/dialog
work is preserved and owns its separate overlay acceptance. No package upgrade,
audio protocol or participant logging change was introduced by this layout task.
These are desktop Chrome results; physical phones, browser zoom, Safari/WebViews,
screen readers, real H10/headset behavior and live final-refinement deployment
were not qualified by this check. RTK was unavailable; native bounded commands
were used. See PROJECT.md and the publication task for observed Git/release state.

## Patient/ECG and recenter release checks — 2026-11-05

For the user's all-changes publication request, refreshed remote main matched
starting HEAD `ac39ec4`. Reviewed all 26 changed/new files: runtime/source,
tests, instructions, native recenter SVG, synthetic NeuroKit JSON/binary and
the parent subproject contract. Local participant exports and diagnostic/voice
artifacts remain ignored. No hosting settings or authored audio were changed.

Fresh release checks pass all 18 Node tests, seven scenario pairs, UI state,
context and whitespace. The added background-stream regression records at
source indices 4 and 20 around an idle interval, retaining the same epoch and
sample array; each export contains only its own recorded interval. Existing
pause checks verify ongoing ECG sample changes with frozen audio time. This
proves the current demo's acquisition/recording separation, not real H10 input.

Native Chrome reran all 25 patient/integration cases, actual >30-second loss,
natural 68-second completion, six CSV routes and four fallback cases without
page errors. All 50 recenter cases passed on the final tree. An initial
concurrent recenter run timed out waiting for Pretext readiness; a diagnostic
rerun passed without a product-code change or page errors. Exact deployed
commit, live asset integrity and browser evidence will be recorded here after
the Pages build. The Pages connector returned an internal error; the existing
authenticated GitHub CLI/API confirmed the main/root Pages source and prior
`ac39ec4` build, and is the deployment-verification fallback.

Initial release `e2b6e32` pushed successfully with remote/local SHA equal.
While its Pages build ran, additional `app.js`, `index.html` and `style.css`
stretch edits arrived. They improve DOM/text fit checks, locale/font/content
remeasurement, dynamic-viewport sizing, patient/CSV vertical distribution,
column gaps and the readable reflow path. Reviewed their complete diff and
reran all 50 Chrome cases against this updated working tree: all passed, no
page errors, clipping or horizontal overflow. Extended browser evidence records
two successive runs at source positions 2313–2436 and 2607 onward, proving the
background source advanced through the intervening menu/idle time. Every CSV
ECG value matched the shared binary; 102 paused ECG rows retained one audio
time while ECG values changed. This late layout work is included in the same
authorized publication, preserving the existing central controls and recenter
placeholder contract.

Pages run `42059584563` succeeded for `e2b6e3230df84870109ac9dc3932c53b891736b4`;
the Pages API reported that exact commit built. All 25 public changed files,
including the ECG binary and documentation, matched committed SHA-256 bytes.
Both slash/no-slash entry URLs returned HTTP 200. Live Chrome passed 50 cases
with no page errors. The first recording started at source sample 5300 and
ended at 5424; the next started at 5605. Every exported ECG value matched the
published binary, and 102 paused ECG rows retained frozen audio time while
ECG values changed. The desktop screenshot was reviewed. Live scripts/receipts
and synthetic downloads stay ignored as `live-release-assets.json` and
`recenter-live-*` under `output/playwright/`.

The final native Chrome stretch matrix passed all 156 cases with no page
errors. It checks independent width/height budgets and breakpoint sides,
patient spacing growth, column/type growth, preserved center/footer anchors,
every screen and all six playback/pause/resume routes, short landscape,
enlarged text, long content/recovery and unavailable measurement. Screenshots
and the detailed receipt remain ignored under `stretch-*`. The final source
also uses rectangular round cards in short landscape and four practice columns
where necessary to retain bounded panels. Publication of this follow-up still
requires exact-commit deployment evidence.

Follow-up source/evidence commit `72334681b5e447f7298aacf58b255fc5543b99b6`
pushed with local/remote SHA equal. A fresh check of that committed layout passed
all 156 Chrome cases. Pages run `42060990832` completed build and artifact upload,
but deployment attempt 1 returned GitHub HTTP 500 while creating the deployment.
The GitHub connector successfully requested rerun of failed jobs only; successful
build/report jobs and the artifact are reused. A Pages API `built` response alone
does not establish that the deploy job succeeded. Later CSV/ZIP and Pretext guard
edits remain local under their separate active tasks, outside this release snapshot.

Deployment attempt 2 also returned HTTP 500; attempt 3 retries only the failed
job after a delay. The artifact-size warning is a separate standing concern:
this upload is 2,142,307,246 bytes versus 2,142,305,761 bytes for the successful
preceding release, so the warning alone does not establish this failure's cause.
[GitHub's limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)
state a 1 GB published-site limit. The official
[GitHub status page](https://www.githubstatus.com/) reported Pages operational
and no incident on 5 November at the diagnostic check. Do not describe these
repository-specific failures as a confirmed platform-wide outage or remove
existing site media without a separate reviewed plan.

Attempt 3 succeeded: run `42060990832` reports success for exact commit
`72334681b5e447f7298aacf58b255fc5543b99b6`, and the Pages API reports it built.
All 25 public files changed since `ac39ec4` match their committed SHA-256 bytes;
both entry URL forms return 200. A final sequential live Chrome pass verifies
all 50 recenter/layout/no-op cases with real Pretext and no page errors. Two
recordings start at source positions 7395 and 7689 (first ends at 7523); every
ECG value matches the published binary. The 103 paused ECG samples retain one
audio position while ECG values change. The final desktop screenshot was reviewed.
Receipts/screenshots/synthetic CSVs stay ignored as
`live-release-assets-7233468.json` and `recenter-live-final-*` in `output/playwright/`.

Initial concurrent final browser runs hit the 30-second gate timeout while
the ECG binary and a Pretext module were still downloading. No JS or validation
error occurred; an independent uncached binary fetch took 13.93 seconds versus
1.32 seconds for a query-qualified fetch, with identical bytes. The sequential
rerun allowed a 90-second network timeout and passed without changing released
code. This establishes functionality, not a loading-speed budget or physical
phone qualification. The local final-source stretch matrix passed 156 cases;
the wider live matrix stopped at that same initial download timeout, while the
final 50-case live matrix completed successfully.

At final Git observation local and remote main equal `7233468`. Newer CSV/ZIP
work, an extra Pretext capability guard and these closing notes remain local
in the shared checkout and are outside the verified release snapshot. They
must not be reported as deployed by this task.

A separate fresh live navigation also reached Polar found, connected and the
round menu. `live-rounds-fresh.png` was reviewed with the heading, both footer
actions and both corner controls visibly rendered; their DOM names, colors,
bounds and hit targets were confirmed. An earlier automated capture omitted
the header/footer paint despite visible measured bounds, so use this stable
fresh capture for the final visual evidence rather than inferring pixels from
geometry alone. The only console warning in that diagnostic was the site's
unrelated favicon 404; the functional final 50-case run had no page errors.

## Patient entry, NeuroKit mock and participant CSVs — 2026-11-05

All 17 Node tests pass, including existing audio/scenario regressions and new
identity/filename validation, pause-aware sample export, missing-signal gaps,
delayed-stop boundary, signed pre-start waiting, checkpoint reconstruction and
shipped mock-source integrity. UI-state, seven scenario JSON/CSV pairs, context,
JS syntax and Git whitespace checks pass. The actual Python generator produced
460,800 finite float32 samples at 256 Hz using NeuroKit2 0.2.13. Binary SHA-256:
`ebe8407430f905f53118ce66c3858815d4de9383bad2a8ef984689ecf8dc6187`.
No runtime speech, audio samples or authored scenario timing changed.

Native Chrome passed 25 rendered/integration cases: patient entry at 320×568,
390×844, 768×1024, 1280×720, 1920×1080, 568×320, 844×390, 480×640 and
481×640 CSS px; 320px with 200% root text, long German labels and WCAG-style
text spacing; round selection; playing/paused states for all six scenarios;
and the CSV recovery view. Normal supported portrait/desktop panels stayed
bounded; compressed/enlarged text used readable document reflow with no inner
scrolling, clipped labels or horizontal overflow. Actual locked Pretext reported
ready/fit. All checked controls retained at least 48px targets and playback
remained at the viewport midpoint. Phone, desktop and enlarged screenshots were
reviewed; local valid-input screenshots were also captured.

All six session routes downloaded the expected `Fej_35_M_probe1..4.csv` and
`Fej_35_M_main1..2.csv`, with common identity/schema, correct scenario, actual
source values and status. Audio elapsed froze during pause while mock ECG
continued. Reload retained completed exports and counters. A real uninterrupted
loss stayed in the session through 29 seconds, then returned to Polar beyond
30 seconds, retained patient details and exported `main3` as aborted with blank
ECG values during the gap. A full real-time 68-second Anleitung naturally
completed and exported `probe5` with `completed` status. Reduced-motion Chrome
used no indicator animation. There were no page errors.

Additional Chrome checks passed native form validation, keyboard Tab/radio-arrow
order with correct M/F/O/N ID updates, unchanged typing focus on sensor updates,
JavaScript-disabled submission protection, missing ECG asset blocking/retry,
unavailable Pretext reflow and interrupted-run recovery with network ECG blocked.
The independent read-only reviewer reproduced and verified fixes for delayed
counter reservations losing signal marks/extending aborted tails, stale failed
attempts aborting a newer run, and pre-start pauses losing signed audio lead.
The reviewer verified IndexedDB checkpoint recovery without network source
access and that another tab cannot recover a recording while its Web Lock is
held; teardown released the lock. No P1/P2 findings remained in that review.

Detailed receipts/scripts remain local under `output/playwright/`:
`check-patient-data.mjs`, `patient-data-results.json`,
`check-patient-fallbacks.mjs`, `patient-fallback-results.json` and the screenshots.
RTK was unavailable at its documented location, so bounded native commands were
used. These are local desktop Chrome checks, not physical-phone/Safari testing,
H10 acquisition, acoustic synchronization, study-data qualification or live
publication. Recovery ends at the last durable checkpoint and can omit an
uncommitted tail. See DATA-LOGGING.md for the complete logging contract.

## Persistent VR recenter placeholder — 2026-11-05

Native headless Chrome passed 50 rendered cases using the local HTTP-served
page. Patient entry covered 320×568, 390×844, 768×1024, 1280×720, 1920×1080,
568×320 and 844×390 CSS px. Round selection, both option menus, CSV recovery,
Anleitung and 15-minute main ready screens covered six phone/desktop/landscape
sizes; additional checks covered Polar discovery/ECG, playing, paused, 320px
with 200% text plus text-spacing overrides, and scrolling the enlarged page.
The button stayed fixed, visible and topmost with a 48×48px target. Heading text
cleared the control. No horizontal overflow, clipped measured labels or page
errors occurred; actual Pretext was ready and the labels fitted. Bounded
playback retained its viewport midpoint. Reflow remains available where needed.

Pointer clicks and Enter/Space activation left route, UI state and audio state
unchanged on patient entry, Polar, every menu/recovery/ready screen, playing,
paused and scrolled reflow. Keyboard focus remained visible. The control exposes
its German name and unavailable description; it has no handler, transport event
or command. Native SVG parsed as XML with no embedded raster. The phone patient,
paused-player and desktop screenshots were reviewed. The executable, 50-case
receipt and screenshots stay ignored under `output/playwright/check-recenter.mjs`,
`recenter-results.json` and `recenter-*.png`.

UI-state, context and Git whitespace checks passed. This task adds no JavaScript
behavior, dependency or participant logger. RTK was absent at its configured
path and on PATH, so bounded native shell reads were used. Physical phone safe
areas, Safari/WebViews, screen-reader output and actual headset recentering were
not qualified by this desktop Chrome check. No commit/push/deployment was
performed; existing patient/logging work was preserved in the shared checkout.

## Combined release and nested skill baseline — 2026-11-05

The subproject entry chain now routes `AGENTS.md` → `for-ai/README.md` →
`for-ai/SKILLS.md`, using the installed For-AI scaffold's full shared baseline
and existing menu-specific rules. Review confirmed RTK, GSD Core, Superpowers,
Ponytail, Pretext, accessibility, performance, MobX and conditional native/remote
routes, one state owner and Git/publication safeguards. The unavailable older
remote skill identifier is explicitly mapped to the installed BRSP skill.
No product dependency or native stack selection was added.

Fresh prepublication checks passed all 13 Node audio tests, seven scenario
JSON/CSV pairs, UI state, context and whitespace. Native Chrome passed the
13-case round-layout check, path-only SVG alpha checks, 24 player-layout cases,
and all six scenario Start/Pause/Resume/End routes with exact JSON receipt
hashes, nine shared WAV fetches, missing-JSON failure and successful retry.
No browser errors occurred. The existing ignored executables and receipts in
`output/playwright/` own detailed evidence; prior native offline-render evidence
is recorded in the scenario section below.

Release `5a0e3a93e6bccf70c1d30091aaa66e72b1d7869e` was pushed to `main`,
with local and remote HEAD equal. Pages run `42048199569` succeeded; the Pages
API reported that exact commit `built`. On the public site all 26 checked text
assets matched the committed content: HTML/CSS/app and audio modules, the shared
manifest, two SVGs, the AGENTS/README/SKILLS entry chain and seven JSON/CSV pairs.
Both slash/no-slash URLs resolved successfully. Live Chrome verified both round
branches at 900×553 and 320×568 CSS px with real Pretext, no page overflow and
no page errors; the desktop screenshot was inspected. A separate live browser
pass exercised every one of the six scenario Start/Pause/Resume/End paths,
exact JSON receipt hashes, nine shared WAV fetches and missing-JSON failure/retry.
All passed with no page errors. Receipts/screenshots remain ignored under
`round-infographics-live*` and `scenario-player-live*` in `output/playwright/`.
Hardware and listening acceptance remain separate open checks.

## Scenario audio protocol — 2026-11-05

The six menu choices and engine-only 20-minute diagnostic each have JSON/CSV
timelines in `audio/scenarios/`. Playback, native offline diagnostics and WAV
verification exports now consume saved JSON. The nine shared library WAVs and
approved timing remain unchanged; a regression proves all seven saved plans
equal their former calculated plans, with explicit null assets for silence.

All 13 Node audio tests passed, including proof that an edited JSON asset is
actually scheduled, receipt hashes match loaded JSON, invalid assets/timing/
metadata/library revisions are rejected, and missing/invalid files prevent any
fallback playback. Seven JSON/CSV consistency checks, the UI-state check,
context checker and Git whitespace check passed. A failing LF-byte regression
caught Windows CRLF changing the pinned manifest hash; scoped `.gitattributes`
now fixes LF for the selected manifest and scenario JSON/CSV. Its LF hash is
`1a4fe0f2546d5850865bb2aab8c737cff05244ff40286044189e7a3cd53fb275`,
identical to the published Git blob. Audio sample bytes were not changed.

Native Chrome silently rendered all seven scenarios at both 44.1 and 48 kHz
(14 renders). Every output sample matched the selected clips/silence with zero
maximum error; the exact 68/300/600/900/1200-second totals and silent 300-second
main baselines held. The verification exporter also produced the exact 68-second
Anleitung from saved JSON. All 886 participant text-score ranges, German scripts
and clip references matched the scenario events; their source hashes were refreshed.

Real browser checks passed Start/Pause/Resume/End for all six menu routes,
correct scenario/JSON hashes in receipts and one preload of exactly nine WAVs
and seven JSONs. A missing scenario blocked playback with retry feedback;
restoring it allowed preparation to recover. No page errors occurred. The phone
playback screenshot was inspected. Local executable checks, native render
receipts and screenshots are ignored under `output/playwright/scenario-*` and
`check-scenario-*.mjs`. These checks establish local browser scheduling, not
physical acoustic/video synchronization, headset/H10 behavior, listening
acceptance or deployment. No push/publication was performed for this change.

## Playback buttons and progress — 2026-11-05

The controller displays elapsed/total time and native progress from the same
Web Audio clock as playback. Its extended regression first failed against the
old countdown, then all eight audio tests passed with exact 600/900-second main
totals, a paused 12-second position, the 68-second Anleitung total, completion,
replay reset, signal-recovery gating and cancelled-unlock receipt protection.
The UI-state check, context checker, JS syntax and Git whitespace checks passed.

Native headless Chrome passed 24 rendered cases. Ready, playing and paused
states at 320×568, 390×844, 768×1024, 1280×720, 1920×1080, 568×320 and
844×390 CSS px stayed bounded without document/internal scrolling, clipped
labels, overlap or page errors. Both actions retained at least 48px targets;
the primary button stayed within 2px of the viewport midpoint. Real Pretext
reported ready/fit. Short landscape groups phase and time above the bar; loading
or failure status occupies the upper spare track rather than shifting playback.

The remaining cases verified 320px with 200% root text, combined long German
labels/text-spacing/200% text, and unavailable Pretext. They retained readable
document reflow without horizontal overflow or clipping. Keyboard Space starts,
pauses and resumes; Escape stops. Native progress advanced during playback,
remained exactly frozen while paused and advanced again on resume. Progress is
read-only and is not a seek control. Screenshot review covered phone, desktop
and enlarged-text layouts. The local runnable check, JSON receipts and PNGs
are ignored under `output/playwright/check-player.cjs`, `player-results.json`
and `player-*.png`. This is Chrome/local verification; physical phone browsers,
Safari, H10/headset execution and live deployment were not checked by this task.

## Round-button SVG infographics — 2026-11-05

The user simplified the initial diagrams to practice lungs with one repeat
arrow and a research/data-collection clipboard with a three-point data chart.
Two new built-in image generations were converted with VTracer to two and three
SVG paths, respectively. Runtime SVGs are 8,907 and 12,982 bytes, with the same
240×160 view box and two-color palette. They contain no bitmap, background
rectangle, script or external reference; transparent outer corners and internal
gaps were checked in rasterized alpha. The page loads the SVGs with revised
image cache tokens, preserving visible German labels and link names. Source
PNGs and exact prompts/conversion settings live in `assets/rounds/`.

The 13-case rendered/keyboard/fallback check below was repeated successfully
for the simplified revision. It is now published and live-verified by the
combined release above; `a79275f` and Pages run `42044006553` refer to the
earlier artwork.

Native headless Chrome passed 13 rendered cases: 320×568, 390×844, 768×1024,
900×553, 1280×720, 1920×1080, 568×320, 844×390, 480×480 and 481×480 CSS px,
plus 320px cases with 200% text, user text-spacing overrides and long German
labels. Images decoded, actual Pretext reported ready/fit, labels and images
did not overlap, cards had no internal overflow, and ordinary viewports had
no page scrolling. Enlarged text used readable document reflow. Phone and
desktop screenshots were visually inspected.

Both links retained exactly their visible accessible names, visible keyboard
focus and Enter activation into the expected four/two choices. Blocking Pretext
retained readable choices in document flow. No page errors or failed runtime
requests occurred. Local receipts, screenshots and the runnable check are in
ignored `output/playwright/round-infographics-*` and
`output/playwright/check-round-infographics.mjs`. Physical phones and Safari
were not tested. All eight audio tests, UI-state, context and Git whitespace
checks also passed. RTK was unavailable at the documented Windows path, so
bounded native commands supplied the verification output.

## Centered main UI — 2026-11-05

Before correction, native Chromium measurements showed left-aligned single
footer actions, an off-center sensor-selection button and primary controls
shifted vertically by the unequal header/footer and playback-feedback rows.
The shared shell now reserves equal measured edge tracks, and the play body
reserves equal feedback tracks around its primary control. Headings, single
footer actions, the sensor selection, ECG and choice groups share the centerline.
The shell follows the dynamic viewport, with its existing small-viewport fallback.

Headless Chrome passed 79 rendered cases at 320×568, 390×844, 768×1024,
1280×720, 1920×1080, 568×320 and 844×390 CSS px. Sensor selection, ECG,
both choice grids and idle/playing/paused buttons matched the viewport midpoint
within two CSS px in bounded layouts; single footer actions matched its
horizontal midpoint. All visible labels had real Pretext fit results, with no
clipped text, page overflow or page errors. Phone/desktop screenshots were
visually reviewed. The later 18-case pass covered six routes at 320 CSS px
with 200% text, text-spacing overrides and readable measurement fallback.
Keyboard checks passed sensor selection, Continue, both menu levels,
Start/Space, pause/resume and Escape. All eight audio tests, UI-state/context
checks, JavaScript syntax and Git whitespace checks passed. Local evidence is
under ignored `output/playwright/`; physical phones, Safari and headset behavior
are not established by these Chromium viewport checks.

Public verification: alignment commit `e4f76d9` was carried into `a79275f` after
the first Pages run was superseded by the concurrent asset update. Pages run
`42044006553` succeeded for `a79275f`. Live `index.html`, `style.css` and `app.js`
returned HTTP 200 and matched the alignment commit byte-for-byte. The same
79-case Chromium matrix then passed at <https://viscereality.org/atmungsraum/>,
with no page errors, overflow or text-fit failures. Local geometry error was
zero horizontally and at most 0.0078125 CSS px vertically for bounded primary
controls. Keep concurrent scenario/progress work separate from this evidence;
later changes require fresh checks against their own committed deployment.

## Audio protocol text review — 2026-10-27

`audio-review/` inventories all eight checked-in audio preparation/export tools
plus runtime sequencing/controller/selection and retained manifests. It includes
a participant walkthrough, an explicitly separate audition comparison, and full
text timelines for all six menu choices plus the engine-only 20-minute option.
The timelines were generated from production `createProtocol()` and the runtime
manifest, recording SHA-256 for that manifest, engine, selection and controller.

Fresh checks passed all nine runtime WAV byte counts/hashes; seven timelines
had contiguous events, expected total duration/cycle/rest counts and exactly
two breath variants per third complete cycle. Written source-sample entries
matched the production event counts. Totals were 68/300/600/900 seconds for
Anleitung/practice, 600/900 for main rounds and 1200 for the engine-only option.
An independent reread checked every saved event range, exact German quote,
cycle/block annotation and source hash against the current production plan;
all passed. All eight audio tool scripts were present in the inventory.
All eight Node audio tests, UI-state check, context check and Git whitespace
check passed. Product code and audio assets were unchanged, so rendered UI and
native offline audio checks were not repeated for this documentation task.

The opening promises counting in each phase, while saved holds are uncounted;
this is identified for review without changing speech. Main rounds begin with
300 seconds of silence. Anleitung is one replayable 68-second example, not an
endlessly looping sound sequence. The newer Halten! audition is separate from
the runtime's Und halten. Its documented generated package and source cache
are absent locally; no regeneration, API call, listening assessment, hardware
measurement or fresh public-site verification was performed.
## Voice sample subpage — 2026-10-28

`samples/` contains twenty fully decoded MP3s (397,476 bytes total), each bound
to its saved profile and the exact requested German phrase. The original MP3
hashes and timing are preserved. `node for-ai/scripts/check-samples.mjs` checks
the twenty unique names, both 0–9 series, LANG valence ordering, all source
links/downloads, byte hashes, and truthful transcript checks without private
profile identifiers or machine paths. Native raw Whisper scores are preserved;
digits 2/3/4 and German zwei/drei/vier have a separately documented equivalence
check. Later single-take samples use independent German ASR on the final decoded
MP3, without fabricating a native checked-take score.

Headless, audio-muted Chromium passed seven ordinary viewport checks:
320×568, 390×844, 568×320, 760×800, 761×800, 1280×800, and 1920×1080.
At 320 px, 200% text, WCAG text-spacing overrides, and a long German profile
label passed without horizontal page overflow or clipped measured text.
Pretext measured all 76 text regions with Arial parity. Enlarged text can
enter explicit readable reflow; no font shrinking masks a failure.
All twenty MP3s loaded real positive duration metadata. Silent actual playback
confirmed starting a second sample pauses the first. Missing Pretext and
JavaScript-disabled pages retain all twenty native players and downloads.
No ordinary JavaScript errors occurred. Physical-phone and perceptual listening
checks were not performed. The existing UI-state and context checks also passed;
the session scheduler and its audio assets were unchanged.

Public deployment of feature commit `1f3f5db` was verified on 2026-10-28.
GitHub Pages run `40730446610` succeeded and the Pages API reported `built`.
`/atmungsraum/samples/`, its catalog, and all twenty MP3 URLs returned HTTP 200;
all public audio hashes matched the generated originals. Live 390×844 Chromium
reported twenty players/downloads, ready Pretext measurement and no horizontal
overflow. Real keyboard Space play/pause and the MP3 download filename passed
with browser audio muted. The context checker with `-RequireRemote` passed.

## Browser prototype completion — 2026-10-13

The completion pass preserved the latest approved menu, baseline, playback-control,
and audio-asset decisions. The UI-state checker, all eight audio tests, context
checker, JavaScript syntax checks, Python syntax checks for both pending listening
helpers, and Git whitespace check passed.

The existing native-browser baseline check passed 41 rendered cases covering
main choices, both final screens, playback, pause/resume, 320 CSS px, phone
landscape, desktop, 200% text, text-spacing overrides, and long German text.
The current phone screenshot was visually inspected. Native OfflineAudioContext
checks passed both 600/900-second main sessions at 44.1/48 kHz: every baseline
sample was silent and all scheduled audio samples matched with zero error.

Release `d6af2f2` was pushed to `main`; GitHub Pages run `38460690617`
completed successfully for that exact commit. The live
<https://viscereality.org/atmungsraum/> was checked on 2026-10-13 at 20:16 UTC:
the six HTML/CSS/JavaScript entry files matched the checkout, the no-slash URL
redirected correctly, and browser checks passed the demo gate, both round
choices, four practice choices, instruction playback, and both main sessions'
baseline, pause/resume, stop, and timing receipts. No page errors or failed
site requests occurred. The remote context gate passed and the source tree was
clean after the release. Local QA output is in ignored `output/playwright/`.

The existing runtime samples remain selected; local listening revisions are not
promoted by this release. Real H10 acquisition, headset transport, and VR-event
execution are explicitly outside this verified browser prototype. No hardware
or acoustic-output verification is implied.

## Observed connected count phrases — 2026-10-12

Two complete ElevenLabs phrases were generated with the selected website voice,
request settings and character alignment. Independent PCM comparison proved that
every output word retains its native sample length and waveform, apart from gain
and short boundary fades: interior reconstruction residual was below 0.6 PCM
units and all speech tempo factors are one. Source takes are 4.24/4.48 seconds;
pause adjustment yields exact four-second phase clips. Actual word starts are
0/1.056/2/3 seconds for inhale and 0/1.164/2/3 for exhale. They are recorded exactly
in JSON rather than claimed as uniformly spaced counts. All 13 WAV/JSON pairs,
frame/hash/headroom checks and component/cycle reconstruction pass. The quiet
Halten! clips are byte-identical to the preceding set; breath gain remains fixed.
Each full cycle is exactly 16 seconds. Runtime assets remain unchanged.
Unprompted Whisper-small review recognized all ten requested words in the cycle.

## Observed quiet Halten! listening revision — 2026-10-12

One new cached ElevenLabs take used literal Halten! and the selected website
voice/settings. Both hold roles reuse the same 0.7527-second prompt PCM, followed
by 3.2473 seconds of exact silence. All 13 WAV/JSON pairs passed independent
frame/hash/headroom checks. Holds have one word slot and speech/quiet segments;
the full voice cycle has ten requested words. Unprompted Whisper-small recognized
those ten words with no hold counts. Ten component clips remain four seconds and
three cycles remain 16 seconds, with sample-identical sums/concatenation. Inhale,
exhale, voice/mix and breath samples are byte-identical to the preceding set.
Runtime assets remain unchanged; the previous listening package is preserved.

## Observed constant-breath/Halten listening revision — 2026-10-12

Thirteen adjacent WAV/JSON pairs passed independent frame/hash/metadata checks:
ten four-second clips and three 16-second cycles. Both hold files use exactly
the same cached German Halten/count PCM, with separate role metadata. Transcripts
contain no Pause. Voice counts still descend; voice + breath equals mix at every
sample and cycles concatenate exactly. Against the retimed source recordings,
the breath stems have a constant scalar throughout the interior: fitted gains
vary by less than 0.01% across seconds and reconstruction residual is under one
PCM unit. Only the 10 ms boundaries fade. Average component power remains matched
within 0.0001 dB; all audio has at least 3 dB sample headroom. Runtime assets remain
unchanged, and no new synthesis/API call was needed.

## Observed minimal UI — 2026-10-12

The final UI replaces three playback buttons with one Starten/Pause/Fortsetzen
button and folds Beenden into the existing back action during playback. Round
icons/descriptions, repeated final-plan text, ready/playback messages, and their
unused wrappers/styles are removed. Search retry is shown only for recovery.

All eight audio tests, the UI-state check, context/source check, JavaScript syntax
checks, and Git whitespace check pass. The controller check proves the same
button pauses/resumes the original schedule, leaves Pause usable during signal
recovery, gates Fortsetzen on current ECG, aborts via Beenden, and preserves the
receipt guard when cancellation interrupts startup.

Chrome rendered round/practice choices without page or element overflow at
320×568, 390×844, 768×1025, 1280×720, 1920×1080, 568×320, and 844×390 CSS px.
The main choices also passed at 356×631, 434×938, 853×1138, 1422×800,
2133×1200, 631×356, and 938×434 CSS px. Actual CSS sizes were read from the DOM
because this Chrome profile uses 90% zoom. Pretext reported ready and full labels
fitting. At 320×568 with 200% root text, the main choices use deliberate vertical
document reflow with no horizontal overflow; the final Starten/Pause/Fortsetzen
screen remains bounded without overflow. Button focus persists through the
playback label changes. Physical phone/headset checks remain pending.

The final screen also passed at 1280×720, 1920×1080, and 568×320 CSS px,
including paused playback in landscape. Longer German choice labels and WCAG
text-spacing overrides at 320×568 wrap/reflow without clipped text or horizontal
overflow. Beenden was observed returning to Haupt-Runde with sessionActive=false.
These are rendered DOM/text bounds checks; browser screenshot capture timed out,
so this pass includes no saved pixel capture.

## Observed main-round normal-breathing baseline — 2026-10-12

All eight Node audio tests, the six-route UI-state checker, context/source
checker, JavaScript syntax checks and Git whitespace check passed. Main choices
are 10/15 minutes total, with five normal-breathing minutes followed by the
existing five-/ten-minute guided sequence. Tests prove that no source starts
before t0 + 300 seconds, the welcome begins at that boundary, the first cycle
starts at t0 + 340 seconds, and pause/resume preserves this same clock/plan.
Receipts distinguish total, baseline and guided duration.

Isolated headless Chromium checked the main menu and both ready/playing leaves
at 320×568, 390×844, 768×1024, 1280×720, 1920×1080, 568×320 and 844×390 CSS px.
All ordinary page/text bounds fitted, without horizontal/vertical page overflow
or an inner scrollbar. Both leaves displayed Normal atmen, had exact 600/900-second
audio spans, paused/resumed the baseline countdown, and saved the correct receipt.
At 320 CSS px, 200% root text, text-spacing overrides and a long German hint
passed full-text/no-horizontal-overflow checks using deliberate document reflow.
Restricting the two-choice width rule to bounded layout fixed the enlarged-text
overflow. Real phone/headset tests and physical acoustic timing remain pending.

Native OfflineAudioContext renders through `tests/offline-audio.html?round=haupt`
passed for both totals at 44.1 and 48 kHz: exact 600/900-second lengths, every
baseline sample zero, and all later samples matching the shifted reusable
assets with zero maximum sample error. These checks use the production scheduler.

## Observed selected ElevenLabs listening set — 2026-10-12

The user rejected the OmniVoice set and selected the existing website's British
Breathing Instructor. Saved profile name/ID and Eleven v4 German TTS support
were confirmed through the API. Only Halten was newly synthesized; website
inhale/exhale/counts and cached Pause were reused. Thirteen adjacent WAV/JSON
pairs passed independent format/frame/hash/transcript-slot checks: ten exact
four-second clips and three exact 16-second component cycles. Voice fades and
breath rises; powers agree within 0.0001 dB during inhale/exhale; all samples
have at least 3 dB headroom; component sums and concatenation match exactly.
Unprompted Whisper-small recognized all sixteen words. Runtime assets remain
unchanged. The rejected 26-pair listening set is preserved in the local archive.

## Observed Start/Pause controls — 2026-10-12

All seven audio tests, the UI-state checker, context/source checker with the
remote gate, JavaScript syntax checks, and Git whitespace check passed. The new
engine test proves that pause/resume retains the original sources and audio-clock
anchors, prevents a second start, records transitions, and leaves a cancelled
resume aborted. Controller coverage checks button visibility and enabled states.

Native browser playback showed the main start button followed by a smaller
Start/Pause row. Pause held the phase and countdown across repeated size changes;
Start resumed with identical start/end anchors and a moving countdown. Stop
removed the smaller controls and enabled a fresh main start. No document overflow
was observed at 320×568, 390×844, 768×1024, 1280×720, 1920×1080, 568×320, and
844×390 CSS px. After giving the landscape countdown five character widths,
Pretext reported every visible final-stage label fitting at 320×568 and 568×320.
At 200% root text and 320×568, all labels fitted in deliberate document reflow,
with no horizontal overflow or inner scroll container. Physical headset/VR-event
pause synchronization remains unimplemented.

## Observed listening asset packaging — 2026-10-12

The earlier package, now in `audio-assets/archived-omnivoice/`, contains 26
same-folder WAV/JSON pairs across
female/male sets: 20 four-second phase/component clips and six 16-second voice,
breath and mix cycles. Independent reading checked formats, exact frames,
transcripts/slot counts, authored timing, matching hashes, voice + breath = mix
at every sample, and exact cycle concatenation including silent breath holds.
Generated files are ignored; runtime assets remain unchanged. Folder opening
and subsequent user listening are separate from these sample checks.

## Observed counted hold variants — 2026-10-12

Two restrained voices now have voice-only and voice-and-breath auditions with
counted Einatmen/Halten/Ausatmen/Pause. Independent finished-PCM reading checked
all sixteen four-second phase WAVs (176,400 frames at 44.1 kHz) and four 16-second
montages (705,600 frames). Sample peaks have at least 3 dB headroom, boundaries
and beat tails are quiet, voice beats descend, breath beats ascend, inhale/exhale
component powers differ by less than 0.0001 dB, holds match between mix versions,
and montages concatenate their elements exactly. ZIP contents match the local
files and pass integrity checking. Runtime asset hashes still match the manifest.
Source Pause takes passed the native speech check; saved Halten/counts were reused.
Final unprompted Whisper-small review recognized all sixteen requested words for
both cycles. Preserving 30 ms before Pause's speech threshold corrected the male
cycle's initial recognition error. Listening judgment remains a separate check.
Human acceptance and clinical release remain pending.

## Observed voice-to-breath prototype — 2026-10-12

Two restrained profiles produced standalone Halten in one take each with zero
source word edits. Four phase files per voice and 16-second montages were rendered
locally. Independent reading of the finished PCM confirmed exact 176,400-frame
phases, 705,600-frame montages, correct hashes, at least 3 dB sample-peak headroom,
descending voice and ascending breath per-second RMS, component power differences
below 0.0001 dB, quiet hold tails, and sample-identical montage concatenation.
The current browser manifest and asset hashes remain unchanged. Human mix/voice
acceptance is pending; no runtime or clinical deployment is claimed.

## Observed calm German voice sourcing — 2026-10-12

The original LANG XLS workbook was read with xlrd in an isolated ignored tools
directory. All 240 recordings were screened against their auditory norms; none
met the provisional joint positive-valence ≥6/arousal ≤3 threshold after the
explicitly inferred valence-polarity reversal. Raw scores and source hashes are
retained. EmoDB's neutral speaker 13 and 15 references use confidence and
naturalness ≥0.8; numeric calm/high-valence norms are not claimed for them.

Secret Tunnel MCP calls returned internal errors; the installed app's documented
local CLI reported model/ASR ready, created two profiles, and generated identical
German welcomes. Both passed its Whisper word check in one take with zero edits.
Strict MP3 decoding passed. Matched-level review WAVs and full receipts remain
ignored/local. Model CC-BY-NC terms and unestablished speaker-clone consent mean
these checks do not establish unrestricted clinical/commercial release rights.
The user reviewed both and found them too robotic; calm/warm acceptance failed.
Positive reference auditions from female speaker 8 and male speaker 3 also
completed in one take with zero recognized word edits and strict MP3 decode
success. Review WAVs match -24 dBFS RMS and preserve native model timing. They
were rejected by the user as too energetic. Two restrained profiles then
generated welcome and connected counted-cycle speech. Four separate four-second
phase files per candidate and a 16-second review montage passed frame/hash checks;
unprompted secondary ASR recognized the entire requested sequence. Their listening
acceptance remains pending. Publisher metadata lists
CC-BY-4.0 while archive metadata lists CC0: preserve both and attribution.
See `audio/voice-sourcing.md`.

## Current soft-v2 audio checks — 2026-10-12

The current nine-asset manifest uses a 40-second welcome, four separate four-second
phases with Einatmen/Ausatmen plus counts and Und halten plus quiet, a 12-second ending, a 20-second
rest, and two four-second breath-overlay variants. Exact total 5/10/15/20-minute
protocols have 15/33/50/68 whole cycles, 0/1/2/3 rests, and 8/0/8/0 seconds recovery.
All six Node audio tests, the UI-state checker, the context checker, and Git
whitespace check pass after the quiet-hold update. The extra controller regression
test proves that cancelling a new audio unlock cannot save the preceding run's
receipt under the new scenario.

Native offline renders were repeated after final Einatmen/Ausatmen/Und halten regeneration:
every output sample matched at 44.1 kHz for all four totals and at 48 kHz for five
minutes, with zero maximum sample error and exact total frame counts.
Actual browser Play/Stop was verified for all four durations, with 300/600/900/1200
second audio-clock spans. Stop/Escape/back navigation and 20-minute double-click
protection were observed. A complete 68-second instruction example reached its
native completed/replay state. Playing phase/count/time and real Pretext fit were
observed at 320×568 CSS px with no document overflow. The current regenerated
sample set was loaded and started/stopped again after reloading. See
[PLAYBACK-AUDIT.md](PLAYBACK-AUDIT.md). Full target-device,
clinical listening, physical onset/synchronization, and final UI resize checks
remain pending. Historical v1 evidence below refers to the preserved v1 set.

## Context and source

From `atmungsraum/`, run:

```powershell
./for-ai/scripts/check-context.ps1 -ProjectRoot .
```

The current checker verifies the guidance, AI document set, four practice/two
main choices, black background, local artwork/runtime assets, and locked Pretext
version. It does not verify an H10 connection, ECG, or stretch-layout behavior.
After pushing the validated change to `main`, add `-RequireRemote` to verify that
the local repository HEAD matches the remote publishing branch.

## H10 gate and ECG acceptance

When implemented, verify the first screen is the H10 gate, including direct links
to session branches. Exercise no device, connecting, verified connection, missing
ECG data, and disconnect/data interruption. Session actions must not appear ready
before connection verification, and stale data must not appear live.

Use a real Polar H10 to verify the waveform comes from its ECG samples and is
displayed in a centered box. Test the chosen connection path on each supported
smartphone/browser. Mock data may check rendering and state transitions but cannot
qualify the sensor connection or live ECG. Record those hardware checks as pending
until observed.

## Stretch-accordion menu acceptance

Verify every menu function, including the gate and ECG view, against the installed
`uncodixfy-pretext` Pretext and accordion-stretch references. Check phone portrait
and landscape, desktop, 320 CSS px reflow, 200% text/zoom, long/localized labels,
user text-spacing overrides, focus order, visible focus, and touch controls.

Test minimum width with tall height, minimum height with wide width, both minimum,
expanded dimensions, and both sides of every breakpoint. Check header/footer
anchors, ordered control groups, centered ECG bounds, non-overlap, and growth of
group/cell spacing as space increases. Inspect panel, status, and document bounds:
the bounded menu has no internal scrolling, and a full-window panel must not make
the page scroll. Hidden overflow is not evidence of fit.

Verify real Pretext measurements with loaded matching fonts, readable no-fit and
unavailable fallbacks, and rendered DOM parity. Record the tested size/browser
range and unresolved failures; do not infer a pass from package presence.

## Observed four-choice UI tree verification — 2026-10-12

The later minimal UI replaces the anatomical images with sprout/wave symbols,
removes global context and the ECG return button, and adds measured one-sentence
exercise descriptions. Its exercise choice passed bounded page/text checks at
all seven sizes below. Narrow portrait tiles grow taller to fit the descriptions.
At 320 CSS px and 200% text, the final copy reflows without horizontal overflow
or clipped text. The new compact recording gate passed at 320×568, 390×844,
1280×720, and 568×320; refresh hides/clears ECG and changes Continue to red and
disabled, while recording makes it green and enabled.

The pure state check proves retries every five seconds, preserving the current
flow through exactly 30 seconds, expiry strictly after 30 seconds, and recovery
cancelling retries/deadline. A browser fixture also verified temporary loss stays
on the exercise screen, emits a retry, and returns to ECG only after an injected
expired deadline; Continue is then disabled and the old trace absent. These are
emulated signal-state tests; physical reconnection remains unverified. Runtime
phase/time/stop controls are shown only during playback.

The browser-local tree was checked in the Codex in-app Chromium browser at
320×568, 390×844, 768×1024, 1280×720, 1920×1080, 568×320, and 844×390 CSS px.
The connected ECG gate, two round choices, both four-box menus, and the final
play screen passed page/text bounds checks at these ordinary sizes. Header and
footer anchors remained visible; panels had no internal scrollbar or page
scroll. Both round choices stayed square and visible together.

All eight choices routed to the matching final play screen; Anleitung carries
no duration. Browser back/forward and heading focus passed. Connecting, waiting
for ECG, synthetic ECG reception, missing ECG, disconnect, and recovery were
exercised. Missing ECG/disconnect cleared the trace and set Continue disabled.
The Node self-check verifies all valid leaves, cold deep-link gating, invalid
routes, signal loss, reset, and duplicate-start protection. Run it with:

```powershell
node for-ai/scripts/check-ui.mjs
```

At 320 CSS px, long German labels and the WCAG text-spacing override passed
rendered text/page-width checks. At 200% text enlargement, readable document
reflow with one-column choices preserves full labels without horizontal overflow
or an inner scrollbar; vertical page scrolling is deliberate outside the bounded
normal-size matrix. Pretext predictions use loaded Arial and remain marked ready;
text-spacing parity is assessed in the DOM because word spacing is not modeled.
A missing Pretext module switches to measurement=unavailable and readable flow
while retaining the decision tree. No-JavaScript behavior retains the initial
unverified gate. Real phone browsers and physical H10/headset tests are pending.

A concurrent audio task connected the final play action to browser audio during
this UI pass. The initial final screen hides runtime phase/time/stop controls
until playback starts; preserve the audio task's engine and controller. The
above UI checks establish navigation/layout, not acoustic timing or real headset
execution. See the audio-specific verification below for that separate scope.

## Earlier radio-menu verification — 2026-10-12

Windows desktop Chrome rendered the start at 320×568, 390×844, 768×1024,
1280×720, 1920×1080, 568×320, and 844×390 CSS px. Both choices remained square
and visible together with no horizontal page overflow or clipped text. Probe
branch text bounds were checked at 320×568, 390×844, 1280×720, and 568×320;
the main branch and every leaf choice were exercised at 390×844.

At 320 CSS px, 200% text enlargement, text-spacing overrides, and a longer
German start label passed rendered text-bound/page-width checks. Both branches
also passed the 200% text page-width check. Text enlargement and short landscape
branch views use vertical document flow where needed; full bounded/no-page-scroll
stretch acceptance remains pending. No inner scroll container or hidden overflow
is used to conceal text.

Round selection, every probe/main radio option, confirmation output, Zurück,
browser history, heading focus, keyboard radio selection, JavaScript-disabled
navigation/selection, and the Pretext-unavailable fallback passed. Runtime
measurement reached `ready` using the locally shipped package; ordinary checks
reported no JavaScript errors. Reduced-motion preference was also exercised.
Source/context and Git whitespace checks passed. No publication or H10/headset
hardware checks are part of this round-menu validation.

## Observed audio cue verification — 2026-10-12

All five `assets/audio/breathwork/de/*.mp3` files fully decoded with FFmpeg.
FFprobe confirmed MP3 audio at 44.1 kHz with positive durations; the combined
duration is 13.885 seconds and the combined file size is 227,587 bytes. Every
file matched its recorded byte count and SHA-256 hash. Measured peak levels
ranged from -2.5 to -1.8 dBFS; each file contained non-silent audio.

The exact German texts and generation settings are recorded in the adjacent
`cues.json`. Pronunciation, accent, and subjective voice quality have not been
listened to and remain pending. No playback scheduling, UI integration,
publication, or headset audio check is part of this asset verification.

## Observed reusable box-breathing verification — 2026-10-12

Both supplied YouTube reference tracks were downloaded as AAC/M4A and decoded
with FFmpeg strict error checking. Native German/English captions were extracted
and analyzed locally; these caption timings are not acoustic onset measurements.

All six reusable WAVs have exactly 176,400 mono 16-bit PCM frames at 44.1 kHz,
four seconds each. Hashes and byte counts matched the asset manifest; each file
contains speech with silent boundaries and sample peaks below full scale.
Five Node tests passed: exact complete-cycle protocols/remainders, actual sample
integrity, common-clock scheduling, cancellation/interrupted contexts/stale
callbacks, and corrupt-asset rejection.

The production scheduler was rendered with native Chromium OfflineAudioContext.
Every output sample matched the expected repeated blocks with zero maximum
difference for 5/10/15 minutes at 44.1 kHz and 5 minutes at 48 kHz. Full PCM
verification exports also matched their sample-indexed protocols, including
silent recovery, at exactly 300, 600, and 900 seconds. Runtime playback reuses
the six samples; concatenated exports are local verification artifacts only.

Local Whisper transcription recognized the four phase prompts and ending;
the compact instruction's short words were joined by ASR. Human listening remains
pending. No realtime phone/headset acoustic timing, H10 synchronization, host
receipt persistence, menu integration, or publication is claimed by these checks.

## Deployment

Prepublication check on 2026-10-12: the UI-state check, all six audio tests,
context/source checker, and Git whitespace check passed. GitHub Pages source was
confirmed as the existing `main` branch, root `/`, with `viscereality.org`.
The user authorized publishing this navigation prototype and browser audio.
Physical H10/headset acquisition and the VR event sequence remain pending.

- Verify the GitHub Pages build reports `built` for the published commit, without
  a build error.
- Request both `/atmungsraum` and `/atmungsraum/`; a redirect from the first to the
  second is normal for a directory index.
- Confirm the final response is HTTP 200 and contains the expected page source,
  then inspect the public page in a browser.
- Verify the deployed `atmungsraum/for-ai/README.md` is reachable.

Report failed or pending checks explicitly; do not infer deployment from a push.

## Observed local verification — 2026-10-09

The context/source checker and Git whitespace check passed. Browser checks at
1280 × 720 and 320 × 568 CSS px showed an empty body with zero child elements,
`rgb(0, 0, 0)` as the root background, and no horizontal or vertical overflow.
Both rendered views were visually solid black. Deployment must still be checked
against the published commit using the procedure above.
