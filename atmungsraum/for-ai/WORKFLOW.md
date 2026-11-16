# Workflow

## Current boot/update checkpoint — 2026-11-16

Actual Meta-installed15 is confirmed by exact APK digest and Meta installer/
initiator, with retained UID. The separate same-signer boot helper is installed,
overlay Allowed and enabled by the user. Genuine reboot automatically launches15:
trusted native cold admission, first submitted frame, OpenXR focus and advancing
frames are observed. The wearer confirms single-menu recovery and protected
three-departure escape; native terminal/task-removal/retired-owner logs corroborate
escape. Helper27policy checks, signed package inspection and independent review pass.

Meta16 was published with the Quest off. Processing completes, malware passes
and security details say all tests passed. The next genuine boot still launches15.
A69-sample window over about208seconds observes no16; available version readbacks
are15 with validated Wi-Fi and charging;
later closed/asleep readbacks remain15. This is a bounded negative result, not
proof Meta updating never works or that kiosk causes it. No host install, host
app launch or manual Update established the test result. Both actual experiment
versions must continue through Meta distribution. Phone control stays with the
user, Location stays ON; normal main updates do not require helper updates.

The user now approves an official in-app startup check/download/install/relaunch.
Use the same [active contract](BOOT-KIOSK-UPDATE.md) extension tasks5–9. Native
Platform SDK207 has a16KiB-aligned ARM64 loader; implementation is in progress,
not built or hardware-qualified. Preserve signer, data/trust, private visuals,
controller-only policy and audio/calibration math. A shared native maintenance
fence must protect calibration, preparation and recording. SDK acknowledgement
is not installed-version evidence. Overall update acceptance remains open.

Verified split source: PR27 merge `4bae41fa3c0d4d30a10996e4e2c569419eefab38`,
exact Pages43921087088 success and18 public files byte-match Git. Public policy
now advertises available16/minimum11; installed15 is still compatible. The new
hardware-evidence/policy increment and SDK implementation await publication.
Historical checkpoint notes below describe their observation time.


## Operator layout qualification — 2026-11-16

The [layout contract](UI-LAYOUT.md) preserves native/domain authority while
splitting preparation, tracking, running, recordings and settings. Use the real
app/gateway fixture and actual Pretext, not static mockup HTML. Check configured
viewport width (mobile overflow can expand `innerWidth`), painted text/control
bounds, header/body/footer separation,48px targets and document reflow. Preserve
six-poll scroll/editor/caret and zero-command local-navigation checks.

With the fixture's paused browser clock, Chromium full-page/captureBeyondViewport
screenshots can paint a stale footer position after measurement/viewport changes,
even when actual viewport paint and before/after DOM regions are correct. Use
actual configured-viewport screenshots and scrolled viewport tiles as visual
evidence, restoring scroll afterward. Do not change production CSS or suppress
geometry failures to accommodate a capture artifact. Native sample cadence and
command assertions remain separate deterministic checks.

## Current hardware-test handoff — 2026-11-16

The user defers new-mode testing. Current focus is physical yellow-circle/
percentage stepping under Automatic, diagnosed read-only on the connected Pixel.
Actual current assets are loaded; blue126px/gold122px is the requested touching
inset. Uneven arrivals truncate the original composite blends. The corrective
PWA uses100ms coalesced preview-only reads and bounded280ms adjacent GPU-sample
playout; endpoint expiry stays500ms. PR22/Pages43877648195 and27 live/public-render checks pass. Exact same-phone
confirmation is pending because the user no longer has the phone available.
Read-only observers ended; owned9226 CDP forward and phone/port leases are released.
Keep phone control with the user. Wait for their return before requesting refresh
or readiness; do not switch modes or calibrate. APK11/native remain unchanged.

Code11 (0.2.10-dev) is installed with exact byte/version readback and retained
UID after the user confirmed no recording. Trusted MAIN/LAUNCHER cold start and
Android activity focus are verified. Controller-only is true, live hand and
simultaneous tracking false; initial driver0. Automatic H10 candidate admission
is observed; signal quality requires the phone/wearer check. Location staysON.

Matching PWA is live through PR20/Pages43848286245 with27 live-file matches and
public phone/desktop checks. The user refreshed/reconnected, took control and
confirmed0.2.10-dev/code11 in the phone version display. They retain phone control.
Fresh wearer Ready is requested before fixed/adaptive mode/display acceptance.
No agent calibration/recording or Meta upload occurred. The finite observer,
owned ADB child and all owned build/device/lifecycle leases are released. Keep
requested Wi-Fi ADB and LocationON. See PROJECT for remaining physical checks.

The user explicitly requests retaining system Location ON on the test Quest
after the attended scan/calibration work. This supersedes the earlier temporary
restore-to-OFF bench instruction. App code must still use normal attended
permission/settings flows rather than toggling the system setting itself.

The latest user preference is to control the phone themselves. Earlier
“try yourself” permission applied to that diagnosis and does not override this
later preference. Observe read-only unless the user renews phone-operation
authorization. After a test pause, require fresh wearer readiness
before starting calibration. Announce actual collection only after an accepted
native calibration has begun advancing completed frames. A running Automatic
loop or a waiting-for-cycle state does not establish collection.

Keep iterative Developer APKs local until relevant hardware acceptance. Preserve
the existing signing key and memberships. Record installation/hash and basic
startup separately from visual parity, worn calibration and Meta delivery.
When diagnosing startup, begin a bounded filtered live log capture before the
launch: busy renderer output can displace the startup interval before a later
log dump. Do not clear shared buffers or change global logging to compensate.

### Launch through the trusted launcher path

Resolve the selected APK's launcher component, then use the normal explicit
launcher intent for an authorized launch:

```text
adb -s <serial> shell am start -W -a android.intent.action.MAIN -c android.intent.category.LAUNCHER -n <package>/<activity>
```

A bare `am start -n` can display the stimulus while failing guarded launcher
admission. The code-9 test launch did this: startup logged rejected with
`launchEpoch=0`, so its trusted automatic-connection and kiosk initialization
were skipped. Do not weaken the production guard to accommodate that malformed
test command. A running process, ready gateway and closed panel do not prove
trusted startup. Capture admission and effective automatic-connection/kiosk
readback separately. Test the actual Quest launcher intent too when available;
do not assume its categories from a generic host fixture.

### Build identity, boot selection and update delivery

The 2026-11-16 private-channel upload of boot-enabled code13 was rejected by
Meta for unsupported `SYSTEM_ALERT_WINDOW`. The existing boot feature requires
that permission. Both versions in the user's update experiment must be installed
through Meta; local12 only served as a packaging smoke-check. Do not count a
local Dev Hub/ADB installation or a manually selected Update as automatic Meta
delivery, spoof installer attribution, or remove a prerequisite while claiming
the feature works. The user approved a separately provisioned one-time helper;
the main Meta builds omit internal boot/overlay authority. Qualify the actual
split setup and real boot before claiming combined support; see
[the current boot/update contract](BOOT-KIOSK-UPDATE.md).

Read actual package version and generated packaging fingerprint through the
authenticated APK status. Keep `quest-release.json` minimum/latest versions and
the APK record current together; verify installed versus required on the phone.
The identifier is not the signed APK byte hash. Unknown metadata or failed
policy reads cannot be presented as an up-to-date result.

Code 9 and code 10 do not register the optional system-boot receiver. Normal
trusted cold launch and soft kiosk are separate from power-on autostart. Do not
claim boot qualification from a successful manual launch. The existing optional
receiver uses MAIN/LAUNCHER and requires its selected feature and user-granted
overlay permission; changing that selection needs its own verified increment.

Meta release-channel membership and delivery are separate from local APK
installation. Eligible channel members receive updated builds, but do not
guarantee installation before an immediate app or boot launch. Use an idle update
window and verify the installed version before a session. See Meta's
[channel membership](https://developers.meta.com/vr/resources/publish-release-channels-add-users/)
and [headset update guidance](https://www.meta.com/help/quest/172903867975450/).
Keep current iterative APKs local until relevant hardware qualification; no
automatic Meta upload is authorized by a version-indicator change.

Custom private-particle assembly requires applying its approved
`experiment.env.json` to the child process for both initial resolution and
assembly, then restoring the caller environment. A feature lock alone does not
supply those caller environment inputs. Retain failed run logs and use a fresh
run name; never mistake a partial failed staging directory for a built APK.

## Normal development

1. Read the root guidance and this control plane; inspect `git status` before edits.
2. Keep the local checkout sparse and the fetch blob-filtered. Select only this
   subproject and the necessary root guidance; do not clone the full media-heavy
   repository, run `git sparse-checkout disable`, or fetch large assets needlessly.
3. Base edits on the current remote `main`. Preserve unrelated changes and files
   outside the sparse checkout; absence locally does not mean deletion remotely.
4. Keep product code outside `for-ai/`. Update these notes and the parent
   `for-ai/subprojects.md` when the page contract changes.
5. Run `for-ai/scripts/check-context.ps1 -ProjectRoot .` from `atmungsraum/` and
   the rendered checks in [VERIFICATION.md](VERIFICATION.md).
6. Follow the user's standing 2026-11-05 Superpowers/GitHub instruction in the
   [root constraints](../../for-ai/constraints.md). Complete relevant design,
   implementation, review and verification practices; update For-AI evidence;
   review/stage intended paths, commit and push completed verified work through
   existing `origin/main` without another permission prompt. The 2026-11-06
   extension requires publishing each completed verified increment as soon as
   it is ready, rather than waiting for the end of a larger task. Fetch/reconcile first
   and preserve unrelated unfinished work. Local source recordings, voice
   auditions, listening copies, and full-session exports stay ignored.
   Never force-push or bypass branch protections.
7. After publication, run the checker with `-RequireRemote`, check the GitHub Pages
   build for the pushed commit, and verify the public URL. A successful push alone
   is not evidence that the page is live. Keep following the exact run in this
   task until deployment succeeds and affected live assets/behavior are checked;
   do not hand off with only a pending build. Report concrete blockers accurately.
8. After every live Atmungsraum PWA release, refresh or relaunch the installed
   app on the lab Pixel 10 and verify the deployed version on the phone.
   This is part of release completion. Follow the device check below.

This standing authorization applies to future tasks in this repository unless
the user asks for local-only work. Before handoff, verify Git synchronization,
record exact publication evidence and remaining changes in PROJECT.md, and
report any concrete blocker instead of leaving completed work silently local.
The live target is <https://viscereality.org/atmungsraum/>. Documentation-only
updates follow the same prompt push/deployment rule, with scoped context/diff
and published-file checks rather than unrelated runtime test reruns.

If Pages times out and a fresh build is rejected because that same deployment
is still in progress, clear only this task's timed-out deployment. Inspect
`gh api repos/GeorgeFejer91/viscereality.org/pages/deployments/<old-sha>`;
use POST on that path's `/cancel` endpoint, verify `deployment_cancelled`,
then rerun the latest build's failed deployment job. A failed Actions run or
an immediately cancelled retry does not guarantee the Pages backend released
its deployment lock. Preserve unrelated active deployments. The existing
large-artifact warning alone is not evidence of the timeout's cause.

Local setup shortcuts and `LOCAL-PROJECT.md` are excluded by `.git/info/exclude`.
Do not publish them or machine-specific paths.

## Quest APK changes and releases

The user's 2026-11-15 correction requires testing iterative APKs locally on the
headset before Meta Developer submission. Do not upload every test build to the
Dev Hub; hand off a qualified release after the relevant hardware checks.
APK startup must request required runtime permissions using the attended
platform flow even when its app panel starts closed. A one-off permission
pre-grant is only a bench exception when explicitly authorized by the user.

Follow the standing Superpowers and GitHub workflow above for every change
that affects a Quest APK, including native code, assembly, features, toolchain
or dependency pins, assets, and packaging-only changes.

1. Identify the last verified release in the [Quest APK log](../quest/apks/README.md).
   Inspect the native and website Git state. Review the complete change
   against that release, including selected build features and asset changes.
   Commit each coherent native source change with a descriptive message.
2. In Atmungsraum, commit the corresponding public native patch and any
   affected assembly, public assets, or browser code. A native commit in a
   separate checkout is only one part of the record. Add the version and
   version code, native commit and tree, upstream base, concrete changes
   from the previous release, build recipe or changed pins, APK byte count
   and SHA-256, and verification evidence to the APK log. Update
   [PROJECT.md](PROJECT.md) with the current status and next step.
3. Run the focused checks for the affected behavior. Confirm the signed
   package, ARM64 loader, signer, and installed APK checksum on Quest 2.
   Record launch and the relevant physical acceptance checks. Mark any
   checks still pending. If a change crosses into the Pixel PWA, record
   the tested browser release with the APK.
4. Review and stage only the intended files, run the context checker, then
   commit, push, and follow the exact Pages deployment. Publish signed APKs
   under `quest/apks/` when their bundled assets are authorized for public
   distribution. For the public build, keep the selected source assets and
   SDK library notices in `quest/build-assets/` and verify the bundled
   build from clean pinned checkouts. Update the public asset inventory
   whenever the release changes those files. Keep signing material,
   unpublished provider inputs, and raw participant data in the lab.

## Local UI workflow

For a numbered smartphone review pack, use the
[phone screenshot protocol](PHONE-SCREENSHOTS.md). Its Chromium fixture
keeps hardware and account requests local.

Serve `atmungsraum/` with a local static HTTP server (for example,
`python -m http.server 8876 --bind 127.0.0.1`) and inspect the page over HTTP.
Run `node --test tests/quest-rehearsal.test.mjs tests/quest-data.test.mjs` for
controller and transfer checks, then verify the rendered entry and recovery routes. Local QA fixtures and
screenshots go in ignored `output/playwright/`.
There is no site build step. The committed `vendor/pretext/` files are the
deployment runtime; `node_modules/` is ignored.

Reload after edits. If the browser serves stale modules/styles, update the local
`app.js`/`style.css` version query in index.html; keep both tokens aligned. This
does not change the static runtime or require a build step.

When restoring/updating the measurement dependency, run
`pnpm install --frozen-lockfile` and `pnpm run vendor`. Keep the exact version,
lockfile, copied runtime modules, and license aligned, and rerun rendered checks.
The vendor script copies only the main API's runtime module graph and license.

## Smartphone PWA

George's standing instruction of 2026-11-13 includes updating the installed
Atmungsraum app on the lab Pixel 10 after each deployment that changes the PWA.
The installed app loads the live website. Once the release is live, target
the authorized Pixel over USB and reload its standalone Atmungsraum page
through Chrome DevTools, or reopen it from the home-screen icon. Reload when
the app is idle and preserve its site data, saved Quest pairing and recordings.
Confirm standalone mode, the expected screen, and that the loaded versioned
runtime assets match the release. Record the deployed commit and phone
readback in [PROJECT.md](PROJECT.md). If the Pixel is unavailable, locked, or
in use, record the blocker and leave the device check pending.

`pwa-install.js` owns the permanent install action and browser prompt lifecycle.
Its manual help covers Android, iPhone and desktop Brave, including embedded
browser recovery. Hide the action on `appinstalled`, in standalone mode, or
when the browser returns the matching web app through `getInstalledRelatedApps`.
Keep the manifest self relation aligned with the canonical app ID and URL.
Its modal must preserve playback on Escape/close and restore focus. All visible
install labels/dialog text use the existing Pretext measurement owner. The dock
observes its actual height for zoom/reflow clearance; short landscape keeps the
install icon and folder outside the central content column.

Run native desktop installation tests in a separate OS account or VM. A
temporary browser profile still shares the Windows taskbar. DevTools
`PWA.install` can leave a pin whose icon file disappears on uninstall. The
local `output/playwright/check-brave-install-ui.mjs` checks the manifest, icons,
installed-state UI and help dialog without invoking a native install. The older
native install probes now require `ATMUNGSRAUM_QA_ISOLATED_OS=1` for that step.

The primary delivery target is an installed smartphone PWA. Keep
`manifest.webmanifest` linked from the entry page, with stable `/atmungsraum/`
ID, start URL and scope, standalone display and local PNG icons. Native browser
installation supplies the home-screen launcher; the app currently requires a
network connection and has no service-worker cache. Do not add offline session
or background-recording behavior implicitly.

The four app icons derive from the root site's `Images/logo.png` (Git blob
`bcd94ca74f6305eb7cd2e8a9a8b4a66cf99f6caa`): composite the unchanged, contained
logo on an opaque black square, preserving its aspect ratio. Ordinary 192/512px
and Apple 180px icons use 86% maximum logo height; the separate 512px maskable
icon uses 66%, keeping all artwork inside the central 80%-diameter safe circle.
If the source logo changes, regenerate all four together and inspect them.

Verify manifest parsing and browser installability over HTTPS or localhost,
icon decoding/dimensions, canonical launch URL and standalone display. Serve the
repository root for local PWA checks so `/atmungsraum/` resolves as published.
Check portrait/landscape and 320px reflow; keep physical Android/iPhone install,
relaunch, storage and lifecycle qualification distinct from desktop automation.
On Android Chrome, use the browser menu's install action; on iPhone Safari,
use Share → Add to Home Screen and enable Open as Web App when offered.

## Participant data and Polar H10

Use [DATA-LOGGING.md](DATA-LOGGING.md) for participant identity, filenames,
and CSV fields. The controller confirms fresh ECG and accelerometer streams
before participant Arm. Quest records and writes one CSV per run. The phone
verifies its SHA-256, keeps it in IndexedDB, and delivers the same CSV to
the private Drive folder. Quest's original streams support checkpoint recovery.

Verify raw ECG and accelerometer row counts, sensor timestamps, applied
radius values, pause timing, and the CSV hash on Quest, phone, and Drive.
Run the focused native recording suite, Node tests, context checker and
rendered checks for touched routes. Keep participant exports and checkpoints
outside Git. The retired browser recorder's completed files are available
for read-only download at `#daten`.

## Audio cue generation

### Exact TXT scripts and wording approval — 2026-11-06

The user requires text first, then explicit approval before creating,
regenerating or replacing audio. Prepare the concrete per-asset scripts for
review and wait. Approval of an earlier script does not authorize a later
wording change. This approval gate takes precedence over the standing automatic
publication rule for unapproved narration revisions. The later explicit request
for a working UI authorizes silent placeholders and labelled draft TXT/JSON/CSV
in scenario folders, plus their normal verification/publication. This exception
does not approve final wording or speech synthesis. Wording approval authorizes the normal
generation, verification and standing publication workflow; do not ask again
for the same approved step.

Every WAV/MP3 in the asset library has its own adjacent, same-basename `.txt`,
including reused cues, older retained sets, breath-overlay variants and published
listening samples. New or revised assets must carry their TXT in the same change.
Use UTF-8/LF, preserving the exact words, punctuation and German characters;
end nonempty text with one newline. A fully silent or breath-only asset has an
empty TXT. Exception: silent UI placeholders carry their intended draft script
in TXT and `text`, with `spoken_text: ""`, `placeholder: true`,
`audio_status: silent-placeholder` and a pending script status in JSON/CSV.
Do not put filenames, timestamps, delivery tags, stage directions,
review notes or silence explanations inside the spoken-word TXT.

Keep the approved TXT, JSON transcript, generated CSV and actual recording
consistent. Before approval, a proposed TXT may differ from the currently
published audio/JSON; record that pending status in PROJECT.md and the existing
audio-review folder. Do not regenerate audio or update its authoritative
transcript/hash to make an unapproved proposal appear current.

Participant-facing narration/UI must not call the preparation “Atemkalibrierung”
or “Atmungskalibrierung”. Use plain wording such as “Vorbereitung” and “Zwei
tiefe Atemzüge”. Technical calibration identifiers, timing events and research
references remain valid internal names. The pending Instruction 1 proposal
begins “Wir beginnen mit zwei ruhigen, tiefen Atemzügen.” Its audio is unchanged
until approval. See `audio-review/04-AUDIO-TEXT-APPROVAL.txt`.

2026-11-06 calibration exception: the user requests Secret Tunnel for the new
two-breath Instruction 1 narration, using its saved `Atmungsraum - Existing Instructor`
clone. `python tools/build-calibration-cue.py` uses the documented native voice
CLI when the MCP connector is unavailable, without altering access settings.
It caches native receipts/MP3s under ignored `reference-audio/`, verifies three
short phrases with Whisper, and packages Instruction 1, a separate15-second
silent Calibration, and Outro with transcript/class JSON/CSV. No speech stretching is applied. The initial long take exceeded the
native ASR checker's input geometry and remains unused; shorter checked phrases
avoid that failure. Read `audio/calibration.md` for timing and Respyra limits.
After regeneration, review/rebind all six scenario manifest hashes, regenerate
CSV and refresh audio-review snapshots. Other speech retains the existing
ElevenLabs workflow below. Real respiratory calibration and listening acceptance
remain open; do not equate the audio phase with a measured sensor result.

Every final sequence must use its own `assets/audio/scenarios/<scenario>/` folder and a saved
JSON/CSV pair in `audio/scenarios/`. JSON is the runtime authority; edit its
frame-indexed events, then run `pnpm run scenarios:csv` and
`pnpm run check:scenarios`. Do not add per-scenario full audio tracks or a
calculated runtime fallback. Package one independent WAV/CSV/TXT/JSON group per unique cue,
named with a four-digit first-use order and first-onset timestamp; repeat cues
in the playlist/timeline instead of duplicating every occurrence. Use
the same basename for all four files. Each per-cue CSV must list all of that
cue's occurrences in chronological order, with exact frame/second timing,
global event order, cycle/block, draft script and audio/script status. Run
`python -m unittest discover -s tests -p test_scenario_assets.py` for this contract,
and `python tools/build-scenario-placeholders.py --check` for the current placeholders.
Bind the timeline to the selected library manifest
hash; changing that manifest requires reviewing/rebinding every scenario and
rerunning audio checks. See `audio/scenarios/README.md` for fields and mappings.
Native offline diagnostics and verification exports also read the saved JSON.

For reviewing the current participant experience, use `audio-review/` rather
than mixing review text into runtime assets or ignored audition output. Its
plain-text inventory distinguishes runtime, selected-voice auditions and older
sets; `timelines/` contains all five menu choices plus the engine-only 20-minute
sequence. These are fixed snapshots, checked against the saved scenario JSON
and library manifest, with source hashes in each timeline. Refresh the text
when its source hashes change, keeping exact German speech separate from
unspoken review notes. A text score does not establish listening acceptance.

Generate speech through the ElevenLabs API from a local process. Read the API
key from an external credential file or environment variable; never put it in
page source, saved request metadata, repository files, or command output.

`assets/audio/breathwork/de/cues.json` records the exact scripts, selected voice,
model, request settings, and output format. To regenerate, submit each cue's
`text` with those settings to the documented
[text-to-speech endpoint](https://elevenlabs.io/docs/api-reference/text-to-speech/convert).
The multilingual v2 model uses the German input text directly; it does not
support the `language_code` request parameter. Preserve existing outputs and
avoid blindly retrying generation after an ambiguous response, since requests
consume account credits. Update durations/hashes and fully decode every new MP3
before recording technical verification. Listening review is a separate check.

For reusable box-breathing samples, use the PCM assets and frame contract in
`assets/audio/box-breathing/de/soft-v2/manifest.json`, not the earlier variable-duration
MP3 cue set. Keep the four phase files separate and reuse them on every cycle.
Run `pnpm run test:audio` and the silent `tests/offline-audio.html` diagnostic
when changing the scheduler or sample set. `pnpm run render:audio -- 5` (or 10/15/20)
provides an optional offline verification export; it is not the runtime strategy.
For the sole main scenario, open `tests/offline-audio.html?round=haupt`: it checks
the exact 900-second total, separate calibration window, fifty complete cycles
and Outro at both 44.1 and 48 kHz. There is no main baseline.
Original downloads, full caption transcripts, raw synthesis, and concatenated
exports stay under ignored `reference-audio/`. See [audio/README.md](../audio/README.md)
for the reference analysis and reproducible timing/export workflow.

For local German corpus auditions, run `tools/prepare-calm-voice-references.py`
with the external database folder. Keep originals, auditions, Secret Tunnel
profile IDs, and full receipts under ignored `reference-audio/`. Record source,
speaker, and model permissions separately; see `audio/voice-sourcing.md`. Native
Secret Tunnel CLI operations are a local fallback when MCP voice tools fail;
use its existing authentication and do not alter access settings.

For new Secret Tunnel slow-voice auditions (27 October 2026 user correction),
set pace during native synthesis. Do not apply post-generation time stretching.
The earlier stretched `Ein-atmen` listening copies were rejected for audible
artifacts. Use native target-duration controls and supported style instructions,
record active speech span separately from total duration, and retain original
generator WAVs plus settings/hashes. Comparison tracks may add silent gaps while
preserving the source samples exactly. Peak/decode/ASR checks do not establish
acceptable pronunciation, softness or calmness; user listening selection remains
required. See [voice sourcing](../audio/voice-sourcing.md) for current evidence.
The 2026-10-28 user request separately authorizes the twenty exact-phrase MP3s
under `samples/audio/` and their listening subpage. Keep source references,
profile UUIDs, and full native receipts private. `samples/samples.json` contains
only public source-rating summaries, generated durations, MP3 hashes, and
truthful transcript evidence. Preserve native raw word scores; digit/word
equivalence (`2=zwei`, `3=drei`, `4=vier`) is recorded as a separate check.
Run `node for-ai/scripts/check-samples.mjs` from `atmungsraum/` after changing
the listening page or its selected samples. Existing runtime voice selection
and session timing are unaffected by this audition page.

For the local voice-to-breath prototype, use `tools/build-power-crossfade-audition.py`
with `--voices-dir`, `--sounds-dir`, and an ignored `--output-dir`. It consumes
the restrained phase files and cached standalone Halten takes. Inspect finished
PCM layer powers, gain progression, duration, headroom, hold silence, and montage
identity; preserve provenance. This audition does not update runtime assets.

For counted Halten/Pause alternatives, use `tools/build-counted-phase-variants.py`
with `--voices-dir`, `--mixes-dir`, and an ignored `--output-dir` containing saved
verified Pause receipts. It reuses existing counted inhale/exhale sources and
mix layers, preserves complete labels while fitting beats, and creates two
four-element sets per voice plus listening montages and a ZIP. Check the finished
PCM durations, beat gaps, power progression, montage identity, and archive; keep
listening acceptance distinct from speech recognition and sample measurements.

Run `python tools/build-elevenlabs-listening-cues.py` with cached connected phrases
and standalone Halten!, then `python tools/package-listening-assets.py` to package
the selected ElevenLabs counted cues in `audio-assets/ready-to-listen/elevenlabs/`.
The selected wording uses Halten! once in both holds, followed by silence. Cache
its synthesis as `halten-exclamation.mp3` and matching request JSON. Each hold
stays four seconds; JSON records one word slot and speech/quiet segments. Breath
uses constant gain with natural dynamics and 10 ms boundary fades; counts descend
during inhale/exhale only.
Inhale/exhale now require complete `inhale-connected.mp3`/`exhale-connected.mp3`
takes with matching `.request.json` and `.response.json` from ElevenLabs'
[speech-with-timing endpoint](https://elevenlabs.io/docs/api-reference/text-to-speech/convert-with-timestamps).
Generate each whole phrase in one request with the website profile/settings.
The helper uses character alignment to locate word windows, preserves native
speech samples and adjusts only silence. It fails rather than speeding speech
that cannot fit. Actual word starts can depart slightly from nominal beats and
are saved in frame-indexed sidecars. Source responses stay local/ignored.
Every WAV receives an adjacent JSON with timing/transcript/hash metadata;
the current voice exposes separate voice,
breath and mix layers plus complete 16-second cycles. It verifies component
reconstruction and sample-identical concatenation without regeneration or API
calls. Preserve differing existing outputs and choose a new `--output-dir` for
a revised set. Only the folder guide and helper are source; generated copies
stay ignored. Historical OmniVoice sources require explicit `--source-dir` and
a separate output path; rejected copies stay in `audio-assets/archived-omnivoice/`.
See `audio-assets/README.md` for the layout.
