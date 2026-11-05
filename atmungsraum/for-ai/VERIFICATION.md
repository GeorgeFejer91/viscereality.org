# Verification

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
