# Verification

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
