# Verification

## Observed listening asset packaging — 2026-10-12

`audio-assets/ready-to-listen/` contains 26 same-folder WAV/JSON pairs across
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

The current checker verifies the guidance, AI document set, four choices per
round, black background, local artwork/runtime assets, and locked Pretext
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
