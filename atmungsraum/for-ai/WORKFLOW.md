# Workflow

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

This standing authorization applies to future tasks in this repository unless
the user asks for local-only work. Before handoff, verify Git synchronization,
record exact publication evidence and remaining changes in PROJECT.md, and
report any concrete blocker instead of leaving completed work silently local.
The live target is <https://viscereality.org/atmungsraum/>. Documentation-only
updates follow the same prompt push/deployment rule, with scoped context/diff
and published-file checks rather than unrelated runtime test reruns.

Local setup shortcuts and `LOCAL-PROJECT.md` are excluded by `.git/info/exclude`.
Do not publish them or machine-specific paths.

## Local UI workflow

Serve `atmungsraum/` with a local static HTTP server (for example,
`python -m http.server 8876 --bind 127.0.0.1`) and inspect the page over HTTP.
Run `node for-ai/scripts/check-ui.mjs` for the sensor/route state check, then test
real rendered sizes as documented in VERIFICATION.md. Local QA fixtures and
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

## Participant data and mock ECG

Every participant-data writer must follow [DATA-LOGGING.md](DATA-LOGGING.md)
and reuse `participantIdentity()` / `sessionFilename()` from `data-logging.js`.
Run `npm test`, `pnpm run check:ui`, the context checker and rendered checks
after changes. `npm test` includes audio, source-integrity and data-logging
regressions. Do not publish participant exports, browser databases or checkpoints.
If exports are saved inside the checkout, use the ignored `session-data/` folder.

`pnpm run generate:ecg` runs the local Python generator using the installed
NeuroKit/NumPy/SciPy versions recorded in the manifest. Preserve/pin those
versions for reproduction; review any changed source bytes/settings and manifest
hash together. The browser loads only the generated local float32 trace and JSON.
Exercise CSV download/reload recovery, signal gaps, signed pre-start pause timing
and another live tab's checkpoint ownership. The current recorder requires
secure-context IndexedDB/Web Locks/Web Crypto; physical browser qualification
remains distinct from desktop viewport checks.

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
calculated runtime fallback. Package one independent WAV/JSON/TXT per unique cue,
named with a four-digit first-use order and first-onset timestamp; repeat cues
in the playlist/timeline instead of duplicating every occurrence. Use
`python tools/build-scenario-placeholders.py --check` for the current placeholders.
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
