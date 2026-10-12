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
6. Review and stage only intended paths. Publishing requires user authorization;
   the 2026-10-12 request authorizes publishing the current navigation prototype
   and its browser audio on the existing site. Local source recordings, voice
   auditions, listening copies, and full-session exports stay ignored.
   Never force-push or bypass branch protections.
7. After publication, run the checker with `-RequireRemote`, check the GitHub Pages
   build for the pushed commit, and verify the public URL. A successful push alone
   is not evidence that the page is live.

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

## Audio cue generation

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
Original downloads, full caption transcripts, raw synthesis, and concatenated
exports stay under ignored `reference-audio/`. See [audio/README.md](../audio/README.md)
for the reference analysis and reproducible timing/export workflow.

For local German corpus auditions, run `tools/prepare-calm-voice-references.py`
with the external database folder. Keep originals, auditions, Secret Tunnel
profile IDs, and full receipts under ignored `reference-audio/`. Record source,
speaker, and model permissions separately; see `audio/voice-sourcing.md`. Native
Secret Tunnel CLI operations are a local fallback when MCP voice tools fail;
use its existing authentication and do not alter access settings.

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

Run `python tools/package-listening-assets.py` to package the latest counted
auditions in `audio-assets/ready-to-listen/`. Every WAV receives an adjacent JSON
with timing/transcript/hash metadata; female/male sets expose separate voice,
breath and mix layers plus complete 16-second cycles. It verifies component
reconstruction and sample-identical concatenation without regeneration or API
calls. Preserve differing existing outputs and choose a new `--output-dir` for
a revised set. Only the folder guide and helper are source; generated copies
stay ignored. See `audio-assets/README.md` for the layout.
