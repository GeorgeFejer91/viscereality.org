# Scenario audio timelines

The [German main study recordings](../../assets/audio/main-study/de/README.md)
include the two matched controls, the approved instruction, 18 phase takes
and the outro. The phase takes document the earlier audition. Use the main-study staging option to build the candidate
authored `haupt-15` plan. Its first five minutes contain both controls and
the instruction, followed by ten minutes of guided breathing. The authored
Quest recording includes the ensuing 19-second outro. Meta Developer 33
uses the shared preparation and main loop.

## Numbered scenario cue libraries

Every final sequence uses one of these JSON/CSV pairs and its own folder under
[`assets/audio/scenarios/`](../../assets/audio/scenarios/). Each unique cue has
a numbered WAV, CSV, JSON and TXT with the same basename. Numbers follow first occurrence, including quiet
recovery; repeated cycles reuse the same cue. Each folder also has an asset CSV,
complete timeline CSV, manifest and M3U8 playlist. The original shared
[`manifest.json`](../../assets/audio/box-breathing/de/soft-v2/manifest.json) pins
the protocol's frame lengths and the canonical cyclic recordings. Numbered
cyclic WAVs are copies of those recordings or their existing breath overlays.
Their TXT and `spoken_text` fields contain the recorded words. Non-cyclic
numbered WAVs retain their silent placeholders and draft script status.
CSV includes both audio and script status.
Each per-cue CSV lists all its occurrences in playback order, with global event
numbers, exact frames/start/end times, cycle/block numbers, draft words and
audio/script status. For example, `0004_01m21s_inhale.wav`, `.csv`, `.txt` and
`.json` are one matched group in `haupt-15/`. Sort Name ascending for first-use
order; use `0000_timeline.csv` or `0000_playlist.m3u8` for the complete sequence,
including repeated cues. The menu folders are `haupt-15`, `probe-anleitung`,
`probe-5`, `probe-10` and `probe-15`; `box-20` is a separate engine diagnostic.
Run `python -m unittest discover -s tests -p test_scenario_assets.py` to check
all numbered CSV/TXT pairs against the authored scenarios.
Run `python tools/build-scenario-placeholders.py --check` to verify the package;
omit `--check` to regenerate the packs from the bound library. The helper can
promote a phase from its exact silent placeholder and preserves differing recordings.
Hash-bound JSON and generated CSV use LF line endings, pinned in `.gitattributes`
so Windows checkouts and GitHub Pages validate the same bytes.
Existing v1/MP3 sets and the separate voice-audition page are retained references,
not sources for these final sequences.

| Scenario | JSON / CSV | Active duration |
| --- | --- | --- |
| Probe: Anleitung | `probe-anleitung.json` / `.csv` | 68-second replayable example |
| Probe: 5 minutes | `probe-5.json` / `.csv` | 300 seconds |
| Probe: 10 minutes | `probe-10.json` / `.csv` | 600 seconds |
| Probe: 15 minutes | `probe-15.json` / `.csv` | 900 seconds |
| Haupt: direct 15 minutes | `haupt-15.json` / `.csv` | 26 + 15 + 40 + (50 × 16) + 19 = 900 seconds |
| Engine diagnostic, outside the menu | `box-20.json` / `.csv` | 1200 seconds |

The only main scenario goes directly from the Haupt-Runde card to Start.
Its sequence is Instruction 1 → Calibration → Instruction 2 → fifty complete
inhale/pause/exhale/pause cycles → Outro. There is no baseline or main rest cue.
The calibration window is a separate silent asset at 26–41 seconds; exact source
frames and planned audio-clock times are available to future accelerometer
integration. No calibration value is currently measured. See the
[German transcript and Respyra distinction](../calibration.md).
`instruction_1_frames`, `calibration_start_frame`, and `calibration_frames`
identify the preparation/window. `baseline_frames` is explicitly 0. Main `minutes`
and `total_frames` both represent the complete15-minute budget, including every
asset. Practice remains unchanged. CSV includes `transcript_de` for each asset.

JSON drives playback. CSV is the generated timing sheet for the same sequence;
edit JSON, then run `pnpm run scenarios:csv`. Run `pnpm run check:scenarios`
to reject stale CSVs or invalid JSON, and `pnpm run test:audio` before use.
There is no independent CSV runtime or automatic fallback to a calculated plan.

Each JSON pins the shared library's manifest hash. Its ordered `events` array
defines every slot, including silence:

```json
{
  "id": "inhale",
  "asset": "inhale",
  "offset_frames": 1764000,
  "duration_frames": 176400,
  "cycle": 1,
  "breathing_block": 1
}
```

At `sample_rate: 44100`, this starts at 40 seconds and lasts 4 seconds. `id`
describes the phase shown by the UI; `asset` selects the actual library clip.
`inhale_breath`/`exhale_breath` identify the intended breath variants.
`recovery` uses `asset: null` for explicit practice silence; `calibration` is
a reusable15-second zero-PCM WAV with its own class and measurement metadata. Each slot must
start where the preceding one ends, and a sounding slot must equal its clip's
full duration. Keep cycle/block annotations and summary counts consistent.
Frame counts are exact; CSV also shows start, duration and end in seconds, row
order, asset ID and asset path relative to the Atmungsraum root.

`audio_assets_base_directory` and `audio_assets` bind each event's ID to the
scenario's own file, frame count and SHA-256. The browser preloads and validates
the library metadata and every scenario before Start. Identical content hashes
share a decoded buffer within one audio context, reducing the placeholder
download to seven files; future different recordings retain separate buffers.
All cues are queued on one Web Audio clock. Pause freezes that clock; resume
continues the same nodes. Receipts retain the scenario ID, JSON hash and library
manifest hash, plus `audio_asset_mode: silent-placeholders` for the current set.
`createProtocol()` remains an authoring/regression reference;
runtime playback and verification exports read the saved JSON.

The standalone browser engine and offline diagnostics can exercise the saved
cue timelines. The production phone controller delegates playback to the
paired Quest. Physical playback of the authored package remains pending.
