# Scenario audio timelines

Every final sequence uses one of these JSON/CSV pairs. All sound comes from the
shared [`assets/audio/`](../../assets/audio/) library. The selected runtime
registry is [`manifest.json`](../../assets/audio/box-breathing/de/soft-v2/manifest.json):
each asset ID resolves to one reusable WAV with its frame count, transcript,
byte count and SHA-256. Scenarios reference IDs; they do not contain audio copies.
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
| Haupt: 10 minutes | `haupt-10.json` / `.csv` | 300 silent + 300 guided seconds |
| Haupt: 15 minutes | `haupt-15.json` / `.csv` | 300 silent + 600 guided seconds |
| Engine diagnostic, outside the menu | `box-20.json` / `.csv` | 1200 seconds |

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
`inhale_breath`/`exhale_breath` select the existing premixed breath variants.
`baseline` and `recovery` use `asset: null` for explicit silence. Each slot must
start where the preceding one ends, and a sounding slot must equal its clip's
full duration. Keep cycle/block annotations and summary counts consistent.
Frame counts are exact; CSV also shows start, duration and end in seconds, row
order, asset ID and asset path relative to the Atmungsraum root.

The browser preloads and validates the library and every scenario before Start.
All cues are queued on one Web Audio clock. Pause freezes that clock; resume
continues the same nodes. Receipts retain the scenario ID, JSON hash and library
manifest hash. `createProtocol()` remains an authoring/regression reference;
runtime playback and verification exports read the saved JSON.

The current integration is browser audio. Actual video/headset execution and
physical audiovisual synchronization remain pending; scheduling receipts are
not measured acoustic onsets.
