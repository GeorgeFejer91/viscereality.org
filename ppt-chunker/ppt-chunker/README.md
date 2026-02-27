# Hybrid PPT Chunker v2

Windows-first PowerPoint pipeline for GitHub Pages slide-style playback.

## What You Get

- Backward-compatible commands:
  - `analyze <pptx>`
  - `chunk <mp4> --config timing_config.json`
- New end-to-end command:
  - `run <pptx>`
- Slide player behavior:
  - slide chunks loop
  - `Next` plays transition once then loops next slide
  - `Previous` jumps directly to previous slide
- Double-buffer player template (`player/index.html`)
- Strict validation + fail-fast ffmpeg errors
- Chunk size guardrail (`--max-chunk-mb`)

## Requirements

- Python 3.10+
- `pywin32` (required for `run` COM export and optional COM probe)
- ffmpeg/ffprobe binaries (PATH or winget install locations)

Install:

```bash
py -3 -m pip install pywin32
```

## Commands

### 1) Analyze PPT timings

```bash
py -3 chunk_presentation.py analyze "presentation.pptx" --com-probe -o output
```

Outputs: `output/timing_config.json`

- Uses XML timing extraction by default.
- `--com-probe` adds PowerPoint COM timing probe.
- Writes legacy `segments` and v2 `defaults/slides` blocks.

### 2) Chunk an existing MP4 (legacy-compatible flow)

```bash
py -3 chunk_presentation.py chunk "presentation.mp4" \
  --config output/timing_config.json \
  --output-dir output \
  --max-chunk-mb 95 \
  --generate-player
```

### 3) Run full end-to-end pipeline

```bash
py -3 chunk_presentation.py run "presentation.pptx" \
  --output-dir output \
  --timing-mode ppt \
  --fit-duration \
  --fps 30 \
  --height 1080 \
  --max-chunk-mb 95 \
  --generate-player
```

This runs:
1. XML analysis (+ COM probe for `ppt`/`hybrid`)
2. Timing resolution (override > COM > XML > defaults)
3. PowerPoint COM export to raw MP4
4. Web normalization encode
5. Optional duration-fit warp (`--fit-duration`) when drift exceeds tolerance
6. Chunking + validation
7. Manifest generation
8. Player scaffold copy (`index.html`)

## Timing Modes

- `ppt`: prefer authored timings (COM/XML), fallback to defaults
- `hybrid`: like `ppt`, intended for mixed/manual override usage
- `uniform`: force defaults per slide/transition

## Config Compatibility

`timing_config.json` supports both:

- Legacy:
  - `default_slide_duration_s`
  - `segments[]` with `slide_duration_s`, `transition_duration_s`, `label`
- V2 additions:
  - `defaults.slide_sec`, `defaults.transition_sec`
  - `slides.<n>.slide_sec`, `slides.<n>.transition_sec`, `slides.<n>.label`
  - optional `export` object

## Output Structure

```text
output/
  timing_config.json
  <presentation_id>_master.mp4
  manifest.json
  index.html             # if --generate-player
  chunks/
    slide_01.mp4
    trans_02.mp4
    slide_02.mp4
    ...
  build/
    <presentation_id>_raw.mp4
```

## Manifest

`manifest.json` keeps old `chunks` for player compatibility and adds v2 fields:

- `presentation_id`
- `source_ppt`
- `generated_at_utc`
- `encoding`
- `segments` with timeline bounds (`start_sec`, `end_sec`)
- `slides` map
- `player_defaults`

## GitHub Pages Integration

1. Copy `output/` artifacts into your repo path:
   - `presentations/<deck>/manifest.json`
   - `presentations/<deck>/chunks/*`
   - `presentations/<deck>/index.html`
2. Enable Pages on your branch.
3. Open:
   - `https://<user>.github.io/<repo>/presentations/<deck>/`

For large decks, host chunks on external storage (e.g., R2) and change `BASE_PATH` in `index.html`.

## Critical Notes

- PowerPoint COM can intermittently reject calls (`RPC_E_CALL_REJECTED`); retries/backoff are built in.
- Hidden pages are URL-obscured, not authenticated.
- `run` preserves source PPT by using a temporary working copy.

## Troubleshooting

- `ffmpeg not found`: pass `--ffmpeg-bin` / `--ffprobe-bin` explicitly.
- Duration mismatch errors: tune timings in config or increase `--duration-tolerance`.
- Oversized chunk error: increase `--max-chunk-mb` or reduce quality settings.
- COM export failure: close open modal dialogs in PowerPoint, retry command.
