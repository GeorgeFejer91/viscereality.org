# PPT Chunker v3 (Strict, Feature-Driven)

Windows-first PowerPoint pipeline for building GitHub Pages deck players with strict validation.

## Staged Workflow

1. Inspect

```bash
py -3 chunk_presentation.py inspect "presentation.pptx" -o output --profile balanced1080
```

Outputs `feature_report.json`.

2. Build

```bash
py -3 chunk_presentation.py build "presentation.pptx" -o output --strict
```

Outputs:

- `manifest.json`
- `timing_decisions.json`
- `feature_report.json`
- `hiccup_report.json`
- `chunks/*.mp4` (hashed names)
- `assets/*.png` for static slides

3. Validate

```bash
py -3 chunk_presentation.py validate output --strict
```

Outputs:

- `validation_report.json`
- `hiccup_report.json` (updated)
- `validation.ok` on pass

4. Publish

```bash
py -3 chunk_presentation.py publish output --deck alpCHI
```

Copies validated artifacts to `presentations/<deck>/`.

Optional git push:

```bash
py -3 chunk_presentation.py publish output --deck alpCHI --git-push
```

## Defaults

- Profile: `balanced1080`
- Static slide duration: `4.0s`
- Tiny transition rule: `<=0.01s` -> treated as `none`
- Strict mode: enabled by default for `build` and `validate`
- Source PPT: never mutated (temp working copy only)

## Manifest Contract

Legacy `chunks` stays available.

Extended fields include:

- `segments[]` with `start_sec/end_sec`
- per-slide `asset_kind` (`video` or `image`)
- optional `deck_meta`

## Player Behavior

- Slide chunks loop indefinitely.
- `Next`: transition chunk plays once, then next slide loops.
- `Prev`: jump directly to previous slide.
- Static slides can be image assets and wait for user navigation.
- Filename rules can override playback metadata at runtime.

## Filename Rules (Runtime-Parsed)

Chunk filenames can carry play hints that the player parses:

- Sequence prefix: `NN_` so files sort in playback order
- Type token: `slide` or `transition`
- Duration token: `dur<seconds>` (use `p` for decimals, e.g. `dur2p5`)
- Transition navigation token: `navauto`, `navmanual`, or `navimmediate`

Examples:

- `01_Slide_01_dur4.mp4`
- `02_Transition_01_to_02_dur0_navmanual.mp4` (transition effectively skipped)
- `04_Transition_02_to_03_dur0p033_navimmediate.mp4` (single-frame immediate handoff)
- `04_Transition_02_to_03_dur1p2_navauto.mp4` (auto-advance after current slide)

## Legacy Compatibility

Legacy commands still exist and route to v3 internals:

- `analyze`
- `chunk`
- `run`
- `upload`

## Requirements

- Python 3.10+
- PowerPoint + `pywin32`
- ffmpeg / ffprobe

Install pywin32:

```bash
py -3 -m pip install pywin32
```
