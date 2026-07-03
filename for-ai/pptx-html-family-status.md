# PPTX HTML Family Status

Last updated: 2026-07-03

This is the living implementation/status note for replacing the three Viscereality presentation players with object-based HTML scene players that share one asset library.

Every future AI agent working on this pipeline should read this file before editing `tools/pptx-html-presenter/`, `presentations/*-scene/`, `presentations/shared-assets/`, `presentations/index.html`, or `presentations/shared/decks.js`. Update this file whenever the build pipeline, QA status, asset policy, or known blockers change.

## Active Goal

Replace the current chunked public players for:

- `MuC`: `presentations/Viscereality_MuC.pptx`
- `alpCHI`: `presentations/Viscereality_alpCHI_v2.pptx`
- `BBD26`: `presentations/20260512_BreathworkDays_Berlin_new.pptx`

with three separate scene-rendered HTML players:

- `/presentations/MuC/`
- `/presentations/alpCHI/`
- `/presentations/BBD26/`

Underneath those public URLs, duplicate media should be reused from:

```text
presentations/shared-assets/viscereality/
```

## Current Implementation State

- Branch in use: `codex/pptx-html-scene-player`.
- Main tool directory: `tools/pptx-html-presenter/`.
- Family config: `presentations/viscereality-family.config.json`.
- Shared default presenter config: `presentations/viscereality-family.defaults.json`.
- Current staging output targets:
  - `presentations/MuC-scene/`
  - `presentations/alpCHI-scene/`
  - `presentations/BBD26-scene-new/`
- Current public scene player targets:
  - `presentations/MuC/`
  - `presentations/alpCHI/`
  - `presentations/BBD26/`
- Archived chunked fallback targets:
  - `presentations/MuC-chunked/`
  - `presentations/alpCHI-chunked/`
  - `presentations/BBD26-chunked/`
- Shared asset index:
  - `presentations/shared-assets/viscereality/asset-index.json`
- Latest family build status: `ok`.
- Latest family HTML visual audit status: `ok`.
- Latest family publish status: `ok`; the public presentation hub points at the three scene players and keeps chunked fallback links secondary.
- PowerPoint MP4 oracle QA: not yet run for the full three-deck family in the current pass; do not claim oracle parity until reference MP4 export/frame comparison is completed.

Implemented family CLI commands:

```powershell
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family inspect presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family build presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family visual-audit presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family publish presentations\viscereality-family.config.json
```

## Parsed Deck Facts

Current source PPTX parse results from family preflight:

- `MuC`: 17 slides, 39 assets, all parsed transitions are Morph.
- `alpCHI`: 25 slides, 48 assets, 24 Morph transitions and 1 custom transition.
- `BBD26`: 32 slides, 57 assets, 31 Morph transitions and 1 custom transition.
- Current chunked public folders total about 1.12 GiB.
- Source media across all three decks totals about 3.32 GiB before dedupe.
- Unique source media across the family is about 1.53 GiB, so the family approach can save about 1.80 GiB versus naive per-deck source duplication.

## Important Design Constraints

- This is not a screenshot/chunked-video replacement pipeline.
- PowerPoint objects should remain browser objects whenever possible.
- Explicit PowerPoint groups become DOM groups.
- Inferred rounded white carousel panels become DOM panel groups.
- Panel children move in lockstep with the panel.
- Panel outlines/borders render above their child contents.
- Forward and backward Morph should use the same mapping mirrored.
- GIF/video clocks should keep looping while objects move forward or backward.
- Full-slide raster fallback is disabled by default; use only smallest-object fallback for unsupported effects.
- Public runtime assets must be GitHub Pages-safe unless Git LFS or another hosting policy is explicitly approved.

## Recent Problems And Solutions

- Problem: The first shared-asset hoist marked pruned `sourceFile` originals as missing, even when the runtime `file` had already been optimized and shared.
  - Solution: `share_deck_assets()` now processes runtime `file` first and maps pruned `sourceFile` provenance to the shared runtime file when appropriate.
- Problem: A MuC GIF produced an optimized MP4 around 116 MiB, still above the GitHub Pages hard limit.
  - Solution: GIF optimization now treats over-hard-limit MP4 as failed and tries WebM/WebP alternatives before accepting a runtime asset.
- Problem: Shared `source/` provenance can still contain huge originals even when the runtime uses a small optimized file.
  - Solution: family shared-asset pruning now removes unreferenced oversized source blobs after scene manifests point at optimized runtime files. The MuC 386 MiB source GIF was pruned from `presentations/shared-assets/viscereality/source/`.
- Problem: Opaque, video-like GIFs may be huge and slow if preserved as GIF or animated WebP.
  - Solution: for opaque GIFs, the optimizer now tries MP4 first, then progressively smaller MP4 variants before animated WebP fallback. The MuC oversized GIF now builds as an under-limit MP4 runtime asset.
- Problem: Public GitHub Pages cannot safely host very large original animation blobs just for provenance when the runtime uses a smaller optimized equivalent.
  - Solution: family builds now publish only referenced GitHub-safe runtime assets in the shared library; oversized originals remain represented by hashes/source metadata in reports rather than copied into the public asset tree.
- Problem: Visual audit initially failed with `playwright-missing`.
  - Solution: use the bundled Node executable with Playwright's pnpm package root, for example `--node-bin C:\Users\gfeje\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe --playwright-dir C:\Users\gfeje\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules\.pnpm\playwright@1.61.1`.
- Problem: Visual audit capture initially hung because the capture server served only a deck folder while scene manifests referenced `../shared-assets/...`.
  - Solution: `browser_capture.mjs` now serves the enclosing presentations folder and opens `/<deck-folder>/index.html`, allowing shared asset URLs to resolve.
- Durable rule: super-large assets must be converted to visually lossless or visually acceptable HTML-friendly formats that stay GitHub-compatible. Use MP4/WebM/WebP according to alpha/playback needs; do not publish giant original blobs unless LFS/external hosting is explicitly chosen.

## Latest HTML Visual Audit

The latest browser-based visual audit captured and passed:

- `MuC`: 145 samples, 17 settled slides, 112 forward transition samples, 16 reverse midpoint samples, 0 failures.
- `alpCHI`: 217 samples, 25 settled slides, 168 forward transition samples, 24 reverse midpoint samples, 0 failures.
- `BBD26`: 280 samples, 32 settled slides, 217 forward transition samples, 31 reverse midpoint samples, 0 failures.

Contact sheets were manually inspected at audit scale for settled slides and transition midpoints. No blank frames, missing shared media, or obvious panel-child drift were observed in that review. This is not a substitute for PowerPoint oracle SSIM QA.

## Latest Public Publish

The latest publish copied the validated scene players into the canonical public folders:

- `/presentations/MuC/`
- `/presentations/alpCHI/`
- `/presentations/BBD26/`

The previous chunked players were moved to:

- `/presentations/MuC-chunked/`
- `/presentations/alpCHI-chunked/`
- `/presentations/BBD26-chunked/`

`presentations/index.html` exposes three separate presentation cards and secondary chunked fallback links. `presentations/shared/decks.js` registers only the public scene deck IDs `MuC`, `alpCHI`, and `BBD26`.

## Remaining Work

1. Run PowerPoint MP4 oracle QA when disk space allows; do not claim full oracle pass without it.
2. Continue comparing future PPTX revisions against contact sheets and PowerPoint oracle frames before replacing public decks again.
3. Commit and push only intended files; do not stage unrelated root `index.html` changes.

## Verification Commands

```powershell
py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family inspect presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family build presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family visual-audit presentations\viscereality-family.config.json
```

PowerPoint oracle QA still requires enough free disk for reference MP4 export and frame extraction.
