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
- MuC deck-specific presenter config: `presentations/MuC-scene.config.json`.
- alpCHI deck-specific presenter config: `presentations/alpCHI-scene.config.json`.
- BBD26 deck-specific presenter config: `presentations/BBD26-scene.config.json`.
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
- Latest family inspect status: `blocked` only because the local disk free-space preflight is below the configured 8 GiB floor; source PPTX parsing still reports 3 decks and about 1.53 GiB unique source media.
- Latest family HTML visual audit status: `ok`.
- Latest family publish status: `ok`; the public presentation hub points at the three scene players and keeps chunked fallback links secondary.
- PowerPoint MP4 oracle QA: slide-1 smoke passes have run for MuC, alpCHI, and BBD26. They are useful for calibration but still fail strict SSIM; do not claim oracle parity until full reference MP4 export/frame comparison passes or reviewed exceptions are written.

Implemented family CLI commands:

```powershell
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family inspect presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family build presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family visual-audit presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family oracle-qa presentations\viscereality-family.config.json --ffmpeg-bin "C:\path\to\ffmpeg.exe"
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
- Super-large public assets must be converted to HTML-friendly runtime formats before publish. The default asset policy now treats oversized assets as disallowed, optimizes oversized static images to WebP, transcodes GIF/video when useful, and reports both the 50 MiB preferred ceiling and the 100 MiB hard ceiling.

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
- Problem: Rebuilding MuC changed optimized media content hashes even when source assets were unchanged, causing avoidable shared-asset churn.
  - Solution: ffmpeg transcodes now strip metadata/chapters and use single-threaded bitexact-oriented WebM/MP4/WebP output settings. This is slower, especially for VP9-alpha, but should make content-hashed optimized assets reproducible across rebuilds.
- Problem: PowerPoint-oracle timing compared transition frames with the wrong reference lead behavior inherited from earlier BBD26-style defaults.
  - Solution: shared family QA defaults now use `transition_reference_lead_fraction: 0.0`; MuC, alpCHI, and BBD26 public/staging scene metadata now use that lead value. Each deck has a deck-specific transition 1->2 Morph progress map.
- Problem: MuC oracle capture briefly blocked on `visible-video-not-ready` for a newly encoded WebM even though the frame could be captured.
  - Solution: `browser_capture.mjs` now waits longer for visible videos to reach `HAVE_CURRENT_DATA` after seek and reports readiness diagnostics if that still fails.
- Problem: Visual audit initially failed with `playwright-missing`.
  - Solution: use the bundled Node executable with Playwright's pnpm package root, for example `--node-bin C:\Users\gfeje\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe --playwright-dir C:\Users\gfeje\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules\.pnpm\playwright@1.61.1`.
- Problem: Visual audit capture initially hung because the capture server served only a deck folder while scene manifests referenced `../shared-assets/...`.
  - Solution: `browser_capture.mjs` now serves the enclosing presentations folder and opens `/<deck-folder>/index.html`, allowing shared asset URLs to resolve.
- Durable rule: super-large assets must be converted to visually lossless or visually acceptable HTML-friendly formats that stay GitHub-compatible. Use MP4/WebM/WebP according to alpha/playback needs; do not publish giant original blobs unless LFS/external hosting is explicitly chosen.
- Problem: The earlier default config still allowed oversized assets and did not attempt static-image optimization.
  - Solution: `AssetPolicy.allow_oversize_assets` now defaults to `false`, `optimize_static_images` defaults to `true`, the family defaults file enables static-image WebP optimization, and the CLI exposes `--image-optimize/--no-image-optimize` for debugging.
- Problem: A future deck could publish a large original `sourceFile` beside a safe optimized runtime asset.
  - Solution: family asset sharing now uses a 50 MiB preferred public-asset ceiling. If an original source blob is above that ceiling and the runtime `file` is already a shared safe optimized asset, `sourceFile` is rewritten to the runtime file and the original remains represented by hash/path metadata rather than being copied into the public shared tree.
- Problem: A future oversized static PNG/JPEG/TIFF/BMP could pass through unchanged if it was not GIF/video.
  - Solution: oversized still images are now candidates for conservative WebP optimization: a high-quality 4K WebP first, then a 1080p WebP fallback only when needed to satisfy the hard limit.
- Problem: PowerPoint can store a high-fidelity HDPhoto/WDP image layer plus a lower-fidelity PNG fallback for a single visible object. MuC slide 1 `Picture 10` used this for the raster title/text block, but the compiler selected the PNG fallback.
  - Solution: the parser now detects PowerPoint image layers (`a14:imgLayer`), prefers the related `.wdp` media asset for that object, and asset preparation converts WDP to browser-safe PNG via Windows WIC. A diagnostic MuC build rendered slide 1 with the converted WDP title object and improved settled slide-1 SSIM from `0.788991` to `0.865641` against the PowerPoint reference.
- Problem: full family rebuilds are still slow because large transparent GIFs transcode to deterministic VP9-alpha WebM single-threaded.
  - Solution: family builds now seed an optimized-asset cache from the existing shared asset index and the current public/staging scene manifests. `prepare_assets()` consults this cache by original PPT media SHA before starting WDP/GIF/static-image/video conversion, copies the previously accepted runtime file into the local build, and records `optimized-asset-reused-from-shared-cache`. Path-only fallback is used only when no source SHA is available, to avoid accidentally reusing stale media when a future PPTX changes bytes but keeps an internal `ppt/media/...` filename.
- Problem: shared optimized assets previously did not retain the original PPT media SHA, making future cache reuse harder and less deterministic.
  - Solution: `share_deck_assets()` now writes `sourceSha256` and `sourceSha256s` into `asset-index.json` for each shared entry when the scene asset provides a source hash. Current shared index has source SHA metadata for all optimized entries.
- Problem: PowerPoint HDPhoto image layers can include `a14:brightnessContrast` effects. BBD26 and alpCHI slide 1 used `bright="100000"` on the HDPhoto layer, so selecting the WDP asset without the image-layer effect made white logos render as colored/dark variants.
  - Solution: the scene schema now carries object-level `mediaEffects.brightnessContrast`, the parser reads `a14:imgLayer/a14:imgEffect/a14:brightnessContrast`, and the HTML runtime applies a CSS filter to the media element. The `bright=100000` case maps visible pixels to white while preserving alpha. This is object-level, so it fits the one-to-one mapper direction better than baking a deck-specific asset.

Current shared public asset library check after the WDP/cache family rebuild:

- `presentations/shared-assets/viscereality/` contains 76 runtime/source files, about 362.57 MiB total.
- Largest shared asset is about 48.879 MiB.
- Files above 50 MiB: 0.
- Files above 100 MiB: 0.
- Optimized cache reuse in the latest build:
  - `MuC`: 15 cached optimized assets reused.
  - `alpCHI`: 21 cached optimized assets reused.
  - `BBD26`: 22 cached optimized assets reused.
- The latest build also converted one WDP/HDPhoto-derived asset per public deck into shared PNG runtime assets.
- Family builds now emit `sharedAssetLimits` with `preferredAssetSafe`, `softOversizeAssets`, and `oversizeAssets` so future agents can verify the public shared library gate without hunting through per-deck reports.

## Latest HTML Visual Audit

The latest browser-based visual audit captured and passed:

- `MuC`: 145 samples, 17 settled slides, 112 forward transition samples, 16 reverse midpoint samples, 0 failures.
- `alpCHI`: 217 samples, 25 settled slides, 168 forward transition samples, 24 reverse midpoint samples, 0 failures.
- `BBD26`: 280 samples, 32 settled slides, 217 forward transition samples, 31 reverse midpoint samples, 0 failures.

Contact sheets were manually inspected at audit scale for settled slides and transition midpoints. No blank frames, missing shared media, or obvious panel-child drift were observed in that review. This is not a substitute for PowerPoint oracle SSIM QA.

After the first-transition calibration updates, current public HTML visual-audit status is:

- `MuC`: 145 samples, 0 failures.
- `alpCHI`: 217 samples, 0 failures.
- `BBD26`: 280 samples, 0 failures.

After the WDP/cache rebuild and public publish on 2026-07-03, `family visual-audit` passed again for all three decks with 0 failures. After the subsequent `mediaEffects.brightnessContrast` runtime/parser change, `family visual-audit` passed again for all three decks with 0 failures. This validates browser load/capture, shared asset URLs, settled slides, forward transition samples, and reverse midpoint samples for the rebuilt scene players. It is still not a PowerPoint-oracle SSIM pass.

## Latest PowerPoint Oracle Smoke

Slide-1 smoke passes have now run after adding `family oracle-qa`:

```powershell
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family oracle-qa presentations\viscereality-family.config.json --decks MuC --slides 1 --target public --force --min-free-gb 0 --ffmpeg-bin "C:\path\to\ffmpeg.exe"
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family oracle-qa presentations\viscereality-family.config.json --decks alpCHI --slides 1 --target public --force --min-free-gb 0 --ffmpeg-bin "C:\path\to\ffmpeg.exe"
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family oracle-qa presentations\viscereality-family.config.json --decks BBD26 --slides 1 --target public --force --min-free-gb 0 --ffmpeg-bin "C:\path\to\ffmpeg.exe"
```

Current smoke results after the media-effects rebuild/public publish:

- `MuC`: status `failed`, no blockers, 8 comparisons, minimum SSIM about `0.623`, settled slide 1 about `0.789`, transition start about `0.916`.
- `alpCHI`: status `failed`, no blockers, 8 comparisons, minimum SSIM about `0.497`, settled slide 1 improved to about `0.814`, transition start improved to about `0.932`, transition endpoint about `0.922`.
- `BBD26`: status `failed`, no blockers, 8 comparisons, minimum SSIM still about `0.323`, settled slide 1 improved to about `0.785`, transition start improved to about `0.714`, calibrated middle samples now reach about `0.862`.

Interpretation:

- The player is coherent and assets load, but strict PowerPoint visual parity is not achieved yet.
- The transition 1->2 Morph progress calibrations and HDPhoto brightness effects improved specific samples, but remaining differences include PowerPoint glow/soft-edge effects, text/raster antialiasing, text placement/scale, background/video brightness/phase, panel timing late in BBD26, and full-frame composition differences.
- Do not publish claims of PowerPoint-oracle success until all three decks pass full oracle QA or have reviewed exceptions.

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

1. Run full PowerPoint MP4 oracle QA for MuC, alpCHI, and BBD26 when disk space allows; do not claim full oracle pass without it.
2. Continue fine-tuning deck-specific configs for oracle parity: first-transition smoke is improved but not passing on any deck.
3. Continue improving oracle parity after the WDP/cache public rebuild. The public scene decks have been rebuilt, visually audited, and republished with WDP conversion and shared optimized cache reuse, but strict PowerPoint oracle QA is still the main unresolved quality gate.
4. Continue calibrating text/layout metrics, media phase/brightness, and additional Morph progress maps.
5. Continue comparing future PPTX revisions against contact sheets and PowerPoint oracle frames before replacing public decks again.
6. Commit and push only intended files; do not stage unrelated root `index.html` changes.

## Verification Commands

```powershell
py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family inspect presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family build presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family visual-audit presentations\viscereality-family.config.json
```

PowerPoint oracle QA still requires enough free disk for reference MP4 export and frame extraction.
