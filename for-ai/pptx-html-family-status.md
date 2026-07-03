# PPTX HTML Family Status

Last updated: 2026-07-04

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
- Public asset-size policy:
  - default public/runtime target is under 50 MiB per file.
  - files above 100 MiB remain hard GitHub Pages blockers.
  - `asset_policy.allow_oversize_assets: false` now makes files above the 50 MiB preferred limit publish blockers too, not just warnings.
  - `allow_oversize_assets: true` is only for explicitly reviewed local/staging exceptions and does not make >100 MiB files GitHub-safe.
- Latest family build status: `ok`.
- Latest family inspect status: `blocked` only because the local disk free-space preflight is below the configured 8 GiB floor; source PPTX parsing still reports 3 decks and about 1.53 GiB unique source media.
- Latest family HTML visual audit status: `ok`; the latest direct public-folder audits also pass for all three decks.
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

- `MuC`: 17 slides, 39 assets, all parsed transitions are Morph. Note that an older planning note expected 18 slides, but the current source PPTX and generated scene manifest both verify 17 slides.
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
- Problem: A future public build could still accept an asset between 50 MiB and 100 MiB because the earlier per-deck asset report treated the 50 MiB limit as only a warning.
  - Solution: `prepare_assets()` now reports `hardLimitSafe`, `preferredAssetSafe`, and `publishAssetSafe`. With `allow_oversize_assets: false`, a post-optimization asset above the 50 MiB preferred limit gets `github-soft-limit-blocker`, `publishAssetSafe: false`, and the build status becomes `blocked-by-asset-size`.
- Problem: Oversized alpha GIFs, opaque GIFs, and videos could stop after one acceptable-under-100-MiB transcode even when the result was still too large for the preferred public asset ceiling.
  - Solution: GIF and video conversion now targets the publish policy, not only the hard limit. Transparent GIFs keep alpha and try alpha-safe WebM/WebP outputs; opaque GIF/video-like loops and large videos try progressively smaller MP4/WebM/WebP variants before falling back to a blocked staged build.
- Problem: Family sharing could upgrade a deck build back to `ok` when the shared library was hard-limit safe but still had preferred-limit violators.
  - Solution: `share_deck_assets()` now only restores `ok` when shared assets are both `githubPagesSafe` and `preferredAssetSafe`; otherwise the staged deck remains `blocked-by-asset-size`.
- Problem: PowerPoint can store a high-fidelity HDPhoto/WDP image layer plus a lower-fidelity PNG fallback for a single visible object. MuC slide 1 `Picture 10` used this for the raster title/text block, but the compiler selected the PNG fallback.
  - Solution: the parser now detects PowerPoint image layers (`a14:imgLayer`), prefers the related `.wdp` media asset for that object, and asset preparation converts WDP to browser-safe PNG via Windows WIC. A diagnostic MuC build rendered slide 1 with the converted WDP title object and improved settled slide-1 SSIM from `0.788991` to `0.865641` against the PowerPoint reference.
- Problem: full family rebuilds are still slow because large transparent GIFs transcode to deterministic VP9-alpha WebM single-threaded.
  - Solution: family builds now seed an optimized-asset cache from the existing shared asset index and the current public/staging scene manifests. `prepare_assets()` consults this cache by original PPT media SHA before starting WDP/GIF/static-image/video conversion, copies the previously accepted runtime file into the local build, and records `optimized-asset-reused-from-shared-cache`. Path-only fallback is used only when no source SHA is available, to avoid accidentally reusing stale media when a future PPTX changes bytes but keeps an internal `ppt/media/...` filename.
- Problem: shared optimized assets previously did not retain the original PPT media SHA, making future cache reuse harder and less deterministic.
  - Solution: `share_deck_assets()` now writes `sourceSha256` and `sourceSha256s` into `asset-index.json` for each shared entry when the scene asset provides a source hash. Current shared index has source SHA metadata for all optimized entries.
- Problem: PowerPoint HDPhoto image layers can include `a14:brightnessContrast` effects. BBD26 and alpCHI slide 1 used `bright="100000"` on the HDPhoto layer, so selecting the WDP asset without the image-layer effect made white logos render as colored/dark variants.
  - Solution: the scene schema now carries object-level `mediaEffects.brightnessContrast`, the parser reads `a14:imgLayer/a14:imgEffect/a14:brightnessContrast`, and the HTML runtime applies a CSS filter to the media element. The `bright=100000` case maps visible pixels to white while preserving alpha. This is object-level, so it fits the one-to-one mapper direction better than baking a deck-specific asset.
- Problem: PowerPoint uses `a:effectLst/a:glow` on many picture/SVG-like objects, including BBD26 title graphics and alpCHI/BBD26 logo elements. Without this, HTML objects looked too flat compared with the PowerPoint MP4 oracle.
  - Solution: the scene schema now carries object-level `visualEffects.glow` with PowerPoint radius/color/alpha. The parser reads `a:glow`, and the HTML runtime renders a conservative single CSS `drop-shadow(...)`/`text-shadow` scaled from EMUs to the current slide frame. A double-shadow version over-bloomed the alpCHI logo, so the current implementation intentionally uses one shadow.
- Problem: BBD26 transition 1->2 had a catastrophic oracle failure at 75% and 90% progress. The slide-2 panel object `track-0011` was present and opaque in the DOM, but was attached to the old explicit PowerPoint wrapper group from slide 1 (`track-group-71b...`), placing it offscreen at about `left=2134px`.
  - Solution: transition parenting now uses a stable parent track only when the parent track matches between the from/to states. If PowerPoint changes the wrapper-group identity while the object track itself is stable, the runtime renders that object in root slide coordinates for Morph. Panel children still attach to the stable panel track, so panel contents remain locked to the border.
- Problem: Manual unmatched-object fade timing changes for MuC transition 1->2 looked plausible but worsened the PowerPoint-oracle score when tested.
  - Solution: do not hard-code those guesses. `candidate-sweep` can now vary `enter-fade-end` and `exit-fade-end` by passing `unmatchedFadeOverride` through `browser_capture.mjs` into the runtime capture path. This lets future agents score fade timing candidates against oracle frames without rebuilding the deck or changing production scene JSON.
- Calibration result: on 2026-07-03, MuC `trans-001-002-025` sweeps for `exit-fade-end` values `0.05:1:0.05` and `enter-fade-end` values `0.05:1:0.05` both failed to improve the current baseline; best score observed was `0.591602`, below the existing smoke baseline around `0.624`. Treat fade-window sweeps as a diagnostic tool, not a currently accepted production override.
- Problem: Family default config values such as `visual_effects.glow_scale` were correctly merged by `_deck_presenter_config()` but were lost when `build_presentation()` copied fields into its title/slug-overlaid `effective_config`.
  - Solution: `build_presentation()` now carries `visual_effects` through to scene compilation, and the build smoke test asserts that non-default glow settings appear in `deck.scene.json`. The rebuilt public manifests now show `runtime.visualEffects.glowScale = 0.5` for `MuC`, `alpCHI`, and `BBD26`.
- Calibration result: on 2026-07-04, candidate sweeps showed a production `glow_scale` of `0.5` improves PowerPoint-oracle slide-1/early-transition samples for BBD26 and alpCHI without changing object structure. The family defaults now set `visual_effects.glow_scale` to `0.5`; `glow_alpha_scale` remains `1.0`.
- Problem: BBD26 visual-audit capture failed on reverse transition `27->26` because visible WebM `track-0109` could remain at `readyState = 0` in deterministic capture mode.
  - Solution: `browser_capture.mjs` now forces visible videos to `preload="auto"`, calls `load()` when metadata/data is absent, waits for metadata/current-frame readiness, seeks visible videos to their sample clocks, and only then screenshots. This does not reverse/restart media in the runtime player; it makes QA capture deterministic for visible media.
- Problem: alpCHI transition `1->2` treated the large title/text SVG block as an unmatched exit/enter pair (`Graphic 2` / `Graphic 15`) because one slide selected the SVG media layer and the next selected the PNG fallback for the same PowerPoint graphic.
  - Solution: `_selected_media_target()` now prefers SVG media over bitmap fallbacks when PowerPoint provides both. The public alpCHI scene now has `Graphic 2` and `Graphic 15` sharing `track-0010` and the same SVG asset, so the title block moves as a Morph object instead of fading in place.
- Problem: alpCHI public visual audit later hit `visible-video-not-ready` on endpoint sample `trans-006-007-100`, with visible WebM `track-0045` stuck at metadata-only `readyState = 1`.
  - Solution: browser capture now briefly plays stubborn visible videos muted if load/seek alone does not decode a current frame, then pauses them before screenshot. This is capture-only determinism; runtime media still loops independently during navigation.

Current shared public asset library check after the visual-effects/public-audit rebuild:

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
- Latest direct filesystem gate check on 2026-07-04: 76 shared public asset files, 362.57 MiB total, 0 files above 50 MiB, 0 files above 100 MiB. The largest runtime file is `optimized/3a907ddc7cdfe95de185fc64f27eaf69f5251c46deff568927ade6b6b9bca5b9.mp4` at about 48.879 MiB. This confirms the current public build follows the visually-lossless/html-friendly/GitHub-compatible asset rule.
- Latest stricter asset-policy verification on 2026-07-04: `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter` passed 125 tests, including soft-limit blocker tests. A direct shared-library filesystem gate again found 76 files, 362.57 MiB total, 0 files above 50 MiB, 0 files above 100 MiB, largest about 48.879 MiB. A full family rebuild was intentionally not run in this pass because `family inspect` reported only 4.85 GiB free against the configured 8 GiB preflight floor.

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

After the WDP/cache rebuild and public publish on 2026-07-03, `family visual-audit` passed again for all three decks with 0 failures. After the subsequent `mediaEffects.brightnessContrast` runtime/parser change, `family visual-audit` passed again for all three decks with 0 failures. After the `visualEffects.glow` runtime/parser change, `family visual-audit` passed again for all three decks with 0 failures. After the transition-parenting fix for mismatched explicit wrapper groups, `family visual-audit` passed again for all three decks with 0 failures. After the clean rebuild/publish that removed the rejected MuC transition-specific fade override and added capture-only unmatched-fade candidate sweeps, `family visual-audit` passed again for all three public decks with 0 failures.

After the `visual_effects` config propagation fix, deterministic visible-video capture hardening, and SVG-over-bitmap fallback preference on 2026-07-04, direct public-folder audits passed:

- `MuC`: 145 samples, 17 settled slides, 112 forward transition samples, 16 reverse midpoint samples, 0 failures, 0 warnings.
- `alpCHI`: 217 samples, 25 settled slides, 168 forward transition samples, 24 reverse midpoint samples, 0 failures, 0 warnings.
- `BBD26`: 280 samples, 32 settled slides, 217 forward transition samples, 31 reverse midpoint samples, 0 failures, 0 warnings.

This validates browser load/capture, shared asset URLs, settled slides, forward transition samples, reverse midpoint samples, the later BBD26 visible-video capture case, and the later alpCHI endpoint WebM capture case for the rebuilt scene players. It is still not a PowerPoint-oracle SSIM pass.

## Latest PowerPoint Oracle Smoke

Slide-1 smoke passes have now run after adding `family oracle-qa`:

```powershell
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family oracle-qa presentations\viscereality-family.config.json --decks MuC --slides 1 --target public --force --min-free-gb 0 --ffmpeg-bin "C:\path\to\ffmpeg.exe"
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family oracle-qa presentations\viscereality-family.config.json --decks alpCHI --slides 1 --target public --force --min-free-gb 0 --ffmpeg-bin "C:\path\to\ffmpeg.exe"
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family oracle-qa presentations\viscereality-family.config.json --decks BBD26 --slides 1 --target public --force --min-free-gb 0 --ffmpeg-bin "C:\path\to\ffmpeg.exe"
```

Current smoke results after the visual-effects config propagation fix, `glow_scale = 0.5`, SVG fallback preference, and public visual-audit rebuild:

- `MuC`: status `failed`, no blockers, 8 comparisons, minimum SSIM about `0.624`, settled slide 1 about `0.789`, transition start about `0.916`, transition 25% about `0.624`, transition midpoint about `0.868`, transition 75% about `0.857`.
- `alpCHI`: status `failed`, no blockers, 8 comparisons, minimum SSIM about `0.508`, settled slide 1 about `0.839`, transition start about `0.957`, transition 10% about `0.609`, transition 25% about `0.508`, transition midpoint about `0.746`, transition 75% about `0.781`. The 25% frame is visually improved because the large title block now slides left as a matched SVG object, though strict SSIM remains low and the 10% score dropped relative to the previous smoke.
- `BBD26`: status `failed`, no blockers, 8 comparisons, minimum SSIM about `0.754`, settled slide 1 about `0.826`, transition start about `0.754`, transition 25% about `0.898`, transition midpoint about `0.945`, transition 75% about `0.874`. This remains far above the previous catastrophic `0.323` late-transition failure caused by offscreen panel parenting.

Interpretation:

- The player is coherent and assets load, but strict PowerPoint visual parity is not achieved yet.
- The transition 1->2 Morph progress calibrations, HDPhoto brightness effects, conservative glow effects with calibrated family scaling, SVG fallback consistency, deterministic visible-video QA capture, and stable transition-parenting rule improved specific samples, but remaining differences include text/raster antialiasing, text placement/scale, remaining PowerPoint effects, background/video brightness/phase, early transition timing, and full-frame composition differences.
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
3. Continue improving oracle parity after the WDP/cache/visual-effects/SVG fallback public rebuild. The public scene decks have been rebuilt, visually audited, and republished with WDP conversion, shared optimized cache reuse, calibrated glow scaling, SVG fallback consistency, and deterministic visible-video capture, but strict PowerPoint oracle QA is still the main unresolved quality gate.
4. Continue calibrating text/layout metrics, media phase/brightness, additional Morph progress maps, and any remaining explicit-group wrapper transitions where PowerPoint changes group IDs but the visible object identity remains stable.
5. Continue comparing future PPTX revisions against contact sheets and PowerPoint oracle frames before replacing public decks again.
6. Commit and push only intended files; do not stage unrelated root `index.html` changes.

## Verification Commands

```powershell
py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family inspect presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family build presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family visual-audit presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py candidate-sweep presentations\MuC --sample trans-001-002-025 --vary exit-fade-end --values 0.05:1:0.05 --reference-frame presentations\MuC\qa\reference\trans-001-002-025.png
```

PowerPoint oracle QA still requires enough free disk for reference MP4 export and frame extraction.
