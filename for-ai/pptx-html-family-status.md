# PPTX HTML Family Status

Last updated: 2026-07-05

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
  - `family asset-check` now includes `largeSourceConversions`, a proof table for oversized PPT source assets. It confirms that each source asset above the 50 MiB preferred public limit is represented by a GitHub-safe optimized runtime asset and that no giant original leaked into `presentations/shared-assets/viscereality/source/`.
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
- Problem: A future operator could run `family publish` without first running `family asset-check`, allowing a stale shared library or runtime scene reference to ship an oversized or non-web asset.
  - Solution: `publish_family()` now runs the family asset gate before copying public decks and refuses to publish unless the shared asset library and all runtime references pass, unless `--force` is used after explicit review. Verification on 2026-07-04: `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter` passed 148 tests; `family asset-check` reports `max-mb=48.879`, `soft-oversize=0`, and `hard-oversize=0`.
- Problem: It was hard to prove from the top-level asset report that super-large PPT media had actually been converted rather than merely hidden behind a passing max-file-size summary.
  - Solution: `asset_check_family()` now emits `largeSourceConversions` with every source asset above the preferred 50 MiB limit, the optimized runtime file, source/runtime sizes, compression ratio, alpha/animation metadata, and leak/missing/unconverted lists. This report blocks publish if an oversized source remains unconverted, if its runtime file is missing or above the preferred limit, or if a large original source blob is still published in the shared source bucket.
- Latest large-asset verification on 2026-07-04: `py -3 tools\pptx-html-presenter\pptx-html-presenter.py family asset-check presentations\viscereality-family.config.json` passed with `largeSourceAssetSafe=true`. It checked 12 large source assets; all 12 use optimized runtime files. The largest original source was `386.763 MiB`, the largest optimized runtime file was `48.879 MiB`, and the report found 0 unconverted, 0 leaked, and 0 missing large-source runtime assets.
- Large static image nuance from 2026-07-04: MuC's shared master background `ppt/media/image1.png` is only about `29.452 MiB` but is very high resolution (`13333 x 7500`). Temporary 1080p and 4K WebP runtime derivatives were tested against PowerPoint reference frames for `slide-002-settled` and `slide-003-settled`; neither improved SSIM (`1080p` was slightly worse, `4K` was flat-to-slightly-worse). Do not add a blanket pixel-count downsampling policy without an oracle/contact-sheet quality gate. Current production policy should keep converting assets that exceed public delivery size limits, while treating pixel-count optimization as opt-in or QA-proven.
- Problem: Oversized alpha GIFs, opaque GIFs, and videos could stop after one acceptable-under-100-MiB transcode even when the result was still too large for the preferred public asset ceiling.
  - Solution: GIF and video conversion now targets the publish policy, not only the hard limit. Transparent GIFs keep alpha and try alpha-safe WebM/WebP outputs; opaque GIF/video-like loops and large videos try progressively smaller MP4/WebM/WebP variants before falling back to a blocked staged build.
- Problem: Family sharing could upgrade a deck build back to `ok` when the shared library was hard-limit safe but still had preferred-limit violators.
  - Solution: `share_deck_assets()` now only restores `ok` when shared assets are both `githubPagesSafe` and `preferredAssetSafe`; otherwise the staged deck remains `blocked-by-asset-size`.
- Problem: PowerPoint can store a high-fidelity HDPhoto/WDP image layer plus a lower-fidelity PNG fallback for a single visible object. MuC slide 1 `Picture 10` used this for the raster title/text block, but the compiler selected the PNG fallback.
  - Solution: the parser now detects PowerPoint image layers (`a14:imgLayer`), prefers the related `.wdp` media asset for that object, and asset preparation converts WDP to browser-safe PNG via Windows WIC. A diagnostic MuC build rendered slide 1 with the converted WDP title object and improved settled slide-1 SSIM from `0.788991` to `0.865641` against the PowerPoint reference.
- Problem: full family rebuilds are still slow because large transparent GIFs transcode to deterministic VP9-alpha WebM single-threaded.
  - Solution: family builds now seed an optimized-asset cache from the existing shared asset index and the current public/staging scene manifests. `prepare_assets()` consults this cache by original PPT media SHA before starting WDP/GIF/static-image/video conversion, copies the previously accepted runtime file into the local build, and records `optimized-asset-reused-from-shared-cache`. Path-only fallback is used only when no source SHA is available, to avoid accidentally reusing stale media when a future PPTX changes bytes but keeps an internal `ppt/media/...` filename.
- Problem: cached optimized media could previously be reused when it was below the 100 MiB hard ceiling but still above the active 50 MiB public target, causing a future build to inherit a too-large derivative instead of attempting a smaller HTML-friendly conversion.
  - Solution: `_try_reuse_cached_optimized_asset()` now uses the same publish-size gate as fresh conversions. With `allow_oversize_assets: false`, cached derivatives above the preferred public limit are rejected with `cached-optimized-asset-over-publish-limit`, forcing the optimizer/fallback report path to run instead of silently accepting the cached file.
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
- Problem: Candidate sweeps produced large local image folders and reported only the best candidate, which made it too easy to mistake a low-scoring or non-improving sweep for a production calibration.
  - Solution: `candidate-sweep` reports now include `baseline`, `summary.baselineSsim`, `summary.bestDelta`, `summary.improvesBaseline`, and `summary.meaningfulImprovement`. The CLI prints the delta when available. The generated `presentations/*/qa/candidate-sweep/**` frame folders are ignored by Git; keep only deliberately curated conclusions in this status file or in compact QA reports.
- Calibration result: on 2026-07-04, MuC `2->3` exit-fade-end sweeps at samples `25%`, `50%`, and `75%` did not produce a meaningful improvement. Best observed SSIM values were about `0.785622`, `0.781919`, and `0.772079`, matching or slightly under the current failed oracle baseline. Do not add a production exit-fade override for MuC `2->3` based on those sweeps.
- Problem: Family default config values such as `visual_effects.glow_scale` were correctly merged by `_deck_presenter_config()` but were lost when `build_presentation()` copied fields into its title/slug-overlaid `effective_config`.
  - Solution: `build_presentation()` now carries `visual_effects` through to scene compilation, and the build smoke test asserts that non-default glow settings appear in `deck.scene.json`. The rebuilt public manifests now show `runtime.visualEffects.glowScale = 0.5` for `MuC`, `alpCHI`, and `BBD26`.
- Calibration result: on 2026-07-04, candidate sweeps showed a production `glow_scale` of `0.5` improves PowerPoint-oracle slide-1/early-transition samples for BBD26 and alpCHI without changing object structure. The family defaults now set `visual_effects.glow_scale` to `0.5`; `glow_alpha_scale` remains `1.0`.
- Problem: BBD26 visual-audit capture failed on reverse transition `27->26` because visible WebM `track-0109` could remain at `readyState = 0` in deterministic capture mode.
  - Solution: `browser_capture.mjs` now forces visible videos to `preload="auto"`, calls `load()` when metadata/data is absent, waits for metadata/current-frame readiness, seeks visible videos to their sample clocks, and only then screenshots. This does not reverse/restart media in the runtime player; it makes QA capture deterministic for visible media.
- Problem: alpCHI transition `1->2` treated the large title/text SVG block as an unmatched exit/enter pair (`Graphic 2` / `Graphic 15`) because one slide selected the SVG media layer and the next selected the PNG fallback for the same PowerPoint graphic.
  - Solution: `_selected_media_target()` now prefers SVG media over bitmap fallbacks when PowerPoint provides both. The public alpCHI scene now has `Graphic 2` and `Graphic 15` sharing `track-0010` and the same SVG asset, so the title block moves as a Morph object instead of fading in place.
- Problem: alpCHI public visual audit later hit `visible-video-not-ready` on endpoint sample `trans-006-007-100`, with visible WebM `track-0045` stuck at metadata-only `readyState = 1`.
  - Solution: browser capture now briefly plays stubborn visible videos muted if load/seek alone does not decode a current frame, then pauses them before screenshot. This is capture-only determinism; runtime media still loops independently during navigation.
- Problem: MuC transition `1->2` still showed the slide-1 title/text raster layer (`track-0031`, `Picture 10`) too visibly at the 25% PowerPoint-oracle sample, even after global progress calibration. The desired reference frame has already cleared most of that title block by the time the lung/body focus appears.
  - Solution: scene schema and runtime now support per-track unmatched fade overrides. `transition_unmatched_fade_overrides` accepts `track_id`, `track_ids`, or `tracks`, and the runtime checks `transition.unmatchedFade.tracks[trackId]` before falling back to transition/global unmatched fade timing. MuC config now fades `track-0031` out from `0.0` to `0.25` only for transition `1->2`.
- Calibration result: after the MuC `track-0031` per-track fade override and rebuild/publish on 2026-07-04, the targeted candidate check for `MuC` sample `trans-001-002-025` improved from the previous smoke baseline around `0.624` to `0.688544`. This is useful progress but still below the strict Morph target, so do not treat it as an oracle pass.
- Durable asset rule update from user: super-large source assets should be converted into visually lossless or visually acceptable HTML-friendly runtime formats before public publish. Preserve transparency and looping semantics when needed; use video formats for opaque/video-like animations when smaller; keep runtime files GitHub-compatible and do not publish giant originals merely for provenance.
- Asset rule revalidation on 2026-07-04: the public shared library still satisfies the GitHub-friendly runtime policy after the latest MuC oracle refresh. Direct filesystem gate: 81 public shared files, 362.645 MiB total, 0 files above 50 MiB, 0 files above 100 MiB, largest file `optimized/3a907ddc7cdfe95de185fc64f27eaf69f5251c46deff568927ade6b6b9bca5b9.mp4` at 48.879 MiB. Continue treating >50 MiB public files as blockers unless explicitly reviewed for non-GitHub hosting.
- Problem: The first-transition oracle failures involve object clusters, especially panel/frame/media clusters, but `candidate-sweep --vary track-progress` could previously vary only one track at a time. That made it too hard to test the user's core requirement that grouped/panel contents move together.
  - Solution: candidate sweeps now accept comma-separated track clusters for `--track-id` on `track-progress` and `phase` sweeps. For example, `--track-id track-0011,track-0012` applies the same candidate progress or media phase to both tracks and records `candidateSweep.trackIds` in the generated samples.
- Problem: Very large comma-separated track clusters in `candidate-sweep --track-id` could produce output folder names long enough to fail on Windows with `WinError 123`.
  - Solution: candidate-sweep output folder/sample IDs now compact long track labels to `tracks-<count>-<hash>`, while preserving the full `candidateSweep.trackIds` metadata in `samples.json` and reports. This keeps panel-cluster sweeps usable without losing provenance.
- Calibration result: on 2026-07-04, alpCHI `trans-001-002-025` was tested with global progress, `track-0010` title progress, `track-0011` panel/media progress, media phase for `track-0011`, and clustered `track-0011,track-0012` progress. The best cluster score was only `0.518489`, and the best individual tests stayed around `0.52`, so do not commit those as production overrides. The remaining alpCHI gap is likely a combination of PowerPoint export timing, full-frame composition, object effects, and/or deeper grouping semantics rather than a simple per-track progress tweak.
- Problem: A broader MuC PowerPoint-oracle pass for slides `1-3` found a worse failure after the opener: transition `2->3` midpoint scored only `0.621540`. The side-by-side showed PowerPoint effectively holding slide 2 through the early/mid samples and then reaching slide 3 quickly, while HTML interpolated linearly and left the incoming measurement panel too far to the right.
  - Solution: MuC config now adds a transition `2->3` progress map that holds at `0.0` through progress `0.5` and jumps to `1.0` by progress `0.75`, mirrored for reverse navigation by the runtime's existing progress-map mirror.
- Calibration result: after rebuilding/publishing that MuC `2->3` hold-then-snap map, rerunning `family oracle-qa --decks MuC --slides 1-3 --target public --force --min-free-gb 0` improved the bounded MuC minimum SSIM from `0.621540` to `0.658322`. Transition `2->3` samples improved from `0.747345 -> 0.785634` at 25%, `0.621540 -> 0.781949` at 50%, `0.647073 -> 0.742557` at 75%, and `0.803880 -> 0.853141` at 90%. Strict oracle thresholds are still not met.
- Problem: MuC transition `3->4` still had carousel-panel ghosting and an oracle midpoint around `0.710758`, with the HTML Morph timing lagging the PowerPoint reference.
  - Solution: candidate sweeps were run against the PowerPoint oracle frames for `trans-003-004` and MuC config now adds a `3->4` progress map: `0%=0.0`, `10%=0.1`, `25%=0.35`, `50%=0.6`, `75%=0.8`, `90%=0.9`, `100%=1.0`. This is mirrored by the runtime for reverse navigation.
- Calibration result: after rebuilding/publishing the MuC `3->4` map, the bounded MuC slides `1-3` oracle run still fails strict thresholds with minimum SSIM `0.658322`, but `3->4` improves at the most visually important early/mid samples: `25% 0.776646 -> 0.809591`, `50% 0.710758 -> 0.753245`, and `100% 0.825763 -> 0.825751` effectively unchanged. It regresses `10% 0.895287 -> 0.837739`, `75% 0.769792 -> 0.736685`, and `90% 0.805204 -> 0.743021`, so treat this as a partial timing calibration, not solved oracle parity.
- Problem: MuC transition `1->2` at 25% was still the bounded oracle minimum at `0.658322`. The side-by-side showed the PowerPoint reference using a gentler crossfade between the slide-1 animated person/background layer and the slide-2 lung/background/title layers than the global unmatched fade window (`0.5 -> 0.75`) provided.
  - Solution: `candidate-sweep` now supports track-scoped unmatched fade sweeps for `enter-fade-end` and `exit-fade-end` via `--track-id`, including comma-separated track clusters. This lets agents score fade timing for one object or group without disturbing the full transition.
- Calibration result: MuC config now keeps `track-0003` (slide-1 person/background WebM) fading out across the full `1->2` transition and fades in slide-2 tracks `track-0032` through `track-0036` across the full transition, while preserving the earlier fast exit for `track-0031` (title raster). After rebuilding/publishing and rerunning `family oracle-qa --decks MuC --slides 1-3 --target public --force --min-free-gb 0`, bounded MuC minimum SSIM improved from `0.658322` to `0.736685`. The targeted `trans-001-002-025` sample improved to `0.740554`. Strict oracle thresholds are still not met; the new worst sample is now `trans-003-004-075`.
- Problem: After the `1->2` crossfade fix, MuC `3->4` late samples became the new oracle floor: `trans-003-004-075` was `0.736685` and `trans-003-004-090` was `0.743021`. Visual inspection showed the second carousel panel content lagging behind the PowerPoint reference.
  - Solution: fresh global progress candidate sweeps for `3->4` showed the transition should hold at 10%, move less at 25%, keep roughly the same midpoint, and then accelerate hard: best raw candidates were `10%=0.0`, `25%=0.2`, `50%=0.6`, `75%=0.95`, and `90%=1.0`. Because the deck already had a progress map, those raw sweep positions were translated into mapped interpolation values in the production config: `0%=0.0`, `10%=0.0`, `25%=0.2667`, `50%=0.68`, `75%=0.95`, `90%=1.0`, `100%=1.0`.
- Calibration result: after rebuilding/publishing that `3->4` v2 map and rerunning `family oracle-qa --decks MuC --slides 1-3 --target public --force --min-free-gb 0`, bounded MuC minimum SSIM improved from `0.736685` to `0.740554`. The `3->4` late samples improved substantially: `75% 0.736685 -> 0.790713`, `90% 0.743021 -> 0.791718`, and `25% 0.809591 -> 0.853097`. The strict oracle threshold is still not met; the worst samples are now `trans-001-002-025` (`0.740554`) and `trans-002-003-075` (`0.740774`).
- Calibration result: a fresh bounded MuC public oracle refresh on 2026-07-04 still fails strict QA with minimum SSIM `0.739402`. Worst samples: `trans-002-003-075` (`0.739402`), `trans-001-002-025` (`0.740554`), `trans-002-003-050` (`0.781949`), `trans-002-003-025` (`0.785634`), and `slide-001-settled` (`0.788892`).
- Diagnostic result: MuC `trans-001-002-025` global progress sweep best was raw `0.25` at SSIM `0.780312`; `track-0003` phase-offset sweep best was `-0.5s` at SSIM `0.793062`. MuC `slide-001-settled` phase-offset sweep for `track-0003` scored `0.975002` for several negative offsets, but the transition sample prefers a different offset. Do not add a transition-only media-phase hack unless it preserves continuous loop playback; this needs a coherent PowerPoint-vs-browser media-clock model.
- Diagnostic result: MuC `trans-002-003-075` global progress sweep best was endpoint `1.0` at SSIM `0.769885`, while global, central-panel, and broad incoming enter/exit fade sweeps were neutral around `0.746083`. Visual inspection of the single 1920x1080 frames suggests the remaining gap is not simple timing: HTML panel fills/text render more opaque/larger than the PowerPoint reference. Next likely work is shape fill/opacity semantics, panel-local text autofit/metrics, and possibly panel fill vs outline opacity separation.
- Rejected experiment: on 2026-07-04, applying raw-time/global unmatched fade behavior to inferred synthetic Morph endpoints regressed bounded MuC oracle QA badly (`trans-001-002-025` dropped to `0.441867` and `trans-001-002-050` to `0.536599`). The experiment was backed out before commit. Future agents should not make inferred panel/object Morph endpoints obey global unmatched fade timing unless a focused oracle sweep proves it improves the target samples without harming transition `1->2`.
- Problem: MuC `trans-001-002-025` still had a large visual gap even after geometry/fade calibration. A phase-offset candidate sweep for looped slide-1 track `track-0003` found that the PowerPoint MP4 oracle is closer when that loop starts at a different phase (`+2.75s` scored `0.793067` for `trans-001-002-025`; slide 1 settled can reach about `0.975` under several equivalent loop offsets).
  - Solution: `presentations/MuC-scene.config.json` now applies a deck-specific `media_phase_overrides` row for slide 1 `track-0003` / asset `asset-0928f3a3fc7358dc` with `phase_sec: 2.75`. This preserves the independent looping-media model; it does not reverse/restart media during Morph.
- Problem: MuC transition `2->3` had a bogus inferred carousel foreground motion for `track-0004`, a bottom sponsor/footer strip that is missing on slide 3. The object was being kept opaque and moved with the carousel, making the 75% oracle frame worse and violating the intended panel/cluster logic.
  - Solution: the inferred panel/foreground motion heuristic now excludes shallow, wide objects in the global slide footer region. The regression test `test_inferred_motions_do_not_slide_footer_sponsor_strips_with_carousel` ensures footer sponsor strips are not treated as panel children or carousel foreground, while the existing large VR/body foreground test still passes.
- Problem: The remaining MuC `2->3` oracle gap visibly includes PowerPoint-vs-browser text metric differences: HTML text was slightly too large/heavy inside the moving panel, but candidate sweeps previously had no way to isolate text rendering from Morph timing.
  - Solution: the presenter now has a reusable `text_rendering` runtime policy (`font_scale`, `regular_weight`, `bold_weight`) and `candidate-sweep --vary text-scale|bold-weight` diagnostics. Capture plumbing passes `textRenderOverrides` into the browser runtime so future agents can test text metric hypotheses without manually editing generated HTML. Focused sweeps on MuC showed `font_scale: 0.9` improved `trans-002-003-075` from about `0.766` to about `0.772`, and improved `slide-003-settled` from about `0.898814` to `0.902273`, so `presentations/MuC-scene.config.json` now applies `text_rendering.font_scale: 0.9` only for MuC. This is a small oracle-parity gain, not a pass; the remaining gap is still mostly full-frame composition/effects/timing.
- Problem: Some remaining oracle failures looked as if PowerPoint may be compositing matched Morph objects with different opacity/alpha than the HTML runtime, but the existing candidate sweeps could only vary unmatched fades, not the opacity of already-matched objects.
  - Solution: capture mode now supports `trackOpacityOverrides`, and `candidate-sweep --vary track-opacity|object-opacity|opacity --track-id ...` generates track-scoped opacity multiplier candidates. This is capture-only diagnostic plumbing; it does not change public playback unless a future verified config feature deliberately promotes an opacity rule.
- Diagnostic result: on 2026-07-04, targeted track-opacity sweeps did not materially improve the current worst oracle samples. For alpCHI `trans-001-002-025`, `track-0010`, `track-0011`, `track-0012`, and the broad cluster `track-0003..track-0008,track-0010` all bested at opacity `0.0` with SSIM only about `0.512105`, barely above the existing failed baseline around `0.508`. For MuC `trans-002-003-075`, a panel/content cluster also bested at `0.0` with SSIM about `0.772081`, effectively unchanged from the current floor. Treat opacity as a useful diagnostic axis, not an accepted production fix.
- Problem: inherited PowerPoint background image fills were emitted as plain media even though `<p:bg>` can carry the same crop, media alpha, luminance, and visual-effect metadata as a normal picture fill.
  - Solution: `_background_object()` now propagates background `crop`, media opacity, `visualEffects`, and `mediaEffects` into scene objects. `_media_effects()` now also reads DrawingML `<a:blip><a:lum bright="..." contrast="..."/></a:blip>` into the existing `brightnessContrast` runtime filter path. Verification: `test_inherits_master_background` now asserts inherited background `srcRect` and luminance metadata, and the full `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter` suite passed 137 tests.
- Problem: DrawingML color transforms were previously discarded. PowerPoint uses `lumMod`, `lumOff`, `tint`, `shade`, `satMod`, `satOff`, and `alpha` on theme/RGB colors in these decks, especially MuC text colors and alpha-filled overlay rectangles.
  - Solution: `_solid_color()` / `_color_from_node()` now preserve transformed scheme colors as deterministic runtime tokens such as `scheme:tx1|lumMod=50000|lumOff=50000`, resolve transformed RGB colors to CSS `#rrggbb`/`rgba(...)`, and the browser runtime resolves transformed scheme colors against its theme color map. The runtime also resolves transformed colors for outline detection and glow colors. Important nuance: shape fill alpha is still represented once through object `opacity`; the parser intentionally strips alpha from shape fill color tokens to avoid double-applying transparency. Text colors and stroke colors can still carry alpha transforms.
  - Verification: full `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter` passed 142 tests. Family rebuild and publish passed. `family asset-check` passed with max public runtime asset 48.879 MiB and no oversize assets. `family visual-audit` passed all three decks after the corrected color/alpha build. Public manifests now show MuC transformed rich-text colors while alpCHI/BBD26 alpha overlay rectangles render as `fill: scheme:tx1` plus `opacity: 0.62`, avoiding the rejected double-alpha regression.
- Diagnostic result: on 2026-07-04, MuC `trans-002-003-075` was checked against the current PowerPoint reference after the opacity diagnostics. Media phase sweeps for the two visible panel animation tracks were not useful production fixes: `track-0022` bested at `-3.0s` with SSIM `0.772273`, and `track-0023` bested at `+0.25s` with SSIM `0.772059`, essentially unchanged from the current failed floor. A `trans-002-003-090` phase sweep for `track-0022` bested at `+2.75s` with SSIM `0.883435`, only a small improvement. Global progress sweeps for `trans-002-003-050/075/090` also produced no meaningful improvement (`0.0`, `0.75`, and `0.75` best raw values respectively). A quick offline brightness/contrast transform suggested only tiny transition-frame gains. Do not commit a MuC 2->3 phase/progress/brightness override without a stronger scoped oracle result.

Current shared public asset library check after the visual-effects/public-audit rebuild:

- `presentations/shared-assets/viscereality/` contains 81 runtime/source files, about 362.645 MiB total.
- Largest shared asset is about 48.879 MiB.
- Files above 50 MiB: 0.
- Files above 100 MiB: 0.
- Optimized cache reuse in the latest build:
  - `MuC`: 15 cached optimized assets reused.
  - `alpCHI`: 21 cached optimized assets reused.
  - `BBD26`: 22 cached optimized assets reused.
- The latest build also converted one WDP/HDPhoto-derived asset per public deck into shared PNG runtime assets.
- Family builds now emit `sharedAssetLimits` with `preferredAssetSafe`, `softOversizeAssets`, and `oversizeAssets` so future agents can verify the public shared library gate without hunting through per-deck reports.
- Latest direct filesystem gate check on 2026-07-04: 81 shared public asset files, 362.645 MiB total, 0 files above 50 MiB, 0 files above 100 MiB. The largest runtime file is `optimized/3a907ddc7cdfe95de185fc64f27eaf69f5251c46deff568927ade6b6b9bca5b9.mp4` at about 48.879 MiB. This confirms the current public build follows the visually-lossless/html-friendly/GitHub-compatible asset rule.
- Latest stricter asset-policy/tooling verification on 2026-07-04: `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter` passed 130 tests, including soft-limit blocker tests, track-scoped candidate-sweep fade tests, stable browser-capture diagnostic asset paths, and compact output IDs for long clustered sweeps. A direct shared-library filesystem gate again found 81 files, 362.645 MiB total, 0 files above 50 MiB, 0 files above 100 MiB, largest about 48.879 MiB.
- Latest post-publish shared-asset gate after the MuC per-track fade rebuild on 2026-07-04: `family-build-report.json` reports `githubPagesSafe: true`, `preferredAssetSafe: true`, `maxAssetMb: 48.879`, `softOversizeAssets: []`, `oversizeAssets: []`, 31 optimized files, 45 source/provenance-safe files, and about 362.57 MiB total. This confirms the current public build still follows the super-large-asset conversion policy.
- Latest post-rebuild filesystem asset gate on 2026-07-04: `presentations/shared-assets/viscereality/` contains 81 public shared files, about 362.645 MiB total, 0 files above 50 MiB, 0 files above 100 MiB. Largest files are optimized MP4s at about 48.879 MiB, 42.672 MiB, 37.708 MiB, and 34.916 MiB. This is the current proof point for the rule that super-large PPT assets must be converted to visually acceptable, HTML-friendly, GitHub-compatible runtime formats.
- Latest post-MuC-calibration asset gate on 2026-07-04: still 81 public shared files, about 362.645 MiB total, 0 files above 50 MiB, 0 files above 100 MiB, largest asset about 48.879 MiB. The clean family build report generated at `2026-07-04T02:34:29+00:00` has sane public-size preflight numbers again after deleting local candidate-sweep scratch output: `MuC` about 433.147 MiB, `alpCHI` about 368.839 MiB, `BBD26` about 452.66 MiB.
- Latest explicit user constraint revalidated on 2026-07-04: super-large PPT assets must never be published as giant public blobs just for convenience or provenance. The family defaults enforce `allow_oversize_assets: false`, `transcode_gif: true`, `transcode_video: true`, `optimize_static_images: true`, and `transparent_animation: preserve-alpha`. Targeted asset tests passed (`py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter -k asset`, 13 tests). Direct filesystem gate still reports 81 shared public files, about 362.645 MiB total, 0 files above 50 MiB, 0 files above 100 MiB, largest about 48.879 MiB. `presentations/shared-assets/viscereality/family-build-report.json` reports `status: ok`, `githubPagesSafe: true`, `preferredAssetSafe: true`, `softOversizeAssets: []`, and `oversizeAssets: []`. The shared index has 76 content-hashed asset records, including source SHA metadata for all entries, so future builds can reuse optimized outputs without re-copying oversized originals.
- Latest post-text-calibration asset gate on 2026-07-04: after rebuilding/publishing all three public scene players with the shared text-rendering runtime, the shared public asset library is unchanged at 81 files, about 362.645 MiB total, 0 files above 50 MiB, 0 files above 100 MiB, largest about 48.879 MiB. `family-build-report.json` still reports `status: ok`, `githubPagesSafe: true`, and `preferredAssetSafe: true`.
- Latest asset-cache hardening on 2026-07-04: cached optimized assets now must pass the active publish-size policy before reuse, so an old derivative above the 50 MiB preferred ceiling will be rejected and regenerated rather than carried forward. Verification: `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter -k asset` passed 14 tests, full `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter` passed 134 tests, and the direct shared-library gate still reports 81 public files, 362.645 MiB total, 0 files above 50 MiB, 0 files above 100 MiB, largest about 48.879 MiB.
- Latest explicit oversized-asset gate on 2026-07-04: added `pptx-html-presenter family asset-check <family-config>`, which writes `presentations/shared-assets/viscereality/family-asset-check-report.json` and exits nonzero if any public shared asset exceeds either the 50 MiB preferred ceiling or 100 MiB hard ceiling. Verification: `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter -k asset` passed 15 tests, and `py -3 tools\pptx-html-presenter\pptx-html-presenter.py family asset-check presentations\viscereality-family.config.json` reported `family-asset-check=ok max-mb=48.879 soft-oversize=0 hard-oversize=0`. The report lists 31 optimized runtime files, 45 source/provenance-safe files, 362.57 MiB total, and no oversize assets. Future agents must run this gate after touching media, rebuild/publish, or shared-asset hoisting.
- Latest runtime-format asset gate hardening on 2026-07-04: `family asset-check` now scans the public and staging `deck.scene.json` manifests in addition to the shared library. Every runtime `asset.file` must resolve to an actual file, stay under the same 50 MiB preferred / 100 MiB hard public limits, and use a browser-friendly runtime extension (`gif`, `jpeg`, `jpg`, `mp4`, `png`, `svg`, `webm`, or `webp`). This prevents future builds from accidentally pointing the HTML player at PowerPoint-native or non-web files such as WDP/HDPhoto, TIFF, or BMP, even if the file is small. Verification: full `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter` passed 138 tests, and `family asset-check` passed with 6 scene manifests checked, 288 runtime asset references, `runtimeFormatSafe: true`, `runtimeFilesExist: true`, `runtimePreferredAssetSafe: true`, no missing runtime files, no unsupported runtime formats, no runtime assets above 50 MiB, and max runtime asset size 48.879 MiB.
- Latest clustered-Morph config support on 2026-07-04: `transition_track_progress_overrides` now accepts `track_ids`, `trackIds`, or `tracks` in addition to a singular `track_id`. The scene compiler expands clustered rows into deterministic per-track runtime rows, letting future calibration express "this white panel plus its children use one Morph progress curve" directly in deck config instead of duplicating rows by hand. Verification: targeted clustered-track tests passed and full `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter` passed 144 tests. Diagnostic alpCHI sweeps confirmed this is the right abstraction for panel/title clusters, although no new production alpCHI timing override was accepted yet because strict oracle gains remained small.
- Latest asset gate revalidation on 2026-07-04: `py -3 tools\pptx-html-presenter\pptx-html-presenter.py family asset-check presentations\viscereality-family.config.json` reports `family-asset-check=ok max-mb=48.879 soft-oversize=0 hard-oversize=0`. The refreshed report checks 6 public/staging scene manifests, 288 runtime asset references, browser-friendly runtime formats only, 31 optimized runtime files, 45 source/provenance-safe files, about 362.57 MiB total, and no file above the 50 MiB preferred ceiling. This directly enforces the user constraint that super-large PPT assets must be converted to visually lossless or visually acceptable HTML-friendly formats before public publish.
- Latest multi-cluster candidate sweep support on 2026-07-04: `candidate-sweep --vary track-progress-matrix` now tests semicolon-separated track clusters with independent progress values, for example `--track-id "track-0010;track-0011,track-0012"`. This renders the Cartesian product of candidate values and writes ordinary `trackProgressOverrides` into each capture sample, allowing title clusters, panel clusters, and child clusters to be scored together against a PowerPoint frame. Verification: full `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter` passed 145 tests.
- Latest alpCHI 1->2 calibration on 2026-07-04: a matrix sweep on `trans-001-002-025` found `track-0010` title progress `0.70` and `track-0011,track-0012` panel/image progress `0.35` improved the scored reference-frame SSIM from `0.506195` to `0.528427`. Adjacent coarse sweeps suggested a monotonic production curve with title progress reaching `0.8` by raw `0.1` and panel progress `0.0 -> 0.35 -> 0.8` over raw `0.1/0.25/0.5`. This was added to `presentations/alpCHI-scene.config.json`, rebuilt through family build, published to `/presentations/alpCHI/deck.scene.json`, and direct `visual-audit presentations\alpCHI` passed 217 samples with 0 failures. This is a bounded improvement, not a strict oracle pass.
- Latest asset-size revalidation after the alpCHI calibration build on 2026-07-04: `family asset-check` still reports `family-asset-check=ok max-mb=48.879 soft-oversize=0 hard-oversize=0`. This confirms the current public scene players and shared library keep every runtime asset below the 50 MiB preferred GitHub-friendly ceiling after the latest rebuild/publish.
- Latest BBD26 current-source remap on 2026-07-04: the current BBD26 source deck has 32 slides, but several deck-specific fixes in `presentations/BBD26-scene.config.json` still targeted the earlier 26-slide build. This caused old carousel overrides to apply to current slide `19->20` tracks that are no longer the video carousel objects. The config now remaps the old carousel block from slides `16-21` onto current slides `20-25`: decorative `Picture 48` is `track-0093`, carousel videos/panels use `track-0094`, `track-0097`, `track-0101`, `track-0102`, `track-0103`, and `track-0104`, and the old full-slide static fallback / transition-time override file references were removed. After rebuild/publish, public `/presentations/BBD26/deck.scene.json` shows `mediaPhaseOverridesApplied: 5`, `rasterFallbacksApplied: 0`, no stale custom media/track rows on transitions `18->19` or `19->20`, the remapped media row on `22->23`, and remapped track progress rows on `23->24`. Verification: `visual-audit presentations\BBD26` passed 280 samples; `family asset-check` reports `max-mb=48.879 soft-oversize=0 hard-oversize=0`; full unit tests passed 145 tests. This fixes a current-object-graph correctness issue, not strict PowerPoint oracle parity.
- Latest stale-override guard on 2026-07-04: the compiler now emits `qa.configOverrideValidation` in every `deck.scene.json`, copies summary fields into `build-report.json`, and marks builds as `blocked-by-stale-overrides` when manual config rows point at missing slides, missing transitions, missing tracks, missing media objects, ambiguous media phase targets, or skipped/stale raster fallback rows. Publish also refuses non-`ok` build statuses unless explicitly forced. This is meant to catch exactly the class of silent drift found in BBD26 when a new PPTX revision changed slide and track identities. Verification: focused stale/healthy override tests and full `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter` passed 147 tests; `family build --force` reports `configOverrideSafe=True` for MuC, alpCHI, and BBD26 with 0 blockers; `family asset-check` still reports `max-mb=48.879 soft-oversize=0 hard-oversize=0`. Current warning-only rows: alpCHI has 2 track-only progress calibrations, and BBD26 has 2 track-only carousel progress calibrations. These warnings do not block, but future agents should add object/name/asset selectors to calibrated rows when practical because raw track IDs can shift between PPTX revisions.
- Calibration result: on 2026-07-04, a `morph-progress` full-frame calibration pass regenerated the MuC `1->2` and `3->4` transition progress maps in `presentations/MuC-scene.config.json`, tagged `MuC-slide-1-2-morph-progress-full-frame-calibration-2026-07-04` and `MuC-slide-3-4-morph-progress-full-frame-calibration-2026-07-04`. `1->2` now maps `0.25=0.6`, `0.5=0.85`, `0.75=0.95`; `3->4` now maps `0.25=0.3`, `0.5=0.7`. After family rebuild/publish, the bounded MuC slides `1-3` public oracle run improved the minimum SSIM from `0.768571` to `0.772084` across 24 comparisons. Status remains `failed` against the strict `0.965` Morph threshold; treat this as an incremental timing gain, not oracle parity.
- Housekeeping in the same 2026-07-04 rebuild/publish: the republished public manifests for MuC, alpCHI, and BBD26 now embed the `qa.configOverrideValidation` summary (MuC 0 warnings; alpCHI and BBD26 each carry the 2 known warning-only track-only-selector calibration rows). Stale per-deck legacy oracle artifacts `qa/report.json`, `qa/samples.json`, and `qa/contact-sheet.html` were removed from `presentations/BBD26/` and `presentations/alpCHI/` because they described superseded single-slide smoke runs; `presentations/MuC/qa/report.json` now reflects the latest bounded slides `1-3` run. Verified on 2026-07-05: full unit suite passes 151 tests, `family asset-check` reports `max-mb=48.879 soft-oversize=0 hard-oversize=0`, per-deck `build-report.json` all `ok`, visual audits all `passed`, `family-publish-report.json` `ok`.
- Live-site verification on 2026-07-05: `main` was fast-forwarded to the scene-player branch and GitHub Pages deployed commit `08623b7`. A Playwright pass against `https://viscereality.org/presentations/` verified: hub returns 200 with all three deck cards; each public player returns 200 with the expected manifest (`generatedAtUtc` 2026-07-04T05:17/05:18, 17/25/32 slides); zero console errors and zero failed asset requests inside all three players; all visible videos playing (loop wraparound observed, confirming independent media clocks); forward ArrowRight Morph mid-frames, ArrowLeft reverse Morph mid-frames, settled round-trips, and panel/carousel cluster motion (MuC 3->4, alpCHI 1->2, BBD26 22->23) all render correctly on the live site.
- Problem found during live verification: the published hub `presentations/index.html` and `presentations/shared/decks.js` referenced `<deck>/preview.jpg` thumbnails that no pipeline step ever generated, producing three 404s and broken-image cards on the live hub.
  - Solution: preview JPGs (1280x720, quality 82, from live settled slide-1 captures) now live at `presentations/shared-assets/viscereality/previews/<deckId>.jpg`, which `publish_family()` does not wipe (it only rmtrees public deck folders). `viscereality-family.config.json` now sets `preview_image` per deck to the absolute `/presentations/shared-assets/viscereality/previews/<deckId>.jpg` path so future `family publish` regenerates the hub/decks.js with the same references, and the currently generated hub/decks.js were updated to match. `family asset-check` still reports `max-mb=48.879 soft-oversize=0 hard-oversize=0` with the previews included. If a deck's slide 1 changes visually, regenerate its preview JPG from a fresh settled capture.

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

After adding the per-track unmatched fade override support and republishing MuC on 2026-07-04, direct public-folder audits again passed:

- `MuC`: 145 samples, 17 settled slides, 112 forward transition samples, 16 reverse midpoint samples, 0 failures, 0 warnings.
- `alpCHI`: 217 samples, 25 settled slides, 168 forward transition samples, 24 reverse midpoint samples, 0 failures, 0 warnings.
- `BBD26`: 280 samples, 32 settled slides, 217 forward transition samples, 31 reverse midpoint samples, 0 failures, 0 warnings.

This confirms the new runtime fade-map logic did not break settled rendering, forward Morph captures, reverse midpoint captures, or shared-media loading. It is still not a PowerPoint-oracle SSIM pass.

After adding the MuC `2->3` hold-then-snap progress map on 2026-07-04, direct public MuC visual audit passed again:

- `MuC`: 145 samples, 17 settled slides, 112 forward transition samples, 16 reverse midpoint samples, 0 failures, 0 warnings.

This validates the new transition map does not create browser capture failures or obvious reverse-transition structural problems. It is still not a PowerPoint-oracle SSIM pass.

After adding the MuC `3->4` carousel-speed progress map on 2026-07-04, direct public MuC visual audit passed again:

- `MuC`: 145 samples, 17 settled slides, 112 forward transition samples, 16 reverse midpoint samples, 0 failures, 0 warnings.

This validates that the new `3->4` progress map remains playable forward and backward, keeps media clocks available during capture, and does not introduce blank/partial frames. It is still not a PowerPoint-oracle SSIM pass.

After adding track-scoped unmatched fade sweep support and the MuC `1->2` slow crossfade overrides on 2026-07-04, direct public MuC visual audit passed again:

- `MuC`: 145 samples, 17 settled slides, 112 forward transition samples, 16 reverse midpoint samples, 0 failures, 0 warnings.

This validates that the track-scoped crossfade calibration did not introduce blank frames, capture failures, or forward/reverse browser-playback regressions. It is still not a PowerPoint-oracle SSIM pass.

After adding the MuC `3->4` v2 late-accelerating progress map on 2026-07-04, direct public MuC visual audit passed again:

- `MuC`: 145 samples, 17 settled slides, 112 forward transition samples, 16 reverse midpoint samples, 0 failures, 0 warnings.

This validates that the revised carousel timing remains playable forward and backward and does not create browser-capture failures. It is still not a PowerPoint-oracle SSIM pass.

After the final 2026-07-04 clean rebuild/publish and asset-gate verification, full family visual audit passed again:

- `MuC`: passed.
- `alpCHI`: passed.
- `BBD26`: passed.

This verifies the public scene players still load shared assets correctly and capture settled slides, forward transition samples, and reverse midpoint samples without browser failures. It remains separate from strict PowerPoint-oracle parity.

After the MuC slide-1 media phase override and footer-strip inferred-motion exclusion on 2026-07-04:

- `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter` passed 131 tests.
- Direct `visual-audit presentations\MuC` passed 145 samples.
- Full `family visual-audit presentations\viscereality-family.config.json` passed all 3 decks again.

This verifies the phase/config change and shared heuristic do not break browser playback/capture. It is still not a PowerPoint-oracle SSIM pass.

After adding text-rendering runtime/candidate-sweep support and applying MuC `text_rendering.font_scale: 0.9` on 2026-07-04:

- `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter` passed 133 tests.
- Direct `visual-audit presentations\MuC` passed 145 samples.
- Full `family visual-audit presentations\viscereality-family.config.json` passed all 3 decks again:
  - `MuC`: 145 samples, 17 settled slides, 112 forward transition samples, 16 reverse midpoint samples, 0 failures, 0 warnings.
  - `alpCHI`: 217 samples, 25 settled slides, 168 forward transition samples, 24 reverse midpoint samples, 0 failures, 0 warnings.
  - `BBD26`: 280 samples, 32 settled slides, 217 forward transition samples, 31 reverse midpoint samples, 0 failures, 0 warnings.

This verifies the shared runtime update and MuC text-scale calibration do not break browser playback/capture. It is still not a PowerPoint-oracle SSIM pass.

After adding DrawingML color-transform parsing/runtime resolution and fixing shape fill alpha so it is not double-applied on 2026-07-04:

- `py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter` passed 142 tests.
- `py -3 tools\pptx-html-presenter\pptx-html-presenter.py family build presentations\viscereality-family.config.json --force ...` passed.
- `py -3 tools\pptx-html-presenter\pptx-html-presenter.py family publish presentations\viscereality-family.config.json --force --no-archive-chunked` passed.
- `py -3 tools\pptx-html-presenter\pptx-html-presenter.py family asset-check presentations\viscereality-family.config.json` passed.
- `py -3 tools\pptx-html-presenter\pptx-html-presenter.py family visual-audit presentations\viscereality-family.config.json ...` passed all three decks.

This validates browser playback/capture and asset safety for the corrected color-transform build. It is still not a PowerPoint-oracle SSIM pass.

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

Additional bounded MuC oracle result after the `2->3` hold-then-snap calibration:

- `MuC` slides `1-3`: status `failed`, no blockers, 24 comparisons, minimum SSIM `0.658322`. This is better than the pre-calibration bounded run minimum `0.621540`, but all samples still remain below strict slide/transition thresholds.

Additional bounded MuC oracle result after the subsequent `3->4` carousel-speed calibration:

- `MuC` slides `1-3`: status `failed`, no blockers, 24 comparisons, minimum SSIM `0.658322`. `3->4` early/mid samples improved, but late samples regressed; the main remaining blocker is still first-transition and full-composition parity, not browser playability.

Additional bounded MuC oracle result after the subsequent `1->2` track-scoped crossfade calibration:

- `MuC` slides `1-3`: status `failed`, no blockers, 24 comparisons, minimum SSIM `0.736685`. `trans-001-002-025` improved from `0.658322` to `0.740554`; the worst current samples are now `trans-003-004-075` (`0.736685`) and `trans-003-004-090` (`0.743021`). The remaining gap is still large and should be treated as unresolved PowerPoint-oracle parity work.

Additional bounded MuC oracle result after the subsequent `3->4` v2 progress-map calibration:

- `MuC` slides `1-3`: status `failed`, no blockers, 24 comparisons, minimum SSIM `0.740554`. `trans-003-004-075` improved to `0.790713`, `trans-003-004-090` improved to `0.791718`, and `trans-003-004-025` improved to `0.853097`. Overall strict parity is still unresolved; current bottlenecks are `1->2` at 25% and `2->3` at 75%.

Additional bounded MuC oracle regression check after the final 2026-07-04 clean rebuild/publish:

- `MuC` slides `1-3`: status `failed`, 24 comparisons, minimum SSIM `0.740554`. Lowest samples were `trans-001-002-025` (`0.740554`), `trans-002-003-075` (`0.743023`), `trans-002-003-050` (`0.781949`), `trans-002-003-025` (`0.785634`), and `slide-001-settled` (`0.788892`). This confirms the rejected inferred-fade experiment was not present in the public output, but strict oracle parity is still unresolved.

Additional bounded MuC oracle result after the MuC media-phase and footer-strip inferred-motion fixes:

- `MuC` slides `1-3`: status `failed`, 24 comparisons, minimum SSIM improved to `0.766376`. Current lowest samples are `trans-002-003-075` (`0.766376`), `trans-002-003-050` (`0.781949`), `trans-002-003-025` (`0.785634`), `trans-001-002-010` (`0.789210`), and `trans-001-002-025` (`0.789845`). This is real progress from the prior `0.740554` floor, but still below the strict transition target `0.965`.

Additional bounded MuC oracle result after the MuC `text_rendering.font_scale: 0.9` calibration:

- `MuC` slides `1-3`: status `failed`, 24 comparisons, minimum SSIM improved from `0.766376` to `0.772079`. Current lowest samples are `trans-002-003-075` (`0.772079`), `trans-002-003-050` (`0.781949`), `trans-002-003-025` (`0.785634`), `trans-001-002-010` (`0.789210`), `trans-003-004-075` (`0.792089`), and `trans-001-002-025` (`0.792934`). This confirms text metrics contribute to the mismatch, but the strict `0.965` Morph threshold remains unresolved.

Additional oracle refresh after DrawingML color-transform parsing/runtime resolution and the corrected single-application shape fill alpha:

- `MuC` slides `1-3`: status `failed`, 24 comparisons, minimum SSIM `0.768571`. Current lowest samples are `trans-002-003-075` (`0.768571`), `trans-002-003-050` (`0.781917`), `trans-002-003-025` (`0.785626`), `trans-001-002-010` (`0.789210`), and `trans-003-004-075` (`0.792079`).
- `alpCHI` slide `1`: status `failed`, 8 comparisons, minimum SSIM `0.506195`. Current lowest samples are `trans-001-002-025` (`0.506195`), `trans-001-002-010` (`0.608198`), and `trans-001-002-050` (`0.746171`).
- `BBD26` slide `1`: status `failed`, 8 comparisons, minimum SSIM `0.742680`. Current lowest samples are `trans-001-002-000` (`0.742680`), `trans-001-002-010` (`0.774193`), and `slide-001-settled` (`0.814801`).

Interpretation: color transforms are now represented more faithfully in the scene/runtime, but strict oracle parity remains unresolved and the bounded scores are roughly comparable to the previous failed baseline rather than a pass. A temporary build that applied shape fill alpha both in `fill` and object `opacity` regressed BBD26/alpCHI substantially; that double-alpha behavior was fixed before commit.

Additional bounded MuC oracle result after the 2026-07-04 full-frame `morph-progress` recalibration of transitions `1->2` and `3->4`:

- `MuC` slides `1-3`: status `failed`, 24 comparisons, minimum SSIM `0.772084` (previous refresh floor was `0.768571`). This is the current bounded public-oracle floor for MuC after the regenerated progress maps. Strict slide/transition thresholds are still not met; the remaining gap is full-frame composition, text/raster antialiasing, effects, and media clock parity rather than gross Morph timing.

Interpretation:

- The player is coherent and assets load, but strict PowerPoint visual parity is not achieved yet.
- The transition 1->2 Morph progress calibrations, HDPhoto brightness effects, conservative glow effects with calibrated family scaling, SVG fallback consistency, deterministic visible-video QA capture, stable transition-parenting rule, and MuC text-scale calibration improved specific samples, but remaining differences include text/raster antialiasing, remaining PowerPoint effects, background/video brightness/phase, early transition timing, and full-frame composition differences.
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
6. If adding pixel-count-based static-image optimization, make it quality-gated and deterministic: generate candidate derivatives, compare settled/transition samples against PowerPoint oracle frames, and accept only when visual parity is neutral or improved.
7. Commit and push only intended files; do not stage unrelated root `index.html` changes.

## Verification Commands

```powershell
py -3 -m unittest tools.pptx-html-presenter.tests.test_presenter
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family inspect presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family build presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family asset-check presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py family visual-audit presentations\viscereality-family.config.json
py -3 tools\pptx-html-presenter\pptx-html-presenter.py candidate-sweep presentations\MuC --sample trans-001-002-025 --vary exit-fade-end --values 0.05:1:0.05 --reference-frame presentations\MuC\qa\reference\trans-001-002-025.png
py -3 tools\pptx-html-presenter\pptx-html-presenter.py candidate-sweep presentations\MuC --sample trans-001-002-025 --vary enter-fade-end --track-id track-0032,track-0033,track-0034,track-0035,track-0036 --values 0.05:1:0.05 --reference-frame presentations\MuC\qa\reference\trans-001-002-025.png
py -3 tools\pptx-html-presenter\pptx-html-presenter.py candidate-sweep presentations\alpCHI --sample trans-001-002-025 --vary track-progress --track-id track-0011,track-0012 --values 0:1:0.05 --reference-frame presentations\alpCHI\qa\reference\trans-001-002-025.png
```

PowerPoint oracle QA still requires enough free disk for reference MP4 export and frame extraction.

