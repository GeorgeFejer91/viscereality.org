# PPTX HTML Presenter

Reusable compiler for turning a PowerPoint deck into a static HTML presentation.

The goal is a scene-rendered player, not a video-chunk player: PPTX objects become
HTML/SVG/media layers, assets are extracted once with content hashes, and Morph is
approximated by interpolating persistent object tracks between slide states.

## Commands

```powershell
py -3 tools\pptx-html-presenter\pptx-html-presenter.py inspect deck.pptx -o out\inspect

py -3 tools\pptx-html-presenter\pptx-html-presenter.py build deck.pptx `
  --out presentations\MyDeck `
  --profile github-pages-1080 `
  --qa

py -3 tools\pptx-html-presenter\pptx-html-presenter.py build deck.pptx `
  --out out\scene-dry-run `
  --asset-mode manifest-only

py -3 tools\pptx-html-presenter\pptx-html-presenter.py qa presentations\MyDeck `
  --reference deck_reference.mp4

py -3 tools\pptx-html-presenter\pptx-html-presenter.py qa presentations\MyDeck `
  --reference deck_reference.mp4 `
  --slides 20 `
  --reuse-html

py -3 tools\pptx-html-presenter\pptx-html-presenter.py media-phase presentations\MyDeck `
  --reference deck_reference.mp4 `
  --slides 17-21 `
  --step-sec 0.5 `
  --include-transitions `
  --overrides-out out\media-phase-overrides.json

py -3 tools\pptx-html-presenter\pptx-html-presenter.py transition-time presentations\MyDeck `
  --reference deck_reference.mp4 `
  --reuse-html `
  --overrides-out out\transition-time-overrides.json

py -3 tools\pptx-html-presenter\pptx-html-presenter.py morph-progress presentations\MyDeck `
  --reference deck_reference.mp4 `
  --slides 18 `
  --compare-mode auto `
  --overrides-out out\morph-progress-overrides.json

py -3 tools\pptx-html-presenter\pptx-html-presenter.py static-fallback presentations\MyDeck `
  --reference deck_reference.mp4 `
  --settled-only `
  --overrides-out out\static-fallback-overrides.json

py -3 tools\pptx-html-presenter\pptx-html-presenter.py reference deck.pptx `
  --scene presentations\MyDeck `
  --out presentations\MyDeck\qa\powerpoint-reference.mp4

py -3 tools\pptx-html-presenter\pptx-html-presenter.py publish presentations\MyDeck `
  --deck MyDeck

py -3 tools\pptx-html-presenter\pptx-html-presenter.py family inspect `
  presentations\viscereality-family.config.json

py -3 tools\pptx-html-presenter\pptx-html-presenter.py family build `
  presentations\viscereality-family.config.json

py -3 tools\pptx-html-presenter\pptx-html-presenter.py family visual-audit `
  presentations\viscereality-family.config.json

py -3 tools\pptx-html-presenter\pptx-html-presenter.py family publish `
  presentations\viscereality-family.config.json
```

## Family Mode

Family mode compiles several related PPTX decks as separate playable
presentations while sharing one content-hashed asset library underneath the UI.
For Viscereality, the family config is:

```text
presentations/viscereality-family.config.json
```

It maps:

- `presentations/Viscereality_MuC.pptx` to `presentations/MuC-scene`, then public `presentations/MuC`.
- `presentations/Viscereality_alpCHI_v2.pptx` to `presentations/alpCHI-scene`, then public `presentations/alpCHI`.
- `presentations/20260512_BreathworkDays_Berlin_new.pptx` to `presentations/BBD26-scene-new`, then public `presentations/BBD26`.

Shared media is written once under:

```text
presentations/shared-assets/viscereality/
```

Each generated `deck.scene.json` keeps normal browser object layers but rewrites
asset URLs to `../shared-assets/viscereality/...`, so `/presentations/MuC/`,
`/presentations/alpCHI/`, and `/presentations/BBD26/` stay independent public
players while duplicate images, GIFs, videos, and SVGs live only once in Git.

`family publish` archives current chunked public folders as `*-chunked` before
copying validated scene builds into the canonical deck URLs, then rewrites
`presentations/shared/decks.js` and `presentations/index.html` for the three
public deck IDs.

## Output Contract

- `index.html`: static browser player.
- `deck.scene.json`: deterministic scene graph.
- `assets/source/*`: original extracted media, content-hashed.
- `assets/optimized/*`: optional optimized publish media.
- `assets/fallback/*`: optional content-hashed PowerPoint-rendered static
  fallback overlays.
- `inspect-report.json`: slide, media, transition, reuse, and unsupported-feature report.
- `asset-report.json`: asset provenance, sizes, GitHub Pages warnings.
- `provenance.json`: object-to-PPTX source mapping.
- `build-report.json`: build status.
- `qa/report.json`: frame sample plan and QA status.
- `qa/media-phase-report.json`: optional per-object video/GIF phase calibration
  against the PowerPoint reference.
- `qa/reference/*.png`, `qa/html/*.png`, `qa/diff/*.png`: emitted when
  `ffmpeg`, Node, Playwright, and a reference MP4 are available.

## Config

`pptx-html-presenter.config.json` can define:

```json
{
  "title": "Deck title",
  "slug": "deck-slug",
  "output_path": "presentations/DeckSlug",
  "profile": "github-pages-1080",
  "asset_policy": {
    "mode": "copy",
    "soft_max_mb": 50,
    "hard_max_mb": 100,
    "transcode_gif": true,
    "transcode_video": true,
    "webp_quality": 88,
    "video_crf": 18,
    "allow_oversize_assets": true,
    "prune_unreferenced_source_assets": true
  },
  "morph_policy": {
    "match_threshold": 55,
    "duration_default_sec": 2,
    "easing": "easeInOutQuad",
    "fade_unmatched": true,
    "unmatched_fade_start": 0,
    "unmatched_fade_end": 1,
    "transition_unmatched_fade_overrides": [
      {
        "from": 2,
        "to": 3,
        "enter_start": 0.95,
        "enter_end": 1.0
      }
    ],
    "transition_easing_overrides": [
      {
        "from": 17,
        "to": 18,
        "easing": "cubic-bezier(.2,0,.2,1)"
      }
    ],
    "transition_progress_overrides": [
      {
        "from": 17,
        "to": 18,
        "points": [
          {"progress": 0.0, "value": 0.0},
          {"progress": 0.1, "value": 0.0},
          {"progress": 0.5, "value": 0.467},
          {"progress": 1.0, "value": 1.0}
        ]
      }
    ],
    "transition_progress_overrides_file": "DeckSlug/qa/morph-progress-overrides.json"
  },
  "qa_policy": {
    "slide_ssim": 0.985,
    "morph_ssim": 0.965,
    "transition_samples": [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
    "slide_hold_sec": 3,
    "settled_offset_sec": 0.5,
    "transition_reference_lead_fraction": 0.5
  },
  "media_phase_overrides": [
    {
      "slide": 17,
      "track_id": "track-0078",
      "asset_id": "asset-c1478e568641a55a",
      "name": "Combined01_spedup_30s_hq_playback",
      "phase_sec": 7.5,
      "source": "media-phase"
    }
  ],
  "transition_time_overrides_file": "DeckSlug/qa/transition-time-overrides.json",
  "transition_media_phase_overrides_file": "DeckSlug/qa/transition-media-phase-overrides.json",
  "raster_fallback_overrides_file": "DeckSlug-fallbacks/static-fallback-overrides.json",
  "transition_time_overrides": [
    {
      "from": 6,
      "to": 7,
      "reference_offset_sec": -0.25,
      "progress_offsets": [
        {"progress": 0.5, "reference_offset_sec": -0.1}
      ],
      "sample_count": 5,
      "source": "transition-time"
    }
  ],
  "transition_media_phase_overrides": [
    {
      "from": 18,
      "to": 19,
      "track_id": "track-0086",
      "phase_sec": -0.115,
      "source": "media-phase-transition"
    }
  ],
  "raster_fallback_overrides": [
    {
      "slide": 17,
      "file": "../DeckSlug-fallbacks/static-slide-017.png",
      "object_id": "s17-static-fallback",
      "track_id": "track-static-fallback-017",
      "settled_only": true,
      "z": 1000,
      "source": "static-fallback"
    }
  ]
}
```

## Current Coverage

Supported as scene objects:

- pictures and GIFs as image/media layers.
- videos as persistent looped video layers.
- PowerPoint video objects that contain both a poster image relationship and a
  video relationship; the scene compiler prefers the real video asset and keeps
  the poster relation as `posterAssetId` for paused states.
- text boxes as HTML text layers.
- basic autoshapes as HTML shape layers.
- groups with coordinate remapping.
- crop, z-order, opacity, rotation, and flips.
- Morph-like transitions through stable `trackId` matching.
- optional PowerPoint-rendered raster fallback image layers for static
  material that the browser cannot reproduce faithfully enough.

Unsupported or partial features are listed per object in `provenance.json` and
`inspect-report.json`. The intended fallback is the smallest unsupported layer,
not a full-slide video chunk.

For Morph transitions, matched objects interpolate geometry, crop, rotation, and
opacity. Unmatched objects fade according to `morph_policy.unmatched_fade_start`
and `morph_policy.unmatched_fade_end`, both normalized from `0` to `1` across
the eased transition. This makes decks with late PowerPoint crossfades tunable
without turning the transition into rendered frame slices.
Use `morph_policy.transition_unmatched_fade_overrides` for slide pairs where
PowerPoint delays unmatched entering or exiting objects differently from the
deck-wide fade window.
Use `morph_policy.transition_easing_overrides` when a specific Morph slide pair
needs a different progress curve. The player accepts `easeInOutQuad`, `linear`,
`power:0.85`, and CSS-style `cubic-bezier(x1,y1,x2,y2)` strings.
Use `morph_policy.transition_progress_overrides` when PowerPoint's Morph
progress itself needs local calibration. Its `points` map raw transition
progress to the final interpolation progress used by the player.

When a matched transition spans an image/video pair, the runtime prefers a video
element during the transition if one side is video. Non-slide-timed loop media
keeps an independent clock even when staged offscreen, so GIF-derived or
otherwise animated-loop assets can continue animating while they move into view.
Per-object `mediaTiming.phaseSec` is applied once when a loop media element is
introduced, then the same browser media element keeps running through Morph
instead of restarting on each slide.

PowerPoint media timing commands are preserved where possible. `playFrom(...)`
sets the start frame for slide-timed media, and immediate `togglePause` calls
freeze that media at the requested frame instead of letting the browser loop it.
If PowerPoint provided a poster image for a paused video object, the player
renders that poster as a still image rather than assuming video frame zero is
the correct placeholder.

Raster fallbacks are deliberately not video chunks. The `static-fallback`
command extracts settled PowerPoint reference frames, punches transparent holes
over live video/GIF-equivalent media objects, and emits config-ready
`raster_fallback_overrides`. With `settled_only: true`, these overlays are shown
only on settled slides and hidden during Morph playback/capture, so animated
media keeps running as independent browser media. Panel-level fallbacks can use
explicit geometry, `replace_track_ids`, and `settled_only: false` when a small
PowerPoint-rendered static layer, such as a rounded panel border skin, should
Morph on the original matched panel track. Full-slide fallbacks should remain
settled-only.

## GitHub Pages Policy

The compiler flags files above 50 MiB and marks builds blocked when output files
exceed 100 MiB. It can still build local staging output, but `publish` refuses
blocked builds unless `--force` is used after manual review.

Set `asset_policy.prune_unreferenced_source_assets` for public GitHub Pages
builds. The compiler still extracts originals first so transcoding and reports
are based on source bytes, but it removes source copies that are not the actual
render asset after optimized publish copies have been created.
Use `asset_policy.video_crf` to tune MP4 publish quality; lower values produce
larger, more faithful video layers, while higher values favor smaller output.

Animated media is not flattened into slide frames. Original GIFs are always
preserved in `assets/source`, but animated GIF publish copies are made
clock-controllable when `transcode_gif` is enabled. Opaque GIF loops may become
MP4 when that is the smaller faithful representation of a video-like GIF.
Transparent GIF loops are never flattened to opaque MP4; they use
alpha-preserving output first: VP9 WebM with alpha, then animated WebP, then
the original GIF if no better alpha-safe copy can be made.

## Breathwork Days First Build

Recommended staging command:

```powershell
py -3 tools\pptx-html-presenter\pptx-html-presenter.py build `
  "C:\Users\gfeje\Downloads\20260512_BreathworkDays_Berlin.pptx" `
  --out presentations\BBD26-scene `
  --title "Berlin Breathwork Days 2026" `
  --slug BBD26-scene `
  --profile github-pages-1080 `
  --qa
```

If large GIFs remain above GitHub limits, install `ffmpeg` and rebuild so the
optimizer can choose MP4 for opaque loops and alpha-safe WebM/WebP for transparent
loops.

For very large decks, use `--asset-mode manifest-only` first. It emits the scene,
player, reports, and Morph tracks without copying media into the output directory.

## QA Dependencies

Complete visual QA needs:

- PowerPoint-exported high-quality MP4 reference for the source deck.
- `ffmpeg`/`ffprobe` for reference frame extraction.
- Node with Playwright installed for HTML player capture.
- Pillow and NumPy for comparison metrics and diff heatmaps.

The player exposes `window.PptxHtmlPresenter.captureAt(slide, progress)`, which
the Playwright runner uses to capture settled slides and transition progress
frames one-to-one against the PowerPoint reference.

QA uses local-window luminance SSIM (`local-uniform-11-luma`) rather than a
single global covariance score. This keeps visually close but high-frequency
slides from being over-penalized while still flagging large phase, timing, and
layout mismatches.

Each QA comparison writes a PowerPoint reference frame, an HTML-rendered frame,
a diff heatmap, and a side-by-side triptych at
`qa/side-by-side/<sample>.png` containing reference, HTML, and diff panels.

`media-phase` is a targeted calibration pass for animated media. It crops the
PowerPoint reference frame to each visible video object's slide bounds, searches
candidate frames from the extracted media asset, and reports the best matching
media timestamp. Add `--include-transitions` to also inspect Morph samples; the
calibrator uses the same effective transition object and visible-slice crop as
the browser player, so a partially offscreen media object is compared against the
same partial source-frame region instead of a squeezed full frame. With
`--apply`, strong matches are written back as per-object `mediaTiming.phaseSec`
values in `deck.scene.json`; the asset remains a normal browser media element.

For repeatable builds, use `--overrides-out` and copy the emitted
`media_phase_overrides` entries into the deck config. The selectors are slide
number plus stable PPTX object identity (`object_id`, `track_id`, `asset_id`, or
`name`), so future builds can restore the calibrated loop phase without mutating
the source PPTX or relying on rendered frame chunks.
Use `transition_media_phase_overrides` when the same media object needs one
clock phase during a Morph and a different phase after the destination slide has
settled. This keeps live incoming GIF/video panels moving during the transition
without damaging the settled slide's media timing. When transition samples for
one object disagree by a wide margin, the calibrator leaves out the transition
override instead of emitting a misleading median phase.

`transition-time` is the matching calibration pass for the PowerPoint MP4 oracle
timeline. It searches a small window around each transition sample, groups the
best reference-frame deltas by transition pair, and emits
`transition_time_overrides`. These offsets make later QA compare HTML Morph
states against the PowerPoint frame that represents the same visual moment,
instead of relying on one global export-timing assumption for every transition.
When PowerPoint's Morph progress is non-linear inside the exported MP4, the
generated `progress_offsets` refine the oracle timestamp at individual Morph
sample points without changing the HTML player's own animation.

Some PowerPoint MP4 exports center transition frames around the automatic advance
boundary instead of starting the transition exactly at that boundary. Use
`qa_policy.transition_reference_lead_fraction` to shift reference transition
sampling earlier by a fraction of the transition duration while keeping HTML
capture at the requested Morph progress. The Breathwork Days validation deck uses
`1.0` based on boundary-frame inspection, with transition-specific offsets
available through `transition_time_overrides`.

`morph-progress` calibrates the player's object interpolation curve. It captures
many candidate HTML Morph states with an identity progress map, compares them to
the PowerPoint reference frames, and emits `transition_progress_overrides`. In
`auto` mode it crops comparisons around inferred large moving anchors, especially
rounded panel containers and their panel-border fallback tracks, so grouped
slide-within-slide panels drive the curve instead of unrelated animated media.
The final player still renders live HTML/SVG/media layers; the calibration only
changes how raw transition progress maps to object positions.

`static-fallback` is a targeted fidelity pass for non-media rendering gaps. It
creates PNG overlays from the PowerPoint MP4 reference and masks out live media
bounding boxes so GIFs and videos remain live assets. For repeatable builds,
store the emitted PNGs outside the generated build directory and point
`raster_fallback_overrides_file` at a sidecar JSON. The build copies those PNGs
back into `assets/fallback` with content-hashed filenames and prunes stale
fallback files.

On Windows, the `reference` command creates the MP4 oracle through PowerPoint
COM without mutating the source PPTX. Pass `--scene` to normalize slide advance
and transition durations to the generated scene timeline before export. When the
scene contains `qa.slideHoldSec`, that value is used as the PowerPoint reference
hold duration unless `--default-slide-sec` overrides it.
