# PPTX-To-HTML One-To-One Mapper Plan

Last updated: 2026-07-03

This is the target architecture for `tools/pptx-html-presenter/`: a PowerPoint-to-HTML compiler that recreates the PPTX object model as browser objects. The goal is not to export full-slide screenshots or video chunks. The goal is a static website presentation where every PowerPoint element that can remain an object does remain an object, and Morph transitions move those objects or object groups directly.

## Core Principle

The compiler must preserve PowerPoint layering and identity:

1. Backgrounds become background image/SVG/shape layers.
2. Images, videos, GIFs, SVGs, shapes, text boxes, and groups become individual HTML/SVG/media objects.
3. PowerPoint groups become HTML groups with a local coordinate system.
4. Visual clusters inferred from layout, such as rounded white carousel frames with contents inside, become explicit HTML panel groups when PowerPoint does not expose useful group metadata.
5. The top outline of a panel remains a top layer, and everything visually inside its boundary moves in lockstep with that panel.
6. Raster fallback is allowed only for isolated unsupported objects or effects, not as the normal representation of a whole slide.

## Scene Graph

Emit `deck.scene.json` as a real presentation object graph:

- `slides[]`: source slide states in PPT order.
- `objects[]`: leaf objects with PowerPoint ids, names, z order, geometry, crop, opacity, fills, strokes, text runs, media timing, asset references, and provenance.
- `groups[]`: explicit or inferred parent objects with local coordinate systems.
- `relationships[]`: parent-child edges, including PowerPoint groups, panel containment, masks, and overlay relationships.
- `transitions[]`: Morph mappings between adjacent slide states.
- `assets[]`: deduplicated source assets and optimized publish assets.

Every emitted object should keep provenance back to the source PPTX XML path, relationship id, media target, and shape id or creation id when available.

## Object Mapping

Map PowerPoint elements to browser-native equivalents first:

- Pictures: `<img>` with exact crop, opacity, rotation, scale, z order, and clipping.
- Videos: `<video>` with poster, loop/autoplay timing, independent clock, crop, opacity, and transforms.
- GIFs: keep as GIF when alpha/loop behavior matters; transcode to video only when visually equivalent or requested.
- SVG/EMF/WMF: preserve as vector where possible; otherwise isolated raster fallback.
- Text boxes: positioned HTML text with per-run styling, font fallback logs, alignment, margins, and autofit strategy.
- Shapes: SVG or CSS/SVG shape objects, including fills, strokes, rounded corners, opacity, and transforms.
- Groups: HTML wrapper with relative child coordinates and inherited transform.
- Masks/clips: CSS/SVG clips when possible.

Do not flatten a slide just because the slide is complex. Flatten only the smallest unsupported visual unit.

## Panel And Carousel Cluster Rule

For decks like BBD26, rounded white frames are carousel-slide objects. The compiler must detect them as panels and build them as groups:

- A panel frame is a large rounded rectangle or equivalent raster/vector panel border.
- Children are objects whose visible bounds or center fall inside the panel bounds and whose dimensions plausibly fit within it.
- The panel has one local coordinate system.
- Child geometry is stored relative to the panel.
- The panel's frame/border is rendered above children.
- Child media remains live media, not baked into the panel.
- If a child crosses a panel edge, preserve the PowerPoint clipping/overflow behavior.

During Morph:

- Match panels by PowerPoint group id, creation id, explicit name, child asset identity, relative child layout, and geometry.
- Once a panel is matched, all child movement is expressed as panel transform plus optional local child transform.
- If a child stays in the same relative position inside a matched panel, it should have no independent global motion.
- If a child changes inside the panel, interpolate its local transform within the panel.
- Per-panel progress/easing applies to the whole panel group, including children and top outline.

This prevents the current failure mode where a white frame and the media inside it appear to snap, drift apart, or use different timing.

## Morph Identity Model

Create a stable identity graph before rendering transitions:

1. Parse PowerPoint ids, creation ids, names, shape ids, relationship ids, media hashes, group paths, and z order.
2. Build candidate matches between adjacent slide states.
3. Match explicit PowerPoint groups first.
4. Match inferred panels/clusters second.
5. Match children within matched groups using local coordinates.
6. Match remaining ungrouped leaf objects.
7. Fade only unmatched entering/exiting objects.

Morph interpolation should operate on:

- group transform: x, y, width, height, rotation, scale, opacity.
- local child transform: relative x, y, width, height, rotation, crop, opacity.
- media state: independent playback clocks, phase offsets, paused/poster state.
- z order: preserve PowerPoint layer order unless PowerPoint implies a change.

Reverse navigation must use the same transition mapping in reverse. Pressing Back/Left should run the Morph from slide `N` to slide `N-1`, not jump to a settled previous slide.

## Runtime Player

The player should render a persistent scene graph:

- HTML elements are keyed by object or group track ids, not recreated as unrelated slide snapshots.
- Groups are wrapper elements; children are positioned inside wrappers.
- Panel groups own their child transforms during transitions.
- Top panel border/frame layers remain above children.
- Media elements persist across slide states when the asset identity persists, so GIF/video clocks do not restart unnecessarily.
- Navigation supports `next()`, `prev()`, `goTo(slide)`, click/touch/keyboard, fullscreen, direct hashes, and `captureAt(slide, progress)`.
- `prev()` must animate backward through Morph when a reverse transition exists.

## Asset Strategy

Extract original assets byte-for-byte first:

- Deduplicate by content hash.
- Preserve source assets in a provenance folder.
- Create publish-optimized copies only when useful.
- Keep alpha-capable assets alpha-capable.
- Keep GIFs or alpha animations live unless a better equivalent is proven.
- Keep video as video when source video exists.
- Keep SVG as SVG when browser-compatible.
- Enforce GitHub Pages file-size limits with reports, not silent quality loss.
- In family/shared-asset publishing, do not keep an oversized original source asset in the public shared asset tree when the runtime uses an optimized copy. Preserve the original asset hash/source path in reports and `asset-index.json`, but publish only GitHub-safe runtime assets unless the user explicitly approves large-file/LFS handling.
- Treat 50 MiB as the preferred public-asset ceiling and 100 MiB as the hard GitHub-compatible ceiling. Future family builds should be `needs-review` if any public shared asset remains above the preferred ceiling, and blocked from ordinary publish if any runtime asset remains above the hard ceiling.
- Oversized static raster images are runtime optimization candidates too, not only GIFs/videos. Prefer high-quality WebP for PNG/JPEG/TIFF/BMP sources, preserving alpha when needed; only downscale to presentation-sized WebP when the larger visually lossless output cannot satisfy the hard limit.
- Default policy should disallow oversize public assets (`allow_oversize_assets: false`) and optimize static images (`optimize_static_images: true`). Use explicit overrides only for debugging or when Git LFS/external hosting has been consciously chosen.
- Optimized media must be reproducible, not merely small. FFmpeg transcodes should strip volatile metadata and avoid non-deterministic multithreaded encoder settings where practical, because shared asset filenames are content hashes and must not churn across identical rebuilds.

## Multi-Deck Family Strategy

The current reusable target includes deck families, not only single decks:

- Three public deck URLs (`/presentations/MuC/`, `/presentations/alpCHI/`, `/presentations/BBD26/`) must remain separate and clickable.
- Shared media should be stored once under `presentations/shared-assets/viscereality/`.
- Each `deck.scene.json` should reference shared assets with relative URLs such as `../shared-assets/viscereality/optimized/<hash>.<ext>`.
- Public sharing must be invisible to users: browser history, fullscreen, navigation, and player state remain per deck.
- Duplicate assets are deduped by SHA-256 across the family; optimized outputs are deduped by output content hash.
- Existing chunked players may be archived as `MuC-chunked`, `alpCHI-chunked`, and `BBD26-chunked` after scene QA passes.

## QA Workflow

PowerPoint remains the visual oracle.

For every build:

1. Export or reuse a high-quality PowerPoint MP4/reference.
2. Render HTML frames with the browser capture API.
3. Compare settled slides and Morph progress frames.
4. Generate side-by-side images and heatmaps.
5. Report SSIM/mean delta/p95 delta per sample.
6. Flag mismatches by object/group when possible.

Required samples:

- every settled slide.
- Morph progress at `0%, 10%, 25%, 50%, 75%, 90%, 100%`.
- reverse Morph samples for previous navigation.
- animated media loop-boundary samples for GIF/video-heavy slides.

Acceptance targets:

- settled slides: SSIM `>= 0.985`.
- Morph frames: SSIM `>= 0.965`.
- lower scores need either a fix or a reviewed exception tied to a specific unsupported PowerPoint feature.

## Implementation Order

1. Stop using full-slide settled fallbacks by default. Keep them only as temporary QA overlays or explicit emergency fallback.
2. Add a canonical group/panel scene model to `deck.scene.json`.
3. Convert detected panel contents to local coordinates under panel groups.
4. Render groups in the player as wrapper elements with child objects inside them.
5. Render panel borders as top overlay layers within each panel group.
6. Make Morph interpolation group-first, then child-local.
7. Add reverse Morph playback for Back/Left navigation.
8. Recalibrate BBD26 only after the object/group model is correct.
9. Add tests for group detection, panel child locking, top-border layering, media persistence, reverse Morph, and fallback minimization.
10. Update publish gating so a deck is publishable only when object-mapped QA passes or exceptions are explicitly reviewed.

## BBD26 Specific Rule

For `20260512_BreathworkDays_Berlin.pptx`, the large rounded white frames are carousel panels. Treat each frame and its contained assets as a single semantic moving unit. The internal videos/images remain live objects, but their global movement follows the panel. The panel border must render above internal content.
