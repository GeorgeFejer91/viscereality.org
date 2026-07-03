# Subprojects

Last updated: 2026-07-02

This is the working map of website-adjacent subprojects. Update it when a subproject is added, retired, or gains durable behavioral constraints.

## Root Website

- Path: `index.html`, `assets/css/site.css`, `assets/js/site.js`
- Purpose: public project page for Viscereality, including overview, explainers, research ingredients, team, publications, collaborations, advisory, affiliations, hire-us, presentations, and contact.
- Constraints: keep content readable without heavy media; preserve desktop and mobile navigation; keep research claims citation-backed or carefully framed.

## Media Assets

- Paths: `assets/media/`, `assets/ascii-media/`, `Images/`, `Scientific_Ingredients/`, `Video/`, `AnimatedSVG/`
- Purpose: videos, posters, GIFs, ASCII-frame media, logos, and research visuals used by the public site.
- Constraints: preserve referenced filenames; optimize first-viewport media; keep generated media manifests in sync with rendered assets.

## Presentations Hub

- Path: `presentations/index.html`
- Purpose: public index of presentation players.
- Constraints: cards should link to playable subpages and use stable preview images, clear titles, and conference/context links when available.

## Presentation Players

- Paths: `presentations/MuC/`, `presentations/alpCHI/`, `presentations/BBD26/`, `presentations/shared/`
- Purpose: browser-based presentation playback with slide chunks, transitions, and manifests.
- Constraints: preserve authored chunk order and timing; update `manifest.json` and `presentations/shared/decks.js` together when changing a deck; keep controls usable on touch and keyboard.

## Sync Relay

- Path: `presentations/sync-relay/`
- Purpose: synchronization support for presentation playback.
- Constraints: keep deployment/configuration details documented locally; avoid committing credentials or environment-specific secrets.

## PPT Chunker

- Path: `ppt-chunker/ppt-chunker/`
- Purpose: Python tooling for exporting or chunking PowerPoint presentations into web-playable media and manifests.
- Constraints: keep tests focused on manifest, timing, and pipeline behavior; generated chunks should be reproducible from documented inputs.

## PPTX HTML Presenter

- Path: `tools/pptx-html-presenter/`, generated decks under `presentations/*-scene/`
- Purpose: reusable static-site compiler for PowerPoint decks into object-based HTML presentation players.
- Constraints: target a one-to-one PPTX object mapper, not full-slide video chunks or screenshots. Preserve PowerPoint layering, object identities, groups, media clocks, text, shapes, crops, and Morph transitions. Inferred panel/carousel clusters, such as rounded white frames with contents inside, must become HTML groups with children locked to the frame and the frame/border rendered on top. Reverse navigation should play the same Morph animation backward.
- Workflow: follow [pptx-html-one-to-one-mapper.md](pptx-html-one-to-one-mapper.md) before changing compiler/runtime behavior.

## Generated Documents

- Path: `generated-documents/`
- Purpose: generated Markdown/LaTeX/PDF-style document assets.
- Constraints: keep source templates separate from outputs; document any generation command needed to rebuild deliverables.

## Control

- Path: `control/`
- Purpose: standalone control or demo page.
- Constraints: preserve as an independent web surface unless intentionally integrated into the root page.

## Writing Group

- Path: `writing-group/`
- Purpose: standalone writing-group page or microsite.
- Constraints: treat as a separate subpage with its own content expectations.

## Research Supplements

- Path: `cardiac_coherence/`
- Purpose: supplementary material and media for cardiac coherence research content.
- Constraints: preserve publication-facing filenames and avoid changing supplements without explicit user intent.
