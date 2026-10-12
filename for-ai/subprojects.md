# Subprojects

Last updated: 2026-10-12

This is the working map of website-adjacent subprojects. Update it when a subproject is added, retired, or gains durable behavioral constraints.

## Root Website

- Path: `index.html`, `assets/css/site.css`, `assets/js/site.js`
- Purpose: public project page for Viscereality, including overview, explainers, research ingredients, team, publications, collaborations, advisory, affiliations, hire-us, presentations, and contact.
- Constraints: keep content readable without heavy media; preserve desktop and mobile navigation; keep research claims citation-backed or carefully framed.

## Atmungsraum

- Path: `atmungsraum/index.html`
- Public URL: <https://viscereality.org/atmungsraum/> (the directory index also serves `/atmungsraum`).
- Purpose (2026-10-12): a smartphone-compatible browser interface that initiates actions on a VR headset screen through a corresponding app that mirrors the website's UI.
- Intended interaction: phone browser controls trigger actions in the headset app; keep corresponding screens, controls, and action meanings aligned across both surfaces. The headset platform, action set, and connection architecture are Undecided.
- First step (2026-10-12): verify that a Polar H10 is connected, then show its live ECG in a centered box before the session menu. The intended browser receives ECG from the headset; acquisition and transport remain Undecided. A verified feed enables Weiter to the round menu.
- Menu layout: use `uncodixfy-pretext` with outside-in stretch-accordion geometry for all UI functions, preserving anchors and ordered groups, actual bounded-text measurement, readable reflow, and no internal panel scrollbar.
- Current implementation (2026-10-12): a browser-local UI demo emulates headset connection and ECG reception, enabling Weiter only with synthetic ECG. The minimal gate shows Polar search/selection/connection status, ECG only during recording, and Refresh/Continue; Continue turns red to green. Soft sprout/wave round choices with short descriptions lead to four boxes each: practice Anleitung (unlimited guided exploration)/5/10/15 minutes; main 5/10/15/20 minutes. Every box goes directly to a final play button. Browser-local audio is wired by a separate controller; Start/Pause appear below the main start button during playback. Pause freezes audio and the timer; Start resumes the same session. There is no ECG return button after Continue. Signal loss requests reconnect every five seconds; only loss beyond 30 seconds after retries returns to ECG and locks Continue. Recovery cancels the deadline. Local Pretext 0.0.9 measures bounded text; ordinary phone/desktop panels fit without scrolling, enlarged text uses readable page reflow. The user approved this prototype for publication as the main VR-control navigation. Real H10 acquisition and headset transport/execution remain pending; the bottom-level Start action is intended to combine audio with a VR event sequence whose timing and integration are still undecided.
- AI control plane: [atmungsraum/for-ai/README.md](../atmungsraum/for-ai/README.md), discovered through `atmungsraum/AGENTS.md`.
- Audio component: [reusable German box-breathing engine](../atmungsraum/audio/README.md), wired to the final browser Play button, with four repeating four-second phase samples, a direct du welcome/ending, 20-second rests after complete 288-second breathing blocks, third-cycle breath overlays, and exact 5/10/15/20-minute totals. Native offline renders passed; physical output timing and H10/headset synchronization remain pending. Alternative calm German voice auditions are local; OmniVoice model licensing and speaker permission do not establish unrestricted clinical use. Original references, audition profiles, and full-session verification exports stay local and ignored.
- Listening assets: local/ignored `atmungsraum/audio-assets/ready-to-listen/` keeps each audition WAV beside its timing/transcript JSON, with separate voice/breath/mix components, four-second phases and full 16-second cycles. See [folder guide](../atmungsraum/audio-assets/README.md). Runtime assets remain separately selected.
- Workflow: share the existing GitHub Pages deployment from `main`, root `/`; keep the local checkout sparse and avoid downloading unrelated large assets.

## Media Assets

- Paths: `assets/media/`, `assets/ascii-media/`, `Images/`, `Scientific_Ingredients/`, `Video/`, `AnimatedSVG/`
- Purpose: videos, posters, GIFs, ASCII-frame media, logos, and research visuals used by the public site.
- Constraints: preserve referenced filenames; optimize first-viewport media; keep generated media manifests in sync with rendered assets.

## Presentations Hub

- Path: `presentations/index.html`
- Purpose: public index of presentation players.
- Constraints: cards should link to playable subpages and use stable preview images, clear titles, and conference/context links when available.

## Presentation Players

- Paths: `presentations/MuC/`, `presentations/alpCHI/`, `presentations/BBD26/`, `presentations/shared/`, `presentations/shared-assets/viscereality/`
- Purpose: browser-based presentation playback. The target public players are scene-rendered HTML recreations with shared content-hashed media assets; chunked players remain fallback/reference artifacts only.
- Constraints: preserve authored timing and navigation; update `deck.scene.json`, shared asset references, and `presentations/shared/decks.js` together when changing a scene deck; keep controls usable on touch and keyboard. If chunked fallbacks are archived as `*-chunked`, keep their links secondary on the presentation hub.

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
- Family mode: `presentations/viscereality-family.config.json` maps `MuC`, `alpCHI`, and `BBD26` source PPTX files to staging scene builds and a shared asset library under `presentations/shared-assets/viscereality/`.
- Workflow: follow [pptx-html-one-to-one-mapper.md](pptx-html-one-to-one-mapper.md) and [pptx-html-family-status.md](pptx-html-family-status.md) before changing compiler/runtime behavior.

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
