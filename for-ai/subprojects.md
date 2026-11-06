# Subprojects

Last updated: 2026-11-06

This is the working map of website-adjacent subprojects. Update it when a subproject is added, retired, or gains durable behavioral constraints.

## Root Website

- Path: `index.html`, `assets/css/site.css`, `assets/js/site.js`
- Purpose: public project page for Viscereality, including overview, explainers, research ingredients, team, publications, collaborations, advisory, affiliations, hire-us, presentations, and contact.
- Constraints: keep content readable without heavy media; preserve desktop and mobile navigation; keep research claims citation-backed or carefully framed.

## Atmungsraum

- Polar loss (2026-11-06): more than 30 seconds of continuous missing ECG aborts
  the scenario and all scheduled audio, clears the scenario selection and
  returns to the Polar selection gate while retaining patient details. The
  deadline uses elapsed time while playing or paused and requires no prior
  reconnect retry; late recovery cannot revive the expired run. Loss through
  exactly 30 seconds retains the existing recovery behavior. See Atmungsraum
  PROJECT.md and VERIFICATION.md for browser-demo evidence and hardware limits.

- Scenario assets (2026-11-06): every scenario owns numbered unique WAV/CSV/TXT/JSON groups under `atmungsraum/assets/audio/scenarios/`, with asset/timeline CSVs, manifest and playlist. Same-basename per-cue CSVs list all chronological occurrences with exact timing and matching draft words; names encode first-use order and onset, while the full playlist/timeline records repeated cycles. All current runtime cues are silent placeholders, explicitly authorized for a working UI; TXT is intended draft wording and spoken text remains empty. Final narration still requires wording approval. Main goes directly to one exact 15-minute sequence; its breathing window remains 26–41 seconds. See the scenario README.

- Audio scripts (2026-11-06): each library WAV/MP3 requires an adjacent exact-
  wording TXT, including empty TXT for silent/breath-only assets. Prepare text
  and wait for explicit wording approval before audio generation/replacement.
  Final speech waits for approval; the later request allows silent UI placeholders with labelled draft scripts. Participant narration/UI
  uses plain wording rather than the technical German calibration term. See
  [the owning audio workflow](../atmungsraum/for-ai/WORKFLOW.md#exact-txt-scripts-and-wording-approval--2026-11-06).

- Main round (2026-11-06 latest correction): Haupt-Runde goes directly to one
  exactly 15-minute Start screen. There is no main duration menu, 10-minute main
  scenario or five-minute baseline. Shared classes are Instruction 1(26 s),
  Calibration(15 s), Instruction 2(40 s), fifty 16-second cycles(800 s), and Outro(19 s).
  Calibration is explicitly scheduled at 26–41 audio-clock seconds. JSON/CSV,
  receipts and an advance event expose its window for the future ECG-sensor
  accelerometer; measured value/unit/method remain unset, status not-integrated.
  Instruction 1 uses the Secret Tunnel existing-instructor clone. Original
  assets/practice remain unchanged. See [the transcript/protocol](../atmungsraum/audio/calibration.md).

- Responsive proportions (2026-11-06): patient entry uses a centered composition
  capped at 560px high, a 640px form, equal columns/shared gutters and bounded
  inter-group spacing. Body type ranges from 16 to 18px; enlarged/no-fit content
  uses readable reflow. All three corner surfaces scale equally from 56 to 112px
  with viewport width and height, superseding the earlier fixed 144px size. Polar embeds
  the recovered dissertation H10 belt illustration (PNG source). The belt and
  frame pulse green with ECG and red after loss, with steady reduced-motion
  glows and no separate dot. A luminance mask keeps the glow on the artwork. Reserve their
  space with portrait top/bottom clearance and a short-landscape central column;
  use readable document reflow at the revised small-screen no-fit boundary.

- Saved data (2026-11-05): a permanent bottom-right SVG folder opens a scrollable
  native dialog of this browser's saved session CSVs on every screen. As of
  2026-11-06 it is the sole visible data entry point; CSV-Dateien footer links
  are removed, while the legacy recovery URL remains supported. Individual
  downloads, an all-file ZIP and native file sharing support email/messaging apps
  where available; download/manual attachment is the fallback. Opening/dismissing
  the dialog preserves playback. This cannot enumerate the OS Downloads folder.
  The QR action (2026-11-06) guides ZIP upload on Wormhole and generates a QR
  locally from the pasted complete link/key. Upload is manual: Wormhole rejects
  site-origin CORS preflight and Pages has no backend. One-click upload remains
  pending a hosting/provider decision. Link/key never goes to a QR service.
  See Atmungsraum DATA-LOGGING.md and VERIFICATION.md for persistence and QA limits.

- Path: `atmungsraum/index.html`
- Public URL: <https://viscereality.org/atmungsraum/> (the directory index also serves `/atmungsraum`).
- Voice samples: <https://viscereality.org/atmungsraum/samples/>. The separately
  requested 2026-10-28 listening page publishes one exact-phrase MP3 for each
  of twenty source-prefixed LANG/EmoDB profiles; it preserves the main demo's
  selected session voice and timing. Original references and full receipts
  remain private. See the Atmungsraum workflow for sample verification.
- Purpose (2026-10-12): a smartphone-compatible browser interface that initiates actions on a VR headset screen through a corresponding app that mirrors the website's UI.
- Intended interaction: phone browser controls trigger actions in the headset app; keep corresponding screens, controls, and action meanings aligned across both surfaces. The headset platform, action set, and connection architecture are Undecided.
- First steps (2026-11-05): collect the first three surname letters, whole-year age and explicit gender, show the generated participant ID, then verify Polar H10 connection/ECG before the session menu. The current demo continuously replays a seeded NeuroKit2 ECG source. The intended product receives real ECG from the headset; acquisition and transport remain Undecided. A corner indicator breathes green, flashes red after loss (steady with reduced motion), and loss beyond 30 seconds returns to the Polar gate while retaining patient details.
- Participant data logging: every writer follows [Atmungsraum DATA-LOGGING.md](../atmungsraum/for-ai/DATA-LOGGING.md) and the shared runtime filename/identity functions. Session exports use `Fej_35_M_probe1.csv` / `Fej_35_M_main1.csv`, explicit scenario/timing/source metadata, and per-participant/condition run indices. CSV downloads and IndexedDB recovery are browser-local; no real H10 acquisition or central study store is established.
- Menu layout: use `uncodixfy-pretext` with outside-in stretch-accordion geometry for all UI functions, including patient entry and CSV recovery. Preserve equal anchors and ordered label/control groups; stretch form/list group spacing and matrix columns with both dimensions. Short landscape menus use four practice columns and wider round/main cards. Keep actual bounded-text and painted-DOM checks, readable no-fit/recovery, and no internal panel scrollbar. [Concrete settings and supported-size exceptions](../atmungsraum/for-ai/SKILLS.md#main-ui-stretch-settings) own the layout contract.
- Current implementation (2026-10-12): a browser-local UI demo emulates headset connection and ECG reception, enabling Weiter only with synthetic ECG. The minimal gate shows Polar search/selection/connection status, ECG only during recording, and Refresh/Continue; Continue turns red to green. Round choices with generated SVG infographics above their HTML labels (2026-11-05) lead to four practice boxes (Anleitung unlimited guided exploration/5/10/15 minutes) and two main boxes (10/15 minutes total). Main sessions begin with five silent minutes of normal breathing, followed by the existing five- or ten-minute guided box-breathing sequence; the choice cards show the breakdown. Every box goes directly to a final play button. Browser-local audio uses one Starten/Pause/Fortsetzen button. Pause freezes audio and the timer; Fortsetzen resumes the same session. The back action reads Beenden during playback and aborts before returning to the choice menu. UI elements and copy follow ponytail: keep only necessary choices, state, and recovery feedback; the search retry appears only where it aids recovery. There is no ECG return button after Continue. Signal loss requests reconnect every five seconds; loss beyond 30 seconds aborts the scenario and returns to Polar selection, independent of retry count or playback pause. Recovery before expiry cancels the deadline; late recovery cannot revive the old run. Local Pretext 0.0.9 measures bounded text; ordinary phone/desktop panels fit without scrolling, enlarged text uses readable page reflow. The user approved this prototype for publication as the main VR-control navigation. Real H10 acquisition and headset transport/execution remain pending; the bottom-level Start action is intended to combine audio with a VR event sequence whose timing and integration are still undecided.
- AI control plane: [atmungsraum/for-ai/README.md](../atmungsraum/for-ai/README.md), discovered through `atmungsraum/AGENTS.md`.
- Audio component: [reusable German box-breathing engine](../atmungsraum/audio/README.md), wired to the final browser Play button, with four repeating four-second phase samples, a direct du welcome/ending, 20-second rests after complete 288-second breathing blocks, third-cycle breath overlays, and exact 5/10/15/20-minute standalone totals. Hauptrunde adds a five-minute silent normal-breathing baseline to the five- or ten-minute guided sequence, giving 10/15-minute totals. Native offline renders passed; physical output timing and H10/headset synchronization remain pending. Alternative calm German voice auditions are local; OmniVoice model licensing and speaker permission do not establish unrestricted clinical use. Original references, audition profiles, and full-session verification exports stay local and ignored.
- Scenario protocol (2026-11-05): every final sequence reads an authored JSON timeline and matching CSV under `atmungsraum/audio/scenarios/`, referencing the single shared `atmungsraum/assets/audio/` library. JSON owns clip order and frame timing; CSV is generated/checked from it. Playback receipts bind scenario and library hashes. All six menu routes and the existing 20-minute diagnostic have pairs; existing sample paths, voice and timing are preserved. See the scenario README and Atmungsraum verification notes.
- Listening assets: the user selected the website's ElevenLabs British Breathing Instructor and rejected OmniVoice. Local/ignored `atmungsraum/audio-assets/ready-to-listen/elevenlabs/` keeps each WAV beside its timing/transcript JSON, with separate voice/breath/mix components, four-second phases and full 16-second cycles. Rejected sets are archived locally. See [folder guide](../atmungsraum/audio-assets/README.md). Runtime assets remain separately selected.
- Workflow: share the existing GitHub Pages deployment from `main`, root `/`; keep the local checkout sparse and avoid downloading unrelated large assets.
- UI alignment (2026-11-05): keep main controls symmetric about the dynamic viewport midpoint, headings and single footer actions horizontally centered, and playback centered as its timer/status changes. Preserve enlarged-text document reflow; see the Atmungsraum verification notes for the rendered matrix.
- Recenter placeholder (2026-11-05): keep the icon-only `VR zentrieren` button fixed at the top left on every screen and while scrolling. Its native SVG shows a headset with four inward arrows. It is accessibly marked unavailable and has no action; future headset integration will recenter the central field of view. See the Atmungsraum project contract.
- Playback controls (2026-11-05): compact Starten/Pause/Fortsetzen and bordered Beenden buttons accompany a native read-only progress bar and elapsed/total times in every player state. Progress follows the audio clock, freezes on pause, and resets on replay; Anleitung shows its actual 68-second example. Seeking and video media are outside this UI change.

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
