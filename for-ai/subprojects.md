# Subprojects

Last updated: 2026-11-12

This is the working map of website-adjacent subprojects. Update it when a subproject is added, retired, or gains durable behavioral constraints.

## Root Website

- Path: `index.html`, `assets/css/site.css`, `assets/js/site.js`
- Purpose: public project page for Viscereality, including overview, explainers, research ingredients, team, publications, collaborations, advisory, affiliations, hire-us, presentations, and contact.
- Constraints: keep content readable without heavy media; preserve desktop and mobile navigation; keep research claims citation-backed or carefully framed.

## Atmungsraum

- Primary delivery target (2026-11-06): a progressive web app for smartphones.
  Users install it from the browser to the home screen; the existing Viscereality
  logo icon launches the live `/atmungsraum/` app in standalone mode. Prioritize
  touch, portrait/landscape, safe areas and installed-app lifecycle while keeping
  desktop access and the intended VR-headset control purpose. This install
  increment remains online and does not add offline session caching. See the
  subproject PROJECT.md for exact verification and physical-phone limits.
  Desktop Brave uses the canonical manifest identity to detect an existing
  install. The install help names `brave://apps` for reopening that app.

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

- Path: `atmungsraum/index.html`
- Public URL: <https://viscereality.org/atmungsraum/> (the directory index also serves `/atmungsraum`).
- Voice samples: <https://viscereality.org/atmungsraum/samples/>. The separately
  requested 2026-10-28 listening page publishes one exact-phrase MP3 for each
  of twenty source-prefixed LANG/EmoDB profiles; it preserves the selected
  listening assets. Original references and full receipts
  remain private. See the Atmungsraum workflow for sample verification.
- Purpose: the PC or smartphone browser acts as the external Viscereality operator controller. Quest owns the scenario, clock, Polar H10 connection and recording. The isolated rehearsal defaults to Remote-only. VR-Probe finds the Quest through Bluetooth, connects over Wi-Fi, selects a packaged rehearsal condition, and uses an enrolled connection to display live ECG and retrieve the verified original recording.
- Physical acceptance (2026-11-12): use the installed Android PWA and a Meta Quest 2. The headset owns media and timing. Verify the full control path, the packaged scenario JSON and replaceable cue library, and headset video and audio with paired device evidence. See the [Atmungsraum project checklist](../atmungsraum/for-ai/PROJECT.md#active-milestone-smartphone-pwa-and-quest-2-rehearsal).
- The phone browser remembers one verified Quest pairing and automatically tries to reconnect when opened. Bluetooth permissions and trusted controller enrollment are granted during first setup.
- Participant setup collects the first three surname letters, age in whole years,
  and an explicit gender code. The controller checks live ECG and accelerometer
  streams before starting a participant run on Quest.
- Quest records raw Polar H10 ECG and three-axis acceleration, renderer radius,
  classified breathing phase, and session timing in one dated CSV per run. The
  installed Pixel PWA verifies the file and delivers those same bytes to the
  private research Drive. Earlier browser CSVs remain downloadable at `#daten`.
  See the [collection procedure](../atmungsraum/for-ai/QUEST-PHONE-SYNC.md) and
  [CSV schema](../atmungsraum/for-ai/DATA-LOGGING.md#participant-csv-v2).
- Keep the existing Pretext measurement, keyboard flow and readable reflow for
  the controller, participant entry and recovery screens.
- AI control plane: [atmungsraum/for-ai/README.md](../atmungsraum/for-ai/README.md), discovered through `atmungsraum/AGENTS.md`.
- Audio component: [authored scenario timelines and audio verification](../atmungsraum/audio/README.md).
  Quest owns playback. The main scenario lasts 15 minutes and includes the
  26–41 second preparation window.
- Listening assets: the user selected the website's ElevenLabs British Breathing Instructor and rejected OmniVoice. Local/ignored `atmungsraum/audio-assets/ready-to-listen/elevenlabs/` keeps each WAV beside its timing/transcript JSON, with separate voice/breath/mix components, four-second phases and full 16-second cycles. Rejected sets are archived locally. See [folder guide](../atmungsraum/audio-assets/README.md). Runtime assets remain separately selected.
- Workflow: share the existing GitHub Pages deployment from `main`, root `/`; keep the local checkout sparse and avoid downloading unrelated large assets.
- UI alignment (2026-11-05): keep main controls symmetric about the dynamic viewport midpoint, headings and single footer actions horizontally centered, and playback centered as its timer/status changes. Preserve enlarged-text document reflow; see the Atmungsraum verification notes for the rendered matrix.
- The top-left `VR zentrieren` control sends the enrolled Quest recenter request
  when the headset reports that the action is available.

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
