# Atmungsraum for-ai

This is the AI control plane for the `/atmungsraum` subpage. Product code lives
outside this folder, currently in `../index.html` and `../app.js`.

At entry, read [SKILLS.md](SKILLS.md) for the nested shared baseline and select
only the task-matching installed skill bodies. It includes `for-ai`, RTK, GSD
Core, Superpowers, Ponytail and the conditional UI, accessibility, performance,
state-management and native/remote routes. Resume the goal/checklist in
[PROJECT.md](PROJECT.md); before handoff, record evidence, blockers, the next
action and observed Git/publication state in that same owner.

The primary delivery goal is a progressive web app (PWA) for smartphones.
Users install Atmungsraum from their browser to the home screen; its Viscereality
logo icon opens the live `/atmungsraum/` URL in a standalone app window.
Prioritize touch, portrait/landscape, safe areas and installed-app behavior.
The PWA is the control interface for a VR headset app that mirrors its UI.
Its required first step collects patient details
and shows the generated participant ID, then verifies a Polar H10 connection
and shows ECG in a centered box. All menu
UI functions use `uncodixfy-pretext` with stretch-accordion geometry. See
[SKILLS.md](SKILLS.md#main-ui-stretch-settings) for the concrete anchors,
dimension-responsive spacing/type, minimum-size and no-fit settings, and
[PROJECT.md](PROJECT.md) for implementation status and open decisions.

The browser prototype is a local UI demo: patient details → simulated headset/ECG check →
round choice → four practice options or direct entry to the sole 15-minute main
sequence → final Starten button. Haupt-Runde has no duration-choice screen or
five-minute baseline. Instruction 1 (26 s), a separate Calibration window (15 s),
Instruction 2 (40 s), fifty complete breathing cycles (800 s), and Outro (19 s)
total exactly 900 seconds. The player labels preparation “Vorbereitung” and the measurement window “Zwei tiefe Atemzüge”. Each scenario now loads its own numbered silent placeholder WAVs with JSON/CSV and draft TXT scripts from `assets/audio/scenarios/`; final narration waits for wording approval. Calibration runs at
26–41 seconds on the audio clock; receipts and an advance event expose its
start/end for the future ECG-sensor accelerometer. Its measured value remains
unset until integration. See [the transcript/protocol](../audio/calibration.md).
The same button becomes Pause during playback and Fortsetzen while paused. The
back action reads Beenden during playback and has a bordered button surface.
A read-only progress bar with elapsed/total time stays visible before, during
and after playback, including the 68-second Anleitung example. Pause freezes audio and the timer;
Fortsetzen resumes the same session. Ponytail applies to UI elements/copy: keep
only necessary controls, choice information, state, and recovery feedback. Anleitung is
unlimited guided exploration. Real H10/headset connection and execution remain
pending. See PROJECT.md for the precise demo and product boundaries.
The UI has no global context header or ECG return button after Continue. Signal
loss uses a 30-second reconnect window. Beyond 30 seconds the scenario is
aborted, all audio is stopped, its selection is cleared, and the Polar selection
gate returns while patient details remain. The deadline uses elapsed time even
while paused, without requiring a prior retry; late ECG recovery cannot revive it.
The corner Polar belt artwork and its frame pulse together in green with ECG
and red after loss; reduced motion keeps both glows steady. There is no separate dot.
The base corner surfaces scale from 56 to 112 CSS px with viewport dimensions.
The 2026-11-07 smartphone correction doubles the two top controls and their
artwork (normally 112px), while keeping the folder at its existing size. An
`App installieren` action sits beside the folder, or as a named icon above it
in short landscape. It opens the native install prompt when available, otherwise
a keyboard-accessible dialog with Android/iPhone and embedded-browser steps.
Installed standalone apps hide the install action. Patient entry uses a compact centered composition,
aligned form columns and bounded group spacing rather than filling desktop height.
The Polar indicator embeds
the recovered dissertation belt illustration (the source was PNG, not SVG).
The layout reserves their space; narrow/short no-fit states use readable page reflow.
The top-left `VR zentrieren` headset/arrows button stays visible on every screen
and while scrolling. It is an inert, accessibly marked placeholder; future work
will recenter the headset's central field of view. See PROJECT.md for its contract.
The mock ECG is a continuously replayed seeded NeuroKit2 trace. Each started run
exports a participant-linked CSV and retains it for download recovery. All data logging must follow
[DATA-LOGGING.md](DATA-LOGGING.md); it owns the required naming convention.
The permanent bottom-right folder SVG opens a scrollable saved-CSV dialog on
every screen and is the sole visible entry to collected data; no separate
CSV-Dateien footer button is shown. It offers individual downloads, one all-file
ZIP, and native file sharing to available email/messaging apps, with a download fallback.
It lists this browser's saved exports rather than the OS Downloads directory.
Its **Per QR-Code teilen** flow can encrypt/upload the ZIP to Wormhole through
an authenticated upload relay (Render Free setup), display a local QR and send
the full link directly to Discord after a masked password prompt. The webhook
is published only as authenticated ciphertext; the password decrypts it on the
device. Discord uses no relay. Operator connection settings are per dialog; acknowledged
shares remain only in page memory. Manual upload/pasted-link QR and ordinary
downloads remain available. See [sharing setup](../sharing/README.md) and
[the data boundary](DATA-LOGGING.md). Hosted relay activation and
physical-phone qualification are recorded separately in PROJECT/VERIFICATION.

Read common rules first, then the documents needed for the task:

- [PROJECT.md](PROJECT.md): purpose, current page contract, and deployment facts.
- [WORKFLOW.md](WORKFLOW.md): scoped editing and publishing workflow.
- [SKILLS.md](SKILLS.md): always and conditional skill routes.
- [VERIFICATION.md](VERIFICATION.md): context, rendering, and deployment checks.
- [DECISIONS.md](DECISIONS.md): durable decisions and their reasons.
- [DATA-LOGGING.md](DATA-LOGGING.md): mandatory participant identity, filename,
  CSV schema, mock-source provenance, persistence and recovery contract.
- [Reusable audio engine](../audio/README.md): four separate repeating German
  phase cues, reference analysis, exact session timing, and verification.
- [Scenario protocol](../audio/scenarios/README.md): the JSON runtime timelines
  and matching CSVs for every final sequence, using independent numbered scenario assets.
- [Listening asset folder](../audio-assets/README.md): ready-to-listen local WAVs
  with adjacent timing/transcript JSONs, separate layers and 16-second cycles.
- [Audio review](../audio-review/00-START-HERE.txt): protocol/script inventory,
  participant listening walkthrough, audition comparison and seven complete
  timestamped text scores. This is a dated source-derived snapshot; refresh
  it when the scheduler, selection mapping or runtime samples change.
- [Voice samples](../samples/index.html): twenty published exact-phrase
  auditions from the LANG and EmoDB profile series, with source ratings and
  separate transcript evidence. This does not change the session voice.
- [Playback completion audit](PLAYBACK-AUDIT.md): evidence for the local browser
  Play/Stop behaviour and exact 5/10/15/20-minute audio totals.

The parent repository's [constraints](../../for-ai/constraints.md) also apply.
Update the affected notes with behavior changes; do not store chat transcripts,
credentials, build output, or speculative feature plans here.

Audio wording now follows the [text-first approval rule](WORKFLOW.md#exact-txt-scripts-and-wording-approval--2026-11-06):
every library audio asset has an adjacent exact-wording TXT. Prepare scripts and
wait for explicit user approval before generating/replacing audio. Avoid the
technical German calibration term in participant-facing narration/UI. Current
local proposals and approval status are in PROJECT.md and
`audio-review/04-AUDIO-TEXT-APPROVAL.txt`; published audio remains unchanged.
