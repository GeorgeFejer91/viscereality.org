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
PC and smartphone browsers serve exclusively as the external controller during
Quest runs. They send operator requests and display native state and receipts.
The VR-Probe controller discovers the Quest by Bluetooth, joins its gateway
over Wi-Fi, and starts Polar setup on Quest. The phone saves one verified Quest
pairing and tries it again when the app opens. The isolated rehearsal APK
defaults to Remote-only. Android handles Bluetooth permission. Enrolled
operators can see the live H10 preview and download the verified native
recording after each rehearsal.
Quest shows only the Viscereality scenario to the wearer and owns the runtime,
media, Polar H10 acquisition and recordings. **Meta Quest 2 and the installed
Android PWA are the required physical acceptance targets.** Desktop browsers
can run controller fixtures. The [active rehearsal checklist](PROJECT.md#quest-2-control-media-and-ui-acceptance--2026-11-12)
covers control, native media packaging and playback, UI stress, and evidence from
both devices. The study UI collects patient details and
shows the generated participant ID. It can display Quest-reported H10 readiness
and authorized ECG telemetry once qualified. All menu
UI functions use `uncodixfy-pretext` with stretch-accordion geometry. See
[SKILLS.md](SKILLS.md#main-ui-stretch-settings) for the concrete anchors,
dimension-responsive spacing/type, minimum-size and no-fit settings, and
[PROJECT.md](PROJECT.md) for implementation status and open decisions.

The live browser app opens the Quest controller. Quest owns recording,
audio playback and the Polar H10 connection. The phone displays the live ECG
preview and retains finalized files. New recordings contain a single participant
CSV with ECG, accelerometer, classified breathing and the applied renderer
radius. The controller downloads or sends the same file to the existing
private Drive folder. Earlier browser CSVs remain available from
[`#daten`](../index.html#daten). See [DATA-LOGGING.md](DATA-LOGGING.md).

Read common rules first, then the documents needed for the task:

- [PROJECT.md](PROJECT.md): purpose, current page contract, and deployment facts.
- [VR-UI-TESTING.md](VR-UI-TESTING.md) covers Meta XR Operator setup, VR UI
  interaction checks and headset evidence.
- [QUEST-AWAKE-TESTING.md](QUEST-AWAKE-TESTING.md) covers bounded awake holds,
  headset readiness, recovery, and cleanup for bench and rehearsal tests.
- [QUEST-PHONE-SYNC.md](QUEST-PHONE-SYNC.md): Quest 2 → installed Pixel PWA → private Google Drive collection procedure and stress test.
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
