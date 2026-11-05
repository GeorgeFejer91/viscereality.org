# Atmungsraum for-ai

This is the AI control plane for the `/atmungsraum` subpage. Product code lives
outside this folder, currently in `../index.html` and `../app.js`.

At entry, read [SKILLS.md](SKILLS.md) for the nested shared baseline and select
only the task-matching installed skill bodies. It includes `for-ai`, RTK, GSD
Core, Superpowers, Ponytail and the conditional UI, accessibility, performance,
state-management and native/remote routes. Resume the goal/checklist in
[PROJECT.md](PROJECT.md); before handoff, record evidence, blockers, the next
action and observed Git/publication state in that same owner.

The product goal is a smartphone browser control interface for a VR headset app
that mirrors the website's UI. Its required first step verifies a Polar H10
connection and shows the connected sensor's live ECG in a centered box. All menu
UI functions use `uncodixfy-pretext` with stretch-accordion geometry. See
[PROJECT.md](PROJECT.md) for the contract, implementation status, and open decisions.

The browser prototype is a local UI demo: simulated headset/ECG check →
round choice → four practice options or two main options → final Starten
play button. Haupt-Runde offers 10 or 15 minutes total: five minutes of normal
breathing, followed by the existing five- or ten-minute guided box-breathing
sequence. The normal-breathing period is silent and displays Normal atmen.
The same button becomes Pause during playback and Fortsetzen while paused. The
back action reads Beenden during playback and has a bordered button surface.
A read-only progress bar with elapsed/total time stays visible before, during
and after playback, including the 68-second Anleitung example. Pause freezes audio and the timer;
Fortsetzen resumes the same session. Ponytail applies to UI elements/copy: keep
only necessary controls, choice information, state, and recovery feedback. Anleitung is
unlimited guided exploration. Real H10/headset connection and execution remain
pending. See PROJECT.md for the precise demo and product boundaries.
The UI has no global context header or ECG return button after Continue. Signal
loss uses a 30-second reconnect window before returning to the minimal ECG gate.

Read common rules first, then the documents needed for the task:

- [PROJECT.md](PROJECT.md): purpose, current page contract, and deployment facts.
- [WORKFLOW.md](WORKFLOW.md): scoped editing and publishing workflow.
- [SKILLS.md](SKILLS.md): always and conditional skill routes.
- [VERIFICATION.md](VERIFICATION.md): context, rendering, and deployment checks.
- [DECISIONS.md](DECISIONS.md): durable decisions and their reasons.
- [Reusable audio engine](../audio/README.md): four separate repeating German
  phase cues, reference analysis, exact session timing, and verification.
- [Scenario protocol](../audio/scenarios/README.md): the JSON runtime timelines
  and matching CSVs for every final sequence, using the shared audio library.
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
