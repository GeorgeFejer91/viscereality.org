# Project

- Purpose: provide a smartphone-compatible browser interface that controls actions
  on a VR headset screen through a corresponding Atmungsraum app.
- Intended user: someone operating the VR experience from a smartphone browser.
- Repository: <https://github.com/GeorgeFejer91/viscereality.org> (public).
- Default and publishing branch: `main`.
- Public URL: <https://viscereality.org/atmungsraum/>.
- Entrypoint: `atmungsraum/index.html`, relative to the repository root.
- Current website stack: static HTML, `style.css`, and plain JavaScript; local
  Pretext runtime pinned to `@chenglou/pretext` 0.0.9.
- Hosting: the existing GitHub Pages branch deployment from `main`, root `/`,
  with the existing `viscereality.org` custom domain.

## Intended product contract — 2026-10-12

The user approved this prototype for publication on 2026-10-12 as the main
navigation logic for the UI controlling the VR setup. It remains a prototype.
The bottom-level Start action is intended to play an audio sequence and initiate
a corresponding sequence of VR events. Browser audio already works; the VR event
set, command transport, timing, and synchronization still need to be defined.

The website is the smartphone-accessible control surface for the VR headset app.
The planned app should mirror the website's UI, with corresponding screens,
controls, labels, and actions. Adapt layout to the phone and headset while keeping
the meaning of those controls consistent.

Actions initiated in the phone browser should trigger the corresponding actions
on the VR headset screen. The intended interaction runs from phone browser to
headset app; controls must ultimately affect the headset experience, not only
the browser's local UI.

UI mirroring refers to corresponding interface elements. Whether live screen or
video streaming is needed is Undecided. Do not infer a breathing exercise or a
specific action set from the page name.

## First step: Polar H10 verification — 2026-10-12

The first page state is a verification gate that checks whether a Polar H10 is
connected. This precedes the round/session menu; session actions must not appear
ready before the H10 connection is verified.

When the H10 is connected, display its live ECG waveform in a box in the middle
of the interface. Use actual ECG samples from that sensor. A selected device,
heart-rate reading, or simulated waveform does not establish a working ECG feed.
Represent connecting, disconnected, and connected-but-no-ECG states accurately;
do not leave a stale trace presented as live after the connection or data stops.

The intended browser checks the headset and receives ECG from it. The H10
acquisition path within the headset app and the transport remain Undecided.
Choose the transport against the actual supported smartphones and browsers. The
verified ECG view enables a Weiter button to the round menu.

## Menu layout contract — 2026-10-12

Use the installed `uncodixfy-pretext` skill and its outside-in stretch-accordion
layout for every UI function in this menu, including the H10 gate and ECG view.
Allocate the outer panel first, retain header and action/footer anchors, preserve
ordered groups and label/control pairs, and distribute extra width and height
between groups and cells. Use shrinkable Grid/Flex tracks with `min-width: 0` and
`min-height: 0`; derive spacing and preferred type from both dimensions.

The bounded menu panel has no internal scrollbar. Reflow controls or revise the
supported minimum when they do not fit. Preserve readable type, touch targets,
keyboard order, visible focus, and user-enlarged text. This stretch geometry does
not require collapsible sections. Use actual locked Pretext measurements for all
bounded labels and status text, with font parity and a readable unavailable/no-fit
fallback. Verify the rendered layout; CSS controls geometry and Pretext measures
text. See [SKILLS.md](SKILLS.md) and [VERIFICATION.md](VERIFICATION.md).

## Round menu and UI prototype — 2026-10-12

Use easy, simple German throughout the interface. The working tree implements
this browser-local decision tree:

1. Emulate the headset searching for Polar, finding it, selecting/connecting it,
   then waiting for ECG. One compact status line shows the current step, with
   Polar auswählen available when found. The ECG box appears only during
   recording. Below it are only Erneut suchen and Weiter; Weiter is red/disabled
   until recording, then green/enabled. The status and canvas identify the demo.
2. Ask "Welche Übung wird durchgeführt?" and show Probe-Runde and Haupt-Runde
   together, with small inline sprout/wave symbols. Their descriptions are
   "Zur Eingewöhnung und für die erste Übung." and "Für die Hauptstudie."
   Remove the global brand/technical header. Use soft dark surfaces, muted sage
   and cream, gentle corners, and restrained borders. Narrow portrait choices
   may grow taller to keep descriptions readable; normal wide choices are square.
3. Show exactly four clickable boxes for the chosen round:
   - Probe-Runde: Anleitung (guided free exploration, no time limit), 5, 10,
     or 15 Minuten atmen. Anleitung and Frei ausprobieren are the same option.
   - Haupt-Runde: 5, 10, 15, or 20 Minuten.
4. Every box goes directly to a final screen with a **Session starten** play
   button. The selected round/option is shown for orientation. Starting is
   browser-local; the separately integrated audio controller owns playback.

There are no intermediate radio confirmations or separate instruction pages.
The initial final screen shows the play action; runtime phase/time/stop controls
appear only during playback. Real headset session execution remains pending.
Hash navigation supports Zurück and browser history; cold deep links still pass
through the connection gate. After Weiter there is no ECG return button. Signal
loss after this verified transition keeps the current screen and existing audio
while requesting reconnection every five seconds. Return to ECG only after more
than 30 seconds without recovery and at least one retry; clear the old waveform
and lock Weiter. Recovery before expiry cancels the deadline and retry count.
New sessions cannot start without current ECG. JavaScript is required for the demo; without it the
initial gate stays visible with an explanatory message.

Runtime assets remain local; Pretext 0.0.9 measures bounded text with the painted
Arial font. The header and footer anchor a bounded panel, and the middle controls
use shrinkable container-sized tracks. At enlarged text/no-fit the page explicitly
reflows into document flow, with no inner scrollbar and no font shrinking.

The actual product should receive verified connection state and live H10 ECG
from the headset. This UI demo implements neither real sensor acquisition nor
browser-to-headset transport. Do not present its synthetic waveform or local
playback as hardware verification or a command delivered to a headset.

The future headset transport can report `atmungsraum:sensor-state` CustomEvents
with `{ state: 'ecg' | 'no-ecg' | 'disconnected' }`. The browser emits
`atmungsraum:reconnect-request` with the retry number. These hooks implement the
UI's recovery behavior; an actual transport and verified ECG sample source are
still required. Do not mistake events or synthetic traces for hardware validation.

## German audio cues — 2026-10-12

Five separate German speech cues are saved in `assets/audio/breathwork/de/`:
Ankommen, Einatmen, Ausatmen, natürlich atmen, and Abschluss. Their exact text,
filenames, generation settings, durations, and SHA-256 hashes live in `cues.json`
beside the MP3 files. ElevenLabs returned the saved voice named
`British Breathing Instructor` (ID `rt9KyyQmwqa1uhHGJkMa`); use this voice for
these Atmungsraum cues. The generated set uses `eleven_multilingual_v2`, a speed
of 0.85, and 44.1 kHz / 128 kbps MP3 output.

These are local audio assets. Playback, breathing-cycle timing, and headset
integration are not implemented by their generation. All files passed full MP3
decode and integrity checks; German pronunciation and subjective voice review
remain pending.

## Reusable box-breathing engine — 2026-10-12

`audio/box-breathing.js` loads nine reusable German PCM WAV samples from
`assets/audio/box-breathing/de/soft-v2/manifest.json`. The phase samples are exactly
four seconds at 44.1 kHz. Repeat the four separate files `inhale.wav`, `hold_in.wav`,
`exhale.wav`, and `hold_out.wav` in that order. The instruction and ending samples
play once (40 and 12 seconds). The two hold files are independent roles with
identical Und halten text followed by quiet. Inhale/exhale say Einatmen/Ausatmen
with counts two/three/four. Welcome, ending, and rest use du. Delivery is softer
and slower; the exact phase wording follows the user's latest instruction.
Do not generate or concatenate new speech for each cycle.

The user selected exact 5/10/15/20-minute total durations including welcome,
ending, and rests. Complete 16-second cycles form blocks of at most 18 cycles
(288 seconds, the nearest complete-cycle duration below five minutes). A 20-second
natural-breathing rest separates blocks when another full cycle fits. Session
cycle counts are 15/33/50/68, with 0/1/2/3 rests and 8/0/8/0 seconds quiet recovery.
Every third complete cycle overlays both inhale and exhale with the supplied
breath sounds. Use the Web Audio clock and preload
all samples; JavaScript timers and ended callbacks must not trigger phase audio.
Hidden pages, interrupted audio clocks, failed preparation, or missed start
deadlines invalidate/abort an attempt. Host-side saving of receipts and physical
audio/sensor synchronization remain integration work. The browser controller
saves the last ten scheduling receipts locally; these are not acoustic onset
measurements or an integrated study data store.

The original German/English reference audio and native-caption transcripts are
downloaded under ignored `reference-audio/`. Newly synthesized German samples
use British Breathing Instructor. See [audio/README.md](../audio/README.md) for
the source analysis, scripts, timing contract, API, verification, and limits.
The round menu's final Play button is wired to browser playback through
`audio/session-ui.js`; Stop/Escape/navigation and interrupted clocks abort the run.
The main start button remains visible after starting, with smaller Start/Pause
buttons directly below it. Pause deliberately suspends the Web Audio context and
freezes the countdown; Start resumes the existing schedule from the same position.
Timed totals count active audio-clock time, so deliberate pauses extend wall-clock
duration. Pause/resume records are included in the local scheduling receipt.
No H10 or headset behavior is established by the audio checks.

Alternative voice sourcing is local audition work: two neutral, two positive,
and two restrained EmoDB profiles are saved in Secret Tunnel. The user rejected
the neutral takes as robotic and the positive takes as too energetic. Restrained
16-second counted cycles await listening. Archive CC0 and publisher CC-BY-4.0
metadata conflict; preserve attribution. LANG norms contain no low-arousal/high-positive-valence
match at the current thresholds. OmniVoice's pretrained model is CC-BY-NC and
speaker clone consent is not established by corpus metadata. Do not replace or
publish runtime samples as unrestricted clinic voices on this evidence. See
[voice-sourcing.md](../audio/voice-sourcing.md).

## Current implementation

The initial 2026-10-09 page was a blank black placeholder. The working tree now
contains the functional German UI prototype described above, including synthetic
ECG gating, four choices per round, and final play-button routes. Real H10 data,
headset connection, and headset session execution remain unimplemented. Browser
audio is wired separately through audio/session-ui.js. Keep the black visual
foundation and preserve work in progress.

The AI guidance lives at `atmungsraum/for-ai/`; it is not page content. This
subproject shares the parent Git repository and deployment, with no nested repo.

## Undecided

- H10 connection owner, transport, ECG acquisition, and supported phone browsers.
- Hardware verification of the headset-delivered ECG gate.
- Headset implementation of the selected session start and unlimited Anleitung.
- Headset platform, app stack, and app source location.
- Browser-to-headset transport, hosting requirements, and supported networks.
- Device discovery, pairing, access control, and headset-state feedback.
- Whether live screen or video streaming is part of the experience.
- The VR event sequence initiated by Start, including audio/event timing and
  synchronization between the browser and headset.
