# Decisions

## 2026-10-12 — Publish the main VR-control navigation prototype

The user approves publishing the current page online and adopts it as the main
navigation logic for the UI controlling the VR setup. This does not establish a
completed headset integration. At the bottom level, Start should run an audio
sequence and initiate a VR event sequence; event definitions, transport, timing,
and synchronization remain to be worked out. Publish the existing browser audio
and keep local audition/reference files out of the repository/deployment.

## 2026-10-12 — Audio and metadata kept together for listening

Keep ready-to-listen audition copies in `audio-assets/ready-to-listen/`, with
each WAV beside a same-basename JSON containing transcript, authored timing,
duration, frame count, hash, component and role. Package female/male counted
variants with separate voice, breath and mixed components, four-second phase
cues and full 16-second cycles. Breaths overlay inhale/exhale only; their cycle
has silent holds. These copies stay local/ignored and do not alter runtime assets.

## 2026-10-12 — Counted hold alternatives for audition

Add local variants with four counted phrases: Einatmen, Halten, Ausatmen, Pause,
each followed by zwei/drei/vier on one-second beats. Every reusable phase WAV is
exactly four seconds. Compare voice-only and fading breath-overlay versions for
both restrained voices; breath overlays remain on inhale/exhale only. Preserve
the quiet-hold auditions and current browser assets until a listening selection
is made. This request adds an audition alternative, not a runtime replacement.

## 2026-10-12 — Audition voice-to-breath fades before integration

Create four-second inhale/exhale elements with descending voice counts and a
rising breath recording. Match average mean-square power of the two layers
over the complete four-second block after envelopes, then apply shared mix
headroom gain. Holds say Halten once and remain quiet. The user explicitly
requested only an element audition first: preserve the current browser sample
set and cadence. Two restrained-voice prototypes and their isolated layers are
local/ignored; see `audio/voice-sourcing.md`. Listening acceptance is pending.

## 2026-10-12 — Connected phase wording with quiet holds

The latest requested wording is Einatmen … zwei … drei … vier … und halten …
Ausatmen … zwei … drei … vier … und halten. Keep the four independent four-second
roles. Spoken counts occur only during inhale/exhale; Und halten begins each
hold, followed by quiet. Welcome/rest/ending continue using du. This supersedes
Atme ein/Atme aus and the previous Pause/count hold text, while preserving the
timing, 20-second breaks, third-cycle breath overlays, and exact total duration.

The user found positive OmniVoice auditions too energetic after finding neutral
ones robotic. Restrained references combine mostly neutral and one positive
sentence from the same speaker. Use full 16-second cycle auditions to assess the
new wording and spacing. A chosen freely reusable clinic voice is still pending;
no OmniVoice audition is installed as a clinical runtime voice.

## 2026-10-12 — Minimal, friendly UI and delayed signal-loss return

Replace anatomical figures with small sprout/wave SVG symbols and gentle dark
surfaces, sage/cream accents, modest corners, and quiet borders. Remove global
branding and technical context. Exercise choice uses a short question and one
sentence per type; Haupt-Runde's description is only "Für die Hauptstudie."

The ECG gate has one headset/Polar status line, selection when a Polar is found,
an ECG view visible only while recording, and two bottom actions. Continue is
red/disabled before recording and green/enabled afterward. Do not add a debug
signal selector or explanatory paragraphs to this screen.

After Continue there is no return-to-ECG button. A lost signal requests reconnect
every five seconds while preserving the current screen and existing playback.
Only loss lasting more than 30 seconds after retries returns to ECG; recovery
within that window cancels the deadline. Keep all signal states explicitly
simulated until a real headset transport and sample feed are implemented.

## 2026-10-12 — Calm German voice sourcing

The user requested a locally synthesized OmniVoice profile through Secret Tunnel,
with a freely reusable German voice, and supplied local emotional-speech corpora
for low-arousal/high-valence selection. Prepare local auditions from licensed
references; do not describe emotion labels as numeric valence/arousal norms.
LANG's happy samples are high arousal and none meets the current combined screen.
The downloaded EmoDB metadata declares CC0, while the publisher record declares
CC-BY-4.0; preserve attribution and both terms. The user found both initial neutral
auditions too robotic and requested warmer, nicer delivery with more energy.
Positive-reference auditions are local comparisons, not selected runtime assets.
OmniVoice pretrained weights remain CC-BY-NC regardless of reference copyright.
Clinical/commercial output and speaker clone permission remain unresolved. Keep
auditions separate from runtime assets. See `audio/voice-sourcing.md`.

## 2026-10-12 — Direct, gentle speech and breathing rests

Use du and speak directly to the listener, including Atme ein/Atme aus. Give a
general welcoming introduction intended for clinic listeners. Use Pause with
counts instead of repeated harsh hold commands. Four phase files still repeat.
Exact 5/10/15/20-minute totals include welcome, ending, 20-second natural-breathing
rests, and quiet recovery. End breathing blocks at 18 complete cycles (288 seconds)
before a rest. Every third complete cycle overlays both inhale and exhale using
the supplied breath recordings. See `audio/README.md` for exact frame timing.

## 2026-10-12 — Emulated headset/ECG gate and four-choice tree

For now implement only the UI behavior. Emulate headset connection and ECG
reception, then enable Weiter to the two round choices. Actual headset transport
and H10 acquisition remain pending. Synthetic ECG must be clearly labeled and
cleared on missing data/disconnect.

Each round has exactly four direct choices. Anleitung and Frei ausprobieren are
one unlimited guided-exploration option; the other practice choices are 5, 10,
and 15 minutes. Main choices are 5, 10, 15, and 20 minutes. Each box leads directly
to a final Session starten play button, without an additional setup level. A separate audio integration owns browser-local playback;
its runtime controls stay hidden until playback begins.
This replaces the earlier five-option practice menu and native radio leaves.

## 2026-10-12 — Reuse four phase samples with exact session timing

The user requires four separate four-second audio cues, repeated in order:
inhale, hold after inhale, exhale, hold after exhale. Keep two separate hold files
and role identifiers, even when their spoken content is identical. Add a single
instruction and ending sample. Use newly synthesized German speech with the
saved ElevenLabs British Breathing Instructor voice.

The user chose exact 5/10/15-minute total sessions including instruction/ending,
complete 16-second cycles, and quiet recovery for the remainder. Preload reusable
PCM WAVs and schedule against one Web Audio clock. Runtime playback must not
depend on full-session files or chained JavaScript timers. Source recordings and
transcripts remain local analysis inputs. See [audio/README.md](../audio/README.md).

## 2026-10-12 — German round tree and repository artwork

The round menu has two square choices: Probe-Runde and Haupt-Runde. Use simple
German across the UI. Probe-Runde offers Anleitung, Frei ausprobieren, and
5/10/15-minute breathwork choices; Haupt-Runde offers 5/10/15/20 minutes. Each round
opens its own next decision layer with a Zurück route. These leaf selections are
local; content, actual sessions, and headset integration remain undecided.

Reuse the actual vector person paths from `Images/Picture1.svg` in both start
choices, adapted to white/black, with surrounding whitespace cropped and a small
waist-area trace-artifact mask. Attribution lives in `assets/person.svg`.
Keep static HTML/CSS, hash links, and native radio inputs. Pin Pretext 0.0.9 in
`package.json` and `pnpm-lock.yaml`, and ship its browser modules and MIT license
under `vendor/pretext/`; no CDN or framework is required.

## 2026-10-12 — H10 gate and menu geometry

The first page step must verify that a Polar H10 is connected, then display that
sensor's live ECG in a centered box. This precedes the session menu. Sensor
connection ownership, transport, and the onward transition are Undecided.

The user requires the combined `uncodixfy-pretext` skill's outside-in
stretch-accordion geometry for all UI functions in this menu. Keep anchors and
ordered groups, distribute available width and height, measure bounded text with
Pretext, and reflow on no-fit without an internal panel scrollbar. This supersedes
the earlier unspecified first screen and records the required UI layout policy.

## 2026-10-12 — Smartphone control surface and VR app

Atmungsraum's product goal is a smartphone-compatible browser interface that can
initiate actions on a VR headset screen. A corresponding headset app should
mirror the website's UI, keeping screens, controls, and action meanings aligned
across the two surfaces. Record this as the intended product contract while the
current page remains the blank placeholder. The headset platform, action set,
and connection architecture are Undecided; no implementation or publication is
part of this context update.

## 2026-10-09 — Initial page

The user requested `/atmungsraum` live on the existing GitHub Pages site, with a
subproject-local `for-ai/` folder, and specified a blank black page for now.
Use a static directory index with an empty body and a black document background.
Retain a descriptive browser-tab title without adding visible content.

## 2026-10-09 — Repository and workflow

Use the existing public `GeorgeFejer91/viscereality.org` repository and its verified
Pages source (`main`, `/`). Maintain a sparse, shallow, blob-filtered local checkout
because the user explicitly does not want the whole repository downloaded.

Adapt the `for-ai` skill's control-plane structure to this nested subproject.
Keep root guidance and the root subproject map as discovery routes. No separate
repository, site framework, build system, or future feature scaffold is needed.
