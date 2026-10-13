# Decisions

## 2026-10-12 — Ponytail applies to UI elements and copy

Keep only controls, descriptors, and feedback needed to make a choice, understand
the current state, or recover from a failure. Remove decoration and duplicated
information rather than moving it elsewhere. This supersedes the separate
Start/Pause row and the round illustrations/descriptions below.

The final screen has one Starten/Pause/Fortsetzen button. Its existing back link
reads Beenden during playback and stops the session before returning to the
choice menu; Escape still stops in place. Keep phase/countdown, completion/error
feedback, and the main-round timing breakdown on the choice menu. Hide the search
retry when it cannot aid recovery. Ordinary loading-to-ready and playback
transitions do not add explanatory messages. Preserve accessibility, measured
full labels, and the existing audio/ECG gates.

## 2026-10-12 — Main round normal-breathing control period

Haupt-Runde offers only two total durations: 10 and 15 minutes. Both start
with exactly five minutes of normal breathing, then run the existing five-
or ten-minute guided box-breathing sequence. This supersedes the four main
choices below. Probe-Runde keeps Anleitung and 5/10/15 minutes.

The control period is silent and labelled Normal atmen. Schedule the following
welcome at the 300-second boundary using the same Web Audio clock; pause/resume
preserves that boundary. The existing welcome/ending/rest/recovery timing stays
inside the guided-segment budget. Menu cards show the breakdown, and the final
screen shows the selected total; receipts distinguish total, baseline and
guided duration.

## 2026-10-12 — Start and Pause beneath the main start button

Keep the final screen's main start button. After it starts a session, show a
smaller Start/Pause row directly underneath. Pause freezes audio and the timer;
Start resumes that same session without resetting or queuing another protocol.
Disable the main button during the run and enable only the relevant smaller
control. Stop/Escape/navigation and the existing signal-loss deadline also abort
paused sessions. Explicit pauses extend wall-clock duration while preserving the
selected active audio duration. Future VR-event synchronization must account for
these pauses when that transport is implemented.

## 2026-10-12 — Publish the main VR-control navigation prototype

The user approves publishing the current page online and adopts it as the main
navigation logic for the UI controlling the VR setup. This does not establish a
completed headset integration. At the bottom level, Start should run an audio
sequence and initiate a VR event sequence; event definitions, transport, timing,
and synchronization remain to be worked out. Publish the existing browser audio
and keep local audition/reference files out of the repository/deployment.

## 2026-10-12 — Generate complete count phrases, then adjust pauses

Always synthesize Einatmen … zwei … drei … vier and Ausatmen … zwei … drei …
vier as complete phrases with the selected ElevenLabs voice. Preserve the native
word audio and phrase prosody; adjust only gaps to make each phase exactly four
seconds, alongside the existing count-level fades. Complete words take precedence
over nominal one-second count starts: defer a count briefly when needed and store
the actual output-frame timing in JSON. Do not accelerate speech or join separate
single-word synthesis takes. Retain quiet Halten! holds and constant breath gain.

## 2026-10-12 — Halten! once and quiet in both holds

Omit zwei/drei/vier after each hold and synthesize literal Halten! with the
selected website ElevenLabs profile and existing soft/calm/slow settings.
Both independent four-second hold files contain the prompt once, then silence.
Retain inhale/exhale count fades and constant-gain breathing clips. Every phase
stays four seconds and every component cycle stays 16 seconds. Update transcripts,
word slots and speech/silence segments; preserve the earlier counted-hold package
locally. This supersedes counted holds in the selected listening set.

## 2026-10-12 — Halten in both holds and constant breath gain

The user prefers Halten to Pause because Pause sounds English in the chosen
voice. Use the cached German Halten take in both independent hold roles, with
the existing counts. Remove the added breath-volume ramp: apply one constant
gain to each retimed inhale/exhale recording, retaining natural source dynamics
and only short boundary fades. Keep descending voice counts and equal average
four-second voice/breath component power. Update the local listening WAVs and
adjacent JSONs, preserving the prior ramped listening set under reference-audio.
This supersedes Pause and rising breath in the selected listening prototype.

## 2026-10-12 — Retain the website's ElevenLabs voice

The user rejected the OmniVoice listening voices and selected the existing
website ElevenLabs British Breathing Instructor profile for further cues.
Rebuild the counted/fading listening elements with that profile, reusing website
inhale/exhale/count recordings and cached Pause; synthesize only standalone
Halten with the same model/settings. Keep four-second phases, all three separate
component types and 16-second cycles with adjacent JSONs. Archive the rejected
local sets; make ElevenLabs the default listening package. Runtime voice remains
the existing ElevenLabs profile; this change does not revise playback scheduling.

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
