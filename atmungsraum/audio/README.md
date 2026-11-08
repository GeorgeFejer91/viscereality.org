# Reusable German box-breathing audio

Every final sequence loads its authored JSON timeline from
[`scenarios/`](scenarios/README.md), with a matching CSV timing sheet. All clips
come from each scenario’s own numbered folder under
[`assets/audio/scenarios/`](../assets/audio/scenarios/). All current runtime WAVs
are silent UI placeholders with matching draft TXT/JSON/CSV; final speech awaits
wording approval. The shared library recordings below are retained references. JSON determines timing/order.
CSV is generated from JSON and checked for drift with `pnpm run check:scenarios`.

## Browser background music

The browser demo starts [the supplied music](../assets/audio/background/ji3ji3.mp3)
from Starten and repeats it using the browser's audio player. It pauses with the
session, resumes at the same position, and releases the track on completion,
abort, or when the page is hidden. The next session starts from the beginning.
Music load failure leaves the timed cues and session controls available.

The public 7.65 MB MP3 keeps the full source timing and playback level. It is a
stereo 48 kHz export made with FFmpeg `-map 0:a:0 -map_metadata -1 -c:a libmp3lame -q:a 4`.
The same-basename TXT is empty for this background track. The scenario WAVs,
frame schedules, and receipts retain their existing role. In a Quest rehearsal
the headset owns stimulus audio.

## Session timing

**Main round (2026-11-06):** Haupt-Runde goes directly to one exactly 15-minute
sequence. Instruction 1 (26 s), Calibration (15 s), Instruction 2 (40 s), fifty
complete 16-second cycles (800 s) and Outro (19 s) total 900 seconds. There is no
baseline, main duration menu or between-cycle-block rest. See [the German
transcript, asset classes and Respyra comparison](calibration.md). Calibration
runs at 26–41 seconds and exposes an advance event/receipt for later H10
accelerometer integration; measured value/unit/method remain unset.
The new Instruction 1 uses Secret Tunnel's existing-instructor clone. The
original nine ElevenLabs samples and practice behavior retain their bytes.

The browser loads reusable samples once. It repeats four separate phase cues:
**inhale -> pause after inhale -> exhale -> pause after exhale**. Every phase is
exactly four seconds. Inhale/exhale cue and counts occur at 0, 1, 2, and 3
seconds. Each hold begins with Und halten and continues quietly to its boundary.
Instruction and ending play once. A separate rest sample plays between longer
breathing blocks. No speech generation or downloads occur during a session.

## Current sample set

The canonical manifest is `../assets/audio/box-breathing/de/soft-v2/manifest.json`.
All files are mono 16-bit PCM WAV at 44,100 Hz. The original v1 samples remain
available for comparison; the engine uses soft-v2.

| File | Length | Speech/role |
| --- | --- | --- |
| `instruction_1.wav` | 26 s | German preparation for two comfortable deep breaths |
| `calibration.wav` | 15 s | Separate silent measurement window |
| `outro.wav` | 19 s | Seven quiet seconds plus the unchanged12-second ending speech |
| `instruction.wav` | 40 s | Welcoming introduction, gentle rhythm explanation, permission to breathe naturally or stop |
| `inhale.wav` | 4 s | Einatmen. Zwei. Drei. Vier. |
| `hold_in.wav` | 4 s | Und halten. Then quiet |
| `exhale.wav` | 4 s | Ausatmen. Zwei. Drei. Vier. |
| `hold_out.wav` | 4 s | Und halten. Then quiet |
| `ending.wav` | 12 s | Gentle return to natural breathing and a moment to finish |
| `rest.wav` | 20 s | Natural-breathing rest, then a resumption prompt at 16 s |
| `inhale_breath.wav` | 4 s | The same inhale voice/counts with the supplied inhale recording beneath them |
| `exhale_breath.wav` | 4 s | The same exhale voice/counts with the supplied exhale recording beneath them |

The hold files are independent roles with identical spoken text. Their softer
phrase is Und halten, following the user's latest connected cue wording. There
are no spoken hold counts. Every third
complete cycle uses both breath-overlay variants. Cycle numbers count across
rests; rest and introduction speech do not change the overlay cadence.

Welcome, rest, and ending directly address the listener with du. The latest phase
labels use the requested Einatmen/Ausatmen/Und halten wording. Speech uses the
saved British Breathing Instructor voice, Eleven v4, and
requested soft/calm/slow delivery tags. Voice sample peaks are -8 dBFS. Long-form
narration is additionally retimed with pitch preserved to 80% speed; counts are
placed individually on exact one-second beats during inhale/exhale. Einatmen and
Ausatmen are slightly retimed to fit before the next beat. Und halten retains its
1.085-second utterance followed by silence. This is authored delivery and PCM
processing, not evidence of a human listening judgment.

The exact German welcome/ending/rest texts, settings, hashes, source durations,
and mix provenance are recorded in the manifest. The supplied Proton Drive
inhale (3.502 s) and exhale (4.411 s) are gently retimed to four seconds, faded at
the boundaries, and mixed at -20 dBFS sample peak beneath the voice. Original
recordings and synthesis responses stay in ignored `reference-audio/soft-box-v2/`.
No credential or private absolute source path is stored in the shipped manifest.

The requested alternative free German voice is being evaluated separately. See
[voice sourcing](voice-sourcing.md) for corpus norms, local audition profiles,
and the distinction between recording, speaker, and model permissions.

A newer voice-to-breath fade is an audition only: descending counts, rising
breath, equal four-second component RMS, and Halten with a quiet tail. The user
requested a listening element first, so it does not replace this runtime set.
See [voice sourcing](voice-sourcing.md#voice-fading-into-breath--local-audition-only).

Local counted-hold alternatives add Halten … zwei … drei … vier and Pause …
zwei … drei … vier, with voice-only and voice/breath versions. Their reusable
phase files are each exactly four seconds. They also remain auditions;
see [counted variants](voice-sourcing.md#counted-halten-and-pause-variants--local-audition-only).

The selected voice is the website's ElevenLabs British Breathing Instructor;
the user rejected the OmniVoice listening versions. Ready-to-listen copies live
in `audio-assets/ready-to-listen/elevenlabs/`.
The selected listening version says Halten! once in both hold phases, followed
by silence. It keeps inhale/exhale count fades and constant-gain breathing
recordings with natural dynamics.
Inhale/exhale are now synthesized as complete count phrases, then fitted to four
seconds by adjusting pauses while preserving native word speed. Their JSONs
record actual word timing, including small count delays needed for complete words.
Every WAV has an adjacent timing/transcript JSON. Separate voice, breath and mix
components and full 16-second cycles are packaged together; see the
[folder guide](../audio-assets/README.md). This local folder is ignored and does
not replace the browser sample manifest.

## Exact session timing

The user retained exact total durations including the welcome, ending, and
rests, and chose the nearest complete-cycle breathing block under five minutes:
18 cycles = 288 seconds. A 20-second natural-breathing rest follows such a block
only when another complete breathing cycle fits afterward. The last block may
be shorter; no cycle is cut. Any remainder follows the ending as quiet recovery.

| Total | Welcome | Breathing blocks (cycles) | Loop time | Rests | Ending | Quiet recovery |
| --- | --- | --- | --- | --- | --- | --- |
| 5 min / 300 s | 40 s | 15 | 240 s | 0 s | 12 s | 8 s |
| 10 min / 600 s | 40 s | 18 + 15 | 528 s | 20 s | 12 s | 0 s |
| 15 min / 900 s | 40 s | 18 + 18 + 14 | 800 s | 40 s | 12 s | 8 s |
| 20 min / 1200 s | 40 s | 18 + 18 + 18 + 14 | 1088 s | 60 s | 12 s | 0 s |

The manifest defines each asset's frame count; each saved scenario defines its
ordered, frame-indexed playback slots. The engine derives every scheduled
start/end from those slots and one AudioContext anchor. A phase always uses 176,400
source frames; rests always use 882,000. The first rest begins at t0 + 328 s,
after the 40-second welcome and 288 seconds of box breathing. Subsequent full
blocks start after 20-second rests. Recovery and rest are natural breathing,
not additional breath holds.

## Working browser UI

The final Play button preloads and verifies the samples, then starts the selected
scenario on a user action. Practice offers Anleitung and 5/10/15 minutes.
Haupt-Runde opens the single 15-minute main scenario directly. Its complete
900-second timeline and class boundaries are documented above. The calibration
window begins at t0+26 seconds, Instruction 2 at t0+41, and paced cycles at t0+81.
Receipts contain `calibration_start_frame`, `calibration_frames`, `instruction_1_frames`
and the complete `total_frames`; `baseline_frames` is 0.

The player shows the current phase/count and
remaining total time. The same button becomes Pause/Fortsetzen, retaining the
audio schedule and countdown across deliberate pauses. The back link reads
Beenden during playback and aborts before returning to the choice menu. Escape,
navigation away, the expired demo ECG recovery deadline, and
hidden/interrupted audio contexts cancel the attempt. Start is available again
after completion/abort. The unlimited instruction screen plays a 68-second
welcome/one-cycle/ending example; the screen itself has no time limit and the
example can be replayed.

Headset/ECG verification is an explicitly labelled simulation in this preview.
Audio is actually scheduled in the local browser. No sensor or headset connection
is established by browser playback.

The controller records selected scenario ID and JSON hash, asset-manifest hash, scheduling times,
audio sample rate, available latency/clock estimates, ended notifications, and
abort reason. The last ten receipts are saved locally under
`atmungsraum.audio-receipts`; no participant identity is collected. Receipts can
be inspected/exported through the experiment's eventual data system. Main-thread
ended notifications are not measured acoustic onsets.

## Engine API and reproduction

```js
import { BoxBreathingEngine } from './audio/box-breathing.js';
const engine = new BoxBreathingEngine();
await engine.prepare();
// From a user-initiated Start action after preparation:
const session = await engine.start(20);
const receipt = await session.done;
// engine.position() uses that same audio clock.
// engine.abort('user-abort') stops queued sources.
// await engine.dispose() cleans up the owned context.
```

An existing audioContext may be supplied to the constructor. The caller may pass
`startAt` to `start(minutes, { startAt })` to share an experimental audio anchor.
For the sole main round use `start(15, { mainRound: true })` (900 seconds total).
`engine.protocol(15, { mainRound: true })` returns its saved timeline.
`createProtocol()` remains an authoring/regression reference, not runtime timing.
The advance `atmungsraum:calibration-scheduled` event and receipt expose the
planned 26–41 second audio-clock measurement window; no sensor value is fabricated.
Default scheduling lead is 250 ms, outside the measured session; explicit starts
require at least 50 ms lead. `pause()` suspends the context, freezes its clock and
keeps all scheduled sources; `resume()` continues that same run. Both return
whether the transition completed and add a receipt record. Explicit pauses extend
wall-clock duration, while the selected active audio duration remains unchanged.
Unplanned context interruption still aborts; aborting during a pending resume
invalidates the transition and stops its sources.

`tools/build-soft-box-cues.py --key-file <external-file> --sounds-dir <folder>`
rebuilds the sample set using installed NumPy and FFmpeg. Successful ElevenLabs
responses are cached with their nonsecret request parameters. Changed requests
are rejected against existing cache entries; preserve/rename the cache before
regeneration. The key is read into memory and never written or printed.

Run `pnpm run test:audio`, `pnpm run check:scenarios` and `pnpm run check:ui`.
After editing scenario JSON, run `pnpm run scenarios:csv` to refresh the paired
CSVs. Serve this tree over HTTP and
open `tests/offline-audio.html` for a silent native Web Audio render of the actual
scheduler, comparing all output samples to the planned reused assets. Adding
`?round=haupt` checks the single main total and all class boundaries, at both
44.1 and 48 kHz. Optional
`pnpm run render:audio -- 20` exports a full verification WAV and adjacent
sample-indexed protocol. `--preview` exports the instruction example. Runtime
playback uses reusable samples, not these concatenated verification exports.
Existing exports are not overwritten; defaults include the manifest version.

Physical speaker latency, long-run device clock drift, phone/headset/Polar H10
synchronization, and suitability of voice delivery for the intended patients
require target-device and clinical listening review. This component does not
define study consent, eligibility, or medical/therapeutic outcomes.

## Analysis of the supplied reference recordings

Both original audio streams were downloaded into the ignored `reference-audio/`
folder. The final `source-m4a.m4a` files were decoded with FFmpeg's strict error
checking. Native-language caption transcripts and source/decode receipts are
saved beside them. Captions provide approximate speech timings, not acoustic
measurements. The earlier Opus download had a packet decoding error; the final
AAC/M4A copy is the verified source for each recording.

The [German MindCare Media guide](https://www.youtube.com/watch?v=wazCdqIBi2c)
has about 45 seconds of introductory speech. Its native German captions specify
four equal phases, with nasal inhalation and mouth exhalation. They list 12 phase
cues over three cycles, beginning around 46.350 s, then leave a long caption gap
until the closing remarks around 352.110 s. Early cues are descriptive; later
cues become short phase labels. Consecutive cue caption starts range from 3.877
to 4.118 seconds. The decoded original lasts 370.939 seconds. It is not an exact
five-minute total file. The adaptation retains the phase instructions and omits
benefit claims, channel promotion, and the longer introduction.

The [English Hands-On Meditation guide](https://www.youtube.com/watch?v=FDikCuovqxk)
starts by asking the listener to settle comfortably, then explains the same
four equal phases and the nose/mouth route. The native English captions contain
152 phase labels, consistent with 38 complete cycles, starting around 26.600 s
and ending with a hold cue around 632.520 s. It adds spoken counts within phases
and occasional encouragement. Its ending returns to natural breathing and a
quiet seated pause, followed by promotion. Caption cue intervals range from
3.839 to 4.880 seconds; these include first-cycle wording and caption placement.
The decoded original lasts 716.521 seconds. The adaptation retains the
four-phase order and natural-breathing ending. It uses one short cue per slot
and newly synthesized counts on exact one-second beats instead of importing
the recording or its uneven caption cadence.

Full downloaded transcripts stay local; the browser assets use newly synthesized
German speech. The reference recordings and their background tracks are not
included in the runtime asset set.
