# Atemkalibrierung vor der Haupt-Runde

## Respyra reference

Respyra's [session description](https://github.com/embodied-computation-group/respyra#session-flow)
describes a 15-second breathing-range calibration. Its actual
[participant screen](https://github.com/embodied-computation-group/respyra/blob/dev/respyra/core/runner.py#L267-L408)
asks for normal, comfortable breathing across the full natural range without
straining, to set difficulty. The screen uses written instructions and a ready
key; it does not prescribe exactly two breaths or provide a German audio script.
Its README describes comfortable deep breaths; the screen emphasizes comfort
and natural range rather than maximal effort.

The runner collects belt-force samples, rejects extremes with percentile-derived
bounds, checks saturation, and offers accept/repeat. The
[default configuration](https://github.com/embodied-computation-group/respyra/blob/dev/respyra/configs/experiment_config.py#L72-L89)
uses 15 seconds, P5/P95 bounds and a target amplitude of 0.8 times the half-range
(with a 0.5 N floor in the runner). A separate natural-breathing baseline follows
and recenters trials. Sources checked 2026-11-06, including the current `dev`
runner and configuration; README wording was cross-checked against code.

## German audio transcript

This original adaptation asks for the user's requested **two** comfortable deep
breaths. It preserves Respyra's comfort/no-straining principle. Two breaths, the
spoken delivery and the class-based main sequence are Atmungsraum choices,
not a verbatim implementation of Respyra's protocol.

> Bevor wir beginnen, machen wir eine kurze Atemkalibrierung.
>
> Atme dafür gleich zweimal ruhig und tief ein und wieder aus. Nutze deinen
> natürlichen Atemraum, ohne dich anzustrengen.
>
> Atme nur so tief, wie es angenehm ist, und halte die Luft nicht an. Beginne jetzt.

**[Danach folgen 15 Sekunden Stille als eigene Calibration-Phase; diese Anmerkung wird nicht gesprochen.]**

## Runtime contract and asset classes

Haupt-Runde opens the sole `haupt-15` Start screen directly. There is no main
length-choice screen or five-minute baseline. The saved JSON/CSV timeline totals
exactly 900 seconds with these shared classes:

| Class | Asset | Start–end | Duration |
| --- | --- | --- | --- |
| Instruction 1 | `instruction_1.wav` | 00:00–00:26 | 26 s |
| Calibration | `calibration.wav` (silent measurement window) | 00:26–00:41 | 15 s |
| Instruction 2 | existing `instruction.wav` | 00:41–01:21 | 40 s |
| Inhale, pause, exhale, pause | existing four 4-second phase files | 01:21–14:41 | 50 × 16 s = 800 s |
| Outro | `outro.wav` | 14:41–15:00 | 19 s |

Instruction 1 is the German narration above. Instruction 2 is the existing
German welcome/box-breathing explanation in the manifest. Outro contains seven
seconds of quiet settling followed by the unchanged 12-second ending recording.
The main cycles are continuous; they use the existing every-third-cycle breath
variants without extra rest cues. Practice behavior and original nine WAVs are
unchanged. `main_asset_classes` in the manifest and `main-asset-classes.csv` name
all eight classes and their variants; adjacent JSON sidecars retain transcripts,
frame counts and hashes. Scenario CSV has exact offsets and `transcript_de`.
No complete per-scenario audio file is required.

## Calibration timing for the later accelerometer input

Calibration is a distinct event/asset, separate from its spoken preparation.
It starts at frame **1,146,600** and lasts **661,500 frames** at 44.1 kHz, ending
at frame **1,808,100**. Thus its audio-clock window is **t0 + 26 to t0 + 41 s**.
All nodes, including this silent window, are queued on one Web Audio clock.
Pause/Fortsetzen freezes and resumes that clock without retriggering calibration;
a new attempt schedules one fresh window.

The receipt's `calibration` object contains `offset_frames`, `duration_frames`,
`scheduled_start_audio_seconds` and `scheduled_end_audio_seconds`. The controller
emits `atmungsraum:calibration-scheduled` in advance with those fields, session
UUID, scenario ID, sample rate and clock name. This is a **planned audio-clock
window**, not a callback claiming acoustic onset. A later receiver must align
its sensor clock with the session audio clock and handle pause/abort; the event
is not an independent wall-clock timer or a successful measurement report.
Participant CSV `phase` uses `instruction_1`, `calibration`, `instruction_2`,
`inhale`, `hold_in`, `exhale`, `hold_out`, and `outro`; full planned duration is900 s.

The user specifies the ECG sensor's accelerometer as the future breathing input.
That source is not integrated yet. Measurement metadata explicitly uses
`signal_source: h10-accelerometer-pending`, `status: not-integrated`, and null
value/unit/method. No value is invented from the synthetic ECG. Respiratory
filtering, range estimation, quality/repeat rules and recording of the actual
calibration value are future integration work. Respyra's Vernier force units
and saturation thresholds must not be applied directly to an H10 accelerometer.

## Generation and evidence

The user requested Secret Tunnel on 2026-11-06. Instruction 1 uses its saved
existing-instructor clone, with native slow generation and no speech stretching.
Three short phrases passed native Whisper with zero word edits. The earlier
long take hit the Whisper length failure and remains unused; a fourth return-
to-normal phrase and the initial 48-second composite are retained privately as
superseded work. Other cues preserve the existing ElevenLabs recordings.
Private profile IDs, paths and complete native receipts remain ignored.
Technical decoding, word checks and scheduling do not establish listening
acceptance or hardware calibration.
