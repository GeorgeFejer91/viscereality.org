# Haupt-Runde cue placeholders

Three ordered segments make up the 15-minute sequence.

| Segment | Order | Time |
| --- | --- | ---: |
| Opening | `control_1.wav` (26 s), `control_2.wav` (15 s), `instructions.wav` (40 s) | 81 s |
| Breathing | `inhale.wav`, `hold_in.wav`, `exhale.wav`, `hold_out.wav` (4 s each), repeated 50 times | 800 s |
| Closing | `end.wav` | 19 s |

The eight files contain silent mono PCM at 44.1 kHz and 16 bits. The empty
TXT sidecars match the silence. The JSON and CSV sidecars describe each cue
and its occurrences. [manifest.json](manifest.json) defines the order and
repetitions. [timeline.csv](timeline.csv) lists all 204 events.

This set uses the planned 15-minute timing. The selected Quest playback
uses the separate [native loop](../../../../../audio/box-loop.json).
