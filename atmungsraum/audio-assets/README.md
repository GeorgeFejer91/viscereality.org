# Listening assets

Open `ready-to-listen/` for the current local German counted-cue auditions.
The `female/` and `male/` folders each contain:

- Four separate four-second voice cues: Einatmen, Halten, Ausatmen, Pause.
- The separate four-second inhale/exhale breathing layers.
- Four separate four-second mixed cues; holds contain voice alone.
- Three full 16-second cycles: voice, breath, and mix.

Every WAV has a JSON with the same basename **beside it**, containing the
transcript, duration, sample rate, frame count, SHA-256, role and timing.
`00-cycle-16s-mix.wav` is the complete voice-and-breath listening preview.
`00-cycle-16s-voice.wav` is the voice-only comparison. The breath-only cycle
has silent holds. Counts fade; the inhale/exhale breath grows in level.

JSON word times describe authored one-second slots, not measured phoneme onset
or duration. Cycle phases begin at 0/4/8/12 seconds. Separate metadata identifies
speech, breath, and silence. `manifest.json` lists all audio/JSON pairs.

Run `python tools/package-listening-assets.py` from the subproject to reproduce
this folder from the existing verified auditions, without synthesis or API calls.
The helper verifies source hashes, frame counts, pairing, component reconstruction,
and cycle concatenation. Different existing output files are preserved with an
error; choose a new `--output-dir` to package a revised source set.

These generated files remain local/ignored and do not change browser playback.
Existing source, speaker, and model permissions are recorded in
`audio/voice-sourcing.md`; this folder does not establish clinical release rights.
