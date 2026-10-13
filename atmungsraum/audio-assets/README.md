# Listening assets

Open `ready-to-listen/elevenlabs/` for the current local German breathing cues.
They use **British Breathing Instructor**, the same saved ElevenLabs profile
as the website. The folder contains:

- Four separate four-second voice cues: Einatmen with counts, Halten!,
  Ausatmen with counts, Halten!.
  The two Halten roles have distinct files, numbered 02 and 04.
- The separate four-second inhale/exhale breathing layers.
- Four separate four-second mixed cues; holds contain voice alone.
- Three full 16-second cycles: voice, breath, and mix.

Every WAV has a JSON with the same basename **beside it**, containing the
transcript, duration, sample rate, frame count, SHA-256, role and timing.
`00-cycle-16s-mix.wav` is the complete voice-and-breath listening preview.
`00-cycle-16s-voice.wav` is the voice-only comparison. The breath-only cycle
has silent holds. Each spoken hold is Halten! once, followed by silence; no hold
counts are spoken. Inhale/exhale counts fade; their recordings use constant gain,
preserving their natural dynamics. Only 10 ms boundary fades prevent abrupt joins.

Inhale/exhale are each generated as one complete phrase. Only their pauses are
adjusted to fit four seconds; word audio is never accelerated or shortened.
Count starts target 1/2/3 seconds, with a small delay when the preceding complete
word needs it. JSON records the actual authored word slots, not measured phoneme
onset or duration. Cycle phases begin at 0/4/8/12 seconds. Separate metadata identifies
speech, breath, and silence. `manifest.json` lists all audio/JSON pairs.

Run `python tools/build-elevenlabs-listening-cues.py`, then
`python tools/package-listening-assets.py` from the subproject to reproduce
this folder from the selected voice's cached complete phrases and listening sources,
without synthesis or API calls. Complete inhale/exhale takes are cached with their
request and character-alignment response JSONs. Standalone Halten is cached with the
website's profile, model and voice settings. The Halten! take was synthesized
with the literal exclamation mark. Inhale/exhale phase labels/counts fade;
the supplied inhale/exhale sound keeps a fixed gain with equal average component
power. Both independent hold roles say Halten! and continue quietly.
The helper verifies source hashes, frame counts, pairing, component reconstruction,
and cycle concatenation. Different existing output files are preserved with an
error; choose a new `--output-dir` to package a revised source set.

The user rejected the OmniVoice auditions. They are preserved separately under
`archived-omnivoice/`; ElevenLabs is the selected voice for subsequent cues.
Historical sources can be packaged explicitly with `--source-dir` and a different
`--output-dir`. Generated listening files remain local/ignored and do not change
browser playback. See `audio/voice-sourcing.md` for provenance and history.
