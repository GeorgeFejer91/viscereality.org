# Shared audio asset library

Store reusable final-sequence audio here once. Scenario timelines under
[`audio/scenarios/`](../../audio/scenarios/) reference asset IDs from the selected
[`box-breathing/de/soft-v2/manifest.json`](box-breathing/de/soft-v2/manifest.json),
which owns filenames, duration, transcripts and integrity hashes. The current
runtime uses its nine WAVs; no scenario-specific full-session audio is loaded.

The earlier `box-breathing/de/` v1 and `breathwork/de/` MP3 sets retain their
existing paths for reference. Local source recordings, rejected auditions and
full verification exports remain ignored; the separate `samples/` voice
comparison page is outside final-sequence playback.
