# German main study recordings

[manifest.json](manifest.json) lists the names, original attachments, durations,
SHA-256 hashes, transcripts and playback roles. Each MP3 is the original
recording under a descriptive name, with a same-name TXT. The 25 attachments
contain 24 distinct files: the 14:20:45 ElevenLabs export is identical to
`Ein2.mp3` and is listed as a second source for `einatmen-02.mp3`.

| Role | Recordings |
| --- | --- |
| Instruction, eyes closed | [Atem lauschen – Augen geschlossen](atem-lauschen-augen-geschlossen.txt) |
| Instruction, eyes open | [Atem lauschen – Augen offen](atem-lauschen-augen-offen.txt) |
| Box breathing instruction | [Main study instruction](main-study-instructions.txt) |
| Probe-Runde instruction | [Five-minute practice](probe-instructions.txt) |
| Previous box breathing instruction | [Boxatmung – Anleitung](boxatmung-anleitung.txt) |
| Inhale pool | `einatmen-01` through `einatmen-07` |
| Exhale pool | `ausatmen-01` through `ausatmen-09` |
| Hold pool | `halten-01` and `halten-02`, for both holds |
| Background | `wisps-of-whorls.mp3` (instrumental; empty TXT) |

Speech was transcribed locally with Whisper and normalized for German spelling,
punctuation and written-out counts. [Wispr Flow](https://wisprflow.ai/notetaker/how-to-transcribe-in-person-meetings)
does not accept prerecorded MP3s. The combined count-and-hold originals retain
their full spoken TXT. The manifest preserves the earlier phase-pool edit
boundaries. The authored programme now uses the canonical Zen cyclic recordings
from the [shared manifest](../../box-breathing/de/soft-v2/manifest.json).

## Review and staging

From the `atmungsraum` directory, use Node and FFmpeg:

```sh
node tools/render-main-study.mjs <new-output.wav>
node quest/stage-audio.mjs <new-output-directory> --main-study
```

Each control lasts 127.338775510 seconds (5,615,640 samples). The quiet
listening breaks follow the prompts to settle, notice inhaling and exhaling,
and return attention. The next sections resume at 0:28, 0:57, and 1:33 in
both controls. A final quiet period follows each closing prompt. The two
controls and the 45.322449-second instruction fill exactly five minutes.

The guided section runs from 5:00 to 15:00. It repeats the canonical Zen
inhale, hold, exhale and hold, with the recorded breath sound underneath
inhale and exhale. It ends with a
four-second inhale and four-second exhale after 37 four-phase cycles. The
full review then plays the 19-second recorded outro. The music comes in at
the first guided inhale and fades out by 15:00.

The Probe-Runde stages the supplied 45.401-second instruction as pre-session
audio and then 300 seconds of guided breathing. It reuses the same phase
takes and ends with a final inhale, hold and exhale after 18 full cycles.
The instruction lives outside the five-minute Quest recording timeline.

The staged Quest programme uses a lossless mix of 39,690,000 frames. Its
decoded PCM matches the first 900 seconds of the review render, and the
catalog binds the original outro separately. The authored build also includes the
first calibration instruction and the full Freie Erkundung recording from the
playground. Freie Erkundung has 960,768 instruction frames and 18,755,025 music
frames at 44.1 kHz. The Quest reader passes the original stereo MP3 to Android
with its instruction boundary and catalog hash verified. The authored
Haupt-Runde uses the timed source for recording and plays the outro after a
complete save. Existing signed packages use the continuous shared loop.

Source recordings are unchanged from the [calibration instruction](https://github.com/GeorgeFejer91/atmungsraum-ui/commit/9ed333e)
and [Freie Erkundung](https://github.com/GeorgeFejer91/atmungsraum-ui/commit/234ebd0ac154b8a5c16e7cbae6814534904b723d)
playground commits.

Music: [“Wisps of Whorls”](https://incompetech.com/music/royalty-free/index.html?Search=Search&isrc=USUAN1200082)
by Kevin MacLeod (incompetech.com), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The preview uses an excerpt at reduced level with fades.
