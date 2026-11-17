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
their full spoken TXT. The manifest records where to end each phase excerpt
before the attached hold and the words retained in the runtime cue. Each
excerpt fits its four-second slot using tail trim, pitch-preserving tempo
adjustment when needed, and silence. Holds use the two standalone takes.

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

The guided section runs from 5:00 to 15:00. It rotates the 18 voice takes,
with the recorded breath sound underneath inhale and exhale. It ends with a
four-second inhale and four-second exhale after 37 four-phase cycles. The
full review then plays the 19-second recorded outro. The music comes in at
the first guided inhale and fades out by 15:00.

The Probe-Runde stages the supplied 45.401-second instruction as pre-session
audio and then 300 seconds of guided breathing. It reuses the same phase
takes and ends with a final inhale, hold and exhale after 18 full cycles.
The instruction lives outside the five-minute Quest recording timeline.

The staged Quest programme stays at 39,690,000 frames. Its catalog includes
the outro as separate post-session audio. The currently released APK uses
the continuous shared loop. Staging this programme prepares the authored
assets for native integration and qualification.

Music: [“Wisps of Whorls”](https://incompetech.com/music/royalty-free/index.html?Search=Search&isrc=USUAN1200082)
by Kevin MacLeod (incompetech.com), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The preview uses an excerpt at reduced level with fades.
