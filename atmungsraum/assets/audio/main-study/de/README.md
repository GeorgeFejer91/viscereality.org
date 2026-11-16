# German main study recordings

[manifest.json](manifest.json) lists the names, original attachments, durations,
SHA-256 hashes, transcripts and playback roles. Each MP3 is the original
recording under a descriptive name, with a same-name TXT. The 23 attachments
contain 22 distinct files: the 14:20:45 ElevenLabs export is identical to
`Ein2.mp3` and is listed as a second source for `einatmen-02.mp3`.

| Role | Recordings |
| --- | --- |
| Instruction, eyes closed | [Atem lauschen – Augen geschlossen](atem-lauschen-augen-geschlossen.txt) |
| Instruction, eyes open | [Atem lauschen – Augen offen](atem-lauschen-augen-offen.txt) |
| Box breathing instruction | [Boxatmung – Anleitung](boxatmung-anleitung.txt) |
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

The 900-second review plays the three instructions once in the listed order,
45 full 16-second cycles, and a closing tail. It rotates the takes in each
phase pool in manifest order and uses different holds within each cycle.
The ordered asset IDs and exact frame positions are in the staged
`audio/scenarios/haupt-15.json`. The preview starts the music at the first
inhale and fades it out during the closing tail. No spoken closing clip was
supplied.

The staging switch emits the narration and cue WAVs for the authored Quest
scenario decoder. It does not change production assembly, the selected
continuous preparation loop, or any installed APK. The preview music mix is
separate from the staged narration. Confirm the study timing before switching
the selected Quest playback owner.

Music: [“Wisps of Whorls”](https://incompetech.com/music/royalty-free/index.html?Search=Search&isrc=USUAN1200082)
by Kevin MacLeod (incompetech.com), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The preview uses an excerpt at reduced level with fades.
