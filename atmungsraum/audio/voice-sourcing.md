# Calm German voice auditions

**Current choice:** the user rejected the OmniVoice listening versions and chose
the website's saved ElevenLabs **British Breathing Instructor** profile. Further
cues use that profile. Ready-to-listen copies are in
`audio-assets/ready-to-listen/elevenlabs/`; the rejected sets are archived locally.
The corpus and OmniVoice work below is retained as provenance, not a selected
voice or an active sourcing task.

The target is low arousal and positive valence: calm, warm delivery, clear German,
direct **du** address, and natural pauses. Emotional category labels are not enough
to establish those properties. Keep local auditions separate from the browser's
current production samples until listening and rights review are complete.

## Local corpus screening — 2026-10-12

`tools/prepare-calm-voice-references.py --database-dir <local-database-folder>`
reads the original LANG-audition workbook and EmoDB 2.0 metadata. It preserves
originals and writes norms, references, hashes, exact transcripts, and a selection
receipt to ignored `reference-audio/calm-voice-auditions/`.

LANG-audition contains 240 single-word recordings by two speakers. The
[publication](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0030086)
explains that ratings combine word meaning and prosody. Its positive material
was selected for high arousal; neutral material was selected for low arousal.
The local workbook confirms:

| Category | Recordings | Auditory arousal range (1–9) | Raw auditory valence range |
| --- | ---: | --- | --- |
| Negative | 80 | 6.50–8.13 | 6.30–8.13 |
| Neutral | 80 | 1.87–3.00 | 4.70–5.20 |
| Positive | 80 | 5.63–7.73 | 2.17–3.23 |

The workbook's auditory valence runs opposite to its visual valence: negative
categories have high auditory values. The tool retains the raw values and labels
`10 - raw` as an **inferred** positive-valence polarity conversion. On that
conversion, zero recordings meet the provisional screening thresholds of
positive valence ≥6 and arousal ≤3. These thresholds are a selection aid, not a
validated breathwork criterion. Do not label the least-arousing happy recordings
as low-arousal stimuli.

EmoDB supplies emotion-recognition confidence and naturalness, not continuous
valence/arousal norms. The provisional shortlist uses neutral recordings from
speaker 13 (female) and speaker 15 (male), with confidence and naturalness ≥0.8.
Each reference combines whole sentences from one speaker. Gain is adjusted once;
pitch and speed are preserved. Full references are 17.243 and 18.438 seconds.
Three-sentence lossless FLAC references fit the installed app's local API request
budget. Exact reference transcripts match only those included sentences.

Two saved Secret Tunnel profiles are named:

- `Atmungsraum EmoDB 13 neutral - audition`
- `Atmungsraum EmoDB 15 neutral - audition`

Secret Tunnel's MCP voice status/list calls returned internal errors. Its installed
local CLI reports native OmniVoice and ASR ready and successfully created both
profiles. Use the app's documented authenticated loopback CLI; do not expose a new
service or change connector permissions to work around a failed connector.

Both profiles generated the identical German welcome at speed 0.85, 32 model
steps, guidance 2.0, and seed 4120. Each completed in one take. Secret Tunnel's
Whisper word check recognized all 15 requested words with zero word edits.
The initial outputs were about five seconds for 15 words. Review WAVs therefore
use pitch-preserving 0.75× retiming, 30 ms boundary fades, and a single constant
gain to match -24 dBFS RMS, with peaks verified below -3 dBFS. Decoded files, hashes, exact
settings, and receipts are local. This verifies text fidelity and technical
integrity, not calmness, warmth, consent, or clinical suitability.

The user heard these neutral auditions and found both too robotic. The next
comparison uses positive/happiness references from speaker 8 (female) and 3
(male), again with confidence and naturalness ≥0.8. These categorical happiness
references may have more energy; they do not have numeric low-arousal norms.
Use `--emotion happiness --output-dir reference-audio/warm-voice-auditions` to
prepare them. Their whole-sentence reference montages are 16.120 and 7.327 seconds.
Saved profiles are `Atmungsraum EmoDB 8 positive - audition` and
`Atmungsraum EmoDB 3 positive - audition`. Compare model-produced speech at 0.85
speed without extra pitch/time stretching, to preserve its conversational rhythm.
Both positive auditions completed in one take with zero recognized word edits.
They passed strict MP3 decoding. Review WAVs preserve native model duration and
use matched -24 dBFS RMS with 30 ms fades; peaks remain below -3 dBFS. User
listening acceptance is pending. Source transcripts, clip IDs, settings, and
hashes are in the ignored warm-audition folder.

The user found the positive takes too energetic. A third comparison uses speakers
8 and 3 with predominantly neutral references and one short happiness sentence
from the same speaker (about one quarter of source speech time). This is a
reference selection experiment, not a validated emotional mixture or a model
emotion-strength parameter. Both restrained welcomes completed with zero word
edits at native speed 0.8. Profiles are named `Atmungsraum EmoDB 8 restrained - audition`
and `Atmungsraum EmoDB 3 restrained - audition`.

The user then specified **Einatmen … zwei … drei … vier … und halten … Ausatmen …
zwei … drei … vier … und halten**. Each restrained profile generated that connected
speech with preserved provenance and verified source text. Local Whisper-small
word alignments guide extraction; `tools/assemble-counted-voice-auditions.py`
places inhale/exhale words at 0/1/2/3 seconds and the two separate hold phrases
at their phase starts. Holds continue quietly. Four separate four-second WAVs
and a 16-second listening montage are saved per candidate under ignored
`reference-audio/restrained-voice-auditions/`. A second transcription of each
assembled cycle, without the reference prompt, recognized the entire requested
sequence. Exact frame/hash receipts distinguish source speech, approximate ASR
boundaries, trimming, fades, any beat retiming, and matched review level.
They remain unselected local auditions; subjective calmness/warmth is pending.

## Voice fading into breath — local audition only

The user requested inhale/exhale counts that progressively quieten while the
supplied breathing sound becomes louder, with both layers having equal average
power. The user explicitly chose to audition the element before deciding its
runtime cadence. Do not change the browser's samples or every-third-cycle rule
on the basis of this prototype.

`tools/build-power-crossfade-audition.py` renders local elements for the two
restrained profiles. Guide word/count onsets remain at 0/1/2/3 seconds. Voice beat
RMS targets descend by 4 dB per beat, while breath targets ascend in reverse.
Breath gain transitions are smoothed over 100 ms and its boundaries fade over
10 ms. Original inhale/exhale recordings are retimed with pitch preserved to
exactly four seconds.

After these envelopes, each complete four-second voice and breath layer is
normalized to unit RMS, including the guide's silent gaps. One shared gain is
then applied to both layers and all candidates before summation, with at least
3 dB sample-peak headroom. This makes the component mean-square powers equal;
it does not establish equal perceived loudness or constant instantaneous mix
power. Both holds use a fresh, source-verified **Halten** take followed by quiet.

Outputs are under ignored `reference-audio/power-crossfade-audition/`: four
separate phase WAVs per voice, isolated voice/breath layers for inspection,
16-second listening montages, and settings/hash/power receipts. The finished
16-bit PCM layers differ in power by less than 0.0001 dB. Voice per-beat RMS is
approximately -24/-28/-32/-36 dBFS; breath per-beat RMS is approximately
-35.9/-31.8/-27.8/-24.1 dBFS. All phases are exactly 176,400 frames; holds have
silent tails; no layer or mix clips; montage samples match the concatenated
phase files. Runtime asset hashes still match the existing browser manifest.
Human assessment of this balance and intelligibility remains pending. Existing
model, speaker, and reference licence constraints still apply.

## Counted Halten and Pause variants — local audition only

The next requested comparison counts every phase: **Einatmen … zwei … drei …
vier / Halten … zwei … drei … vier / Ausatmen … zwei … drei … vier / Pause …
zwei … drei … vier**. This adds an audition alternative to the quiet holds,
without changing the current browser samples or overlay cadence. Here Pause
is the fourth box-breathing phase, distinct from the engine's 20-second recovery.

`tools/build-counted-phase-variants.py` reuses the restrained voices' verified
inhale/exhale words and the existing fading mix stems. Fresh Pause labels passed
the local speech check; the male result was recovered from the saved catalogue
after a CLI timeout, with five automatic verification attempts. Halten uses the
previously verified standalone take. Labels are trimmed and faded, and the male
Halten is retimed with pitch preserved to fit its first beat without truncation.
Pause keeps 30 ms before the detected speech threshold to preserve its initial
consonant. A second unprompted Whisper-small transcription of each final cycle
recognized all sixteen requested words after this adjustment. The saved counts
are reused at 1/2/3 seconds; label onsets are at zero. All four
spoken beats descend by 4 dB. No breathing recording overlays either hold.

Each voice has a voice-only and a voice-and-breath set under ignored
`reference-audio/counted-phase-variants/`, with four separate 176,400-frame
four-second PCM WAVs, a 705,600-frame 16-second listening montage, and source,
settings, timing and hash receipts. Both sets use one shared additional 1.208 dB
attenuation for at least 3 dB sample-peak headroom. Inhale/exhale component powers
remain matched over each four-second block. The ZIP contains all sixteen reusable
elements, four listening montages, and receipts. Independent finished-PCM checks
confirmed durations, beat gaps, descending voice/rising breath, power differences
below 0.0001 dB, identical holds between mix versions, exact montage concatenation,
and archive integrity. Current runtime asset hashes still match their manifest.
Listening acceptance and the existing rights limits remain unchanged.

## Selected website voice — ElevenLabs listening set

`tools/build-elevenlabs-listening-cues.py` now uses complete, connected inhale and
exhale phrases synthesized with the same website profile/settings. Both holds reuse a
cached Halten! take with the literal exclamation mark and the same settings.
The complete phrases and Halten! use British Breathing Instructor
(`rt9KyyQmwqa1uhHGJkMa`), the same Eleven v4 model, soft/calm/slow tags, stability
0.5, similarity 0.65, German language and seed 42. The saved profile's name/ID and
the model's German TTS support were read from ElevenLabs before generation.

The selected roles are counted Einatmen / Halten! once / counted Ausatmen /
Halten! once. The user omitted hold counts and requested the exclamation mark.
Each hold contains about 0.75 seconds of native-speed speech and 3.25 seconds
of silence, with an exact four-second file duration. The user found
Pause's pronunciation English and preferred Halten in both holds. The four
inhale/exhale words fade by 4 dB per word; breath uses constant gain with natural
source dynamics and only 10 ms boundary fades. The added rising envelope was
rejected. The earlier ramped listening package is preserved under reference-audio.
Both count phrases are synthesized as complete utterances. Provider character
alignment locates word windows; native word samples retain their tempo while
silence is adjusted. Actual count starts are recorded in JSON, with brief delays
when a complete phase word exceeds a nominal one-second slot. This replaces
assembling the guide from separate single-word synthesis requests.
Full four-second voice/breath powers are matched before summation, with shared
gain and at least 3 dB sample-peak headroom. Hold labels retain complete native
speech and match the initial inhale cue's sample peak before the quiet tail.

The current package has 13 WAV/JSON pairs: four separate voice cues, two breath
components, four mixed cues, and three full 16-second voice/breath/mix cycles.
Independent PCM checks passed exact durations, hashes, authored slots, gain
progression, constant breath gain, component powers, sample reconstruction and
concatenation. Unprompted Whisper-small review of the initial set recognized all
sixteen requested words. Review of the current quiet-hold cycle recognized all
ten words with no hold counts; its second hold reuses the first hold PCM exactly.
Runtime assets and
scheduling remain unchanged; this is a listening set for the selected voice.

## Rights and release boundary

| Component | Evidence | Consequence |
| --- | --- | --- |
| EmoDB 2.0 recordings | Archive `db.yaml`: CC0-1.0/unrestricted; [publisher record](https://zenodo.org/records/17651657) API: CC-BY-4.0 | Conflicting metadata: retain both declarations and full attribution; do not describe the whole release as unambiguously public domain |
| LANG-audition publication/data | [PLOS publication](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0030086), Creative Commons Attribution | Preserve attribution and the actual source terms |
| GAUDIE audio | [Repository](https://osf.io/xyr6j/), restricted approved research access; no audio downloaded | Exclude from this freely reusable shortlist |
| OmniVoice pretrained weights | [Official model card](https://huggingface.co/k2-fsa/OmniVoice), CC-BY-NC | A permissive reference recording does not remove the model's noncommercial restriction |
| OmniVoice code/runtime | Apache 2.0; Secret Tunnel MIT | Code licences do not replace model licences |
| Higgs audio tokenizer | [Boson licence](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/blob/main/LICENSE), incorporating Meta terms | Additional model dependency terms remain applicable |
| Speaker likeness/clone permission | Not established by downloaded corpus metadata | Do not call the audition voices fully cleared for routine clinical/commercial deployment |

[CC0](https://creativecommons.org/publicdomain/zero/1.0/) allows copying and
adaptation, including commercial uses, but does not eliminate other people's
privacy/publicity rights or imply endorsement. Dataset copyright, permission to
clone a speaker, model use, and output deployment need distinct evidence.

The Zenodo record API was checked on 2026-10-12 and its licence evidence is saved
locally. To preserve attribution under the publisher's CC-BY-4.0 declaration,
credit **Felix Burkhardt, Astrid Paeschke, Miriam Kienast, Walter F. Sendlmeier,
and Benjamin Weiss**, *Berlin EmoDB 2.0*, v3,
[DOI 10.5281/zenodo.17651657](https://doi.org/10.5281/zenodo.17651657).
Changes for these auditions: selection/concatenation, gain adjustment, lossless
reference encoding and app resampling; new synthesized speech is derivative
evaluation material, with model settings and review processing recorded locally.
Preserve the [CC-BY-4.0 terms](https://creativecommons.org/licenses/by/4.0/).

Local OmniVoice auditions are for evaluation only. They are not shipped in
`assets/audio/` and do not replace the current ElevenLabs session voice. The user
has not resolved the intended clinical/commercial use against OmniVoice's
noncommercial terms. A reference explicitly contributed for synthetic speech,
such as [Thorsten Voice](https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full),
is another option, but would still need a compatible generation model.

Listening review should compare calmness, warmth, clarity, German pronunciation,
and repeated-cue comfort on the same welcoming text and at matched loudness.
Selecting a neutral reference or reducing playback speed does not establish
high valence or suitability for patients.
