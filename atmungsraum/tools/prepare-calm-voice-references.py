"""Prepare local voice auditions; never classify neutral speech as validated calm.

Requires numpy, soundfile and xlrd. Originals stay untouched. Outputs belong in
the ignored reference-audio directory, not the public runtime asset directory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import xlrd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(paths, output):
    segments, rate = [], None
    for path in paths:
        data, sample_rate = sf.read(path, dtype="float32")
        assert data.ndim == 1, path
        assert rate is None or rate == sample_rate
        rate = sample_rate
        # Preserve speed and pitch; use one gain for the whole reference.
        segments.extend([data, np.zeros(round(rate * .15), dtype=np.float32)])
    data = np.concatenate(segments[:-1])
    data *= 10 ** (-8 / 20) / max(float(np.max(np.abs(data))), 1e-9)
    sf.write(output, data, rate, subtype="PCM_16")
    return len(data) / rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reference-audio/calm-voice-auditions"))
    parser.add_argument("--emotion", choices=("neutral", "happiness", "restrained"), default="neutral")
    args = parser.parse_args()
    root, output = args.database_dir, args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    book = xlrd.open_workbook(root / "LANG_audition/data/LANG-audition.xls")
    sheet = book.sheet_by_index(0)
    norms = []
    for i in range(4, sheet.nrows):
        r = sheet.row_values(i)
        norms.append({"word": r[0], "file": r[2], "category": r[3], "speaker": r[4],
                      "auditory_valence_raw": r[14], "auditory_valence_sd": r[15],
                      "auditory_arousal_mean": r[16], "auditory_arousal_sd": r[17],
                      # Polarity inferred from category/rating correspondence:
                      # negative mean 7.72, positive mean 2.69 in this workbook.
                      "pleasantness_reversed_inferred": 10 - r[14]})
    with (output / "lang-audition-norms.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=norms[0].keys())
        writer.writeheader()
        writer.writerows(norms)
    summary = {}
    for cat in ("negative", "neutral", "positive"):
        rows = [r for r in norms if r["category"] == cat]
        summary[cat] = {"count": len(rows),
                        "arousal_range": [min(r["auditory_arousal_mean"] for r in rows), max(r["auditory_arousal_mean"] for r in rows)],
                        "raw_valence_range": [min(r["auditory_valence_raw"] for r in rows), max(r["auditory_valence_raw"] for r in rows)]}

    db = root / "EmoDB_2.0/data/emodb_2.0"
    files = {r["file"]: r for r in csv.DictReader((db / "db.files.csv").open())}
    emotions = list(csv.DictReader((db / "db.emotion.csv").open()))
    # Sentences from one speaker per reference; no mixed speaker identities.
    texts = {"a01": "Der Lappen liegt auf dem Eisschrank.", "a02": "Das will sie am Mittwoch abgeben.",
             "a04": "Heute abend könnte ich es ihm sagen.", "a05": "Das schwarze Stück Papier befindet sich da oben neben dem Holzstück.",
             "a07": "In sieben Stunden wird es soweit sein.", "b01": "Was sind denn das für Tüten, die da unter dem Tisch stehen.",
             "b02": "Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter.",
             "b03": "An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.",
             "b09": "Ich will das eben wegbringen und dann mit Karl was trinken gehen.",
             "b10": "Die wird auf dem Platz sein, wo wir sie immer hinlegen."}
    candidates = []
    speakers = (("13", "female"), ("15", "male")) if args.emotion == "neutral" else (("8", "female"), ("3", "male"))
    for speaker, gender in speakers:
        eligible = [r for r in emotions if r["emotion"] in ({"neutral", "happiness"} if args.emotion == "restrained" else {args.emotion})
                    and files[r["file"]]["speaker"] == speaker
                    and float(r["emotion.confidence"]) >= .8 and r["emotion.naturalness"]
                    and float(r["emotion.naturalness"]) >= .8]
        eligible.sort(key=lambda r: (-float(r["emotion.naturalness"]), -float(r["emotion.confidence"]), r["file"]))
        if args.emotion == "restrained":
            # Whole sentences from the same person, with about three quarters
            # neutral speech. This is an audition reference, not an emotion
            # interpolation control or a validated arousal score.
            neutral = [r for r in eligible if r["emotion"] == "neutral"]
            positive = [r for r in eligible if r["emotion"] == "happiness"]
            eligible = neutral[:2 if gender == "female" else 3] + positive[:1]
        selected, duration, sentence_ids = [], 0, set()
        for row in eligible:
            sentence = files[row["file"]]["transcription"]
            if sentence in sentence_ids:
                continue
            selected.append(row)
            sentence_ids.add(sentence)
            duration += sf.info(db / row["file"]).duration + .15
            if duration >= 16:
                break
        name = f"emodb-{args.emotion}-{int(speaker):02}"
        path = output / f"{name}.wav"
        seconds = assemble([db / r["file"] for r in selected], path)
        # The installed local control API has a 256 KiB request budget. A
        # lossless three-sentence reference fits it without cutting a sentence.
        profile_rows = selected if args.emotion == "restrained" else selected[:3]
        profile_path = output / f"{name}-profile.wav"
        profile_seconds = assemble([db / r["file"] for r in profile_rows], profile_path)
        profile_flac = profile_path.with_suffix(".flac")
        profile_data, profile_rate = sf.read(profile_path, dtype="int16")
        sf.write(profile_flac, profile_data, profile_rate, subtype="PCM_16")
        assert profile_flac.stat().st_size < 190_000, "Reference exceeds local API budget"
        candidates.append({"candidate": name, "speaker_id": speaker, "gender": gender,
                           "reference_file": path.name, "seconds": seconds, "sha256": digest(path),
                           "reference_text": " ".join(texts[files[r["file"]]["transcription"]] for r in selected),
                           "profile_reference_file": profile_flac.name, "profile_reference_seconds": profile_seconds,
                           "profile_reference_sha256": digest(profile_flac),
                           "profile_reference_text": " ".join(texts[files[r["file"]]["transcription"]] for r in profile_rows),
                           "profile_positive_speech_fraction": sum(sf.info(db / r["file"]).duration for r in profile_rows if r["emotion"] == "happiness") / sum(sf.info(db / r["file"]).duration for r in profile_rows),
                           "source_license": "Archive db.yaml declares CC0-1.0; publisher record declares CC-BY-4.0. Preserve attribution and both declarations.",
                           "source_metadata_sha256": digest(db / "db.yaml"),
                           "source_url": "https://zenodo.org/records/17651657",
                           "emotion_evidence": f"{args.emotion} recognition and naturalness, not numeric valence/arousal",
                           "voice_cloning_consent": "not established by corpus metadata",
                           "intended_use": "local audition only; no clinical release clearance",
                           "sources": [{**r, "sha256": digest(db / r["file"])} for r in selected]})

    receipt = {"lang_norms": summary,
               "screening_threshold": {"pleasantness_reversed_inferred_min": 6, "arousal_max": 3},
               "lang_matching_recordings": sum(r["pleasantness_reversed_inferred"] >= 6 and r["auditory_arousal_mean"] <= 3 for r in norms),
               "scale_note": "Workbook auditory valence decreases with positive categories. Reversal 10-x is an explicitly inferred polarity conversion; raw scores retained. Ratings combine semantics and prosody.",
               "lang_source": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0030086",
               "omnivoice_model_license": "CC-BY-NC; independent of reference copyright licence",
               "omnivoice_model_source": "https://huggingface.co/k2-fsa/OmniVoice",
               "candidates": candidates}
    (output / "selection.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"lang_matching_recordings": receipt["lang_matching_recordings"],
                      "candidates": [{k: c[k] for k in ("candidate", "gender", "seconds", "reference_file")} for c in candidates]}, indent=2))


if __name__ == "__main__":
    main()
