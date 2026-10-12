"""Make local 4 × 4-second auditions from word-aligned connected speech.

Input MP3s and Whisper word alignments are local evaluation material. Output
never goes into public assets. Check the resulting speech by listening before
selecting a voice. ASR boundaries are approximate, not acoustic ground truth.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 44100
FRAMES = 4 * RATE
EXPECTED = ['einatmen', 'zwei', 'drei', 'vier', 'und', 'halten',
            'ausatmen', 'zwei', 'drei', 'vier', 'und', 'halten']


def pcm(path):
    result = subprocess.run([shutil.which('ffmpeg'), '-v', 'error', '-xerror', '-nostdin',
                             '-i', str(path), '-ar', str(RATE), '-ac', '1', '-f', 's16le', 'pipe:1'],
                            capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype='<i2').astype(np.float64) / 32768


def write(path, samples):
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(RATE)
        wav.writeframes(np.clip(np.rint(samples * 32767), -32768, 32767).astype('<i2').tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fragment(data, start, end, limit, leading_padding=.005):
    samples = data[round(start * RATE):round(end * RATE)].copy()
    active = np.flatnonzero(np.abs(samples) > 10 ** (-48 / 20))
    if not len(active):
        raise ValueError('A required word contains no speech.')
    samples = samples[max(0, int(active[0]) - round(leading_padding * RATE)):min(len(samples), int(active[-1]) + 441)]
    factor = 1.0
    if len(samples) > round(limit * RATE):
        factor = len(samples) / (limit * RATE)
        if factor > 1.5:
            raise ValueError('Speech cannot fit its beat without excessive acceleration.')
        raw = np.clip(np.rint(samples * 32767), -32768, 32767).astype('<i2').tobytes()
        result = subprocess.run([shutil.which('ffmpeg'), '-v', 'error', '-nostdin',
                                 '-f', 's16le', '-ar', str(RATE), '-ac', '1', '-i', 'pipe:0',
                                 '-af', f'atempo={factor}', '-f', 's16le', 'pipe:1'],
                                input=raw, capture_output=True, check=True)
        samples = np.frombuffer(result.stdout, dtype='<i2').astype(np.float64) / 32768
        if len(samples) > round(limit * RATE):
            raise ValueError('Retimed word still exceeds its beat; preserve it for review.')
    n = min(220, len(samples) // 2)
    samples[:n] *= np.linspace(0, 1, n); samples[-n:] *= np.linspace(1, 0, n)
    return samples, factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    prepared = []
    for source in sorted(args.directory.glob('*-cycle-source.mp3')):
        candidate = source.name.removesuffix('-cycle-source.mp3')
        alignment = json.loads((args.directory / f'{candidate}-cycle-alignment.json').read_text(encoding='utf-8'))
        words = alignment['words']
        if [re.sub(r'[^a-zäöüß]', '', w['word'].lower()) for w in words] != EXPECTED:
            raise ValueError(f'{candidate}: word alignment differs from the requested script.')
        data = pcm(source)
        boundaries = [0.] + [(a['end'] + b['start']) / 2 for a, b in zip(words, words[1:])] + [len(data) / RATE]
        if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
            raise ValueError('Word boundaries are not ordered.')
        groups = [('inhale', [0, 1, 2, 3]), ('hold_in', [(4, 5)]),
                  ('exhale', [6, 7, 8, 9]), ('hold_out', [(10, 11)])]
        clips, pieces = {}, []
        for phase, indices in groups:
            clip = np.zeros(FRAMES)
            for beat, index in enumerate(indices):
                first, last = index if isinstance(index, tuple) else (index, index)
                samples, factor = fragment(data, boundaries[first], boundaries[last + 1], 1.6 if phase.startswith('hold') else .94)
                clip[beat * RATE:beat * RATE + len(samples)] = samples
                pieces.append({'phase': phase, 'beat': beat, 'source_start': boundaries[first],
                               'source_end': boundaries[last + 1], 'speech_frames': len(samples), 'tempo_factor': factor})
            clips[phase] = clip
        combined = np.concatenate(list(clips.values()))
        rms = float(np.sqrt(np.mean(combined ** 2))); peak = float(np.max(np.abs(combined)))
        prepared.append((candidate, source, clips, combined, rms, peak, pieces))
    if not prepared:
        raise ValueError('No aligned cycle source files.')
    level = min(-24., *(float(-3 - 20 * np.log10(peak / rms)) for _, _, _, _, rms, peak, _ in prepared))
    receipts = []
    for candidate, source, clips, combined, rms, peak, pieces in prepared:
        out = args.directory / f'{candidate}-blocks'; out.mkdir(exist_ok=True)
        gain = 10 ** (level / 20) / rms
        blocks = {phase: {'file': f'{phase}.wav', 'frames': FRAMES, 'sha256': write(out / f'{phase}.wav', clip * gain)} for phase, clip in clips.items()}
        preview = args.directory / f'{candidate}-cycle-16s.wav'
        preview_hash = write(preview, combined * gain)
        receipt = {'candidate': candidate, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'preview': preview.name, 'preview_sha256': preview_hash, 'sample_rate': RATE, 'frames': 16 * RATE,
                   'phase_order': list(clips), 'blocks': blocks, 'pieces': pieces, 'rms_dbfs': level,
                   'peak_dbfs': float(20 * np.log10(peak * gain)), 'alignment': 'Whisper-small approximate word boundaries, with intervening pauses; listening required',
                   'deployment': 'Local audition only; model and speaker release restrictions unresolved'}
        (out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        receipts.append(receipt)
        print(json.dumps({'candidate': candidate, 'seconds': 16, 'preview': str(preview), 'rms_dbfs': level, 'peak_dbfs': receipt['peak_dbfs']}))
    (args.directory / 'cycle-receipts.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
