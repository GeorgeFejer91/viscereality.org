"""Package verified audition components with adjacent timing/transcript JSONs."""
import argparse
import hashlib
import io
import json
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RATE, FRAMES = 44100, 176400
PHASES = [('inhale', 'einatmen', 'Einatmen'), ('hold_in', 'halten', 'Halten'),
          ('exhale', 'ausatmen', 'Ausatmen'), ('hold_out', 'pause', 'Pause')]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f'Preserving different existing output: {path}')
    path.write_bytes(data)


def save_json(path, value):
    save(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def samples(data, seconds):
    with wave.open(io.BytesIO(data), 'rb') as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) != (1, 2, RATE, seconds * RATE):
            raise ValueError('Unexpected audio format or duration.')
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.int32)


def encode(pcm):
    if np.max(np.abs(pcm)) > 32767:
        raise ValueError('Component exceeds PCM headroom.')
    output = io.BytesIO()
    with wave.open(output, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(pcm.astype('<i2').tobytes())
    return output.getvalue()


def slots(label, offset=0):
    return [{'text': word, 'start_seconds': offset + beat, 'start_frame': (offset + beat) * RATE,
             'slot_end_seconds': offset + beat + 1, 'slot_end_frame': (offset + beat + 1) * RATE}
            for beat, word in enumerate([label, 'zwei', 'drei', 'vier'])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'audio-assets/ready-to-listen')
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    source_root = ROOT / 'reference-audio/counted-phase-variants'
    entries = []
    for voice, candidate in [('female', 'emodb-restrained-08'), ('male', 'emodb-restrained-03')]:
        out = args.output_dir / voice; out.mkdir(exist_ok=True)
        receipt = json.loads((source_root / candidate / 'voice-and-breath/receipt.json').read_text(encoding='utf-8'))
        cycles = {kind: [] for kind in ['voice', 'breath', 'mix']}
        cycle_segments = {kind: [] for kind in cycles}
        def asset(name, data, kind, phase, transcript, words, segments, source):
            pcm = samples(data, 16 if phase == 'cycle' else 4)
            audio_path = out / (name + '.wav'); json_path = out / (name + '.json')
            save(audio_path, data)
            metadata = {'audio_file': audio_path.name, 'sha256': sha(data), 'voice': voice, 'candidate': candidate,
                        'role': phase, 'component': kind, 'format': 'pcm_s16le', 'channels': 1,
                        'sample_rate': RATE, 'frames': len(pcm), 'duration_seconds': len(pcm) / RATE,
                        'transcript': transcript, 'word_slots': words, 'segments': segments,
                        'timing_basis': 'Authored PCM beat/phase placement; word slots are not measured acoustic boundaries.',
                        'source': source, 'use': receipt['use']}
            save_json(json_path, metadata)
            entries.append({'audio': audio_path.relative_to(args.output_dir).as_posix(),
                            'json': json_path.relative_to(args.output_dir).as_posix(),
                            'sha256': metadata['sha256'], 'frames': len(pcm), 'seconds': len(pcm) / RATE})
            assert json.loads(json_path.read_text(encoding='utf-8'))['sha256'] == sha(audio_path.read_bytes())
            return pcm
        for index, (phase, filename, label) in enumerate(PHASES):
            pcm, data = {}, {}
            sources = {}
            for kind, variant in [('voice', 'voice-only'), ('mix', 'voice-and-breath')]:
                original = source_root / candidate / variant / f'{phase}.wav'
                r = json.loads((original.parent / 'receipt.json').read_text(encoding='utf-8'))
                data[kind] = original.read_bytes()
                assert sha(data[kind]) == r['blocks'][phase]['sha256']
                pcm[kind] = samples(data[kind], 4)
                sources[kind] = {'project_file': original.relative_to(ROOT).as_posix(), 'sha256': sha(data[kind])}
            pcm['breath'] = pcm['mix'] - pcm['voice']
            assert np.array_equal(pcm['voice'] + pcm['breath'], pcm['mix'])
            data['breath'] = encode(pcm['breath'])
            sources['breath'] = {'derivation': 'Finished mixed PCM minus finished voice PCM; exact component reconstruction.',
                                'mix_sha256': sha(data['mix']), 'voice_sha256': sha(data['voice'])}
            if phase.startswith('hold'):
                assert not np.any(pcm['breath'])
            else:
                ratio = 10 * np.log10(np.mean(pcm['voice'].astype(float)**2) / np.mean(pcm['breath'].astype(float)**2))
                assert abs(ratio) < .0001
            for kind in cycles:
                speech = kind != 'breath'
                content = ('speech_and_breath' if kind == 'mix' and not phase.startswith('hold') else 'speech') if speech else ('silence' if phase.startswith('hold') else 'breath')
                segment = {'role': phase, 'content': content, 'start_seconds': index * 4,
                           'end_seconds': (index + 1) * 4, 'start_frame': index * FRAMES, 'end_frame': (index + 1) * FRAMES}
                cycle_segments[kind].append(segment)
                cycles[kind].append(pcm[kind])
                if kind == 'breath' and phase.startswith('hold'):
                    continue
                asset(f'{index + 1:02d}-{filename}-{kind}', data[kind], kind, phase,
                      receipt['blocks'][phase]['text'] if speech else '', slots(label) if speech else [],
                      [{**segment, 'start_seconds': 0, 'end_seconds': 4, 'start_frame': 0, 'end_frame': FRAMES}], sources[kind])
        for kind, blocks in cycles.items():
            combined = np.concatenate(blocks)
            transcript = ' '.join(receipt['blocks'][p]['text'] for p, _, _ in PHASES) if kind != 'breath' else ''
            words = [word for i, (_, _, label) in enumerate(PHASES) for word in slots(label, i * 4)] if kind != 'breath' else []
            exported = asset(f'00-cycle-16s-{kind}', encode(combined), kind, 'cycle', transcript, words,
                             cycle_segments[kind], {'derivation': 'Exact concatenation of the four phase components.'})
            assert np.array_equal(exported, combined)
    save_json(args.output_dir / 'manifest.json', {'asset_count': len(entries), 'phase_seconds': 4,
              'cycle_seconds': 16, 'component_types': ['voice', 'breath', 'mix'], 'assets': entries})
    save(args.output_dir / 'START-HERE.txt',
         ('Open female or male, then play 00-cycle-16s-mix.wav.\n'
          'Compare 00-cycle-16s-voice.wav and 00-cycle-16s-breath.wav.\n'
          'Files numbered 01-04 are the separate four-second phase cues.\n'
          'Every WAV has its timing and transcript in the adjacent JSON.\n'
          'Order: Einatmen / Halten / Ausatmen / Pause, with zwei / drei / vier.\n'
          'Local audition copies; browser playback is unchanged.\n').encode('utf-8'))
    print(json.dumps({'folder': str(args.output_dir.resolve()), 'audio_json_pairs': len(entries),
                      'verified': 'Source hashes, exact duration/frames, paired JSONs, component sums, cycle concatenation'}))


if __name__ == '__main__':
    main()
