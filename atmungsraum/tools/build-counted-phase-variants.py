"""Local counted-hold auditions; four exact four-second files per variant.

Reuses the verified inhale/exhale words and mix stems. Halten/Pause labels
come from saved, word-verified Secret Tunnel takes. No runtime assets change.
"""
import argparse
import json
import shutil
import subprocess
import wave
import zipfile
from pathlib import Path

import numpy as np

import importlib.util

spec = importlib.util.spec_from_file_location('mix', Path(__file__).with_name('build-power-crossfade-audition.py'))
mix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mix)
fragment = mix.load_tool('assemble-counted-voice-auditions.py').fragment
RATE, FRAMES, ORDER = mix.RATE, mix.FRAMES, mix.ORDER
TEXTS = dict(zip(ORDER, ['Einatmen … zwei … drei … vier.', 'Halten … zwei … drei … vier.',
                        'Ausatmen … zwei … drei … vier.', 'Pause … zwei … drei … vier.']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voices-dir', type=Path, required=True)
    parser.add_argument('--mixes-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parents[1] / 'assets/audio/box-breathing/de/soft-v2/manifest.json'
    runtime_before = mix.digest(runtime)
    prepared = []
    for record in sorted(args.output_dir.glob('*-pause-generated.json')):
        candidate = record.name.removesuffix('-pause-generated.json')
        previous = json.loads((args.mixes_dir / candidate / 'receipt.json').read_text(encoding='utf-8'))
        gain = 10 ** (previous['shared_component_rms_dbfs'] / 20)
        voice, breath, labels = {}, {}, {}
        for phase in ['inhale', 'exhale']:
            voice[phase] = mix.pcm(args.mixes_dir / candidate / f'{phase}-voice.wav')
            breath[phase] = mix.pcm(args.mixes_dir / candidate / f'{phase}-breath.wav')
        counts = mix.pcm(args.voices_dir / f'{candidate}-blocks/inhale.wav')
        for phase, label, folder in [('hold_in', 'halten', args.mixes_dir), ('hold_out', 'pause', args.output_dir)]:
            take = json.loads((folder / f'{candidate}-{label}-generated.json').read_text(encoding='utf-8-sig'))
            if take['clip']['word_fidelity']['word_edits'] != 0 or take['clip']['text'].lower() != label + '.':
                raise ValueError('Label source failed its word/content check.')
            source = args.output_dir / f'{candidate}-{label}-source.mp3'
            shutil.copyfile(take['path'], source)
            data = mix.pcm(source)
            word, tempo = fragment(data, 0, len(data) / RATE, 1.6, leading_padding=.03 if label == 'pause' else .005)
            if len(word) > .94 * RATE:
                # FFmpeg atempo duration is approximate. Target 900 ms and check
                # the complete result rather than cutting a word at the beat.
                tempo = len(word) / (.9 * RATE)
                if tempo > 1.5:
                    raise ValueError('Label needs a shorter source take.')
                raw = np.rint(word * 32767).astype('<i2').tobytes()
                result = subprocess.run([shutil.which('ffmpeg'), '-v', 'error', '-nostdin', '-f', 's16le',
                                         '-ar', str(RATE), '-ac', '1', '-i', 'pipe:0', '-af', f'atempo={tempo}',
                                         '-f', 's16le', 'pipe:1'], input=raw, capture_output=True, check=True)
                word = np.frombuffer(result.stdout, dtype='<i2').astype(np.float64) / 32768
                if len(word) > .94 * RATE:
                    raise ValueError('Complete retimed label still exceeds its beat.')
                word[:220] *= np.linspace(0, 1, 220); word[-220:] *= np.linspace(1, 0, 220)
            assembled = counts.copy(); assembled[:RATE] = 0
            assembled[:len(word)] = word
            voice[phase], _ = mix.faded_beats(assembled, mix.VOICE_DB)
            voice[phase] *= gain
            labels[phase] = {'label': label, 'clip_id': take['clip']['clip_id'], 'source_sha256': mix.digest(source),
                             'source_word_edits': 0, 'speech_frames': len(word), 'tempo_factor': tempo,
                             'counts_source_sha256': mix.digest(args.voices_dir / f'{candidate}-blocks/inhale.wav')}
        prepared.append((candidate, voice, breath, labels, previous))
    if not prepared:
        raise ValueError('No saved Pause takes. Generate labels before rendering.')
    peak = max(float(np.max(np.abs(voice[p] + breath.get(p, 0))))
               for _, voice, breath, _, _ in prepared for p in ORDER)
    peak = max(peak, *(float(np.max(np.abs(v))) for _, voice, _, _, _ in prepared for v in voice.values()))
    attenuation = min(1., 10 ** (-3 / 20) / peak)
    receipts, packaged = [], []
    for candidate, voice, breath, labels, previous in prepared:
        for variant in ['voice-only', 'voice-and-breath']:
            out = args.output_dir / candidate / variant; out.mkdir(parents=True, exist_ok=True)
            blocks, pcm_blocks = {}, []
            for phase in ORDER:
                data = (voice[phase] + (breath.get(phase, 0) if variant == 'voice-and-breath' else 0)) * attenuation
                path = out / f'{phase}.wav'
                blocks[phase] = {'file': path.name, 'text': TEXTS[phase], 'frames': FRAMES,
                                 'sha256': mix.write(path, data), 'sample_peak_dbfs': mix.db(float(np.max(np.abs(data))))}
                # Read the finished PCM so the montage is sample-identical to its elements.
                with wave.open(str(path), 'rb') as wav:
                    assert (wav.getnframes(), wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (FRAMES, RATE, 1, 2)
                    pcm_blocks.append(wav.readframes(FRAMES))
                packaged.append(path)
            preview = args.output_dir / f'{candidate}-{variant}-counted-16s.wav'
            with wave.open(str(preview), 'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(RATE); wav.writeframes(b''.join(pcm_blocks))
            receipt = {'candidate': candidate, 'variant': variant, 'rate': RATE, 'channels': 1, 'format': 'pcm_s16le',
                       'phase_frames': FRAMES, 'phase_order': ORDER, 'word_onsets_frames': [0, RATE, 2 * RATE, 3 * RATE],
                       'voice_beat_targets_db': mix.VOICE_DB.tolist(), 'breath_only_on': ['inhale', 'exhale'] if variant == 'voice-and-breath' else [],
                       'shared_extra_attenuation_db': mix.db(attenuation), 'blocks': blocks, 'label_sources': labels,
                       'previous_mix_receipt_sha256': mix.digest(args.mixes_dir / candidate / 'receipt.json'),
                       'preview': preview.name, 'preview_frames': 4 * FRAMES, 'preview_sha256': mix.digest(preview),
                       'use': 'Local audition only; existing source, speaker and model permissions remain unresolved.'}
            receipt_path = out / 'receipt.json'; receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            receipts.append(receipt); packaged.extend([preview, receipt_path])
            print(json.dumps({'candidate': candidate, 'variant': variant, 'clip_seconds': 4, 'cycle_seconds': 16, 'preview': str(preview)}))
    if mix.digest(runtime) != runtime_before:
        raise ValueError('Runtime manifest changed during audition build.')
    (args.output_dir / 'variant-receipts.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(args.output_dir / 'counted-phase-variants.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in packaged:
            archive.write(path, path.relative_to(args.output_dir))


if __name__ == '__main__':
    main()
