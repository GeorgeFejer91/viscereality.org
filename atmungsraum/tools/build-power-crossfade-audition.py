"""Audition four-second voice/breath elements with equal mean-square power.

Outputs stay local, outside runtime assets. RMS equality is measured after gain
envelopes, over the entire four-second block including speech pauses. It is not
a claim of equal perceived loudness. Requires NumPy and installed FFmpeg.
"""
import argparse
import hashlib
import importlib.util
import json
import wave
from pathlib import Path

import numpy as np

def load_tool(filename):
    spec = importlib.util.spec_from_file_location(filename.replace('-', '_'), Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pcm = load_tool('assemble-counted-voice-auditions.py').pcm
cue_tools = load_tool('build-soft-box-cues.py')
retime, trim = cue_tools.retime, cue_tools.trim

RATE, FRAMES = 44100, 4 * 44100
VOICE_DB = np.array([0., -4., -8., -12.])
BREATH_DB = VOICE_DB[::-1].copy()
ORDER = ['inhale', 'hold_in', 'exhale', 'hold_out']


def rms(samples):
    return float(np.sqrt(np.mean(samples ** 2)))


def db(level):
    return float(20 * np.log10(max(level, 1e-15)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    if np.max(np.abs(data)) >= 1:
        raise ValueError('Refusing to clip a layer or its mix.')
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(RATE)
        wav.writeframes(np.rint(data * 32767).astype('<i2').tobytes())
    return digest(path)


def faded_beats(data, targets_db, smooth=False):
    if len(data) != FRAMES:
        raise ValueError('Source must be exactly four seconds.')
    levels = np.array([rms(data[i * RATE:(i + 1) * RATE]) for i in range(4)])
    if np.any(levels <= 1e-8):
        raise ValueError('A required speech/breath beat is silent.')
    gains = 10 ** (targets_db / 20) / levels
    envelope = np.repeat(gains, RATE)
    # Breath is continuous: smooth the four gain changes over 100 ms. Speech
    # already has quiet gaps and faded word boundaries at the second markers.
    if smooth:
        half = round(.05 * RATE)
        for i in range(1, 4):
            left, right = i * RATE - half, i * RATE + half
            blend = (1 - np.cos(np.linspace(0, np.pi, right - left))) / 2
            envelope[left:right] = np.exp(np.log(gains[i - 1]) * (1 - blend) + np.log(gains[i]) * blend)
        fade = round(.01 * RATE)
        envelope[:fade] *= np.linspace(0, 1, fade)
        envelope[-fade:] *= np.linspace(1, 0, fade)
    shaped = data * envelope
    shaped /= rms(shaped)
    measured = [db(rms(shaped[i * RATE:(i + 1) * RATE])) for i in range(4)]
    if (smooth and not np.all(np.diff(measured) > 0)) or (not smooth and not np.all(np.diff(measured) < 0)):
        raise ValueError('The measured gain progression is not monotonic.')
    return shaped, measured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voices-dir', type=Path, required=True)
    parser.add_argument('--sounds-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the current browser sample set throughout this audition.
    runtime_manifest = Path(__file__).resolve().parents[1] / 'assets/audio/box-breathing/de/soft-v2/manifest.json'
    runtime_before = digest(runtime_manifest)
    breaths, provenance = {}, {}
    for phase, filename in [('inhale', 'inhale.wav'), ('exhale', 'Exhale.wav')]:
        source = args.sounds_dir / filename
        original = pcm(source)
        fitted, tempo = retime(original, 4)
        breaths[phase], levels = faded_beats(fitted, BREATH_DB, smooth=True)
        provenance[phase] = {'source_file': filename, 'source_sha256': digest(source),
                             'source_seconds': len(original) / RATE, 'tempo_factor': tempo,
                             'pitch_preserved': True, 'envelope_targets_db': BREATH_DB.tolist(),
                             'measured_unit_rms_beat_db': levels, 'gain_transition_seconds': .1}

    prepared = []
    for hold_record in sorted(args.output_dir.glob('*-halten-generated.json')):
        candidate = hold_record.name.removesuffix('-halten-generated.json')
        hold_take = json.loads(hold_record.read_text(encoding='utf-8-sig'))
        if hold_take['clip']['word_fidelity']['word_edits'] != 0:
            raise ValueError('Standalone Halten did not pass the source speech check.')
        source_hold = Path(hold_take['path'])
        hold_original = args.output_dir / f'{candidate}-halten-source.mp3'
        hold_original.write_bytes(source_hold.read_bytes())
        hold_word = trim(pcm(hold_original))
        if len(hold_word) > 1.6 * RATE:
            raise ValueError('Halten is too long for a short prompt and quiet hold.')
        fade = min(220, len(hold_word) // 2)
        hold_word[:fade] *= np.linspace(0, 1, fade)
        hold_word[-fade:] *= np.linspace(1, 0, fade)
        pairs, voice_sources = {}, {}
        for phase in ['inhale', 'exhale']:
            source = args.voices_dir / f'{candidate}-blocks' / f'{phase}.wav'
            voice, measured = faded_beats(pcm(source), VOICE_DB)
            breath = breaths[phase].copy()
            pairs[phase] = {'voice': voice, 'breath': breath, 'mixed': voice + breath}
            voice_sources[phase] = {'source_sha256': digest(source), 'envelope_targets_db': VOICE_DB.tolist(),
                                    'measured_unit_rms_beat_db': measured, 'word_onsets_frames': [0, RATE, 2 * RATE, 3 * RATE]}
        # Keep Halten's sample peak consistent with the initial spoken cue.
        hold_word *= np.max(np.abs(pairs['inhale']['voice'][:RATE])) / np.max(np.abs(hold_word))
        hold = np.zeros(FRAMES); hold[:len(hold_word)] = hold_word
        prepared.append((candidate, pairs, hold, voice_sources, hold_take, len(hold_word), hold_original))
    if not prepared:
        raise ValueError('No verified standalone Halten source takes.')

    # One shared gain across every candidate, layer, and mix preserves equality
    # and permits a comparison at matched component RMS without a limiter.
    peak = max(max(float(np.max(np.abs(layer))) for pair in pairs.values() for layer in pair.values())
               for _, pairs, _, _, _, _, _ in prepared)
    peak = max(peak, *(float(np.max(np.abs(hold))) for _, _, hold, _, _, _, _ in prepared))
    gain = min(10 ** (-26 / 20), 10 ** (-3 / 20) / peak)
    receipts = []
    for candidate, pairs, hold, voice_sources, hold_take, hold_frames, hold_original in prepared:
        out = args.output_dir / candidate; out.mkdir(exist_ok=True)
        blocks, measurements = {}, {}
        for phase, pair in pairs.items():
            scaled = {name: layer * gain for name, layer in pair.items()}
            for name, data in scaled.items():
                filename = f'{phase}.wav' if name == 'mixed' else f'{phase}-{name}.wav'
                file_hash = write(out / filename, data)
                if name == 'mixed': blocks[phase] = {'file': filename, 'frames': FRAMES, 'sha256': file_hash}
            power_voice = float(np.mean(scaled['voice'] ** 2)); power_breath = float(np.mean(scaled['breath'] ** 2))
            if abs(power_voice / power_breath - 1) > 1e-10:
                raise ValueError('Component mean-square powers differ.')
            measurements[phase] = {'voice_power': power_voice, 'breath_power': power_breath,
                                   'power_ratio_db': float(10 * np.log10(power_voice / power_breath)),
                                   'component_rms_dbfs': db(gain), 'mix_peak_dbfs': db(float(np.max(np.abs(scaled['mixed'])))),
                                   'voice_beat_rms_dbfs': [db(rms(scaled['voice'][i * RATE:(i + 1) * RATE])) for i in range(4)],
                                   'breath_beat_rms_dbfs': [db(rms(scaled['breath'][i * RATE:(i + 1) * RATE])) for i in range(4)]}
        for phase in ['hold_in', 'hold_out']:
            blocks[phase] = {'file': f'{phase}.wav', 'frames': FRAMES, 'text': 'Halten.',
                             'sha256': write(out / f'{phase}.wav', hold * gain)}
        montage = np.concatenate([pairs[phase]['mixed'] if phase in pairs else hold for phase in ORDER]) * gain
        preview = args.output_dir / f'{candidate}-power-crossfade-16s.wav'
        preview_hash = write(preview, montage)
        receipt = {'candidate': candidate, 'sample_rate': RATE, 'channels': 1, 'format': 'pcm_s16le',
                   'phase_frames': FRAMES, 'phase_order': ORDER, 'blocks': blocks,
                   'preview_file': preview.name, 'preview_frames': len(montage), 'preview_sha256': preview_hash,
                   'normalization_scope': 'Full four-second block after envelopes, including speech pauses; both layers have identical mean-square power before summation',
                   'shared_component_rms_dbfs': db(gain), 'normalization_is_perceived_loudness': False,
                   'breaths': provenance, 'voice_sources': voice_sources, 'measurements': measurements,
                   'hold_speech_frames': hold_frames, 'hold_source_sha256': digest(hold_original),
                   'hold_source_word_edits': 0, 'hold_clip_id': hold_take['clip']['clip_id'],
                   'runtime_manifest_unchanged': digest(runtime_manifest) == runtime_before,
                   'use': 'Local audition only. Existing OmniVoice/source/speaker licence constraints still apply.'}
        (out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        receipts.append(receipt)
        print(json.dumps({'candidate': candidate, 'duration_seconds': 16, 'component_rms_dbfs': db(gain),
                          'power_ratios_db': {phase: data['power_ratio_db'] for phase, data in measurements.items()},
                          'preview': str(preview)}))
    if digest(runtime_manifest) != runtime_before:
        raise ValueError('Runtime manifest changed during a local audition build.')
    (args.output_dir / 'mix-receipts.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
