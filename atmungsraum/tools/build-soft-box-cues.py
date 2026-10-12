"""Build reusable German voice/count samples and every-third-cycle breath mixes.

Requires installed numpy and FFmpeg. Credentials are read only from --key-file.
Successful API responses are cached locally before processing; retries reuse them.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RATE, PHASE_FRAMES = 44100, 176400
VOICE = 'rt9KyyQmwqa1uhHGJkMa'
SETTINGS = {'stability': 0.5, 'similarity_boost': 0.65}
TAGS = '[softly] [calm] [slowly] '
TEXTS = {
    'instruction': 'Willkommen im Atmungsraum. Schön, dass du dir diesen Moment nimmst. '
        'Mach es dir bequem und lass die Schultern locker. Wir begleiten dich durch '
        'einen ruhigen Atemrhythmus. Auf Ein atmest du sanft ein. Auf Aus lässt du die '
        'Luft wieder ausströmen. Dazwischen ruht der Atem für einen Moment. Wir zählen '
        'jede Phase bis vier. Atme nur so weit, wie es sich angenehm anfühlt. Du kannst '
        'jederzeit normal weiteratmen oder die Übung beenden. Wir beginnen gemeinsam.',
    'ending': 'Unsere Atemrunde ist jetzt zu Ende. Lass deinen Atem ganz frei fließen. '
        'Nimm dir noch einen Moment, bevor du weitergehst.',
    'rest_begin': 'Nimm dir jetzt eine kleine Pause. Lass deinen Atem in deinem eigenen Rhythmus fließen.',
    'rest_resume': 'Wir beginnen wieder gemeinsam.',
    'ein': 'Einatmen.', 'aus': 'Ausatmen.', 'pause': 'Und halten.',
    'zwei': 'Zwei.', 'drei': 'Drei.', 'vier': 'Vier.',
}

def ffmpeg(data=None, args=()):
    result = subprocess.run(['ffmpeg', '-v', 'error', *args], input=data, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype='<i2').astype(np.float64) / 32768

def trim(samples):
    active = np.flatnonzero(np.abs(samples) > 10 ** (-50 / 20))
    if not len(active): raise ValueError('No speech detected.')
    return samples[max(0, int(active[0]) - 220):min(len(samples), int(active[-1]) + 2206)].copy()

def finish(samples, peak_db=-8):
    samples = samples.copy()
    peak = np.max(np.abs(samples))
    if peak: samples *= 10 ** (peak_db / 20) / peak
    fade = min(220, len(samples) // 2)
    if fade:
        samples[:fade] *= np.linspace(0, 1, fade)
        samples[-fade:] *= np.linspace(1, 0, fade)
    return samples

def retime(samples, seconds):
    factor = len(samples) / (RATE * seconds)
    pcm = np.clip(np.rint(samples * 32767), -32768, 32767).astype('<i2').tobytes()
    result = ffmpeg(pcm, ['-f', 's16le', '-ar', str(RATE), '-ac', '1', '-i', 'pipe:0',
        '-af', f'atempo={factor:.12f}', '-f', 's16le', 'pipe:1'])
    output = np.zeros(round(seconds * RATE))
    output[:min(len(result), len(output))] = result[:len(output)]
    return output, factor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--sounds-dir', type=Path, required=True)
    args = parser.parse_args()
    raw = ROOT / 'reference-audio/soft-box-v2'
    out = ROOT / 'assets/audio/box-breathing/de/soft-v2'
    raw.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    key = args.key_file.read_text(encoding='utf-8-sig').strip().strip('"').strip("'")
    manifest = {'id': 'box-breathing-4-4-4-4-de-v2', 'version': 2, 'language': 'de',
        'sample_rate': RATE, 'block_frames': PHASE_FRAMES,
        'phase_order': ['inhale', 'hold_in', 'exhale', 'hold_out'],
        'supported_minutes': [5, 10, 15, 20], 'cycles_per_block': 18,
        'rest_frames': 20 * RATE, 'overlay_every_cycles': 3,
        'remainder_policy': 'quiet-recovery-after-ending', 'generated_on': '2026-10-12',
        'voice': {'name': 'British Breathing Instructor', 'id': VOICE}, 'provider': 'ElevenLabs',
        'model_id': 'eleven_v4', 'voice_settings': SETTINGS, 'delivery_tags': TAGS.strip(),
        'narration_tempo_factor': 0.8,
        'format': 'pcm_s16le_mono_wav', 'blocks': {},
        'review_status': 'Generated; human listening and technical verification pending.'}
    speech = {}
    for name, text in TEXTS.items():
        payload = {'text': TAGS + text, 'model_id': manifest['model_id'], 'voice_settings': SETTINGS,
            'language_code': 'de', 'seed': 42}
        source, request_file = raw / f'{name}.mp3', raw / f'{name}.request.json'
        if source.exists():
            if not request_file.exists() or json.loads(request_file.read_text(encoding='utf-8')) != payload:
                raise ValueError(f'Cached request changed: {name}; preserve it before regenerating.')
        else:
            request = urllib.request.Request(f'https://api.elevenlabs.io/v1/text-to-speech/{VOICE}?output_format=mp3_44100_128',
                data=json.dumps(payload).encode(), headers={'xi-api-key': key, 'Content-Type': 'application/json'}, method='POST')
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    audio = response.read()
                    if response.headers.get_content_type() != 'audio/mpeg' or len(audio) < 1000:
                        raise ValueError('Unexpected speech response.')
            except urllib.error.HTTPError as error:
                sys.exit(f'ElevenLabs returned HTTP {error.code} for {name}; response content suppressed.')
            except Exception as error:
                sys.exit(f'Request failed for {name}: {type(error).__name__}; check history before retrying.')
            source.write_bytes(audio)
            request_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        samples = trim(ffmpeg(args=['-xerror', '-i', str(source), '-ac', '1', '-ar', str(RATE), '-f', 's16le', 'pipe:1']))
        if name in ['instruction', 'ending', 'rest_begin', 'rest_resume']:
            samples, _ = retime(samples, len(samples) / RATE / 0.8)
        speech[name] = finish(samples)
        print(json.dumps({'speech': name, 'trimmed_seconds': len(samples) / RATE}), flush=True)

    def write(id, samples, text, processing):
        pcm = np.clip(np.rint(samples * 32767), -32768, 32767).astype('<i2')
        target = out / f'{id}.wav'
        with wave.open(str(target), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(RATE); wav.writeframes(pcm.tobytes())
        data = target.read_bytes()
        manifest['blocks'][id] = {'file': target.name, 'text': text, 'frames': len(pcm),
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'processing': processing}

    for name in ['ein', 'aus', 'pause', 'zwei', 'drei', 'vier']:
        limit = 1.6 if name == 'pause' else .94
        if len(speech[name]) > limit * RATE:
            factor = len(speech[name]) / (limit * RATE)
            if factor > 1.5: raise ValueError(f'{name} cannot fit a readable one-second count beat.')
            speech[name], _ = retime(speech[name], limit)
            speech[name] = finish(speech[name])

    for id, word in [('inhale', 'ein'), ('hold_in', 'pause'), ('exhale', 'aus'), ('hold_out', 'pause')]:
        samples = np.zeros(PHASE_FRAMES)
        words = [word] if id in ['hold_in', 'hold_out'] else [word, 'zwei', 'drei', 'vier']
        for beat, name in enumerate(words):
            offset = beat * RATE
            samples[offset:offset + len(speech[name])] = speech[name]
        write(id, samples, ' '.join(TEXTS[name] for name in words),
            {'word_onsets_frames': [beat * RATE for beat in range(len(words))], 'sample_peak_dbfs': -8})

    for id in ['instruction', 'ending']:
        frames = math.ceil((len(speech[id]) + .4 * RATE) / PHASE_FRAMES) * PHASE_FRAMES
        samples = np.zeros(frames); samples[:len(speech[id])] = speech[id]
        write(id, samples, TEXTS[id], {'speech_frames': len(speech[id]), 'sample_peak_dbfs': -8})
    rest = np.zeros(20 * RATE)
    if len(speech['rest_begin']) > 15 * RATE or len(speech['rest_resume']) > 4 * RATE:
        raise ValueError('Rest narration would overlap the resumption or next phase.')
    rest[:len(speech['rest_begin'])] = speech['rest_begin']
    rest[16 * RATE:16 * RATE + len(speech['rest_resume'])] = speech['rest_resume']
    write('rest', rest, TEXTS['rest_begin'] + ' ' + TEXTS['rest_resume'], {'resume_onset_frame': 16 * RATE})

    for id, name in [('inhale_breath', 'inhale.wav'), ('exhale_breath', 'Exhale.wav')]:
        original = args.sounds_dir / name
        data = original.read_bytes()
        (raw / name).write_bytes(data)
        samples = ffmpeg(args=['-xerror', '-i', str(original), '-ac', '1', '-ar', str(RATE), '-f', 's16le', 'pipe:1'])
        source_duration = len(samples) / RATE
        effect, factor = retime(samples, 4)
        effect = finish(effect, -20)
        phase = id.split('_')[0]
        with wave.open(str(out / f'{phase}.wav'), 'rb') as wav:
            voice = np.frombuffer(wav.readframes(PHASE_FRAMES), dtype='<i2').astype(np.float64) / 32768
        mixed = voice + effect
        if np.max(np.abs(mixed)) >= .95: raise ValueError('Breath mix lacks headroom.')
        write(id, mixed, manifest['blocks'][phase]['text'], {'voice_block': phase,
            'breath_peak_dbfs': -20, 'source_file': name, 'source': 'User-supplied Proton Drive recording',
            'source_sha256': hashlib.sha256(data).hexdigest(), 'source_duration_seconds': source_duration,
            'breath_tempo_factor': factor, 'target_duration_seconds': 4})
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'manifest': str(out / 'manifest.json'), 'intro_seconds': manifest['blocks']['instruction']['frames'] / RATE,
        'ending_seconds': manifest['blocks']['ending']['frames'] / RATE, 'rest_seconds': 20}), flush=True)

if __name__ == '__main__':
    main()
