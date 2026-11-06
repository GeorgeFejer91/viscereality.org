"""Generate the shared two-breath cue through Secret Tunnel's native voice CLI.

Uses the saved existing-instructor clone; no stretching or per-round copies.
Re-running reuses matching cached responses; differing published assets are kept.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RATE = 44100
TEXTS = {
    'prepare_intro': 'Bevor wir beginnen, machen wir eine kurze Atemkalibrierung.',
    'prepare_breaths': 'Atme dafür gleich zweimal ruhig und tief ein und wieder aus. '
        'Nutze deinen natürlichen Atemraum, ohne dich anzustrengen.',
    'prepare_comfort': 'Atme nur so tief, wie es angenehm ist, und halte die Luft nicht an. Beginne jetzt.',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--voice-app', type=Path,
        default=Path(os.environ['LOCALAPPDATA']) / 'Secret Tunnel 6/secret-tunnel-v6.exe')
    parser.add_argument('--voice-name', default='Atmungsraum - Existing Instructor')
    parser.add_argument('--replace', action='store_true', help='Replace a reviewed earlier generated cue.')
    args = parser.parse_args()
    folder = ROOT / 'assets/audio/box-breathing/de/soft-v2'
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    cache = ROOT / 'reference-audio/calibration-secret-tunnel-de'
    cache.mkdir(parents=True, exist_ok=True)
    def native(request):
        result = subprocess.run([str(args.voice_app), '--voice-cli'],
            input=json.dumps(request, ensure_ascii=False), text=True, encoding='utf-8',
            capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW, timeout=600)
        if result.returncode:
            raise RuntimeError(f'Native voice CLI exited {result.returncode}.')
        response = json.loads(result.stdout)
        if not response.get('ok'):
            raise RuntimeError(str(response.get('error', 'Native generation failed.')))
        return response['result']

    profiles = [p for p in native({'operation': 'list'})['profiles'] if p['name'] == args.voice_name]
    if len(profiles) != 1:
        raise ValueError('Expected one saved profile with the requested name.')
    speech, source_segments = {}, []
    for name, text in TEXTS.items():
        payload = {'voiceId': profiles[0]['id'], 'text': text, 'language': 'German',
                   'mode': 'slow', 'speed': 0.85, 'guidanceScale': 2.0,
                   'seed': 611060 + len(speech), 'whisperCheck': True}
        request_file, response_file = cache / f'{name}.request.json', cache / f'{name}.response.json'
        if request_file.exists():
            if json.loads(request_file.read_text(encoding='utf-8')) != payload:
                raise ValueError(f'Cached request differs: {name}. Preserve it before regenerating.')
            if not response_file.exists():
                raise ValueError(f'Uncertain prior request: {name}. Inspect provider history before retrying.')
        else:
            request_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(f'Generating {name} with Secret Tunnel; native word check enabled.', flush=True)
            response = native({'operation': 'generate', 'input': payload})
            response_file.write_text(json.dumps(response, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        response = json.loads(response_file.read_text(encoding='utf-8'))
        source = cache / f'{name}.mp3'
        if not source.exists():
            shutil.copyfile(response['path'], source)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if response.get('source_mp3_sha256', source_hash) != source_hash:
            raise ValueError(f'Cached MP3 hash changed: {name}.')
        if 'source_mp3_sha256' not in response:
            response['source_mp3_sha256'] = source_hash
            response_file.write_text(json.dumps(response, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        fidelity = response['clip'].get('word_fidelity', {})
        if fidelity.get('status') != 'scored' or fidelity.get('word_edits') != 0:
            raise ValueError(f'Native word check did not pass: {name}.')
        pcm = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(source), '-ac', '1',
            '-ar', str(RATE), '-f', 's16le', 'pipe:1'], capture_output=True, check=True).stdout
        samples = np.frombuffer(pcm, dtype='<i2').astype(np.float64)
        active = np.flatnonzero(np.abs(samples) > 32768 * 10 ** (-50 / 20))
        if not len(active):
            raise ValueError(f'No speech: {name}')
        samples = samples[max(0, active[0] - 220):active[-1] + 2206].copy()
        samples *= 32767 * 10 ** (-8 / 20) / np.max(np.abs(samples))
        samples[:220] *= np.linspace(0, 1, 220)
        samples[-220:] *= np.linspace(1, 0, 220)
        speech[name] = np.rint(samples).astype('<i2')
        source_segments.append({'id': name, 'text': text,
            'source_mp3_sha256': source_hash,
            'speech_frames': len(samples), 'generation_parameters': response['clip']['generation_parameters'],
            'word_fidelity': fidelity})
        print(json.dumps({'segment': name, 'speech_seconds': len(samples) / RATE}), flush=True)

    prepare = np.concatenate([speech['prepare_intro'], np.zeros(round(.4 * RATE), dtype='<i2'),
        speech['prepare_breaths'], np.zeros(round(.4 * RATE), dtype='<i2'), speech['prepare_comfort']])
    range_start = (math.ceil(len(prepare) / RATE) + 1) * RATE
    samples = np.zeros(range_start, dtype='<i2')
    samples[:len(prepare)] = prepare

    def write_cue(name, pcm, text, metadata):
        wav_bytes = io.BytesIO()
        with wave.open(wav_bytes, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(RATE); wav.writeframes(pcm.tobytes())
        data = wav_bytes.getvalue()
        target = folder / f'{name}.wav'
        if target.exists() and target.read_bytes() != data and not args.replace:
            raise ValueError(f'Existing {name} differs; review before using --replace.')
        target.write_bytes(data)
        block = {'file': target.name, 'text': text, 'frames': len(pcm), 'bytes': len(data),
                 'sha256': hashlib.sha256(data).hexdigest(), **metadata}
        manifest['blocks'][name] = block
        (folder / f'{name}.json').write_text(json.dumps({'sample_rate': RATE, **block}, ensure_ascii=False,
            indent=2) + '\n', encoding='utf-8', newline='\n')

    write_cue('instruction_1', samples, ' '.join(TEXTS.values()), {
        'provider': 'Secret Tunnel native OmniVoice', 'voice': args.voice_name,
        'model_id': response['clip']['model'],
        'processing': {'sample_peak_dbfs': -8, 'speech_time_stretch': False,
            'source_segments': source_segments,
            'source': 'Respyra comfortable natural-range instructions; two-breath adaptation',
            'source_url': 'https://github.com/embodied-computation-group/respyra/blob/dev/respyra/core/runner.py#L267-L408',
            'generated_on': '2026-11-06', 'listening_status': 'Human delivery acceptance pending'}})
    write_cue('calibration', np.zeros(15 * RATE, dtype='<i2'), '', {
        'role': 'Silent breathing measurement window after Instruction 1',
        'measurement': {'signal_source': 'h10-accelerometer-pending', 'value': None, 'unit': None,
            'method': None, 'status': 'not-integrated'},
        'processing': {'range_start_frame': 0, 'range_duration_frames': 15 * RATE}})
    with wave.open(str(folder / manifest['blocks']['ending']['file']), 'rb') as wav:
        ending = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
    # Seven seconds of settling plus the unchanged 12-second speech closes the
    # 900-second main after fifty complete 16-second cycles; no cycle is cut.
    write_cue('outro', np.concatenate([np.zeros(7 * RATE, dtype='<i2'), ending]),
        manifest['blocks']['ending']['text'], {'processing': {'speech_start_frame': 7 * RATE,
            'source_asset': 'ending', 'source_sha256': manifest['blocks']['ending']['sha256'],
            'source_samples_unchanged': True}})
    manifest['main_asset_classes'] = {'instruction_1': ['instruction_1'], 'calibration': ['calibration'],
        'instruction_2': ['instruction'], 'inhale': ['inhale', 'inhale_breath'], 'hold_in': ['hold_in'],
        'exhale': ['exhale', 'exhale_breath'], 'hold_out': ['hold_out'], 'outro': ['outro']}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    with (folder / 'calibration.csv').open('w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(['segment', 'start_seconds', 'duration_seconds', 'offset_frames', 'duration_frames', 'transcript_de'])
        writer.writerow(['calibration', 0, 15, 0, 15 * RATE, ''])
    with (folder / 'main-asset-classes.csv').open('w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(['class_id', 'asset_id', 'asset_file', 'duration_seconds', 'transcript_de'])
        for class_id, assets in manifest['main_asset_classes'].items():
            for asset in assets:
                block = manifest['blocks'][asset]
                writer.writerow([class_id, asset, block['file'], block['frames'] / RATE, block['text']])
    print(json.dumps({'instruction_1_seconds': range_start / RATE, 'calibration_seconds': 15,
        'outro_seconds': 19, 'main_asset_classes': list(manifest['main_asset_classes'])}))


if __name__ == '__main__':
    main()
