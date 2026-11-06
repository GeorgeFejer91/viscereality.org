"""Build numbered, independent silent cue assets for each saved UI scenario.

TXT files are draft scripts. No speech is synthesized and existing recordings
are preserved. Run with --check to verify generated files without writing.
"""
import argparse
import csv
import hashlib
import io
import json
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def csv_bytes(columns, rows):
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(columns)
    writer.writerows(rows)
    return output.getvalue().encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    library_dir = ROOT / 'assets/audio/box-breathing/de/soft-v2'
    library_bytes = (library_dir / 'manifest.json').read_bytes()
    library = json.loads(library_bytes)
    library_hash = hashlib.sha256(library_bytes).hexdigest()
    output = ROOT / 'assets/audio/scenarios'
    totals = []
    pcm_cache = {}

    def put(path, data, audio=False):
        if path.exists() and path.read_bytes() == data:
            return
        if args.check:
            raise ValueError(f'Missing/stale generated file: {path.relative_to(ROOT)}')
        if audio and path.exists():
            raise ValueError(f'Existing WAV differs; preserving it: {path.relative_to(ROOT)}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def silence(frames, rate):
        key = frames, rate
        if key not in pcm_cache:
            data = io.BytesIO()
            with wave.open(data, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(bytes(frames * 2))
            pcm_cache[key] = data.getvalue()
        return pcm_cache[key]

    for path in sorted((ROOT / 'audio/scenarios').glob('*.json')):
        plan = json.loads(path.read_bytes())
        scenario = plan['scenario_id']
        if scenario != path.stem or not scenario.replace('-', '').isalnum():
            raise ValueError('Unsafe scenario ID.')
        if plan['library_manifest_sha256'] != library_hash:
            raise ValueError(f'Stale source library: {scenario}')
        folder = output / scenario
        rate = plan['sample_rate']
        unique, registry, timeline, frame = {}, {}, [], 0
        for order, event in enumerate(plan['events'], 1):
            if event['offset_frames'] != frame:
                raise ValueError(f'Non-contiguous scenario: {scenario}')
            frame += event['duration_frames']
            key = event['id'], event['asset'], event['duration_frames']
            if key not in unique:
                cue_order = len(unique) + 1
                seconds = event['offset_frames'] // rate
                role = event['id'].replace('_', '-')
                if event['asset'] in ['inhale_breath', 'exhale_breath']:
                    role += '-with-breath'
                stem = f'{cue_order:04}_{seconds // 60:02}m{seconds % 60:02}s_{role}'
                block = library['blocks'].get(event['asset'], {})
                source_txt = library_dir / Path(block['file']).with_suffix('.txt') if block else None
                own_txt = folder / f'{stem}.txt'
                if own_txt.exists():
                    script = own_txt.read_text(encoding='utf-8').rstrip('\n')
                elif source_txt and source_txt.exists():
                    script = source_txt.read_text(encoding='utf-8').rstrip('\n')
                else:
                    script = block.get('text', '')
                data = silence(event['duration_frames'], rate)
                meta = {'id': event['asset'] or 'recovery', 'file': stem + '.wav',
                    'frames': event['duration_frames'], 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest(), 'text': script,
                    'spoken_text': '', 'placeholder': True, 'audio_status': 'silent-placeholder',
                    'script_status': 'awaiting-wording-approval' if script else 'no-speech',
                    'sample_rate': rate, 'scenario_id': scenario, 'cue_order': cue_order,
                    'phase_id': event['id'], 'first_offset_frames': event['offset_frames'],
                    'first_start_seconds': event['offset_frames'] / rate,
                    'duration_seconds': event['duration_frames'] / rate,
                    'transcript_file': stem + '.txt', 'occurrence_offsets_frames': []}
                if 'measurement' in block:
                    meta['measurement'] = block['measurement']
                put(folder / meta['file'], data, audio=True)
                put(own_txt, (script + ('\n' if script else '')).encode('utf-8'))
                unique[key] = meta
                if event['asset'] is not None:
                    registry[event['asset']] = meta
            meta = unique[key]
            meta['occurrence_offsets_frames'].append(event['offset_frames'])
            timeline.append({**event, 'order': order, 'file': meta['file']})
        if frame != plan['total_frames']:
            raise ValueError(f'Incorrect total: {scenario}')
        for meta in unique.values():
            meta['occurrence_count'] = len(meta['occurrence_offsets_frames'])
            put(folder / Path(meta['file']).with_suffix('.json'), encoded(meta))
        plan['audio_assets_base_directory'] = f'../../assets/audio/scenarios/{scenario}/'
        plan['audio_assets'] = registry
        put(path, encoded(plan))
        manifest = {'scenario_id': scenario, 'sample_rate': rate, 'duration_seconds': frame / rate,
            'placeholder': True, 'source_scenario_sha256': hashlib.sha256(encoded(plan)).hexdigest(),
            'assets': list(unique.values()), 'events': timeline}
        put(folder / '0000_manifest.json', encoded(manifest))
        put(folder / '0000_assets.csv', csv_bytes(
            ['cue_order', 'file', 'phase_id', 'first_start_seconds', 'duration_seconds', 'occurrence_count', 'audio_status', 'script_status', 'draft_script'],
            [[m[k] for k in ['cue_order', 'file', 'phase_id', 'first_start_seconds', 'duration_seconds', 'occurrence_count', 'audio_status', 'script_status', 'text']] for m in unique.values()]))
        put(folder / '0000_timeline.csv', csv_bytes(
            ['order', 'file', 'phase_id', 'start_seconds', 'duration_seconds', 'cycle'],
            [[e['order'], e['file'], e['id'], e['offset_frames'] / rate, e['duration_frames'] / rate, e['cycle']] for e in timeline]))
        playlist = ['#EXTM3U']
        for event in timeline:
            playlist += [f'#EXTINF:{event["duration_frames"] / rate:g},{event["file"]}', event['file']]
        put(folder / '0000_playlist.m3u8', ('\n'.join(playlist) + '\n').encode('utf-8'))
        put(folder / '0000_README.txt', (
            f'{scenario}: {frame / rate:g} seconds; {len(unique)} unique cues.\n\n'
            'All WAVs are intentional silent placeholders for UI testing.\n'
            'TXT files contain draft intended wording, not speech in these silent WAVs.\n'
            'Final narration requires explicit wording approval before synthesis.\n'
            'Sort Name ascending: four-digit cue numbers follow first appearance.\n'
            'The filename timestamp is the first onset; repeated uses are in the timeline CSV.\n'
            'Open 0000_playlist.m3u8 for the complete ordered scenario.\n'
            'Each WAV has a matching JSON and TXT; every scenario owns independent files.\n'
        ).encode('utf-8'))
        totals.append([scenario, frame / rate, len(unique), len(timeline)])
    put(output / '0000_INDEX.csv', csv_bytes(['scenario', 'duration_seconds', 'unique_cues', 'events'], totals))
    print(json.dumps({'scenarios': len(totals), 'unique_placeholder_wavs': sum(row[2] for row in totals),
        'playback_events': sum(row[3] for row in totals), 'check_only': args.check}))


if __name__ == '__main__':
    main()
