"""Check numbered scenario cue sidecars against the authored playback events."""
import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScenarioAssetTests(unittest.TestCase):
    def test_each_numbered_cue_has_matching_csv_and_txt_for_every_occurrence(self):
        scenarios = ROOT / 'assets/audio/scenarios'
        self.assertTrue(all((scenarios / name).is_dir() for name in
                            ['haupt-15', 'probe-anleitung', 'probe-5', 'probe-10', 'probe-15']))
        count = 0
        for path in sorted((ROOT / 'audio/scenarios').glob('*.json')):
            plan = json.loads(path.read_bytes())
            folder = scenarios / plan['scenario_id']
            manifest = json.loads((folder / '0000_manifest.json').read_bytes())
            previous_onset = -1
            for cue_order, asset in enumerate(manifest['assets'], 1):
                with self.subTest(scenario=path.stem, file=asset['file']):
                    stem = Path(asset['file']).stem
                    self.assertEqual(int(stem.split('_')[0]), cue_order)
                    events = [(order, event) for order, event in enumerate(plan['events'], 1)
                              if event['id'] == asset['phase_id']
                              and (event['asset'] or 'recovery') == asset['id']]
                    self.assertGreater(events[0][1]['offset_frames'], previous_onset)
                    previous_onset = events[0][1]['offset_frames']
                    seconds = previous_onset // plan['sample_rate']
                    self.assertEqual(stem.split('_')[1], f'{seconds // 60:02}m{seconds % 60:02}s')
                    self.assertEqual((folder / f'{stem}.txt').read_text(encoding='utf-8'),
                                     asset['text'] + ('\n' if asset['text'] else ''))
                    with (folder / f'{stem}.csv').open(encoding='utf-8', newline='') as source:
                        rows = list(csv.DictReader(source))
                    self.assertEqual(len(rows), len(events))
                    for row, (order, event) in zip(rows, events):
                        self.assertEqual(int(row['order']), order)
                        self.assertEqual(row['asset_file'], asset['file'])
                        self.assertEqual(row['event_id'], event['id'])
                        self.assertEqual(row['asset_id'], event['asset'] or '')
                        self.assertEqual(int(row['offset_frames']), event['offset_frames'])
                        self.assertEqual(int(row['duration_frames']), event['duration_frames'])
                        self.assertEqual(float(row['start_seconds']), event['offset_frames'] / plan['sample_rate'])
                        self.assertEqual(float(row['duration_seconds']), event['duration_frames'] / plan['sample_rate'])
                        self.assertEqual(float(row['end_seconds']),
                                         (event['offset_frames'] + event['duration_frames']) / plan['sample_rate'])
                        self.assertEqual(row['cycle'], str(event['cycle'] or ''))
                        self.assertEqual(row['breathing_block'], str(event['breathing_block'] or ''))
                        self.assertEqual(row['transcript_de'], asset['text'])
                        self.assertEqual(row['audio_status'], asset['audio_status'])
                        self.assertEqual(row['script_status'], asset['script_status'])
                    count += 1
        self.assertEqual(count, 53)


if __name__ == '__main__':
    unittest.main()
