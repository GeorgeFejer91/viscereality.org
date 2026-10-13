"""Build local counted/fading cues from the website's ElevenLabs voice.

Consumes cached speech only; never changes runtime assets or calls an API.
Standalone Halten must be generated with the website manifest's settings first.
"""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mix', Path(__file__).with_name('build-power-crossfade-audition.py'))
mix = importlib.util.module_from_spec(spec); spec.loader.exec_module(mix)
fragment = mix.load_tool('assemble-counted-voice-auditions.py').fragment
RATE, FRAMES = mix.RATE, mix.FRAMES
PHASES = [('inhale', 'Einatmen'), ('hold_in', 'Halten'), ('exhale', 'Ausatmen'), ('hold_out', 'Halten')]


def label(path):
    data = mix.pcm(path)
    word, tempo = fragment(data, 0, len(data) / RATE, 3.75, leading_padding=.03)
    return word, tempo


def counted_phrase(path, first_word):
    """Keep one take's speech at native speed; change only pauses and gain."""
    source = mix.pcm(path)
    result = json.loads(path.with_suffix('.response.json').read_text(encoding='utf-8'))
    alignment = result.get('normalized_alignment') or result['alignment']
    text = ''.join(alignment['characters'])
    words = []
    for word in [first_word, 'zwei', 'drei', 'vier']:
        match = re.search(r'\b' + re.escape(word) + r'\b', text, flags=re.I)
        if not match:
            raise ValueError('Connected phrase alignment is missing a requested word.')
        words.append({'word': word, 'start': alignment['character_start_times_seconds'][match.start()],
                      'end': alignment['character_end_times_seconds'][match.end() - 1]})
    boundaries = [0] + [round((a['end'] + b['start']) / 2 * RATE) for a, b in zip(words, words[1:])] + [len(source)]
    assert all(a < b for a, b in zip(boundaries, boundaries[1:]))
    output = np.zeros(FRAMES); pieces = []; previous_end = 0
    for beat, word in enumerate(words):
        chunk = source[boundaries[beat]:boundaries[beat + 1]]
        active = np.flatnonzero(np.abs(chunk) > 10 ** (-48 / 20))
        if not len(active):
            raise ValueError('Requested word has no detected speech.')
        start = boundaries[beat] + max(0, int(active[0]) - round(.03 * RATE))
        end = boundaries[beat] + min(len(chunk), int(active[-1]) + 441)
        piece = source[start:end].copy()
        fade = min(220, len(piece) // 2)
        piece[:fade] *= np.linspace(0, 1, fade); piece[-fade:] *= np.linspace(1, 0, fade)
        # Preserve each complete word, deferring a count only when the preceding
        # word needs more than its nominal one-second slot. Never stretch speech.
        onset = max(beat * RATE, previous_end + round(.03 * RATE)) if beat else 0
        if onset + len(piece) > FRAMES:
            raise ValueError('Complete native speech cannot fit four seconds; preserve this take for review.')
        piece *= 10 ** (mix.VOICE_DB[beat] / 20) / mix.rms(piece)
        output[onset:onset + len(piece)] = piece
        pieces.append({'word': word['word'], 'source_start_frame': start, 'source_end_frame': end,
                       'output_start_frame': onset, 'output_end_frame': onset + len(piece), 'speech_tempo_factor': 1,
                       'relative_word_rms_db': float(mix.VOICE_DB[beat])})
        previous_end = onset + len(piece)
    output /= mix.rms(output)
    return output, pieces


def main():
    runtime = ROOT / 'assets/audio/box-breathing/de/soft-v2/manifest.json'
    before = mix.digest(runtime); manifest = json.loads(runtime.read_text(encoding='utf-8'))
    assert manifest['voice']['id'] == 'rt9KyyQmwqa1uhHGJkMa', 'Cached speech belongs to the selected British Breathing Instructor profile.'
    cache = ROOT / 'reference-audio/elevenlabs-counted-fade'
    old_cache = ROOT / 'reference-audio/soft-box-v2'
    candidate = 'elevenlabs-british-breathing-instructor'
    target = cache / candidate; target.mkdir(parents=True, exist_ok=True)
    voice, breath, provenance = {}, {}, {}
    for phase in ['inhale', 'exhale']:
        source = cache / (phase + '-connected.mp3')
        request = json.loads(source.with_suffix('.request.json').read_text(encoding='utf-8'))
        first_word = 'Einatmen' if phase == 'inhale' else 'Ausatmen'
        assert request['text'] == manifest['delivery_tags'] + ' ' + first_word + ' … zwei … drei … vier.'
        assert request['model_id'] == manifest['model_id'] and request['voice_settings'] == manifest['voice_settings']
        voice[phase], pieces = counted_phrase(source, first_word)
        original = old_cache / ('inhale.wav' if phase == 'inhale' else 'Exhale.wav')
        assert mix.digest(original) == manifest['blocks'][phase + '_breath']['processing']['source_sha256']
        fitted, tempo = mix.retime(mix.pcm(original), 4)
        # Preserve the recording's natural dynamics with one constant gain.
        # Only the first/last 10 ms fade to avoid an abrupt file boundary.
        fade = round(.01 * RATE)
        fitted[:fade] *= np.linspace(0, 1, fade)
        fitted[-fade:] *= np.linspace(1, 0, fade)
        breath[phase] = fitted / mix.rms(fitted)
        provenance[phase] = {'voice_source_sha256': mix.digest(source), 'breath_source_sha256': mix.digest(original),
                             'voice_source': 'One complete connected ElevenLabs phrase', 'request': request,
                             'pause_adjustment_only': True, 'word_pieces': pieces,
                             'breath_tempo_factor': tempo, 'pitch_preserved': True,
                             'breath_gain': 'Constant over the phase, with 10 ms boundary fades only; natural source dynamics retained.'}
    for phase, source in [('hold_in', cache / 'halten-exclamation.mp3'),
                          ('hold_out', cache / 'halten-exclamation.mp3')]:
        request = json.loads(source.with_suffix('.request.json').read_text(encoding='utf-8'))
        assert request['text'] == manifest['delivery_tags'] + ' Halten!'
        assert request['model_id'] == manifest['model_id'] and request['voice_settings'] == manifest['voice_settings']
        word, tempo = label(source)
        # Match the initial inhale cue's sample peak; a solitary word should
        # not be amplified by normalizing its long silent hold to full RMS.
        # Retain the previously selected hold level while revising counted speech.
        reference_cue, _ = mix.faded_beats(mix.pcm(runtime.parent / 'inhale.wav'), mix.VOICE_DB)
        word *= np.max(np.abs(reference_cue[:RATE])) / np.max(np.abs(word))
        block = np.zeros(FRAMES); block[:len(word)] = word
        voice[phase] = block
        provenance[phase] = {'label_source_sha256': mix.digest(source), 'speech_frames': len(word),
                             'label_tempo_factor': tempo, 'pitch_preserved': True, 'request': request,
                             'spoken_counts': False, 'silence_after_frame': len(word)}
    peak = max(float(np.max(np.abs(voice[p] + breath.get(p, 0)))) for p, _ in PHASES)
    peak = max(peak, *(float(np.max(np.abs(v))) for v in voice.values()))
    gain = min(10 ** (-26 / 20), 10 ** (-3 / 20) / peak)
    for variant in ['voice-only', 'voice-and-breath']:
        out = target / variant; out.mkdir(exist_ok=True)
        blocks = {}
        for phase, name in PHASES:
            data = (voice[phase] + (breath.get(phase, 0) if variant == 'voice-and-breath' else 0)) * gain
            hold = phase.startswith('hold')
            blocks[phase] = {'file': phase + '.wav', 'text': 'Halten!' if hold else name + ' … zwei … drei … vier.',
                             'spoken_words': ['Halten!'] if hold else [name, 'zwei', 'drei', 'vier'],
                             'word_onsets_frames': [0] if hold else [p['output_start_frame'] for p in provenance[phase]['word_pieces']],
                             'voice_beat_targets_db': [0] if hold else mix.VOICE_DB.tolist(),
                             'speech_frames': provenance[phase]['speech_frames'] if hold else None, 'frames': FRAMES,
                             'sha256': mix.write(out / (phase + '.wav'), data), 'sample_peak_dbfs': mix.db(float(np.max(np.abs(data))))}
        receipt = {'candidate': candidate, 'voice_key': 'elevenlabs', 'voice_profile': manifest['voice'],
                   'provider': manifest['provider'], 'model_id': manifest['model_id'], 'variant': variant,
                   'rate': RATE, 'phase_frames': FRAMES, 'phase_order': [p for p, _ in PHASES],
                   'blocks': blocks,
                   'voice_beat_targets_db': mix.VOICE_DB.tolist(),
                   'breath_gain_mode': 'constant', 'breath_boundary_fade_seconds': .01,
                   'shared_component_rms_dbfs': mix.db(gain), 'source_manifest_sha256': before, 'provenance': provenance,
                   'use': 'Listening version of the existing website ElevenLabs voice; mix listening acceptance pending.'}
        (out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    assert mix.digest(runtime) == before
    print(json.dumps({'voice': manifest['voice']['name'], 'component_rms_dbfs': mix.db(gain),
                      'source_dir': str(cache), 'runtime_manifest_unchanged': True}))


if __name__ == '__main__':
    main()
