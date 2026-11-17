import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

const sourceDirectory = 'assets/audio/playground';
const explorationId = 'free-exploration';
const calibrationId = 'calibration-prelude';
const sampleRate = 44_100;

const recordings = {
  [explorationId]: {
    sha256: '1f1e942ba5ec3c74eb680958047b90ba0e8385e3d4dc3eb641514c57b6f6ab2a',
    duration_frames: 19_715_793,
  },
  [calibrationId]: {
    sha256: '9d723c7fd5e397c087acf8be5e16c38b04344a7deed4c908057a645ac2e0c67a',
    duration_frames: 1_952_640,
  },
};

async function packagedClip(sourceRoot, id) {
  const path = sourceDirectory + '/' + id + '.mp3';
  const bytes = await readFile(join(sourceRoot, path));
  const { sha256, duration_frames } = recordings[id];
  if (createHash('sha256').update(bytes).digest('hex') !== sha256) {
    throw new Error('Authored playback asset differs: ' + path);
  }
  return { bytes, clip: { path, bytes: bytes.length, sha256, duration_frames } };
}

export async function loadAuthoredPlayback(sourceRoot, libraryHash) {
  const [exploration, calibration] = await Promise.all([
    packagedClip(sourceRoot, explorationId), packagedClip(sourceRoot, calibrationId),
  ]);
  const instructionFrames = 960_768;
  const plan = {
    schema_version: 1,
    scenario_id: explorationId,
    sample_rate: sampleRate,
    library_manifest_sha256: libraryHash,
    total_frames: exploration.clip.duration_frames,
    events: [
      { id: 'instruction', offset_frames: 0, duration_frames: instructionFrames },
      { id: 'rest', offset_frames: instructionFrames, duration_frames: exploration.clip.duration_frames - instructionFrames },
    ],
    encoded_audio: exploration.clip,
  };
  const files = new Map([
    [exploration.clip.path, exploration.bytes],
    [calibration.clip.path, calibration.bytes],
  ]);
  return { plan, calibration: calibration.clip, files };
}
