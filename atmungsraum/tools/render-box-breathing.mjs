import { readFile, mkdir, open } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createProtocol, validateManifest } from '../audio/box-breathing.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const assetDir = join(root, 'assets/audio/box-breathing/de/soft-v2');
const manifest = validateManifest(JSON.parse(await readFile(join(assetDir, 'manifest.json'), 'utf8')));
const preview = process.argv[2] === '--preview';
const minutes = Number(process.argv[2]);
const protocol = createProtocol(preview ? 'anleitung' : minutes, manifest);
const target = resolve(process.argv[3] ?? join(root, 'reference-audio/sessions',
  preview ? `preview-v${manifest.version}.wav` : `box-breathing-${minutes}min-v${manifest.version}.wav`));

// Decode RIFF chunks instead of assuming all WAV encoders use a 44-byte header.
function pcm(wav, frames) {
  if (wav.toString('ascii', 0, 4) !== 'RIFF' || wav.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Expected a RIFF/WAVE file.');
  }
  let data, validFormat = false;
  for (let offset = 12; offset + 8 <= wav.length;) {
    const id = wav.toString('ascii', offset, offset + 4);
    const size = wav.readUInt32LE(offset + 4);
    const content = offset + 8;
    if (content + size > wav.length) throw new Error('Truncated WAV chunk.');
    if (id === 'fmt ') validFormat = size >= 16 && wav.readUInt16LE(content) === 1
      && wav.readUInt16LE(content + 2) === 1 && wav.readUInt32LE(content + 4) === manifest.sample_rate
      && wav.readUInt16LE(content + 14) === 16;
    if (id === 'data') data = wav.subarray(content, content + size);
    offset = content + size + (size % 2);
  }
  if (!validFormat || data?.length !== frames * 2) throw new Error('Wrong WAV format/duration.');
  return data;
}

const blocks = {};
for (const [id, block] of Object.entries(manifest.blocks)) {
  const file = await readFile(join(assetDir, block.file));
  if (createHash('sha256').update(file).digest('hex') !== block.sha256) throw new Error(`Hash mismatch: ${id}.`);
  blocks[id] = pcm(file, block.frames);
}
const length = protocol.total_frames * 2;
const header = Buffer.alloc(44);
header.write('RIFF'); header.writeUInt32LE(36 + length, 4); header.write('WAVEfmt ', 8);
header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22);
header.writeUInt32LE(manifest.sample_rate, 24); header.writeUInt32LE(manifest.sample_rate * 2, 28);
header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34); header.write('data', 36); header.writeUInt32LE(length, 40);
const output = Buffer.alloc(44 + length);
header.copy(output);
for (const event of protocol.events) {
  if (event.id !== 'recovery') blocks[event.asset].copy(output, 44 + event.offset_frames * 2);
}
await mkdir(dirname(target), { recursive: true });
const file = await open(target, 'wx'); // Never overwrite another experimental render.
try { await file.writeFile(output); } finally { await file.close(); }
const plan = await open(`${target}.protocol.json`, 'wx');
try { await plan.writeFile(`${JSON.stringify(protocol, null, 2)}\n`); } finally { await plan.close(); }
console.log(JSON.stringify({ output: target, duration_seconds: protocol.total_frames / protocol.sample_rate,
  cycles: protocol.cycles ?? 1, recovery_seconds: (protocol.recovery_frames ?? 0) / protocol.sample_rate,
  sha256: createHash('sha256').update(output).digest('hex') }));
