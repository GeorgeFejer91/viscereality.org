import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildMainStudyScenario } from '../quest/main-study-audio.mjs';

const output = process.argv[2];
if (!output || process.argv.length !== 3 || !output.toLowerCase().endsWith('.wav')) {
  console.error('Usage: node tools/render-main-study.mjs <new output.wav>');
  process.exit(2);
}

const root = fileURLToPath(new URL('../', import.meta.url));
const original = JSON.parse(await readFile(new URL('../audio/scenarios/haupt-15.json', import.meta.url), 'utf8'));
const { plan, files, music } = await buildMainStudyScenario(root, original);
const start = plan.events.find(event => event.id === 'inhale').offset_frames;
const length = plan.total_frames - start;
const mix = `[1:a]aresample=${plan.sample_rate},aformat=channel_layouts=mono,atrim=end_sample=${length},`
  + `volume=-24dB,afade=t=in:d=2,afade=t=out:ss=${length - 3 * plan.sample_rate}:ns=${3 * plan.sample_rate},`
  + `adelay=${start}S[music];[0:a][music]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95:level=0:latency=1`;
const ffmpeg = spawn('ffmpeg', ['-nostdin', '-v', 'error', '-n', '-f', 's16le', '-ar', String(plan.sample_rate),
  '-ac', '1', '-i', 'pipe:0', '-i', music, '-filter_complex', mix, '-map_metadata', '-1',
  '-metadata', 'comment=Music: Wisps of Whorls by Kevin MacLeod (incompetech.com), CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); excerpt, level adjusted and faded.',
  '-ac', '1', '-c:a', 'pcm_s16le', resolve(output)], { stdio: ['pipe', 'ignore', 'inherit'], windowsHide: true });
const result = once(ffmpeg, 'close');
ffmpeg.stdin.on('error', error => { if (error.code !== 'EPIPE') console.error(error.message); });
for (const event of plan.events) {
  if (ffmpeg.stdin.destroyed) break;
  const asset = event.asset === null ? null : plan.audio_assets[event.asset];
  const pcm = asset ? files.get('assets/audio/scenarios/haupt-15/' + asset.file).subarray(44)
    : Buffer.alloc(event.duration_frames * 2);
  if (!ffmpeg.stdin.write(pcm)) await Promise.race([once(ffmpeg.stdin, 'drain'), result]);
}
ffmpeg.stdin.end();
const [exitCode] = await result;
if (exitCode !== 0) process.exitCode = 1;
else console.log(`${resolve(output)} — ${plan.total_frames / plan.sample_rate}s, ${plan.cycles} cycles`);
