export const SCENARIOS = Object.freeze([
  'probe-anleitung', 'probe-5', 'probe-10', 'probe-15', 'haupt-10', 'haupt-15', 'box-20',
]);
export const AUDIO_LIBRARY = '../../assets/audio/box-breathing/de/soft-v2/manifest.json';
const PHASES = ['inhale', 'hold_in', 'exhale', 'hold_out'];
const SILENCE = ['baseline', 'recovery'];

export function scenarioId(minutes, { baselineMinutes = 0 } = {}) {
  if (baselineMinutes === 5 && [5, 10].includes(minutes)) return `haupt-${minutes + 5}`;
  if (baselineMinutes === 0) {
    if (minutes === 'anleitung') return 'probe-anleitung';
    if ([5, 10, 15].includes(minutes)) return `probe-${minutes}`;
    if (minutes === 20) return 'box-20';
  }
  throw new RangeError('Unsupported scenario selection.');
}

// Frame-indexed, contiguous slots make silence and every reused asset explicit.
export function validateScenario(plan, library, { scenarioId: expectedId, manifestHash } = {}) {
  const fail = detail => { throw new Error(`Invalid scenario ${expectedId ?? plan?.scenario_id ?? ''}: ${detail}.`); };
  if (plan?.schema_version !== 1 || !SCENARIOS.includes(plan.scenario_id)
      || (expectedId && plan.scenario_id !== expectedId)
      || plan.id !== library.id || plan.sample_rate !== library.sample_rate
      || plan.audio_library !== AUDIO_LIBRARY
      || !/^[a-f0-9]{64}$/.test(plan.library_manifest_sha256)
      || (manifestHash && plan.library_manifest_sha256 !== manifestHash)
      || !Number.isSafeInteger(plan.total_frames) || plan.total_frames <= 0
      || !Array.isArray(plan.events) || plan.events.length === 0 || plan.events.length > 1000) fail('header/library');
  const main = plan.scenario_id.startsWith('haupt-');
  const lesson = plan.scenario_id === 'probe-anleitung';
  const minutes = lesson ? null : Number(plan.scenario_id.split('-')[1]);
  const baseline = main ? 300 * library.sample_rate : 0;
  if (plan.minutes !== minutes || plan.box_minutes !== (main ? minutes - 5 : minutes)
      || plan.baseline_frames !== baseline
      || plan.mode !== (lesson ? 'instruction-example' : main ? 'main-session' : 'session')
      || plan.total_frames !== (lesson ? library.blocks.instruction.frames + 4 * library.block_frames
        + library.blocks.ending.frames : minutes * 60 * library.sample_rate)) fail('selection/total');
  let next = 0;
  const counts = {};
  const cycleIds = new Set();
  const blocks = [];
  for (const event of plan.events) {
    if (!event || ![...PHASES, ...SILENCE, 'instruction', 'ending', 'rest'].includes(event.id)
        || !Number.isSafeInteger(event.offset_frames) || event.offset_frames !== next
        || !Number.isSafeInteger(event.duration_frames) || event.duration_frames <= 0
        || !Number.isSafeInteger(next + event.duration_frames)) fail('event timing/type');
    if (SILENCE.includes(event.id)) {
      if (event.asset !== null) fail('silence asset');
    } else if (!Object.hasOwn(library.blocks, event.asset)
        || library.blocks[event.asset].frames !== event.duration_frames) fail('unknown asset/duration');
    if (PHASES.includes(event.id)) {
      if (!Number.isSafeInteger(event.cycle) || event.cycle <= 0
          || !Number.isSafeInteger(event.breathing_block) || event.breathing_block <= 0
          || event.breathing_block > 100 || event.duration_frames !== library.block_frames) fail('cycle');
      cycleIds.add(event.cycle);
      if (event.id === 'inhale') blocks[event.breathing_block - 1] = (blocks[event.breathing_block - 1] ?? 0) + 1;
    } else if (event.cycle !== null || event.breathing_block !== null) fail('non-cycle annotation');
    counts[event.id] = (counts[event.id] ?? 0) + 1;
    next += event.duration_frames;
  }
  const totalFor = id => plan.events.filter(event => event.id === id).reduce((sum, event) => sum + event.duration_frames, 0);
  if (next !== plan.total_frames || counts.instruction !== 1 || counts.ending !== 1
      || (counts.baseline ?? 0) !== (main ? 1 : 0) || totalFor('baseline') !== baseline
      || (main && plan.events[0].id !== 'baseline')
      || plan.events[main ? 1 : 0].id !== 'instruction'
      || plan.events.at(-1).id !== ((counts.recovery ?? 0) ? 'recovery' : 'ending')
      || (counts.recovery ?? 0) > 1
      || totalFor('instruction') !== plan.intro_frames || plan.intro_frames !== library.blocks.instruction.frames
      || totalFor('ending') !== plan.ending_frames || plan.ending_frames !== library.blocks.ending.frames
      || totalFor('recovery') !== plan.recovery_frames || plan.recovery_frames >= 4 * library.block_frames
      || cycleIds.size !== plan.cycles || (counts.rest ?? 0) !== plan.rests
      || !PHASES.every(id => counts[id] === plan.cycles)
      || JSON.stringify(blocks) !== JSON.stringify(plan.breathing_blocks)) fail('timeline summary');
  for (const event of plan.events) Object.freeze(event);
  Object.freeze(plan.events);
  Object.freeze(plan.breathing_blocks);
  return Object.freeze(plan);
}

export function scenarioCsv(plan, library) {
  const columns = ['order', 'event_id', 'asset_id', 'asset_file', 'start_seconds', 'duration_seconds',
    'end_seconds', 'offset_frames', 'duration_frames', 'cycle', 'breathing_block'];
  const cell = value => {
    const text = String(value ?? '');
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  const rows = plan.events.map((event, index) => [index + 1, event.id, event.asset,
    event.asset === null ? '' : `assets/audio/box-breathing/de/soft-v2/${library.blocks[event.asset].file}`,
    event.offset_frames / plan.sample_rate, event.duration_frames / plan.sample_rate,
    (event.offset_frames + event.duration_frames) / plan.sample_rate,
    event.offset_frames, event.duration_frames, event.cycle, event.breathing_block]);
  return [columns, ...rows].map(row => row.map(cell).join(',')).join('\n') + '\n';
}
