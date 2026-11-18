import { SCENARIOS } from './audio/scenario-protocol.js?v=20261106-placeholders';

export function participantIdentity({ surnamePrefix = '', age = '', gender = '' } = {}) {
  if (typeof surnamePrefix !== 'string' || !['string', 'number'].includes(typeof age)) throw new Error('Invalid participant');
  let name = surnamePrefix.trim().normalize('NFC');
  if (!/^\p{L}{3}$/u.test(name)) throw new Error('Enter three surname letters');
  const lower = name.toLocaleLowerCase('de-DE');
  const styled = lower[0].toLocaleUpperCase('de-DE') + lower.slice(1);
  if (/^\p{L}{3}$/u.test(styled)) name = styled;
  const years = String(age).trim() === '' ? NaN : Number(age);
  const genders = { male: 'M', female: 'F', other: 'O', 'prefer-not-to-say': 'N' };
  const genderCode = Object.hasOwn(genders, gender) ? genders[gender] : null;
  if (!Number.isSafeInteger(years) || years < 0 || !genderCode) throw new Error('Invalid age or gender');
  return { name, age: years, genderCode, id: `${name}_${years}_${genderCode}` };
}
export function datedSessionParts(name) {
  const parts = /^(\p{L}{3}_(?:0|[1-9][0-9]{0,15})_[MFON])_(\d{4}-\d{2}-\d{2})_([a-z0-9-]+)_r([1-9][0-9]{0,15})\.csv$/u.exec(name);
  if (!parts || (parts[3] !== 'free-exploration' && !SCENARIOS.includes(parts[3]))) return null;
  const date = new Date(parts[2]);
  return Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === parts[2] ? parts : null;
}

export const isDatedSessionFilename = name => datedSessionParts(name) !== null;

export const likertLabels = Object.freeze([
  'Stimme gar nicht zu', 'Stimme eher nicht zu', 'Teils, teils', 'Stimme eher zu', 'Stimme voll zu',
]);
const agreement = Object.freeze({ min: 1, max: 5, min_label: likertLabels[0], max_label: likertLabels.at(-1) });
export const mainQuestionnaireItems = Object.freeze([
  { type: 'slider', prompt: 'Wie ruhig fühlst du dich gerade?',
    scale: { min: 0, max: 100, min_label: 'Gar nicht ruhig', max_label: 'Sehr ruhig' } },
  { type: 'likert', prompt: 'Die Atemübung war angenehm.', scale: agreement },
  { type: 'likert', prompt: 'Ich konnte meine Atmung bewusst wahrnehmen.', scale: agreement },
  { type: 'likert', prompt: 'Ich fühlte mich im virtuellen Raum wohl.', scale: agreement },
  { type: 'slider', prompt: 'Wie intensiv war deine Erfahrung?',
    scale: { min: 0, max: 100, min_label: 'Gar nicht intensiv', max_label: 'Sehr intensiv' } },
]);

export function completedMainQuestionnaire(csv, bytes) {
  const parts = datedSessionParts(csv.name);
  if (csv.schema_version !== 4 || parts?.[3] !== 'haupt-15') return null;
  const [, participant_id, , scenario, run] = parts;
  // The checked v4 file ends in one native terminal row. A completed row has
  // no quoted fields, and a short tail avoids decoding the full recording.
  const tail = new TextDecoder().decode(bytes.subarray(-2048)).trimEnd();
  const row = tail.slice(tail.lastIndexOf('\n') + 1).split(',');
  if (row.length !== 24 || row[0] !== '4' || row[1] !== participant_id
    || row[2] !== csv.session_id || row[3] !== run || row[4] !== scenario || row[5] !== 'end'
    || row[8] !== '900000000' || row[21] !== 'completed' || row[22] !== 'completed') return null;
  return { participant_id, session_id: csv.session_id, recording_filename: csv.name };
}

export function questionnaireFile(run, answers, stamp = { utc: new Date().toISOString(), source: 'phone-system-clock' }) {
  if (answers.length !== mainQuestionnaireItems.length || mainQuestionnaireItems.some(({ scale }, index) => {
    const answer = answers[index];
    return !Number.isInteger(answer) || answer < scale.min || answer > scale.max;
  })) throw Error('Bitte beantworte alle fünf Fragen mit einer gültigen Auswahl.');
  const record = {
    schema_version: 2, ...run, started_at_utc: stamp.utc, submitted_at_utc: stamp.utc, clock_source: stamp.source,
    ...(stamp.round_trip_ms === undefined ? {} : { clock_round_trip_ms: stamp.round_trip_ms }),
    answers: mainQuestionnaireItems.map(({ type, prompt, scale }, index) => ({
      question_id: 'q' + (index + 1), type, prompt, scale, answer: answers[index],
    })),
  };
  const filename = run.recording_filename.replace(/\.csv$/u, '_questionnaire.json');
  return new File([JSON.stringify(record, null, 2) + '\n'], filename, { type: 'application/json' });
}

export async function savedBrowserExports() {
  const db = await new Promise((resolve, reject) => {
    const request = indexedDB.open('atmungsraum-session-data');
    request.onupgradeneeded = () => request.transaction.abort();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => {
      if (request.error?.name === 'AbortError') resolve(null);
      else reject(request.error);
    };
  });
  if (!db?.objectStoreNames.contains('exports')) {
    db?.close();
    return [];
  }
  try {
    const records = await new Promise((resolve, reject) => {
      const request = db.transaction('exports', 'readonly').objectStore('exports').getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return records.sort((first, second) => second.created_at.localeCompare(first.created_at));
  } finally { db.close(); }
}
