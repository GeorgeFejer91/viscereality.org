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
export function isDatedSessionFilename(name) {
  const parts = /^\p{L}{3}_(0|[1-9][0-9]{0,15})_[MFON]_(\d{4}-\d{2}-\d{2})_([a-z0-9-]+)_r[1-9][0-9]{0,15}\.csv$/u.exec(name);
  if (!parts || !SCENARIOS.includes(parts[3])) return false;
  const date = new Date(parts[2]);
  return Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === parts[2];
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
