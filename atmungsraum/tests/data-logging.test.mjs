import test from 'node:test';
import assert from 'node:assert/strict';
import { participantIdentity, isDatedSessionFilename } from '../data-logging.js';

test('Participant identity normalizes names and rejects invalid form values', () => {
  const patient = { surnamePrefix: ' FeJ ', age: '35', gender: 'male' };
  assert.deepEqual(participantIdentity(patient), { name: 'Fej', age: 35, genderCode: 'M', id: 'Fej_35_M' });
  assert.equal(participantIdentity({ ...patient, surnamePrefix: 'ßab' }).id, 'ßab_35_M');
  assert.equal(participantIdentity({ ...patient, surnamePrefix: 'MU\u0308L' }).id, 'Mül_35_M');
  for (const [gender, code] of [['male', 'M'], ['female', 'F'], ['other', 'O'], ['prefer-not-to-say', 'N']]) {
    assert.equal(participantIdentity({ ...patient, gender }).id, `Fej_35_${code}`);
  }
  for (const invalid of [
    { surnamePrefix: '../' }, { surnamePrefix: '=1+' }, { surnamePrefix: 'Fejer' },
    { age: '' }, { age: '-1' }, { age: '3.5' }, { gender: '' }, { gender: '__proto__' },
  ]) assert.throws(() => participantIdentity({ ...patient, ...invalid }));
});

test('Dated Quest CSV filenames bind participant, UTC date, scenario and positive attempt', () => {
  assert.equal(isDatedSessionFilename('Fej_35_M_2026-11-13_probe-5_r1.csv'), true);
  assert.equal(isDatedSessionFilename('Mül_35_F_2028-02-29_haupt-15_r2.csv'), true);
  for (const name of [
    'Fej_35_M_2026-02-29_probe-5_r1.csv',
    'Fej_35_M_2026-11-13_probe-5_r0.csv',
    'Fej_35_M_2026-11-13_haupt-10_r1.csv',
    'Fej_35_M_2026-11-13_../probe-5_r1.csv',
  ]) assert.equal(isDatedSessionFilename(name), false);
});
