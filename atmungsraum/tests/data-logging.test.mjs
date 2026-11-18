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
  assert.equal(isDatedSessionFilename('Fej_35_M_2026-11-17_free-exploration_r1.csv'), true);
  for (const name of [
    'Fej_35_M_2026-02-29_probe-5_r1.csv',
    'Fej_35_M_2026-11-13_probe-5_r0.csv',
    'Fej_35_M_2026-11-13_haupt-10_r1.csv',
    'Fej_35_M_2026-11-13_../probe-5_r1.csv',
  ]) assert.equal(isDatedSessionFilename(name), false);
});

test('a questionnaire belongs to the completed native main CSV and requires five answers', async () => {
  const { completedMainQuestionnaire, questionnaireFile } = await import('../data-logging.js');
  const name = 'Mül_35_F_2026-11-18_haupt-15_r2.csv';
  const session_id = '01234567-89ab-4def-8abc-0123456789ab';
  const csv = { name, session_id, schema_version: 4 };
  const row = ['4', 'Mül_35_F', session_id, '2', 'haupt-15', 'end', '2026-11-18T10:00:00Z',
    '900000000', '900000000', 'ended', 'ended', ...Array(10).fill(''), 'completed', 'completed', ''];
  const bytes = cells => new TextEncoder().encode('previous row\r\n' + cells.join(',') + '\r\n');
  const run = completedMainQuestionnaire(csv, bytes(row));
  assert.deepEqual(run, { participant_id: 'Mül_35_F', session_id, recording_filename: name });
  for (const [column, value] of [[1, 'Tes_30_N'], [2, 'another-session'], [3, '1'], [4, 'probe-5'],
    [5, 'breath'], [8, '830000000'], [21, 'aborted']]) {
    const wrong = row.with(column, value);
    assert.equal(completedMainQuestionnaire(csv, bytes(wrong)), null);
  }
  assert.equal(completedMainQuestionnaire({ ...csv, name: name.replace('haupt-15', 'probe-5') }, bytes(row)), null);
  assert.throws(() => questionnaireFile(run, ['eins', 'zwei', 'drei', 'vier', ' ']), /fünf/);
  const file = questionnaireFile(run, [' eins ', 'zwei', 'drei', 'vier', 'fünf']);
  assert.equal(file.name, name.replace('.csv', '_questionnaire.json'));
  const saved = JSON.parse(await file.text());
  assert.equal(saved.participant_id, run.participant_id);
  assert.equal(saved.session_id, session_id);
  assert.equal(saved.recording_filename, name);
  assert.deepEqual(saved.answers.map(({ question_id, answer }) => [question_id, answer]),
    [['q1', 'eins'], ['q2', 'zwei'], ['q3', 'drei'], ['q4', 'vier'], ['q5', 'fünf']]);
});
