import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const page = new URL('../../samples/', import.meta.url);
const catalog = JSON.parse(await readFile(new URL('samples.json', page), 'utf8'));
const document = await readFile(new URL('index.html', page), 'utf8');
const expected = ['einatmen', 'zwei', 'drei', 'vier'];
const numbers = { 2: 'zwei', 3: 'drei', 4: 'vier' };
const words = text => (text.normalize('NFKC').toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).map(word => numbers[word] || word);

assert.equal(catalog.text, 'Einatmen … zwei … drei … vier.');
assert.equal(catalog.samples.length, 20);
assert.equal(new Set(catalog.samples.map(sample => sample.name)).size, 20);
assert.equal((document.match(/<audio\b/g) || []).length, 20);
assert.equal((document.match(/preload="none"/g) || []).length, 20);
assert.equal((document.match(/class="download"/g) || []).length, 20);
for (const prefix of ['LANG', 'EmoDB']) {
  assert.deepEqual(catalog.samples.filter(sample => sample.name.startsWith(prefix)).map(sample => sample.name), Array.from({ length: 10 }, (_, number) => `${prefix} Atmungsraum ${number}`));
}
const lang = catalog.samples.filter(sample => sample.database === 'LANG-audition');
assert.deepEqual(lang.map(sample => sample.ratings.valence_positive_high), lang.map(sample => sample.ratings.valence_positive_high).sort((a, b) => a - b));
for (const sample of catalog.samples) {
  assert.match(sample.audio, /^audio\/(lang|emodb)-atmungsraum-[0-9]\.mp3$/);
  assert.equal(sample.words.verified.words_match, true);
  assert.deepEqual(words(sample.words.verified.transcript), expected);
  assert.ok(sample.duration_seconds > 0);
  const bytes = await readFile(new URL(sample.audio, page));
  assert.ok(bytes.length > 1000);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), sample.sha256);
  assert.ok(document.includes(`src="${sample.audio}"`));
  if (sample.database === 'EmoDB') {
    assert.equal(sample.ratings.basis, 'qualitative_neutral_emotion_estimate');
    assert.equal(sample.ratings.arousal, 'unmeasured');
  }
}
assert.ok(!JSON.stringify(catalog).match(/[A-Z]:\\|cogpsy-vrlab|voice_id|profile_id|token/i));
assert.ok(!document.includes('autoplay'));
console.log(`PASS: 20 unique profiles, source order, transcripts, MP3 hashes, downloads, and privacy checks (${fileURLToPath(page)})`);
