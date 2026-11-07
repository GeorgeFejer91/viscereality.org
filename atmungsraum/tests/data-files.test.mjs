import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';

test('ZIP extracts every CSV byte-for-byte, including Unicode and colliding filenames', async () => {
  const { createExportsZip } = await import('../data-files.js');
  const records = [
    { session_id: 'first', filename: 'Mül_35_F_probe1.csv', csv: 'name,value\r\nMüller,1\r\n' },
    { session_id: 'second', filename: 'Mül_35_F_probe1.csv', csv: 'name,value\r\nMüller,2\r\n' },
    { session_id: 'third', filename: 'Fej_35_M_main1.csv', csv: 'name,value\r\nFej,3\r\n' },
  ];
  const zip = createExportsZip(records);
  assert.equal(zip.type, 'application/zip');
  const result = spawnSync('python', ['-c', [
    'import sys, io, zipfile, json',
    'with zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read())) as z:',
    ' assert z.testzip() is None',
    ' print(json.dumps({n:z.read(n).decode("utf-8") for n in z.namelist()}))',
  ].join('\n')], { input: Buffer.from(await zip.arrayBuffer()), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), {
    'Mül_35_F_probe1.csv': 'name,value\r\nMüller,1\r\n',
    'second/Mül_35_F_probe1.csv': 'name,value\r\nMüller,2\r\n',
    'Fej_35_M_main1.csv': 'name,value\r\nFej,3\r\n',
  });
});

test('Wormhole links preserve the decryption key and reject unsafe or incomplete URLs', async () => {
  const { validateWormholeUrl } = await import('../wormhole-link.js');
  assert.equal(typeof validateWormholeUrl, 'function');
  const link = 'https://wormhole.app/nb7dMz#fnQhVRrakH6ZUgQyUeAFfA';
  assert.equal(validateWormholeUrl(`  ${link}\n`), link);
  for (const invalid of [
    '', 'https://wormhole.app/', 'https://wormhole.app/nb7dMz',
    'https://wormhole.app/nb7dMz#short',
    'http://wormhole.app/nb7dMz#fnQhVRrakH6ZUgQyUeAFfA',
    'https://wormhole.app.evil.test/nb7dMz#fnQhVRrakH6ZUgQyUeAFfA',
    'https://evil.test@wormhole.app/nb7dMz#fnQhVRrakH6ZUgQyUeAFfA',
    'https://wormhole.app:444/nb7dMz#fnQhVRrakH6ZUgQyUeAFfA',
    'https://wormhole.app/nb7dMz?redirect=evil#fnQhVRrakH6ZUgQyUeAFfA',
    'https://wormhole.app/nb7dMz/more#fnQhVRrakH6ZUgQyUeAFfA',
    'https://wormhole.app/nb7dMz#fnQhVRrakH6ZUgQyUeAF\nfA',
    'javascript:alert(1)',
  ]) assert.throws(() => validateWormholeUrl(invalid), /Wormhole/);
});
