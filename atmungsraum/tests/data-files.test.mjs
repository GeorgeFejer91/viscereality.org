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
