import { cpSync, mkdirSync } from 'node:fs';

const source = new URL('../../node_modules/@chenglou/pretext/', import.meta.url);
const destination = new URL('../../vendor/pretext/', import.meta.url);
mkdirSync(destination, { recursive: true });
for (const file of ['layout.js', 'analysis.js', 'bidi.js', 'line-break.js',
  'line-text.js', 'measurement.js', 'generated/bidi-data.js']) {
  cpSync(new URL(`dist/${file}`, source), new URL(file, destination), { recursive: true });
}
cpSync(new URL('LICENSE', source), new URL('LICENSE', destination));
