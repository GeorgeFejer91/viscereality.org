// No package scripts run: download pinned MIT sources and verify npm integrity.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, readdir, writeFile, copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const root = new URL('../../', import.meta.url);
const packages = [
  ['wormhole-crypto', '0.3.1', 'P6dsua8VTTbaLozTSbeDWBuXcoAJABH8a+XutfMg9Vb2a/roUPLFvkgtuJcXorp8hwLM5RguQEWtMZqgpJYohA=='],
  ['base64-js', '1.5.1', 'AKpaYlHn8t4SVbOHCy+b5+KKgvR4vrsD8vbvrbiQJps7fKDTkjkDry6ji0rUJjC0kzbNePLwzxq8iypo41qeWA=='],
];
for (const [name, version, integrity] of packages) {
  const response = await fetch(`https://registry.npmjs.org/${name}/-/${name}-${version}.tgz`);
  if (!response.ok) throw new Error(`Cannot download ${name}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (createHash('sha512').update(bytes).digest('base64') !== integrity) throw new Error(`${name} integrity mismatch`);
  const work = new URL(`output/playwright/vendor-${name}/`, root);
  await mkdir(work, { recursive: true });
  await writeFile(new URL('package.tgz', work), bytes);
  execFileSync('tar', ['-xf', fileURLToPath(new URL('package.tgz', work)), '-C', fileURLToPath(work)]);
  const source = new URL('package/', work), destination = new URL('vendor/wormhole-crypto/', root);
  await mkdir(new URL('lib/', destination), { recursive: true });
  if (name === 'wormhole-crypto') {
    for (const file of ['index.js', 'LICENSE', ...(await readdir(new URL('lib/', source))).filter(file => file.endsWith('.js')).map(file => `lib/${file}`)]) {
      await copyFile(new URL(file, source), new URL(file, destination));
    }
    const keychain = new URL('lib/keychain.js', destination);
    await writeFile(keychain, (await readFile(keychain, 'utf8')).replace("from 'base64-js'", "from './base64.js'"));
  } else {
    const sourceText = await readFile(new URL('index.js', source), 'utf8');
    const module = sourceText.replace(/^exports\.(byteLength|toByteArray|fromByteArray) = \1\r?\n/gm, '')
      + '\nexport default { byteLength, toByteArray, fromByteArray };\n';
    await writeFile(new URL('lib/base64.js', destination), module);
    await copyFile(new URL('LICENSE', source), new URL('BASE64-LICENSE', destination));
  }
  console.log(`Vendored ${name}@${version} with verified SHA-512`);
}
