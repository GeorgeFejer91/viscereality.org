import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('Quest setup uses Bluetooth discovery and button-only pairing without manual connection fields', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const setup = html.slice(html.indexOf('<div id="quest-pairing"'), html.indexOf('id="quest-clock"'));
  assert.equal((setup.match(/id="quest-find"/g) ?? []).length, 1);
  assert.match(setup, /id="quest-pairing"[^>]*hidden/);
  assert.match(setup, /id="quest-pairing-confirm"/);
  assert.match(setup, /id="quest-pairing-cancel"/);
  assert.match(setup, /id="quest-find"[^>]*><span data-measure>Mit Bluetooth verbinden<\/span>/);
  assert.doesNotMatch(setup, /<(?:form|input|details)\b|Controller-ID|Quest-Schlüssel/);
});
