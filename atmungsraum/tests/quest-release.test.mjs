import test from 'node:test';
import assert from 'node:assert/strict';
import * as release from '../quest-release.js';
import { readFile } from 'node:fs/promises';

const packageName = 'org.viscereality.atmungsraum.developer';
const build = 'a'.repeat(64);
const owner = { device: 'b'.repeat(32), serial: 'TEST', epoch: '7', native_epoch: '8' };
const apk = (version_code = 10, patch = {}) => ({ v: 1, package_name: packageName, version_code,
  version_name: '0.2.9-dev', build_id: build, ...patch });
const status = value => ({ apk: value, build: { v: 1, packaging_fingerprint: build },
  o: { v: 1, s: 'observed', e: '8', a: 5, i: '100' } });
const policy = (patch = {}) => ({ v: 1, packages: { [packageName]: {
  minimum_version_code: 10, latest_version_code: 10, version_name: '0.2.9-dev', ...patch } } });

test('actual installed numeric APK code compares older, equal and newer against the package policy', async () => {
  assert.equal(typeof release.ApkRelease, 'function');
  const model = new release.ApkRelease({ fetcher: async () => Response.json(policy()) });
  await model.refresh();
  for (const [code, expected] of [[9, 'required'], [10, 'compatible'], [11, 'compatible']]) {
    assert.equal(model.update({ owner }, status(apk(code))).state, expected);
  }
  const optional = new release.ApkRelease({ fetcher: async () => Response.json(policy({ latest_version_code: 12, version_name: '0.3.0-dev' })) });
  await optional.refresh();
  const view = optional.update({ owner }, status(apk(10)));
  assert.equal(view.state, 'recommended');
  assert.match(view.required, /Mindestens Code 10/);
  assert.match(view.message, /empfohlen/);
});

test('missing, malformed, mismatched-build and unknown-package metadata never gets a compatible verdict', async () => {
  const model = new release.ApkRelease({ fetcher: async () => Response.json(policy()) });
  await model.refresh();
  for (const value of [undefined, null, apk(0), apk(10, { version_name: '<bad>' }), apk(10, { build_id: 'c'.repeat(64) }),
    apk(10, { package_name: 'org.other.application' }), apk(10, { version_code: '10' })]) {
    const view = model.update({ owner }, status(value));
    assert.equal(view.state, 'unknown');
    assert.equal(view.compatible, false);
  }
  for (const patch of [{ i: 'invalid' }, { e: '9' }, { a: 1501 }]) {
    const native = status(apk()); Object.assign(native.o, patch);
    assert.equal(model.update({ owner }, native).installed, null);
  }
  assert.equal(model.update({ owner: { native_epoch: '8' } }, status(apk())).installed, null);
});

test('policy failures invalidate a previous compatible verdict and never reuse a cached policy', async () => {
  const calls = []; let fail = false;
  const model = new release.ApkRelease({ fetcher: async (url, options) => {
    calls.push({ url: String(url), options });
    if (fail) throw Error('offline');
    return Response.json(policy());
  } });
  await model.refresh(); assert.equal(model.update({ owner }, status(apk())).state, 'compatible');
  fail = true; await model.refresh();
  const view = model.update({ owner }, status(apk()));
  assert.equal(view.state, 'check-unavailable'); assert.equal(view.compatible, false);
  assert.equal(calls.length, 2);
  for (const call of calls) { assert.ok(call.url.endsWith('/quest-release.json')); assert.equal(call.options.cache, 'no-store'); assert.equal(call.options.redirect, 'error'); }
  for (const bad of [null, { v: 2 }, policy({ minimum_version_code: 11 }), policy({ latest_version_code: '10' })]) {
    const invalid = new release.ApkRelease({ fetcher: async () => Response.json(bad) });
    await invalid.refresh(); assert.equal(invalid.update({ owner }, status(apk())).state, 'check-unavailable');
  }
});

test('a different trusted actor clears installed identity and does not reuse the prior actor status', async () => {
  const model = new release.ApkRelease({ fetcher: async () => Response.json(policy()) });
  await model.refresh();
  assert.ok(model.update({ owner }, status(apk())).installed);
  const next = { owner: { ...owner, native_epoch: '9' } };
  assert.equal(model.update(next, status(apk())).installed, null);
  assert.equal(model.update(next, null).installed, null);
  const current = status(apk(11)); current.o.e = '9';
  assert.equal(model.update(next, current).installed.version_code, 11);
  assert.equal(model.update(null, null).installed, null);
});

test('policy refresh is one bounded read-only request and does not mutate headset state', async () => {
  let finish; let reads = 0;
  const model = new release.ApkRelease({ fetcher: async () => { reads++; return await new Promise(resolve => { finish = resolve; }); } });
  const first = model.refresh(); const second = model.refresh();
  assert.equal(reads, 1);
  const native = status(apk()); const before = structuredClone(native);
  assert.equal(model.update({ owner }, native).state, 'checking');
  finish(Response.json(policy())); await Promise.all([first, second]);
  assert.deepEqual(native, before);
  assert.equal(model.update({ owner }, native).state, 'compatible');
});

test('the published release policy declares the selected Developer package minimum independently of installed APK metadata', async () => {
  const actual = JSON.parse(await readFile(new URL('../quest-release.json', import.meta.url), 'utf8'));
  assert.deepEqual(actual, policy({ minimum_version_code: 11, latest_version_code: 11, version_name: '0.2.10-dev' }));
  const model = new release.ApkRelease({ fetcher: async () => Response.json(actual) });
  await model.refresh();
  assert.equal(model.update({ owner }, status(apk(9))).state, 'required');
  assert.equal(model.update({ owner }, status(null)).state, 'unknown');
});
