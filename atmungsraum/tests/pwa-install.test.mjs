import test from 'node:test';
import assert from 'node:assert/strict';
import { mountAppInstall } from '../pwa-install.js';

function fixture(standalone = false, getInstalledRelatedApps) {
  const environment = new EventTarget();
  const media = Object.assign(new EventTarget(), { matches: standalone });
  environment.matchMedia = () => media;
  environment.location = { href: 'https://viscereality.org/atmungsraum/#patient' };
  environment.navigator = { getInstalledRelatedApps };
  const button = Object.assign(new EventTarget(), {
    hidden: false, disabled: false, attrs: {}, focused: false,
    setAttribute(name, value) { this.attrs[name] = value; },
    focus() { this.focused = true; },
  });
  const dialog = Object.assign(new EventTarget(), {
    open: false,
    showModal() { this.open = true; },
    close() { this.open = false; this.dispatchEvent(new Event('close')); },
  });
  const closeButton = new EventTarget();
  mountAppInstall({ button, dialog, closeButton, environment });
  return { environment, media, button, dialog, closeButton };
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test('A fresh browser tab suppresses installation when the browser already has this app', async () => {
  const f = fixture(false, async () => [{ platform: 'webapp', id: 'https://viscereality.org/atmungsraum/' }]);
  let prompts = 0;
  f.environment.dispatchEvent(Object.assign(new Event('beforeinstallprompt'), { prompt: async () => { prompts++; } }));
  await tick();
  assert.equal(f.button.hidden, true);
  f.button.dispatchEvent(new Event('click'));
  assert.equal(prompts, 0);
  assert.equal(f.dialog.open, false);
});

test('An unavailable or unrelated installed-app result keeps installation help available', async () => {
  for (const lookup of [async () => [{ platform: 'webapp', id: 'https://viscereality.org/' }], async () => { throw new Error('Unavailable'); }]) {
    const f = fixture(false, lookup);
    await tick();
    assert.equal(f.button.hidden, false);
    f.button.dispatchEvent(new Event('click'));
    assert.equal(f.dialog.open, true);
  }
});

test('Installation is prompted only by a user click, once per browser event, with a retryable manual fallback', async () => {
  const f = fixture();
  let calls = 0, finish;
  const event = Object.assign(new Event('beforeinstallprompt', { cancelable: true }), {
    prompt: async () => { calls++; },
    userChoice: new Promise(resolve => { finish = resolve; }),
  });
  f.environment.dispatchEvent(event);
  assert.equal(event.defaultPrevented, true);
  assert.equal(calls, 0);
  f.button.dispatchEvent(new Event('click'));
  f.button.dispatchEvent(new Event('click'));
  await tick();
  assert.equal(calls, 1);
  assert.equal(f.button.disabled, true);
  finish({ outcome: 'dismissed' });
  await tick();
  assert.equal(f.button.disabled, false);
  assert.equal(f.button.hidden, false);
  f.button.dispatchEvent(new Event('click'));
  assert.equal(f.dialog.open, true);
  assert.equal(calls, 1);
});

test('Browsers without an install event get a closable dialog; Escape cannot reach session shortcuts', () => {
  const f = fixture();
  f.button.dispatchEvent(new Event('click'));
  assert.equal(f.dialog.open, true);
  assert.equal(f.button.attrs['aria-expanded'], 'true');
  const escape = new Event('keydown');
  Object.defineProperty(escape, 'key', { value: 'Escape' });
  let stopped = false;
  escape.stopPropagation = () => { stopped = true; };
  f.dialog.dispatchEvent(escape);
  assert.equal(stopped, true);
  f.closeButton.dispatchEvent(new Event('click'));
  assert.equal(f.dialog.open, false);
  assert.equal(f.button.attrs['aria-expanded'], 'false');
  assert.equal(f.button.focused, true);
});

test('A rejected prompt falls back, while accepted consent alone never claims installation', async () => {
  const f = fixture();
  const send = prompt => f.environment.dispatchEvent(Object.assign(new Event('beforeinstallprompt'), { prompt, userChoice: Promise.resolve({ outcome: 'accepted' }) }));
  send(async () => { throw new Error('Browser refused prompt'); });
  f.button.dispatchEvent(new Event('click'));
  await tick();
  assert.equal(f.dialog.open, true);
  assert.equal(f.button.disabled, false);
  f.dialog.close();
  send(async () => {});
  f.button.dispatchEvent(new Event('click'));
  await tick();
  assert.equal(f.button.hidden, false);
  f.environment.dispatchEvent(new Event('appinstalled'));
  assert.equal(f.button.hidden, true);
});

test('Installed standalone apps hide installation controls and close obsolete help', () => {
  assert.equal(fixture(true).button.hidden, true);
  const f = fixture();
  f.button.dispatchEvent(new Event('click'));
  f.media.matches = true;
  f.media.dispatchEvent(new Event('change'));
  assert.equal(f.button.hidden, true);
  assert.equal(f.dialog.open, false);
  assert.equal(f.button.attrs['aria-expanded'], 'false');
});
