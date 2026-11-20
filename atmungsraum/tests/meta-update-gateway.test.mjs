import test from 'node:test';
import assert from 'node:assert/strict';
import { createHmac } from 'node:crypto';
import { joinQuest } from '../quest-gateway.js';

const owner = { device: 'a'.repeat(32), serial: 'TEST', epoch: '7', native_epoch: '8' };
const controller = 'b'.repeat(32), key = 'c'.repeat(64), build = 'd'.repeat(64);
const mac = text => createHmac('sha256', key).update(text).digest('hex');

async function peer({ version = 2, control = 1, waiting = false, preparation = undefined, music, musicReply, mix, mixReply } = {}) {
  const calls = [];
  const fetcher = async (_url, init) => {
    if (!init.body) return Response.json({ ...owner, v: 1, enabled: true, remote_available: true });
    const request = JSON.parse(init.body), { action, nonce } = request;
    assert.equal(request.mac, mac(`RQG1|${controller}|7|${nonce}|${action}|${request.payload}`));
    let value;
    if (action === 'status') value = { v: 1, wifi_snapshot: 1, control_mode: 'remote', recordings: 1,
      build: { v: 1, packaging_fingerprint: build }, p: 'IDLE', g: 0, r: 0, ms: 0, pending: false, rec: false,
      o: { v: 1, s: 'observed', e: '8', a: 5, i: '1', g: 0, r: 0 }, meta_update_control: control,
      ...(preparation === undefined ? {} : { meta_update_preparation: preparation }),
      ...(music === undefined ? {} : { background_music: music }),
      ...(mix === undefined ? {} : { audio_mix: mix }),
      meta_update: { v: version }, controller_access: { v: 1, role: waiting ? 'waiting' : 'active', active_phone: 'Test', revision: 2 } };
    else if (action === 'audio-mix') {
      const body = JSON.parse(request.payload);
      calls.push(body);
      value = mixReply ?? { ...body, revision: mix.revision + 1 };
    } else if (action === 'background-music') {
      const body = JSON.parse(request.payload);
      calls.push(body);
      value = musicReply ?? { ...music, ...(typeof body.playing === 'boolean'
        ? { state: body.playing ? 'starting' : 'off' } : { volume: body.volume }), revision: music.revision + 1 };
    } else {
      assert.equal(action, 'meta-update'); calls.push(JSON.parse(request.payload));
      value = { v: 1, id: calls.at(-1).id, state: 'accepted', cycle: '2', revision: '8', attempt: '20', reason: '' };
    }
    const payload = JSON.stringify(value);
    return Response.json({ v: 1, device: owner.device, epoch: '7', nonce, action, payload,
      mac: mac(`RQG1-REPLY|${controller}|${owner.device}|7|${nonce}|${action}|${payload}`) });
  };
  return { calls, gateway: await joinQuest({ endpoint: '192.168.0.10', serial: owner.serial, build, enrollment: { controller, key }, fetcher }) };
}

test('enrolled update action authenticates the exact bounded check body and original receipt', async () => {
  const { gateway, calls } = await peer();
  const body = { v: 1, id: 'f'.repeat(16), op: 'check-and-update', cycle: '2', revision: '7' };
  assert.equal((await gateway.metaUpdate(body)).state, 'accepted');
  await gateway.metaUpdate({ v: 1, id: body.id, op: 'receipt' });
  assert.deepEqual(calls, [body, { v: 1, id: body.id, op: 'receipt' }]);
  for (const bad of [{ ...body, url: 'https://invalid/' }, { ...body, target: 20 }, { ...body, op: 'install' },
    { ...body, cycle: 2 }, { ...body, revision: '0' }, { ...body, id: 'x' }]) assert.throws(() => gateway.metaUpdate(bad));
  assert.equal(calls.length, 2); gateway.close();
});

test('only advertised preparation capability accepts the exact preparation-end request', async () => {
  const body = { v: 1, id: 'f'.repeat(16), op: 'end-preparation-and-check', cycle: '2', revision: '7' };
  for (const preparation of [undefined, 0, 1]) {
    const { gateway, calls } = await peer({ preparation });
    if (preparation === 1) {
      await gateway.metaUpdate(body); assert.deepEqual(calls, [body]);
      for (const bad of [{ ...body, target: 20 }, { ...body, revision: 7 }, { ...body, cycle: '0' }])
        assert.throws(() => gateway.metaUpdate(bad));
    } else assert.throws(() => gateway.metaUpdate(body));
    gateway.close();
  }
});

test('legacy advertisement and a waiting controller cannot invoke the update action', async () => {
  for (const options of [{ version: 1 }, { control: 0 }]) {
    const { gateway } = await peer(options); assert.equal(gateway.metaUpdate, undefined); gateway.close();
  }
  const { gateway, calls } = await peer({ waiting: true });
  assert.throws(() => gateway.metaUpdate({ v: 1, id: 'f'.repeat(16), op: 'receipt' }));
  assert.equal(calls.length, 0); gateway.close();
});

test('enrolled music control sends only bounded playback and gain changes and requires matching native readback', async () => {
  const music = { v: 1, state: 'off', volume: 35, revision: 0 };
  const { gateway, calls } = await peer({ music });
  const play = { v: 1, playing: true };
  const mute = { v: 1, volume: 0 };
  assert.deepEqual(await gateway.backgroundMusic(play), { ...music, state: 'starting', revision: 1 });
  assert.deepEqual(await gateway.backgroundMusic(mute), { ...music, volume: 0, revision: 1 });
  for (const bad of [{ ...play, volume: 12 }, { ...play, url: 'https://example.org' }, { ...play, playing: 1 },
    ...[-1, 101, 0.5, '12'].map(volume => ({ v: 1, volume }))]) {
    await assert.rejects(gateway.backgroundMusic(bad));
  }
  assert.deepEqual(calls, [play, mute]);
  gateway.close();

  const mismatch = await peer({ music, musicReply: { ...music, revision: 1 } });
  await assert.rejects(mismatch.gateway.backgroundMusic(play), /nicht bestätigt/);
  mismatch.gateway.close();
});

test('older APKs, malformed music status and waiting phones cannot send music commands', async () => {
  const music = { v: 1, state: 'off', volume: 35, revision: 0 };
  for (const options of [{}, { music: { ...music, volume: 101 } }]) {
    const { gateway } = await peer(options);
    assert.equal(gateway.backgroundMusic, null);
    gateway.close();
  }
  const { gateway, calls } = await peer({ music, waiting: true });
  await assert.rejects(gateway.backgroundMusic({ v: 1, playing: true }));
  assert.deepEqual(calls, []);
  gateway.close();
});

test('the round mix sends the two exact gains and accepts only matching Quest readback', async () => {
  const mix = { v: 1, music: 30, instructions: 90, revision: 0 };
  const value = { v: 1, music: 0, instructions: 72 };
  const { gateway, calls } = await peer({ mix });
  assert.deepEqual(await gateway.audioMix(value), { ...value, revision: 1 });
  for (const bad of [{ ...value, music: -1 }, { ...value, instructions: 101 },
    { ...value, instructions: '90' }, { ...value, playing: true }]) {
    await assert.rejects(gateway.audioMix(bad));
  }
  assert.deepEqual(calls, [value]);
  gateway.close();
  const mismatch = await peer({ mix, mixReply: { ...mix, revision: 1 } });
  await assert.rejects(mismatch.gateway.audioMix(value), /nicht bestätigt/);
  mismatch.gateway.close();
  const waiting = await peer({ mix, waiting: true });
  await assert.rejects(waiting.gateway.audioMix(value));
  assert.deepEqual(waiting.calls, []);
  waiting.gateway.close();
});
