import assert from 'node:assert/strict';
import test from 'node:test';
import { createDecipheriv, createHmac } from 'node:crypto';
import { discoverQuest, readQuest } from '../quest-gateway.js';

const serial = 'QUEST-2-TEST';
const owner = { v: 1, mode: 'remote', device: 'a'.repeat(32), serial, epoch: '7', native_epoch: '8', gateways: [] };
const enrollment = { controller: 'b'.repeat(32), key: 'c'.repeat(64) };
const hotspot = { ssid: 'Pixel-Prüfung', passphrase: 'temporary-password' };
const challenge = Buffer.from('0123456789abcdef0123456789abcdef', 'hex');
const endpoint = 'http://192.168.54.12:8766/v1/gateway';

function fakeQuest(metadata = owner, reply = '192.168.54.12') {
  const chunks = [];
  const buffer = (value) => {
    const bytes = Buffer.from(value);
    return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  };
  const gatt = {
    connected: false,
    async connect() { this.connected = true; return this; },
    disconnect() { this.connected = false; },
    async getPrimaryService() {
      return { async getCharacteristic(uuid) {
        if (uuid.endsWith('0102')) return { async readValue() { return buffer(metadata.serial); } };
        if (uuid.endsWith('0103')) return { async readValue() { return buffer(JSON.stringify(metadata)); } };
        assert.ok(uuid.endsWith('0104'));
        return {
          async readValue() { return buffer(chunks.length ? reply : challenge); },
          async writeValueWithResponse(bytes) { chunks.push(Buffer.from(bytes)); }
        };
      } };
    }
  };
  return { gatt, chunks, bluetooth: { async requestDevice() { return { gatt }; } } };
}

test('Bluetooth hotspot setup encrypts credentials for the selected Quest and reads its Wi-Fi address', async () => {
  const peer = fakeQuest();
  const result = await discoverQuest(peer.bluetooth, new AbortController().signal, { serial, enrollment, hotspot });
  assert.deepEqual(result.endpoints, [endpoint]);
  assert.equal(peer.gatt.connected, false);
  assert.ok(peer.chunks.length > 1);
  peer.chunks.forEach((bytes, index) => {
    assert.equal(bytes[0], index);
    assert.equal(bytes[1], Number(index === peer.chunks.length - 1));
    assert.ok(bytes.length <= 20);
  });
  const packet = Buffer.concat(peer.chunks.map((chunk) => chunk.subarray(2)));
  const binding = [enrollment.controller, owner.device, owner.epoch, owner.native_epoch].join('|');
  const key = createHmac('sha256', enrollment.key).update('RQG1-WIFI-KEY|' + binding).digest();
  const decipher = createDecipheriv('aes-256-gcm', key, packet.subarray(0, 12));
  decipher.setAAD(Buffer.concat([Buffer.from('RQG1-WIFI|' + binding + '|'), challenge]));
  decipher.setAuthTag(packet.subarray(-16));
  const plaintext = Buffer.concat([decipher.update(packet.subarray(12, -16)), decipher.final()]);
  assert.equal(plaintext[0], Buffer.byteLength(hotspot.ssid));
  assert.equal(plaintext.subarray(1, 1 + plaintext[0]).toString(), hotspot.ssid);
  assert.equal(plaintext.subarray(1 + plaintext[0]).toString(), hotspot.passphrase);
});

test('Bluetooth setup sends no credentials to a different serial or without enrollment', async () => {
  for (const settings of [{ serial: 'OTHER-QUEST', enrollment, hotspot }, { serial, hotspot }]) {
    const peer = fakeQuest();
    await assert.rejects(discoverQuest(peer.bluetooth, new AbortController().signal, settings));
    assert.deepEqual(peer.chunks, []);
    assert.equal(peer.gatt.connected, false);
  }
});

test('Bluetooth setup refuses an invalid Wi-Fi address and disconnects on headset rejection', async () => {
  for (const reply of ['failed', '8.8.8.8']) {
    const peer = fakeQuest(owner, reply);
    await assert.rejects(discoverQuest(peer.bluetooth, new AbortController().signal, { serial, enrollment, hotspot }));
    assert.equal(peer.gatt.connected, false);
  }
});

test('Bluetooth discovery on an existing LAN still uses the advertised gateway', async () => {
  const peer = fakeQuest({ ...owner, gateways: [endpoint] });
  const result = await discoverQuest(peer.bluetooth, new AbortController().signal);
  assert.deepEqual(result.endpoints, [endpoint]);
  assert.deepEqual(peer.chunks, []);
  assert.equal(peer.gatt.connected, false);
});

test('reconnect uses the saved hotspot only when needed and checks the stable headset ID before sending it', async () => {
  const signal = new AbortController().signal;
  const saved = { serial, device: owner.device, hotspot, enrollment };
  const peer = fakeQuest();
  const found = await readQuest(peer, signal, saved);
  assert.deepEqual(found.endpoints, []);
  assert.deepEqual(peer.chunks, []);
  const rejoined = await readQuest(peer, signal, { ...saved, forceHotspot: true });
  assert.deepEqual(rejoined.endpoints, [endpoint]);
  assert.ok(peer.chunks.length > 0);
  const other = fakeQuest({ ...owner, device: 'd'.repeat(32) });
  await assert.rejects(readQuest(other, signal, { ...saved, forceHotspot: true }));
  assert.deepEqual(other.chunks, []);
});
