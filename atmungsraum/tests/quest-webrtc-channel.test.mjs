import assert from 'node:assert/strict';
import test from 'node:test';
import { bindQuestRtcChannel } from '../quest-webrtc-channel.js';

class Link extends EventTarget {
  writes = [];
  closed = [];
  sendControl(peerKey, data) {
    this.writes.push(data);
    this.other.dispatchEvent(new CustomEvent('controlmessage', { detail: { peerKey, data } }));
    return true;
  }
  closePeer(peerKey) { this.closed.push(peerKey); }
}

function pair(handler) {
  const phone = new Link();
  const quest = new Link();
  phone.other = quest;
  quest.other = phone;
  const target = bindQuestRtcChannel(quest, 'paired', handler);
  const client = bindQuestRtcChannel(phone, 'paired');
  return { phone, quest, client, target };
}

const frame = (id, data) => ({ v: 1, id: id.repeat(32), tag: 'e'.repeat(64), iv: '0'.repeat(16), data });

test('Unicode phone labels keep their exact text through ASCII-only WebRTC framing', async () => {
  const request = { v: 3, id: 'a'.repeat(32), request: { phone_name: 'Téléphone 日本語 📱' } };
  const { client, target, phone, quest } = pair(input => input);
  try {
    assert.deepEqual(await client.request(request, AbortSignal.timeout(500)), request);
    assert.ok([...phone.writes, ...quest.writes].every(text => !/[^\x20-\x7e]/.test(text)));
  } finally { client.close(); target.close(); }
});

test('one ordered WebRTC lane preserves large frames and matches concurrent replies', async () => {
  const one = frame('a', 'a'.repeat(87000));
  const two = frame('b', 'b'.repeat(20000));
  let release;
  const { client, quest } = pair(async input => {
    if (input.id === one.id) await new Promise(resolve => { release = resolve; });
    return input;
  });
  const signal = new AbortController().signal;
  const first = client.request(one, signal);
  assert.deepEqual(await client.request(two, signal), two);
  release();
  assert.deepEqual(await first, one);
  assert.ok(quest.writes.length > 10);
  assert.ok(quest.writes.every(part => part.length < 10000));
  client.close();
});

test('an abort releases a pending request and a different peer cannot fulfill it', async () => {
  let release;
  const { phone, quest, client, target } = pair(async input => {
    await new Promise(resolve => { release = resolve; });
    return input;
  });
  const signal = new AbortController();
  const body = frame('a', 'AA==');
  const call = client.request(body, signal.signal);
  const rejected = assert.rejects(call, { name: 'AbortError' });
  signal.abort();
  await rejected;
  release();
  await new Promise(setImmediate);
  for (const part of quest.writes) {
    phone.dispatchEvent(new CustomEvent('controlmessage', { detail: { peerKey: 'other', data: part } }));
  }
  client.close();
  target.close();
  assert.deepEqual(phone.closed, ['paired']);
  assert.deepEqual(quest.closed, ['paired']);
});

test('invalid fragments close only the bound peer without dispatch', async () => {
  let dispatches = 0;
  const { phone, quest, client, target } = pair(input => { dispatches++; return input; });
  phone.sendControl('other', 'malformed');
  assert.deepEqual(quest.closed, []);
  phone.sendControl('paired', JSON.stringify({ v: 1, id: 'a'.repeat(32), part: 1, count: 2, text: '{}' }));
  assert.deepEqual(quest.closed, ['paired']);
  assert.equal(dispatches, 0);
  client.close();
  target.close();
});
