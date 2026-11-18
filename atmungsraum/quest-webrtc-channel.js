const fragmentSize = 8192;
const frameLimit = 88000;
const fragmentLimit = Math.ceil(frameLimit / fragmentSize);
const inFlightLimit = 8;
const idPattern = /^[a-f0-9]{32}$/;
const unavailable = () => Error('Der WebRTC-Kanal zur Quest ist nicht verfügbar.');

export function bindQuestRtcChannel(carrier, peerKey, exchange) {
  const pending = new Map();
  const active = new Set();
  let packet;
  let assemblyDeadline;
  let closed = false;

  function clearPacket() {
    clearTimeout(assemblyDeadline);
    packet = undefined;
  }

  function close() {
    if (closed) return;
    closed = true;
    clearPacket();
    carrier.removeEventListener('controlmessage', receive);
    carrier.removeEventListener('peerclose', departed);
    for (const reply of pending.values()) reply.reject(unavailable());
    active.clear();
    carrier.closePeer(peerKey);
  }

  function departed({ detail }) {
    if (detail.peerKey === peerKey) close();
  }

  function send(frame) {
    const encoded = JSON.stringify(frame);
    if (!idPattern.test(frame?.id) || !encoded || encoded.length > frameLimit) throw unavailable();
    const count = Math.ceil(encoded.length / fragmentSize);
    for (let part = 0; part < count; part++) {
      const text = encoded.slice(part * fragmentSize, (part + 1) * fragmentSize);
      if (!carrier.sendControl(peerKey, JSON.stringify({ v: 1, id: frame.id, part, count, text }))) {
        throw unavailable();
      }
    }
  }

  function assemble(data) {
    if (typeof data !== 'string' || data.length > fragmentSize + 200) throw unavailable();
    const row = JSON.parse(data);
    if (row?.v !== 1 || Object.keys(row).length !== 5 || typeof row.id !== 'string' || !idPattern.test(row.id)
      || !Number.isInteger(row.part) || !Number.isInteger(row.count) || row.count < 1 || row.count > fragmentLimit
      || row.part < 0 || row.part >= row.count || typeof row.text !== 'string' || !row.text.length
      || row.text.length > fragmentSize || (row.part < row.count - 1 && row.text.length !== fragmentSize)
      || /[^\x20-\x7e]/.test(row.text)) throw unavailable();

    if (row.part === 0 && !packet) {
      packet = { id: row.id, count: row.count, parts: [] };
      assemblyDeadline = setTimeout(close, 6500);
    }
    if (!packet || packet.id !== row.id || packet.count !== row.count || packet.parts.length !== row.part) throw unavailable();
    packet.parts.push(row.text);
    if (packet.parts.length < packet.count) return;
    const text = packet.parts.join('');
    clearPacket();
    if (text.length > frameLimit) throw unavailable();
    const frame = JSON.parse(text);
    if (frame?.id !== row.id) throw unavailable();
    return frame;
  }

  async function receive({ detail }) {
    if (closed || detail.peerKey !== peerKey) return;
    try {
      const frame = assemble(detail.data);
      if (!frame) return;
      if (!exchange) {
        pending.get(frame.id)?.resolve(frame);
        return;
      }
      if (active.size >= inFlightLimit || active.has(frame.id)) throw unavailable();
      active.add(frame.id);
      try {
        const reply = await exchange(frame);
        if (!closed) {
          if (reply?.id !== frame.id) throw unavailable();
          send(reply);
        }
      } finally { active.delete(frame.id); }
    } catch { close(); }
  }

  carrier.addEventListener('controlmessage', receive);
  carrier.addEventListener('peerclose', departed);
  return {
    close,
    async request(frame, signal) {
      signal.throwIfAborted();
      if (exchange || closed || pending.size >= inFlightLimit || pending.has(frame.id)) throw unavailable();
      return new Promise((resolve, reject) => {
        const done = handler => value => {
          pending.delete(frame.id);
          signal.removeEventListener('abort', abort);
          handler(value);
        };
        const reply = { resolve: done(resolve), reject: done(reject) };
        const abort = () => reply.reject(signal.reason);
        pending.set(frame.id, reply);
        signal.addEventListener('abort', abort, { once: true });
        try { send(frame); }
        catch { close(); }
      });
    },
  };
}
