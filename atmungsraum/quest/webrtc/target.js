import { VdoNinjaTransport } from './vdo-ninja-transport.js';
import { bindQuestRtcChannel } from './quest-webrtc-channel.js';

const configuration = globalThis.QuestRtcNative.configuration();
if (/^[a-f0-9]{32}:[a-f0-9]{64}$/.test(configuration)) {
  const [room, sharedSecret] = configuration.split(':');
  const carrier = new VdoNinjaTransport({ role: 'target', room, sharedSecret, label: 'Atmungsraum' });
  carrier.addEventListener('peeropen', ({ detail }) => {
    bindQuestRtcChannel(carrier, detail.peerKey, frame => {
      const reply = globalThis.QuestRtcNative.exchange(JSON.stringify(frame));
      if (!reply) throw Error('Native gateway declined the WebRTC frame');
      return JSON.parse(reply);
    });
  });
  let closed = false;
  let retry;
  function start() {
    if (closed || globalThis.QuestRtcNative.configuration() !== configuration) return;
    void carrier.start().catch(() => {
      if (!closed) retry = setTimeout(start, 2000);
    });
  }
  window.addEventListener('pagehide', () => {
    closed = true;
    clearTimeout(retry);
    void carrier.stop().catch(() => {});
  }, { once: true });
  start();
}
