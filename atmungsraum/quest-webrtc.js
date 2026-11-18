import { VdoNinjaTransport } from './vendor/quest-webrtc/vdo-ninja-transport.js';
import { bindQuestRtcChannel } from './quest-webrtc-channel.js';
import { createQuestRtcTransport } from './quest-webrtc-session.js';
import './vendor/quest-webrtc/vdoninja-sdk.min.js';

export async function openQuestWebRtc(discovery, enrollment, signal) {
  const { owner, rtc } = discovery;
  const readySignal = AbortSignal.any([signal, AbortSignal.timeout(5000)]);
  readySignal.throwIfAborted();
  const carrier = new VdoNinjaTransport({ role: 'controller', room: rtc.r, sharedSecret: rtc.p });
  const stop = () => { void carrier.stop().catch(() => {}); };
  try {
    const peer = await new Promise((resolve, reject) => {
      const finish = handler => value => {
        carrier.removeEventListener('peeropen', open);
        carrier.removeEventListener('status', status);
        readySignal.removeEventListener('abort', abort);
        handler(value);
      };
      const complete = finish(resolve);
      const fail = finish(reject);
      const open = ({ detail }) => complete(detail.peerKey);
      const status = ({ detail }) => {
        if (detail.phase === 'error') fail(Error('Das WebRTC-Signal ist nicht verfügbar.'));
      };
      const abort = () => fail(readySignal.reason);
      readySignal.addEventListener('abort', abort, { once: true });
      carrier.addEventListener('peeropen', open, { once: true });
      carrier.addEventListener('status', status);
      void carrier.start().catch(fail);
    });
    readySignal.throwIfAborted();
    const link = bindQuestRtcChannel(carrier, peer);
    return await createQuestRtcTransport({
      owner, enrollment, room: rtc.r, channel: {
        request: link.request,
        close() { link.close(); stop(); },
      },
    });
  } catch (error) {
    stop();
    throw error;
  }
}
