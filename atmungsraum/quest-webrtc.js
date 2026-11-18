import { VdoNinjaTransport } from './vendor/quest-webrtc/vdo-ninja-transport.js';
import { bindQuestRtcChannel } from './quest-webrtc-channel.js';
import { createQuestRtcTransport } from './quest-webrtc-session.js';
import { commissioningBroker, recoverCommissionedEnrollment } from './quest-commissioning.js?v=4';
import './vendor/quest-webrtc/vdoninja-sdk.min.js';

export async function openQuestWebRtc(discovery, enrollment, signal) {
  const { owner, rtc } = discovery;
  const fallbackDeadline = enrollment && discovery.endpoints.length ? AbortSignal.timeout(5000) : null;
  const readySignal = AbortSignal.any([signal, AbortSignal.timeout(15000)]);
  readySignal.throwIfAborted();
  const carrier = new VdoNinjaTransport({ role: 'controller', room: rtc.r, sharedSecret: rtc.p });
  const stop = () => { void carrier.stop().catch(() => {}); };
  try {
    const peer = await new Promise((resolve, reject) => {
      const finish = handler => value => {
        carrier.removeEventListener('peeropen', open);
        carrier.removeEventListener('status', status);
        readySignal.removeEventListener('abort', abort);
        fallbackDeadline?.removeEventListener('abort', fallback);
        handler(value);
      };
      const complete = finish(resolve);
      const fail = finish(reject);
      const open = ({ detail }) => complete(detail.peerKey);
      const status = ({ detail }) => {
        if (detail.phase === 'error') fail(Error('Das WebRTC-Signal ist nicht verfügbar.'));
        // A selected peer may still be exchanging ICE or TURN candidates.
        if (detail.phase === 'connecting-peer') fallbackDeadline?.removeEventListener('abort', fallback);
      };
      const abort = () => fail(readySignal.reason);
      const fallback = () => fail(fallbackDeadline.reason);
      readySignal.addEventListener('abort', abort, { once: true });
      fallbackDeadline?.addEventListener('abort', fallback, { once: true });
      carrier.addEventListener('peeropen', open, { once: true });
      carrier.addEventListener('status', status);
      void carrier.start().catch(fail);
    });
    readySignal.throwIfAborted();
    const link = bindQuestRtcChannel(carrier, peer);
    const membership = enrollment ?? await recoverCommissionedEnrollment({
      discovery, channel: link, broker: commissioningBroker,
      signal: AbortSignal.any([signal, AbortSignal.timeout(7500)]),
    });
    const transport = await createQuestRtcTransport({
      owner, enrollment: membership, room: rtc.r, channel: {
        request: link.request,
        close() { link.close(); stop(); },
      },
    });
    return { transport, enrollment: membership };
  } catch (error) {
    stop();
    throw error;
  }
}
