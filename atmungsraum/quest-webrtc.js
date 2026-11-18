import { VdoNinjaTransport } from './vendor/quest-webrtc/vdo-ninja-transport.js';
import { bindQuestRtcChannel } from './quest-webrtc-channel.js?v=2';
import { createQuestRtcTransport } from './quest-webrtc-session.js';
import { commissioningBroker, recoverCommissionedEnrollment } from './quest-commissioning.js?v=5';
import { pairPhone } from './quest-pairing.js?v=9';
import './vendor/quest-webrtc/vdoninja-sdk.min.js';

export async function openQuestWebRtc(discovery, enrollment, signal, pairing) {
  const { owner, rtc } = discovery;
  const fallbackDeadline = enrollment && discovery.endpoints.length ? AbortSignal.timeout(5000) : null;
  const readySignal = AbortSignal.any([signal, AbortSignal.timeout(15000)]);
  readySignal.throwIfAborted();
  const carrier = new VdoNinjaTransport({ role: 'controller', room: rtc.r, sharedSecret: rtc.p });
  const stop = () => { void carrier.stop().catch(() => {}); };
  let link;
  let transport;
  let pairingAttempted = false;
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
    link = bindQuestRtcChannel(carrier, peer);
    let membership = enrollment;
    if (!membership && typeof pairing?.onCode === 'function') {
      const id = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join('');
      const exchange = async (frame, requestSignal, field) => {
        const reply = await link.request(frame, requestSignal);
        if (reply?.v !== 3 || reply.id !== frame.id || Object.keys(reply).length !== 3 || !Object.hasOwn(reply, field))
          throw Error('Die Quest unterstützt diese Verbindung noch nicht.');
        return reply[field];
      };
      const supported = await exchange({ v: 3, id: id() }, AbortSignal.any([signal, AbortSignal.timeout(6500)]), 'pairing');
      if (supported !== 1) throw Error('Die Quest unterstützt diese Verbindung noch nicht.');
      pairingAttempted = true;
      const ticket = await pairPhone({ owner, signal, onCode: pairing.onCode, transport: {
        async request(request, requestSignal) {
          const reply = await exchange({ v: 3, id: id(), request }, requestSignal, 'reply');
          return Response.json(reply);
        },
      } });
      membership = { controller: ticket.controller, key: ticket.key };
      signal.throwIfAborted();
      await pairing.onEnrollment?.(membership);
    }
    membership ??= await recoverCommissionedEnrollment({
      discovery, channel: link, broker: commissioningBroker,
      signal: AbortSignal.any([signal, AbortSignal.timeout(7500)]),
    });
    signal.throwIfAborted();
    transport = await createQuestRtcTransport({
      owner, enrollment: membership, room: rtc.r, channel: {
        request: link.request,
        close() { try { link.close(); } finally { stop(); } },
      },
    });
    signal.throwIfAborted();
    return { transport, enrollment: membership };
  } catch (error) {
    try {
      if (transport) transport.close();
      else {
        try { link?.close(); } finally { stop(); }
      }
    } catch { /* Preserve the connection failure after stopping the carrier. */ }
    if (pairingAttempted) {
      const failure = new Error(error?.message ?? 'Die Verbindungsanfrage wurde beendet.');
      failure.name = error?.name ?? 'Error';
      failure.cause = error;
      failure.pairingAttempted = true;
      throw failure;
    }
    throw error;
  }
}
