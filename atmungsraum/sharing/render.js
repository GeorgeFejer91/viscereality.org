import { createServer } from 'node:http';
import { pathToFileURL } from 'node:url';
import worker from './worker.js';

const origin = 'https://viscereality.org';
const local = 'http://relay.invalid';
const headers = { 'Access-Control-Allow-Origin': origin, 'Cache-Control': 'no-store' };

export function createRenderRelay(token) {
  if (!/^[A-Za-z0-9_-]{32,128}$/.test(token ?? '')) throw new Error('Configure LINK_RELAY_TOKEN before starting the upload relay.');
  const env = { ALLOWED_ORIGIN: origin, LINK_RELAY_TOKEN: token };
  return createServer({ headersTimeout: 15000, requestTimeout: 120000 }, async (incoming, outgoing) => {
    try {
      const url = new URL(incoming.url, local);
      if (url.origin !== local || url.search || url.hash) return outgoing.writeHead(404, headers).end();
      if (url.pathname === '/health' && incoming.method === 'GET') return outgoing.writeHead(204, headers).end();
      if (!(url.pathname === '/wormhole/chunk' || /^\/wormhole\/api\/room(?:\/|$)/.test(url.pathname))) {
        return outgoing.writeHead(404, headers).end();
      }
      const request = new Request(url, {
        method: incoming.method, headers: incoming.headers,
        ...(incoming.method === 'GET' || incoming.method === 'HEAD' || incoming.method === 'OPTIONS'
          ? {} : { body: incoming, duplex: 'half' }),
      });
      const response = await worker.fetch(request, env);
      const bytes = response.body ? Buffer.from(await response.arrayBuffer()) : undefined;
      if (!outgoing.destroyed) outgoing.writeHead(response.status, Object.fromEntries(response.headers)).end(bytes);
    } catch {
      if (!outgoing.destroyed) outgoing.writeHead(502, headers).end();
    }
  });
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  createRenderRelay(process.env.LINK_RELAY_TOKEN).listen(Number(process.env.PORT ?? 10000), '0.0.0.0');
}
