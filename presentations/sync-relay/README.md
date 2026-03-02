# Presentation Sync Relay (Cloudflare Worker + Durable Object)

Realtime room relay for presenter-controlled slide sync.

## Endpoints

- `GET /health` - health check
- `WS /ws?room=<room>&deck=<deck>&role=<viewer|controller>` - realtime sync socket

## Message Protocol

Client -> server:

- `{"type":"auth_controller","secret":"..."}`
- `{"type":"set_slide","targetSlideIndex":7,"action":"jump"}`
- `{"type":"pong","ts":1710000000000}`

Server -> client:

- `{"type":"hello","role":"viewer","state":{...}}`
- `{"type":"auth_ok","state":{...}}`
- `{"type":"state_update","action":"next","state":{...}}`
- `{"type":"ping","ts":1710000000000}`
- `{"type":"error","code":"AUTH_FAILED","message":"Invalid controller secret."}`

## Deploy

From this folder:

```powershell
wrangler deploy
```

After deploy, set `relayWsBase` in:

- `presentations/shared/decks.js`

to the Worker WebSocket URL:

```text
wss://<your-worker-domain>/ws
```

## Notes

- One active controller per room.
- Room is bound to a single deck.
- Controller secret is hashed in Durable Object storage.
- Viewers are read-only.
