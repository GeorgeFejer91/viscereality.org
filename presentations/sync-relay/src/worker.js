const HEARTBEAT_INTERVAL_MS = 15000;
const STALE_MS = 45000;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return json({ ok: true, service: "presentation-sync-relay", now: new Date().toISOString() });
    }

    if (url.pathname !== "/ws") {
      return json({
        ok: false,
        message: "Use /ws for WebSocket and /health for health checks.",
      }, 404);
    }

    if (request.headers.get("Upgrade") !== "websocket") {
      return json({ ok: false, message: "WebSocket upgrade required." }, 426);
    }

    const room = normalizeRoom(url.searchParams.get("room"));
    if (!room) {
      return json({ ok: false, message: "Missing room query param." }, 400);
    }

    const id = env.ROOMS.idFromName(room);
    const stub = env.ROOMS.get(id);
    return stub.fetch(request);
  },
};

export class RoomSession {
  constructor(state, env) {
    this.state = state;
    this.env = env;

    this.roomState = {
      deckId: null,
      controllerSecretHash: null,
      slideIndex: 0,
      seq: 0,
      controllerConnected: false,
      updatedAt: new Date().toISOString(),
    };

    this.connections = new Map();
    this.controllerSocket = null;
    this.heartbeat = setInterval(() => {
      this.runHeartbeat().catch((err) => {
        console.error("Heartbeat failed:", err);
      });
    }, HEARTBEAT_INTERVAL_MS);

    this.state.blockConcurrencyWhile(async () => {
      const stored = await this.state.storage.get("roomState");
      if (stored) {
        this.roomState = {
          ...this.roomState,
          ...stored,
        };
      }
    });
  }

  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/health") {
      return json({ ok: true, room: true, state: this.publicState() });
    }

    if (request.headers.get("Upgrade") !== "websocket") {
      return json({ ok: false, message: "WebSocket upgrade required." }, 426);
    }

    const role = (url.searchParams.get("role") || "").toLowerCase();
    const deck = (url.searchParams.get("deck") || "").trim();
    if (role !== "viewer" && role !== "controller") {
      return json({ ok: false, message: "role must be viewer or controller." }, 400);
    }
    if (!deck) {
      return json({ ok: false, message: "Missing deck query param." }, 400);
    }

    if (this.roomState.deckId && deck !== this.roomState.deckId) {
      return json({
        ok: false,
        message: `Deck mismatch for room. Expected "${this.roomState.deckId}", got "${deck}".`,
      }, 409);
    }

    if (role === "controller" && this.isControllerActive()) {
      return json({
        ok: false,
        message: "Controller already connected for this room.",
      }, 409);
    }

    const pair = new WebSocketPair();
    const client = pair[0];
    const server = pair[1];
    server.accept();

    const conn = {
      id: crypto.randomUUID(),
      ws: server,
      role,
      deck,
      authed: role === "viewer",
      lastPongAt: Date.now(),
    };

    this.connections.set(server, conn);
    if (role === "controller") {
      this.controllerSocket = server;
      this.roomState.controllerConnected = true;
      this.roomState.updatedAt = new Date().toISOString();
      await this.persistRoomState();
    }

    server.addEventListener("message", (event) => {
      this.onMessage(server, event).catch((err) => {
        console.error("Message handler failed:", err);
        this.safeSend(server, {
          type: "error",
          code: "SERVER_ERROR",
          message: "Internal error while processing message.",
        });
      });
    });

    server.addEventListener("close", () => {
      this.onClose(server).catch((err) => {
        console.error("Close handler failed:", err);
      });
    });

    server.addEventListener("error", () => {
      this.onClose(server).catch((err) => {
        console.error("Error handler failed:", err);
      });
    });

    this.safeSend(server, {
      type: "hello",
      role,
      state: this.publicState(),
    });

    return new Response(null, { status: 101, webSocket: client });
  }

  async onMessage(socket, event) {
    const conn = this.connections.get(socket);
    if (!conn) return;

    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (_) {
      this.safeSend(socket, {
        type: "error",
        code: "BAD_JSON",
        message: "Invalid JSON payload.",
      });
      return;
    }

    if (!msg || typeof msg !== "object") {
      this.safeSend(socket, {
        type: "error",
        code: "BAD_MESSAGE",
        message: "Malformed message payload.",
      });
      return;
    }

    if (msg.type === "pong") {
      conn.lastPongAt = Date.now();
      return;
    }

    if (msg.type === "ping") {
      this.safeSend(socket, { type: "pong", ts: msg.ts || Date.now() });
      return;
    }

    if (conn.role === "controller") {
      await this.onControllerMessage(conn, msg);
      return;
    }
  }

  async onControllerMessage(conn, msg) {
    if (msg.type === "auth_controller") {
      const secret = String(msg.secret || "").trim();
      if (!secret) {
        this.safeSend(conn.ws, {
          type: "error",
          code: "AUTH_FAILED",
          message: "Secret cannot be empty.",
        });
        return;
      }

      const hash = await sha256Hex(secret);
      if (!this.roomState.controllerSecretHash) {
        this.roomState.controllerSecretHash = hash;
      }

      if (hash !== this.roomState.controllerSecretHash) {
        this.safeSend(conn.ws, {
          type: "error",
          code: "AUTH_FAILED",
          message: "Invalid controller secret.",
        });
        try { conn.ws.close(4003, "auth failed"); } catch (_) {}
        return;
      }

      if (!this.roomState.deckId) {
        this.roomState.deckId = conn.deck;
      } else if (this.roomState.deckId !== conn.deck) {
        this.safeSend(conn.ws, {
          type: "error",
          code: "DECK_MISMATCH",
          message: `Room is bound to deck "${this.roomState.deckId}", got "${conn.deck}".`,
        });
        try { conn.ws.close(4004, "deck mismatch"); } catch (_) {}
        return;
      }

      conn.authed = true;
      this.roomState.controllerConnected = true;
      this.roomState.updatedAt = new Date().toISOString();
      await this.persistRoomState();
      this.safeSend(conn.ws, { type: "auth_ok", state: this.publicState() });
      return;
    }

    if (msg.type === "set_slide") {
      if (!conn.authed) {
        this.safeSend(conn.ws, {
          type: "error",
          code: "NOT_AUTHENTICATED",
          message: "Controller must authenticate first.",
        });
        return;
      }

      let target = Number(msg.targetSlideIndex);
      if (!Number.isFinite(target)) {
        this.safeSend(conn.ws, {
          type: "error",
          code: "BAD_TARGET",
          message: "targetSlideIndex must be a number.",
        });
        return;
      }

      target = Math.max(0, Math.trunc(target));
      const action = normalizeAction(msg.action);

      this.roomState.slideIndex = target;
      this.roomState.seq += 1;
      this.roomState.updatedAt = new Date().toISOString();
      await this.persistRoomState();

      const payload = {
        type: "state_update",
        action,
        state: this.publicState(),
      };
      this.broadcast(payload);
      return;
    }
  }

  async onClose(socket) {
    const conn = this.connections.get(socket);
    if (!conn) return;
    this.connections.delete(socket);

    if (socket === this.controllerSocket) {
      this.controllerSocket = null;
      this.roomState.controllerConnected = false;
      this.roomState.updatedAt = new Date().toISOString();
      await this.persistRoomState();
    }
  }

  isControllerActive() {
    if (!this.controllerSocket) return false;
    return this.connections.has(this.controllerSocket);
  }

  publicState() {
    return {
      slideIndex: this.roomState.slideIndex,
      seq: this.roomState.seq,
      deck: this.roomState.deckId,
      updatedAt: this.roomState.updatedAt,
    };
  }

  async persistRoomState() {
    await this.state.storage.put("roomState", this.roomState);
  }

  broadcast(payload) {
    for (const [socket] of this.connections) {
      this.safeSend(socket, payload);
    }
  }

  safeSend(socket, payload) {
    try {
      socket.send(JSON.stringify(payload));
      return true;
    } catch (_) {
      this.onClose(socket).catch((err) => {
        console.error("safeSend close failed:", err);
      });
      return false;
    }
  }

  async runHeartbeat() {
    const now = Date.now();
    for (const [socket, conn] of this.connections) {
      if (now - conn.lastPongAt > STALE_MS) {
        try { socket.close(4000, "stale connection"); } catch (_) {}
        await this.onClose(socket);
        continue;
      }
      this.safeSend(socket, { type: "ping", ts: now });
    }
  }
}

function normalizeRoom(room) {
  if (!room) return "";
  const cleaned = String(room).trim().toLowerCase().replace(/[^a-z0-9\-_]/g, "");
  return cleaned.slice(0, 64);
}

function normalizeAction(action) {
  const v = String(action || "").toLowerCase();
  if (v === "next" || v === "prev" || v === "jump") return v;
  return "jump";
}

async function sha256Hex(text) {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj, null, 2), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}
