(function () {
  "use strict";

  // Replace with your deployed Worker endpoint, e.g.:
  // wss://viscereality-sync.your-subdomain.workers.dev/ws
  const relayWsBase = "wss://replace-with-your-relay-domain/ws";

  window.PRESENTATION_SYNC = Object.freeze({
    relayWsBase,
    reconnectBaseMs: 1000,
    reconnectMaxMs: 12000,
  });

  window.PRESENTATION_DECKS = Object.freeze({
    MuC: Object.freeze({
      id: "MuC",
      title: "Mensch und Computer 2025",
      viewerPath: "/presentations/MuC/",
      manifestPath: "/presentations/MuC/manifest.json",
      previewImage: "/presentations/MuC/preview.jpg",
      conferenceUrl: "https://muc2025.mensch-und-computer.de/en/",
      conferenceLabel: "MuC 2025",
    }),
    alpCHI: Object.freeze({
      id: "alpCHI",
      title: "alpCHI 2026",
      viewerPath: "/presentations/alpCHI/",
      manifestPath: "/presentations/alpCHI/manifest.json",
      previewImage: "/presentations/alpCHI/preview.jpg",
      conferenceUrl: "https://alpchi.org/",
      conferenceLabel: "alpCHI",
    }),
  });
})();
