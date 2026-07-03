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
    BBD26: Object.freeze({
      id: "BBD26",
      title: "Berlin Breathwork Days 2026",
      viewerPath: "/presentations/BBD26/",
      manifestPath: "/presentations/BBD26/manifest.json",
      previewImage: "/presentations/BBD26/preview.jpg",
      conferenceUrl: "https://www.berlinbreathwork.org/en/bbd26-programm",
      conferenceLabel: "Berlin Breathwork Days",
    }),
    "BBD26-scene": Object.freeze({
      id: "BBD26-scene",
      title: "Berlin Breathwork Days 2026 - Scene Player",
      viewerPath: "/presentations/BBD26-scene/",
      manifestPath: "/presentations/BBD26-scene/deck.scene.json",
      previewImage: "/presentations/BBD26-scene/preview.jpg",
      conferenceUrl: "https://www.berlinbreathwork.org/en/bbd26-programm",
      conferenceLabel: "Berlin Breathwork Days",
    }),
  });
})();
