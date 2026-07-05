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
      title: "Viscereality - MuC 2025",
      viewerPath: "/presentations/MuC/",
      manifestPath: "/presentations/MuC/deck.scene.json",
      previewImage: "/presentations/shared-assets/viscereality/previews/MuC.jpg",
      conferenceUrl: "https://muc2025.mensch-und-computer.de/en/",
      conferenceLabel: "MuC 2025",
    }),
    alpCHI: Object.freeze({
      id: "alpCHI",
      title: "Viscereality - alpCHI 2026",
      viewerPath: "/presentations/alpCHI/",
      manifestPath: "/presentations/alpCHI/deck.scene.json",
      previewImage: "/presentations/shared-assets/viscereality/previews/alpCHI.jpg",
      conferenceUrl: "https://alpchi.org/",
      conferenceLabel: "alpCHI 2026",
    }),
    BBD26: Object.freeze({
      id: "BBD26",
      title: "Viscereality - Berlin Breathwork Days",
      viewerPath: "/presentations/BBD26/",
      manifestPath: "/presentations/BBD26/deck.scene.json",
      previewImage: "/presentations/shared-assets/viscereality/previews/BBD26.jpg",
      conferenceUrl: "https://www.berlinbreathwork.org/en/bbd26-programm",
      conferenceLabel: "Berlin Breathwork Days",
    }),
  });
})();
