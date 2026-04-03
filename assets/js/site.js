(() => {
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const coarsePointer = window.matchMedia("(pointer: coarse)");
  const isNarrowViewport = window.matchMedia("(max-width: 767px)");
  const saveData = Boolean(navigator.connection && navigator.connection.saveData);
  const lowPowerMode = saveData || prefersReducedMotion.matches || coarsePointer.matches;

  document.documentElement.classList.toggle("low-power", lowPowerMode);

  function initOrbVisualization() {
    const container = document.getElementById("orbContainer");
    const vizElement = document.getElementById("orbViz");
    const orbCard = document.getElementById("orb-card");

    if (!container || !vizElement || !orbCard) {
      return;
    }

    const RADIUS_MIN = 80;
    const RADIUS_MAX = 140;
    const BREATH_CYCLE = 12000;
    const PARTICLE_SIZE = lowPowerMode ? 10 : 12;
    const SUBDIVISIONS = lowPowerMode ? 1 : 2;

    let isPaused = false;
    let isVisible = false;
    let rafId = 0;
    let lastTime = performance.now();
    let currentRadius = RADIUS_MIN;

    function generateIcosphere(subdivisions) {
      const vertices = [];
      const t = (1 + Math.sqrt(5)) / 2;

      const baseVertices = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
      ].map((point) => {
        const len = Math.sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
        return [point[0] / len, point[1] / len, point[2] / len];
      });

      vertices.push(...baseVertices);

      let faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
      ];

      for (let step = 0; step < subdivisions; step += 1) {
        const newFaces = [];
        const midCache = new Map();

        const getMidpoint = (i1, i2) => {
          const key = i1 < i2 ? `${i1}-${i2}` : `${i2}-${i1}`;
          if (midCache.has(key)) {
            return midCache.get(key);
          }

          const v1 = vertices[i1];
          const v2 = vertices[i2];
          const midpoint = [
            (v1[0] + v2[0]) / 2,
            (v1[1] + v2[1]) / 2,
            (v1[2] + v2[2]) / 2
          ];
          const len = Math.sqrt(midpoint[0] * midpoint[0] + midpoint[1] * midpoint[1] + midpoint[2] * midpoint[2]);
          const normalized = [midpoint[0] / len, midpoint[1] / len, midpoint[2] / len];

          const idx = vertices.length;
          vertices.push(normalized);
          midCache.set(key, idx);
          return idx;
        };

        faces.forEach(([a, b, c]) => {
          const ab = getMidpoint(a, b);
          const bc = getMidpoint(b, c);
          const ca = getMidpoint(c, a);
          newFaces.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
        });

        faces = newFaces;
      }

      return vertices;
    }

    const vertices = generateIcosphere(SUBDIVISIONS);

    const oscillators = vertices.map((pos) => {
      const particle = document.createElement("div");
      particle.className = "orb-particle";
      particle.style.width = `${PARTICLE_SIZE}px`;
      particle.style.height = `${PARTICLE_SIZE}px`;
      particle.style.left = "50%";
      particle.style.top = "50%";
      container.appendChild(particle);

      const rotationY = Math.atan2(pos[0], pos[2]) * (180 / Math.PI);
      const rotationX = -Math.asin(pos[1]) * (180 / Math.PI);
      const spatialFreq = 0.15 * Math.sin(pos[0] * 5) * Math.cos(pos[1] * 5);
      const randomFreq = (Math.random() - 0.5) * 0.12;

      return {
        element: particle,
        pos,
        phase: Math.random() * 2 * Math.PI,
        naturalFreq: 0.15 + randomFreq + spatialFreq,
        rotX: rotationX,
        rotY: rotationY,
        neighbors1: [],
        neighbors2: []
      };
    });

    oscillators.forEach((oscillator, index) => {
      const distances = oscillators
        .map((other, otherIndex) => {
          if (index === otherIndex) {
            return { index: otherIndex, dist: Infinity };
          }

          const dx = oscillator.pos[0] - other.pos[0];
          const dy = oscillator.pos[1] - other.pos[1];
          const dz = oscillator.pos[2] - other.pos[2];
          return { index: otherIndex, dist: dx * dx + dy * dy + dz * dz };
        })
        .sort((a, b) => a.dist - b.dist);

      oscillator.neighbors1 = distances.slice(0, 6).map((item) => item.index);
      oscillator.neighbors2 = distances.slice(6, 18).map((item) => item.index);
    });

    function couplingCurve1(t) {
      const centered = (t - 0.5) * 2;
      return 0.1 + 1.2 * (centered * centered);
    }

    function couplingCurve2(t) {
      const centered = (t - 0.5) * 2;
      return 0.02 + 0.4 * (centered * centered);
    }

    function updatePhases(dt, radiusNorm, time) {
      const K1 = couplingCurve1(radiusNorm);
      const K2 = couplingCurve2(radiusNorm);
      const wavePhase1 = time * 0.006;
      const wavePhase2 = time * 0.01;
      const updates = new Float32Array(oscillators.length);

      for (let i = 0; i < oscillators.length; i += 1) {
        const oscillator = oscillators[i];
        let coupling = 0;

        for (let j = 0; j < oscillator.neighbors1.length; j += 1) {
          const neighborIndex = oscillator.neighbors1[j];
          coupling += K1 * Math.sin(oscillators[neighborIndex].phase - oscillator.phase);
        }

        for (let j = 0; j < oscillator.neighbors2.length; j += 1) {
          const neighborIndex = oscillator.neighbors2[j];
          coupling += K2 * Math.sin(oscillators[neighborIndex].phase - oscillator.phase);
        }

        const wave1 = 0.02 * Math.sin(oscillator.pos[0] * 8 + oscillator.pos[1] * 8 + wavePhase1);
        const wave2 = 0.015 * Math.cos(oscillator.pos[0] * 6 - oscillator.pos[2] * 6 + wavePhase2);
        const noise = (Math.random() - 0.5) * 0.025;

        updates[i] = oscillator.naturalFreq + coupling / 18 + wave1 + wave2 + noise;
      }

      for (let i = 0; i < oscillators.length; i += 1) {
        oscillators[i].phase += updates[i] * dt;
        oscillators[i].phase %= (2 * Math.PI);
      }
    }

    const colorCache = new Array(1000);
    for (let i = 0; i < colorCache.length; i += 1) {
      const hue = (i * 0.36) % 360;
      const saturation = 75 + 25 * Math.sin(i * 0.1);
      const lightness = 60 + 20 * Math.cos(i * 0.08);
      colorCache[i] = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    }

    function phaseToColor(phase) {
      const idx = Math.floor((phase / (2 * Math.PI)) * colorCache.length) % colorCache.length;
      return colorCache[idx];
    }

    function easeInOutCubic(t) {
      return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    function getBreathState(time) {
      const cycleTime = time % BREATH_CYCLE;
      const quarter = BREATH_CYCLE / 4;

      if (cycleTime < quarter) {
        return easeInOutCubic(cycleTime / quarter);
      }

      if (cycleTime < quarter * 2) {
        return 1;
      }

      if (cycleTime < quarter * 3) {
        return 1 - easeInOutCubic((cycleTime - quarter * 2) / quarter);
      }

      return 0;
    }

    function updateVisuals(radius) {
      for (let i = 0; i < oscillators.length; i += 1) {
        const oscillator = oscillators[i];
        const x = radius * oscillator.pos[0];
        const y = radius * oscillator.pos[1];
        const z = radius * oscillator.pos[2];
        const element = oscillator.element;

        element.style.color = phaseToColor(oscillator.phase);
        element.style.transform = `translate(-50%, -50%) translate3d(${x}px, ${y}px, ${z}px) rotateY(${oscillator.rotY}deg) rotateX(${oscillator.rotX}deg)`;
      }
    }

    function shouldAnimate() {
      return isVisible && !document.hidden && !isPaused;
    }

    function stopAnimation() {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
    }

    function tick(timestamp) {
      if (!shouldAnimate()) {
        stopAnimation();
        return;
      }

      const dt = Math.min((timestamp - lastTime) / 1000, 0.1);
      lastTime = timestamp;

      const targetProgress = getBreathState(timestamp);
      const targetRadius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * targetProgress;
      currentRadius += (targetRadius - currentRadius) * 0.15;

      const radiusNorm = (currentRadius - RADIUS_MIN) / (RADIUS_MAX - RADIUS_MIN);
      updatePhases(dt * 5, radiusNorm, timestamp);
      updateVisuals(currentRadius);

      rafId = requestAnimationFrame(tick);
    }

    function startAnimation() {
      if (rafId || !shouldAnimate()) {
        return;
      }

      lastTime = performance.now();
      rafId = requestAnimationFrame(tick);
    }

    vizElement.addEventListener("click", () => {
      isPaused = !isPaused;
      vizElement.classList.toggle("orb-paused", isPaused);

      if (isPaused) {
        stopAnimation();
      } else {
        startAnimation();
      }
    });

    const visibilityObserver = new IntersectionObserver((entries) => {
      const entry = entries[0];
      isVisible = entry.isIntersecting;
      vizElement.classList.toggle("orb-active", isVisible);

      if (isVisible) {
        startAnimation();
      } else {
        stopAnimation();
      }
    }, { threshold: 0.2 });

    visibilityObserver.observe(orbCard);

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        stopAnimation();
      } else {
        startAnimation();
      }
    });
  }

  function initDeferredVideos() {
    const videos = document.querySelectorAll("[data-defer-video]");

    if (!videos.length) {
      return;
    }

    function resolveSource(sourceElement) {
      const mobileSrc = sourceElement.dataset.mobileSrc;
      const desktopSrc = sourceElement.dataset.desktopSrc;

      if (mobileSrc || desktopSrc) {
        return isNarrowViewport.matches ? (mobileSrc || desktopSrc) : (desktopSrc || mobileSrc);
      }

      return sourceElement.dataset.src || sourceElement.getAttribute("src");
    }

    function hydrateVideo(video) {
      if (video.dataset.loaded === "true") {
        return;
      }

      let hasUpdates = false;

      video.querySelectorAll("source").forEach((sourceElement) => {
        const resolvedSource = resolveSource(sourceElement);
        if (resolvedSource && sourceElement.getAttribute("src") !== resolvedSource) {
          sourceElement.setAttribute("src", resolvedSource);
          hasUpdates = true;
        }
      });

      const directSource = video.dataset.src;
      if (directSource && video.getAttribute("src") !== directSource) {
        video.setAttribute("src", directSource);
        hasUpdates = true;
      }

      if (hasUpdates) {
        video.dataset.loaded = "true";
        video.load();
      }
    }

    function syncPlayback(video) {
      const shouldPlay = video.dataset.playWhenVisible !== undefined && video.__isVisible && !document.hidden;

      if (shouldPlay) {
        const playPromise = video.play();
        if (playPromise && typeof playPromise.catch === "function") {
          playPromise.catch(() => {});
        }
      } else {
        video.pause();
      }
    }

    const videoObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const video = entry.target;

        if (entry.isIntersecting) {
          hydrateVideo(video);
        }

        video.__isVisible = entry.isIntersecting;
        syncPlayback(video);
      });
    }, { rootMargin: "500px 0px", threshold: 0.15 });

    videos.forEach((video) => {
      video.__isVisible = false;
      videoObserver.observe(video);
    });

    document.addEventListener("visibilitychange", () => {
      videos.forEach(syncPlayback);
    });
  }

  function initLiteYouTubeEmbeds() {
    const placeholders = document.querySelectorAll(".youtube-lite");

    placeholders.forEach((placeholder) => {
      placeholder.addEventListener("click", () => {
        const videoId = placeholder.dataset.youtubeId;
        const title = placeholder.dataset.youtubeTitle || "Video";

        if (!videoId) {
          return;
        }

        const iframe = document.createElement("iframe");
        iframe.src = `https://www.youtube-nocookie.com/embed/${videoId}?autoplay=1&rel=0&playsinline=1&vq=hd1080`;
        iframe.title = title;
        iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture";
        iframe.allowFullscreen = true;
        iframe.loading = "lazy";
        iframe.referrerPolicy = "strict-origin-when-cross-origin";

        placeholder.replaceWith(iframe);
      }, { once: true });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    function setMobileVH() {
      const vh = window.innerHeight * 0.01;
      document.documentElement.style.setProperty("--mobile-vh", `${vh}px`);
    }

    setMobileVH();
    window.addEventListener("resize", setMobileVH, { passive: true });
    window.addEventListener("orientationchange", () => {
      window.setTimeout(setMobileVH, 100);
    }, { passive: true });

    const textSections = Array.from(document.querySelectorAll(".text-section"));
    if (prefersReducedMotion.matches) {
      textSections.forEach((section) => section.classList.add("visible"));
    } else {
      textSections.forEach((section, index) => {
        window.setTimeout(() => {
          section.classList.add("visible");
        }, 500 + index * 3000);
      });
    }

    const revealTargets = document.querySelectorAll(".team-member, .publication-item, .advisory-member, .contact-card, .explainer-video, .funding-card, .collaboration-card, .ingredient-card, .hire-card");
    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) {
          return;
        }

        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      });
    }, { threshold: 0.2 });

    revealTargets.forEach((target) => {
      target.classList.add("reveal-ready");
      revealObserver.observe(target);
    });

    const rootStyle = document.documentElement.style;
    const pretextApi = window.Pretext && typeof window.Pretext.prepareWithSegments === "function" && typeof window.Pretext.walkLineRanges === "function"
      ? window.Pretext
      : null;
    const desktopMenu = document.querySelector(".desktop-menu");
    const mobileMenu = document.querySelector(".mobile-menu");
    const desktopMenuLinks = Array.from(document.querySelectorAll(".desktop-menu a"));
    const mobileMenuLinks = Array.from(document.querySelectorAll(".mobile-menu a"));
    const navLinks = Array.from(document.querySelectorAll(".side-menu a, .mobile-menu a"));
    const sectionLinks = navLinks.filter((link) => {
      const href = link.getAttribute("href");
      return href && href.startsWith("#");
    });
    const sectionLinkMap = new Map(sectionLinks.map((link) => [link.getAttribute("href").slice(1), link]));
    const preparedLabelCache = new Map();
    const menuMeasureState = { scheduled: false };
    let currentNavSection = "";

    function addMediaQueryChangeListener(mediaQueryList, handler) {
      if (!mediaQueryList) {
        return;
      }

      if (typeof mediaQueryList.addEventListener === "function") {
        mediaQueryList.addEventListener("change", handler);
      } else if (typeof mediaQueryList.addListener === "function") {
        mediaQueryList.addListener(handler);
      }
    }

    function clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }

    function getNavLabel(link) {
      return link.dataset.navLabel || link.textContent.replace(/\s+/g, " ").trim();
    }

    function getFontShorthand(link) {
      const styles = window.getComputedStyle(link);
      return `${styles.fontStyle} ${styles.fontVariant} ${styles.fontWeight} ${styles.fontSize} ${styles.fontFamily}`
        .replace(/\s+/g, " ")
        .trim();
    }

    function getPreparedLabel(label, font) {
      if (!pretextApi) {
        return null;
      }

      const cacheKey = `${font}\u0000${label}`;
      const cached = preparedLabelCache.get(cacheKey);
      if (cached) {
        return cached;
      }

      const prepared = pretextApi.prepareWithSegments(label, font, { whiteSpace: "pre-wrap" });
      preparedLabelCache.set(cacheKey, prepared);
      return prepared;
    }

    function measurePreparedLabel(prepared) {
      let widestLine = 0;
      let lineCount = 0;

      pretextApi.walkLineRanges(prepared, Number.POSITIVE_INFINITY, (line) => {
        lineCount += 1;
        widestLine = Math.max(widestLine, line.width);
      });

      return {
        widestLine,
        lineCount: lineCount || 1,
      };
    }

    function updateMobileMenuHeight() {
      if (!mobileMenu) {
        return;
      }

      rootStyle.setProperty("--mobile-menu-height", `${Math.ceil(mobileMenu.offsetHeight)}px`);
    }

    function centerActiveMobileLink(sectionId, behavior) {
      if (!mobileMenu || !isNarrowViewport.matches) {
        return;
      }

      const activeMobileLink = mobileMenuLinks.find((link) => link.getAttribute("href") === `#${sectionId}`);
      if (!activeMobileLink) {
        return;
      }

      activeMobileLink.scrollIntoView({
        behavior,
        inline: "center",
        block: "nearest",
      });
    }

    function measureMenuLayout() {
      menuMeasureState.scheduled = false;

      if (!desktopMenu && !mobileMenu) {
        return;
      }

      if (!pretextApi) {
        updateMobileMenuHeight();
        return;
      }

      if (desktopMenuLinks.length) {
        const desktopFont = getFontShorthand(desktopMenuLinks[0]);

        desktopMenuLinks.forEach((link) => {
          const prepared = getPreparedLabel(getNavLabel(link), desktopFont);
          const { widestLine, lineCount } = measurePreparedLabel(prepared);
          const linkWidth = clamp(Math.ceil(widestLine + 34), 92, 168);
          const linkHeight = Math.max(30, Math.ceil(lineCount * 14.4 + 12));

          link.style.setProperty("--nav-link-width", `${linkWidth}px`);
          link.style.setProperty("--nav-link-height", `${linkHeight}px`);
        });
      }

      if (mobileMenuLinks.length) {
        const mobileFont = getFontShorthand(mobileMenuLinks[0]);
        const viewportWidth = document.documentElement.clientWidth;
        const mobileChipMax = Math.max(72, Math.floor(Math.min(136, viewportWidth * 0.36)));

        mobileMenuLinks.forEach((link) => {
          const prepared = getPreparedLabel(getNavLabel(link), mobileFont);
          const { widestLine, lineCount } = measurePreparedLabel(prepared);
          const chipWidth = clamp(Math.ceil(widestLine + 18), 72, mobileChipMax);
          const chipHeight = Math.max(44, Math.ceil(lineCount * 10.8 + 16));

          link.style.setProperty("--nav-chip-width", `${chipWidth}px`);
          link.style.setProperty("--nav-chip-height", `${chipHeight}px`);
        });
      }

      updateMobileMenuHeight();

      if (currentNavSection) {
        centerActiveMobileLink(currentNavSection, "auto");
      }
    }

    function scheduleMenuLayoutMeasurement() {
      if (menuMeasureState.scheduled) {
        return;
      }

      menuMeasureState.scheduled = true;
      window.requestAnimationFrame(measureMenuLayout);
    }

    function setActiveNav(sectionId, options = {}) {
      if (!sectionId) {
        return;
      }

      if (!options.force && sectionId === currentNavSection) {
        return;
      }

      currentNavSection = sectionId;

      navLinks.forEach((link) => {
        const href = link.getAttribute("href");
        link.classList.toggle("active", Boolean(href && href === `#${sectionId}`));
      });

      centerActiveMobileLink(sectionId, options.mobileBehavior || "auto");
    }

    sectionLinks.forEach((link) => {
      link.addEventListener("click", (event) => {
        const targetId = link.getAttribute("href");
        if (!targetId || !targetId.startsWith("#")) {
          return;
        }

        const targetSection = document.querySelector(targetId);
        if (!targetSection) {
          return;
        }

        event.preventDefault();
        setActiveNav(targetId.slice(1), {
          force: true,
          mobileBehavior: prefersReducedMotion.matches ? "auto" : "smooth",
        });
        targetSection.scrollIntoView({
          behavior: prefersReducedMotion.matches ? "auto" : "smooth",
          block: "start"
        });
      });
    });

    const sectionObserver = new IntersectionObserver((entries) => {
      const visibleEntries = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio);

      if (!visibleEntries.length) {
        return;
      }

      const activeSectionId = visibleEntries[0].target.getAttribute("id");
      if (activeSectionId && sectionLinkMap.has(activeSectionId)) {
        setActiveNav(activeSectionId);
      }
    }, {
      rootMargin: "-25% 0px -45% 0px",
      threshold: [0.2, 0.45, 0.7]
    });

    document.querySelectorAll("section[id]").forEach((section) => {
      sectionObserver.observe(section);
    });

    window.addEventListener("resize", scheduleMenuLayoutMeasurement, { passive: true });
    window.addEventListener("orientationchange", () => {
      window.setTimeout(scheduleMenuLayoutMeasurement, 100);
    }, { passive: true });
    addMediaQueryChangeListener(isNarrowViewport, scheduleMenuLayoutMeasurement);
    if (document.fonts && document.fonts.ready && typeof document.fonts.ready.then === "function") {
      document.fonts.ready.then(scheduleMenuLayoutMeasurement).catch(() => {});
    }

    setActiveNav("hero");
    scheduleMenuLayoutMeasurement();

    initDeferredVideos();
    initLiteYouTubeEmbeds();
    initOrbVisualization();
  });
})();
