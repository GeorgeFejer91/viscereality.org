from __future__ import annotations

from pathlib import Path


def write_player(out_dir: Path) -> None:
    (out_dir / "index.html").write_text(PLAYER_HTML, encoding="utf-8")


PLAYER_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Presentation</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #050505;
    --text: #f2f2f2;
    --muted: #9b9b9b;
    --line: rgba(255,255,255,.18);
    --hud: rgba(10,10,10,.76);
  }
  * { box-sizing: border-box; }
  html, body { width: 100%; height: 100%; margin: 0; }
  body {
    overflow: hidden;
    background: var(--bg);
    color: var(--text);
    font: 14px/1.35 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
  }
  #viewport {
    position: fixed;
    inset: 0;
    display: grid;
    place-items: center;
    background: #000;
  }
  #frame {
    position: relative;
    width: min(100vw, calc(100vh * var(--aspect)));
    height: min(100vh, calc(100vw / var(--aspect)));
    overflow: hidden;
    background: #000;
  }
  .obj {
    position: absolute;
    transform-origin: center center;
    will-change: left, top, width, height, opacity, transform;
    overflow: visible;
  }
  .obj-content,
  .obj-children,
  .obj-outline {
    position: absolute;
    inset: 0;
    box-sizing: border-box;
  }
  .obj-content {
    overflow: hidden;
    z-index: 0;
  }
  .obj-children {
    overflow: visible;
    pointer-events: none;
    z-index: 1;
  }
  .obj-outline {
    overflow: hidden;
    pointer-events: none;
    z-index: 2;
  }
  .obj-content > img,
  .obj-content > video,
  .obj-content > svg,
  .obj-outline > img,
  .obj-outline > video,
  .obj-outline > svg {
    display: block;
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: fill;
  }
  .obj > video { background: transparent; }
  .text,
  .shape {
    width: 100%;
    height: 100%;
    white-space: pre-wrap;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: .25em;
  }
  .shape { border-style: solid; border-width: 0; box-sizing: border-box; }
  .text-line {
    display: block;
    width: 100%;
  }
  .text.autofit {
    white-space: nowrap;
  }
  .text.autofit .text-line {
    align-self: center;
    max-width: none;
    transform-origin: center center;
    white-space: nowrap;
    width: max-content;
  }
  #hud {
    position: fixed;
    left: 16px;
    bottom: 14px;
    z-index: 30;
    display: flex;
    gap: 12px;
    align-items: center;
    color: var(--text);
    background: var(--hud);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 8px 10px;
    opacity: 0;
    transition: opacity .16s ease;
    pointer-events: none;
  }
  body.show-ui #hud { opacity: 1; }
  #count { color: var(--muted); }
  #fullscreen {
    position: fixed;
    right: 14px;
    bottom: 14px;
    z-index: 31;
    border: 1px solid var(--line);
    border-radius: 8px;
    color: var(--text);
    background: rgba(10,10,10,.72);
    padding: 8px 10px;
    cursor: pointer;
    opacity: 0;
    transition: opacity .16s ease, background .16s ease;
  }
  body.show-ui #fullscreen { opacity: 1; }
  body.capture-mode #hud,
  body.capture-mode #fullscreen { display: none; }
  #fullscreen:hover { background: rgba(28,28,28,.9); }
  #loading {
    position: fixed;
    inset: 0;
    display: grid;
    place-items: center;
    color: var(--muted);
    background: #000;
    z-index: 50;
  }
  #loading.hidden { display: none; }
</style>
</head>
<body class="show-ui">
<div id="viewport">
  <div id="frame" aria-live="polite"></div>
</div>
<div id="hud"><span id="title"></span><span id="count"></span></div>
<button id="fullscreen" type="button" aria-label="Toggle fullscreen">Fullscreen</button>
<div id="loading">Loading presentation...</div>
<script>
(() => {
  "use strict";

  const frame = document.getElementById("frame");
  const loading = document.getElementById("loading");
  const titleEl = document.getElementById("title");
  const countEl = document.getElementById("count");
  const fullscreenBtn = document.getElementById("fullscreen");
  const nodes = new Map();
  let scene = null;
  let currentIndex = 0;
  let transitioning = false;
  let uiTimer = null;
  let autoAdvanceTimer = null;
  const captureMode = new URLSearchParams(location.search).has("captureSlide");
  if (captureMode) {
    document.body.classList.add("capture-mode");
    document.body.classList.remove("show-ui");
  }

  fetch("deck.scene.json")
    .then((response) => response.json())
    .then((data) => {
      scene = data;
      document.title = data.deck.title || "Presentation";
      const aspect = data.deck.slideSize.width / data.deck.slideSize.height;
      frame.style.setProperty("--aspect", String(aspect || 1.7777778));
      applyInitialRoute();
      loading.classList.add("hidden");
      exposeApi();
      installCaptureEvents();
      startMediaWatchdog();
      showUi();
    })
    .catch((error) => {
      loading.textContent = "Could not load deck.scene.json";
      console.error(error);
    });

  function assetById(id) {
    return scene.assets.find((asset) => asset.id === id);
  }

  function slideAt(index) {
    return scene.slides[Math.max(0, Math.min(scene.slides.length - 1, index))];
  }

  function transitionFor(fromIndex, toIndex) {
    const forward = scene.transitions.find((item) => item.from === fromIndex + 1 && item.to === toIndex + 1);
    if (forward) return { ...forward, reverse: false };
    const reverse = scene.transitions.find((item) => item.from === toIndex + 1 && item.to === fromIndex + 1);
    if (reverse) return { ...reverse, reverse: true };
    return null;
  }

  function showSlide(index, options = {}) {
    cancelAutoAdvance();
    currentIndex = Math.max(0, Math.min(scene.slides.length - 1, index));
    const slide = slideAt(currentIndex);
    reconcile(slideObjects(slide), new Map());
    updateHud();
    preload(currentIndex + 1);
    scheduleAutoAdvance(currentIndex, options);
  }

  function slideObjects(slide) {
    return Array.isArray(slide?.nodes) && slide.nodes.length ? slide.nodes : (slide?.objects || []);
  }

  function reconcile(objects, previousStates) {
    const activeTracks = new Set(objects.map((obj) => obj.trackId));
    for (const obj of objects) {
      ensureNode(obj);
    }
    for (const obj of objects) {
      const node = ensureNode(obj);
      attachNode(node, obj, obj, activeTracks);
      applyState(node, obj, obj, 1, false, null);
      node.style.display = "";
    }
    for (const [track, node] of nodes.entries()) {
      if (!activeTracks.has(track) && !previousStates.has(track)) {
        pauseMedia(node);
        node.remove();
        nodes.delete(track);
      }
    }
  }

  function removeInactiveTransitionNodes(activeTracks) {
    for (const [track, node] of [...nodes.entries()]) {
      if (activeTracks.has(track)) continue;
      pauseMedia(node);
      node.remove();
      nodes.delete(track);
    }
  }

  function ensureNode(obj) {
    const renderAsset = renderAssetForObject(obj);
    const renderAssetId = renderAsset?.id || obj.assetId || "";
    const renderKind = renderAsset && renderAsset.kind === "video" ? "video" : (renderAsset ? "image" : obj.kind);
    const existing = nodes.get(obj.trackId);
    const nodeRole = nodeRoleForObject(obj);
    const outlineMode = outlineModeForObject(obj);
    if (
      existing
      && existing.dataset.assetId === String(renderAssetId)
      && existing.dataset.kind === renderKind
      && existing.dataset.nodeRole === nodeRole
      && existing.dataset.outlineMode === outlineMode
    ) {
      existing.dataset.settledOnly = isSettledOnlyRasterFallback(obj) ? "1" : "0";
      return existing;
    }
    if (existing) {
      existing.remove();
      nodes.delete(obj.trackId);
    }
    const node = document.createElement("div");
    node.className = "obj";
    node.dataset.trackId = obj.trackId;
    node.dataset.kind = renderKind;
    node.dataset.nodeRole = nodeRole;
    node.dataset.outlineMode = outlineMode;
    node.dataset.assetId = String(renderAssetId);
    node.dataset.settledOnly = isSettledOnlyRasterFallback(obj) ? "1" : "0";
    node.classList.toggle("panel", nodeRole === "panel");
    node.classList.toggle("group", nodeRole === "group");
    node.classList.toggle("top-outline", outlineMode === "top");
    buildNodeShell(node, obj);
    frame.appendChild(node);
    nodes.set(obj.trackId, node);
    return node;
  }

  function buildNodeShell(node, obj) {
    if (nodeRoleForObject(obj) === "group") {
      const children = document.createElement("div");
      children.className = "obj-children";
      node.appendChild(children);
      return;
    }
    if (nodeRoleForObject(obj) === "panel") {
      const base = document.createElement("div");
      base.className = "obj-content panel-base";
      const baseShape = document.createElement("div");
      baseShape.className = "shape panel-fill";
      base.appendChild(baseShape);
      node.appendChild(base);
      const children = document.createElement("div");
      children.className = "obj-children";
      node.appendChild(children);
      const outline = document.createElement("div");
      outline.className = "obj-outline";
      outline.appendChild(createPanelOutlineChild(obj));
      node.appendChild(outline);
      return;
    }
    const content = document.createElement("div");
    content.className = "obj-content";
    content.appendChild(createChild(obj));
    node.appendChild(content);
    if (outlineModeForObject(obj) === "top") {
      const outline = document.createElement("div");
      outline.className = "obj-outline";
      outline.appendChild(createOutlineChild(obj));
      node.appendChild(outline);
    }
  }

  function createPanelOutlineChild(obj) {
    const outline = createOutlineChild(obj);
    outline.classList.add("panel-outline-shape");
    return outline;
  }

  function createOutlineChild(obj) {
    if (obj.kind === "shape" && needsSvgShape(obj.shape)) {
      return createSvgShape(obj.shape);
    }
    const outline = document.createElement("div");
    outline.className = "shape object-outline-shape";
    return outline;
  }

  function nodeRoleForObject(obj) {
    if (obj?.nodeRole === "panel" || obj?.panelRole === "container") return "panel";
    if (obj?.nodeRole === "group" || obj?.kind === "group") return "group";
    return "object";
  }

  function outlineModeForObject(obj) {
    const role = nodeRoleForObject(obj);
    if (role === "panel") return "panel";
    if (role === "group") return "none";
    return shouldUseTopOutline(obj) ? "top" : "inline";
  }

  function shouldUseTopOutline(obj) {
    const style = outlineStyle();
    if (!style.borderOnTop) return false;
    if (!isWhiteStroke(obj?.stroke)) return false;
    return ["image", "video", "svg", "shape"].includes(String(obj?.kind || ""));
  }

  function attachNode(node, from, to, activeTracks) {
    const parentTrack = panelParentTrack(from, to);
    const parentNode = parentTrack && parentTrack !== node.dataset.trackId && activeTracks.has(parentTrack)
      ? nodes.get(parentTrack)
      : null;
    const target = parentNode?.querySelector(":scope > .obj-children") || frame;
    if (node.parentElement !== target) {
      target.appendChild(node);
    }
    node.dataset.parentTrackId = parentNode ? parentTrack : "";
  }

  function createChild(obj) {
    const asset = renderAssetForObject(obj);
    if (asset && asset.kind === "video" && asset.file) {
      const video = document.createElement("video");
      video.src = asset.file;
      const posterAsset = obj.posterAssetId ? assetById(obj.posterAssetId) : null;
      if (posterAsset?.file) video.poster = posterAsset.file;
      video.loop = true;
      video.muted = true;
      video.playsInline = true;
      video.preload = "auto";
      return video;
    }
    if ((obj.kind === "image" || obj.kind === "svg" || asset) && asset && asset.file) {
      const image = document.createElement("img");
      image.src = asset.file;
      image.alt = obj.name || "";
      return image;
    }
    if (obj.kind === "shape" && needsSvgShape(obj.shape)) {
      return createSvgShape(obj.shape);
    }
    const div = document.createElement("div");
    div.className = obj.kind === "text" ? "text" : "shape";
    return div;
  }

  function renderAssetForObject(obj) {
    const asset = obj.assetId ? assetById(obj.assetId) : null;
    const posterAsset = obj.posterAssetId ? assetById(obj.posterAssetId) : null;
    if (isPausedMedia(obj) && posterAsset?.file) return posterAsset;
    return asset;
  }

  function applyState(node, from, to, progress, isTransition, transition = null) {
    const state = lerpState(from, to, progress, transition);
    const g = stateGeometryForNode(node, state);
    node.style.left = (g.leftPct * 100) + "%";
    node.style.top = (g.topPct * 100) + "%";
    node.style.width = (g.widthPct * 100) + "%";
    node.style.height = (g.heightPct * 100) + "%";
    node.style.zIndex = String(transitionZIndex(state, isTransition, transition));
    const sx = g.flipH ? -1 : 1;
    const sy = g.flipV ? -1 : 1;
    node.style.transform = `rotate(${g.rotation || 0}deg) scale(${sx}, ${sy})`;
    node.style.opacity = String(Math.max(0, Math.min(1, state.opacity)));
    if (node.dataset.nodeRole === "panel") {
      applyPanelState(node, state);
      syncMediaPlayback(node, state, isTransition);
      return;
    }
    if (node.dataset.nodeRole === "group") {
      return;
    }
    if (node.dataset.outlineMode === "top") {
      applyOutlinedObjectState(node, state);
      syncMediaPlayback(node, state, isTransition);
      return;
    }
    const child = visualChild(node);
    if (child && (child.tagName === "IMG" || child.tagName === "VIDEO")) {
      const content = contentBox(node);
      content.style.borderColor = cssColor(state.stroke);
      content.style.borderStyle = state.stroke ? "solid" : "none";
      content.style.borderWidth = state.stroke ? cssStrokeWidth(state) : "0";
      content.style.borderRadius = cssShapeRadius(state);
      applyMediaCrop(child, state.crop);
      applyMediaEffects(child, state.mediaEffects);
    } else {
      const content = contentBox(node);
      if (content) {
        content.style.borderWidth = "0";
        content.style.borderRadius = "0";
      }
    }
    if (child && child.classList.contains("shape")) {
      child.style.background = cssColor(state.fill);
      child.style.borderColor = cssColor(state.stroke);
      child.style.borderWidth = state.stroke ? cssStrokeWidth(state) : "0";
      child.style.borderRadius = cssShapeRadius(state);
    }
    if (child && child.classList.contains("shape-svg")) {
      applySvgShape(child, state);
    }
    if (child && child.classList.contains("text")) {
      child.style.background = cssColor(state.fill);
      child.style.borderColor = cssColor(state.stroke);
      child.style.borderStyle = state.stroke ? "solid" : "none";
      child.style.borderWidth = state.stroke ? cssStrokeWidth(state) : "0";
      child.style.borderRadius = cssShapeRadius(state);
      applyTextStyle(child, state.textStyle || {});
      renderText(child, state);
      fitText(child, state);
    }
    syncMediaPlayback(node, state, isTransition);
  }

  function applyOutlinedObjectState(node, state) {
    const content = contentBox(node);
    const contentChild = content?.firstElementChild || null;
    const outline = outlineBox(node);
    const outlineChild = outline?.firstElementChild || null;
    if (content && contentChild) {
      applyVisualBoxState(content, contentChild, state, { border: false, fill: true });
    }
    if (outline && outlineChild) {
      applyVisualBoxState(outline, outlineChild, state, { border: true, fill: false });
    }
  }

  function stateGeometryForNode(node, state) {
    if (node.dataset.parentTrackId && state.localGeometry) return state.localGeometry;
    return state.geometry;
  }

  function contentBox(node) {
    return node.querySelector(":scope > .obj-content");
  }

  function outlineBox(node) {
    return node.querySelector(":scope > .obj-outline");
  }

  function visualChild(node) {
    return contentBox(node)?.firstElementChild || outlineBox(node)?.firstElementChild || null;
  }

  function applyPanelState(node, state) {
    const base = contentBox(node);
    const baseChild = base?.firstElementChild || null;
    const outline = outlineBox(node);
    const outlineChild = outline?.firstElementChild || null;
    if (base && baseChild) {
      applyVisualBoxState(base, baseChild, state, { border: false, fill: true, defaultFill: "#000000" });
    }
    if (outline && outlineChild) {
      applyVisualBoxState(outline, outlineChild, state, { border: true, fill: false });
    }
  }

  function applyVisualBoxState(box, child, state, options) {
    const stroke = options.border ? effectiveOutlineStroke(state) : null;
    const strokeWidth = options.border && stroke ? cssStrokeWidth(state) : "0";
    if (child.tagName === "IMG" || child.tagName === "VIDEO") {
      box.style.borderColor = cssColor(stroke);
      box.style.borderStyle = stroke ? "solid" : "none";
      box.style.borderWidth = strokeWidth;
      box.style.borderRadius = cssShapeRadius(state);
      applyMediaCrop(child, state.crop);
      applyMediaEffects(child, state.mediaEffects);
      return;
    }
    if (child.classList.contains("shape")) {
      child.style.background = options.fill ? cssColor(state.fill || options.defaultFill || null) : "transparent";
      child.style.borderColor = cssColor(stroke);
      child.style.borderStyle = stroke ? "solid" : "none";
      child.style.borderWidth = strokeWidth;
      child.style.borderRadius = cssShapeRadius(state);
    }
    if (child.classList.contains("shape-svg")) {
      applySvgShape(child, {
        ...state,
        fill: options.fill ? (state.fill || options.defaultFill || null) : null,
        stroke,
        strokeWidthPct: options.border && stroke ? normalizedStrokeWidthPct(state) : 0,
      });
    }
  }

  function lerpState(from, to, progress, transition) {
    if (!from) {
      const fadeProgress = unmatchedFadeProgress(progress, transition, "enter");
      const state = structuredClone(to);
      state.opacity = (to.opacity ?? 1) * fadeProgress;
      return state;
    }
    if (!to) {
      const fadeProgress = unmatchedFadeProgress(progress, transition, "exit");
      const state = structuredClone(from);
      state.opacity = (from.opacity ?? 1) * (1 - fadeProgress);
      return state;
    }
    const state = structuredClone(to);
    state.geometry = {
      ...to.geometry,
      leftPct: lerp(from.geometry.leftPct, to.geometry.leftPct, progress),
      topPct: lerp(from.geometry.topPct, to.geometry.topPct, progress),
      widthPct: lerp(from.geometry.widthPct, to.geometry.widthPct, progress),
      heightPct: lerp(from.geometry.heightPct, to.geometry.heightPct, progress),
      rotation: lerpAngle(from.geometry.rotation || 0, to.geometry.rotation || 0, progress),
      flipH: progress < .5 ? from.geometry.flipH : to.geometry.flipH,
      flipV: progress < .5 ? from.geometry.flipV : to.geometry.flipV,
    };
    if (from.localGeometry || to.localGeometry) {
      const fromLocal = from.localGeometry || from.geometry;
      const toLocal = to.localGeometry || to.geometry;
      state.localGeometry = {
        ...toLocal,
        leftPct: lerp(fromLocal.leftPct, toLocal.leftPct, progress),
        topPct: lerp(fromLocal.topPct, toLocal.topPct, progress),
        widthPct: lerp(fromLocal.widthPct, toLocal.widthPct, progress),
        heightPct: lerp(fromLocal.heightPct, toLocal.heightPct, progress),
        rotation: lerpAngle(fromLocal.rotation || 0, toLocal.rotation || 0, progress),
        flipH: progress < .5 ? fromLocal.flipH : toLocal.flipH,
        flipV: progress < .5 ? fromLocal.flipV : toLocal.flipV,
      };
      state.parentTrackId = to.parentTrackId || from.parentTrackId || to.panelParentTrackId || from.panelParentTrackId || null;
      state.panelParentTrackId = state.parentTrackId;
    }
    state.opacity = lerp(from.opacity ?? 1, to.opacity ?? 1, progress);
    state.strokeWidthPct = lerp(from.strokeWidthPct || 0, to.strokeWidthPct || 0, progress);
    state.textStyle = {
      ...(to.textStyle || {}),
      fontSizePt: lerp((from.textStyle || {}).fontSizePt, (to.textStyle || {}).fontSizePt, progress),
    };
    if (transition) {
      const mediaObj = transitionMediaObject(from, to);
      if (mediaObj) {
        state.id = mediaObj.id;
        state.kind = mediaObj.kind;
        state.assetId = mediaObj.assetId;
        state.posterAssetId = mediaObj.posterAssetId;
        state.mediaEffects = structuredClone(mediaObj.mediaEffects || {});
        state.mediaTiming = structuredClone(mediaObj.mediaTiming || {});
        const phaseOverride = transitionMediaPhaseOverride(state, transition);
        if (phaseOverride !== null) {
          state.mediaTiming.phaseSec = phaseOverride;
        }
      }
    }
    return state;
  }

  function unmatchedFadeProgress(progress, transition, direction) {
    const runtime = scene.runtime || {};
    if (runtime.fadeUnmatched === false) {
      return progress >= 1 ? 1 : 0;
    }
    const fade = transition?.unmatchedFade || {};
    const prefix = direction === "exit" ? "exit" : "enter";
    const rawStart = Number(fade[`${prefix}Start`] ?? runtime.unmatchedFadeStart ?? 0);
    const rawEnd = Number(fade[`${prefix}End`] ?? runtime.unmatchedFadeEnd ?? 1);
    const start = Math.max(0, Math.min(1, Number.isFinite(rawStart) ? rawStart : 0));
    const end = Math.max(0, Math.min(1, Number.isFinite(rawEnd) ? rawEnd : 1));
    if (end <= start) {
      return progress >= start ? 1 : 0;
    }
    return Math.max(0, Math.min(1, (progress - start) / (end - start)));
  }

  function goNext() {
    if (transitioning || currentIndex >= scene.slides.length - 1) return;
    runTransition(currentIndex, currentIndex + 1);
  }

  function goPrev() {
    if (transitioning || currentIndex <= 0) return;
    runTransition(currentIndex, currentIndex - 1);
  }

  function goTo(index) {
    if (transitioning) return;
    showSlide(Number(index) - 1);
  }

  function cancelAutoAdvance() {
    if (!autoAdvanceTimer) return;
    clearTimeout(autoAdvanceTimer);
    autoAdvanceTimer = null;
  }

  function scheduleAutoAdvance(index, options = {}) {
    if (captureMode || options.autoAdvance === false || options.direction === "reverse") return;
    const rule = autoAdvanceRule(index + 1);
    if (!rule) return;
    const toIndex = Math.max(0, Math.min(scene.slides.length - 1, Number(rule.to || index + 2) - 1));
    if (toIndex === index) return;
    const delayMs = Math.max(0, Number(rule.delaySec || 0) * 1000);
    autoAdvanceTimer = setTimeout(() => {
      autoAdvanceTimer = null;
      if (transitioning || currentIndex !== index) return;
      runTransition(index, toIndex, undefined, { autoAdvance: true });
    }, delayMs);
  }

  function autoAdvanceRule(slideNumber) {
    const rows = [
      ...(scene?.runtime?.autoAdvance || []),
      ...(scene?.runtime?.autoSegments || []),
    ];
    return rows.find((row) => Number(row.from) === Number(slideNumber)) || null;
  }

  function runTransition(fromIndex, toIndex, fixedProgress, captureOptions = null) {
    cancelAutoAdvance();
    const fromSlide = slideAt(fromIndex);
    const toSlide = slideAt(toIndex);
    const transition = transitionFor(fromIndex, toIndex) || { durationSec: 0 };
    const duration = Math.max(0, Number(transition.durationSec || 0)) * 1000;
    const fromByTrack = new Map(slideObjects(fromSlide).filter((obj) => !isSettledOnlyRasterFallback(obj)).map((obj) => [obj.trackId, obj]));
    const toByTrack = new Map(slideObjects(toSlide).filter((obj) => !isSettledOnlyRasterFallback(obj)).map((obj) => [obj.trackId, obj]));
    const tracks = new Set([...fromByTrack.keys(), ...toByTrack.keys()]);
    const inferredMotions = inferredMotionMap(transition);
    hideSettledOnlyFallbacks();
    const renderProgress = (raw) => {
      const baseProgress = interpolationProgress(raw, transition);
      const rows = [...tracks].map((track) => {
        const from = fromByTrack.get(track) || null;
        const to = toByTrack.get(track) || null;
        const inferred = inferredMotions.get(track) || null;
        const [effectiveFrom, effectiveTo] = applyInferredMotion(from, to, inferred);
        const objForNode = transitionNodeObject(effectiveFrom, effectiveTo);
        return { track, effectiveFrom, effectiveTo, objForNode };
      }).sort((a, b) => transitionRenderOrder(a.objForNode) - transitionRenderOrder(b.objForNode));
      const activeTracks = new Set(rows.map((row) => row.track));
      removeInactiveTransitionNodes(activeTracks);
      for (const row of rows) {
        ensureNode(row.objForNode);
      }
      for (const row of rows) {
        const { track, effectiveFrom, effectiveTo, objForNode } = row;
        const trackProgress = trackInterpolationProgress(raw, track, transition, captureOptions);
        const parentTrack = panelParentTrack(effectiveFrom, effectiveTo);
        const parentTrackProgress = trackProgress === null && parentTrack
          ? trackInterpolationProgress(raw, parentTrack, transition, captureOptions)
          : null;
        const eased = trackProgress ?? parentTrackProgress ?? baseProgress;
        const node = ensureNode(objForNode);
        attachNode(node, effectiveFrom, effectiveTo, activeTracks);
        node.style.display = "";
        applyState(node, effectiveFrom, effectiveTo, eased, true, transition);
      }
    };
    if (typeof fixedProgress === "number") {
      renderProgress(Math.max(0, Math.min(1, fixedProgress)));
      return;
    }
    if (duration <= 16) {
      showSlide(toIndex, { direction: toIndex < fromIndex ? "reverse" : "forward" });
      return;
    }
    transitioning = true;
    const started = performance.now();
    const step = (now) => {
      const raw = Math.min(1, (now - started) / duration);
      renderProgress(raw);
      if (raw < 1) {
        requestAnimationFrame(step);
      } else {
        transitioning = false;
        showSlide(toIndex, { direction: toIndex < fromIndex ? "reverse" : "forward" });
      }
    };
    requestAnimationFrame(step);
  }

  function transitionNodeObject(from, to) {
    if (!from || !to) return to || from;
    return transitionMediaObject(from, to) || to;
  }

  function inferredMotionMap(transition) {
    const map = new Map();
    for (const row of transition?.inferredMotions || []) {
      if (!row?.trackId) continue;
      if (transition?.reverse) {
        map.set(row.trackId, {
          ...row,
          fromGeometry: row.toGeometry,
          toGeometry: row.fromGeometry,
          source: `${row.source || "inferred"}-reverse`,
        });
      } else {
        map.set(row.trackId, row);
      }
    }
    return map;
  }

  function applyInferredMotion(from, to, inferred) {
    if (!inferred) return [from, to];
    if (!from && to && inferred.fromGeometry) {
      const syntheticFrom = structuredClone(to);
      syntheticFrom.geometry = structuredClone(inferred.fromGeometry);
      syntheticFrom.opacity = inferred.preserveOpacity ? (to.opacity ?? 1) : 0;
      syntheticFrom.inferredMotionSource = inferred.source || "inferred";
      return [syntheticFrom, to];
    }
    if (from && !to && inferred.toGeometry) {
      const syntheticTo = structuredClone(from);
      syntheticTo.geometry = structuredClone(inferred.toGeometry);
      syntheticTo.opacity = inferred.preserveOpacity ? (from.opacity ?? 1) : 0;
      syntheticTo.inferredMotionSource = inferred.source || "inferred";
      return [from, syntheticTo];
    }
    return [from, to];
  }

  function panelParentTrack(from, to) {
    return from?.parentTrackId || to?.parentTrackId || from?.panelParentTrackId || to?.panelParentTrackId || null;
  }

  function transitionRenderOrder(obj) {
    if (!obj) return 0;
    if (nodeRoleForObject(obj) === "group") return -200000 + Number(obj.z || 0);
    if (nodeRoleForObject(obj) === "panel") return -100000 + Number(obj.z || 0);
    return Number(obj.z || 0);
  }

  function transitionZIndex(state, isTransition, transition) {
    const base = Number(state?.renderZ ?? state?.panelClusterMaxZ ?? state?.z ?? 0);
    if (!isTransition || !transition) return Math.round(base);
    const override = transitionLayerOverride(transition);
    if (!override) return Math.round(base);
    const runtimeConfig = scene?.runtime || {};
    const track = String(state?.trackId || "");
    const decorativeTracks = new Set([
      ...((runtimeConfig.layerPolicy || {}).decorativeTracks || []),
      ...(override.decorativeTracks || []),
    ].map(String));
    if (decorativeTracks.has(track)) {
      return Math.round(base - Number(override.decorativeZDrop ?? 0));
    }
    if (shouldLiftPanelForTransition(state, override)) {
      return Math.round(base + Number(override.zBoost ?? 1000));
    }
    return Math.round(base);
  }

  function transitionLayerOverride(transition) {
    const runtimeConfig = scene?.runtime || {};
    const policy = runtimeConfig.layerPolicy || {};
    const rows = policy.transitionLayerOverrides || [];
    return rows.find((row) => (
      Number(row.from) === Number(transition.from)
      && Number(row.to) === Number(transition.to)
      && String(row.mode || "panels-above-decorative") === "panels-above-decorative"
    )) || null;
  }

  function shouldLiftPanelForTransition(state, override) {
    const panelTracks = new Set((override.panelTracks || []).map(String));
    const track = String(state?.trackId || "");
    const parentTrack = String(state?.parentTrackId || state?.panelParentTrackId || "");
    const panelCluster = String(state?.panelClusterId || "");
    const isPanelState = nodeRoleForObject(state) === "panel" || Boolean(panelCluster);
    if (!isPanelState) return false;
    if (!panelTracks.size) return true;
    return panelTracks.has(track) || panelTracks.has(parentTrack) || panelTracks.has(panelCluster);
  }

  function transitionMediaObject(from, to) {
    if (!from || !to) return to || from;
    const fromAsset = from.assetId ? assetById(from.assetId) : null;
    const toAsset = to.assetId ? assetById(to.assetId) : null;
    const sameAsset = from.assetId && from.assetId === to.assetId;
    if (fromAsset?.kind === "video" && sameAsset) {
      if (!isPausedMedia(to) && !isStateVisible(from) && hasExplicitMediaPhase(to) && !isAnimatedLoopAsset(fromAsset)) return to;
      if (isPausedMedia(from) && !isPausedMedia(to) && !isStateVisible(from) && !isAnimatedLoopAsset(fromAsset)) return to;
      if (!isPausedMedia(to) || isStateVisible(from) || isAnimatedLoopAsset(fromAsset)) return from;
    }
    if (fromAsset?.kind === "video" && (isStateVisible(from) || toAsset?.kind !== "video")) return from;
    if (toAsset?.kind === "video") return to;
    return null;
  }

  function hasExplicitMediaPhase(state) {
    const timing = state?.mediaTiming || {};
    return Number.isFinite(Number(timing.phaseSec));
  }

  function transitionMediaPhaseOverride(state, transition) {
    const rows = transition?.mediaPhaseOverrides || [];
    for (const row of rows) {
      if (row.trackId && row.trackId !== state.trackId) continue;
      if (row.objectId && row.objectId !== state.id) continue;
      if (row.assetId && row.assetId !== state.assetId) continue;
      if (row.name && row.name !== state.name) continue;
      const phase = Number(row.phaseSec ?? row.phase_sec);
      if (Number.isFinite(phase)) return phase;
    }
    return null;
  }

  function isAnimatedLoopAsset(asset) {
    if (!asset) return false;
    const source = `${asset.sourceFile || ""} ${asset.sourcePath || ""} ${asset.extension || ""}`.toLowerCase();
    return Boolean(asset.animated) || source.includes(".gif");
  }

  function isSettledOnlyRasterFallback(obj) {
    return Boolean(obj?.rasterFallback?.settledOnly);
  }

  function hideSettledOnlyFallbacks() {
    for (const node of nodes.values()) {
      if (node.dataset.settledOnly === "1") {
        pauseMedia(node);
        node.style.display = "none";
      }
    }
  }

  function preload(index) {
    const slide = scene.slides[index];
    if (!slide) return;
    for (const obj of slide.objects) {
      if (!obj.assetId) continue;
      const asset = assetById(obj.assetId);
      if (!asset || !asset.file) continue;
      if (asset.kind === "video") {
        const video = document.createElement("video");
        video.preload = "metadata";
        video.src = asset.file;
      } else {
        const image = new Image();
        image.src = asset.file;
      }
      if (obj.posterAssetId) {
        const posterAsset = assetById(obj.posterAssetId);
        if (posterAsset?.file) {
          const poster = new Image();
          poster.src = posterAsset.file;
        }
      }
    }
  }

  function updateHud() {
    titleEl.textContent = scene.deck.title || "";
    countEl.textContent = `${currentIndex + 1} / ${scene.slides.length}`;
    location.hash = `slide-${currentIndex + 1}`;
  }

  function toggleFullscreen() {
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else {
      document.documentElement.requestFullscreen().catch(() => {});
    }
  }

  function captureAt(slideNumber, progress = 1, options = null) {
    const direction = options?.direction === "reverse" ? "reverse" : "forward";
    const fromIndex = Math.max(0, Number(slideNumber) - 1);
    const toIndex = direction === "reverse"
      ? Math.max(0, fromIndex - 1)
      : Math.min(scene.slides.length - 1, fromIndex + 1);
    if (progress <= 0 || fromIndex === toIndex) {
      showSlide(fromIndex, { autoAdvance: false });
    } else if (progress >= 1) {
      showSlide(toIndex, { autoAdvance: false });
    } else {
      runTransition(fromIndex, toIndex, progress, options);
    }
    return Promise.resolve();
  }

  function exposeApi() {
    const api = {
      next: goNext,
      prev: goPrev,
      goTo,
      toggleFullscreen,
      captureAt,
      scene,
    };
    window.PptxHtmlPresenter = api;
    globalThis.PptxHtmlPresenter = api;
  }

  function installCaptureEvents() {
    document.addEventListener("pptx-html-presenter:capture-at", (event) => {
      const detail = event.detail || {};
      captureAt(Number(detail.slide || 1), Number(detail.progress ?? 1), detail);
    });
  }

  function applyInitialRoute() {
    const params = new URLSearchParams(location.search);
    const captureSlide = Number(params.get("captureSlide") || params.get("slide") || 0);
    const progressRaw = params.get("progress");
    if (captureSlide > 0) {
      const progress = progressRaw == null ? 0 : Number(progressRaw);
      captureAt(captureSlide, Number.isFinite(progress) ? progress : 0);
      return;
    }
    const hashMatch = String(location.hash || "").match(/slide-(\d+)/);
    if (hashMatch) {
      showSlide(Number(hashMatch[1]) - 1);
      return;
    }
    showSlide(0);
  }

  function showUi() {
    if (captureMode) return;
    document.body.classList.add("show-ui");
    clearTimeout(uiTimer);
    uiTimer = setTimeout(() => document.body.classList.remove("show-ui"), 2200);
  }

  function startMediaWatchdog() {
    if (window.__pptxHtmlPresenterMediaWatchdog) return;
    window.__pptxHtmlPresenterMediaWatchdog = window.setInterval(() => {
      if (!scene || transitioning) return;
      for (const node of nodes.values()) {
        const video = ownVideo(node);
        if (!video || video.dataset.pausedByState === "1") continue;
        if (!isNodeVisible(node)) continue;
        if (video.paused || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
          video.play().catch(() => {});
        }
      }
    }, 1200);
  }

  function playMedia(node) {
    const media = ownVideo(node);
    if (media) {
      media.dataset.pausedByState = "0";
      media.play().catch(() => {});
    }
  }

  function pauseMedia(node) {
    const media = ownVideo(node);
    if (media) {
      media.dataset.pausedByState = "1";
      media.pause();
    }
  }

  function syncMediaPlayback(node, state, isTransition) {
    const media = ownVideo(node);
    if (!media) return;
    if (isPausedMedia(state)) {
      seekMedia(media, mediaStartTime(state));
      media.dataset.pausedByState = "1";
      media.pause();
      node.dataset.mediaStarted = "0";
      node.dataset.mediaTimingObjectId = String(state.id || "");
      return;
    }
    if (!isSlideTimedMedia(state)) {
      const mediaSignature = mediaTimingSignature(state);
      if (node.dataset.mediaLoopSignature !== mediaSignature) {
        node.dataset.mediaLoopSignature = mediaSignature;
        seekMedia(media, mediaStartTime(state));
      }
      playMedia(node);
      return;
    }
    if (node.dataset.mediaTimingObjectId !== String(state.id || "")) {
      node.dataset.mediaTimingObjectId = String(state.id || "");
      node.dataset.mediaStarted = "0";
      seekMedia(media, mediaStartTime(state));
    }
    node.dataset.mediaStarted = "1";
    playMedia(node);
  }

  function mediaStartTime(state) {
    const timing = state.mediaTiming || {};
    let start = Number(timing.startSec || 0) + Number(timing.phaseSec || 0);
    if (isSlideTimedMedia(state)) {
      start += Number(scene.qa?.slideTimedVideoPhaseSec || 0);
    }
    return start;
  }

  function mediaTimingSignature(state) {
    const timing = state.mediaTiming || {};
    return [
      state.assetId || "",
      timing.kind || "loop",
      Number(timing.startSec || 0),
      Number(timing.phaseSec || 0),
    ].join("|");
  }

  function seekMedia(media, seconds) {
    const target = mediaLoopTime(media, seconds);
    try {
      media.currentTime = target;
      return;
    } catch {}
    media.dataset.pendingSeekSec = String(seconds);
    if (media.dataset.pendingSeekHandler === "1") return;
    media.dataset.pendingSeekHandler = "1";
    media.addEventListener("loadedmetadata", () => {
      media.dataset.pendingSeekHandler = "0";
      const pending = Number(media.dataset.pendingSeekSec || 0);
      try {
        media.currentTime = mediaLoopTime(media, pending);
      } catch {}
    }, { once: true });
  }

  function mediaLoopTime(media, seconds) {
    const desired = Number(seconds || 0);
    const duration = Number.isFinite(media.duration) && media.duration > 0.08 ? media.duration : 0;
    if (duration <= 0) return Math.max(0, desired);
    const wrapped = ((desired % duration) + duration) % duration;
    return Math.max(0, Math.min(wrapped, Math.max(0, duration - 0.05)));
  }

  function isSlideTimedMedia(state) {
    return (state.mediaTiming || {}).kind === "playFrom";
  }

  function isPausedMedia(state) {
    return Boolean((state.mediaTiming || {}).paused);
  }

  function isStateVisible(state) {
    const g = state.geometry || {};
    const left = Number(g.leftPct || 0);
    const top = Number(g.topPct || 0);
    const width = Number(g.widthPct || 0);
    const height = Number(g.heightPct || 0);
    const opacity = Number(state.opacity ?? 1);
    return opacity > 0.01 && left < 1 && top < 1 && (left + width) > 0 && (top + height) > 0;
  }

  function isNodeVisible(node) {
    if (node.style.display === "none") return false;
    const opacity = Number(node.style.opacity || 1);
    if (opacity <= 0.01) return false;
    const rect = node.getBoundingClientRect();
    const frameRect = frame.getBoundingClientRect();
    return rect.right > frameRect.left && rect.left < frameRect.right && rect.bottom > frameRect.top && rect.top < frameRect.bottom;
  }

  function ownVideo(node) {
    const child = visualChild(node);
    return child?.tagName === "VIDEO" ? child : null;
  }

  function cssColor(value) {
    if (!value) return "transparent";
    if (value.startsWith("scheme:")) {
      const key = value.slice("scheme:".length).toLowerCase();
      const map = {
        bg1: "#fff",
        lt1: "#fff",
        tx1: "#000",
        dk1: "#000",
        bg2: "#1f1f1f",
        tx2: "#f2f2f2",
        dk2: "#1f1f1f",
        lt2: "#f2f2f2",
        accent1: "#2f80ed",
        accent2: "#eb5757",
        accent3: "#27ae60",
        accent4: "#f2c94c",
        accent5: "#9b51e0",
        accent6: "#56ccf2",
      };
      return map[key] || "currentColor";
    }
    return value;
  }

  function outlineStyle() {
    const style = scene?.runtime?.outlineStyle || {};
    return {
      normalizeWhiteOutlines: style.normalizeWhiteOutlines !== false,
      borderOnTop: style.borderOnTop !== false,
      widthPct: Number(style.widthPct ?? 0.0055),
      minPx: Number(style.minPx ?? 3),
      maxPx: Number(style.maxPx ?? 7),
    };
  }

  function effectiveOutlineStroke(state) {
    if (state.stroke) return state.stroke;
    return null;
  }

  function normalizedStrokeWidthPct(state) {
    const style = outlineStyle();
    if (style.normalizeWhiteOutlines && isWhiteStroke(state.stroke)) {
      return Math.max(0.0001, Number.isFinite(style.widthPct) ? style.widthPct : 0.0055);
    }
    return Number(state.strokeWidthPct || 0);
  }

  function isWhiteStroke(value) {
    const raw = String(value || "").trim().toLowerCase();
    if (!raw) return false;
    if (["#fff", "#ffffff", "white", "rgb(255,255,255)", "rgb(255, 255, 255)"].includes(raw)) return true;
    return raw === "scheme:bg1" || raw === "scheme:lt1";
  }

  function cssStrokeWidth(state) {
    const style = outlineStyle();
    const pct = normalizedStrokeWidthPct(state);
    const px = Number(pct || 0) * frame.clientHeight;
    const minPx = style.normalizeWhiteOutlines && isWhiteStroke(state.stroke) ? style.minPx : 1;
    const maxPx = style.normalizeWhiteOutlines && isWhiteStroke(state.stroke) ? style.maxPx : 18;
    return Math.max(minPx, Math.min(maxPx, px || minPx)) + "px";
  }
  function cssShapeRadius(state) {
    const key = String(state.shape || "").toLowerCase();
    if (key.includes("ellipse")) return "50%";
    if (key.includes("round")) return "64px";
    return "0";
  }
  function needsSvgShape(shape) {
    const key = String(shape || "").toLowerCase();
    return key.includes("curveduparrow");
  }
  function createSvgShape(shape) {
    const key = String(shape || "").toLowerCase();
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("shape-svg");
    svg.dataset.shape = key;
    svg.setAttribute("viewBox", "0 0 100 100");
    svg.setAttribute("preserveAspectRatio", "none");
    if (key.includes("curveduparrow")) {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.classList.add("shape-path");
      path.setAttribute("d", "M 7 18 Q 42 88 86 26");
      path.setAttribute("fill", "none");
      path.setAttribute("stroke-linecap", "round");
      path.setAttribute("stroke-linejoin", "round");
      const head = document.createElementNS("http://www.w3.org/2000/svg", "path");
      head.classList.add("shape-head");
      head.setAttribute("d", "M 83 5 L 99 22 L 78 35 Z");
      svg.append(path, head);
    }
    return svg;
  }
  function applySvgShape(svg, state) {
    const key = String(state.shape || svg.dataset.shape || "").toLowerCase();
    const fill = cssColor(state.fill || state.stroke || "scheme:bg1");
    const stroke = cssColor(state.stroke || state.fill || "scheme:bg1");
    if (key.includes("curveduparrow")) {
      const path = svg.querySelector(".shape-path");
      const head = svg.querySelector(".shape-head");
      if (path) {
        path.setAttribute("stroke", fill);
        path.setAttribute("stroke-width", "16");
      }
      if (head) {
        head.setAttribute("fill", fill);
        head.setAttribute("stroke", stroke);
        head.setAttribute("stroke-width", state.stroke ? "1.2" : "0");
      }
      svg.style.overflow = "visible";
    }
  }
  function applyMediaCrop(child, crop) {
    const l = Number(crop?.l || 0);
    const r = Number(crop?.r || 0);
    const t = Number(crop?.t || 0);
    const b = Number(crop?.b || 0);
    const visibleW = Math.max(0.001, 1 - l - r);
    const visibleH = Math.max(0.001, 1 - t - b);
    child.style.left = `${(-l / visibleW) * 100}%`;
    child.style.top = `${(-t / visibleH) * 100}%`;
    child.style.width = `${100 / visibleW}%`;
    child.style.height = `${100 / visibleH}%`;
    child.style.clipPath = "none";
  }
  function applyMediaEffects(child, effects) {
    if (!child || !(child.tagName === "IMG" || child.tagName === "VIDEO")) return;
    const filters = [];
    const bc = effects?.brightnessContrast || null;
    if (bc) {
      const bright = Number(bc.bright || 0);
      const contrast = Number(bc.contrast || 0);
      if (bright >= 0.999) {
        filters.push("brightness(0)", "invert(1)");
      } else if (bright > 0) {
        filters.push(`brightness(${1 + bright})`);
      } else if (bright < 0) {
        filters.push(`brightness(${Math.max(0, 1 + bright)})`);
      }
      if (Math.abs(contrast) > 0.001) {
        filters.push(`contrast(${Math.max(0, 1 + contrast)})`);
      }
    }
    child.style.filter = filters.join(" ");
  }
  function applyTextStyle(child, style) {
    const size = Number(style.fontSizePt || 18);
    child.style.fontSize = cssFontSize(size);
    child.style.lineHeight = "1.08";
    child.style.fontWeight = style.bold ? "700" : "400";
    child.style.fontStyle = style.italic ? "italic" : "normal";
    child.style.color = cssColor(style.color || "scheme:bg1");
    child.style.textAlign = cssTextAlign(style.align);
    child.style.alignItems = cssFlexInlineAlign(style.align);
    child.style.justifyContent = cssFlexBlockAlign(style.anchor);
    child.style.padding = cssInsets(style.insets || {});
    child.classList.toggle("autofit", Boolean(style.autoFit));
    if (style.typeface) {
      child.style.fontFamily = `"${String(style.typeface).replaceAll('"', '\\"')}", sans-serif`;
    }
  }
  function renderText(child, state) {
    const richText = Array.isArray(state.richText) ? state.richText : [];
    const signature = JSON.stringify([state.text || "", richText, state.textStyle || {}]);
    if (child.dataset.textSignature === signature) return;
    child.dataset.textSignature = signature;
    child.replaceChildren();
    if (!richText.length) {
      child.textContent = state.text || "";
      return;
    }
    for (const paragraph of richText) {
      const line = document.createElement("div");
      line.className = "text-line";
      line.style.textAlign = cssTextAlign(paragraph.align || state.textStyle?.align);
      line.style.alignSelf = cssFlexInlineAlign(paragraph.align || state.textStyle?.align);
      for (const run of paragraph.runs || []) {
        const span = document.createElement("span");
        span.textContent = run.text || "";
        applyRunStyle(span, run.style || {}, state.textStyle || {});
        line.appendChild(span);
      }
      child.appendChild(line);
    }
  }
  function fitText(child, state) {
    if (!state.textStyle?.autoFit) return;
    const lines = Array.from(child.querySelectorAll(".text-line"));
    const targets = lines.length ? lines : [child];
    const style = getComputedStyle(child);
    const available = Math.max(1,
      child.clientWidth - parseFloat(style.paddingLeft || "0") - parseFloat(style.paddingRight || "0")
    );
    for (const target of targets) {
      target.style.transform = "";
      const natural = Math.max(target.scrollWidth || 0, target.getBoundingClientRect().width || 0);
      const scale = natural > available ? available / natural : 1;
      target.style.transform = scale < 1 ? `scale(${Math.max(0.1, scale)})` : "";
    }
  }
  function applyRunStyle(span, runStyle, fallbackStyle) {
    const style = { ...fallbackStyle, ...runStyle };
    if (style.fontSizePt) span.style.fontSize = cssFontSize(Number(style.fontSizePt));
    span.style.fontWeight = style.bold ? "700" : "400";
    span.style.fontStyle = style.italic ? "italic" : "normal";
    span.style.color = cssColor(style.color || fallbackStyle.color || "scheme:bg1");
    if (style.typeface) {
      span.style.fontFamily = `"${String(style.typeface).replaceAll('"', '\\"')}", sans-serif`;
    }
  }
  function cssFontSize(points) {
    const slideHeight = Number(scene.deck.slideSize.height || 6858000);
    const px = (Number(points || 18) * 12700 / slideHeight) * frame.clientHeight;
    return Math.max(6, px) + "px";
  }
  function cssInsets(insets) {
    const slideWidth = Number(scene.deck.slideSize.width || 12192000);
    const slideHeight = Number(scene.deck.slideSize.height || 6858000);
    const top = (Number(insets.tIns || 0) / slideHeight) * frame.clientHeight;
    const right = (Number(insets.rIns || 0) / slideWidth) * frame.clientWidth;
    const bottom = (Number(insets.bIns || 0) / slideHeight) * frame.clientHeight;
    const left = (Number(insets.lIns || 0) / slideWidth) * frame.clientWidth;
    return `${top}px ${right}px ${bottom}px ${left}px`;
  }
  function cssTextAlign(value) {
    const key = String(value || "ctr").toLowerCase();
    if (key === "l" || key === "left") return "left";
    if (key === "r" || key === "right") return "right";
    if (key === "just" || key === "justified") return "justify";
    return "center";
  }
  function cssFlexInlineAlign(value) {
    const key = String(value || "ctr").toLowerCase();
    if (key === "l" || key === "left") return "flex-start";
    if (key === "r" || key === "right") return "flex-end";
    return "center";
  }
  function cssFlexBlockAlign(value) {
    const key = String(value || "mid").toLowerCase();
    if (key === "t" || key === "top") return "flex-start";
    if (key === "b" || key === "bottom") return "flex-end";
    return "center";
  }

  function lerp(a, b, t) { return Number(a || 0) + ((Number(b || 0) - Number(a || 0)) * t); }
  function lerpAngle(a, b, t) {
    let diff = ((b - a + 540) % 360) - 180;
    return a + diff * t;
  }
  function easeForTransition(t, transition) {
    const runtime = scene.runtime || {};
    const easing = transition?.easing ?? runtime.easing ?? "easeInOutQuad";
    return applyEasing(t, easing);
  }
  function interpolationProgress(t, transition) {
    const mapped = mirroredProgressMapValue(t, transition?.progressMap || null, transition);
    if (mapped !== null) return mapped;
    return easeForTransition(t, transition);
  }
  function trackInterpolationProgress(t, trackId, transition, captureOptions) {
    const captureValue = captureTrackProgressValue(t, trackId, captureOptions);
    if (captureValue !== null) return captureValue;
    const rows = Array.isArray(transition?.trackProgressOverrides) ? transition.trackProgressOverrides : [];
    for (const row of rows) {
      if (row?.trackId && String(row.trackId) !== String(trackId)) continue;
      const mapped = mirroredProgressMapValue(t, row?.points || null, transition);
      if (mapped !== null) return mapped;
    }
    return null;
  }
  function captureTrackProgressValue(t, trackId, captureOptions) {
    const overrides = captureOptions?.trackProgressOverrides ?? captureOptions?.trackProgress ?? null;
    if (!overrides) return null;
    if (Array.isArray(overrides)) {
      for (const row of overrides) {
        if (row?.trackId && String(row.trackId) !== String(trackId)) continue;
        if (Number.isFinite(Number(row?.value))) return clamp01(Number(row.value));
        const mapped = progressMapValue(t, row?.points || null);
        if (mapped !== null) return mapped;
      }
      return null;
    }
    if (typeof overrides === "object") {
      const value = overrides[trackId];
      if (Number.isFinite(Number(value))) return clamp01(Number(value));
      if (Array.isArray(value)) return progressMapValue(t, value);
    }
    return null;
  }
  function progressMapValue(t, points) {
    if (!Array.isArray(points) || points.length < 2) return null;
    const x = clamp01(Number.isFinite(Number(t)) ? Number(t) : 0);
    const normalized = points
      .map((point) => ({
        progress: clamp01(Number(point.progress)),
        value: clamp01(Number(point.value ?? point.mappedProgress ?? point.interpolationProgress)),
      }))
      .filter((point) => Number.isFinite(point.progress) && Number.isFinite(point.value))
      .sort((a, b) => a.progress - b.progress);
    if (normalized.length < 2) return null;
    if (x <= normalized[0].progress) return normalized[0].value;
    for (let i = 1; i < normalized.length; i += 1) {
      const prev = normalized[i - 1];
      const next = normalized[i];
      if (x <= next.progress) {
        const span = Math.max(0.0001, next.progress - prev.progress);
        return lerp(prev.value, next.value, (x - prev.progress) / span);
      }
    }
    return normalized[normalized.length - 1].value;
  }
  function mirroredProgressMapValue(t, points, transition) {
    if (transition?.reverse) {
      const mapped = progressMapValue(1 - t, points);
      return mapped === null ? null : 1 - mapped;
    }
    return progressMapValue(t, points);
  }
  function applyEasing(t, easing) {
    const x = clamp01(Number.isFinite(Number(t)) ? Number(t) : 0);
    if (typeof easing === "object" && easing) {
      if (easing.type === "power") return powerEase(x, Number(easing.exponent || 1));
      if (Array.isArray(easing.cubicBezier)) return cubicBezierEase(x, easing.cubicBezier);
    }
    const key = String(easing || "easeInOutQuad").trim();
    const lower = key.toLowerCase();
    if (lower === "linear") return x;
    if (lower === "easeinoutquad" || lower === "quad") return easeInOutQuad(x);
    const power = lower.match(/^power[:(]?\s*([0-9.]+)\)?$/);
    if (power) return powerEase(x, Number(power[1]));
    const bezier = key.match(/^cubic-bezier\(\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\)$/i);
    if (bezier) {
      return cubicBezierEase(x, bezier.slice(1).map(Number));
    }
    return easeInOutQuad(x);
  }
  function easeInOutQuad(t) {
    return t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }
  function powerEase(t, exponent) {
    const exp = Number.isFinite(exponent) && exponent > 0 ? exponent : 1;
    return Math.pow(t, exp);
  }
  function cubicBezierEase(t, points) {
    if (!points || points.length !== 4 || points.some((value) => !Number.isFinite(Number(value)))) {
      return easeInOutQuad(t);
    }
    const [x1, y1, x2, y2] = points.map(Number);
    const sampleX = (u) => bezierCoord(u, x1, x2);
    let lo = 0;
    let hi = 1;
    let u = t;
    for (let i = 0; i < 12; i += 1) {
      const x = sampleX(u);
      const dx = bezierDerivative(u, x1, x2);
      if (Math.abs(x - t) < 0.0005) break;
      if (Math.abs(dx) < 0.0001) break;
      u = clamp01(u - (x - t) / dx);
    }
    if (Math.abs(sampleX(u) - t) > 0.001) {
      lo = 0;
      hi = 1;
      for (let i = 0; i < 16; i += 1) {
        u = (lo + hi) / 2;
        if (sampleX(u) < t) lo = u;
        else hi = u;
      }
    }
    return clamp01(bezierCoord(u, y1, y2));
  }
  function bezierCoord(t, p1, p2) {
    const inv = 1 - t;
    return (3 * inv * inv * t * p1) + (3 * inv * t * t * p2) + (t * t * t);
  }
  function bezierDerivative(t, p1, p2) {
    const inv = 1 - t;
    return (3 * inv * inv * p1) + (6 * inv * t * (p2 - p1)) + (3 * t * t * (1 - p2));
  }
  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "ArrowRight" || event.key === "PageDown" || event.key === " ") {
      event.preventDefault();
      goNext();
    } else if (event.key === "ArrowLeft" || event.key === "PageUp") {
      event.preventDefault();
      goPrev();
    } else if (event.key.toLowerCase() === "f") {
      toggleFullscreen();
    }
    showUi();
  });
  document.addEventListener("click", (event) => {
    if (event.target === fullscreenBtn) return;
    goNext();
    showUi();
  });
  document.addEventListener("mousemove", showUi);
  fullscreenBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    toggleFullscreen();
  });
})();
</script>
</body>
</html>
"""
