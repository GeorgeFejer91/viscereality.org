import { state, hasSignal, canNavigate, confirmECG, setSensor, updateReconnect, route } from './ui-logic.js';
import { mountSessionAudio } from './audio/session-ui.js';
const byId = (id) => document.getElementById(id);
const root = document.documentElement;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
let current = 'h10';
let attempt = 0;
const text = (id, value) => { if (byId(id).textContent !== value) byId(id).textContent = value; };
const sessionAudio = mountSessionAudio();

function render() {
  const next = route(location.hash.slice(1) || 'h10');
  if (next !== 'session-start' && location.hash !== `#${next}`) history.replaceState(null, '', `#${next}`);
  if (next === 'session-start' && current !== 'session-start') state.started = false;
  current = next;
  document.querySelectorAll('.screen').forEach((screen) => { screen.hidden = screen.id !== next; });
  root.dataset.layout = root.dataset.measurement === 'unavailable' ? 'reflow' : 'bounded';
  const labels = { searching: 'Suche Polar … (Demo)', found: 'Polar H10 gefunden (Demo)', connecting: 'Verbinde Polar … (Demo)', ecg: 'ECG wird aufgezeichnet (Demo)', 'no-ecg': 'Warte auf ECG … (Demo)', disconnected: 'Polar getrennt (Demo)' };
  text('sensor-status', labels[state.sensor]);
  byId('select-polar').hidden = state.sensor !== 'found';
  document.querySelector('.ecg-box').hidden = !hasSignal();
  byId('sensor-next').setAttribute('aria-disabled', String(!hasSignal()));
  byId('sensor-next').tabIndex = hasSignal() ? 0 : -1;
  if (state.selection) {
    const { branch, value } = state.selection;
    text('session-title', `${branch === 'probe' ? 'Probe-Runde' : 'Haupt-Runde'} · ${value === 'anleitung' ? 'Anleitung ohne Zeitlimit' : `${value} Minuten`}`);
    byId('session-back').href = `#${branch}-runde`;
  }
  sessionAudio.update(next, state.selection, canNavigate());
  if (!hasSignal()) byId('play').disabled = true;
  drawECG();
  byId(next).querySelector('h1').focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

window.addEventListener('hashchange', render);
byId('sensor-next').addEventListener('click', (event) => { if (!confirmECG()) event.preventDefault(); });

// The future headset transport reports signal state here. No hardware feed is implemented yet.
window.addEventListener('atmungsraum:sensor-state', (event) => {
  if (!['ecg', 'no-ecg', 'disconnected'].includes(event.detail?.state)) return;
  setSensor(event.detail.state);
  render();
});

function searchPolar() {
  const pending = ++attempt;
  setSensor('searching'); render();
  setTimeout(() => {
    if (pending !== attempt) return;
    setSensor('found'); render();
  }, 700);
}

byId('connect').addEventListener('click', searchPolar);
byId('select-polar').addEventListener('click', () => {
  if (state.sensor !== 'found') return;
  const pending = ++attempt;
  setSensor('connecting'); render();
  setTimeout(() => {
    if (pending !== attempt) return;
    setSensor('no-ecg'); render();
    setTimeout(() => {
      if (pending !== attempt) return;
      setSensor('ecg'); render();
    }, 700);
  }, 650);
});
const canvas = byId('ecg');
const ctx = canvas.getContext('2d');
function drawECG() {
  if (current !== 'h10' || !ctx) return;
  const { width, height } = canvas.getBoundingClientRect();
  const ratio = Math.min(devicePixelRatio || 1, 2);
  const pixelWidth = Math.max(1, Math.round(width * ratio));
  const pixelHeight = Math.max(1, Math.round(height * ratio));
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) { canvas.width = pixelWidth; canvas.height = pixelHeight; }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#222'; ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 24) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
  for (let y = 0; y < height; y += 24) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
  if (!hasSignal() || document.hidden) return; // Missing data clears the old trace immediately.
  const phase = reducedMotion.matches ? 0 : Date.now() / 1000;
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.beginPath();
  const pulse = (t, center, spread, size) => size * Math.exp(-(((t - center) / spread) ** 2));
  for (let x = 0; x < width; x++) {
    const t = (x / 120 + phase) % 1;
    const value = pulse(t, .18, .045, .1) - pulse(t, .37, .014, .15) + pulse(t, .4, .014, .9) - pulse(t, .43, .018, .25) + pulse(t, .67, .07, .2);
    const y = height * .6 - value * height * .45;
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// Text measurement is an enhancement; the UI state machine remains usable if it fails.
async function measureLabels() {
  try {
    const { prepareWithSegments, measureLineStats, measureNaturalWidth, setLocale } = await import('./vendor/pretext/layout.js');
    await document.fonts.load('16px Arial');
    await document.fonts.ready;
    setLocale(root.lang);
    const cache = new Map();
    let frame;
    const measure = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const results = [...document.querySelectorAll('[data-measure]')].filter((element) => element.getClientRects().length).map((element) => {
          const style = getComputedStyle(element);
          const value = element.textContent.trim();
          const font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
          const letterSpacing = parseFloat(style.letterSpacing) || 0;
          const key = JSON.stringify([value, font, letterSpacing, style.whiteSpace, style.wordBreak]);
          if (!cache.has(key)) {
            if (cache.size >= 128) cache.clear();
            cache.set(key, prepareWithSegments(value, font, { letterSpacing, whiteSpace: style.whiteSpace, wordBreak: style.wordBreak }));
          }
          const prepared = cache.get(key);
          const width = element.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
          const height = element.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
          const lineHeight = parseFloat(style.lineHeight);
          const stats = measureLineStats(prepared, Math.max(1, width - 2));
          const fits = stats.maxLineWidth <= width + 1 && stats.lineCount * lineHeight <= height + 2;
          return { element, fits, square: element.classList.contains('round-description') ? null : element.closest('.round'), wrap: measureNaturalWidth(prepared) > width - 2 || !fits };
        });
        results.forEach(({ element, fits, square, wrap }) => {
          element.dataset.textFit = fits ? 'fit' : 'reflow';
          if (square) square.dataset.textFit = wrap ? 'reflow' : 'fit';
        });
        const visible = byId(current);
        const groups = [...visible.querySelectorAll('.screen-body, .actions, [data-measure]')];
        const bounds = visible.getBoundingClientRect();
        const footer = visible.querySelector('.actions');
        const bodyLimit = footer ? footer.getBoundingClientRect().top - 2 : bounds.bottom + 2;
        const hasOverflow = groups.some((group) => {
          const box = group.getBoundingClientRect();
          return group.scrollWidth > group.clientWidth + 2 || group.scrollHeight > group.clientHeight + 2 || box.bottom > bounds.bottom + 2;
        }) || [...visible.querySelector('.screen-body').children].some((group) => group.getBoundingClientRect().bottom > bodyLimit);
        if (root.dataset.layout !== 'reflow' && (hasOverflow || results.some(({ fits }) => !fits))) root.dataset.layout = 'reflow';
        root.dataset.measurement = 'ready';
      });
    };
    const observer = new ResizeObserver(measure);
    document.querySelectorAll('[data-measure], .screen-body, .actions, .round').forEach((element) => observer.observe(element));
    new MutationObserver(measure).observe(document.querySelector('main'), { childList: true, characterData: true, subtree: true });
    window.addEventListener('resize', () => { root.dataset.layout = 'bounded'; measure(); });
    window.addEventListener('hashchange', measure);
    measure();
  } catch (error) {
    root.dataset.measurement = 'unavailable';
    root.dataset.layout = 'reflow';
    console.warn('Text measurement unavailable; using readable document flow.', error);
  }
}

searchPolar();
measureLabels();
setInterval(() => {
  const reconnect = updateReconnect();
  if (reconnect === 'attempt') window.dispatchEvent(new CustomEvent('atmungsraum:reconnect-request', { detail: { attempt: state.reconnectAttempts } }));
  else if (reconnect === 'expired') { history.replaceState(null, '', '#h10'); render(); }
  drawECG();
}, 100);
window.addEventListener('resize', drawECG);
