import { state, hasSignal, canNavigate, confirmECG, confirmPatient, setSensor, updateReconnect, route } from './ui-logic.js?v=20261106-polar-timeout';
import { mountSessionAudio } from './audio/session-ui.js?v=20261106-polar-timeout';
import { MockECG } from './mock-ecg.js?v=20261105-data';
import { participantIdentity, mountSessionData } from './data-logging.js?v=20261107-file-delivery';
import { mountDataFiles } from './data-files.js?v=20261107-checkout-receipts';
const byId = (id) => document.getElementById(id);
const root = document.documentElement;
const rehearsalMode = new URLSearchParams(location.search).get('mode') === 'quest';
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
let current = null;
let attempt = 0;
let connectedOnce = false;
let connectionError = '';
const text = (id, value) => { if (byId(id).textContent !== value) byId(id).textContent = value; };
const signal = rehearsalMode ? null : new MockECG();
const sessionData = signal && mountSessionData(signal);
const sessionAudio = sessionData && mountSessionAudio({ data: sessionData });
if (sessionData) mountDataFiles(sessionData);

function render() {
  if (rehearsalMode) {
    if (current === 'quest') return;
    current = 'quest';
    root.dataset.mode = 'quest';
    document.querySelectorAll('.screen').forEach((screen) => { screen.hidden = screen.id !== current; });
    byId('quest-title').focus({ preventScroll: true });
    return;
  }
  const next = route(location.hash.slice(1) || 'h10');
  const changed = next !== current;
  if (next !== 'session-start' && location.hash !== `#${next}`) history.replaceState(null, '', `#${next}`);
  if (next === 'session-start' && current !== 'session-start') state.started = false;
  current = next;
  document.querySelectorAll('.screen').forEach((screen) => { screen.hidden = screen.id !== next; });
  root.dataset.layout = root.dataset.measurement === 'unavailable' ? 'reflow' : 'bounded';
  const labels = { searching: 'Suche Polar … (Demo)', found: 'Polar H10 gefunden (Demo)', connecting: 'Verbinde Polar … (Demo)', ecg: 'ECG-Demo aktiv (NeuroKit2)', 'no-ecg': 'Warte auf ECG … (Demo)', disconnected: 'Polar getrennt (Demo)' };
  text('sensor-status', connectionError || labels[state.sensor]);
  if (hasSignal()) connectedOnce = true;
  byId('belt-signal').dataset.state = hasSignal() ? 'connected' : connectedOnce ? 'lost' : 'waiting';
  const beltStatus = hasSignal() ? 'ECG-Demosignal aktiv' : connectedOnce ? 'ECG-Demosignal verloren' : 'Noch kein ECG-Demosignal';
  text('belt-status', beltStatus);
  byId('belt-signal').title = beltStatus;
  byId('select-polar').hidden = state.sensor !== 'found';
  byId('connect').hidden = !['no-ecg', 'disconnected'].includes(state.sensor);
  document.querySelector('.ecg-box').hidden = !hasSignal();
  byId('sensor-next').setAttribute('aria-disabled', String(!hasSignal()));
  byId('sensor-next').tabIndex = hasSignal() ? 0 : -1;
  if (state.selection) {
    const { branch, value } = state.selection;
    text('session-title', `${branch === 'probe' ? 'Probe-Runde' : 'Haupt-Runde'} · ${value === 'anleitung' ? 'Anleitung ohne Zeitlimit' : `${value} Minuten`}`);
    byId('session-back').href = branch === 'haupt' ? '#runde' : '#probe-runde';
  }
  sessionAudio.update(next, state.selection, canNavigate());
  drawECG();
  if (changed) {
    byId(next).querySelector('h1').focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }
}

window.addEventListener('hashchange', render);
byId('sensor-next').addEventListener('click', (event) => { if (!confirmECG()) event.preventDefault(); });
byId('patient-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const details = Object.fromEntries(new FormData(event.currentTarget));
  if (confirmPatient(details)) {
    location.hash = 'h10';
    searchPolar();
  }
});
byId('patient-form').addEventListener('input', (event) => {
  if (event.target.id === 'patient-surname') event.target.value = event.target.value.normalize('NFC');
  try { text('patient-id', participantIdentity(Object.fromEntries(new FormData(byId('patient-form')))).id); }
  catch { text('patient-id', '—'); }
});

// The future headset transport reports signal state here. No hardware feed is implemented yet.
window.addEventListener('atmungsraum:sensor-state', (event) => {
  if (rehearsalMode) return;
  if (!['ecg', 'no-ecg', 'disconnected'].includes(event.detail?.state)) return;
  ++attempt;
  connectionError = '';
  const reconnect = setSensor(event.detail.state === 'ecg' && !signal.samples ? 'no-ecg' : event.detail.state);
  if (reconnect === 'expired') history.replaceState(null, '', '#h10');
  render();
});

function searchPolar() {
  const pending = ++attempt;
  connectionError = '';
  setSensor('searching'); render();
  Promise.all([sessionData.prepare(), signal.prepare(), new Promise(resolve => setTimeout(resolve, 700))]).then(() => {
    if (pending !== attempt) return;
    setSensor('found'); render();
  }).catch(error => {
    if (pending !== attempt) return;
    connectionError = 'Demo-Daten konnten nicht geladen werden. Erneut suchen.';
    setSensor('disconnected'); render();
    console.warn('Demo preparation failed.', error);
  });
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
  if (!signal.samples) return;
  const end = signal.indexAt(reducedMotion.matches ? signal.epoch + 4000 : performance.now());
  const span = 4 * signal.rate;
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.beginPath();
  for (let x = 0; x < width; x++) {
    const value = signal.sampleAt(end - span + Math.floor(x / width * span));
    const y = height * .6 - value * height * .45;
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// Text measurement is an enhancement; the UI state machine remains usable if it fails.
async function measureLabels() {
  try {
    if (typeof Intl.Segmenter !== 'function' || !document.createElement('canvas').getContext('2d')) {
      throw new Error('Pretext requires text segmentation and Canvas 2D');
    }
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
          const key = JSON.stringify([root.lang, value, font, letterSpacing, style.whiteSpace, style.wordBreak]);
          if (!cache.has(key)) {
            if (cache.size >= 128) cache.clear();
            cache.set(key, prepareWithSegments(value, font, { letterSpacing, whiteSpace: style.whiteSpace, wordBreak: style.wordBreak }));
          }
          const prepared = cache.get(key);
          const width = element.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
          const height = element.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
          const lineHeight = parseFloat(style.lineHeight);
          const stats = measureLineStats(prepared, Math.max(1, width - 2));
          const naturalWidth = measureNaturalWidth(prepared);
          const singleLine = root.dataset.layout !== 'reflow' && !element.closest('dialog') && element.closest('button, .actions a');
          // User word spacing is outside Pretext's model; the painted DOM owns that check.
          const domOnly = (parseFloat(style.wordSpacing) || 0) !== 0 || style.hyphens === 'auto';
          const range = document.createRange();
          range.selectNodeContents(element);
          const box = element.getBoundingClientRect();
          const painted = [...range.getClientRects()];
          const domFits = painted.every(rect => rect.left >= box.left + parseFloat(style.paddingLeft) - 2
            && rect.right <= box.right - parseFloat(style.paddingRight) + 2
            && rect.top >= box.top + parseFloat(style.paddingTop) - 2
            && rect.bottom <= box.bottom - parseFloat(style.paddingBottom) + 2);
          const fits = domFits && (domOnly || (stats.maxLineWidth <= width + 1 && stats.lineCount * lineHeight <= height + 2
            && (!singleLine || (stats.lineCount === 1 && naturalWidth <= width - 2 && lineHeight <= height + 2))));
          return { element, fits, domOnly, square: element.closest('.round'), wrap: naturalWidth > width - 2 || !fits };
        });
        results.forEach(({ element, fits, domOnly, square, wrap }) => {
          element.dataset.textFit = fits ? 'fit' : 'reflow';
          element.dataset.textMeasurement = domOnly ? 'dom' : 'pretext';
          if (square) square.dataset.textFit = wrap ? 'reflow' : 'fit';
        });
        const visible = byId(current);
        // Equal outer tracks keep the main controls at the viewport midpoint.
        const headerHeight = visible.firstElementChild.getBoundingClientRect().height;
        const footerHeight = visible.querySelector('.actions')?.getBoundingClientRect().height ?? 0;
        visible.style.setProperty('--edge-size', `${Math.max(headerHeight, footerHeight)}px`);
        const groups = [document.querySelector('main'), visible, ...visible.querySelectorAll('.screen-body, .actions, .patient-fields, .gender-options, .round, .option, #data-files, [data-measure]')].filter(element => element.getClientRects().length);
        const bounds = visible.getBoundingClientRect();
        const footer = visible.querySelector('.actions');
        const bodyLimit = footer ? footer.getBoundingClientRect().top - 2 : bounds.bottom + 2;
        const hasOverflow = groups.some((group) => {
          const box = group.getBoundingClientRect();
          return group.scrollWidth > group.clientWidth + 2 || group.scrollHeight > group.clientHeight + 2
            || (group !== document.querySelector('main') && (box.top < bounds.top - 2 || box.bottom > bounds.bottom + 2))
            || box.left < -2 || box.right > innerWidth + 2;
        }) || [...visible.querySelector('.screen-body').children].some((group) => group.getBoundingClientRect().bottom > bodyLimit)
          || root.scrollWidth > innerWidth + 2 || root.scrollHeight > innerHeight + 2;
        if (root.dataset.layout !== 'reflow') {
          root.dataset.layoutReason = hasOverflow ? 'geometry' : results.some(({ fits, element }) => !fits && !element.closest('dialog')) ? 'text' : 'fit';
          if (root.dataset.layoutReason !== 'fit') {
            root.dataset.layout = 'reflow';
            measure(); // Recheck wrapped labels even when reflow leaves their box unchanged.
          }
        }
        root.dataset.measurement = 'ready';
      });
    };
    const observer = new ResizeObserver(measure);
    document.querySelectorAll('[data-measure], .screen-body, .actions, .round, .play-feedback').forEach((element) => observer.observe(element));
    const retry = () => { root.dataset.layout = 'bounded'; measure(); };
    new MutationObserver(retry).observe(document.querySelector('main'), { childList: true, characterData: true, subtree: true });
    const dialog = byId('files-dialog');
    new MutationObserver(measure).observe(dialog, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['open'] });
    observer.observe(dialog);
    new MutationObserver(() => { setLocale(root.lang); cache.clear(); retry(); }).observe(root, { attributes: true, attributeFilter: ['lang', 'style'] });
    window.addEventListener('resize', retry);
    document.fonts.addEventListener('loadingdone', retry);
    window.addEventListener('hashchange', measure);
    measure();
  } catch (error) {
    root.dataset.measurement = 'unavailable';
    root.dataset.layout = 'reflow';
    root.dataset.layoutReason = 'measurement-unavailable';
    console.warn('Text measurement unavailable; using readable document flow.', error);
  }
}

render();
measureLabels();
if (rehearsalMode) {
  import('./quest-ui.js?v=1').then(({ mountQuestRehearsal }) => mountQuestRehearsal()).catch(() => {
    text('quest-status', 'Der VR-Probelauf konnte nicht geladen werden.');
  });
} else {
  byId('patient-fields').disabled = false;
  byId('patient-next').disabled = false;
  // Load saved CSVs even on a cold recovery link; sensor search waits for the form.
  sessionData.prepare().catch(() => {});
  setInterval(() => {
    const reconnect = updateReconnect();
    if (reconnect === 'attempt') window.dispatchEvent(new CustomEvent('atmungsraum:reconnect-request', { detail: { attempt: state.reconnectAttempts } }));
    else if (reconnect === 'expired') { history.replaceState(null, '', '#h10'); render(); }
    drawECG();
  }, 100);
  window.addEventListener('resize', drawECG);
}
