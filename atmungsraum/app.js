import { participantIdentity, savedBrowserExports } from './data-logging.js?v=20261113-polar-only';

const byId = id => document.getElementById(id);
const root = document.documentElement;
let questPair;
try { questPair = JSON.parse(localStorage.getItem('atmungsraum-quest-pair')); }
catch { /* Pairing can still start without browser storage. */ }

async function showSavedBrowserExports() {
  const status = byId('data-status');
  try {
    const records = await savedBrowserExports();
    for (const record of records) {
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.download = record.filename;
      link.href = URL.createObjectURL(new Blob([record.csv], { type: 'text/csv;charset=utf-8' }));
      link.textContent = record.filename;
      link.dataset.measure = '';
      item.append(link);
      byId('data-files').append(item);
    }
    status.textContent = records.length ? 'CSV-Datei zum Herunterladen auswählen.' : 'Keine früheren Browser-Aufnahmen gespeichert.';
  } catch {
    status.textContent = 'Frühere Browser-Aufnahmen konnten nicht geladen werden.';
  }
}

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
          return { element, fits, domOnly };
        });
        results.forEach(({ element, fits, domOnly }) => {
          element.dataset.textFit = fits ? 'fit' : 'reflow';
          element.dataset.textMeasurement = domOnly ? 'dom' : 'pretext';
        });
        const visible = document.querySelector('.screen:not([hidden])');
        if (!visible) return;
        // Equal outer tracks keep the main controls at the viewport midpoint.
        const headerHeight = visible.firstElementChild.getBoundingClientRect().height;
        const footerHeight = visible.querySelector('.actions')?.getBoundingClientRect().height ?? 0;
        visible.style.setProperty('--edge-size', `${Math.max(headerHeight, footerHeight)}px`);
        const groups = [document.querySelector('main'), visible, ...visible.querySelectorAll('.screen-body, .actions, .patient-fields, .gender-options, #data-files, [data-measure]')].filter(element => element.getClientRects().length);
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
    document.querySelectorAll('[data-measure], .screen-body, .actions').forEach((element) => observer.observe(element));
    const retry = () => { root.dataset.layout = 'bounded'; measure(); };
    new MutationObserver(retry).observe(document.querySelector('main'), { childList: true, characterData: true, subtree: true });
    const dialogChanges = new MutationObserver(measure);
    for (const id of ['install-dialog']) {
      const dialog = byId(id);
      dialogChanges.observe(dialog, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['open'] });
      observer.observe(dialog);
    }
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

const recovery = location.hash === '#daten';
byId('quest').hidden = recovery;
byId('daten').hidden = !recovery;
measureLabels();
if (recovery) {
  void showSavedBrowserExports();
} else {
  import('./quest-ui.js?v=24').then(({ mountQuestRehearsal }) => mountQuestRehearsal(participantIdentity, questPair)).catch(() => {
    byId('quest-status').textContent = 'Die Quest-Steuerung konnte nicht geladen werden.';
  });
}
