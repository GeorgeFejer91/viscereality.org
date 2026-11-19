import { participantIdentity, savedBrowserExports } from './data-logging.js?v=20261118-questionnaire-completion';

const byId = id => document.getElementById(id);
const root = document.documentElement;
let questPair;
try { questPair = JSON.parse(localStorage.getItem('atmungsraum-quest-pair')); }
catch { /* Pairing can still start without browser storage. */ }

async function showSavedBrowserExports() {
  const status = byId('data-status');
  const select = byId('browser-archives');
  const filename = byId('browser-archive-name');
  const download = byId('browser-export-download');
  try {
    const records = await savedBrowserExports();
    if (records.length) {
      const date = new Intl.DateTimeFormat('de-DE', { day: '2-digit', month: '2-digit', year: '2-digit' });
      for (const [index, record] of records.entries()) {
        const created = new Date(record.created_at);
        const number = records.length - index;
        select.add(new Option(Number.isFinite(created.getTime())
          ? `${date.format(created)} · ${number}` : `Aufnahme ${number}`, index));
      }
      let url;
      const updateDownload = () => {
        if (url) URL.revokeObjectURL(url);
        const record = records[select.selectedIndex];
        url = URL.createObjectURL(new Blob([record.csv], { type: 'text/csv;charset=utf-8' }));
        filename.textContent = record.filename;
        download.download = record.filename;
        download.href = url;
      };
      select.addEventListener('change', updateDownload);
      updateDownload();
      byId('data-files').hidden = false;
    }
    status.textContent = records.length ? '' : 'Keine früheren Browser-Aufnahmen gespeichert.';
    status.hidden = records.length > 0;
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
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = undefined;
        const results = [...document.querySelectorAll('[data-measure]')].filter((element) => element.getClientRects().length).map((element) => {
          const style = getComputedStyle(element);
          const value = element.dataset.measureSample ?? element.textContent.trim();
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
          const stats = measureLineStats(prepared, Math.max(1, width));
          const naturalWidth = measureNaturalWidth(prepared);
          const singleLine = style.whiteSpace === 'nowrap';
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
        const groups = [document.querySelector('main'), visible, ...visible.querySelectorAll('.screen-body, .actions, .patient-fields, .patient-row, .session-tiles, .session-tile, #data-files, [data-measure], aside, details, .operator-common, #quest-data')].filter(element => element.getClientRects().length);
        const survey = visible.querySelector('#questionnaire-form');
        const bounds = visible.getBoundingClientRect();
        const footer = visible.querySelector(':scope > .actions');
        const bodyLimit = footer ? footer.getBoundingClientRect().top - 2 : bounds.bottom + 2;
        const hasOverflow = groups.some((group) => {
          const box = group.getBoundingClientRect();
          return group.scrollWidth > group.clientWidth + 2
            || group !== survey && group.scrollHeight > group.clientHeight + 2
            || group !== document.querySelector('main') && !survey?.contains(group)
              && (box.top < bounds.top - 2 || box.bottom > bounds.bottom + 2)
            || box.left < -2 || box.right > innerWidth + 2;
        }) || !survey && [...visible.querySelector('.screen-body').children].some((group) => group.getBoundingClientRect().bottom > bodyLimit)
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
    // A temporary bounded layout can clamp scrollTop before reflow returns.
    // Telemetry changes only remeasure; reconsider compact layout on navigation
    // or when the viewport/font changes outside an active scroll or editor.
    const retry = () => {
      if (!window.scrollY && !document.activeElement?.matches('input, select, textarea')) root.dataset.layout = 'bounded';
      measure();
    };
    const main = document.querySelector('main');
    new MutationObserver(records => {
      // The bounded 0–100% field is measured against its largest label once.
      // Changing its digits cannot change geometry or require a page-wide pass.
      if (records.some(record => {
        if (record.type === 'attributes') return record.oldValue !== record.target.getAttribute(record.attributeName);
        const element = record.target.nodeType === Node.TEXT_NODE ? record.target.parentElement : record.target;
        return !element?.closest?.('[data-live-value]');
      })) measure();
    }).observe(main, { childList: true, characterData: true, subtree: true,
      attributes: true, attributeFilter: ['hidden', 'data-view'], attributeOldValue: true });
    const screenChanges = new MutationObserver(() => { root.dataset.layout = 'bounded'; measure(); });
    main.querySelectorAll('.screen').forEach((screen) => {
      screenChanges.observe(screen, { attributes: true, attributeFilter: ['hidden'] });
    });
    const dialogChanges = new MutationObserver(records => {
      if (records.some(record => record.type !== 'attributes'
        || record.oldValue !== record.target.getAttribute(record.attributeName))) measure();
    });
    for (const dialog of document.querySelectorAll('dialog')) {
      dialogChanges.observe(dialog, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['open', 'hidden'], attributeOldValue: true });
      observer.observe(dialog);
    }
    new MutationObserver(() => { setLocale(root.lang); cache.clear(); retry(); }).observe(root, { attributes: true, attributeFilter: ['lang', 'style'] });
    const viewport = window.visualViewport;
    const resize = () => {
      const size = `${Math.floor(viewport?.scale === 1 ? viewport.height : innerHeight)}px`;
      if (root.style.getPropertyValue('--usable-height') !== size) root.style.setProperty('--usable-height', size);
      retry();
    };
    window.addEventListener('resize', resize);
    viewport?.addEventListener('resize', resize);
    resize();
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

let questMounted = false;
let refreshingIdlePage = false;
let idleRefreshRequested = false;
function mountOperator() {
  if (questMounted) return;
  questMounted = true;
  import('./quest-ui.js?v=20261120-polar-continuity').then(({ mountQuestRehearsal }) =>
    mountQuestRehearsal(participantIdentity, questPair, () => {
      // A saved search can be busy at the refresh timer's exact deadline.
      // Give the requested check the next idle window before searching again.
      if (idleRefreshRequested) void refreshIdlePage();
      return !refreshingIdlePage;
    })).catch(() => {
    byId('quest-status').textContent = 'Die Quest-Steuerung konnte nicht geladen werden.';
  });
}

async function replaceWithPublishedRelease(canReplace = () => true) {
  const currentEntry = location.href;
  const entry = new URL(currentEntry);
  if (entry.searchParams.has('_pwa')) {
    // This navigation was just checked by the previous document.
    entry.searchParams.delete('_pwa');
    history.replaceState(history.state, '', entry);
    return false;
  }
  entry.searchParams.set('_pwa', `${Date.now()}-${Math.random().toString(36).slice(2)}`);
  const controller = new AbortController();
  const deadline = setTimeout(() => controller.abort(), 2000);
  try {
    // Refresh the browser cache before handing this URL to navigation.
    const response = await fetch(entry, { cache: 'reload', redirect: 'error', signal: controller.signal });
    if (!response.ok) return false;
    const published = document.createElement('template');
    published.innerHTML = await response.text();
    const source = published.content.querySelector('script[type="module"][src^="./app.js?v="]')?.getAttribute('src');
    const release = source && new URL(source, entry).searchParams.get('v');
    if (!release || release === new URL(import.meta.url).searchParams.get('v')) return false;
    if (location.href !== currentEntry || !canReplace()) return false;
    location.replace(entry);
    return true;
  } catch {
    return false;
  } finally {
    clearTimeout(deadline);
  }
}

function idleConnectionPage() {
  const find = byId('quest-find');
  return !document.hidden && root.dataset.operatorStep === 'connect'
    && byId('check-bridge').dataset.check !== 'ok' && !find.hidden && find.getAttribute('aria-disabled') !== 'true'
    && !document.querySelector('dialog[open], #quest-pairing:not([hidden]), #quest-recovery:not([hidden]), #preview-recovery:not([hidden])')
    && ![...byId('patient-form').querySelectorAll('input, select')].some(field => field.value);
}

async function refreshIdlePage() {
  if (refreshingIdlePage || !idleConnectionPage()) return;
  idleRefreshRequested = false;
  refreshingIdlePage = true;
  if (!await replaceWithPublishedRelease(idleConnectionPage)) refreshingIdlePage = false;
}

function requestIdleRefresh() {
  if (refreshingIdlePage) return;
  idleRefreshRequested = true;
  void refreshIdlePage();
}

if (!await replaceWithPublishedRelease()) {
  document.addEventListener('visibilitychange', requestIdleRefresh);
  window.addEventListener('pageshow', event => {
    if (event.persisted) requestIdleRefresh();
  });
  window.setInterval(requestIdleRefresh, 5 * 60_000);
  window.addEventListener('hashchange', mountOperator);
  const recovery = location.hash === '#daten';
  const browserExports = byId('daten');
  browserExports.querySelector('.back').addEventListener('click', event => {
    if (!history.state?.browserExportsFromRecordings || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    history.back();
  });
  let browserExportsLoaded = false;
  const showBrowserExports = () => {
    if (browserExports.hidden) return;
    if (!browserExportsLoaded) { browserExportsLoaded = true; void showSavedBrowserExports(); }
    byId('data-title').focus();
  };
  new MutationObserver(showBrowserExports).observe(browserExports, { attributes: true, attributeFilter: ['hidden'] });
  byId('quest').hidden = recovery;
  browserExports.hidden = !recovery;
  byId('quest-status').textContent = 'Quest-Steuerung wird geladen …';
  measureLabels();
  mountOperator();
}
