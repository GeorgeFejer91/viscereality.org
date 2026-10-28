let measurement;
const samplesContainer = document.getElementById('samples');
const preparedLabels = new Map();
let measurementFrame;

function measureText() {
  const results = [];
  for (const label of document.querySelectorAll('[data-measure]')) {
    const css = getComputedStyle(label);
    const box = label.getBoundingClientRect();
    const width = box.width - parseFloat(css.paddingLeft) - parseFloat(css.paddingRight) - parseFloat(css.borderLeftWidth) - parseFloat(css.borderRightWidth);
    const height = box.height - parseFloat(css.paddingTop) - parseFloat(css.paddingBottom) - parseFloat(css.borderTopWidth) - parseFloat(css.borderBottomWidth);
    const font = `${css.fontStyle} ${css.fontWeight} ${css.fontSize} ${css.fontFamily}`;
    const text = label.textContent.trim();
    const spacing = parseFloat(css.letterSpacing) || 0;
    const key = JSON.stringify([text, font, spacing]);
    if (!preparedLabels.has(key)) preparedLabels.set(key, measurement.prepareWithSegments(text, font, { letterSpacing: spacing, whiteSpace: 'normal', wordBreak: 'normal' }));
    const prepared = preparedLabels.get(key);
    const stats = measurement.measureLineStats(prepared, Math.max(1, width - 1));
    const fits = stats.maxLineWidth <= width + 1 && stats.lineCount * parseFloat(css.lineHeight) <= height + 2;
    label.classList.toggle('measurement-reflow', !fits);
    label.dataset.fit = fits ? 'fit' : 'reflow';
    results.push({ text, width, height, naturalWidth: measurement.measureNaturalWidth(prepared), ...stats, fits });
  }
  while (preparedLabels.size > 160) preparedLabels.delete(preparedLabels.keys().next().value);
  window.atmungsraumMeasurements = results;
  document.documentElement.dataset.measurement = 'ready';
}

function scheduleMeasurement() {
  cancelAnimationFrame(measurementFrame);
  measurementFrame = requestAnimationFrame(measureText);
}

function connectPlayers() {
  const players = [...document.querySelectorAll('audio')];
  for (const audio of players) {
    audio.addEventListener('play', () => {
      for (const other of players) if (other !== audio) other.pause();
    });
    audio.addEventListener('error', () => {
      const article = audio.closest('article');
      if (article.querySelector('.audio-error')) return;
      const error = document.createElement('p');
      error.className = 'audio-error';
      error.setAttribute('role', 'status');
      error.textContent = 'Die Aufnahme konnte nicht geladen werden. Versuche den MP3-Download.';
      error.dataset.measure = '';
      article.append(error);
      if (measurement) scheduleMeasurement();
    });
  }
}

async function initialize() {
  connectPlayers();
  await document.fonts.ready;
  if (!window.Intl?.Segmenter || !document.fonts.check('16px Arial')) {
    document.documentElement.dataset.measurement = 'unavailable';
    return;
  }
  try {
    measurement = await import('../vendor/pretext/layout.js');
  } catch {
    document.documentElement.dataset.measurement = 'unavailable';
    return;
  }
  measurement.setLocale('de');
  scheduleMeasurement();
  new ResizeObserver(scheduleMeasurement).observe(samplesContainer);
  window.addEventListener('resize', scheduleMeasurement);
}

initialize();
