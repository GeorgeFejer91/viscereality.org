(viewport) => {
  const root = document.documentElement;
  const problems = new Set();
  const active = document.querySelector('dialog[open], .screen:not([hidden])');
  const panel = document.querySelector('dialog[open]') ?? active;
  const toolbar = document.querySelector('#operator-toolbar');
  const visible = node => node instanceof HTMLElement && node.getClientRects().length
    && node.checkVisibility({ checkVisibilityCSS: true }) && !node.closest('.sr-only, [aria-hidden="true"]');
  const name = node => node.id || node.className || (node.matches('[data-measure]')
    ? `${node.localName} "${node.textContent.trim().slice(0, 60)}"` : node.localName);
  const contains = (outer, inner, margin = 1) => inner.left >= outer.left - margin
    && inner.right <= outer.right + margin && inner.top >= outer.top - margin && inner.bottom <= outer.bottom + margin;
  const view = { left: visualViewport.offsetLeft, top: visualViewport.offsetTop,
    right: visualViewport.offsetLeft + Math.min(visualViewport.width, viewport.width),
    bottom: visualViewport.offsetTop + Math.min(visualViewport.height, viewport.height) };
  const rect = node => node.getBoundingClientRect();
  const measureOverflow = node => {
    if (node.scrollWidth > node.clientWidth + 1 || node.scrollHeight > node.clientHeight + 1
        || node.scrollTop || node.scrollLeft) problems.add(`${name(node)} scrolls (${node.scrollWidth - node.clientWidth} × ${node.scrollHeight - node.clientHeight})`);
  };
  measureOverflow(root);
  if (root.scrollHeight > viewport.height + 1 || root.scrollWidth > viewport.width + 1
      || innerWidth > viewport.width + 1) problems.add(`document exceeds ${viewport.width} × ${viewport.height}`);
  if (document.querySelector('.viewport-navigation, [data-page-index]')) problems.add('generic page arrows');
  if (!panel) return { problems: ['no visible screen or dialog'] };
  const row = [...toolbar.querySelectorAll('button')].filter(visible).map(rect);
  if (row.length < 5 || row.some(box => box.width < 47.99 || box.height < 47.99
      || Math.abs(box.top - row[0].top) > 1 || Math.abs(box.width - row[0].width) > 1)) {
    problems.add('toolbar needs one row of equal 48 px targets');
  }
  if (rect(toolbar).bottom > rect(panel).top + 1) problems.add('toolbar overlaps the active view');
  for (const node of [panel, ...panel.querySelectorAll('*')].filter(visible)) {
    const box = rect(node);
    if (!contains(view, box)) problems.add(`${name(node)} outside the visual viewport`);
    if (node.clientWidth && node.clientHeight && !node.matches('input, select, progress, canvas')) measureOverflow(node);
    if (node.matches('[data-measure]') && node.textContent.trim()) {
      if (node.dataset.textFit !== 'fit' || !['pretext', 'dom'].includes(node.dataset.textMeasurement)) {
        problems.add(`${name(node)} has no passing text measurement`);
      }
      const range = document.createRange();
      range.selectNodeContents(node);
      if ([...range.getClientRects()].some(line => !contains(box, line, 2))) {
        problems.add(`${name(node)} text crosses its box`);
      }
    }
    if (node.matches('button, input:not([type=radio]), select, summary, a[href]')) {
      const target = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
      if (target && !node.contains(target)) problems.add(`${name(node)} is covered`);
      if (!contains(rect(panel), box)) problems.add(`${name(node)} escapes its panel`);
    }
  }
  const focus = document.activeElement;
  if (visible(focus) && (panel.contains(focus) || toolbar.contains(focus)) && !contains(view, rect(focus))) {
    problems.add(`${name(focus)} focus outside the visual viewport`);
  }
  const header = panel.querySelector(':scope > :is(header, h1, h2)');
  const body = panel.querySelector(':scope > .screen-body');
  const footer = panel.querySelector(':scope > footer');
  if (![header, body, footer].every(visible)) {
    problems.add(`${name(panel)} needs a header, flexible body and action footer`);
    return { problems: [...problems] };
  }
  if (rect(header).bottom > rect(body).top + 1 || rect(body).bottom > rect(footer).top + 1) {
    problems.add(`${name(panel)} has overlapping stretch regions`);
  }
  const style = getComputedStyle(body);
  if (style.minWidth !== '0px' || style.minHeight !== '0px') problems.add('body needs shrinkable tracks');
  const groups = [...(body.querySelector('#patient-fields') ?? body).children].filter(visible);
  const rows = new Map();
  for (const group of groups) {
    const box = rect(group);
    rows.set(box.top, Math.max(rows.get(box.top) ?? 0, box.bottom));
    if (group.matches('.patient-row')) {
      const peers = [...group.querySelectorAll('input, select, output')].filter(visible).map(rect);
      if (peers.some(peer => ['top', 'bottom', 'width'].some(edge => Math.abs(peer[edge] - peers[0][edge]) > 1))) {
        problems.add('participant controls lack shared row tracks');
      }
    }
  }
  const ordered = [...rows].sort(([a], [b]) => a - b);
  const gap = ordered.slice(1).reduce((sum, [top], index) => sum + Math.max(0, top - ordered[index][1]), 0);
  const bounds = rect(body);
  return { problems: [...problems], width: bounds.width, height: bounds.height, rows: rows.size,
    gap, contents: groups.map(name).join(' ') };
}
