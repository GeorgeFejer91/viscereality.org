// ZIP32 with stored entries: keep original CSV bytes, without a runtime dependency.
const crcTable = Uint32Array.from({ length: 256 }, (_, value) => {
  for (let bit = 0; bit < 8; bit++) value = (value >>> 1) ^ ((value & 1) ? 0xedb88320 : 0);
  return value >>> 0;
});
export function createExportsZip(records) {
  const encoder = new TextEncoder(), parts = [], directory = [], names = new Set();
  let offset = 0, directorySize = 0;
  if (records.length > 65535) throw new Error('Too many ZIP entries');
  for (const record of records) {
    const path = names.has(record.filename) ? `${record.session_id}/${record.filename}` : record.filename;
    if (names.has(path) || /[\\/\0]/.test(record.filename) || /[\\/\0]/.test(record.session_id)) throw new Error('Invalid ZIP filename');
    names.add(path);
    const name = encoder.encode(path), bytes = encoder.encode(record.csv);
    if (name.length > 65535 || offset + bytes.length + name.length + 30 > 0xffffffff) throw new Error('ZIP32 size exceeded');
    let crc = 0xffffffff;
    for (const byte of bytes) crc = (crc >>> 8) ^ crcTable[(crc ^ byte) & 255];
    crc = (crc ^ 0xffffffff) >>> 0;
    const local = new Uint8Array(30), header = new DataView(local.buffer);
    header.setUint32(0, 0x04034b50, true);
    header.setUint16(4, 20, true);
    header.setUint16(6, 0x0800, true); // UTF-8 filenames.
    header.setUint16(12, 33, true); // 1980-01-01; recording UTC stays in the CSV.
    header.setUint32(14, crc, true);
    header.setUint32(18, bytes.length, true);
    header.setUint32(22, bytes.length, true);
    header.setUint16(26, name.length, true);
    const central = new Uint8Array(46), entry = new DataView(central.buffer);
    entry.setUint32(0, 0x02014b50, true);
    entry.setUint16(4, 20, true);
    central.set(local.subarray(4, 30), 6);
    entry.setUint32(42, offset, true);
    parts.push(local, name, bytes);
    directory.push(central, name);
    directorySize += central.length + name.length;
    offset += local.length + name.length + bytes.length;
  }
  if (offset + directorySize + 22 > 0xffffffff) throw new Error('ZIP32 size exceeded');
  const end = new Uint8Array(22), footer = new DataView(end.buffer);
  footer.setUint32(0, 0x06054b50, true);
  footer.setUint16(8, records.length, true);
  footer.setUint16(10, records.length, true);
  footer.setUint32(12, directorySize, true);
  footer.setUint32(16, offset, true);
  return new Blob([...parts, ...directory, end], { type: 'application/zip' });
}

export function mountDataFiles(data) {
  const byId = id => document.getElementById(id);
  const dialog = byId('files-dialog'), button = byId('open-files'), list = byId('saved-files');
  const status = byId('files-status'), zipButton = byId('download-zip'), shareButton = byId('share-files');
  let zip, zipUrl, generation = 0, returnFocus, sharing = false;
  const clearZip = () => {
    if (zipUrl) URL.revokeObjectURL(zipUrl);
    zip = zipUrl = null;
    zipButton.disabled = shareButton.disabled = true;
  };
  async function refresh() {
    const version = ++generation;
    clearZip();
    status.textContent = 'Dateien werden geladen …';
    try {
      const records = await data.exports();
      if (version !== generation || !dialog.open) return;
      // Reuse the recovery links and their existing object URLs.
      list.replaceChildren(...[...byId('data-files').children].map(item => item.cloneNode(true)));
      if (!records.length) { status.textContent = 'Noch keine gespeicherten Session-Daten.'; return; }
      status.textContent = `${records.length} ${records.length === 1 ? 'Datei' : 'Dateien'} · ZIP wird vorbereitet …`;
      await new Promise(resolve => setTimeout(resolve, 0));
      if (version !== generation || !dialog.open) return;
      zip = new File([createExportsZip(records)], 'Atmungsraum-Daten.zip', { type: 'application/zip' });
      zipUrl = URL.createObjectURL(zip);
      zipButton.disabled = false;
      let canShare = false;
      try { canShare = Boolean(navigator.share && navigator.canShare?.({ files: [zip] })); } catch { /* Download remains available. */ }
      shareButton.disabled = !canShare;
      status.textContent = `${records.length} ${records.length === 1 ? 'Datei' : 'Dateien'} gespeichert.`;
      byId('files-share-hint').textContent = canShare
        ? 'Über „Teilen“ eine App wählen, z. B. E-Mail oder WhatsApp. Alle Dateien werden als ZIP angehängt.'
        : 'Dieser Browser kann die ZIP nicht direkt teilen. ZIP herunterladen und in E-Mail, WhatsApp oder einer anderen App anhängen.';
    } catch (error) {
      if (version !== generation || !dialog.open) return;
      status.textContent = list.children.length ? 'ZIP konnte nicht vorbereitet werden. Dateien einzeln herunterladen oder erneut öffnen.' : 'Dateien konnten nicht geladen werden. Bitte erneut öffnen.';
      console.warn('File menu unavailable.', error);
    }
  }
  function open() {
    if (dialog.open) return;
    returnFocus = document.activeElement;
    dialog.showModal();
    button.setAttribute('aria-expanded', 'true');
    refresh();
  }
  button.addEventListener('click', open);
  byId('close-files').addEventListener('click', () => dialog.close());
  // Escape dismisses this popup without reaching the player's stop shortcut.
  dialog.addEventListener('keydown', event => { if (event.key === 'Escape') event.stopPropagation(); });
  dialog.addEventListener('close', () => {
    ++generation;
    clearZip();
    button.setAttribute('aria-expanded', 'false');
    if (returnFocus?.isConnected && returnFocus.getClientRects().length) returnFocus.focus({ preventScroll: true });
    else button.focus({ preventScroll: true });
  });
  zipButton.addEventListener('click', () => {
    if (!zipUrl) return;
    const link = document.createElement('a');
    link.href = zipUrl; link.download = zip.name;
    document.body.append(link); link.click(); link.remove();
  });
  shareButton.addEventListener('click', async () => {
    if (!zip || sharing) return;
    sharing = true;
    try {
      // The archive is prepared before this click, preserving transient activation.
      await navigator.share({ files: [zip], title: 'Atmungsraum-Daten' });
      status.textContent = 'ZIP an die gewählte App übergeben.';
    } catch (error) {
      if (error.name !== 'AbortError') status.textContent = 'Teilen nicht möglich. ZIP herunterladen und in der gewünschten App anhängen.';
    } finally { sharing = false; }
  });
  document.addEventListener('atmungsraum:exports-changed', () => { if (dialog.open) refresh(); });
  return { open };
}
