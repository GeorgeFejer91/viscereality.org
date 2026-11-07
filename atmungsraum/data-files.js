import { validateWormholeUrl } from './wormhole-link.js';
import { validateRelaySettings } from './link-relay.js';
import { forwardPasswordDiscordLink } from './discord-share.js';
import { uploadWormholeZip } from './wormhole-upload.js';
export { validateWormholeUrl } from './wormhole-link.js';
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

export async function createWormholeQr(value) {
  const url = validateWormholeUrl(value);
  const { default: qrcode } = await import('./vendor/qrcode/qrcode.js');
  const qr = qrcode(0, 'M');
  qr.addData(url, 'Byte');
  qr.make();
  return { url, image: `data:image/svg+xml,${encodeURIComponent(qr.createSvgTag(6, 24))}` };
}

export function mountDataFiles(data) {
  const byId = id => document.getElementById(id);
  const dialog = byId('files-dialog'), button = byId('open-files'), list = byId('saved-files');
  const status = byId('files-status'), zipButton = byId('download-zip'), shareButton = byId('share-files');
  const qrButton = byId('share-qr'), flow = byId('wormhole-flow'), input = byId('wormhole-link');
  const qrStatus = byId('wormhole-status'), result = byId('wormhole-result');
  const relayStatus = byId('relay-status'), relayButton = byId('relay-send');
  const uploadButton = byId('upload-wormhole'), uploadStatus = byId('upload-status'), cancelButton = byId('cancel-upload');
  let zip, zipUrl, generation = 0, returnFocus, sharing = false;
  let qrGeneration = 0, qrLink = '';
  let forwarding, lastForwarded = '';
  let uploading, archiveRecords = [], uploadedShare;
  const sameArchive = records => uploadedShare?.records.length === records.length
    && records.every((record, i) => ['session_id', 'filename', 'csv'].every(key => record[key] === uploadedShare.records[i][key]));
  const forwardingKey = () => `discord\n${qrLink}`;
  const wasForwarded = key => lastForwarded === key || uploadedShare?.forwardedKey === key;
  const passwordDialog = byId('discord-password-dialog'), passwordInput = byId('discord-password');
  const rememberPassword = byId('remember-discord-password');
  const canRemember = typeof globalThis.PasswordCredential === 'function' && typeof navigator.credentials?.store === 'function';
  rememberPassword.closest('label').hidden = !canRemember;
  passwordDialog.addEventListener('keydown', event => { if (event.key === 'Escape') event.stopPropagation(); });
  function requestPassword(signal) {
    return new Promise(resolve => {
      if (signal.aborted) { resolve(null); return; }
      const form = byId('discord-password-form'), cancelPrompt = byId('cancel-discord-password');
      const finish = value => {
        form.removeEventListener('submit', submit);
        cancelPrompt.removeEventListener('click', cancel);
        passwordDialog.removeEventListener('cancel', cancel);
        passwordDialog.removeEventListener('close', cancel);
        signal.removeEventListener('abort', cancel);
        passwordInput.value = '';
        if (passwordDialog.open) passwordDialog.close();
        if (dialog.open && !flow.hidden) input.focus({ preventScroll: true });
        resolve(value);
      };
      const submit = event => {
        event.preventDefault();
        if (passwordInput.reportValidity()) finish({ password: passwordInput.value, remember: canRemember && rememberPassword.checked });
      };
      const cancel = event => { event.preventDefault(); finish(null); };
      form.addEventListener('submit', submit);
      cancelPrompt.addEventListener('click', cancel);
      passwordDialog.addEventListener('cancel', cancel);
      passwordDialog.addEventListener('close', cancel);
      signal.addEventListener('abort', cancel, { once: true });
      passwordInput.value = '';
      rememberPassword.checked = false;
      passwordDialog.showModal();
      passwordInput.focus({ preventScroll: true });
    });
  }
  function cancelUpload(announce = false) {
    if (!uploading) return;
    const restoreFocus = document.activeElement === cancelButton;
    uploading.abort(); uploading = null;
    cancelButton.hidden = true;
    uploadButton.disabled = !zip;
    if (restoreFocus && dialog.open && !flow.hidden && zip) uploadButton.focus({ preventScroll: true });
    if (announce) uploadStatus.textContent = 'Upload abgebrochen. Gespeicherte Dateien bleiben erhalten.';
  }
  function cancelForwarding() {
    forwarding?.abort();
    forwarding = null;
    relayStatus.textContent = '';
    relayButton.disabled = !qrLink || wasForwarded(forwardingKey());
  }
  function clearRelaySettings() {
    cancelForwarding();
    lastForwarded = '';
    byId('relay-token').value = byId('relay-endpoint').value = '';
    byId('relay-auto').checked = true;
    byId('relay-settings').open = false;
  }
  async function forwardCurrentLink() {
    if (!qrLink || forwarding) return;
    const key = forwardingKey();
    if (wasForwarded(key)) return;
    const controller = new AbortController(), version = qrGeneration;
    forwarding = controller;
    relayButton.disabled = true;
    relayStatus.textContent = 'Passwort für Discord eingeben …';
    try {
      const access = await requestPassword(controller.signal);
      if (version !== qrGeneration || forwarding !== controller || !dialog.open || controller.signal.aborted) return;
      if (access === null) { relayStatus.textContent = 'Nicht gesendet.'; return; }
      relayStatus.textContent = 'Link wird direkt an Discord gesendet …';
      await forwardPasswordDiscordLink({ password: access.password, url: qrLink, signal: controller.signal });
      if (version !== qrGeneration || forwarding !== controller || !dialog.open) return;
      lastForwarded = key;
      if (uploadedShare?.url === qrLink) uploadedShare.forwardedKey = key;
      relayStatus.textContent = 'Von Discord zur Zustellung angenommen. Bitte das Ziel prüfen.';
      if (access.remember) {
        try {
          navigator.credentials.store(new PasswordCredential({ id: byId('discord-password-user').value, password: access.password })).catch(() => {});
        } catch { /* Password manager availability does not change delivery. */ }
      }
    } catch (error) {
      if (version !== qrGeneration || forwarding !== controller || !dialog.open) return;
      relayStatus.textContent = error.name === 'TimeoutError'
        ? 'Weiterleitung nicht bestätigt. Ziel prüfen, bevor du erneut sendest.' : error.message;
    } finally {
      if (forwarding === controller) {
        forwarding = null;
        relayButton.disabled = !qrLink || wasForwarded(forwardingKey());
      }
    }
  }
  const clearQr = () => {
    ++qrGeneration;
    qrLink = '';
    cancelForwarding();
    if (result.contains(document.activeElement) && dialog.open && !flow.hidden) input.focus({ preventScroll: true });
    result.hidden = true;
    byId('wormhole-qr').removeAttribute('src');
    byId('wormhole-open-link').removeAttribute('href');
  };
  const resetWormhole = () => {
    clearQr();
    input.value = '';
    input.setAttribute('aria-invalid', 'false');
    qrStatus.textContent = '';
  };
  const clearZip = () => {
    cancelUpload();
    if (zipUrl) URL.revokeObjectURL(zipUrl);
    zip = zipUrl = null;
    zipButton.disabled = shareButton.disabled = qrButton.disabled = uploadButton.disabled = byId('wormhole-download').disabled = true;
  };
  async function refresh() {
    const version = ++generation;
    clearZip();
    resetWormhole();
    uploadStatus.textContent = '';
    status.textContent = 'Dateien werden geladen …';
    try {
      const records = await data.exports();
      if (version !== generation || !dialog.open) return;
      // Reuse the recovery links and their existing object URLs.
      list.replaceChildren(...[...byId('data-files').children].map(item => item.cloneNode(true)));
      if (!sameArchive(records)) uploadedShare = null;
      archiveRecords = records.map(({ session_id, filename, csv }) => ({ session_id, filename, csv }));
      if (!records.length) { status.textContent = 'Noch keine gespeicherten Session-Daten.'; return; }
      status.textContent = `${records.length} ${records.length === 1 ? 'Datei' : 'Dateien'} · ZIP wird vorbereitet …`;
      await new Promise(resolve => setTimeout(resolve, 0));
      if (version !== generation || !dialog.open) return;
      zip = new File([createExportsZip(records)], 'Atmungsraum-Daten.zip', { type: 'application/zip' });
      zipUrl = URL.createObjectURL(zip);
      zipButton.disabled = qrButton.disabled = uploadButton.disabled = byId('wormhole-download').disabled = false;
      let canShare = false;
      try { canShare = Boolean(navigator.share && navigator.canShare?.({ files: [zip] })); } catch { /* Download remains available. */ }
      shareButton.disabled = !canShare;
      status.textContent = `${records.length} ${records.length === 1 ? 'Datei' : 'Dateien'} gespeichert.`;
      byId('files-share-hint').textContent = canShare
        ? 'Über „Teilen“ eine App wählen, z. B. E-Mail oder WhatsApp. Alle Dateien werden als ZIP angehängt.'
        : 'Dieser Browser kann die ZIP nicht direkt teilen. ZIP herunterladen und in E-Mail, WhatsApp oder einer anderen App anhängen.';
      if (uploadedShare) {
        uploadStatus.textContent = 'Diese ZIP wurde bereits hochgeladen. Der gespeicherte Link ist unten verfügbar.';
        await displayLink(uploadedShare.url, false);
      }
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
    resetWormhole();
    clearRelaySettings();
    flow.hidden = true;
    qrButton.setAttribute('aria-expanded', 'false');
    button.setAttribute('aria-expanded', 'false');
    if (returnFocus?.isConnected && returnFocus.getClientRects().length) returnFocus.focus({ preventScroll: true });
    else button.focus({ preventScroll: true });
  });
  const downloadZip = () => {
    if (!zipUrl) return;
    const link = document.createElement('a');
    link.href = zipUrl; link.download = zip.name;
    document.body.append(link); link.click(); link.remove();
  };
  zipButton.addEventListener('click', downloadZip);
  byId('wormhole-download').addEventListener('click', downloadZip);
  qrButton.addEventListener('click', () => {
    flow.hidden = !flow.hidden;
    qrButton.setAttribute('aria-expanded', String(!flow.hidden));
    if (flow.hidden) { cancelUpload(); resetWormhole(); clearRelaySettings(); }
    else {
      if (uploadedShare && sameArchive(archiveRecords)) displayLink(uploadedShare.url, false);
      uploadButton.focus({ preventScroll: true });
    }
  });
  input.addEventListener('input', () => {
    cancelUpload(true);
    clearQr();
    input.setAttribute('aria-invalid', 'false');
    qrStatus.textContent = '';
  });
  async function displayLink(url, forward = true) {
    clearQr();
    const version = qrGeneration;
    qrLink = validateWormholeUrl(url); input.value = qrLink;
    input.setAttribute('aria-invalid', 'false');
    byId('wormhole-open-link').href = qrLink;
    byId('wormhole-qr').hidden = true;
    result.hidden = false;
    relayButton.disabled = wasForwarded(forwardingKey());
    qrStatus.textContent = 'QR-Code wird auf diesem Gerät erstellt …';
    try {
      const qr = await createWormholeQr(url);
      if (version !== qrGeneration || !dialog.open) return;
      byId('wormhole-qr').src = qr.image;
      byId('wormhole-qr').hidden = false;
      qrStatus.textContent = 'QR-Code für den Wormhole-Link bereit.';
    } catch {
      if (version !== qrGeneration || !dialog.open) return;
      qrStatus.textContent = 'QR-Code konnte nicht erstellt werden. Link aus dem Eingabefeld kopieren oder die Seite neu laden.';
    }
    if (version === qrGeneration && dialog.open && forward && byId('relay-auto').checked) forwardCurrentLink();
  }
  byId('wormhole-form').addEventListener('submit', async event => {
    event.preventDefault();
    cancelUpload(true);
    let url;
    try { url = validateWormholeUrl(input.value); }
    catch (error) {
      clearQr(); input.setAttribute('aria-invalid', 'true'); qrStatus.textContent = error.message; input.focus(); return;
    }
    await displayLink(url);
  });
  uploadButton.addEventListener('click', async () => {
    if (!zip || uploading || forwarding) return;
    if (uploadedShare && sameArchive(archiveRecords)) {
      uploadStatus.textContent = 'Diese ZIP wurde bereits hochgeladen. Der gespeicherte Link ist unten verfügbar.';
      await displayLink(uploadedShare.url);
      return;
    }
    const endpoint = byId('relay-endpoint').value, token = byId('relay-token').value;
    try { validateRelaySettings(endpoint, token); }
    catch (error) {
      byId('relay-settings').open = true; uploadStatus.textContent = error.message;
      byId(endpoint.trim() ? 'relay-token' : 'relay-endpoint').focus(); return;
    }
    clearQr(); input.value = '';
    const controller = new AbortController(), version = generation, snapshot = archiveRecords;
    uploading = controller; uploadButton.disabled = true; cancelButton.hidden = false;
    uploadStatus.textContent = 'ZIP wird auf diesem Gerät verschlüsselt …';
    try {
      const receipt = await uploadWormholeZip({ file: zip, endpoint, token, signal: controller.signal,
        onProgress: progress => {
          if (uploading !== controller || version !== generation) return;
          const percent = Math.floor(progress.completed / progress.total * 100);
          uploadStatus.textContent = progress.phase === 'encrypting'
            ? `ZIP wird auf diesem Gerät verschlüsselt (${percent} %).`
            : `Verschlüsselte ZIP wird hochgeladen (${percent} %).`;
        },
      });
      if (controller.signal.aborted || uploading !== controller || version !== generation || !dialog.open) return;
      uploadedShare = { ...receipt, records: snapshot };
      if (document.activeElement === cancelButton) {
        uploadButton.disabled = false;
        uploadButton.focus({ preventScroll: true });
      }
      cancelButton.hidden = true;
      uploadStatus.textContent = `ZIP mit ${snapshot.length} ${snapshot.length === 1 ? 'Datei' : 'Dateien'} hochgeladen. Link und QR-Code sind unten verfügbar.`;
      await displayLink(receipt.url);
    } catch (error) {
      if (uploading !== controller || version !== generation || !dialog.open) return;
      uploadStatus.textContent = error.name === 'AbortError' ? 'Upload abgebrochen. Gespeicherte Dateien bleiben erhalten.'
        : `${error.message} ZIP kann weiterhin heruntergeladen und manuell auf Wormhole geteilt werden.`;
    } finally {
      if (uploading === controller) {
        uploading = null; uploadButton.disabled = !zip;
        if (document.activeElement === cancelButton && dialog.open && !flow.hidden && zip) uploadButton.focus({ preventScroll: true });
        cancelButton.hidden = true;
      }
    }
  });
  cancelButton.addEventListener('click', () => cancelUpload(true));
  byId('wormhole-copy').addEventListener('click', async () => {
    if (!qrLink) return;
    const version = qrGeneration;
    try {
      await navigator.clipboard.writeText(qrLink);
      if (version === qrGeneration) qrStatus.textContent = 'Wormhole-Link kopiert.';
    } catch {
      if (version !== qrGeneration) return;
      input.focus(); input.select();
      qrStatus.textContent = 'Link im Eingabefeld markiert. Bitte selbst kopieren.';
    }
  });
  relayButton.addEventListener('click', forwardCurrentLink);
  for (const id of ['relay-endpoint', 'relay-token']) byId(id).addEventListener('input', () => {
    lastForwarded = '';
    cancelUpload(true);
    cancelForwarding();
  });
  byId('relay-auto').addEventListener('change', () => {
    if (byId('relay-auto').checked) forwardCurrentLink();
    else cancelForwarding();
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
