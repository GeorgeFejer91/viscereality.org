import { sendDiscordLink, validateRelaySettings } from './link-relay.js?v=20261107-file-delivery';
import { unlockDiscordWebhook } from './discord-share.js?v=20261107-file-delivery';
import { uploadWormholeZip } from './wormhole-upload.js?v=20261107-file-delivery';
import { deliveryKey, receiptPrefix, readSentFiles, deliveryEntries, prepareDeliveryReceipt } from './file-delivery.js?v=20261107-checkout-receipts';
// ZIP32 with stored entries preserves the original recording bytes.
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
    const name = encoder.encode(path), bytes = record.bytes ?? encoder.encode(record.csv);
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
  const dialog = byId('files-dialog'), button = byId('open-files'), close = byId('close-files');
  const counts = byId('files-counts'), status = byId('files-status'), progress = byId('files-progress');
  const sendNew = byId('send-new-files'), sendAll = byId('send-all-files');
  const settings = byId('relay-settings'), endpoint = byId('relay-endpoint'), token = byId('relay-token');
  const passwordDialog = byId('discord-password-dialog'), passwordInput = byId('discord-password');
  const remember = byId('remember-discord-password');
  const canRemember = typeof globalThis.PasswordCredential === 'function' && typeof navigator.credentials?.store === 'function';
  remember.closest('label').hidden = !canRemember;
  let active, returnFocus, generation = 0, total = 0, unsent = 0;

  function syncButtons() {
    sendNew.disabled = Boolean(active) || unsent === 0;
    sendAll.disabled = Boolean(active) || total === 0;
  }

  async function refresh() {
    if (!dialog.open) return;
    const version = ++generation;
    try {
      const entries = await deliveryEntries(await data.exports());
      if (version !== generation || !dialog.open) return;
      const sent = readSentFiles(localStorage);
      total = entries.length;
      unsent = entries.filter(entry => !sent.has(entry.key)).length;
      counts.textContent = total ? total + ' gespeichert · ' + unsent + ' noch nicht gesendet' : 'Keine gespeicherten Dateien.';
    } catch {
      if (version !== generation || !dialog.open) return;
      total = unsent = 0;
      counts.textContent = 'Dateien oder Versandverlauf konnten nicht gelesen werden.';
    }
    syncButtons();
  }

  passwordDialog.addEventListener('keydown', event => { if (event.key === 'Escape') event.stopPropagation(); });
  function requestPassword(signal) {
    return new Promise(resolve => {
      if (signal.aborted) { resolve(null); return; }
      const form = byId('discord-password-form'), cancelButton = byId('cancel-discord-password');
      const finish = value => {
        form.removeEventListener('submit', submit);
        cancelButton.removeEventListener('click', cancel);
        passwordDialog.removeEventListener('cancel', cancel);
        passwordDialog.removeEventListener('close', cancel);
        signal.removeEventListener('abort', cancel);
        passwordInput.value = '';
        if (passwordDialog.open) passwordDialog.close();
        if (dialog.open) status.focus({ preventScroll: true });
        resolve(value);
      };
      const submit = event => {
        event.preventDefault();
        if (passwordInput.reportValidity()) finish({ password: passwordInput.value, remember: canRemember && remember.checked });
      };
      const cancel = event => { event.preventDefault(); finish(null); };
      form.addEventListener('submit', submit);
      cancelButton.addEventListener('click', cancel);
      passwordDialog.addEventListener('cancel', cancel);
      passwordDialog.addEventListener('close', cancel);
      signal.addEventListener('abort', cancel, { once: true });
      passwordInput.value = '';
      remember.checked = false;
      passwordDialog.showModal();
      passwordInput.focus({ preventScroll: true });
    });
  }

  async function deliver(all) {
    if (active || !dialog.open) return;
    const controller = active = new AbortController(), { signal } = controller;
    syncButtons();
    status.textContent = 'Dateien werden geprüft …';
    let notifying = false;
    try {
      await navigator.locks.request(deliveryKey, { ifAvailable: true }, async lock => {
        signal.throwIfAborted();
        if (!lock) { status.textContent = 'In einem anderen Tab läuft bereits ein Versand.'; return; }
        const entries = await deliveryEntries(await data.exports(), signal);
        const storage = localStorage, sent = readSentFiles(storage);
        const selected = all ? entries : entries.filter(entry => !sent.has(entry.key));
        if (!selected.length) {
          status.textContent = all ? 'Keine gespeicherten Dateien.' : 'Keine neuen Dateien zum Senden.';
          return;
        }
        const relay = { endpoint: endpoint.value, token: token.value };
        try { validateRelaySettings(relay.endpoint, relay.token); }
        catch (error) {
          settings.open = true;
          (relay.endpoint.trim() ? token : endpoint).focus();
          status.textContent = error.message;
          return;
        }
        const confirmReceipt = prepareDeliveryReceipt(storage, selected);
        status.textContent = 'Passwort eingeben …';
        const access = await requestPassword(signal);
        signal.throwIfAborted();
        if (!access) { status.textContent = 'Versand abgebrochen.'; return; }
        status.textContent = 'Passwort wird geprüft …';
        const webhook = await unlockDiscordWebhook(access.password, signal);
        status.textContent = 'ZIP wird vorbereitet …';
        const file = new File([createExportsZip(selected.map(entry => entry.record))], 'Atmungsraum-Daten.zip', { type: 'application/zip' });
        let lastPhase;
        const receipt = await uploadWormholeZip({ file, ...relay, signal, onProgress: ({ phase, completed, total: size }) => {
          if (signal.aborted) return;
          if (phase !== lastPhase) status.textContent = phase === 'starting' ? 'Upload-Dienst startet …'
            : phase === 'encrypting' ? 'ZIP wird verschlüsselt …' : 'ZIP wird zu Wormhole hochgeladen …';
          lastPhase = phase;
          progress.hidden = false;
          if (size) progress.value = Math.round(100 * completed / size);
          else progress.removeAttribute('value');
        } });
        signal.throwIfAborted();
        progress.removeAttribute('value');
        status.textContent = 'Link wird an Discord gesendet …';
        notifying = true;
        await sendDiscordLink({ webhook, url: receipt.url, signal });
        notifying = false;
        try { confirmReceipt(); }
        catch {
          status.textContent = 'Discord hat den Link angenommen. Der Versandverlauf konnte nicht gespeichert werden.';
          return;
        }
        status.textContent = selected.length === 1 ? '1 Datei gesendet. Discord hat den Link angenommen.'
          : selected.length + ' Dateien gesendet. Discord hat den Link angenommen.';
        if (access.remember && !signal.aborted) {
          try {
            navigator.credentials.store(new PasswordCredential({ id: byId('discord-password-user').value, password: access.password })).catch(() => {});
          } catch { /* Browser storage preference does not change delivery. */ }
        }
      });
    } catch (error) {
      status.textContent = signal.aborted
        ? (notifying ? 'Versand abgebrochen. Discord vor einem erneuten Versuch prüfen.' : 'Versand abgebrochen.')
        : error.message;
    } finally {
      active = null;
      progress.hidden = true;
      progress.removeAttribute('value');
      syncButtons();
      await refresh();
    }
  }

  function open() {
    if (dialog.open) return;
    returnFocus = document.activeElement;
    total = unsent = 0;
    syncButtons();
    counts.textContent = 'Dateien werden geladen …';
    dialog.showModal();
    button.setAttribute('aria-expanded', 'true');
    refresh();
  }

  function dismiss() {
    active?.abort();
    dialog.close();
  }

  button.addEventListener('click', open);
  close.addEventListener('click', dismiss);
  dialog.addEventListener('keydown', event => { if (event.key === 'Escape') event.stopPropagation(); });
  dialog.addEventListener('cancel', event => { event.preventDefault(); dismiss(); });
  dialog.addEventListener('close', () => {
    if (dialog.open) return;
    ++generation;
    active?.abort();
    endpoint.value = token.value = '';
    settings.open = false;
    button.setAttribute('aria-expanded', 'false');
    (returnFocus?.isConnected && returnFocus.getClientRects().length ? returnFocus : button).focus({ preventScroll: true });
  });
  sendNew.addEventListener('click', () => deliver(false));
  sendAll.addEventListener('click', () => deliver(true));
  document.addEventListener('atmungsraum:exports-changed', refresh);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  window.addEventListener('focus', refresh);
  window.addEventListener('storage', event => {
    if (event.key === null || event.key === deliveryKey || event.key.startsWith(receiptPrefix)) refresh();
  });
  return { open };
}
