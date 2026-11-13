import { createQuestDrive, deliverQuestBundle, deliverQuestPrefix, questDriveScope } from './quest-drive.js?v=2';
import { openQuestRecordingStore } from './quest-recording-store.js?v=3';

const clientIdPattern = /^[0-9]+-[A-Za-z0-9_-]+\.apps\.googleusercontent\.com$/;
const lockName = 'atmungsraum-quest-drive';

export function mountQuestDrive() {
  const panel = document.getElementById('quest-drive');
  const form = document.getElementById('quest-drive-form');
  const clientId = document.getElementById('quest-drive-client');
  const email = document.getElementById('quest-drive-account');
  const button = document.getElementById('quest-drive-connect');
  const status = document.getElementById('quest-drive-status');
  let owner, library, loading, binding, connection, upload;
  let due = 0;

  function report(message) {
    status.hidden = !message;
    if (status.textContent !== message) status.textContent = message;
  }

  function loadLibrary() {
    if (library || loading || !panel.open) return;
    const script = document.createElement('script');
    loading = script;
    script.src = 'https://accounts.google.com/gsi/client';
    script.async = true;
    script.onload = () => {
      loading = null;
      library = window.google?.accounts?.oauth2;
      button.disabled = !library || Boolean(binding);
      if (!library) report('Google-Anmeldung konnte nicht geladen werden.');
    };
    script.onerror = () => {
      loading = null;
      script.remove();
      report('Google-Anmeldung konnte nicht geladen werden.');
    };
    document.head.append(script);
  }

  async function authorize(event) {
    event.preventDefault();
    if (!owner || !library || binding || !form.reportValidity()) return;
    const config = { clientId: clientId.value.trim(), email: email.value.trim().toLowerCase() };
    if (!clientIdPattern.test(config.clientId)) { report('Die Google OAuth-Client-ID ist ungültig.'); return; }
    const attempt = new AbortController();
    const selectedOwner = owner;
    upload?.abort();
    connection = null;
    binding = attempt;
    button.disabled = true;
    const finish = (message) => {
      if (binding !== attempt) return;
      binding = null;
      button.disabled = false;
      report(message);
    };
    try {
      const client = library.initTokenClient({ client_id: config.clientId, scope: questDriveScope,
        login_hint: config.email, include_granted_scopes: false, error_callback: () => finish('Google-Anmeldung wurde nicht abgeschlossen.'),
        callback: async response => {
          try {
            const expiresIn = Number(response?.expires_in);
            if (!response?.access_token || !response.scope?.split(' ').includes(questDriveScope)
              || !Number.isFinite(expiresIn) || expiresIn <= 0) throw Error('Google Drive wurde nicht freigegeben.');
            const drive = createQuestDrive(response.access_token);
            const signal = attempt.signal;
            const user = await drive.account(signal);
            if (user.emailAddress.toLowerCase() !== config.email) throw Error('Das ausgewählte Google-Konto stimmt nicht überein.');
            const destination = await navigator.locks.request(lockName, { signal }, async () => {
              const store = await openQuestRecordingStore(selectedOwner);
              try {
                let saved = await store.readDriveDestination();
                if (saved && (saved.accountId !== user.permissionId || saved.clientId !== config.clientId)) {
                  throw Error('Die Google-Drive-Ablage ist an ein anderes Konto oder einen anderen Client gebunden.');
                }
                if (!saved) {
                  saved = { ...config, accountId: user.permissionId, folderId: await drive.reserveId(signal) };
                  await store.saveDriveDestination(saved);
                }
                await drive.ensureFolder(saved.folderId, signal);
                return saved;
              } finally { store.close(); }
            });
            signal.throwIfAborted();
            connection = { drive, destination, expiresAt: performance.now() + expiresIn * 1000, checkFolderAfter: 0 };
            due = 0;
            panel.open = false;
            panel.querySelector('summary').focus({ preventScroll: true });
            finish('Google Drive verbunden. Gespeicherte Aufnahmen werden abgeglichen.');
          } catch (error) { if (!attempt.signal.aborted) finish(error.message); }
        } });
      client.requestAccessToken({ prompt: 'select_account' });
    } catch { finish('Google-Anmeldung konnte nicht geöffnet werden.'); }
  }

  panel.addEventListener('toggle', loadLibrary);
  form.addEventListener('submit', authorize);

  return {
    async setOwner(next) {
      if (owner?.device === next?.device && owner?.serial === next?.serial) return;
      upload?.abort();
      binding?.abort();
      binding = connection = null;
      button.disabled = !library;
      owner = next;
      panel.hidden = !next;
      report('');
      if (!next) return;
      let store;
      try {
        store = await openQuestRecordingStore(next);
        const settings = await store.readDriveDestination();
        if (owner !== next || !settings) return;
        clientId.value = settings.clientId;
        email.value = settings.email;
        report('Google Drive verbinden, um ausstehende Aufnahmen zu übertragen.');
      } catch { if (owner === next) report('Die Google-Drive-Einstellungen konnten nicht gelesen werden.'); }
      finally { store?.close(); }
    },
    schedule() { due = 0; },
    cancel() { upload?.abort(); },
    async sync(prefix) {
      if (!owner || !connection || binding || upload || performance.now() < due) return;
      const current = connection;
      if (performance.now() >= current.expiresAt) {
        connection = null;
        report('Google Drive erneut verbinden. Die Aufnahmen bleiben gespeichert.');
        return;
      }
      const selectedOwner = owner;
      const attempt = new AbortController();
      const signal = attempt.signal;
      upload = attempt;
      due = performance.now() + 10000;
      try {
        await navigator.locks.request(lockName, { ifAvailable: true }, async lock => {
          if (!lock) return;
          signal.throwIfAborted();
          const store = await openQuestRecordingStore(selectedOwner);
          try {
            const folderId = current.destination.folderId;
            if (prefix) {
              const staged = await store.readPrefix(prefix.id);
              if (staged?.session_id !== prefix.session_id || staged.offset < 256 * 1024 + 1) {
                due = 0;
                return;
              }
              const receipt = await store.readDelivery(prefix.id);
              if (!receipt?.verified && (receipt?.pendingEnd || staged.offset - (receipt?.offset ?? 0) > 256 * 1024)) {
                if (!receipt || performance.now() >= current.checkFolderAfter) {
                  await current.drive.ensureFolder(folderId, signal);
                  current.checkFolderAfter = performance.now() + 30000;
                }
                await deliverQuestPrefix({ drive: current.drive, store, folderId, id: prefix.id, name: staged.name,
                  type: 'text/csv', available: staged.offset, signal,
                  read: (start, end) => store.readPrefixBytes(prefix.id, start, end) });
                const acknowledged = (await store.readDelivery(prefix.id))?.offset ?? 0;
                if (acknowledged) report('Google Drive empfängt die laufende Aufnahme.');
              }
              due = 0;
              return;
            }
            let saved = 0;
            for (const id of (await store.ids()).reverse()) {
              signal.throwIfAborted();
              const receipt = await store.readDelivery(id);
              if (receipt?.verified && receipt.folderId === folderId) { saved++; continue; }
              const entry = await store.get(id);
              if (entry?.bundleVersion !== 1 || !(entry.bundle instanceof File)) continue;
              report('Aufnahmen werden in Google Drive gespeichert …');
              await current.drive.ensureFolder(folderId, signal);
              await deliverQuestBundle({ drive: current.drive, store, folderId, id, file: entry.bundle, signal });
              saved++;
            }
            signal.throwIfAborted();
            report(saved ? `In Google Drive bestätigt: ${saved}` : 'Google Drive verbunden. Warte auf Aufnahmen.');
            due = performance.now() + 60000;
          } finally { store.close(); }
        });
      } catch (error) {
        if (signal.aborted) return;
        if (error.status === 401) {
          connection = null;
          report('Google Drive erneut verbinden. Die Aufnahmen bleiben gespeichert.');
        } else report('Google Drive hat den Abgleich nicht bestätigt. Die App versucht es erneut.');
      } finally { if (upload === attempt) upload = null; }
    },
  };
}
