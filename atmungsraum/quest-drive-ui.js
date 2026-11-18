import { createQuestDrive, deliverQuestBundle, deliverQuestPrefix, questDriveRootName, questDriveScope } from './quest-drive.js?v=6';
import { openQuestRecordingStore } from './quest-recording-store.js?v=5';
import { collectionStart, questionnaireStart, recordingHeadBytes } from './collection-time.js?v=1';
import { datasetDestination, drivePairIdentity, ensureDrivePair } from './quest-drive-datasets.js?v=2';

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
                  const folderId = await drive.findFolder('root', questDriveRootName, signal) ?? await drive.reserveId(signal);
                  saved = { ...config, accountId: user.permissionId, folderId };
                  await store.saveDriveDestination(saved);
                }
                await drive.ensureFolder(saved.folderId, signal);
                const pair = await drivePairIdentity(selectedOwner);
                await ensureDrivePair({ drive, store, root: saved.folderId, pair, signal });
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
            const root = current.destination.folderId;
            let pair;
            async function destination(id, start, observed) {
              pair ??= await drivePairIdentity(selectedOwner);
              await current.drive.ensureFolder(root, signal);
              return datasetDestination({ drive: current.drive, store, root, pair, id, start, observed, signal });
            }
            if (prefix) {
              const staged = await store.readPrefix(prefix.id);
              if (staged?.session_id !== prefix.session_id || staged.offset < 256 * 1024 + 1) {
                due = 0;
                return;
              }
              const receipt = await store.readDelivery(prefix.id);
              if (!receipt?.verified && (receipt?.pendingEnd || staged.offset - (receipt?.offset ?? 0) > 256 * 1024)) {
                const stamp = staged.collectionStart ?? collectionStart(staged,
                  await store.readPrefixBytes(prefix.id, 0, Math.min(recordingHeadBytes, staged.offset)));
                let folderId = receipt?.folderId;
                if (!receipt || performance.now() >= current.checkFolderAfter) {
                  if (folderId === root) await current.drive.ensureFolder(root, signal);
                  else folderId = await destination(prefix.id, stamp, staged.collectionObserved);
                  if (receipt && folderId !== receipt.folderId) throw Error('Die gespeicherte Upload-Zuordnung stimmt nicht überein.');
                  current.checkFolderAfter = performance.now() + 30000;
                }
                await deliverQuestPrefix({ drive: current.drive, store, folderId, id: prefix.id, name: staged.name,
                  type: 'text/csv', available: staged.offset, signal, stamp, recoverExisting: folderId !== root,
                  read: (start, end) => store.readPrefixBytes(prefix.id, start, end) });
                const acknowledged = (await store.readDelivery(prefix.id))?.offset ?? 0;
                if (acknowledged) report('Google Drive empfängt die laufende Aufnahme.');
              }
              due = 0;
              return;
            }
            let saved = 0;
            async function uploadFile(folderId, id, file, stamp, resetDriveUpload = false) {
              signal.throwIfAborted();
              report('Dateien werden in Google Drive gespeichert …');
              await deliverQuestBundle({ drive: current.drive, store, folderId, id, file, signal, stamp, resetDriveUpload,
                recoverExisting: folderId !== root });
            }
            for (const id of (await store.ids()).reverse()) {
              signal.throwIfAborted();
              const receipt = await store.readDelivery(id);
              let folderId = receipt?.folderId;
              if (receipt?.verified) saved++;
              else {
                const entry = await store.get(id);
                if (entry?.bundleVersion === 1 && entry.bundle instanceof File) {
                  const stamp = entry.collectionStart ?? collectionStart({ id, name: entry.bundle.name },
                    entry.bundle.type === 'text/csv' && new Uint8Array(await entry.bundle.slice(0, recordingHeadBytes).arrayBuffer()));
                  if (folderId === root) await current.drive.ensureFolder(root, signal);
                  else folderId = await destination(id, stamp, entry.collectionObserved);
                  await uploadFile(folderId, id, entry.bundle, stamp, entry.resetDriveUpload === true);
                  saved++;
                }
              }
              const form = await store.getQuestionnaire(id);
              if (!(form?.file instanceof File)) continue;
              const answerId = id + '/questionnaire';
              const answerReceipt = await store.readDelivery(answerId);
              if (!answerReceipt?.verified) {
                if (answerReceipt?.folderId === root || folderId === root) {
                  folderId = root;
                  await current.drive.ensureFolder(root, signal);
                } else {
                  if (!folderId) continue;
                  const confirmed = await destination(id);
                  if (confirmed !== folderId) throw Error('Die gespeicherte Upload-Zuordnung stimmt nicht überein.');
                }
                await uploadFile(folderId, answerId, form.file, await questionnaireStart(form.file));
              }
              saved++;
            }
            signal.throwIfAborted();
            report(saved ? 'In Google Drive bestätigt: ' + saved : 'Google Drive verbunden. Warte auf Aufnahmen.');
            if (due !== 0) due = performance.now() + 60000;
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
