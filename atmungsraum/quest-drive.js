export const questDriveScope = 'https://www.googleapis.com/auth/drive.file';
const apiOrigin = 'https://www.googleapis.com';
const filePath = '/drive/v3/files';
const uploadPath = '/upload/drive/v3/files';
const folderMime = 'application/vnd.google-apps.folder';
const fields = 'id,name,mimeType,size,parents,sha256Checksum,ownedByMe,shared,trashed';
const safeId = /^[A-Za-z0-9_-]+$/;

async function sha256(file) {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', await file.arrayBuffer()));
  return Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('');
}

export function createQuestDrive(token, fetcher = fetch) {
  async function request(path, signal, { method = 'GET', headers, body, expected = [], timeout = 30000 } = {}) {
    const url = new URL(path, apiOrigin);
    if (url.origin !== apiOrigin || url.username || url.password) throw Error('Ungültige Google-Drive-Adresse.');
    const requestHeaders = new Headers(headers);
    requestHeaders.set('Authorization', `Bearer ${token}`);
    const response = await fetcher(url.href, { method, headers: requestHeaders, body, redirect: 'error',
      credentials: 'omit', cache: 'no-store', signal: AbortSignal.any([signal, AbortSignal.timeout(timeout)]) });
    if (!response.ok && !expected.includes(response.status)) {
      throw Object.assign(Error('Google Drive hat den Vorgang nicht bestätigt.'), { status: response.status });
    }
    return response;
  }

  async function getFile(id, signal) {
    if (typeof id !== 'string' || !safeId.test(id)) throw Error('Ungültige Google-Drive-Datei-ID.');
    const response = await request(`${filePath}/${id}?fields=${fields}`, signal, { expected: [404] });
    return response.status === 404 ? null : response.json();
  }

  function checkPrivateFile(file, id) {
    if (file?.id !== id || file.ownedByMe !== true || file.shared !== false || file.trashed !== false) {
      throw Error('Der private Google-Drive-Speicherort konnte nicht bestätigt werden.');
    }
  }

  return {
    async account(signal) {
      const response = await request('/drive/v3/about?fields=user(emailAddress,permissionId)', signal);
      const { user } = await response.json();
      if (!user?.emailAddress || typeof user.permissionId !== 'string' || !safeId.test(user.permissionId)) {
        throw Error('Das Google-Konto konnte nicht bestätigt werden.');
      }
      return user;
    },
    async reserveId(signal) {
      const response = await request(`${filePath}/generateIds?count=1&space=drive`, signal);
      const { ids } = await response.json();
      if (typeof ids?.[0] !== 'string' || !safeId.test(ids[0])) throw Error('Google Drive hat keine Datei-ID geliefert.');
      return ids[0];
    },
    async ensureFolder(id, signal) {
      let folder = await getFile(id, signal);
      if (!folder) {
        await request(filePath, signal, { method: 'POST', expected: [409], headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id, name: 'Atmungsraum – Quest-Aufnahmen', mimeType: folderMime }) });
        folder = await getFile(id, signal);
      }
      checkPrivateFile(folder, id);
      if (folder.mimeType !== folderMime) throw Error('Das Google-Drive-Ziel ist kein Ordner.');
    },
    async upload(receipt, file, signal) {
      const { driveId, folderId, sha256: checksum } = receipt;
      let remote = await getFile(driveId, signal);
      if (!remote) {
        const response = await request(`${uploadPath}?uploadType=resumable`, signal, { method: 'POST', expected: [409],
          headers: { 'Content-Type': 'application/json', 'X-Upload-Content-Type': file.type,
            'X-Upload-Content-Length': String(file.size) },
          body: JSON.stringify({ id: driveId, name: file.name, mimeType: file.type, parents: [folderId] }) });
        if (response.status !== 409) {
          const location = response.headers.get('Location');
          const destination = location && new URL(location, apiOrigin);
          if (!destination || destination.origin !== apiOrigin || destination.pathname !== uploadPath
            || !destination.searchParams.has('upload_id')) throw Error('Ungültige Google-Drive-Upload-Adresse.');
          await request(destination.href, signal, { method: 'PUT', headers: { 'Content-Type': file.type },
            body: file, expected: [409], timeout: 300000 });
        }
        remote = await getFile(driveId, signal);
      }
      checkPrivateFile(remote, driveId);
      if (remote.name !== file.name || remote.mimeType !== file.type || String(file.size) !== remote.size
        || remote.parents?.length !== 1 || remote.parents[0] !== folderId) throw Error('Google Drive hat eine andere Datei bestätigt.');
      const actual = remote.sha256Checksum ?? await sha256(await (await request(`${filePath}/${driveId}?alt=media`, signal,
        { timeout: 300000 })).blob());
      if (actual !== checksum) throw Error('Die Google-Drive-Prüfsumme stimmt nicht überein.');
    },
  };
}

export async function deliverQuestBundle({ drive, store, folderId, id, file, signal }) {
  const saved = await store.readDelivery(id);
  if (saved?.folderId === folderId && saved.verified) return;
  const checksum = await sha256(file);
  signal.throwIfAborted();
  const receipt = saved ?? { folderId, driveId: await drive.reserveId(signal), sha256: checksum, verified: false };
  if (receipt.folderId !== folderId || receipt.sha256 !== checksum) throw Error('Die gespeicherte Upload-Zuordnung stimmt nicht überein.');
  if (!saved) await store.saveDelivery(id, receipt);
  signal.throwIfAborted();
  await drive.upload(receipt, file, signal);
  await store.saveDelivery(id, { ...receipt, verified: true });
}
