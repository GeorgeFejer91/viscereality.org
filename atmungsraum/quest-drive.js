export const questDriveScope = 'https://www.googleapis.com/auth/drive.file';
const apiOrigin = 'https://www.googleapis.com';
const filePath = '/drive/v3/files';
const uploadPath = '/upload/drive/v3/files';
const folderMime = 'application/vnd.google-apps.folder';
const fields = 'id,name,mimeType,size,parents,sha256Checksum,ownedByMe,shared,trashed';
const safeId = /^[A-Za-z0-9_-]+$/;
const chunkSize = 256 * 1024;
const batchSize = 4 * chunkSize;
const expiredStatuses = [404, 410];

function uploadDestination(location) {
  const url = location && new URL(location, apiOrigin);
  if (!url || url.origin !== apiOrigin || url.pathname !== uploadPath || url.username || url.password
    || url.hash || url.searchParams.getAll('upload_id').length !== 1 || !url.searchParams.get('upload_id')) {
    throw Error('Ungültige Google-Drive-Upload-Adresse.');
  }
  return url.href;
}

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

  async function beginUpload(receipt, name, type, length, signal) {
    const headers = { 'Content-Type': 'application/json', 'X-Upload-Content-Type': type };
    if (length !== undefined) headers['X-Upload-Content-Length'] = String(length);
    const { driveId, folderId } = receipt;
    const response = await request(`${uploadPath}?uploadType=resumable`, signal, { method: 'POST', expected: [409], headers,
      body: JSON.stringify({ id: driveId, name, mimeType: type, parents: [folderId] }) });
    return response.status === 409 ? null : uploadDestination(response.headers.get('Location'));
  }

  async function putRange(uploadUrl, type, range, signal, body) {
    const response = await request(uploadDestination(uploadUrl), signal, { method: 'PUT', body, expected: expiredStatuses,
      headers: { 'Content-Type': type, 'Content-Range': range, 'X-GUploader-No-308': 'yes' } });
    if (expiredStatuses.includes(response.status)) return { expired: true };
    // Google's upload server exposes its incomplete status without an HTTP redirect.
    if (response.headers.get('X-HTTP-Status-Code-Override') !== '308') {
      if (response.status === 200 || response.status === 201) return { complete: true };
      throw Error('Google Drive hat den Upload-Status nicht bestätigt.');
    }
    const header = response.headers.get('Range');
    const lastByte = header === null ? -1 : Number(/^bytes=0-(\d+)$/.exec(header)?.[1]);
    const offset = lastByte + 1;
    if (!Number.isSafeInteger(offset) || offset < 0) throw Error('Ungültiger Google-Drive-Bytebereich.');
    return { offset };
  }

  async function verifyFile(remote, receipt, file, signal) {
    const { driveId, folderId, sha256: checksum } = receipt;
    checkPrivateFile(remote, driveId);
    if (remote.name !== file.name || remote.mimeType !== file.type || String(file.size) !== remote.size
      || remote.parents?.length !== 1 || remote.parents[0] !== folderId) throw Error('Google Drive hat eine andere Datei bestätigt.');
    const actual = remote.sha256Checksum ?? await sha256(await (await request(`${filePath}/${driveId}?alt=media`, signal,
      { timeout: 300000 })).blob());
    if (actual !== checksum) throw Error('Die Google-Drive-Prüfsumme stimmt nicht überein.');
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
      const { driveId } = receipt;
      let remote = await getFile(driveId, signal);
      if (!remote) {
        const destination = await beginUpload(receipt, file.name, file.type, file.size, signal);
        if (destination) {
          await request(destination, signal, { method: 'PUT', headers: { 'Content-Type': file.type },
            body: file, expected: [409], timeout: 300000 });
        }
        remote = await getFile(driveId, signal);
      }
      await verifyFile(remote, receipt, file, signal);
    },
    async advance(receipt, available, read, persist, signal, file) {
      const { driveId, name, type } = receipt;
      const total = file ? String(file.size) : '*';
      const complete = async () => {
        if (!file) throw Error('Google Drive hat die Datei vorzeitig abgeschlossen.');
        await verifyFile(await getFile(driveId, signal), receipt, file, signal);
      };
      if (file) {
        const remote = await getFile(driveId, signal);
        if (remote) return verifyFile(remote, receipt, file, signal);
      }
      let { uploadUrl, offset = 0, pendingEnd = null } = receipt;
      if (!Number.isSafeInteger(offset) || offset < 0 || offset >= available) {
        throw Error('Ungültiger Google-Drive-Bytebereich.');
      }
      const checkpoint = () => persist({ ...receipt, uploadUrl, offset, pendingEnd });
      const acknowledge = (next, upperBound) => {
        if (!Number.isSafeInteger(next) || next < offset || next > upperBound) {
          throw Error('Ungültiger Google-Drive-Bytebereich.');
        }
        offset = next;
      };
      if (uploadUrl && pendingEnd !== null) {
        const status = await putRange(uploadUrl, type, `bytes */${total}`, signal);
        if (status.complete) return complete();
        if (status.expired) { uploadUrl = null; offset = 0; }
        else acknowledge(status.offset, Math.min(pendingEnd, file ? available - 1 : available));
        pendingEnd = null;
        await checkpoint();
      }
      if (!uploadUrl) {
        uploadUrl = await beginUpload(receipt, name, type, file?.size, signal);
        if (!uploadUrl) return complete();
        await checkpoint();
      }
      do {
        const remaining = available - offset;
        // Keep a tail so the same object can be finalized when its length is known.
        const length = file ? Math.min(batchSize, remaining) : Math.min(batchSize, Math.floor((remaining - 1) / chunkSize) * chunkSize);
        if (length <= 0) break;
        signal.throwIfAborted();
        const end = offset + length;
        const body = await read(offset, end);
        if (!(body instanceof Blob) || body.size !== length) throw Error('Der gespeicherte Bytebereich ist unvollständig.');
        pendingEnd = end;
        await checkpoint();
        const status = await putRange(uploadUrl, type, `bytes ${offset}-${end - 1}/${total}`, signal, body);
        if (status.complete) {
          if (!file || end !== available) throw Error('Google Drive hat die Datei vorzeitig abgeschlossen.');
          return complete();
        }
        if (status.expired) {
          await persist({ ...receipt, uploadUrl: null, offset: 0, pendingEnd: null });
          throw Error('Die Google-Drive-Uploadsitzung ist abgelaufen.');
        }
        if (status.offset === offset) throw Error('Google Drive hat keine weiteren Bytes bestätigt.');
        acknowledge(status.offset, file ? Math.min(end, available - 1) : end);
        pendingEnd = null;
        await checkpoint();
      } while (file);
      if (file) throw Error('Google Drive hat die Datei nicht abgeschlossen.');
    },
  };
}

export async function deliverQuestPrefix({ drive, store, folderId, id, name, type, available, read, signal }) {
  if (!Number.isSafeInteger(available) || available < 0) throw Error('Ungültige Aufnahmelänge.');
  if (available <= chunkSize) return;
  const saved = await store.readDelivery(id);
  if (saved && (saved.folderId !== folderId || (saved.name && (saved.name !== name || saved.type !== type)))) {
    throw Error('Die gespeicherte Upload-Zuordnung stimmt nicht überein.');
  }
  if (saved?.verified || saved?.sha256) return;
  const receipt = saved ?? { folderId, driveId: await drive.reserveId(signal), name, type, offset: 0, verified: false };
  const persist = value => store.saveDelivery(id, value);
  if (!saved) await persist(receipt);
  await drive.advance(receipt, available, read, persist, signal);
}

export async function deliverQuestBundle({ drive, store, folderId, id, file, signal }) {
  const saved = await store.readDelivery(id);
  if (saved?.folderId === folderId && saved.verified) return;
  const checksum = await sha256(file);
  signal.throwIfAborted();
  const receipt = { ...(saved ?? { folderId, driveId: await drive.reserveId(signal), verified: false }), sha256: checksum };
  if (file.type === 'text/csv' && !receipt.name) Object.assign(receipt, { name: file.name, type: file.type });
  if (receipt.folderId !== folderId || (saved?.sha256 && saved.sha256 !== checksum)
    || (saved?.name && (saved.name !== file.name || saved.type !== file.type))) {
    throw Error('Die gespeicherte Upload-Zuordnung stimmt nicht überein.');
  }
  const persist = value => store.saveDelivery(id, value);
  if (!saved?.sha256) await persist(receipt);
  signal.throwIfAborted();
  if (receipt.name) await drive.advance(receipt, file.size, async (start, end) => file.slice(start, end), persist, signal, file);
  else await drive.upload(receipt, file, signal);
  const { uploadUrl, offset, pendingEnd, ...confirmed } = receipt;
  await persist({ ...confirmed, verified: true });
}
