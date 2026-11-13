const storeName = 'recordings';
const driveStore = 'drive';
const prefixStore = 'prefix';
const rangeStore = 'prefix-bytes';

export async function openQuestRecordingStore({ device, serial }) {
  const scope = `${device}/${serial}/`;
  const db = await new Promise((resolve, reject) => {
    const request = indexedDB.open('atmungsraum-quest-recordings', 3);
    request.onupgradeneeded = ({ oldVersion }) => {
      if (oldVersion === 0) request.result.createObjectStore(storeName);
      if (oldVersion < 2) request.result.createObjectStore(driveStore);
      if (oldVersion < 3) {
        request.result.createObjectStore(prefixStore);
        request.result.createObjectStore(rangeStore);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  db.onversionchange = () => db.close();

  function query(mode, operation, name = storeName) {
    return new Promise((resolve, reject) => {
      const tx = db.transaction(name, mode, mode === 'readwrite' ? { durability: 'strict' } : undefined);
      const request = operation(tx.objectStore(name));
      tx.oncomplete = () => resolve(request.result);
      tx.onabort = () => reject(tx.error ?? request.error);
    });
  }

  function appendPrefix(prefix, start, parts) {
    const key = scope + prefix.id;
    return new Promise((resolve, reject) => {
      const tx = db.transaction([prefixStore, rangeStore], 'readwrite', { durability: 'strict' });
      const descriptors = tx.objectStore(prefixStore);
      const ranges = tx.objectStore(rangeStore);
      const existing = descriptors.get(key);
      existing.onsuccess = () => {
        if ((existing.result?.offset ?? 0) !== start) {
          tx.abort();
          return;
        }
        let offset = start;
        for (const part of parts) {
          ranges.add(part, [key, offset]);
          offset += part.size;
        }
        descriptors.put({ ...prefix, offset }, key);
      };
      tx.oncomplete = resolve;
      tx.onabort = () => reject(tx.error ?? Error('Laufende Quest-Aufnahme wurde gleichzeitig geändert.'));
    });
  }

  function readPrefixBytes(id, start, end) {
    const key = scope + id;
    if (end === start) return Promise.resolve(new Blob());
    return new Promise((resolve, reject) => {
      const tx = db.transaction(rangeStore, 'readonly');
      const ranges = tx.objectStore(rangeStore);
      const previous = ranges.openCursor(IDBKeyRange.bound([key, 0], [key, start]), 'prev');
      const parts = [];
      let offset = start;
      previous.onsuccess = () => {
        const first = previous.result;
        if (!first) return;
        const cursor = ranges.openCursor(IDBKeyRange.bound(first.key, [key, end], false, true));
        cursor.onsuccess = () => {
          const current = cursor.result;
          if (!current || offset >= end) return;
          const at = current.key[1];
          if (at > offset || at + current.value.size <= offset) return;
          const part = current.value.slice(offset - at, end - at);
          parts.push(part);
          offset += part.size;
          current.continue();
        };
      };
      tx.oncomplete = () => offset === end ? resolve(new Blob(parts))
        : reject(Error('Der gespeicherte Quest-Bytebereich ist unvollständig.'));
      tx.onabort = () => reject(tx.error);
    });
  }

  function finalizePrefix(id, archive) {
    const key = scope + id;
    return new Promise((resolve, reject) => {
      const tx = db.transaction([storeName, prefixStore, rangeStore], 'readwrite', { durability: 'strict' });
      tx.objectStore(storeName).put(archive, key);
      tx.objectStore(prefixStore).delete(key);
      tx.objectStore(rangeStore).delete(IDBKeyRange.bound([key, 0], [key, Number.MAX_SAFE_INTEGER]));
      tx.oncomplete = resolve;
      tx.onabort = () => reject(tx.error);
    });
  }

  return {
    async ids() {
      const keys = await query('readonly', store => store.getAllKeys(IDBKeyRange.bound(scope, scope + '\uffff')));
      return keys.reverse().map(key => key.slice(scope.length));
    },
    get: id => query('readonly', store => store.get(scope + id)),
    put: (id, archive) => query('readwrite', store => store.put(archive, scope + id)),
    readPrefix: id => query('readonly', store => store.get(scope + id), prefixStore),
    appendPrefix,
    readPrefixBytes,
    finalizePrefix,
    readDriveDestination: () => query('readonly', store => store.get('destination'), driveStore),
    saveDriveDestination: (value) => query('readwrite', store => store.put(value, 'destination'), driveStore),
    readDelivery: id => query('readonly', store => store.get(scope + id), driveStore),
    saveDelivery: (id, value) => query('readwrite', store => store.put(value, scope + id), driveStore),
    close: () => db.close(),
  };
}
