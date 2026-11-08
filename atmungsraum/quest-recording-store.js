const storeName = 'recordings';

export async function openQuestRecordingStore({ device, serial }) {
  const scope = `${device}/${serial}/`;
  const db = await new Promise((resolve, reject) => {
    const request = indexedDB.open('atmungsraum-quest-recordings', 1);
    request.onupgradeneeded = () => request.result.createObjectStore(storeName);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  db.onversionchange = () => db.close();

  function query(mode, operation) {
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, mode, mode === 'readwrite' ? { durability: 'strict' } : undefined);
      const request = operation(tx.objectStore(storeName));
      tx.oncomplete = () => resolve(request.result);
      tx.onabort = () => reject(tx.error ?? request.error);
    });
  }

  return {
    async ids() {
      const keys = await query('readonly', store => store.getAllKeys(IDBKeyRange.bound(scope, scope + '\uffff')));
      return keys.reverse().map(key => key.slice(scope.length));
    },
    get: id => query('readonly', store => store.get(scope + id)),
    put: (id, archive) => query('readwrite', store => store.put(archive, scope + id)),
    close: () => db.close(),
  };
}
