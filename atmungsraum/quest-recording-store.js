const storeName = 'recordings';
const driveStore = 'drive';

export async function openQuestRecordingStore({ device, serial }) {
  const scope = `${device}/${serial}/`;
  const db = await new Promise((resolve, reject) => {
    const request = indexedDB.open('atmungsraum-quest-recordings', 2);
    request.onupgradeneeded = ({ oldVersion }) => {
      if (oldVersion === 0) request.result.createObjectStore(storeName);
      request.result.createObjectStore(driveStore);
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

  return {
    async ids() {
      const keys = await query('readonly', store => store.getAllKeys(IDBKeyRange.bound(scope, scope + '\uffff')));
      return keys.reverse().map(key => key.slice(scope.length));
    },
    get: id => query('readonly', store => store.get(scope + id)),
    put: (id, archive) => query('readwrite', store => store.put(archive, scope + id)),
    readDriveDestination: () => query('readonly', store => store.get('destination'), driveStore),
    saveDriveDestination: (value) => query('readwrite', store => store.put(value, 'destination'), driveStore),
    readDelivery: id => query('readonly', store => store.get(scope + id), driveStore),
    saveDelivery: (id, value) => query('readwrite', store => store.put(value, scope + id), driveStore),
    close: () => db.close(),
  };
}
