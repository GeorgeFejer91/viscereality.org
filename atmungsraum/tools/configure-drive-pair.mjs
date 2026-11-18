import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  check: { type: 'boolean' }, 'quest-device': { type: 'string' },
} });
const [quest, phone] = positionals;
if (positionals.length !== 2 || ![quest, phone].every(value => /^[A-Za-z0-9_-]+$/.test(value)) || quest === phone) {
  throw Error('Usage: node tools/configure-drive-pair.mjs <Quest USB serial> <Pixel USB serial> [--check] [--quest-device <verified app-installation ID>]');
}

const adb = (serial, ...args) => execFileSync('adb', ['-s', serial, ...args], { encoding: 'utf8', timeout: 8000 }).trim();
if (adb(quest, 'get-serialno') !== quest || adb(phone, 'get-serialno') !== phone
  || !/^Quest\b/.test(adb(quest, 'shell', 'getprop', 'ro.product.model'))) throw Error('Expected the selected USB Quest and phone.');

const id = createHash('sha256').update(JSON.stringify(['atmungsraum-usb-pair-v1', quest, phone])).digest('hex').slice(0, 24);
const port = adb(phone, 'forward', '--no-rebind', 'tcp:0', 'localabstract:chrome_devtools_remote');
let socket;
try {
  if (!/^\d+$/.test(port)) throw Error('Chrome debugging forward failed.');
  const list = await (await fetch('http://127.0.0.1:' + port + '/json/list', { signal: AbortSignal.timeout(5000) })).json();
  async function evaluate(page, expression) {
    const endpoint = new URL(page.webSocketDebuggerUrl);
    if (endpoint.protocol !== 'ws:' || endpoint.hostname !== '127.0.0.1' || endpoint.port !== port) {
      throw Error('Unexpected Chrome debugger endpoint.');
    }
    socket = new WebSocket(endpoint);
    try {
      const response = await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(Error('Chrome did not confirm the request.')), 8000);
        socket.addEventListener('open', () => socket.send(JSON.stringify({ id: 1, method: 'Runtime.evaluate',
          params: { expression, awaitPromise: true, returnByValue: true } })), { once: true });
        socket.addEventListener('message', event => {
          const message = JSON.parse(event.data);
          if (message.id === 1) { clearTimeout(timer); resolve(message); }
        });
        socket.addEventListener('error', () => { clearTimeout(timer); reject(Error('Chrome debugging is unavailable.')); }, { once: true });
      });
      if (response.error || response.result?.exceptionDetails) throw Error('Chrome could not read the PWA.');
      return response.result?.result?.value;
    } finally { socket.close(); socket = null; }
  }

  const entries = [];
  for (const page of list.filter(item => item.type === 'page' && item.url.startsWith('https://viscereality.org/atmungsraum/'))) {
    const state = await evaluate(page, `(async () => {
      if (!matchMedia('(display-mode: standalone)').matches) return null;
      const saved = JSON.parse(localStorage.getItem('atmungsraum-quest-pair'));
      const pairs = JSON.parse(localStorage.getItem('atmungsraum-drive-hardware-pairs')) ?? [];
      const hasRecordings = (await indexedDB.databases()).some(db => db.name === 'atmungsraum-quest-recordings');
      return { device: saved?.device, serial: saved?.serial, kind: saved?.identity_kind,
        configured: pairs.some(p => p.device === saved?.device && p.serial === saved?.serial && p.id === ${JSON.stringify(id)}),
        driveConfigured: hasRecordings && await new Promise((resolve, reject) => {
          const request = indexedDB.open('atmungsraum-quest-recordings');
          request.onerror = () => reject(request.error);
          request.onsuccess = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains('drive')) { db.close(); return resolve(false); }
            const get = db.transaction('drive').objectStore('drive').get('destination');
            get.onerror = () => { db.close(); reject(get.error); };
            get.onsuccess = () => { db.close(); resolve(Boolean(get.result)); };
          };
        }),
        driveStatus: document.getElementById('quest-drive-status')?.textContent,
        release: new URL(document.querySelector('script[type="module"][src*="app.js?v="]').src).searchParams.get('v') };
    })()`);
    if (state) entries.push({ page, state });
  }
  if (entries.length !== 1) throw Error('Open exactly one installed Atmungsraum PWA on the selected phone.');
  const [{ page, state }] = entries;
  const matches = state.serial === quest || (state.kind === 'app-installation' && state.serial === ''
    && /^[a-f0-9]{32}$/.test(values['quest-device']) && state.device === values['quest-device']);
  if (values.check) {
    console.log(JSON.stringify({ pairFolder: 'Pair_' + id, ...state, matchesUsbQuest: matches }, null, 2));
  } else {
    if (!state.device || !matches) throw Error('The installed PWA pairing does not identify the selected Quest. Pair this headset first. For a serial-less installation supply its verified --quest-device ID.');
    const binding = JSON.stringify({ device: state.device, serial: state.serial, id });
    const confirmed = await evaluate(page, `(() => {
      const binding = ${binding};
      const saved = JSON.parse(localStorage.getItem('atmungsraum-quest-pair'));
      if (saved?.device !== binding.device || saved?.serial !== binding.serial) return false;
      const key = 'atmungsraum-drive-hardware-pairs';
      const pairs = JSON.parse(localStorage.getItem(key)) ?? [];
      const next = pairs.filter(p => p.device !== binding.device || p.serial !== binding.serial);
      localStorage.setItem(key, JSON.stringify([...next, binding]));
      return JSON.parse(localStorage.getItem(key)).some(p => p.device === binding.device && p.serial === binding.serial && p.id === binding.id);
    })()`);
    if (!confirmed) throw Error('The PWA pairing changed while applying the configuration.');
    console.log('Configured Pair_' + id);
  }
} finally {
  socket?.close();
  if (/^\d+$/.test(port)) adb(phone, 'forward', '--remove', 'tcp:' + port);
}
