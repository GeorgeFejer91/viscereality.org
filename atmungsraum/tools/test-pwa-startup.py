"""Exercise a cached PWA launch against a newly published release in Chromium."""
import re
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from time import sleep
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
PROBE = '_pwa'
OLD = b'./app.js?v=pwa-test-old'
NEW = b'./app.js?v=pwa-test-new'


@contextmanager
def releases():
    index = (ROOT / 'atmungsraum/index.html').read_bytes()
    app_url = re.search(rb'src="(\./app\.js\?v=[^"]+)"', index)[1]
    current = {'app_url': OLD, 'probe_result': None, 'probes': [], 'entries': 0}

    class Site(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            request = urlsplit(self.path)
            if request.path != '/atmungsraum/':
                try:
                    return super().do_GET()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
            probe = parse_qs(request.query).get(PROBE)
            if probe:
                current['probes'].append(probe[0])
            else:
                current['entries'] += 1
            result = current['probe_result'] if probe else None
            if result == 'error':
                self.send_error(503)
                return
            if result == 'slow':
                sleep(3)
            body = index.replace(app_url, NEW if result == 'slow' else current['app_url'])
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'max-age=600')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                if result != 'slow':
                    raise

    with ThreadingHTTPServer(('127.0.0.1', 0), partial(Site, directory=str(ROOT))) as server:
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield f'http://127.0.0.1:{server.server_port}/atmungsraum/?live-sync=off#daten', current
        finally:
            server.shutdown()


QUEST_STARTUP = r"""
const device = 'd'.repeat(32), key = 'b'.repeat(64), controller = 'a'.repeat(32), build = 'c'.repeat(64);
const oldIp = '192.168.10.3', newIp = '192.168.10.9';
const gateway = ip => 'http://' + ip + ':8766/v1/gateway';
const mode = new URL(location.href).searchParams.get('fixture');
const savedSearch = mode === 'saved-search-release';
const unreachable = mode === 'lan-unreachable';
const legacy = mode.startsWith('legacy-commissioning');
const owner = unreachable || legacy ? {v: 2, device, serial: '', epoch: '1', native_epoch: '2', identity_kind: 'app-installation'}
  : {v: 1, device, serial: 'TEST-QUEST', epoch: '1', native_epoch: '2'};
const mark = value => sessionStorage.events = JSON.stringify([...JSON.parse(sessionStorage.events || '[]'), value]);
const storage = 'atmungsraum-quest-pair';
const savedOwner = legacy ? {device: '0'.repeat(32), serial: 'OLD-QUEST', epoch: '1', native_epoch: '2'} : owner;
localStorage.setItem(storage, localStorage.getItem(storage) || JSON.stringify({
  ...savedOwner, enrollment: {controller, key}, bluetoothId: 'saved', build,
  ...(savedSearch ? {} : {endpoint: gateway(oldIp)})
}));
if (mode === 'legacy-commissioning-pending') {
  sessionStorage.setItem('atmungsraum-quest-pending', JSON.stringify({
    owner: savedOwner, op: 'recenter', id: '1'.repeat(16), generation: 0,
  }));
}
const encode = new TextEncoder();
const metadata = {...owner, mode: 'remote', gateways: [newIp],
  ...(unreachable || legacy ? {pairing: 1, rtc: {r: 'e'.repeat(32), p: 'f'.repeat(64)}} : {})};
const radio = new EventTarget();
radio.enabled = mode !== 'radio-off';
radio.getAvailability = async () => radio.enabled;
radio.getDevices = async () => [{
  id: 'other', gatt: {connect() { mark('wrong headset'); }}
}, {id: 'saved', gatt: {
  async connect() {
    mark('ble:' + document.querySelector('script[src^="./app.js?v="]').getAttribute('src'));
    if (mode === 'wifi-only') throw new DOMException('No GATT connection', 'NetworkError');
    this.connected = true;
    return this;
  },
  disconnect() { this.connected = false; },
  async getPrimaryService() {
    return {getCharacteristic: async id => ({readValue: async () => new DataView(encode.encode(
      id.startsWith('9a7b2001-7d6a-4b7f-9d4a-6f7c0a020102') ? owner.serial : JSON.stringify(metadata)).buffer)})};
  }
}}];
radio.requestDevice = async () => { mark('chooser'); throw Error('Unexpected chooser'); };
if (mode === 'companion' || savedSearch) delete radio.getDevices;
Object.defineProperty(navigator, 'bluetooth', {value: radio});
const fetchPage = window.fetch.bind(window);
let witness = 0;
window.fetch = async (url, init) => {
  const address = new URL(url, location.href);
  if (address.searchParams.has('_pwa')) mark('update');
  if (address.port === '8767') {
    if (legacy && address.pathname === '/v1/commissioning') {
      mark('commissioning');
      return Response.json({v: 1, device, identity_kind: owner.identity_kind, serial: owner.serial});
    }
    if (legacy) {
      mark('companion:' + address.pathname);
      return Response.json(address.pathname.endsWith(device) ? {v: 1, state: 'ready', metadata}
        : {v: 1, state: 'searching'});
    }
    mark('companion');
    if (savedSearch) {
      await new Promise(resolve => window.finishSavedSearch = resolve);
      return Response.json({v: 1, state: 'searching'});
    }
    if (mode === 'companion') return Response.json({v: 1, state: 'ready', metadata});
    throw new TypeError('No companion');
  }
  if (address.port !== '8766') return fetchPage(url, init);
  if (!init.body) {
    mark('wifi:' + address.hostname);
    if (unreachable) throw new TypeError('Failed to fetch');
    if (address.hostname === oldIp && !['wifi-only', 'companion'].includes(mode)) throw new TypeError('Unreachable');
    return Response.json({...owner, enabled: true, remote_available: true});
  }
  const {action, nonce} = JSON.parse(init.body);
  if (action !== 'status') throw Error('Unexpected command: ' + action);
  if (legacy) mark('native-status');
  const payload = JSON.stringify({v: 1, wifi_snapshot: 1, build: {v: 1, packaging_fingerprint: build},
    recordings: 0, control_mode: 'remote', p: 'IDLE', g: 0, r: 0, ms: 0, pending: false, rec: false,
    ...(legacy ? {controller_access: {v: 1, role: 'active', active_phone: 'Phone', revision: 0}} : {}),
    o: {v: 1, s: 'observed', e: owner.native_epoch, i: String(++witness), a: 0, g: 0, r: 0}});
  const macKey = await crypto.subtle.importKey('raw', encode.encode(key),
    {name: 'HMAC', hash: 'SHA-256'}, false, ['sign']);
  const bytes = await crypto.subtle.sign('HMAC', macKey,
    encode.encode(['RQG1-REPLY', controller, device, owner.epoch, nonce, action, payload].join('|')));
  const mac = Array.from(new Uint8Array(bytes), b => b.toString(16).padStart(2, '0')).join('');
  return Response.json({v: 1, device, epoch: owner.epoch, nonce, action, payload,
    mac: mode === 'legacy-commissioning-rejected' ? '0'.repeat(64) : mac});
};
if (legacy) {
  let waitForStatus = ['legacy-commissioning', 'legacy-commissioning-superseded'].includes(mode);
  window.testCommissionedTransport = {
    async request(body) {
      if (body && waitForStatus) {
        waitForStatus = false;
        mark('waiting-for-native');
        await new Promise(resolve => window.releaseCommissionedStatus = resolve);
      }
      return window.fetch(gateway(newIp), {body: body ? JSON.stringify(body) : undefined});
    },
    close() { mark('transport-closed'); },
  };
}
"""


def verify_reconnect(browser, url, release):
    from playwright.sync_api import expect

    for mode in ('lan-unreachable', 'changed-address', 'radio-off', 'companion', 'wifi-only', 'recovery'):
        target = url.replace('?live-sync=off#daten', '?fixture=' + mode + ('#daten' if mode == 'recovery' else ''))
        release.update(app_url=OLD, probe_result='error' if mode == 'wifi-only' else None)
        context = browser.new_context(viewport={'width': 320, 'height': 568}, is_mobile=True, has_touch=True)
        context.add_init_script(QUEST_STARTUP)
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        if mode == 'lan-unreachable':
            page.route('**/quest-webrtc.js?v=*', lambda route: route.fulfill(
                content_type='text/javascript',
                body='export async function openQuestWebRtc() { throw Error("Signaling unavailable"); }'))
        if mode == 'changed-address':
            old_page = page.request.get(target).body()
            page.route(re.compile(re.escape(target) + '$'), lambda route:
                       route.fulfill(content_type='text/html', body=old_page))
            release['app_url'] = NEW
        page.goto(target)
        if mode == 'radio-off':
            expect(page.locator('#quest-status')).to_contain_text('Schalte Bluetooth ein')
            expect(page.locator('#check-bluetooth')).to_have_attribute('data-check', 'unconfirmed')
            expect(page.locator('#quest-status')).to_have_attribute('data-text-fit', 'fit')
            assert page.evaluate('document.documentElement.scrollWidth === innerWidth')
            page.evaluate("""() => {
              navigator.bluetooth.enabled = true;
              navigator.bluetooth.dispatchEvent(new Event('availabilitychanged'));
            }""")
        if mode == 'lan-unreachable':
            page.evaluate("document.documentElement.style.fontSize='200%'")
            status = page.locator('#quest-status')
            expect(status).to_have_text('Quest nicht erreichbar. Prüfe Bluetooth, WLAN und den Fernmodus.')
            expect(status).to_have_attribute('data-text-fit', 'fit')
            assert page.evaluate('document.documentElement.scrollWidth === innerWidth')
            events = page.evaluate('JSON.parse(sessionStorage.events)')
            assert events[-1] == 'wifi:192.168.10.9' and 'chooser' not in events, events
        elif mode == 'recovery':
            expect(page.locator('#data-status')).to_contain_text('Keine früheren Browser-Aufnahmen')
            assert page.evaluate('JSON.parse(sessionStorage.events)') == ['update']
        else:
            expect(page.locator('#check-bridge')).to_have_attribute('data-check', 'ok', timeout=3000)
            events = page.evaluate('JSON.parse(sessionStorage.events)')
            assert events[0] == 'update', events
            assert 'chooser' not in events and 'wrong headset' not in events, events
            if mode == 'changed-address':
                assert events == ['update', 'ble:' + NEW.decode(), 'wifi:192.168.10.9'], events
            if mode == 'companion':
                assert events == ['update', 'companion', 'wifi:192.168.10.9'], events
            if mode == 'wifi-only':
                assert events.index('ble:' + OLD.decode()) < events.index('wifi:192.168.10.3'), events
            ip = '192.168.10.3' if mode == 'wifi-only' else '192.168.10.9'
            assert page.evaluate("JSON.parse(localStorage.getItem('atmungsraum-quest-pair')).endpoint") == (
                'http://' + ip + ':8766/v1/gateway')
        assert not errors, errors
        context.close()
    print('PASS ordered startup, saved headset, changed address, companion, Bluetooth recovery and Wi-Fi fallback')
    print('PASS failed signaling and LAN use fitted connection guidance at 320px and 200% text')


def verify_legacy_commissioning(browser, url, release):
    from playwright.sync_api import expect

    for suffix in ('', '-rejected', '-pending', '-superseded', '-questionnaire'):
        target = url.replace('?live-sync=off#daten', '?fixture=legacy-commissioning' + suffix)
        release.update(app_url=OLD, probe_result=None)
        context = browser.new_context()
        context.add_init_script(QUEST_STARTUP)
        page = context.new_page()
        page.route('**/quest-webrtc.js?v=*', lambda route: route.fulfill(
            content_type='text/javascript', body='''export async function openQuestWebRtc(discovery, enrollment) {
              if (enrollment || discovery.owner.device !== 'd'.repeat(32)) throw Error('Unexpected identity');
              return {enrollment: {controller: 'a'.repeat(32), key: 'b'.repeat(64)}, transport: window.testCommissionedTransport};
            }'''))
        page.clock.install()
        if suffix == '-questionnaire':
            page.goto(target + '#daten')
            page.evaluate("""async () => {
              const { openQuestRecordingStore } = await import('./quest-recording-store.js?v=20261118-post-session-questions');
              const store = await openQuestRecordingStore({device: '0'.repeat(32), serial: 'OLD-QUEST'});
              try {
                await store.finalizePrefix('old-recording', {
                  questionnaire: {required: true, run: {session_id: 'old-session'}},
                });
              } finally { store.close(); }
            }""")
            target += '&prepared=1'
        page.goto(target)
        if suffix in ('', '-superseded'):
            page.wait_for_function('Boolean(window.releaseCommissionedStatus)', timeout=5000)
            assert page.evaluate("JSON.parse(localStorage.getItem('atmungsraum-quest-pair')).serial") == 'OLD-QUEST'
            if suffix == '-superseded':
                page.evaluate("""() => {
                  const key = 'atmungsraum-quest-pair';
                  const other = {...JSON.parse(localStorage.getItem(key)), serial: 'RECENTLY-SELECTED'};
                  localStorage.setItem(key, JSON.stringify(other));
                  window.dispatchEvent(new StorageEvent('storage', {key}));
                }""")
            page.evaluate('window.releaseCommissionedStatus()')
            if suffix == '-superseded':
                page.wait_for_function("JSON.parse(sessionStorage.events).includes('transport-closed')")
            else:
                expect(page.locator('#check-bridge')).to_have_attribute('data-check', 'ok')
        elif suffix == '-rejected':
            expect(page.locator('#quest-status')).to_have_text('Die Quest-Antwort konnte nicht bestätigt werden.')
        elif suffix == '-questionnaire':
            expect(page.locator('#questionnaire-screen')).to_be_visible()
            page.clock.run_for(1500)
        else:
            expect(page.locator('#quest-status')).to_contain_text('Suche die gespeicherte Quest')
            page.clock.run_for(1500)
            assert page.evaluate("JSON.parse(sessionStorage.getItem('atmungsraum-quest-pending')).id") == '1' * 16

        state = page.evaluate("""() => ({
            pair: JSON.parse(localStorage.getItem('atmungsraum-quest-pair')),
            events: JSON.parse(sessionStorage.events),
        })""")
        assert 'chooser' not in state['events'], state['events']
        if not suffix:
            assert state['pair']['identity_kind'] == 'app-installation' and state['pair']['device'] == 'd' * 32
            assert state['events'].index('companion:/v1/quest/' + '0' * 32) < state['events'].index('commissioning')
        elif suffix == '-superseded':
            assert state['pair']['serial'] == 'RECENTLY-SELECTED'
        else:
            assert state['pair']['serial'] == 'OLD-QUEST'
            assert ('commissioning' in state['events']) == (suffix == '-rejected')
        context.close()
    print('PASS stale legacy pairing uses verified commissioning and preserves rejected or pending records')


def run():
    with releases() as (url, release), sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
        errors = []
        context.on('page', lambda page: page.on('pageerror', lambda error: errors.append(str(error))))

        def launch(target=url):
            page = context.new_page()
            page.goto(target)
            page.wait_for_function("document.documentElement.dataset.measurement === 'ready'", timeout=5000)
            return page

        def version(page):
            return page.locator('script[src^="./app.js?v="]').get_attribute('src').encode()

        page = launch()
        assert version(page) == OLD
        assert len(set(release['probes'])) == 1, release['probes']
        page.evaluate("localStorage.setItem('pwa-test-sentinel', 'original')")
        page.close()

        release['app_url'] = NEW
        page = launch()
        assert release['entries'] == 1, 'launch must begin with cached HTML'
        assert version(page) == NEW, 'a cached launch must switch to the published release'
        assert page.url == url and page.locator('#daten').is_visible()
        assert page.evaluate("localStorage.getItem('pwa-test-sentinel')") == 'original'
        assert len(set(release['probes'])) == 2, release['probes']

        release['probe_result'] = 'error'
        fallback = url.replace('live-sync=off', 'live-sync=off&check=fallback')
        page = launch(fallback)
        assert page.url == fallback and version(page) == NEW
        assert len(set(release['probes'])) == 3, release['probes']

        release.update(app_url=OLD, probe_result='slow')
        slow = fallback.replace('check=fallback', 'check=slow')
        page = launch(slow)
        assert page.url == slow and version(page) == OLD
        assert len(set(release['probes'])) == 4, release['probes']
        release.update(app_url=OLD, probe_result=None)
        idle = url.replace('#daten', '').replace('live-sync=off', 'live-sync=off&check=return')
        page = launch(idle)
        page.locator('#quest-find').wait_for(state='visible')
        assert version(page) == OLD
        probe_count = len(set(release['probes']))
        release['app_url'] = NEW
        page.locator('#settings-open').click()
        page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
        page.wait_for_timeout(250)
        assert version(page) == OLD and len(set(release['probes'])) == probe_count
        page.locator('#settings-screen [data-operator-return]').click()
        page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
        page.wait_for_selector('script[src="./app.js?v=pwa-test-new"]', state='attached', timeout=5000)
        page.wait_for_function("document.documentElement.dataset.operatorStep === 'connect' && !location.search.includes('_pwa=')")
        assert page.url == idle + '#connect' and page.locator('#quest').is_visible()
        assert len(set(release['probes'])) == probe_count + 1
        assert page.evaluate("localStorage.getItem('pwa-test-sentinel')") == 'original'
        assert not errors, errors
        print('PASS cached launch and idle resume update; settings, URL, storage and error/timeout fallback preserved')
        context.close()
        verify_reconnect(browser, url, release)
        verify_legacy_commissioning(browser, url, release)
        verify_idle_release_timer(browser, url, release)
        verify_release_during_saved_search(browser, url, release)


def verify_idle_release_timer(browser, url, release):
    release.update(app_url=OLD, probe_result=None)
    with browser.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True) as context:
        page = context.new_page()
        page.clock.install()
        idle = url.replace('#daten', '')
        page.goto(idle)
        page.locator('#quest-find').wait_for(state='visible')
        old_script = page.locator('script[src="./app.js?v=pwa-test-old"]')
        assert old_script.count() == 1
        page.evaluate("localStorage.setItem('pwa-test-sentinel', 'retained')")
        probe_count = len(release['probes'])
        release['app_url'] = NEW
        page.locator('#settings-open').click()
        page.clock.fast_forward(5 * 60 * 1000)
        assert old_script.count() == 1 and len(release['probes']) == probe_count
        assert page.locator('#settings-screen').is_visible()
        page.locator('#settings-screen [data-operator-return]').click()
        page.locator('#quest-find').wait_for(state='visible')
        page.clock.fast_forward(5 * 60 * 1000)
        page.wait_for_selector('script[src="./app.js?v=pwa-test-new"]', state='attached', timeout=5000)
        page.wait_for_function("document.documentElement.dataset.operatorStep === 'connect' && !location.search.includes('_pwa=')")
        assert page.url == idle + '#connect' and len(release['probes']) == probe_count + 1
        assert page.evaluate("localStorage.getItem('pwa-test-sentinel')") == 'retained'
        print('PASS open and idle Connect updates once without hiding Settings or dropping storage')


def verify_release_during_saved_search(browser, url, release):
    from playwright.sync_api import expect

    release.update(app_url=OLD, probe_result=None)
    with browser.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True) as context:
        context.add_init_script(QUEST_STARTUP)
        page = context.new_page()
        page.clock.install()
        target = url.replace('?live-sync=off#daten', '?fixture=saved-search-release')
        page.goto(target)
        page.wait_for_function("typeof window.finishSavedSearch === 'function'")
        expect(page.locator('#quest-find')).to_have_attribute('aria-disabled', 'true')
        saved = page.evaluate("localStorage.getItem('atmungsraum-quest-pair')")
        old_script = page.locator('script[src="./app.js?v=pwa-test-old"]')
        probe_count = len(release['probes'])
        release['app_url'] = NEW
        page.clock.fast_forward(5 * 60 * 1000)
        assert old_script.count() == 1 and len(release['probes']) == probe_count
        page.evaluate("window.finishSavedSearch()")
        page.clock.run_for(2200)
        page.wait_for_selector('script[src="./app.js?v=pwa-test-new"]', state='attached', timeout=5000)
        assert len(release['probes']) == probe_count + 1
        assert page.evaluate("localStorage.getItem('atmungsraum-quest-pair')") == saved
        print('PASS release check queued during a saved Bluetooth search runs before the next attempt')


if __name__ == '__main__':
    run()
