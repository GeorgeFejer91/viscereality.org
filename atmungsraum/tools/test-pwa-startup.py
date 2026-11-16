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


if __name__ == '__main__':
    run()
