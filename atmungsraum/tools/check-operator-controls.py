"""Rendered operator controls with synthetic transport and real locked Pretext."""
from argparse import ArgumentParser
from pathlib import Path
import json
from playwright.sync_api import sync_playwright, expect
import importlib.util

spec = importlib.util.spec_from_file_location('capture', Path(__file__).with_name('capture-phone-screens.py'))
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def main():
    parser = ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'screens').mkdir(exist_ok=True)
    paired = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({capture.PAIRED}));"
    base = paired + """window.reviewInitialQuest={polar:'ready',deferSetup:true,
      access:{v:1,role:'active',active_phone:'Fixture',revision:1},
      kiosk:{v:1,g:2,requested:true,effective:false,state:'interrupted',service:'failed',
        blocked:'service-failed',detail:'Accessibility service requires reconnection',retry_available:true}};
      window.reviewInitialSetup={stage:'testing',input:'controller-state',ready:true,
        calibration:'failed',failure:'clock_stalled',attempt:2,target_frames:1411200,
        completed_frames:1411200,playing:true,input_volume01:.1,input_phase:'hold'};"""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        review = capture.Review(browser, args.output)
        for width, height, font in [(390, 844, '100%'), (320, 568, '200%'), (1280, 900, '100%')]:
            label = f'{width}-{font.strip("%")}'
            page = review.page(base, viewport={'width': width, 'height': height}, reduced_motion='no-preference')
            page.evaluate('font=>document.documentElement.style.fontSize=font', font)
            capture.enter_participant(page)
            expect(page.locator('#preview-title')).to_have_text('Controller testen')
            expect(page.locator('#preview-input-label')).to_contain_text('Bestätigte Eingabe: Controller State')
            expect(page.locator('#preview-progress-group')).to_be_hidden()
            expect(page.locator('#preview-detail')).not_to_contain_text('fehlgeschlagen')
            expect(page.locator('#preview-controller')).to_be_hidden()
            expect(page.locator('#preview-calibrate')).to_have_text('Zu Polar wechseln und einrichten')
            review.shot(page, f'controller-{label}', f'Confirmed Controller {label}', page.locator('#preview-calibrate'), 'Operator controls')
            page.locator('#preview-calibrate').click()
            expect(page.locator('#preview-switch')).to_contain_text('Angefragt: Polar einrichten')
            expect(page.locator('#preview-input-label')).to_contain_text('Bestätigte Eingabe: Controller State')
            expect(page.locator('#preview-recovery')).to_be_visible()
            page.clock.run_for(14000)
            expect(page.locator('#preview-acknowledge')).to_have_text('Bestätigung erneut abfragen')
            expect(page.locator('#preview-recovery-detail')).to_contain_text('Atmungsraum-App im Headset')
            page.locator('#preview-acknowledge').click()
            page.clock.run_for(1100)
            assert page.evaluate('window.reviewSetupCommands.length') == 1
            review.shot(page, f'pending-{label}', f'Pending Polar switch, read-only recheck {label}', page.locator('#preview-recovery'), 'Operator controls')
            page.context.close()
            renderer = {'v': 1, 'progress01': .2, 'radius_m': 1.4, 'frame': 1, 'render_generation': 2,
                        'settings_revision': 0, 'observed_ns': '1000000000', 'age_ns': 0}
            page = review.page(base + 'window.reviewInitialSetup.renderer=' + json.dumps(renderer) + ';',
                               viewport={'width': width, 'height': height}, reduced_motion='no-preference')
            page.evaluate('font=>document.documentElement.style.fontSize=font', font)
            capture.enter_participant(page)
            page.clock.pause_at(page.evaluate('Date.now()') + 10)
            renderer = {**renderer, 'frame': 2, 'observed_ns': '2000000000'}
            page.evaluate('renderer=>window.reviewQuest.setup.renderer=renderer', renderer)
            for _ in range(25):
                page.clock.run_for(50)
                if 'aktuell:' in page.locator('#preview-actual-status').text_content():
                    break
            expect(page.locator('#preview-actual-status')).to_contain_text('aktuell:')
            expect(page.locator('#preview-actual-volume')).to_have_text('20 %')
            expect(page.locator('#preview-actual-circle')).to_be_visible()
            expect(page.locator('#preview-input-volume')).to_be_hidden()
            assert page.locator('#preview-actual-circle').evaluate('n=>getComputedStyle(n).borderTopStyle') == 'dashed'
            assert page.locator('#preview-circle').evaluate('n=>getComputedStyle(n).borderTopStyle') == 'solid'
            assert abs(float(page.locator('#preview-actual-circle').evaluate('n=>n.style.transform.match(/scale\\(([^)]+)\\)/)[1]')) - (.46 + .54 * .2)) < 1e-6
            review.shot(page, f'actual-fresh-{label}', f'Fresh renderer ACTUAL diverges from desired audio radius {label}', page.locator('#preview-actual-volume'), 'Operator controls')
            fresh_status_height = page.locator('#preview-actual').bounding_box()['height']
            page.clock.run_for(1800)
            expect(page.locator('#preview-actual-status')).to_contain_text('eingefroren')
            expect(page.locator('#preview-actual-volume')).to_have_text('20 %')
            assert page.locator('#preview-actual').bounding_box()['height'] == fresh_status_height, 'renderer freshness text changed live layout height'
            page.evaluate('document.querySelector(".preview-radius").style.filter="grayscale(1)"')
            review.shot(page, f'actual-stale-{label}', f'ACTUAL radius, replay expiry and grayscale line styles {label}', page.locator('#preview-actual-status'), 'Operator controls')
            assert page.evaluate('window.reviewSetupCommands.length') == 0
            renderer = {**renderer, 'frame': 3, 'observed_ns': '3000000000'}
            capture.wait_setup(page, renderer=renderer, clock_valid=False)
            expect(page.locator('#preview-circle')).to_be_hidden()
            expect(page.locator('#preview-actual-circle')).to_be_visible()
            review.shot(page, f'actual-without-clock-{label}', f'Fresh renderer is independent from invalid desired clock {label}', page.locator('#preview-actual-status'), 'Operator controls')
            capture.wait_setup(page, renderer=None)
            expect(page.locator('#preview-actual-circle')).to_be_hidden()
            expect(page.locator('#preview-actual-volume')).to_be_hidden()
            expect(page.locator('#preview-actual-status')).to_contain_text('nicht verfügbar')
            assert page.evaluate('window.reviewSetupCommands.length') == 0
            page.context.close()

            page = review.page(base.replace('deferSetup:true', 'deferSetup:false'), viewport={'width': width, 'height': height})
            page.evaluate('font=>document.documentElement.style.fontSize=font', font)
            capture.enter_participant(page)
            page.locator('#quest-settings summary').click()
            expect(page.locator('#kiosk-summary')).to_contain_text('angefordert · nicht wirksam')
            page.locator('#kiosk-retry').click()
            expect(page.locator('#kiosk-message')).to_contain_text('Warte auf die Bestätigung')
            page.clock.run_for(14000)
            expect(page.locator('#kiosk-message')).to_contain_text('Wartezeit ist abgelaufen')
            assert page.evaluate('window.reviewKioskCommands.filter(x=>x.op==="retry").length') == 1
            request_id = page.evaluate('window.reviewKioskCommands.find(x=>x.op==="retry").id')
            page.evaluate('id=>window.reviewQuest.kioskReceipt={id,state:"confirmed",detail:"service reconnected"}', request_id)
            page.locator('#kiosk-check').click()
            expect(page.locator('#kiosk-message')).to_contain_text('Dienst wieder verbunden')
            expect(page.locator('#kiosk-summary')).to_contain_text('nicht wirksam')
            review.shot(page, f'kiosk-{label}', f'Kiosk request distinct from effective protection {label}', page.locator('#kiosk-summary'), 'Operator controls')
            page.context.close()
        capture.capture_scroll_regression(review)
        page = review.page(paired + "window.reviewInitialQuest={polar:'off',strapCount:1,connectTimeoutOnce:true};")
        page.locator('#quest-polar-connect').click()
        page.clock.run_for(16000)
        expect(page.locator('#quest-polar-status')).to_contain_text('automatisch in 2 s erneut')
        expect(page.locator('#quest-polar-connect')).to_be_hidden()
        assert page.evaluate('sessionStorage.getItem("atmungsraum-quest-pending")') is None
        commands = page.evaluate('window.reviewQuest.commands')
        assert commands == ['scan', 'select', 'connect'], commands
        review.shot(page, 'connect-timeout-backoff', 'Known native Connect timeout retires receipt while automatic backoff continues', page.locator('#quest-polar-status'), 'Operator controls')
        page.clock.run_for(2200)
        expect(page.locator('#quest-polar-status')).to_contain_text('empfängt EKG und Beschleunigungsdaten')
        assert page.evaluate('window.reviewQuest.commands') == commands, 'automatic recovery replayed a browser mutation'
        capture.enter_participant(page)
        expect(page.locator('#preview-calibrate')).to_be_enabled()
        expect(page.locator('#preview-controller')).to_be_enabled()
        review.shot(page, 'connect-timeout-ready', 'Automatic H10 recovery restores phone setup without command replay', page.locator('#preview-calibrate'), 'Operator controls')
        page.context.close()
        browser.close()
    (args.output / 'rendered-checks.json').write_text(json.dumps({'checks': review.checks, 'regressions': review.regressions, 'errors': review.errors}, indent=2), encoding='utf8')
    print(json.dumps({'rendered': len(review.checks), 'scroll_regressions': len(review.regressions), 'errors': review.errors}))


if __name__ == '__main__':
    main()
