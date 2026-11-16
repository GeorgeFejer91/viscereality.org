"""APK policy/identity UI with real source, Pretext and same-origin policy GETs."""
from argparse import ArgumentParser
from pathlib import Path
import importlib.util
import json
from playwright.sync_api import sync_playwright, expect

spec = importlib.util.spec_from_file_location('capture', Path(__file__).with_name('capture-phone-screens.py'))
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def main():
    parser = ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'screens').mkdir(exist_ok=True)
    policy = json.loads(Path(__file__).parents[1].joinpath('quest-release.json').read_text(encoding='utf8'))['packages']['org.viscereality.atmungsraum.developer']
    current_code, current_name = policy['latest_version_code'], policy['version_name']
    outdated_code = policy['minimum_version_code'] - 1
    metadata = {'v': 1, 'package_name': 'org.viscereality.atmungsraum.developer',
                'version_code': current_code, 'version_name': current_name, 'build_id': 'e' * 64}
    cases = [('old-apk-no-metadata', None, 'unknown', False),
             ('installed-current', metadata, 'compatible', False),
             ('installed-outdated', {**metadata, 'version_code': outdated_code, 'version_name': 'older-dev'}, 'required', False),
             ('policy-unavailable', metadata, 'check-unavailable', True),
             ('unknown-package', {**metadata, 'package_name': 'org.other.application'}, 'unknown', False)]
    receipts = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        review = capture.Review(browser, args.output)
        for width, size in [(390, '100%'), (320, '200%'), (1280, '100%')]:
            for slug, apk, expected, failure in cases:
                review.release_policy_failure = failure
                settings = {'polar': 'ready', 'apk': apk,
                            'access': {'v': 1, 'role': 'active', 'active_phone': 'Fixture', 'revision': 1}}
                initial = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({capture.PAIRED}));" + \
                    'window.reviewInitialQuest=' + json.dumps(settings) + ';' + """
                    window.reviewInitialSetup={stage:'testing',input:'controller-state',ready:true,playing:true};
                    Object.defineProperty(document,'hidden',{configurable:true,get:()=>Boolean(window.reviewHidden)});
                    const originalFetch=window.fetch;
                    window.reviewPolicyRequests=[];
                    window.fetch=(url,options)=>{
                      if(String(url).endsWith('/quest-release.json')) window.reviewPolicyRequests.push({url:String(url),cache:options.cache,credentials:options.credentials});
                      return originalFetch(url,options);
                    };
                    """
                page = review.page(initial, viewport={'width': width, 'height': 844})
                page.evaluate('size=>document.documentElement.style.fontSize=size', size)
                capture.enter_participant(page)
                page.clock.run_for(500)
                expect(page.locator('#quest-version')).to_have_attribute('data-version-state', expected)
                expect(page.locator('#preview-continue')).to_be_enabled()
                if expected == 'required':
                    expect(page.locator('#apk-update-status')).to_contain_text('erforderlich')
                    expect(page.locator('#apk-required')).to_contain_text(f"Mindestens Code {policy['minimum_version_code']}")
                if expected == 'compatible':
                    expect(page.locator('#apk-installed')).to_contain_text(f'{current_name} · Code {current_code} · Build eeeeeeeeeeee')
                if apk is None:
                    expect(page.locator('#apk-installed')).to_have_text('Installiert: nicht verlässlich gemeldet.')
                review.shot(page, f'{slug}-{width}', f'APK {slug} · {width}px/{size}', page.locator('#quest-version'), 'APK release indicator')
                assert page.evaluate('window.reviewSetupCommands.length') == 0
                assert page.evaluate('(window.reviewQuest.commands??[]).length') == 0
                initial_reads = page.evaluate('window.reviewPolicyRequests.length')
                assert initial_reads >= 1
                page.evaluate("window.reviewHidden=true;document.dispatchEvent(new Event('visibilitychange'));window.reviewHidden=false;document.dispatchEvent(new Event('visibilitychange'))")
                page.clock.run_for(1400)
                expect(page.locator('#quest-version')).to_have_attribute('data-version-state', expected)
                requests = page.evaluate('window.reviewPolicyRequests')
                assert len(requests) > initial_reads
                assert all(request['cache'] == 'no-store' and request['credentials'] == 'omit' for request in requests)
                if slug == 'installed-current':
                    summary = page.locator('#apk-identity summary')
                    summary.click()
                    expect(page.locator('#apk-full-identity')).to_contain_text('e' * 64)
                    expect(page.locator('#apk-full-identity')).to_contain_text(metadata['package_name'])
                    summary.focus()
                    page.evaluate('scrollTo(0,Math.min(800,document.documentElement.scrollHeight-innerHeight-10))')
                    before = page.evaluate('scrollY')
                    page.clock.run_for(2300)
                    assert abs(page.evaluate('scrollY') - before) <= 2
                    expect(summary).to_be_focused()
                    review.shot(page, f'full-identity-{width}', f'APK full build identity · {width}px/{size}', page.locator('#apk-full-identity'), 'APK release indicator')
                    # Isolate policy-copy geometry from the existing desired
                    # audio live/stale label by holding the native clock still.
                    page.evaluate("Object.assign(window.reviewQuest.setup,{playing:false,audio_state:'paused'})")
                    page.clock.run_for(1300)
                    summary.focus()
                    page.evaluate('scrollTo(0,document.documentElement.scrollHeight-innerHeight-10)')
                    before = page.evaluate('scrollY')
                    page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
                    page.clock.run_for(1400)
                    expect(page.locator('#quest-version')).to_have_attribute('data-version-state', 'compatible')
                    assert abs(page.evaluate('scrollY') - before) <= 2, 'read-only policy refresh clamped document scroll'
                    expect(summary).to_be_focused()
                    page.evaluate('window.reviewQuest.apk=null')
                    page.clock.run_for(1400)
                    expect(page.locator('#quest-version')).to_have_attribute('data-version-state', 'unknown')
                    expect(page.locator('#apk-installed')).not_to_contain_text('Code 10')
                    expect(page.locator('#apk-full-identity')).not_to_contain_text('e' * 64)
                receipts.append({'case': slug, 'width': width, 'text_size': size, 'state': expected, 'policy_requests': requests,
                                 'experiment_admission_unchanged': True})
                page.context.close()
        browser.close()
    (args.output / 'checks.json').write_text(json.dumps({'geometry': review.checks, 'receipts': receipts, 'errors': review.errors}, indent=2), encoding='utf8')
    print(json.dumps({'rendered': len(review.checks), 'cases': len(receipts), 'errors': review.errors}))


if __name__ == '__main__':
    main()
