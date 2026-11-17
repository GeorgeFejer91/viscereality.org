"""Real app regression: native method confirmation, stable status and gold selection."""
from pathlib import Path
import tempfile
import os
from playwright.sync_api import sync_playwright, expect
from layout_render_fixtures import APK, module, initial, mutations
from release_fixtures import PACKAGE_NAME, release_policy

c = module()
with sync_playwright() as p, tempfile.TemporaryDirectory() as output:
    browser = p.chromium.launch()
    evidence = Path(os.environ.get('ATMUNGSRAUM_INPUT_UI_OUTPUT', output))
    evidence.mkdir(parents=True, exist_ok=True)
    review = c.Review(browser, evidence)
    page = review.page(initial(c, {'participantArm': False}))
    documents = []
    page.on('request', lambda request: documents.append(request.url) if request.resource_type == 'document' else None)
    page.clock.run_for(2000)
    expect(page.locator('#check-bluetooth')).to_have_attribute('data-check', 'ok')
    expect(page.locator('#quest-version')).not_to_be_visible()
    page.locator('#operator-connect-next').click()
    page.locator('#preview-polar-fixed').click()
    page.clock.run_for(1500)
    expect(page.locator('#operator-confirmed-input')).to_be_visible()
    expect(page.locator('#operator-confirmed-input')).to_contain_text('feste Achse')
    # Freshness/readiness can drop without deleting the last method/status box.
    page.evaluate('''() => {
      window.inputVisibility=[];
      new MutationObserver(rs=>{for(const r of rs) if(r.oldValue!==r.target.getAttribute('hidden'))
        inputVisibility.push(r.target.hidden);}).observe(document.querySelector('#operator-input-status'),
          {attributes:true,attributeOldValue:true,attributeFilter:['hidden']});
      reviewQuest.fresh=false;
    }''')
    page.clock.run_for(1200)
    page.evaluate('reviewQuest.fresh=true')
    page.clock.run_for(1400)
    assert page.evaluate('inputVisibility') == []
    for button, label in (('preview-controller', 'Controller State'),
                          ('preview-polar-adaptive', 'dynamische Achse'),
                          ('preview-polar-fixed', 'feste Achse')):
        page.locator('[data-operator-go="input"]:visible').click()
        page.locator('#'+button).click()
        page.clock.run_for(1200)
        expect(page.locator('#operator-confirmed-input')).to_contain_text(label)
        page.locator('[data-operator-go="input"]:visible').click()
        expect(page.locator('#'+button)).to_have_attribute('aria-pressed', 'true')
        assert page.locator('.preview-controls [aria-pressed="true"]').count() == 1
        color = page.locator('#'+button).evaluate('(e)=>getComputedStyle(e).borderTopColor')
        assert color == 'rgb(239, 197, 112)', color
        if button != 'preview-controller':
            expect(page.locator('#preview-controller')).to_have_attribute('aria-pressed', 'false')
        page.locator('#input-screen [data-operator-go="tracking"]').click()
    assert [command['op'] for command in mutations(page)['setup']] == ['polar-fixed','controller','polar-adaptive','polar-fixed']
    assert documents == [], documents
    page.context.close()
    page = review.page(initial(c, {'participantArm': False}))
    expect(page.locator('#check-bridge')).to_have_attribute('data-check', 'ok')
    page.clock.run_for(1500)
    outdated = release_policy()['packages'][PACKAGE_NAME]['minimum_version_code'] - 1
    page.evaluate('code=>reviewQuest.apk.version_code=code', outdated)
    page.clock.run_for(1200)
    page.locator('#operator-connect-next').click()
    expect(page.locator('#update-screen')).to_be_visible()
    expect(page.locator('#operator-update-next')).to_be_disabled()
    page.locator('#update-screen [data-operator-return]').click()
    expect(page.locator('#quest')).to_be_visible()
    page.locator('#operator-connect-next').click()
    expect(page.locator('#update-screen')).to_be_visible()
    page.evaluate('code=>reviewQuest.apk.version_code=code', APK['version_code'])
    page.clock.run_for(1200)
    expect(page.locator('#operator-update-next')).to_be_enabled()
    page.locator('#operator-update-next').click()
    expect(page.locator('#input-screen')).to_be_visible()
    assert mutations(page)['setup'] == []
    page.context.close()
    for width, text_size in ((390, ''), (320, ''), (320, '200%')):
        for mode, button in (('controller-state', 'preview-controller'),
                             ('polar-tilt-fixed', 'preview-polar-fixed'),
                             ('polar-tilt-adaptive', 'preview-polar-adaptive'),
                             ('polar-tilt', 'preview-calibrate')):
            page = review.page(initial(c, {'participantArm': False},
                {'input':mode,'stage':'testing','playing':True,'ready':True,
                 'calibration':'ready' if mode == 'polar-tilt' else 'idle',
                 'audio_state':'playing','parked':False}),
                viewport={'width':width,'height':568 if width == 320 else 844})
            page.evaluate('s=>document.documentElement.style.fontSize=s', text_size)
            page.clock.run_for(1800)
            c.show_operator(page, 'input')
            expect(page.locator('#'+button)).to_have_attribute('aria-pressed','true')
            assert page.locator('#'+button).evaluate('(e)=>getComputedStyle(e).borderTopColor') == 'rgb(239, 197, 112)'
            review.shot(page, f'input-{mode}-{width}-{text_size or "normal"}',
                        'Confirmed input', page.locator('#operator-confirmed-input'), 'Input choice')
            c.show_operator(page, 'tracking')
            review.shot(page, f'tracking-{mode}-{width}-{text_size or "normal"}',
                        'Confirmed input with live circles', page.locator('#operator-confirmed-input'), 'Tracking')
            assert mutations(page)['setup'] == []
            page.context.close()
    assert not review.errors, review.errors
    browser.close()
print('PASS native method confirmation, consistent gold selection and stable status between polls')
