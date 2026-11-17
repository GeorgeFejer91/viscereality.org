"""Ongoing Polar feedback and local battery review with real mounted UI/Pretext."""
from pathlib import Path
import os
import tempfile
from playwright.sync_api import sync_playwright, expect
from layout_render_fixtures import module, initial, mutations

c = module()
with sync_playwright() as p, tempfile.TemporaryDirectory() as temporary:
    output = Path(os.environ.get('ATMUNGSRAUM_DEVICE_UI_OUTPUT', temporary))
    output.mkdir(parents=True, exist_ok=True)
    browser = p.chromium.launch()
    review = c.Review(browser, output)
    page = review.page(initial(c))
    indicator = page.locator('#check-polar')
    expect(indicator).to_have_attribute('data-check', 'ok')
    page.evaluate("Object.assign(reviewQuest,{phase:'RUNNING',host:'vr-session',ecg:false,fresh:true})")
    page.clock.run_for(1400)
    expect(page.locator('#running-screen')).to_be_visible()
    expect(indicator).to_have_attribute('data-check', 'unconfirmed')
    indicator.click()
    expect(page.locator('#connection-diagnostic-message')).to_contain_text('EKG fehlt')
    page.keyboard.press('Escape')
    page.evaluate("Object.assign(reviewQuest,{ecg:true,contactHealth:{state:'not-detected',supported:true,detected:false,age_ms:0}})")
    page.clock.run_for(1400)
    expect(indicator).to_have_attribute('data-health-state', 'contact')
    expect(page.locator('#quest-finish')).to_be_visible()
    page.evaluate("Object.assign(reviewQuest,{contactHealth:{state:'unsupported',supported:false,detected:null,age_ms:0}})")
    page.clock.run_for(1400)
    expect(indicator).to_have_attribute('data-check', 'checking')
    expect(indicator).to_have_attribute('aria-label', 'Polar H10: Tragekontakt unbekannt')
    page.evaluate("Object.assign(reviewQuest,{connectionGeneration:2,contactHealth:{state:'detected',supported:true,detected:true,age_ms:0}})")
    page.clock.run_for(1400)
    expect(indicator).to_have_attribute('data-check', 'ok')
    page.evaluate('reviewQuest.freezeHealth=true')
    page.clock.run_for(5200)
    expect(indicator).not_to_have_attribute('data-check', 'ok')
    page.evaluate('reviewQuest.freezeHealth=false')
    page.clock.run_for(1400)
    expect(indicator).to_have_attribute('data-check', 'ok')
    assert mutations(page)['session'] == mutations(page)['setup'] == []
    page.context.close()

    for slug, viewport, size in [('390', {'width':390,'height':844}, ''),
                                 ('320', {'width':320,'height':568}, ''),
                                 ('320-200text', {'width':320,'height':568}, '200%')]:
        for legacy in (False, True):
            quest = {'participantArm':False, 'healthSupported':not legacy,
                     'power':None if legacy else {'v':1,'quest':{'percent':0,'charging':False},
                         'controllers':[{'name':'0xEC4EF8A9070251C0','percent':None}]}}
            setup = {'input':'controller-state','stage':'testing','playing':True,'ready':True,
                     'audio_state':'playing','parked':False,'input_phase':'hold'}
            page = review.page(initial(c, quest, setup), viewport=viewport)
            page.evaluate('size=>document.documentElement.style.fontSize=size', size)
            expect(page.locator('#operator-connect-next')).to_be_enabled()
            page.locator('#operator-connect-next').click()
            page.locator('#input-screen [data-operator-go="tracking"]').click()
            page.locator('#preview-continue').click()
            expect(page.locator('#battery-screen')).to_be_visible()
            before = mutations(page)
            page.evaluate("location.hash='session'")
            page.clock.run_for(300)
            expect(page.locator('#battery-screen')).to_be_visible()
            expect(page.locator('#quest-play')).not_to_be_visible()
            expect(page.locator('#battery-next')).to_be_enabled()
            if legacy:
                expect(page.locator('#check-polar')).to_have_attribute('data-check','checking')
                expect(page.locator('#battery-quest')).to_have_text('unbekannt')
                expect(page.locator('#battery-polar')).to_have_text('unbekannt')
            else:
                expect(page.locator('#battery-quest')).to_have_text('0 %')
                expect(page.locator('#battery-controllers')).to_have_text('Controller: unbekannt')
                expect(page.locator('#battery-controllers')).not_to_contain_text('0x')
                node = page.locator('#battery-quest').element_handle()
                page.evaluate('reviewQuest.power.quest.percent=1')
                page.clock.run_for(1400)
                assert node.evaluate('n=>n===document.querySelector("#battery-quest")')
                expect(page.locator('#battery-screen')).to_be_visible()
            review.shot(page, f'battery-{slug}-{legacy}', 'Battery review', page.locator('#battery-next'), 'Device review')
            assert mutations(page) == before
            c.review_battery(page)
            expect(page.locator('#quest-play')).to_be_enabled()
            page.locator('#quest-play').click()
            expect(page.locator('#running-screen')).to_be_visible()
            assert mutations(page)['session'] == ['arm', 'start']
            page.context.close()
    # Review follows the existing actor/method/participant binding, never a
    # changing percentage. A new participant or Quest owner must review again.
    page = review.page(initial(c, {}, {'input':'controller-state','stage':'testing',
        'playing':True,'ready':True,'audio_state':'playing','parked':False,'input_phase':'hold'}))
    c.enter_participant(page)
    c.show_operator(page, 'tracking')
    page.locator('#preview-continue').click()
    c.review_battery(page)
    page.evaluate('reviewQuest.power={v:1,quest:{percent:2,charging:false},controllers:[]}')
    page.clock.run_for(1400)
    expect(page.locator('#session-screen')).to_be_visible()
    c.show_operator(page, 'participant')
    page.locator('#patient-age').fill('31')
    page.locator('#patient-next').click()
    c.show_operator(page, 'tracking')
    page.locator('#preview-continue').click()
    expect(page.locator('#battery-screen')).to_be_visible()
    c.review_battery(page)
    c.show_operator(page, 'settings')
    page.locator('#quest-disconnect').click()
    page.evaluate("reviewOwner.native_epoch='9'")
    page.locator('#quest-find').click()
    expect(page.locator('#operator-connect-next')).to_be_enabled()
    page.evaluate("location.hash='session'")
    page.clock.run_for(1000)
    expect(page.locator('#breathing-preview')).to_be_visible()
    page.locator('#preview-continue').click()
    expect(page.locator('#battery-screen')).to_be_visible()
    assert mutations(page)['session'] == []
    page.context.close()
    assert not review.errors, review.errors
    browser.close()
print('PASS ongoing Polar loss/contact/expiry/recovery; six battery renders, old APK/optional controller/zero battery, participant/actor review binding, direct-route guard and original Arm/Start')
