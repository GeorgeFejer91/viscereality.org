"""Optional manual-update instructions use real UI/Pretext and inert transport only."""
from pathlib import Path
import json
import os
import tempfile
from playwright.sync_api import sync_playwright, expect
from layout_render_fixtures import module, initial, mutations, go, SIZES

AUDIT = (Path(__file__).resolve().parents[1] / 'for-ai/scripts/gui-layout.js').read_text(encoding='utf-8')


def settings_entries(review, c):
    kiosk = {'v':1,'g':2,'requested':True,'effective':True,'state':'active',
        'service':'connected','blocked':'','detail':'Quest-App und Kiosk-Dienst verbunden.','retry_available':False}
    pending = {'owner':{'serial':'DEMO-QUEST','device':'d'*32,'epoch':'7','native_epoch':'8'},
        'command':{'v':1,'id':'f'*16,'g':1,'r':0,'op':'calibrate'}}
    for diagnostic in ['kiosk', 'setup-receipt']:
        extra = ('sessionStorage.setItem("atmungsraum-breathing-setup-pending",'+json.dumps(json.dumps(pending))+');'
            if diagnostic == 'setup-receipt' else '')
        page = review.page(initial(c, {'kiosk':kiosk,'receiptState':'outcome_unknown'})+extra,
            viewport={'width':844,'height':390})
        page.clock.run_for(1500)
        before = mutations(page)
        go(page, 'settings')
        audit = page.evaluate(AUDIT, page.viewport_size)
        assert not audit['problems'], audit
        opener = page.locator('#settings-manual-update-help')
        opener.click()
        expect(page.locator('#manual-update-screen')).to_be_visible()
        page.locator('#manual-update-screen [data-operator-return]').click()
        expect(opener).to_be_focused()
        if diagnostic == 'kiosk':
            page.locator('#quest-settings > summary').click()
        else:
            page.locator('#settings-connection > summary').click()
            page.locator('#settings-connection-view').select_option('setup-receipt')
        page.clock.run_for(250)
        audit = page.evaluate(AUDIT, page.viewport_size)
        assert not audit['problems'], audit
        if diagnostic == 'kiosk':
            page.locator('#kiosk-diagnostics > summary').click()
            page.clock.run_for(250)
            audit = page.evaluate(AUDIT, page.viewport_size)
            assert not audit['problems'], audit
        assert mutations(page) == before
        page.context.close()


def active_settings(review, c):
    page = review.page(initial(c), viewport={'width':390,'height':844})
    page.evaluate("document.documentElement.style.fontSize='200%'")
    page.clock.run_for(1500)
    # Previously visited archives retain their real status during the session;
    # this extra visible group exposed the combined Settings overflow.
    go(page, 'recordings')
    page.clock.run_for(1500)
    c.enter_participant(page)
    page.locator('#preview-controller').click()
    page.locator('#preview-continue').click()
    c.review_battery(page)
    page.locator('input[name="quest-scenario"][value="probe-anleitung"]').check()
    page.locator('#quest-play').click()
    before = mutations(page)
    for phase in ['RUNNING', 'PAUSED', 'FINALIZING', 'SAVING']:
        c.wait_state(page, phase=phase, host='vr-session', clock=18000)
        go(page, 'settings')
        page.clock.run_for(250)
        expect(page.locator('#settings-screen #quest-export-status')).to_be_visible()
        for control in ['#quest-play', '#quest-finish', '#music-open', '#settings-manual-update-help']:
            expect(page.locator('#settings-screen ' + control)).to_be_visible()
        audit = page.evaluate(AUDIT, page.viewport_size)
        assert not audit['problems'], (phase, audit)
        ecg = page.locator('#settings-ecg summary')
        ecg.click()
        page.clock.run_for(250)
        expect(page.locator('#settings-ecg dialog')).to_be_visible()
        audit = page.evaluate(AUDIT, page.viewport_size)
        assert not audit['problems'], (phase, audit)
        page.keyboard.press('Escape')
        expect(ecg).to_be_focused()
        opener = page.locator('#settings-manual-update-help')
        opener.click()
        page.locator('#manual-update-screen [data-operator-return]').click()
        expect(opener).to_be_focused()
        page.locator('#settings-screen [data-operator-return]').click()
        expect(page.locator('#running-screen')).to_be_visible()
        assert mutations(page) == before
    page.context.close()


def run():
    c = module()
    with sync_playwright() as p, tempfile.TemporaryDirectory() as temporary:
        evidence = Path(os.environ.get('ATMUNGSRAUM_MANUAL_HELP_OUTPUT', temporary))
        evidence.mkdir(parents=True, exist_ok=True)
        browser = p.chromium.launch()
        review = c.Review(browser, evidence)
        settings_entries(review, c)
        active_settings(review, c)
        for size, viewport, text_size in SIZES:
            page = review.page(initial(c), viewport=viewport)
            page.evaluate('size => document.documentElement.style.fontSize=size', text_size)
            page.clock.run_for(1500)
            c.enter_participant(page)
            before = mutations(page)
            storage = page.evaluate('({local:{...localStorage},session:{...sessionStorage}})')
            go(page, 'settings')
            opener = page.locator('#settings-manual-update-help')
            opener.click()
            guide = page.locator('#manual-update-screen')
            expect(guide).to_be_visible()
            expect(page.locator('#manual-update-title')).to_be_focused()
            expect(guide.locator('img, [data-placeholder]')).to_have_count(0)
            assert not any(word in guide.inner_text().lower() for word in ['todo', 'platzhalter', 'screenshot'])
            selector = page.locator('#manual-update-step')
            for index in range(6):
                selector.select_option(str(index))
                page.clock.run_for(150)
                expect(guide.locator('[data-manual-update-step]:visible')).to_have_count(1)
                review.shot(page, f'manual-{size}-{index}', f'Manual guide {size} step {index+1}',
                    guide.locator('[data-manual-update-step]:visible'), 'Manual update help')
                geometry = page.evaluate('''() => ({width:innerWidth,height:innerHeight,
                  scrollWidth:document.documentElement.scrollWidth,scrollHeight:document.documentElement.scrollHeight})''')
                assert geometry['scrollWidth'] <= geometry['width'] and geometry['scrollHeight'] <= geometry['height'] + 1, geometry
                audit = page.evaluate(AUDIT, viewport)
                assert not audit['problems'], audit
            page.go_back()
            page.clock.run_for(100)
            expect(page.locator('#settings-screen')).to_be_visible()
            expect(opener).to_be_focused()
            page.locator('#settings-screen [data-operator-return]').click()
            expect(page.locator('#input-screen')).to_be_visible()
            expect(page.locator('#patient-surname')).to_have_value('Tes')
            go(page, 'settings')
            opener.click()
            go(page, 'settings')
            page.locator('#settings-screen [data-operator-return]').click()
            expect(page.locator('#input-screen')).to_be_visible()
            assert mutations(page) == before
            assert page.evaluate('({local:{...localStorage},session:{...sessionStorage}})') == storage
            go(page, 'update')
            update_opener = page.locator('#update-manual-update-help')
            update_opener.click()
            guide.locator('[data-operator-return]').click()
            expect(page.locator('#update-screen')).to_be_visible()
            expect(update_opener).to_be_focused()
            page.locator('#update-screen [data-operator-return]').click()
            expect(page.locator('#input-screen')).to_be_visible()
            assert mutations(page) == before
            assert page.evaluate('window.reviewMetaUpdateCommands??[]') == []
            page.context.close()

        page = review.page()
        page.clock.run_for(1500)
        go(page, 'settings')
        page.locator('#settings-manual-update-help').click()
        expect(page.locator('#manual-update-screen')).to_be_visible()
        page.clock.run_for(2000)
        expect(page.locator('#manual-update-screen')).to_be_visible()
        page.locator('#manual-update-screen [data-operator-return]').click()
        expect(page.locator('#settings-screen')).to_be_visible()
        page.context.close()

        page = review.page(initial(c))
        page.clock.run_for(1500)
        page.locator('#operator-connect-next').click()
        page.locator('#patient-surname').fill('Qu')
        page.locator('#patient-age').fill('2')
        draft = page.evaluate('({surname:document.querySelector("#patient-surname").value,age:document.querySelector("#patient-age").value})')
        before = mutations(page)
        go(page, 'settings')
        page.locator('#settings-manual-update-help').click()
        page.locator('#manual-update-screen [data-operator-return]').click()
        page.locator('#settings-screen [data-operator-return]').click()
        expect(page.locator('#participant-screen')).to_be_visible()
        assert page.evaluate('({surname:document.querySelector("#patient-surname").value,age:document.querySelector("#patient-age").value})') == draft
        assert mutations(page) == before
        page.context.close()

        # Native recording and its original pending journal remain authoritative.
        pending = {'op':'pause','id':'f'*16,'generation':0,
            'owner':{'serial':'DEMO-QUEST','device':'d'*32,'epoch':'7','native_epoch':'8'}}
        extra = 'sessionStorage.setItem("atmungsraum-quest-pending",'+json.dumps(json.dumps(pending))+');'
        page = review.page(initial(c, {'phase':'RUNNING','clock':2000})+extra)
        page.clock.run_for(1500)
        before = mutations(page)
        original = page.evaluate('sessionStorage.getItem("atmungsraum-quest-pending")')
        assert json.loads(original) == pending
        go(page, 'settings')
        page.locator('#settings-manual-update-help').click()
        page.clock.run_for(2000)
        expect(page.locator('#manual-update-screen')).to_be_visible()
        page.locator('#manual-update-screen [data-operator-return]').click()
        expect(page.locator('#settings-screen')).to_be_visible()
        page.locator('#settings-screen [data-operator-return]').click()
        expect(page.locator('#running-screen')).to_be_visible()
        assert mutations(page) == before
        assert page.evaluate('sessionStorage.getItem("atmungsraum-quest-pending")') == original
        assert page.evaluate('window.reviewMetaUpdateCommands??[]') == []
        page.context.close()
        assert not review.errors, review.errors
        browser.close()
        print(f'PASS {len(review.checks)} manual-help renders; Back/focus/drafts/pairing/journal/recording retained, disconnected access, zero commands')


if __name__ == '__main__':
    run()
