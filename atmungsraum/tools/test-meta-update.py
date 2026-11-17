"""Read-only native startup status and actual Pretext/reflow; no device or account I/O."""
from pathlib import Path
import tempfile
import json
import os
from playwright.sync_api import sync_playwright, expect
from layout_render_fixtures import APK, module, initial, mutations, go
from release_fixtures import meta_update_fixture


def run():
    c = module()
    with sync_playwright() as p, tempfile.TemporaryDirectory() as output:
        browser = p.chromium.launch()
        evidence = Path(os.environ.get('ATMUNGSRAUM_META_UI_OUTPUT', output))
        evidence.mkdir(parents=True, exist_ok=True)
        review = c.Review(browser, evidence)
        for width, text_size in [(390, ''), (320, ''), (320, '200%')]:
            update = meta_update_fixture(v=1, state='downloading', reason='download-in-progress-'+'x'*70,
                          available_code=APK['version_code']+1, downloaded_bytes=1024,
                          total_bytes=None, session_allowed=False)
            page = review.page(initial(c), viewport={'width':width,'height':568 if width == 320 else 844})
            page.evaluate('size=>document.documentElement.style.fontSize=size', text_size)
            page.clock.run_for(1500)
            page.locator('#operator-connect-next').click()
            expect(page.locator('#patient-surname')).to_be_visible()
            page.locator('#patient-surname').fill('Tes')
            before = mutations(page)
            page.evaluate('window.reviewMetaUpdate='+json.dumps(update))
            page.clock.run_for(1200)
            expect(page.locator('#operator-meta-update')).to_be_hidden()
            expect(page.locator('#update-screen')).to_be_visible()
            expect(page.locator('#operator-meta-summary')).to_contain_text('herunter')
            expect(page.locator('#patient-next')).to_be_disabled()
            go(page, 'update')
            expect(page.locator('#meta-update-status')).to_contain_text('herunter')
            expect(page.locator('#meta-update-progress')).not_to_have_attribute('value', '')
            assert page.locator('#meta-update-progress').get_attribute('value') is None
            expect(page.locator('#meta-update-detail')).to_contain_text('unbekannt')
            review.shot(page, f'meta-{width}-{text_size or "normal"}', 'Meta update', page.locator('#meta-update-status'), 'Meta update')
            assert page.locator('#update-instruction').get_attribute('data-text-measurement') == 'pretext'
            if page.locator('#update-instruction').get_attribute('data-text-fit') != 'fit':
                # Conservative Pretext no-fit must retain readable document flow;
                # shot() independently checks every painted glyph and panel.
                expect(page.locator('html')).to_have_attribute('data-layout', 'reflow')
            identity = page.locator('#meta-update-status').element_handle()
            page.evaluate('reviewMetaUpdate.total_bytes=4096;reviewMetaUpdate.downloaded_bytes=2048')
            page.clock.run_for(1200)
            expect(page.locator('#meta-update-progress')).to_have_js_property('value',2048)
            expect(page.locator('#meta-update-progress')).to_have_js_property('max',4096)
            assert identity.evaluate('node=>node===document.querySelector("#meta-update-status")')
            assert page.locator('#patient-surname').input_value() == 'Tes'
            # A shorter terminal message must not clamp bottom scroll, detach
            # disclosure focus, or discard a draft when native admission opens.
            page.locator('#apk-identity summary').click()
            page.clock.run_for(1200)
            page.evaluate("document.querySelector('#apk-identity summary').focus({preventScroll:true});scrollTo(0,document.documentElement.scrollHeight)")
            page.clock.run_for(100)
            scroll_before = page.evaluate('scrollY')
            page.evaluate("reviewMetaUpdate.state='current';reviewMetaUpdate.session_allowed=true;reviewMetaUpdate.reason=''")
            page.clock.run_for(1200)
            scroll_after = page.evaluate('scrollY')
            assert abs(scroll_after - scroll_before) < 2, {
                'viewport': width, 'text': text_size, 'before': scroll_before,
                'after': scroll_after, 'documentHeight': page.evaluate('document.documentElement.scrollHeight')}
            assert page.locator('#apk-identity summary').evaluate('n=>n===document.activeElement')
            expect(page.locator('#patient-next')).to_be_enabled()
            assert mutations(page) == before
            page.context.close()
        # Every advertised terminal/busy state renders in the actual app, with
        # admission coming only from its native flag. Legacy absence is usable.
        page = review.page(initial(c))
        page.clock.run_for(1500)
        before = mutations(page)
        go(page, 'update')
        expect(page.locator('#meta-update-status')).to_contain_text('nicht gemeldet')
        for state_name, allowed in [('waiting',False),('checking',False),('installing',False),
                                    ('restart_required',False),('failed',False),('failed',True),
                                    ('current',True),('updated',True)]:
            update.update(state=state_name, session_allowed=allowed, reason='')
            page.evaluate('window.reviewMetaUpdate='+json.dumps(update))
            page.clock.run_for(1200)
            expect(page.locator('#meta-update')).to_have_attribute('data-update-state',state_name)
            expect(page.locator('#meta-update')).to_have_attribute('data-update-allowed',str(allowed).lower())
            review.shot(page, f'meta-{state_name}-{allowed}', 'Meta '+state_name, page.locator('#meta-update-status'), 'Meta update')
        page.evaluate('window.reviewMetaUpdate=null')
        page.clock.run_for(1200)
        expect(page.locator('#meta-update-status')).to_contain_text('nicht verlässlich')
        expect(page.locator('#patient-next')).to_be_disabled()
        assert mutations(page) == before
        page.context.close()
        page = review.page(initial(c), viewport={'width':320,'height':568})
        page.evaluate("document.documentElement.style.fontSize='200%'")
        page.clock.run_for(1500)
        go(page, 'update')
        update.update(state='restart_required',session_allowed=False,reason='prior-install-unconfirmed-maintenance-required')
        page.evaluate('window.reviewMetaUpdate='+json.dumps(update))
        page.clock.run_for(1200)
        expect(page.locator('#meta-update-detail')).to_contain_text('Meta-Bibliothek')
        expect(page.locator('#meta-update-detail')).to_contain_text('Betreiberwartung')
        review.shot(page, 'meta-maintenance-320-200text', 'Meta maintenance', page.locator('#meta-update-status'), 'Meta update')
        page.context.close()
        # Repeatable control uses the real mounted UI and native cycle fixture.
        # Status/paint never invokes a check; one manual click creates one ID.
        native2=meta_update_fixture()
        for width,text_size in [(390,''),(320,''),(320,'200%')]:
            page=review.page(initial(c,quest={'metaUpdate':native2}),viewport={'width':width,'height':844})
            page.evaluate('size=>document.documentElement.style.fontSize=size',text_size)
            page.clock.run_for(1500)
            go(page,'update')
            expect(page.locator('#meta-update-check')).to_be_visible()
            expect(page.locator('#meta-update-check')).to_be_enabled()
            assert page.evaluate('window.reviewMetaUpdateCommands??[]') == []
            page.locator('#meta-update-check').click()
            page.clock.run_for(1800)
            expect(page.locator('#meta-update-status')).to_contain_text('prüft')
            expect(page.locator('#meta-update-check')).to_be_disabled()
            commands=page.evaluate('window.reviewMetaUpdateCommands??[]')
            assert len([x for x in commands if x['op']=='check-and-update']) == 1,commands
            required = APK['version_code']+2
            blocked=dict(native2,state='update_required',reason='required-update-not-yet-on-channel',required_code=required,
                         session_allowed=False,available_code=18,attempt='777',revision='9')
            page.evaluate('window.reviewMetaUpdate='+json.dumps(blocked))
            page.clock.run_for(1200)
            expect(page.locator('#meta-update-status')).to_have_text('Meta bietet das benötigte Update noch nicht an.')
            expect(page.locator('#meta-update-detail')).to_contain_text(str(required))
            expect(page.locator('#patient-next')).to_be_disabled()
            review.shot(page,f'meta-repeatable-required-{width}-{text_size or "normal"}','Required update',page.locator('#meta-update-status'),'Meta update')
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            error=dict(v=1,stage=3,code=1971054,http=None,install_result='DUPLICATE_REQUEST')
            for native_state,native_reason,title in [
                ('checking','checking-duplicate-download','Existing Meta download'),
                ('checking','restarting-duplicate-download','Restarting Meta download'),
                ('installing','queued-meta-update-handoff','Queued Meta handoff'),
                ('update_required','duplicate-download-already-retried','Meta duplicate error')]:
                changed=dict(blocked,state=native_state,reason=native_reason,sdk_error=error,can_check=False)
                page.evaluate('window.reviewMetaUpdate='+json.dumps(changed))
                page.clock.run_for(1200)
                expect(page.locator('#patient-next')).to_be_disabled()
                if native_reason=='queued-meta-update-handoff':
                    expect(page.locator('#meta-update-status')).to_contain_text('schließt')
                    expect(page.locator('#meta-update-status')).not_to_contain_text('Installation angefragt')
                review.shot(page,f'meta-sdk-{native_reason}-{width}-{text_size or "normal"}',title,page.locator('#meta-update-status'),'Meta update')
            changed=dict(native2,attempt='777',revision='10')
            page.evaluate('window.reviewMetaUpdate='+json.dumps(changed))
            page.clock.run_for(1200)
            expect(page.locator('#meta-update-status')).to_have_text('Quest-App ist aktuell.')
            expect(page.locator('#meta-update-message')).to_be_hidden()
            expect(page.locator('#meta-update-prerequisites')).not_to_contain_text('Die Quest prüft die erforderliche Version')
            page.context.close()
        assert not review.errors, review.errors
        browser.close()
    print('PASS native Meta read-only display, stable nodes, native gates, 320px/200% and actual Pretext')

if __name__ == '__main__':
    run()
