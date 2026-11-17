"""Compact Update presentation with real mounted UI and Pretext; inert Quest transport."""
from pathlib import Path
import json
import os
import tempfile
from playwright.sync_api import sync_playwright, expect
from layout_render_fixtures import module, initial, mutations, go
from release_fixtures import meta_update_fixture


def run():
    c = module()
    with sync_playwright() as p, tempfile.TemporaryDirectory() as output:
        evidence = Path(os.environ.get('ATMUNGSRAUM_UPDATE_UI_OUTPUT', output))
        evidence.mkdir(parents=True, exist_ok=True)
        browser = p.chromium.launch()
        review = c.Review(browser, evidence)
        for width, size in [(390, ''), (320, ''), (320, '200%')]:
            native = meta_update_fixture(state='updated')
            page = review.page(initial(c, quest={'metaUpdate':native}), viewport={'width':width,'height':844 if width==390 else 568})
            page.evaluate('size=>document.documentElement.style.fontSize=size', size)
            page.clock.run_for(1500)
            go(page, 'update')
            before = mutations(page)
            expect(page.locator('#meta-update-status')).to_have_text('Quest-App ist aktuell.')
            expect(page.locator('#operator-meta-update')).to_be_hidden()
            expect(page.locator('#update-instruction')).to_be_hidden()
            expect(page.locator('#apk-identity')).not_to_have_attribute('open', '')
            expect(page.locator('#apk-installed')).to_be_hidden()
            expect(page.locator('#meta-update-detail')).to_be_hidden()
            expect(page.locator('#meta-update-check')).to_have_text('Erneut prüfen')
            expect(page.locator('#operator-update-next')).to_be_enabled()
            review.shot(page,f'update-current-{width}-{size or "normal"}','Current Quest app',page.locator('#meta-update-status'),'Update presentation')
            page.locator('#apk-identity summary').click()
            expect(page.locator('#apk-installed')).to_be_visible()
            expect(page.locator('#apk-full-identity')).to_contain_text('e'*64)
            expect(page.locator('#meta-update-diagnostic-status')).to_contain_text('Code')
            review.shot(page,f'update-details-{width}-{size or "normal"}','Update details',page.locator('#meta-update-detail'),'Update presentation')
            summary = page.locator('#apk-identity summary')
            summary.focus()
            page.clock.run_for(1200)
            page.evaluate('scrollTo(0,Math.max(0,document.documentElement.scrollHeight-innerHeight-10))')
            scroll_before = page.evaluate('scrollY')
            page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
            page.clock.run_for(1400)
            assert abs(page.evaluate('scrollY')-scroll_before)<=2, 'current Meta/policy polling moved the open diagnostic'
            expect(summary).to_be_focused()
            expect(page.locator('#meta-update-status')).to_have_text('Quest-App ist aktuell.')
            summary.click()
            for state, reason, allowed, instruction in [
                ('downloading','',False,'geladen'),
                ('restart_required','prior-install-unconfirmed-maintenance-required',False,'Meta-Bibliothek'),
                ('failed','offline',True,'Internetverbindung'),
                ('current','',False,'freigegeben'),
                ('update_required','required-update-not-yet-on-channel',False,'Meta'),
            ]:
                changed=dict(native,state=state,reason=reason,session_allowed=allowed,downloaded_bytes=1024,total_bytes=4096,
                             **({'required_code':native['installed_code']+1} if state=='update_required' else {}))
                page.evaluate('window.reviewMetaUpdate='+json.dumps(changed))
                page.clock.run_for(1200)
                expect(page.locator('#meta-update-status')).not_to_have_text('Quest-App ist aktuell.')
                expect(page.locator('#operator-meta-update')).to_be_hidden()
                expect(page.locator('#update-instruction')).to_contain_text(instruction)
                assert not page.locator('#apk-identity').evaluate('n=>n.open')
                if state=='update_required': expect(page.locator('#meta-update-check')).to_have_text('Jetzt aktualisieren')
                if not allowed: expect(page.locator('#operator-update-next')).to_be_disabled()
                review.shot(page,f'update-{state}-{width}-{size or "normal"}',state,page.locator('#update-instruction'),'Update presentation')
            page.evaluate('window.reviewMetaUpdate=null')
            page.clock.run_for(1200)
            expect(page.locator('#meta-update-status')).to_contain_text('nicht verlässlich')
            expect(page.locator('#update-instruction')).to_contain_text('erneut')
            expect(page.locator('#operator-update-next')).to_be_disabled()
            review.shot(page,f'update-unreliable-{width}-{size or "normal"}','Unreliable update status',page.locator('#update-instruction'),'Update presentation')
            assert mutations(page)==before, 'presentation and Details cannot dispatch Quest mutations'
            page.context.close()
        # Retain required headset guidance beside an unresolved original phone
        # request; disclosing Details must never be necessary to discover it.
        native = meta_update_fixture(state='restart_required', session_allowed=False,
            reason='prior-install-unconfirmed-maintenance-required')
        original = 'a'*16
        receipt = {'v':1,'id':original,'state':'accepted','cycle':native['cycle'],
                   'revision':str(int(native['revision'])+1),'attempt':'999','reason':''}
        pending = {'owner':{'serial':'DEMO-QUEST','device':'d'*32,'epoch':'7','native_epoch':'8'},
                   'command':{'v':1,'id':original,'op':'check-and-update',
                              'cycle':native['cycle'],'revision':native['revision']},'accepted':receipt}
        extra = "sessionStorage.setItem('atmungsraum-meta-update-pending',"+json.dumps(json.dumps(pending))+');'
        extra += 'window.reviewReceipts='+json.dumps({original:receipt})+';'
        page = review.page(initial(c,quest={'metaUpdate':native})+extra,viewport={'width':320,'height':568})
        page.evaluate("document.documentElement.style.fontSize='200%'")
        page.clock.run_for(1500)
        go(page,'update')
        expect(page.locator('#meta-update-status')).to_have_text('Update-Bestätigung noch offen.')
        expect(page.locator('#update-instruction')).to_contain_text('ursprüngliche')
        expect(page.locator('#update-instruction')).to_contain_text('Meta-Bibliothek')
        expect(page.locator('#update-instruction')).to_contain_text('Betreiberwartung')
        expect(page.locator('#operator-update-next')).to_be_disabled()
        assert not page.locator('#apk-identity').evaluate('n=>n.open')
        assert not any(v['op']=='check-and-update' for v in page.evaluate('window.reviewMetaUpdateCommands??[]'))
        review.shot(page,'update-pending-maintenance','Pending receipt with native maintenance',page.locator('#update-instruction'),'Update presentation')
        page.context.close()
        # A failed public-policy read cannot be presented as a current app,
        # even when Meta reports an allowed terminal result. No new gate is added.
        review.release_policy_failure = True
        page = review.page(initial(c, quest={'metaUpdate':meta_update_fixture()}))
        page.clock.run_for(1500)
        go(page, 'update')
        expect(page.locator('#meta-update-status')).to_contain_text('Versionsprüfung nicht verfügbar')
        expect(page.locator('#meta-update-status')).not_to_have_text('Quest-App ist aktuell.')
        assert mutations(page)['session']==[]
        review.shot(page,'update-policy-unavailable','Public policy unavailable',page.locator('#update-instruction'),'Update presentation')
        page.context.close()
        assert not review.errors, review.errors
        browser.close()
        print(f'PASS {len(review.checks)} compact Update renders: current/details/progress/recovery/offline/unknown, 320px/200% and actual Pretext; zero Quest mutations')

if __name__=='__main__': run()
