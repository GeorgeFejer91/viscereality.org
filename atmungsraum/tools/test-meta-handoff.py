"""Actual mounted PWA: native update handoff, disconnect/reload, Pretext reflow."""
import json
import os
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
from layout_render_fixtures import module, initial, mutations


def run():
    c = module()
    with sync_playwright() as p, tempfile.TemporaryDirectory() as temporary:
        browser = p.chromium.launch()
        output = Path(os.environ.get('ATMUNGSRAUM_META_UI_OUTPUT', temporary))
        output.mkdir(parents=True, exist_ok=True)
        review = c.Review(browser, output)
        update = dict(v=2, state='installing', reason='queued-meta-update-handoff',
                      installed_code=22, available_code=23, required_code=23,
                      downloaded_bytes=55, total_bytes=100, session_allowed=False,
                      cycle='2', revision='9', attempt='777', can_check=False, busy_reason='')
        for width, text_size in [(390, ''), (320, ''), (320, '200%')]:
            page = review.page(initial(c, quest={'metaUpdate': update}),
                               viewport={'width': width, 'height': 844})
            page.evaluate('s=>document.documentElement.style.fontSize=s', text_size)
            page.clock.run_for(1800)
            expect(page.locator('#operator-meta-summary')).to_contain_text('schließt')
            before = mutations(page)
            page.evaluate('reviewQuest.statusDrop=true')
            page.clock.run_for(1800)
            notice = page.locator('#operator-meta-handoff')
            expect(notice).to_be_visible()
            expect(notice).to_contain_text('Meta-Update übergeben')
            expect(notice).to_contain_text('aktuelle Stand ist auf diesem Smartphone unbekannt')
            expect(notice).to_contain_text('Prüfe das Update im Headset')
            expect(notice).to_contain_text('verbinde dieses Smartphone erneut')
            assert notice.locator('progress').count() == 0
            assert page.locator('#operator-meta-handoff-detail').get_attribute('data-text-measurement') == 'pretext'
            review.shot(page, f'handoff-disconnected-{width}-{text_size or "normal"}',
                        'Disconnected update handoff', notice, 'Meta update')
            assert mutations(page) == before
            # A fresh document has only a saved hint; it cannot infer progress.
            page.add_init_script('window.reviewJoin="fail";window.reviewInitialQuest.statusDrop=true;'
                                 f'addEventListener("DOMContentLoaded",()=>document.documentElement.style.fontSize={json.dumps(text_size)},{{once:true}})')
            page.reload(wait_until='load')
            page.clock.run_for(1800)
            expect(notice).to_be_visible()
            expect(notice).to_contain_text('aktuelle Stand ist auf diesem Smartphone unbekannt')
            review.shot(page, f'handoff-reloaded-{width}-{text_size or "normal"}',
                        'Retained update instructions', notice, 'Meta update')
            # Return with actual current code21, then normal native UI takes over.
            page.evaluate('''() => {
              window.reviewJoin=null;reviewQuest.statusDrop=false;
              reviewQuest.apk.version_code=23;
              Object.assign(reviewQuest.metaUpdate,{state:'current',reason:'',installed_code:23,
                available_code:23,required_code:23,session_allowed:true,revision:'10',can_check:true});
            }''')
            page.locator('#quest-find').click()
            page.clock.run_for(2000)
            expect(notice).to_be_hidden()
            assert page.evaluate('sessionStorage.getItem("atmungsraum-meta-update-handoff")') is None
            assert page.evaluate('window.reviewMetaUpdateCommands??[]') == []
            page.context.close()
        assert not review.errors, review.errors
        browser.close()
    print('PASS actual disconnected/reloaded update notice, reconnect version verification, 320px/200% Pretext')


if __name__ == '__main__':
    run()
