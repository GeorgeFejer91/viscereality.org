"""Regressions reproduced on the attended Pixel/Quest, using real UI and synthetic transport."""
from pathlib import Path
import tempfile
from playwright.sync_api import sync_playwright, expect
from layout_render_fixtures import module, initial, mutations
from release_fixtures import meta_update_fixture
c=module()
with sync_playwright() as p, tempfile.TemporaryDirectory() as output:
    browser=p.chromium.launch()
    review=c.Review(browser,Path(output))
    current=meta_update_fixture()
    for quest, count, final in (({'staleSetupRejects':2},3,'feste Achse'),
                               ({'staleSetupRejects':9},3,'Controller State'),
                               ({'staleSetupRejects':1,'loseStaleSetupReply':True},1,'Controller State'),
                               ({'staleSetupRejects':1,'changeInputOnStale':'polar-tilt-adaptive'},1,'dynamische Achse')):
        page=review.page(initial(c,{'participantArm':False,'metaUpdate':current,**quest},
            {'input':'controller-state','ready':True,'playing':True,'stage':'testing','audio_state':'playing','parked':False}))
        expect(page.locator('#operator-connect-next')).to_be_enabled()
        page.locator('#operator-connect-next').click()
        expect(page.locator('#operator-meta-update')).to_be_hidden()
        page.locator('#input-screen [data-operator-go="tracking"]').click()
        expect(page.locator('#preview-review')).to_be_visible()
        expect(page.locator('#preview-detail')).to_be_hidden()
        appearance=page.locator('#preview-continue').evaluate('(b)=>{const s=getComputedStyle(b);return [s.color,s.backgroundColor,s.borderColor,s.opacity]}')
        c.wait_setup(page,ready=False,r=1,input_phase='unknown')
        expect(page.locator('#preview-review')).to_be_visible()
        expect(page.locator('#preview-continue')).to_be_visible()
        expect(page.locator('#preview-continue')).to_be_disabled()
        expect(page.locator('#preview-detail')).to_be_hidden()
        assert page.locator('#preview-continue').evaluate('(b)=>{const s=getComputedStyle(b);return [s.color,s.backgroundColor,s.borderColor,s.opacity]}') == appearance
        c.wait_setup(page,ready=True,r=2,input_phase='hold')
        expect(page.locator('#preview-continue')).to_be_enabled()
        page.locator('#preview-continue').click()
        expect(page.locator('#session-screen')).to_be_visible()
        c.wait_setup(page,ready=False,r=3,input_phase='unknown')
        expect(page.locator('#session-screen')).to_be_visible()
        expect(page.locator('#quest-play')).to_be_disabled()
        c.wait_setup(page,ready=True,r=4,input_phase='hold')
        expect(page.locator('#quest-play')).to_be_enabled()
        assert mutations(page)['session']==[]
        page.locator('#session-screen [data-operator-back]').click()
        page.locator('[data-operator-go="input"]:visible').click()
        page.locator('#preview-polar-fixed').click()
        page.clock.run_for(2500)
        expect(page.locator('#operator-confirmed-input')).to_contain_text(final)
        sent=mutations(page)['setup']
        assert len(sent)==count,sent
        assert len({v['id'] for v in sent})==count,sent
        assert all(v['op']=='polar-fixed' for v in sent)
        if quest['staleSetupRejects']==9:
            expect(page.locator('#preview-detail')).to_contain_text('geändert')
        page.context.close()
    assert not review.errors,review.errors
    browser.close()
print('PASS healthy-update notice hidden, stable readiness prompt, bound stale retry, no unknown replay or intent drift')
