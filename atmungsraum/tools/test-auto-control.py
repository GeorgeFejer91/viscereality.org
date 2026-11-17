"""Automatic native lease acquisition through the real app and Pretext layout."""
from pathlib import Path
import tempfile
from playwright.sync_api import sync_playwright, expect
from layout_render_fixtures import module, initial, mutations

c = module()
free = dict(v=1, role='waiting', active_phone='', revision=1)
with sync_playwright() as p, tempfile.TemporaryDirectory() as output:
    browser = p.chromium.launch()
    review = c.Review(browser, Path(output))
    page = review.page(initial(c, {'access':free}))
    page.clock.run_for(2500)
    expect(page.locator('#check-control')).to_have_attribute('data-check','ok')
    assert [v['op'] for v in mutations(page)['access']] == ['claim']
    assert mutations(page)['setup'] == mutations(page)['session'] == []
    c.show_operator(page, 'settings')
    page.locator('#quest-control-release').click()
    page.clock.run_for(16000)
    expect(page.locator('#check-control')).to_have_attribute('data-check','unconfirmed')
    expect(page.locator('#quest-control-status')).to_contain_text('freigegeben')
    assert [v['op'] for v in mutations(page)['access']] == ['claim','release']
    page.locator('#quest-control-claim').click()
    expect(page.locator('#check-control')).to_have_attribute('data-check','ok')
    page.context.close()

    page = review.page(initial(c, {'access':{**free,'active_phone':'Andere Bedien-App'}}))
    page.clock.run_for(10000)
    expect(page.locator('#check-bluetooth')).to_have_attribute('data-check','ok')
    expect(page.locator('#check-control')).to_have_attribute('data-control-state','occupied')
    assert mutations(page)['access'] == []
    page.locator('#check-control').click()
    expect(page.locator('#connection-diagnostic-message')).to_contain_text('Schließe die andere Bedien-App')
    page.keyboard.press('Escape')
    page.evaluate('Object.assign(reviewQuest.access,{active_phone:"",revision:2})')
    page.clock.run_for(2000)
    expect(page.locator('#check-control')).to_have_attribute('data-check','ok')
    assert len(mutations(page)['access']) == 1
    page.context.close()

    for lost, state, count in ((True,'unknown',1),(False,'retrying',3)):
        page = review.page(initial(c, {'access':free,'controlNoGrant':True,'loseControlReply':lost}))
        page.wait_for_function('window.reviewStatusReads>0')
        page.clock.run_for(25000)
        expect(page.locator('#check-control')).to_have_attribute('data-control-state',state)
        assert len(mutations(page)['access']) == count
        assert mutations(page)['session'] == mutations(page)['setup'] == []
        if not lost:
            page.clock.run_for(20000)
            assert len(mutations(page)['access']) == 4
            page.clock.run_for(20000)
            assert len(mutations(page)['access']) == 4
            page.clock.run_for(10000)
            assert len(mutations(page)['access']) == 5
        if lost:
            page.reload()
            page.wait_for_function('window.reviewStatusReads>0')
            page.clock.run_for(9000)
            expect(page.locator('#check-control')).to_have_attribute('data-control-state','unknown')
            assert mutations(page)['access'] == []
            page.evaluate('Object.assign(reviewQuest.access,{revision:2});reviewQuest.controlNoGrant=false;reviewQuest.loseControlReply=false')
            page.clock.run_for(2000)
            expect(page.locator('#check-control')).to_have_attribute('data-check','ok')
        page.context.close()

    page = review.page(initial(c, {'access':free,'controlNoGrant':True,'phase':'RUNNING'}))
    page.wait_for_function('window.reviewStatusReads>0')
    page.clock.run_for(25000)
    assert len(mutations(page)['access']) == 3
    expect(page.locator('#check-control')).to_have_attribute('data-control-state','retrying')
    page.clock.run_for(20000)
    assert len(mutations(page)['access']) == 4
    assert mutations(page)['setup'] == mutations(page)['session'] == []
    page.context.close()

    # Backgrounding expires the green grant; foreground resumption reads and
    # reacquires an expired lease without any session/setup operation.
    page = review.page(initial(c))
    page.clock.run_for(2000)
    expect(page.locator('#check-control')).to_have_attribute('data-check','ok')
    page.evaluate("Object.defineProperty(document,'hidden',{configurable:true,get:()=>Boolean(window.reviewHidden)});reviewHidden=true;document.dispatchEvent(new Event('visibilitychange'))")
    expect(page.locator('#check-control')).to_have_attribute('data-check','unconfirmed')
    page.evaluate('Object.assign(reviewQuest.access,{role:"waiting",active_phone:"",revision:2})')
    page.clock.run_for(16000)
    assert mutations(page)['access'] == []
    page.evaluate("reviewHidden=false;document.dispatchEvent(new Event('visibilitychange'))")
    page.clock.run_for(2000)
    expect(page.locator('#check-control')).to_have_attribute('data-check','ok')
    assert len(mutations(page)['access']) == 1
    page.context.close()

    page = review.page(initial(c))
    expect(page.locator('#check-control')).to_have_attribute('data-check','ok')
    page.evaluate('reviewQuest.statusDelay=5000')
    page.clock.run_for(3500)
    expect(page.locator('#check-control')).to_have_attribute('data-control-state','lost')
    page.locator('#check-control').click()
    expect(page.locator('#connection-diagnostic-message')).to_contain_text('nicht mehr bestätigt')
    page.context.close()

    for label, viewport, size in (('320',{'width':320,'height':568},''),('320-200text',{'width':320,'height':568},'200%')):
        page = review.page(initial(c, {'access':{**free,'active_phone':'Ein sehr langer Name der anderen gespeicherten Bedien-App'}}), viewport=viewport)
        page.evaluate('size=>document.documentElement.style.fontSize=size',size)
        page.clock.run_for(2000)
        review.shot(page,'auto-control-'+label,label+' occupied control',page.locator('#quest-control-status'),'Automatic control')
        targets=page.locator('#operator-toolbar button:visible').evaluate_all('(buttons)=>buttons.map(b=>({id:b.id,w:b.getBoundingClientRect().width,h:b.getBoundingClientRect().height}))')
        assert all(v['w']>=48 and v['h']>=48 for v in targets),targets
        page.locator('#check-control').click()
        expect(page.locator('#connection-diagnostic-message')).to_contain_text('Schließe')
        review.shot(page,'auto-control-dialog-'+label,label+' control diagnostic',page.locator('#connection-diagnostic-message'),'Automatic control')
        page.context.close()
    assert not review.errors,review.errors
    browser.close()
print('PASS automatic native control, busy/unknown fences, bounded retry, recording, manual release, resume, 320px/200% Pretext')
