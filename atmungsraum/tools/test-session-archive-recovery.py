"""Mounted production UI regressions for session and local archive recovery."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
from playwright.sync_api import sync_playwright, expect
from layout_render_fixtures import module, initial, mutations, SIZES

c = module()
c.GATEWAY = c.GATEWAY.replace("recordings: async () => ({ v: 1, state: 'ready', entries: [], next_before: null }),",
    "recordings: async () => {window.reviewRecordingInventories=(window.reviewRecordingInventories??0)+1; return { v: 1, state: 'ready', entries: [], next_before: null };},")
c.GATEWAY = c.GATEWAY.replace('const current = snapshot();',
    'if(window.reviewNextStatusPatch){Object.assign(settings,window.reviewNextStatusPatch);delete window.reviewNextStatusPatch;}const current = snapshot();')
OWNER = dict(device='d' * 32, serial='DEMO-QUEST', epoch='7', native_epoch='8')
KEY = 'atmungsraum-breathing-setup-pending'
ARCHIVE = 'record-1731843000000000000'
CSV = 'participant,time,value\nSYNTHETIC,0,17\n'


def mounted(review, init, **options):
    page = review.page(init, **options)
    page.wait_for_function('Boolean(window.reviewQuest)')
    page.wait_for_function('Boolean(document.documentElement.dataset.operatorStep)')
    return page


def journal(owner=None, recovery=False):
    return dict(owner=owner or OWNER, command=dict(v=1, id='f' * 16, g=1, r=0, op='controller'),
                **({'recovery': True} if recovery else {}))


def pending(value):
    return f"sessionStorage.setItem('{KEY}',JSON.stringify({json.dumps(value)}));"


def session_cases(review):
    running = dict(phase='RUNNING', host='vr-session', generation=3, condition='condition-a',
                   receiptState='outcome_unknown')
    setup = dict(stage='main', input='controller-state', playing=True, ready=True)
    for label, old in [('same', journal()), ('reclaimed', journal()), ('retired', journal({**OWNER, 'native_epoch': '6'})),
                       ('old-recovery', journal(recovery=True))]:
        state = {**running, **({'access': dict(v=1, role='waiting', active_phone='', revision=1)} if label == 'reclaimed' else {})}
        page = mounted(review, initial(c, state, setup) + pending(old))
        page.clock.run_for(16000)
        expect(page.locator('#running-screen')).to_be_visible()
        expect(page.locator('#quest-play')).to_be_enabled()
        expect(page.locator('#quest-finish')).to_be_enabled()
        assert mutations(page)['session'] == mutations(page)['setup'] == []
        for button, phase in [('#quest-play', 'PAUSED'), ('#quest-play', 'RUNNING'), ('#quest-finish', 'IDLE')]:
            page.locator(button).click()
            page.wait_for_function('(phase)=>reviewQuest.phase===phase', arg=phase)
            assert page.evaluate(f"JSON.parse(sessionStorage.getItem('{KEY}'))") == old
        assert mutations(page)['session'] == ['pause', 'resume', 'save-next']
        assert mutations(page)['setup'] == []
        queries = page.evaluate('window.reviewSetupReceiptQueries??[]')
        assert all(q == dict(v=1, id=old['command']['id'], op='receipt') for q in queries)
        if label == 'retired':
            assert queries == []
        review.regressions.append(dict(kind='prep-journal-current-session', actor=label, commands=mutations(page)))
        page.context.close()

    for patch in [dict(generation=4), dict(access=dict(v=1,role='active',revision=8,active_phone='Fixture')),
                  dict(access=dict(v=1,role='active',revision=1,active_phone='Changed phone')),dict(statusDelay=5000)]:
        page = mounted(review, initial(c, running, setup) + pending(journal()))
        page.clock.run_for(2000)
        expect(page.locator('#quest-play')).to_be_enabled()
        # Bind the changed reply to this click, after preview polling and before
        # the production click handler captures and refreshes its authority.
        page.evaluate('''patch=>document.querySelector('#quest-play').addEventListener('click',
            ()=>{window.reviewNextStatusPatch=patch;},{capture:true,once:true})''',patch)
        page.locator('#quest-play').click()
        page.clock.run_for(6000)
        assert mutations(page)['session'] == mutations(page)['setup'] == [], patch
        page.context.close()

    barriers = [
        ('occupied', dict(access=dict(v=1, role='waiting', revision=2, active_phone='Other phone')), ''),
        ('native-pending', dict(pending=True), ''),
        ('native-recovering', {}, 'reviewInitialSetup.stage="recovering";reviewInitialSetup.ready=false;'),
        ('unknown-experiment', {}, "sessionStorage.setItem('atmungsraum-quest-pending',JSON.stringify({id:'eeeeeeeeeeeeeeee',op:'pause',generation:3,owner:" + json.dumps(OWNER) + '}));'),
        ('unknown-claim', dict(access=dict(v=1, role='waiting', revision=1, active_phone=''), controlNoGrant=True, loseControlReply=True), ''),
    ]
    for label, patch, extra in barriers:
        page = mounted(review, initial(c, {**running, **patch}, setup) + pending(journal()) + extra)
        page.clock.run_for(16000)
        expect(page.locator('#quest-play')).to_be_disabled()
        expect(page.locator('#quest-finish')).to_be_disabled()
        page.locator('#quest-finish').evaluate('b=>b.click()')
        page.clock.run_for(2000)
        assert mutations(page)['session'] == mutations(page)['setup'] == [], label
        page.context.close()

    page = mounted(review, initial(c, running, setup) + pending(journal()))
    page.clock.run_for(2000)
    page.evaluate('reviewQuest.statusDelay=5000')
    page.clock.run_for(3500)
    expect(page.locator('#quest-play')).to_be_disabled()
    expect(page.locator('#quest-finish')).to_be_disabled()
    assert mutations(page)['session'] == []
    page.context.close()

    for phase in ['IDLE', 'ARMED']:
        page = mounted(review, initial(c, {**running, 'phase': phase}, setup) + pending(journal()))
        page.clock.run_for(16000)
        expect(page.locator('#quest-play')).to_be_disabled()
        page.locator('#quest-finish').evaluate('b=>b.click()')
        page.clock.run_for(2000)
        assert mutations(page)['session'] == mutations(page)['setup'] == []
        page.context.close()


def seed(page):
    page.wait_for_function('Boolean(window.reviewOwner)')
    return page.evaluate('''async ({id,csv})=>{
      const {openQuestRecordingStore}=await import('./quest-recording-store.js?v=4');
      const s=await openQuestRecordingStore(reviewOwner);
      await s.put(id,{download:new File([csv],'synthetic.csv',{type:'text/csv'}),bundleVersion:1,
        bundle:new File(['original bundle'],'synthetic.zip')});
      await s.appendPrefix({id:'record-1731843001000000000'},0,[new TextEncoder().encode('unfinished')]);
      s.close();
      const foreign=await openQuestRecordingStore({...reviewOwner,device:'a'.repeat(32)});
      await foreign.put('record-1731843002000000000',new File(['foreign'],'foreign.zip'));foreign.close();
      return JSON.parse(localStorage.getItem('atmungsraum-quest-pair'));
    }''', dict(id=ARCHIVE, csv=CSV))


def assert_download(page):
    expect(page.locator('#quest-archives')).to_be_visible()
    expect(page.locator('#quest-archives option')).to_have_count(1)
    expect(page.locator('#quest-export-download')).to_be_visible()
    result = page.evaluate('''async()=>{const a=document.querySelector('#quest-export-download');
      const bytes=await (await fetch(a.href)).arrayBuffer();
      return {name:a.download,hash:[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))]
        .map(n=>n.toString(16).padStart(2,'0')).join('')};}''')
    assert result == dict(name='synthetic.csv', hash=hashlib.sha256(CSV.encode()).hexdigest()), result


def archive_cases(review):
    page = mounted(review, initial(c))
    page.clock.run_for(2000)
    seed(page)
    c.show_operator(page,'settings')
    page.locator('#quest-disconnect').click()
    c.show_operator(page,'recordings')
    assert_download(page)
    page.evaluate('window.previousArchiveURL=document.querySelector("#quest-export-download").href')
    page.locator('[data-operator-return]:visible').click()
    page.evaluate('reviewOwner.device="e".repeat(32);reviewQuest.joinDelay=5000;window.reviewPairingEnabled=true')
    page.locator('#quest-find').click()
    expect(page.locator('#quest-pairing')).to_be_visible()
    page.locator('#quest-pairing-confirm').click()
    page.evaluate('reviewPairing.finish("approved")')
    page.wait_for_function('JSON.parse(localStorage.getItem("atmungsraum-quest-pair")).device==="e".repeat(32)')
    assert page.locator('#quest-export-download').get_attribute('href') is None
    assert page.evaluate('async()=>{try{await fetch(previousArchiveURL);return false;}catch{return true;}}')
    page.clock.run_for(6000)
    page.context.close()

    for label, owner in [('hardware', OWNER), ('installation', {**OWNER, 'serial': '', 'identity_kind': 'app-installation'})]:
        fixture = initial(c)
        pair_init, state_init = fixture.split('window.reviewInitialQuest=', 1)
        init = ('if(!sessionStorage.fixtureInitialized){' + pair_init
                + 'sessionStorage.fixtureInitialized="1";}window.reviewInitialQuest=' + state_init
                + 'window.reviewInitialOwner=' + json.dumps(owner) + ';'
                + 'if(sessionStorage.fixtureOffline)window.reviewJoin="fail";')
        page = mounted(review, init)
        page.clock.run_for(2000)
        pair = seed(page)
        page.locator('#quest-export-retry').evaluate('b=>b.click()')
        page.clock.run_for(1500)
        c.show_operator(page, 'recordings')
        assert_download(page)
        c.show_operator(page, 'settings')
        page.locator('#quest-disconnect').click()
        c.show_operator(page, 'recordings')
        assert_download(page)
        before = mutations(page)
        inventories = page.evaluate('window.reviewRecordingInventories??0')
        page.clock.run_for(16000)
        assert mutations(page) == before, 'Offline access sent native commands'
        assert page.evaluate('window.reviewRecordingInventories??0') == inventories
        # Real IndexedDB work is held behind a live transaction to expose races.
        page.evaluate('''async()=>{
          window.reviewArchiveURL=document.querySelector('#quest-export-download').href;
          const request=indexedDB.open('atmungsraum-quest-recordings',3);
          const db=await new Promise(resolve=>request.onsuccess=()=>resolve(request.result));
          const tx=db.transaction('recordings','readwrite');window.reviewArchiveRelease=false;
          function keepAlive(){const r=tx.objectStore('recordings').get('keep-alive');
            r.onsuccess=()=>{if(!window.reviewArchiveRelease)keepAlive();};}keepAlive();
          tx.oncomplete=()=>{db.close();window.reviewArchiveReleased=true;};
          document.querySelector('#quest-archives').dispatchEvent(new Event('change'));
          localStorage.removeItem('atmungsraum-quest-pair');
          window.dispatchEvent(new StorageEvent('storage',{key:'atmungsraum-quest-pair'}));
        }''')
        expect(page.locator('#quest-export-download')).to_be_hidden()
        assert page.evaluate('async()=>{try{await fetch(reviewArchiveURL);return false;}catch{return true;}}')
        page.evaluate('reviewArchiveRelease=true')
        page.wait_for_function('window.reviewArchiveReleased')
        page.clock.run_for(500)
        expect(page.locator('#quest-archives')).to_be_hidden()
        expect(page.locator('#quest-export-download')).to_be_hidden()
        page.evaluate('''pair=>{localStorage.setItem('atmungsraum-quest-pair',JSON.stringify(pair));
          window.dispatchEvent(new StorageEvent('storage',{key:'atmungsraum-quest-pair'}));}''', pair)
        assert_download(page)
        # Enumeration from a restored pairing must not outlive a switch to a
        # different verified device, even if its IDB request completes later.
        page.evaluate('''async pair=>{
          const request=indexedDB.open('atmungsraum-quest-recordings',3);
          const db=await new Promise(resolve=>request.onsuccess=()=>resolve(request.result));
          const tx=db.transaction('recordings','readwrite');window.reviewArchiveRelease=false;window.reviewArchiveReleased=false;
          function keepAlive(){const r=tx.objectStore('recordings').get('keep-alive');
            r.onsuccess=()=>{if(!window.reviewArchiveRelease)keepAlive();};}keepAlive();
          tx.oncomplete=()=>{db.close();window.reviewArchiveReleased=true;};
          const set=p=>{if(p)localStorage.setItem('atmungsraum-quest-pair',JSON.stringify(p));
            else localStorage.removeItem('atmungsraum-quest-pair');
            window.dispatchEvent(new StorageEvent('storage',{key:'atmungsraum-quest-pair'}));};
          set(null);set(pair);set({...pair,device:'e'.repeat(32)});
        }''', pair)
        expect(page.locator('#quest-export-download')).to_be_hidden()
        page.evaluate('reviewArchiveRelease=true')
        page.wait_for_function('window.reviewArchiveReleased')
        page.clock.run_for(500)
        expect(page.locator('#quest-archives')).to_be_hidden()
        page.evaluate('''pair=>{localStorage.setItem('atmungsraum-quest-pair',JSON.stringify(pair));
          window.dispatchEvent(new StorageEvent('storage',{key:'atmungsraum-quest-pair'}));}''', pair)
        assert_download(page)
        # A missing committed payload is a visible local error even without Quest.
        page.evaluate('''async id=>{const {openQuestRecordingStore}=await import('./quest-recording-store.js?v=4');
          const s=await openQuestRecordingStore(reviewOwner);await s.put(id,{});s.close();
          document.querySelector('#quest-archives').dispatchEvent(new Event('change'));}''', ARCHIVE)
        expect(page.locator('#quest-export-status')).to_be_visible()
        expect(page.locator('#quest-export-status')).to_contain_text('konnte nicht geöffnet')
        expect(page.locator('#quest-export-download')).to_be_hidden()
        page.evaluate('''async ({id,csv})=>{const {openQuestRecordingStore}=await import('./quest-recording-store.js?v=4');
          const s=await openQuestRecordingStore(reviewOwner);await s.put(id,new File([csv],'synthetic.csv'));s.close();
          document.querySelector('#quest-archives').dispatchEvent(new Event('change'));}''', dict(id=ARCHIVE,csv=CSV))
        assert_download(page)
        # APK updates retain the device scope; its saved fingerprint is provenance only.
        page.evaluate('''()=>{const p=JSON.parse(localStorage.getItem('atmungsraum-quest-pair'));
          p.build='c'.repeat(64);localStorage.setItem('atmungsraum-quest-pair',JSON.stringify(p));
          sessionStorage.fixtureOffline='1';}''')
        page.reload()
        page.clock.run_for(2000)
        c.show_operator(page, 'recordings')
        assert_download(page)
        review.regressions.append(dict(kind='offline-committed-archive', identity=label,
                                      hash=hashlib.sha256(CSV.encode()).hexdigest(), buildChanged=True))
        for bad in [None, {**pair, 'device': 'e' * 32}, {**pair, 'build': None}]:
            page.evaluate('''value=>{if(value===null)localStorage.removeItem('atmungsraum-quest-pair');
              else localStorage.setItem('atmungsraum-quest-pair',JSON.stringify(value));}''', bad)
            page.reload()
            page.clock.run_for(2000)
            c.show_operator(page, 'recordings')
            expect(page.locator('#quest-export-download')).to_be_hidden()
            expect(page.locator('#quest-archives')).to_be_hidden()
        page.context.close()

    page = mounted(review, initial(c) + pending(journal()))
    page.clock.run_for(2000)
    seed(page)
    c.show_operator(page,'settings')
    page.locator('#quest-disconnect').click()
    expect(page.locator('#recordings-open')).to_be_enabled()
    page.locator('#recordings-open').click()
    assert_download(page)
    expect(page.locator('#recordings-screen #preview-recovery')).to_be_visible()
    assert page.evaluate(f"JSON.parse(sessionStorage.getItem('{KEY}'))") == journal()
    assert mutations(page)['session'] == mutations(page)['setup'] == []
    page.context.close()

    experiment = dict(id='e'*16,op='pause',generation=3,owner=OWNER)
    extra = ("const pair=JSON.parse(localStorage.getItem('atmungsraum-quest-pair'));pair.build='e'.repeat(64);"
             "localStorage.setItem('atmungsraum-quest-pair',JSON.stringify(pair));window.reviewJoin='fail';"
             "sessionStorage.setItem('atmungsraum-quest-pending',JSON.stringify(" + json.dumps(experiment) + '));')
    page = mounted(review, initial(c) + pending(journal()) + extra)
    page.clock.run_for(2000)
    expect(page.locator('#recordings-open')).to_be_disabled()
    assert mutations(page)['session'] == mutations(page)['setup'] == []
    page.context.close()


def layout_cases(review):
    for label, viewport, size in SIZES:
        page = mounted(review, initial(c, dict(phase='RUNNING',host='vr-session',generation=3,
            condition='condition-a',receiptState='outcome_unknown'),
            dict(stage='main',input='controller-state',playing=True,ready=True)) + pending(journal()), viewport=viewport)
        page.evaluate('size=>document.documentElement.style.fontSize=size', size)
        page.clock.run_for(2000)
        expect(page.locator('#quest-finish')).to_be_enabled()
        review.shot(page,'journal-current-session-'+label,label+' current session with old journal',page.locator('#quest-finish'),'Recovery repairs')
        page.context.close()
        page = mounted(review, initial(c) + (pending(journal()) if label == '320-200text' else ''), viewport=viewport)
        page.clock.run_for(2000)
        seed(page)
        c.show_operator(page,'settings')
        page.locator('#quest-disconnect').click()
        c.show_operator(page,'recordings')
        page.evaluate('size=>document.documentElement.style.fontSize=size',size)
        page.clock.run_for(1000)
        assert_download(page)
        review.shot(page,'offline-archive-'+label,label+' offline committed archive',page.locator('#quest-export-download'),'Recovery repairs')
        page.context.close()


def run():
    parser = ArgumentParser()
    parser.add_argument('--scope', choices=['all', 'session', 'archives', 'layout'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        review = c.Review(browser, args.output)
        if args.scope in ['all', 'session']:
            session_cases(review)
        if args.scope in ['all', 'archives']:
            archive_cases(review)
        if args.scope in ['all', 'layout']:
            layout_cases(review)
        assert not review.errors, review.errors
        (args.output / ('session-archive-' + args.scope + '.json')).write_text(json.dumps(review.regressions, indent=2))
        (args.output / 'layout-checks.json').write_text(json.dumps(review.checks, indent=2))
        browser.close()
    print('PASS mounted session journal and offline committed archive recovery: ' + args.scope)


if __name__ == '__main__':
    run()
