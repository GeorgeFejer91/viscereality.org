"""Real app/gateway layout qualification. No static HTML or readiness bypass."""
import json
from playwright.sync_api import expect

SIZES = (("390", {"width":390,"height":844}, ""),
         ("320", {"width":320,"height":568}, ""),
         ("320-200text", {"width":320,"height":568}, "200%"),
         ("landscape", {"width":844,"height":390}, ""),
         ("desktop", {"width":1440,"height":900}, ""))
APK = {"v":1,"package_name":"org.viscereality.atmungsraum.developer",
       "version_code":22,"version_name":"0.2.21-dev","build_id":"e"*64}


def module():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('capture_layout_gateway',Path(__file__).with_name('capture-phone-screens.py'))
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    return c


def initial(c, quest=None, setup=None):
    state = {"polar":"ready","apk":APK,"advanceRenderer":True,"access":{"v":1,"role":"active","active_phone":"Fixture","revision":1}}
    state.update(quest or {})
    setup_state = {"stage":"idle","playing":False,"audio_state":"paused","parked":True,
                   "polar_modes":["calibrated","fixed","adaptive"],
                   "renderer":{"v":1,"progress01":.5,"radius_m":1.5,"frame":1,
                               "render_generation":2,"settings_revision":0,"observed_ns":"200000000","age_ns":0}}
    setup_state.update(setup or {})
    return (f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({c.PAIRED}));"
            + 'window.reviewInitialQuest='+json.dumps(state)+';'
            + 'window.reviewInitialSetup='+json.dumps(setup_state)+';')


def go(page, step):
    direct = page.locator(f'[data-operator-go="{step}"]:visible')
    if direct.count():
        direct.first.click()
    else:
        module().show_operator(page, step)


def mutations(page):
    page.wait_for_function('Boolean(window.reviewQuest)')
    return page.evaluate('({session:reviewQuest.commands??[],setup:reviewSetupCommands,access:reviewAccessCommands,recenter:reviewRecenterCommands,kiosk:window.reviewKioskCommands??[]})')


def participant(c, page):
    c.enter_participant(page)
    expect(page.locator('#input-screen')).to_be_visible()


def capture_layout(review):
    c = module()
    for size, viewport, text in SIZES:
        def new(quest=None, setup=None, extra=''):
            page = review.page(initial(c,quest,setup)+extra,viewport=viewport)
            page.evaluate('size=>document.documentElement.style.fontSize=size',text)
            page.clock.run_for(1500)
            return page
        def shot(page, slug, expected):
            review.shot(page,f'layout-{size}-{slug}',f'{size} · {slug}',page.locator(expected),'Operator layout')

        page = new()
        before = mutations(page)
        shot(page,'connect','#operator-connect-next')
        page.locator('#operator-connect-next').click()
        shot(page,'participant','#patient-form')
        participant(c,page)
        shot(page,'input','#preview-controller')
        go(page,'update')
        page.locator('#apk-identity summary').click()
        expect(page.locator('#quest-version')).to_have_attribute('data-version-state','compatible')
        shot(page,'update-identity','#apk-full-identity')
        page.locator('[data-operator-return]:visible').click()
        go(page,'recordings')
        shot(page,'recordings','#recording-files')
        page.locator('a[href="#daten"]').click()
        shot(page,'legacy-recordings','#data-status')
        page.go_back()
        page.clock.run_for(100)
        expect(page.locator('#recordings-screen')).to_be_visible()
        page.locator('[data-operator-return]:visible').click()
        expect(page.locator('#input-screen')).to_be_visible()
        assert mutations(page) == before, 'Local setup/settings/history navigation mutated gateway'
        assert page.locator('#patient-surname').input_value() == 'Tes'
        page.locator('#preview-calibrate').click()
        assert [x['op'] for x in mutations(page)['setup']] == ['calibrate']
        expect(page.locator('#calibration-screen')).to_be_visible()
        c.wait_setup(page,position_frames=705600,completed_frames=705600)
        expect(page.locator('#preview-progress')).to_have_js_property('value',1)
        shot(page,'calibration-one-cycle','#preview-progress')
        c.wait_setup(page,stage='testing',calibration='ready',input='polar-tilt',ready=True,
                     completed_frames=1411200,position_frames=1499400,input_phase='inhale',input_volume01=.5)
        shot(page,'tracking','#preview-continue')
        expect(page.locator('#preview-actual-circle')).to_be_visible()
        expect(page.locator('#preview-actual-volume')).to_have_text('50 %')
        expect(page.locator('#preview-actual-status')).to_contain_text('aktuell')
        inset=page.evaluate('''() => {
          const blue=document.querySelector('#preview-circle'), gold=document.querySelector('#preview-actual-circle');
          return {blue:getComputedStyle(blue).width,gold:getComputedStyle(gold).width,
            actual:gold.style.transform,desired:blue.style.transform,actualText:document.querySelector('#preview-actual-volume').textContent};
        }''')
        assert float(inset['blue'].removesuffix('px'))-float(inset['gold'].removesuffix('px'))==4,inset
        assert inset['actual']==inset['desired'],inset
        review.regressions.append({'kind':'actual-renderer-inset','size':size,**inset})
        expect(page.locator('#preview-continue')).to_be_enabled()
        page.locator('#preview-continue').click()
        expect(page.get_by_role('region', name='Vorbereitung').get_by_role('radio')).to_have_count(2)
        expect(page.get_by_role('region', name='Hauptstudie').get_by_role('radio')).to_have_count(1)
        expect(page.locator('#quest-condition')).to_have_count(0)
        requested_scenario = {'390':'haupt-15', '320':'probe-5'}.get(size, 'probe-anleitung')
        selection = page.locator(f'input[name="quest-scenario"][value="{requested_scenario}"]')
        selection.check()
        shot(page,'session','#quest-play')
        # A new native attempt invalidates the accepted tracking token without
        # a browser command; the operator must deliberately accept it again.
        c.wait_setup(page,attempt=2)
        expect(page.locator('#breathing-preview')).to_be_visible()
        assert mutations(page)['session']==[]
        page.locator('#preview-continue').click()
        expect(page.locator('#session-screen')).to_be_visible()
        page.locator('#quest-play').click()
        expect(page.locator('#running-screen')).to_be_visible()
        assert page.evaluate('reviewQuest.commands') == ['arm','start']
        assert page.evaluate('reviewQuest.armedScenario') == requested_scenario
        for phase in ('RUNNING','PAUSED','FINALIZING','SAVING'):
            c.wait_state(page,phase=phase,host='vr-session',clock=18000)
            expect(page.locator('#running-screen #quest-ecg')).to_be_visible()
            expect(page.locator('#quest-play')).to_be_visible()
            expect(page.locator('#quest-finish')).to_be_visible()
            if phase in ('FINALIZING','SAVING'):
                expect(page.locator('#quest-play')).to_be_disabled()
                expect(page.locator('#quest-finish')).to_be_disabled()
            shot(page,phase.lower(),'#quest-clock')
            nav_before=mutations(page)
            go(page,'settings')
            expect(page.locator('#settings-screen #quest-play')).to_be_visible()
            expect(page.locator('#settings-screen #quest-finish')).to_be_visible()
            expect(page.locator('#settings-screen #quest-ecg')).to_be_visible()
            shot(page,phase.lower()+'-settings','#quest-clock')
            page.locator('[data-operator-return]:visible').click()
            assert mutations(page)==nav_before
        c.wait_state(page,phase='RUNNING')
        page.locator('#quest-play').click()
        expect(page.locator('#quest-play')).to_contain_text('Fortsetzen')
        page.locator('#quest-play').click()
        expect(page.locator('#quest-play')).to_contain_text('Pause')
        page.locator('#quest-finish').click()
        assert page.evaluate('reviewQuest.commands') == ['arm','start','pause','resume','save-next']
        expect(page.locator('#recordings-screen')).to_be_visible()
        next_before=mutations(page)
        go(page,'session')
        # Arm changed the native session generation, so the existing acceptance
        # token correctly requires another tracking check before selection.
        expect(page.locator('#breathing-preview')).to_be_visible()
        assert mutations(page)==next_before
        assert page.locator('#patient-id').text_content()=='Tes_30_N'
        review.regressions.append({'kind':'layout-native-flow','size':size,'sessionCommands':['arm','start','pause','resume','save-next'],'localMutations':0})
        page.context.close()

        for op,button,expected_input in (('polar-fixed','preview-polar-fixed','polar-tilt-fixed'),
                                        ('polar-adaptive','preview-polar-adaptive','polar-tilt-adaptive'),
                                        ('controller','preview-controller','controller-state')):
            page=new({'polar':'stream-wait','ecg':False,'acc':False} if op=='controller' else None)
            participant(c,page)
            go(page,'input')
            before=mutations(page)
            assert before['setup']==[]
            expect(page.locator('#'+button)).to_be_enabled()
            page.locator('#'+button).click()
            expect(page.locator('#breathing-preview')).to_be_visible()
            assert page.evaluate('reviewQuest.setup.input')==expected_input
            assert [x['op'] for x in mutations(page)['setup']]==[op]
            if op=='controller':
                expect(page.locator('#operator-signal-warning')).to_be_visible()
                expect(page.locator('#preview-continue')).to_be_disabled()
            shot(page,'explicit-'+op,'#preview-actual')
            go(page,'input')
            assert [x['op'] for x in mutations(page)['setup']]==[op], 'Change input sent command'
            review.regressions.append({'kind':'explicit-mode','size':size,'op':op,'input':expected_input,'missingStreams':op=='controller'})
            page.context.close()

        page=new(setup={'stage':'testing','calibration':'failed','failure':'excessive-motion','attempt':1,
                        'target_frames':1411200,'completed_frames':1411200,'playing':True,'parked':False})
        participant(c,page)
        go(page,'input')
        expect(page.locator('#preview-detail')).to_contain_text('fehlgeschlagen')
        shot(page,'localized-calibration-failure','#preview-detail')
        page.context.close()

        for label, quest in (('unknown-apk',{'apk':None}),
                             ('long-controller',{'access':{'v':1,'role':'waiting','revision':1,
                              'active_phone':'Studienleitung mit sehr langem Gerätenamen für diese Übergabe'}})):
            page=new(quest)
            if label=='unknown-apk':
                expect(page.locator('#operator-compatibility')).to_be_visible()
                go(page,'update')
                page.locator('#apk-identity summary').click()
                expect(page.locator('#quest-version')).to_have_attribute('data-version-state','unknown')
                shot(page,label,'#apk-full-identity')
            else:
                expect(page.locator('#quest-control-claim')).to_be_disabled()
                shot(page,label,'#quest-control-status')
            assert mutations(page)['setup']==[]
            page.context.close()

        review.release_policy_failure=True
        page=new()
        expect(page.locator('#operator-compatibility')).to_be_visible()
        go(page,'update')
        expect(page.locator('#quest-version')).to_have_attribute('data-version-state','check-unavailable')
        shot(page,'release-policy-failed','#apk-update-status')
        page.context.close()
        review.release_policy_failure=False

        journal={'owner':{'serial':'DEMO-QUEST','device':'d'*32,'epoch':'7','native_epoch':'8'},
                 'command':{'v':1,'id':'f'*16,'g':1,'r':0,'op':'calibrate'}}
        saved="sessionStorage.setItem('atmungsraum-breathing-setup-pending',JSON.stringify("+json.dumps(journal)+'));'
        page=new({'receiptState':'outcome_unknown'},extra=saved)
        expect(page.locator('#recovery-screen')).to_be_visible()
        expect(page.locator('#preview-recovery')).to_contain_text('Ergebnis der Anfrage unklar')
        shot(page,'pending-original-receipt','#preview-recovery')
        go(page,'settings')
        expect(page.locator('#settings-screen #preview-recovery')).to_be_visible()
        shot(page,'pending-receipt-settings','#preview-recovery')
        assert page.evaluate("JSON.parse(sessionStorage.getItem('atmungsraum-breathing-setup-pending')).command.id")=='f'*16
        assert mutations(page)['setup']==[]
        c.wait_state(page,receiptState='confirmed')
        c.wait_setup(page,r=1,stage='waiting',calibration='waiting',attempt=1,target_frames=1411200)
        page.locator('[data-operator-return]:visible').click()
        c.show_operator(page,'participant')
        assert page.evaluate("sessionStorage.getItem('atmungsraum-breathing-setup-pending')") is None
        assert mutations(page)['setup']==[]
        shot(page,'receipt-resolved-no-replay','#patient-form')
        review.regressions.append({'kind':'pending-receipt','size':size,'originalId':'f'*16,'setupMutations':0})
        page.context.close()

    capture_poll_cadence(review,c)


def capture_poll_cadence(review,c):
    init=initial(c,setup={'stage':'testing','playing':True,'parked':False,'input':'controller-state','ready':True})
    init+="Object.defineProperty(document,'hidden',{configurable:true,get:()=>Boolean(window.reviewHidden)});"
    page=review.page(init)
    participant(c,page)
    go(page,'input')
    go(page,'tracking')
    page.clock.run_for(1500)
    def counts():
        return page.evaluate('({status:window.reviewStatusReads??0,ecg:window.reviewEcgCalls??0,receipts:(window.reviewSetupReceiptQueries??[]).length,max:window.reviewStatusMaxInFlight??0})')
    observations=[]
    for step in ('tracking','input','settings'):
        if step!='tracking':
            go(page,step)
            page.clock.run_for(1500)
        before=counts()
        page.clock.run_for(4000)
        after=counts()
        delta={key:after[key]-before[key] for key in ('status','ecg','receipts')}
        # Forty preview-only reads plus the unchanged four ordinary status reads.
        assert (43<=delta['status']<=45 if step=='tracking' else 3<=delta['status']<=5),(step,delta)
        assert 3<=delta['ecg']<=5,(step,delta)
        assert delta['receipts']==0,(step,delta)
        assert after['max']==1,(step,after)
        observations.append({'step':step,**delta,'maxInFlight':after['max']})
    page.locator('[data-operator-return]:visible').click()
    go(page,'input')
    go(page,'tracking')
    page.evaluate('reviewQuest.statusDelay=250')
    page.clock.run_for(4000)
    assert counts()['max']==1,counts()
    page.evaluate("window.reviewHidden=true;document.dispatchEvent(new Event('visibilitychange'))")
    page.clock.run_for(500)
    hidden=counts()
    page.clock.run_for(4000)
    assert counts()==hidden,(hidden,counts())
    assert mutations(page)['setup']==[] and mutations(page)['session']==[]
    review.regressions.append({'kind':'read-only-preview-cadence','ordinaryAndPreview':observations,'slowMaxInFlight':1,'hiddenDelta':0,'nativeMutations':0})
    page.context.close()

    page=review.page(initial(c))
    participant(c,page)
    go(page,'input')
    page.locator('#preview-calibrate').click()
    expect(page.locator('#calibration-screen')).to_be_visible()
    page.clock.run_for(1500)
    before=counts()
    page.clock.run_for(4000)
    after=counts()
    assert 43<=after['status']-before['status']<=45,(before,after)
    assert 3<=after['ecg']-before['ecg']<=5,(before,after)
    assert after['max']==1 and after['receipts']==before['receipts'],(before,after)
    assert [x['op'] for x in mutations(page)['setup']]==['calibrate']
    review.regressions.append({'kind':'calibration-status-only-cadence','statusReads':after['status']-before['status'],
                               'ecgReads':after['ecg']-before['ecg'],'receiptReads':0,'maxInFlight':after['max']})
    page.context.close()

    journal={'owner':{'serial':'DEMO-QUEST','device':'d'*32,'epoch':'7','native_epoch':'8'},
             'command':{'v':1,'id':'f'*16,'g':1,'r':0,'op':'calibrate'}}
    page=review.page(initial(c,{'receiptState':'pending'})
        +"sessionStorage.setItem('atmungsraum-breathing-setup-pending',JSON.stringify("+json.dumps(journal)+')) ;')
    expect(page.locator('#recovery-screen')).to_be_visible()
    page.clock.run_for(1500)
    before=counts()
    page.clock.run_for(4000)
    after=counts()
    assert 3<=after['receipts']-before['receipts']<=5,(before,after)
    assert 3<=after['status']-before['status']<=5,(before,after)
    assert mutations(page)['setup']==[]
    assert all(q['id']=='f'*16 for q in page.evaluate('reviewSetupReceiptQueries'))
    review.regressions.append({'kind':'pending-original-receipt-cadence','statusReads':after['status']-before['status'],
                               'receiptReads':after['receipts']-before['receipts'],'setupMutations':0,'originalId':'f'*16})
    page.context.close()
