"""Explicit preparation/save/update in the mounted PWA; synthetic enrolled transport."""
from pathlib import Path
import json
import os
import tempfile
from playwright.sync_api import sync_playwright, expect
from layout_render_fixtures import module, initial, go, mutations
from release_fixtures import meta_update_fixture, apk_fixture


def authored_instruction_cases(c, review):
    page = review.page(initial(c, quest={'metaUpdate': meta_update_fixture(), 'updatePrepEnabled': True,
                                         'calibrationInstruction': True}), viewport={'width': 320, 'height': 568})
    page.clock.run_for(1500)
    c.enter_participant(page)
    page.locator('#preview-calibrate').click()
    expect(page.locator('#calibration-instruction-cancel')).to_be_enabled()
    original = page.evaluate('reviewQuest.setup.instruction_id')
    page.evaluate("""Object.assign(reviewQuest.metaUpdate, {can_check:false,
        can_end_preparation:true, busy_reason:'preparation-audio-active'})""")
    page.clock.run_for(1500)
    go(page, 'settings')
    page.locator('#operator-update-open').click()
    check = page.locator('#meta-update-check')
    expect(check).to_have_text('Vorbereitung beenden und aktualisieren')
    expect(check).to_be_enabled()
    assert page.evaluate('window.reviewMetaUpdateCommands??[]') == []
    review.shot(page, 'calibration-instruction-update', 'Calibration narration can end for update', check, 'Preparation update')
    check.click()
    page.clock.run_for(1200)
    attempt = page.evaluate('reviewQuest.metaUpdate.attempt')
    assert attempt == page.evaluate('reviewReceipts[reviewMetaUpdateCommands[0].id].attempt')
    page.evaluate("reviewQuest.metaUpdate.busy_reason = 'narration-park-pending'")
    page.evaluate("document.documentElement.style.fontSize = '200%'")
    page.clock.run_for(16000)
    expect(page.locator('#meta-update-status')).to_have_text('Anleitung wird beendet.')
    detail = page.locator('#update-instruction')
    expect(detail).to_contain_text('schließe Atmungsraum im Headset')
    expect(detail).to_contain_text('öffne die App erneut und verbinde das Smartphone wieder')
    expect(check).to_be_disabled()
    review.shot(page, 'narration-awaiting-release', 'Narration awaits native release', detail, 'Preparation update')
    page.evaluate("Object.assign(reviewQuest.metaUpdate, {reason:'', busy_reason:''})")
    page.clock.run_for(1200)
    expect(page.locator('#meta-update-status')).to_contain_text('prüft den Meta-Kanal')
    expect(detail).not_to_contain_text('schließe Atmungsraum')
    assert page.evaluate('reviewQuest.metaUpdate.attempt') == attempt
    assert [cmd['op'] for cmd in page.evaluate('window.reviewMetaUpdateCommands') if cmd['op'] != 'receipt'] == ['end-preparation-and-check']
    assert page.evaluate("JSON.parse(sessionStorage.getItem('atmungsraum-breathing-setup-pending')).command.id") == original
    assert [cmd['op'] for cmd in mutations(page)['setup']] == ['calibrate']
    assert mutations(page)['session'] == []
    page.context.close()

    page = review.page(initial(c, quest={'metaUpdate': meta_update_fixture(), 'updatePrepEnabled': True,
        'scenarios': ['probe-5'], 'probeInstructionMs': 45401}), viewport={'width': 320, 'height': 568})
    c.enter_participant(page)
    page.locator('#preview-controller').click()
    page.locator('#preview-continue').click()
    c.review_battery(page)
    page.locator('#quest-play').click()
    expect(page.locator('#running-screen-title')).to_have_text('Anleitung zur Probe-Runde')
    original = page.evaluate("JSON.parse(sessionStorage.getItem('atmungsraum-quest-pending')).id")
    go(page, 'settings')
    page.locator('#operator-update-open').click()
    expect(page.locator('#update-screen')).to_be_visible()
    save = page.locator('#meta-update-finish')
    expect(save).to_be_visible()
    expect(save).to_be_disabled()
    expect(page.locator('#update-instruction')).to_contain_text('Anleitung zur Probe-Runde')
    assert page.evaluate("JSON.parse(sessionStorage.getItem('atmungsraum-quest-pending')).id") == original
    review.shot(page, 'probe-instruction-update', 'Guided Start awaits its original receipt', save, 'Preparation update')
    page.clock.run_for(47000)
    expect(save).to_be_enabled()
    assert mutations(page)['session'] == ['arm', 'start']
    assert page.evaluate('window.reviewMetaUpdateCommands??[]') == []
    page.context.close()


def run():
    c = module()
    # The fixture implements the agreed native contract. Production UI, command
    # journals, reflow and Pretext remain the shipped modules.
    c.GATEWAY = c.GATEWAY.replace('...(settings.metaUpdate ? {meta_update_control:1} : {}),',
        '...(settings.metaUpdate ? {meta_update_control:1} : {}), ...(settings.updatePrepEnabled ? {meta_update_preparation:1} : {}),')
    c.GATEWAY = c.GATEWAY.replace("rec: phase !== 'IDLE' && phase !== 'ERROR'", "rec: !settings.suppressRecording && phase !== 'IDLE' && phase !== 'ERROR'")
    c.GATEWAY = c.GATEWAY.replace("if(value.cycle !== current.cycle || value.revision !== current.revision || !current.can_check)",
        "if(value.cycle !== current.cycle || value.revision !== current.revision || !(current.can_check || value.op === 'end-preparation-and-check' && settings.updatePrepEnabled && current.can_end_preparation))")
    c.GATEWAY = c.GATEWAY.replace("Object.assign(current,{state:'checking',revision:String(BigInt(current.revision)+1n)",
        "if(value.op === 'end-preparation-and-check') Object.assign(current,{cycle:String(BigInt(current.cycle)+1n),reason:'ending-preparation',can_end_preparation:false});Object.assign(current,{state:'checking',revision:String(BigInt(current.revision)+1n)")
    c.GATEWAY = c.GATEWAY.replace("if (op === 'save-next') {\n    settings.clock = 0;", """if (op === 'save-next' && settings.deferSave) {
    window.reviewSaveCommand=structuredClone({id,op});
    settings.phase='SAVING';settings.pending=true;
    return receipts[id]={v:1,id,state:'pending'};
  }
  if (op === 'save-next') {
    settings.clock = 0;""")
    c.GATEWAY = c.GATEWAY.replace('const receipts = window.reviewReceipts ??= {};', '''const receipts = window.reviewReceipts ??= {};
window.reviewCompleteSave=(outcome='confirmed')=>{
  settings.pending=false;settings.phase='IDLE';settings.host='vr-preparation';
  if(outcome==='confirmed') Object.assign(settings.metaUpdate,{cycle:String(BigInt(settings.metaUpdate.cycle)+1n),revision:'1',
    attempt:'778',state:'checking',reason:'ending-preparation',can_check:false,can_end_preparation:false,session_allowed:false});
  if(outcome==='confirmed' && settings.saveParkPending) Object.assign(settings.metaUpdate,{attempt:'19',state:'current',reason:'',busy_reason:'setup-park-pending',session_allowed:true});
  if(outcome==='confirmed' && settings.saveOldCycle) Object.assign(settings.metaUpdate,{cycle:String(BigInt(settings.metaUpdate.cycle)-1n),busy_reason:'writer-active'});
  receipts[window.reviewSaveCommand.id]={v:1,id:window.reviewSaveCommand.id,state:outcome};
};''')
    with sync_playwright() as p, tempfile.TemporaryDirectory() as temporary:
        output = Path(os.environ.get('ATMUNGSRAUM_PREPARATION_UPDATE_OUTPUT', temporary))
        output.mkdir(parents=True, exist_ok=True)
        browser = p.chromium.launch()
        review = c.Review(browser, output)
        for width, size in [(390, ''), (320, ''), (320, '200%')]:
            native = meta_update_fixture(state='update_required', installed_code=23,
                required_code=25, available_code=25, session_allowed=False, can_check=False,
                busy_reason='session-active')
            page = review.page(initial(c, quest={'apk':dict(apk_fixture(), version_code=23), 'metaUpdate':native,
                'healthSupported':False}, setup={'playing':True,'stage':'testing','parked':False,
                    'audio_state':'playing'}), viewport={'width':width,'height':844 if width==390 else 568})
            page.evaluate('size=>document.documentElement.style.fontSize=size', size)
            page.clock.run_for(1500)
            expect(page.locator('#operator-connect-next')).to_have_text('Quest-App aktualisieren')
            page.locator('#check-polar').click()
            expect(page.locator('#connection-diagnostic-message')).to_contain_text('Quest-App aktualisieren')
            expect(page.locator('#connection-diagnostic-message')).to_contain_text('Tragekontakt unbekannt')
            page.keyboard.press('Escape')
            page.locator('#operator-connect-next').click()
            expect(page.locator('#update-screen')).to_be_visible()
            expect(page.locator('#update-instruction')).to_contain_text('Quest-App')
            expect(page.locator('#update-instruction')).to_contain_text('Vorbereitung')
            assert not page.locator('#apk-identity').evaluate('n=>n.open')
            review.shot(page, f'legacy-blocker-{width}-{size or "normal"}', 'Legacy preparation blocker',
                        page.locator('#meta-update-check'), 'Preparation update')
            assert mutations(page)['session'] == mutations(page)['setup'] == []
            page.context.close()
        for width, size in [(390, ''), (320, ''), (320, '200%')]:
            for stage, input_mode in [('calibrating','automatic'), ('testing','controller-state'),
                                      ('testing','polar-tilt-fixed'), ('testing','polar-tilt-adaptive')]:
                native = meta_update_fixture(can_check=False, can_end_preparation=True, busy_reason='preparation-active')
                old = {'owner':{'device':'d'*32,'serial':'DEMO-QUEST','epoch':'7','native_epoch':'8'},
                    'command':{'v':1,'id':'f'*16,'g':1,'r':0,'op':'controller'}}
                extra = "sessionStorage.setItem('atmungsraum-breathing-setup-pending',JSON.stringify("+json.dumps(old)+'));'
                page = review.page(initial(c, quest={'metaUpdate':native,'updatePrepEnabled':True,'receiptState':'outcome_unknown'},
                    setup={'stage':stage,'input':input_mode,'playing':True,'parked':False,'audio_state':'playing'})+extra,
                    viewport={'width':width,'height':844 if width==390 else 568})
                page.evaluate('size=>document.documentElement.style.fontSize=size', size)
                page.clock.run_for(1500)
                go(page, 'settings')
                page.locator('#operator-update-open').click()
                expect(page.locator('#update-screen')).to_be_visible()
                button = page.locator('#meta-update-check')
                expect(button).to_have_text('Vorbereitung beenden und aktualisieren')
                expect(button).to_be_enabled()
                assert page.evaluate('window.reviewMetaUpdateCommands??[]') == []
                button.click()
                page.clock.run_for(1200)
                commands = page.evaluate('window.reviewMetaUpdateCommands??[]')
                assert len([cmd for cmd in commands if cmd['op']=='end-preparation-and-check'])==1, commands
                expect(page.locator('#meta-update-status')).to_contain_text('Vorbereitung')
                expect(page.locator('#update-instruction')).to_contain_text('Ruheposition')
                expect(page.locator('#preview-recovery')).to_be_hidden()
                page.locator('#apk-identity summary').click()
                expect(page.locator('#preview-recovery')).to_be_visible()
                page.locator('#apk-identity summary').click()
                page.locator('#apk-identity summary').focus()
                page.evaluate('scrollTo(0,Math.max(0,document.documentElement.scrollHeight-innerHeight-10))')
                before_scroll = page.evaluate('scrollY')
                page.clock.run_for(1400)
                assert abs(page.evaluate('scrollY')-before_scroll)<=2, 'native progress moved the update document'
                expect(page.locator('#apk-identity summary')).to_be_focused()
                assert page.evaluate("JSON.parse(sessionStorage.getItem('atmungsraum-breathing-setup-pending'))") == old
                assert mutations(page)['session'] == mutations(page)['setup'] == []
                review.shot(page, f'end-preparation-{stage}-{input_mode}-{width}-{size or "normal"}',
                    'Explicit preparation end', page.locator('#update-instruction'), 'Preparation update')
                page.context.close()
        # Only the explicit recording button may stop a run. Writer completion
        # and park are separate observations; no elapsed clock can grant either.
        for outcome in ['confirmed', 'outcome_unknown', 'rejected']:
            native = meta_update_fixture(can_check=False, can_end_preparation=False, busy_reason='session-active')
            page = review.page(initial(c, quest={'metaUpdate':native,'updatePrepEnabled':True,'deferSave':True,
                'phase':'RUNNING','host':'vr-session','generation':3,'condition':'condition-a'},
                setup={'stage':'main','playing':True,'parked':False,'audio_state':'playing'}),
                viewport={'width':320,'height':568})
            page.clock.run_for(1500)
            assert mutations(page)['session'] == []
            go(page,'settings')
            expect(page.locator('#quest-finish')).to_be_enabled()
            page.locator('#operator-update-open').click()
            expect(page.locator('#update-screen')).to_be_visible()
            button=page.locator('#meta-update-finish')
            expect(button).to_be_enabled()
            assert page.evaluate('window.reviewMetaUpdateCommands??[]') == []
            review.shot(page, 'recording-explicit-'+outcome, 'Explicit save and update', button, 'Preparation update')
            button.click()
            page.clock.run_for(2500)
            assert mutations(page)['session'] == ['save-next']
            assert page.evaluate('window.reviewMetaUpdateCommands??[]') == []
            expect(page.locator('#meta-update-status')).to_contain_text('speichert')
            page.evaluate('outcome=>reviewCompleteSave(outcome)', outcome)
            page.clock.run_for(2000)
            assert mutations(page)['session'] == ['save-next']
            assert not any(cmd['op']!='receipt' for cmd in page.evaluate('window.reviewMetaUpdateCommands??[]'))
            if outcome=='confirmed':
                expect(page.locator('#update-screen')).to_be_visible()
                expect(page.locator('#update-instruction')).to_contain_text('Ruheposition')
                page.evaluate("Object.assign(reviewQuest.setup,{playing:false,parked:true,audio_state:'paused'});Object.assign(reviewQuest.metaUpdate,{state:'current',reason:'',session_allowed:true,can_check:true})")
                page.clock.run_for(1300)
                expect(page.locator('#meta-update-status')).to_have_text('Quest-App ist aktuell.')
            elif outcome=='outcome_unknown':
                assert page.evaluate("JSON.parse(sessionStorage.getItem('atmungsraum-quest-pending')).id") == page.evaluate('reviewSaveCommand.id')
                page.evaluate("Object.assign(reviewQuest.metaUpdate,{cycle:'3',attempt:'778',can_check:true,state:'current'});")
                page.clock.run_for(16000)
                assert not any(cmd['op']!='receipt' for cmd in page.evaluate('window.reviewMetaUpdateCommands??[]'))
            page.context.close()
        for change in ['automatic', 'manual', 'lease', 'owner', 'cleanup']:
            native=meta_update_fixture(can_check=False,can_end_preparation=False,busy_reason='writer-active')
            page=review.page(initial(c,quest={'metaUpdate':native,'updatePrepEnabled':True,'deferSave':True,'saveParkPending':True,
                'phase':'ARMED','suppressRecording':True,'saveOldCycle':change=='cleanup','host':'vr-session','generation':3,'condition':'condition-a'},
                setup={'stage':'main','playing':True,'parked':False,'audio_state':'playing'}),viewport={'width':320,'height':568})
            page.evaluate("document.documentElement.style.fontSize='200%'")
            page.clock.run_for(1500)
            go(page,'settings')
            expect(page.locator('#quest-finish')).to_be_enabled()
            page.locator('#operator-update-open').click()
            button=page.locator('#meta-update-finish')
            expect(button).to_be_enabled()
            button.click()
            page.clock.run_for(1000)
            page.evaluate('reviewCompleteSave()')
            page.clock.run_for(2000)
            expect(page.locator('#meta-update-status')).to_have_text('Aufnahme gespeichert. Quest bereitet das Update vor.')
            expect(page.locator('#update-instruction')).to_contain_text('Ruheposition')
            expect(page.locator('#meta-update-save-retry')).to_be_enabled()
            assert page.evaluate('window.reviewMetaUpdateCommands??[]') == []
            review.shot(page,'saved-park-pending-'+change,'Saved writer awaits actual park',
                        page.locator('#meta-update-save-retry'),'Preparation update')
            # The old native current/attempt may remain longer than a timer.
            page.clock.run_for(16000)
            expect(page.locator('#meta-update-status')).not_to_have_text('Quest-App ist aktuell.')
            assert mutations(page)['session']==['save-next']
            assert page.evaluate('window.reviewMetaUpdateCommands??[]') == []
            if change=='cleanup':
                page.evaluate("Object.assign(reviewQuest.metaUpdate,{cycle:'3',revision:'1',busy_reason:'setup-park-pending'})")
                page.clock.run_for(2000)
                expect(page.locator('#meta-update-status')).to_have_text('Aufnahme gespeichert. Quest bereitet das Update vor.')
                assert page.evaluate('window.reviewMetaUpdateCommands??[]') == []
            if change=='lease':
                page.evaluate('reviewQuest.access.revision++')
            elif change=='owner':
                page.evaluate("reviewOwner.native_epoch='9'")
            if change in ['automatic','cleanup']:
                page.evaluate("Object.assign(reviewQuest.metaUpdate,{attempt:'778',revision:'2',state:'checking',reason:'',session_allowed:false})")
            else:
                page.evaluate("Object.assign(reviewQuest.setup,{playing:false,parked:true,audio_state:'paused'});Object.assign(reviewQuest.metaUpdate,{can_check:true,busy_reason:''})")
            page.clock.run_for(2000)
            commands=page.evaluate('window.reviewMetaUpdateCommands??[]')
            if change=='manual':
                assert [cmd['op'] for cmd in commands if cmd['op']!='receipt']==['check-and-update'], commands
                page.clock.run_for(4000)
                assert len([cmd for cmd in page.evaluate('window.reviewMetaUpdateCommands??[]') if cmd['op']!='receipt'])==1
            elif change in ['lease','owner']:
                assert commands==[], commands
                expect(page.locator('#update-instruction')).to_contain_text('erneut')
                expect(page.locator('#meta-update-status')).to_have_text('Update noch nicht bestätigt.')
            else:
                assert commands==[], commands
                expect(page.locator('#meta-update-status')).to_contain_text('Meta-Kanal')
            page.context.close()
        authored_instruction_cases(c, review)
        assert not review.errors, review.errors
        browser.close()
        print(f'PASS {len(review.checks)} mounted preparation update renders; narrow/200% actual Pretext')


if __name__ == '__main__':
    run()
