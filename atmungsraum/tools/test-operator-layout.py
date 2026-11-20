"""Focused real-app local-route regressions; no hardware or account requests."""
from pathlib import Path
import importlib.util
import re
import tempfile
from playwright.sync_api import sync_playwright, expect
from layout_render_fixtures import SIZES, initial, mutations

spec = importlib.util.spec_from_file_location('capture_phone', Path(__file__).with_name('capture-phone-screens.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def compact_setup(review):
    for width, height in ((320, 640), (390, 844), (844, 390)):
        page = review.page(initial(c), viewport={'width': width, 'height': height})
        page.clock.run_for(4000)

        def assert_fits():
            expect(page.locator('html')).to_have_attribute('data-measurement', 'ready')
            page.clock.run_for(150)
            result = page.evaluate('''() => {
              const root = document.documentElement;
              const scope = document.querySelector('dialog[open], .screen:not([hidden])');
              const open = document.querySelector('dialog[open]');
              return {
                measured: root.dataset.measurement,
                layout: root.dataset.layout,
                page: [root.scrollWidth, root.scrollHeight],
                viewport: [innerWidth, innerHeight],
                dialogFits: !open || (open.scrollHeight <= open.clientHeight && open.scrollWidth <= open.clientWidth
                  && open.getBoundingClientRect().top >= document.querySelector('#operator-toolbar').getBoundingClientRect().bottom),
                textFits: ![...(open || scope).querySelectorAll('[data-text-fit="reflow"]')]
                  .some(node => node.checkVisibility())
              };
            }''')
            assert result['measured'] == 'ready' and result['page'][0] == result['viewport'][0], result
            assert result['page'][1] == result['viewport'][1] or result['layout'] == 'reflow', result
            assert result['dialogFits'] and result['textFits'], result

        assert_fits()
        header = page.get_by_role('banner', name='Geräte und Aktionen')
        settings = page.locator('#settings-open')
        recordings = page.locator('#recordings-open')
        utility = page.locator('#toolbar-actions') if width < 390 else settings
        expect(utility).to_be_visible(timeout=1500)
        controls = header.locator('button:visible').evaluate_all(
            '(buttons) => buttons.map(button => button.getBoundingClientRect().toJSON())')
        assert len({button['y'] for button in controls}) == 1, controls
        assert all(button['width'] >= 48 and button['height'] >= 48 for button in controls), controls
        utility.focus()
        page.keyboard.press('Enter')
        if width < 390:
            expect(page.locator('#toolbar-dialog')).to_be_visible()
            expect(recordings).to_be_visible()
            assert_fits()
            page.keyboard.press('Escape')
            expect(utility).to_be_focused()
            page.keyboard.press('Enter')
            settings.click()
        expect(page.locator('#settings-screen-title')).to_be_focused()
        page.locator('#settings-screen [data-operator-return]').click()
        expect(utility).to_be_focused()
        assert mutations(page)['session'] == []
        page.locator('#drive-open').click()
        expect(page.locator('#drive-dialog')).to_be_visible()
        assert_fits()
        page.keyboard.press('Escape')
        expect(page.locator('#drive-open')).to_be_focused()
        page.locator('#operator-connect-next').click()
        assert_fits()
        c.enter_participant(page)
        assert_fits()
        page.context.close()


def calibration_instruction(review):
    page = review.page(initial(c, {'calibrationInstruction': True}), viewport={'width': 320, 'height': 640})
    page.clock.run_for(4000)
    c.enter_participant(page)
    page.locator('#preview-calibrate').click()
    screen = page.locator('#calibration-screen')
    cancel = screen.get_by_role('button', name='Anleitung abbrechen')
    expect(screen.get_by_role('heading', name='Anleitung zur Atemübung')).to_be_visible()
    expect(screen.locator('#calibration-change-input')).to_be_hidden()
    expect(cancel).to_be_enabled()
    page.clock.run_for(35000)
    expect(cancel).to_be_enabled()
    expect(page.locator('html')).to_have_attribute('data-measurement', 'ready')
    assert cancel.locator('[data-measure]').get_attribute('data-text-fit') != 'reflow'
    assert page.evaluate('document.documentElement.scrollWidth === innerWidth')
    assert page.evaluate('reviewQuest.setup.attempt') == 0
    assert page.evaluate('reviewSetupCommands.map(command => command.op)') == ['calibrate']
    cancel.focus()
    page.keyboard.press('Enter')
    expect(cancel).to_be_hidden()
    page.clock.run_for(1000)
    expect(screen.get_by_role('button', name='Erneut einrichten')).to_be_enabled()
    assert page.evaluate('''() => {
        const [calibrate, cancel, ...extra] = reviewSetupCommands;
        return cancel.op === 'cancel-instruction' && cancel.target === calibrate.id
            && cancel.id !== calibrate.id && extra.length === 0 && reviewQuest.setup.attempt === 0;
    }''')
    page.context.close()


def recovery_and_recordings(review):
    page = review.page()
    expect(page.locator('#quest-find')).to_be_visible()
    page.locator('[data-operator-go="recordings"]:visible').click()
    expect(page.locator('#recordings-screen')).to_be_visible()
    page.locator('a[href="#daten"]').click()
    expect(page.locator('#data-status')).to_contain_text('Keine früheren Browser-Aufnahmen')
    page.locator('#daten .back').click()
    expect(page.locator('#recordings-screen')).to_be_visible()
    page.context.close()

    init = initial(c, {'access': {'v': 1, 'role': 'waiting', 'active_phone': '', 'revision': 2}}) + """
sessionStorage.setItem('atmungsraum-quest-pending', JSON.stringify({op:'start', id:'f'.repeat(16), generation:0,
    owner:{serial:'DEMO-QUEST', device:'d'.repeat(32), epoch:'6', native_epoch:'7'}}));
"""
    page = review.page(init)
    expect(page.locator('#quest-control-claim')).to_be_visible()
    expect(page.locator('#quest-control-claim')).to_be_enabled(timeout=3000)
    page.locator('#quest-control-claim').click()
    expect(page.locator('#quest-acknowledge')).to_contain_text('Aktuellen Quest-Zustand übernehmen')
    page.locator('#quest-acknowledge').click()
    assert mutations(page)['session'] == []
    page.context.close()

    page = review.page(initial(c))
    page.clock.run_for(4000)
    expect(page.locator('#operator-connect-next')).to_be_enabled()
    page.get_by_role('banner', name='Geräte und Aktionen').get_by_role('button', name='Aufnahmen').click()
    expect(page.locator('#recordings-screen')).to_be_visible()
    page.locator('#recordings-screen #drive-open').click()
    expect(page.locator('#drive-dialog')).to_be_visible()
    page.locator('#quest-drive-account').fill('fixture@example.test')
    page.clock.run_for(2400)
    expect(page.locator('#drive-dialog')).to_be_visible()
    expect(page.locator('#quest-drive-account')).to_have_value('fixture@example.test')
    expect(page.locator('#recordings-screen')).to_be_visible()
    page.evaluate("document.querySelector('#drive-dialog').close();document.querySelector('#operator-next-session').focus()")
    page.clock.run_for(100)
    expect(page.locator('#operator-next-session')).to_be_focused()
    page.locator('[data-operator-go="settings"]:visible').click()
    page.locator('#quest-disconnect').click()
    expect(page.locator('#quest-find')).to_be_focused()
    page.context.close()


def scenario_compatibility(review):
    primary = ['probe-anleitung', 'probe-5', 'haupt-15']
    legacy = ['probe-10', 'probe-15', 'box-20']
    for offered, tracks, visible, selected in (
        (primary + legacy, ['track-ready', 'track-ready'], primary, 'probe-5'),
        (['haupt-15'], ['track-ready', 'track-ready'], ['haupt-15'], 'haupt-15'),
        (['haupt-15', 'free-exploration'], ['track-ready', 'track-ready'], ['free-exploration', 'haupt-15'], 'free-exploration'),
        (legacy, ['track-ready', 'track-ready'], [], None),
        (None, ['track-ready', 'track-ready'], [], None),
        (['probe-5'], ['missing', 'track-ready'], ['probe-5'], None),
    ):
        page = review.page(initial(c, {'scenarios': offered, 'tracks': tracks}),
                           viewport={'width': 320, 'height': 568})
        c.enter_participant(page)
        page.locator('#preview-controller').click()
        page.locator('#preview-continue').click()
        c.review_battery(page)
        expect(page.locator('#session-screen')).to_be_visible()
        expect(page.locator('#quest-condition, #quest-legacy-options')).to_have_count(0)
        radios = page.locator('#quest-scenario').get_by_role('radio')
        outside = page.locator('.session-tile:visible').evaluate_all('''tiles => tiles.filter(tile => {
          const box = tile.getBoundingClientRect();
          const contents = tile.querySelector('.session-tile-text').getBoundingClientRect();
          return contents.top < box.top || contents.bottom > box.bottom
            || contents.left < box.left || contents.right > box.right;
        }).map(tile => tile.textContent.trim())''')
        assert not outside, f'scenario label extends outside its tile: {outside}'
        assert radios.evaluate_all('nodes => nodes.map(node => node.value)') == visible
        if selected:
            page.locator(f'input[value="{selected}"]').check()
            page.locator('#quest-play').click()
            expect(page.locator('#running-screen')).to_be_visible()
            assert page.evaluate('[reviewQuest.armedScenario, reviewQuest.condition]') == [selected, 'condition-a']
            assert mutations(page)['session'] == ['arm', 'start']
            if selected == 'free-exploration':
                c.wait_state(page, duration_ms=447070, clock=22000)
                progress = page.locator('#quest-progress')
                expect(progress).to_have_js_property('max', 447070)
                expect(progress).to_have_js_property('value', 22000)
                page.locator('#quest-play').click()
                expect(page.locator('#running-screen-title')).to_have_text('Sitzung pausiert')
                expect(progress).to_have_js_property('value', 22000)
                page.locator('#quest-play').click()
                expect(page.locator('#running-screen-title')).to_have_text('Sitzung läuft')
                assert mutations(page)['session'] == ['arm', 'start', 'pause', 'resume']
        else:
            expect(page.locator('#quest-play')).to_be_disabled()
            expect(page.locator('#quest-scenario-unavailable')).to_be_visible()
            assert mutations(page)['session'] == []
        page.context.close()


def authored_round_layout(review):
    offer = initial(c, {'scenarios': ['haupt-15', 'probe-5', 'free-exploration']})
    for name, viewport, text_size in SIZES:
        page = review.page(offer, viewport=viewport)
        page.evaluate('size => document.documentElement.style.fontSize = size', text_size)
        c.enter_participant(page)
        page.locator('#preview-controller').click()
        page.locator('#preview-continue').click()
        c.review_battery(page)
        choice = page.get_by_role('radio', name='Freie Erkundung')
        expect(choice).to_be_visible()
        expect(choice).to_be_enabled()
        expect(page.locator('html')).to_have_attribute('data-measurement', 'ready')
        page.clock.run_for(150)
        assert choice.evaluate('''input => {
          const tile = input.closest('label');
          const label = tile.querySelector('[data-measure]');
          const panel = tile.getBoundingClientRect(), text = label.getBoundingClientRect();
          return text.left >= panel.left && text.right <= panel.right
            && text.top >= panel.top && text.bottom <= panel.bottom
            && document.documentElement.scrollWidth === innerWidth;
        }'''), name
        page.context.close()


def guided_probe_instruction(review):
    page = review.page(initial(c, {'scenarios': ['probe-5', 'free-exploration', 'haupt-15'],
                                  'probeInstructionMs': 45401}), viewport={'width': 320, 'height': 640})
    c.enter_participant(page)
    page.locator('#preview-controller').click()
    page.locator('#preview-continue').click()
    c.review_battery(page)
    page.get_by_role('radio', name='Geführte Atmung 5 Min.').check()
    page.locator('#quest-play').click()
    heading = page.locator('#running-screen-title')
    expect(heading).to_have_text('Anleitung zur Probe-Runde')
    expect(page.locator('#quest-status')).to_have_text('Die fünf Minuten beginnen nach der Anleitung im Headset.')
    page.clock.run_for(35000)
    expect(heading).to_have_text('Anleitung zur Probe-Runde')
    progress = page.locator('#quest-progress')
    expect(progress).to_have_js_property('max', 300000)
    expect(progress).to_have_js_property('value', 0)
    assert mutations(page)['session'] == ['arm', 'start']
    page.clock.run_for(12000)
    expect(heading).to_have_text('Sitzung läuft')
    expect(progress).to_have_js_property('value', 0)
    c.wait_state(page, clock=4000)
    expect(progress).to_have_js_property('value', 4000)
    assert mutations(page)['session'] == ['arm', 'start']
    page.context.close()


def optional_participant_navigation(review):
    page = review.page(initial(c, {'participantArm': False, 'scenarios': None}))
    expect(page.locator('#operator-connect-next')).to_be_enabled()
    page.locator('#operator-connect-next').click()
    expect(page.locator('#input-screen')).to_be_visible()
    page.locator('#input-screen [data-operator-back]').press('Enter')
    expect(page.locator('#quest')).to_be_visible()
    expect(page.locator('#quest-title')).to_be_focused()
    sent = mutations(page)
    assert sent['session'] == sent['setup'] == []
    page.context.close()


def completion_status_gap_keeps_recordings_visible(review):
    settings = {'phase': 'RUNNING', 'host': 'vr-session', 'duration_ms': 900000, 'clock': 898760}
    page = review.page(initial(c, settings), suffix='#session')
    try:
        expect(page.locator('#running-screen')).to_be_visible()
        before = mutations(page)
        page.evaluate("Object.assign(reviewQuest, {statusDrop:true, joinDelay:18000})")
        page.clock.run_for(3600)
        expect(page.locator('#recordings-screen')).to_be_visible(timeout=1200)
        expect(page.locator('#quest')).to_be_hidden()
        expect(page.locator('#quest-status')).to_contain_text('Aufnahmestatus')
        expect(page.locator('#quest-export-status')).to_be_hidden()
        expect(page.locator('#recordings-open')).to_be_disabled()
        page.locator('#recordings-open').dispatch_event('click')
        assert page.evaluate("location.hash") == '#session'
        expect(page.locator('#operator-next-session')).to_be_disabled()

        # The phone may display a closeout view but only native IDLE confirms
        # the recording is over. The original command journal stays untouched.
        page.evaluate("Object.assign(reviewQuest, {phase:'IDLE', clock:0, statusDrop:false, joinDelay:0})")
        page.clock.run_for(20000)
        expect(page.locator('#recordings-screen')).to_be_visible()
        assert page.evaluate("location.hash") == '#recordings'
        assert mutations(page) == before
    finally:
        page.context.close()


def closeout_recovery_expires_without_changing_ordinary_connection_loss(review):
    for elapsed in (37400, 898760):
        settings = {'phase': 'RUNNING', 'host': 'vr-session', 'duration_ms': 900000, 'clock': elapsed}
        page = review.page(initial(c, settings), suffix='#session')
        try:
            expect(page.locator('#running-screen')).to_be_visible()
            page.evaluate("Object.assign(reviewQuest, {statusDrop:true, joinDelay:60000})")
            page.clock.run_for(3600)
            if elapsed == 898760:
                expect(page.locator('#recordings-screen')).to_be_visible()
                expect(page.locator('#operator-next-session')).to_be_disabled()
                page.clock.run_for(29000)
            expect(page.locator('#quest')).to_be_visible()
            expect(page.locator('#recordings-screen')).to_be_hidden()
        finally:
            page.context.close()


def calibration_panel_order_survives_method_reentry(review):
    page = review.page(initial(c))
    try:
        c.enter_participant(page)
        page.locator('#preview-calibrate').click()
        page.wait_for_function("reviewQuest.setup.calibration === 'collecting'")
        expect(page.locator('#calibration-screen')).to_be_visible()

        def panel_order():
            return page.locator('#calibration-screen .operator-body').evaluate(
                "body => [...body.children].map(child => child.id).filter(id => "
                "['operator-common','operator-live','operator-input-status'].includes(id))")

        original = panel_order()
        assert original == ['operator-common', 'operator-live', 'operator-input-status'], original
        page.locator('#calibration-change-input').click()
        expect(page.locator('#input-screen')).to_be_visible()
        page.evaluate("location.hash = '#calibration'")
        expect(page.locator('#calibration-screen')).to_be_visible()
        assert panel_order() == original, panel_order()
    finally:
        page.context.close()


def personal_volume_preparation(review):
    page = review.page(initial(c, {'scenarios': ['haupt-15']}, {
        'polar_modes': ['calibrated', 'volume'], 'calibration_capture': 1,
    }), viewport={'width': 390, 'height': 844})
    try:
        c.enter_participant(page)
        volume = page.locator('#preview-polar-volume')
        expect(volume).to_be_visible()
        expect(page.locator('#preview-polar-fixed')).to_be_hidden()
        expect(page.locator('#preview-polar-adaptive')).to_be_hidden()
        volume.click()
        expect(page.locator('#calibration-screen')).to_be_visible()
        page.wait_for_function("['waiting','collecting'].includes(reviewQuest.setup.calibration)")
        first = page.evaluate("reviewSetupCommands.at(-1)")
        assert first['op'] == 'polar-volume' and len(first['preparation_id']) == 36
        c.wait_setup(page, calibration='failed', failure='signal_gap')
        page.locator('#preview-retry').click()
        page.wait_for_function("reviewSetupCommands.length === 2")
        retry = page.evaluate("reviewSetupCommands.at(-1)")
        assert retry['op'] == 'polar-volume' and retry['preparation_id'] == first['preparation_id']
        c.wait_setup(page, stage='testing', calibration='ready', input='polar-volume',
                     input_volume01=.55, ready=True, failure='', completed_frames=1411200)
        expect(page.locator('#operator-confirmed-input')).to_contain_text('Polar Volume')
        page.locator('#preview-continue').click()
        c.review_battery(page)
        page.locator('#quest-play').click()
        page.wait_for_function("reviewQuest.commands.includes('arm')")
        assert page.evaluate("reviewQuest.armedPreparationId") == first['preparation_id']
    finally:
        page.context.close()


def calibration_audio_stall(review):
    page = review.page(initial(c), viewport={'width': 320, 'height': 640})
    try:
        c.enter_participant(page)
        page.locator('#preview-calibrate').click()
        expect(page.locator('#calibration-screen')).to_be_visible()
        before = mutations(page)
        clock = {'generation': 2, 'platform_generation': 2, 'actual_playing': True,
                 'intent_playing': True, 'raw_head': 0, 'timestamp_available': False,
                 'poll_age_ns': 8000000}
        c.wait_setup(page, stage='waiting', calibration='failed', failure='clock_stalled',
                     playing=True, clock_valid=False, audio_generation=2, audio_clock=clock,
                     audio_recovery_available=True)
        detail = page.locator('#preview-detail')
        expect(detail).to_be_visible()
        expect(detail).to_contain_text('Starte das Quest-Headset neu')
        expect(detail).to_contain_text('Die Audio-Wiedergabe der Quest steht bei 0.')
        c.wait_setup(page, audio_recovery_available=False)
        expect(page.locator('#preview-recovery-detail')).to_contain_text('Starte das Quest-Headset neu')
        expect(page.locator('#preview-recovery')).to_be_visible()
        expect(detail).to_be_hidden()
        c.wait_setup(page, audio_recovery_available=True, audio_clock={**clock, 'raw_head': 512})
        expect(detail).to_contain_text('Die Atemschleife lief nicht weiter')
        assert mutations(page) == before
    finally:
        page.context.close()


def completed_session_history(review):
    page = review.page(initial(c, {'scenarios': ['haupt-15'], 'startDelayMs': 2400}))
    c.enter_participant(page)
    page.locator('#preview-controller').click()
    page.locator('#preview-continue').click()
    c.review_battery(page)
    play = page.locator('#quest-play')
    play.click()
    page.wait_for_function('reviewQuest.pending && reviewQuest.commands.at(-1) === "start"')
    expect(play).to_be_disabled()
    assert mutations(page)['session'] == ['arm', 'start']
    page.clock.run_for(3200)
    expect(play).to_have_text('Pause')
    expect(page.locator('#quest-clock')).to_have_text('00:37')
    expect(page.locator('#quest-progress')).to_be_hidden()
    play.click()
    expect(play).to_have_text('Fortsetzen')
    play.click()
    expect(play).to_have_text('Pause')
    page.locator('#quest-finish').click()
    expect(page.locator('#recordings-screen')).to_be_visible()
    page.locator('a[href="#daten"]').click()
    expect(page.locator('#data-title')).to_be_focused()
    page.locator('#daten .back').click()
    expect(page.locator('#recordings-screen')).to_be_visible()
    expect(page.locator('#recordings-screen-title')).to_be_focused()
    assert mutations(page)['session'] == ['arm', 'start', 'pause', 'resume', 'save-next']
    assert page.locator('#patient-surname').input_value() == 'Tes'
    before_next = mutations(page)
    next_session = page.locator('#operator-next-session')
    c.wait_state(page, pending=True)
    expect(page.locator('#recovery-screen')).to_be_visible()
    expect(next_session).to_be_disabled()
    next_session.dispatch_event('click')
    expect(page.locator('#patient-surname')).to_have_value('Tes')
    assert mutations(page) == before_next
    c.wait_state(page, pending=False)
    c.wait_setup(page, stage='idle', input='automatic', playing=False, ready=False, parked=True)
    next_session.click()
    expect(page.locator('#participant-screen')).to_be_visible()
    expect(page.locator('#patient-surname')).to_have_value('')
    expect(page.locator('#patient-age')).to_have_value('')
    expect(page.locator('#patient-gender')).to_have_value('')
    expect(page.locator('#patient-id')).to_have_text('—')
    assert mutations(page) == before_next
    c.enter_participant(page)
    expect(page.locator('#input-screen')).to_be_visible()
    page.locator('#preview-controller').click()
    expect(page.locator('#breathing-preview')).to_be_visible()
    page.locator('#preview-continue').click()
    c.review_battery(page)
    assert mutations(page)['session'] == before_next['session']
    page.context.close()


def stopped_preview_history(review):
    page = review.page(initial(c), suffix='#calibration')
    expect(page.locator('#participant-screen')).to_be_visible()
    c.enter_participant(page)
    for step in ('calibration', 'tracking', 'battery', 'session'):
        page.evaluate('step => { location.hash = step; }', step)
        page.clock.run_for(200)
        expect(page.locator('#input-screen')).to_be_visible()
        assert page.evaluate('location.hash') == '#input'
    assert mutations(page)['setup'] == []
    page.context.close()


def running_volume_fixture(extra=None):
    fixture = initial(c, {'phase':'RUNNING', 'host':'vr-session', 'duration_ms':900000,
                       'generation':7, 'clock':2000, **(extra or {})},
                   {'stage':'main', 'playing':True, 'audio_state':'playing',
                    'renderer':{'v':1, 'progress01':.5, 'radius_m':1.5, 'frame':1,
                                'render_generation':2, 'settings_revision':0,
                                'observed_ns':'200000000', 'age_ns':0,
                                'radius_min_m':1, 'radius_max_m':2, 'cue_progress01':.25,
                                'audio_generation':1}})
    return fixture + """
Object.defineProperty(window.reviewInitialSetup.renderer, 'cue_observed_ns', {
  enumerable: true,
  get: () => String((Math.floor(performance.now()/100)+1)*100000000-20000000)
});
"""


def session_volume(review):
    page = review.page(running_volume_fixture(), viewport={'width':320, 'height':640},
                       reduced_motion='no-preference')
    page.clock.run_for(2200)
    scene = page.locator('#session-scene')
    graph = page.locator('#session-volume-chart')
    actual = page.locator('#session-volume-ist')
    intended = page.locator('#session-volume-soll')
    caption = page.locator('#session-volume-status')
    expect(scene).to_be_visible()
    expect(graph).to_be_visible()
    expect(scene.locator('canvas')).to_have_count(0)
    expect(caption).to_have_text('Quest live')
    expect(page.locator('#session-volume-ist-value')).to_have_text('14,1 m³')
    expect(page.locator('#session-volume-soll-value')).to_have_text('8,2 m³')
    expect(page.locator('#session-volume-min')).to_have_text('4,2 m³')
    expect(page.locator('#session-volume-max')).to_have_text('33,5 m³')
    page.clock.run_for(450)
    expect(actual).to_have_attribute('d', re.compile(r' L'))
    expect(intended).to_have_attribute('d', re.compile(r' L'))

    before = mutations(page)
    page.evaluate('reviewQuest.setup.renderer.radius_m=1.75')
    page.clock.run_for(300)
    expect(page.locator('#session-volume-ist-value')).to_have_text('22,4 m³')

    c.wait_state(page, phase='PAUSED')
    expect(caption).to_have_text('Pausiert.')
    paused = actual.get_attribute('d')
    page.evaluate('reviewQuest.setup.renderer.radius_m=1.25')
    page.clock.run_for(400)
    assert actual.get_attribute('d') == paused

    c.wait_state(page, phase='RUNNING', advanceRenderer=False)
    expect(caption).to_have_text('Warte auf Quest.')
    expect(page.locator('#session-volume-ist-value')).to_have_text('—')
    page.evaluate('reviewQuest.advanceRenderer=true')
    page.clock.run_for(400)
    expect(caption).to_have_text('Quest live')
    expect(page.locator('#session-volume-ist-value')).to_have_text('8,2 m³')
    assert actual.get_attribute('d').count('M') >= 2

    page.evaluate('reviewQuest.setup.renderer.age_ns=250000000')
    c.wait_state(page, generation=8, clock=0, advanceRenderer=False)
    expect(actual).to_have_attribute('d', '')
    expect(intended).to_have_attribute('d', '')
    page.evaluate('Object.assign(reviewQuest,{clock:100,advanceRenderer:true})')
    page.clock.run_for(400)
    assert actual.get_attribute('d').count('M') == 1

    page.evaluate('delete reviewQuest.setup.renderer.cue_progress01')
    page.clock.run_for(300)
    expect(caption).to_have_text('Sollwert fehlt.')
    expect(page.locator('#session-volume-soll-value')).to_have_text('—')
    expect(page.locator('#session-volume-ist-value')).to_have_text('8,2 m³')
    assert mutations(page) == before
    c.wait_state(page, phase='SAVING')
    expect(scene).to_be_hidden()
    page.context.close()

def session_volume_layout(review):
    for width, height, font, spacing in (
        (320, 568, 100, False), (320, 640, 100, False), (390, 844, 100, False),
        (844, 390, 100, False), (1440, 900, 100, False),
        (320, 640, 200, False), (390, 844, 100, True),
    ):
        page = review.page(running_volume_fixture(), viewport={'width':width, 'height':height},
                           reduced_motion='no-preference')
        page.evaluate("size => document.documentElement.style.fontSize = size + '%'", font)
        if spacing:
            page.add_style_tag(content="*{line-height:1.5!important;letter-spacing:.12em!important;"
                                       "word-spacing:.16em!important}")
        expect(page.locator('#session-volume-status')).to_have_text('Quest live')
        page.clock.run_for(500)
        for selector in ['#session-volume-status', '#session-volume-ist-value', '#session-volume-soll-value']:
            expect(page.locator(selector)).to_have_attribute('data-text-fit', 'fit')
            expect(page.locator(selector)).to_have_attribute('data-text-measurement', 'dom' if spacing else 'pretext')
        bounds = page.evaluate('''() => {
          const root = document.documentElement;
          const scene = document.querySelector('#session-scene');
          const pause = document.querySelector('#quest-play').getBoundingClientRect();
          return {
            width: root.scrollWidth, height: root.scrollHeight,
            overflow: scene.scrollHeight > scene.clientHeight || scene.scrollWidth > scene.clientWidth,
            overlapsPause: scene.getBoundingClientRect().bottom > pause.top,
            pauseBottom: pause.bottom,
          };
        }''')
        assert bounds['width'] <= width and not bounds['overflow'] and not bounds['overlapsPause'], bounds
        if font == 100:
            assert bounds['height'] <= height and bounds['pauseBottom'] <= height, (width, height, bounds)
        page.emulate_media(reduced_motion='reduce')
        expect(page.locator('#session-volume-chart')).to_be_visible()
        expect(page.locator('#session-volume-ist-value')).to_have_text('14,1 m³')
        page.context.close()


def active_calibration_protects_participant(review):
    page = review.page(initial(c))
    try:
        c.enter_participant(page)
        page.locator('#preview-calibrate').click()
        page.wait_for_function("reviewQuest.setup.calibration === 'collecting'")
        page.clock.run_for(1200)
        before = mutations(page)
        saved_name = page.locator('#patient-surname').input_value()
        c.show_operator(page, 'recordings')
        next_session = page.locator('#operator-next-session')
        expect(next_session).to_be_disabled(timeout=1000)
        next_session.dispatch_event('click')
        expect(page.locator('#recordings-screen')).to_be_visible()
        expect(page.locator('#patient-surname')).to_have_value(saved_name)
        assert mutations(page) == before
    finally:
        page.context.close()


def run():
    init = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({c.PAIRED}));" + """
window.reviewInitialQuest={polar:'ready',access:{v:1,role:'active',active_phone:'Fixture',revision:1}};
window.reviewInitialSetup={stage:'idle',playing:false,polar_modes:['calibrated','fixed','adaptive']};
"""
    with sync_playwright() as p, tempfile.TemporaryDirectory() as output:
        browser = p.chromium.launch()
        review = c.Review(browser, Path(output))
        recovery_and_recordings(review)
        scenario_compatibility(review)
        authored_round_layout(review)
        guided_probe_instruction(review)
        optional_participant_navigation(review)
        completion_status_gap_keeps_recordings_visible(review)
        closeout_recovery_expires_without_changing_ordinary_connection_loss(review)
        calibration_panel_order_survives_method_reentry(review)
        completed_session_history(review)
        stopped_preview_history(review)
        active_calibration_protects_participant(review)
        page = review.page(init)
        page.clock.run_for(4000)
        expect(page.locator('#quest')).to_be_visible()
        expect(page.locator('#quest #quest-ecg')).to_be_visible()
        expect(page.locator('#check-bridge')).to_have_attribute('data-check', 'ok')
        expect(page.locator('#quest #drive-open')).to_be_visible()
        page.locator('#drive-open').click()
        expect(page.locator('#drive-dialog #quest-drive-form')).to_be_visible()
        expect(page.locator('#quest-drive-connect')).to_be_enabled()
        page.keyboard.press('Escape')
        expect(page.locator('#drive-dialog')).to_be_hidden()
        expect(page.locator('#drive-open')).to_be_focused()
        c.wait_state(page, polar='off')
        polar_check = page.locator('#check-polar')
        expect(polar_check).to_have_attribute('data-check', 'unconfirmed')
        polar_check.click()
        expect(page.locator('#connection-diagnostic-title')).to_have_text('Polar H10')
        expect(page.locator('#connection-diagnostic-message')).to_contain_text('Polar H10')
        page.keyboard.press('Escape')
        expect(polar_check).to_be_focused()
        c.wait_state(page, polar='ready')
        before_connect = mutations(page)
        page.locator('#operator-connect-next').click()
        assert before_connect == mutations(page)
        expect(page.locator('#participant-screen')).to_be_visible()
        page.locator('#patient-surname').fill('Tes')
        page.locator('#patient-age').fill('30')
        page.locator('#patient-gender').select_option('prefer-not-to-say')
        page.locator('#patient-next').click()
        expect(page.locator('#input-screen')).to_be_visible()
        before = mutations(page)
        page.locator('[data-operator-go="recordings"]:visible').click()
        page.locator('a[href="#daten"]').click()
        expect(page.locator('#daten')).to_be_visible()
        page.go_back()
        page.clock.run_for(100)
        expect(page.locator('#recordings-screen')).to_be_visible()
        assert page.locator('.screen:visible').count() == 1
        page.locator('[data-operator-return]:visible').click()
        expect(page.locator('#input-screen')).to_be_visible()
        page.locator('[data-operator-go="recordings"]:visible').click()
        page.locator('a[href="#daten"]').click()
        page.locator('#daten .back').click()
        page.locator('[data-operator-return]:visible').click()
        expect(page.locator('#input-screen')).to_be_visible()
        c.show_operator(page, 'settings')
        page.locator('[data-operator-return]:visible').click()
        expect(page.locator('#input-screen')).to_be_visible()
        page.locator('[data-operator-go="recordings"]:visible').click()
        page.locator('[data-operator-go="settings"]:visible').click()
        page.go_back()
        page.clock.run_for(100)
        expect(page.locator('#recordings-screen')).to_be_visible()
        page.locator('[data-operator-return]:visible').click()
        expect(page.locator('#input-screen')).to_be_visible()
        assert before == mutations(page)
        assert page.locator('#patient-surname').input_value() == 'Tes'
        c.wait_state(page, phase='RUNNING', host='vr-session', clock=18000, liveRecording=True)
        expect(page.locator('#running-screen #quest-export-status')).to_contain_text('Live-Abgleich unterbrochen')
        expect(page.locator('#quest-progress')).to_be_hidden()
        c.wait_state(page, duration_ms=900000)
        expect(page.locator('#quest-progress')).to_have_js_property('value', 18000)
        expect(page.locator('#quest-progress')).to_have_js_property('max', 900000)
        c.wait_setup(page, main_start_frame=705600, main_started=True, stage='main')
        c.wait_state(page, duration_ms=None, scenarios=['haupt-15'])
        expect(page.locator('#quest-progress')).to_have_js_property('value', 18000)
        expect(page.locator('#quest-progress')).to_have_js_property('max', 900000)
        expect(page.locator('#quest-clock')).to_have_text('00:18 / 15:00')
        c.wait_state(page, phase='PAUSED')
        page.clock.run_for(2400)
        expect(page.locator('#quest-progress')).to_have_js_property('value', 18000)
        expect(page.locator('#quest-clock')).to_have_text('00:18 / 15:00')
        c.wait_state(page, phase='RUNNING', clock=21000)
        expect(page.locator('#quest-progress')).to_have_js_property('value', 21000)
        expect(page.locator('#running-screen')).to_be_visible()
        expect(page.locator('#running-screen #quest-data')).to_be_visible()
        expect(page.locator('#quest-play')).to_be_visible()
        expect(page.locator('#quest-finish')).to_be_visible()
        page.locator('[data-operator-go="settings"]:visible').click()
        expect(page.locator('#settings-screen')).to_be_visible()
        expect(page.locator('#settings-screen #quest-play')).to_be_visible()
        expect(page.locator('#settings-screen #quest-finish')).to_be_visible()
        expect(page.locator('#settings-screen #quest-export-status')).to_be_visible()
        page.locator('#settings-connection summary').click()
        expect(page.locator('#quest-control-release')).to_be_visible()
        page.keyboard.press('Escape')
        page.locator('[data-operator-return]:visible').click()
        expect(page.locator('#running-screen')).to_be_visible()
        for phase,heading in (('PAUSED','Sitzung pausiert'),('FINALIZING','Sitzung wird beendet'),('SAVING','Aufnahme wird gespeichert')):
            c.wait_state(page,phase=phase)
            expect(page.locator('#running-screen-title')).to_have_text(heading)
            expect(page.locator('#running-screen #quest-status')).to_be_visible()
            expect(page.locator('#running-screen #quest-data')).to_be_visible()
            expect(page.locator('#quest-play')).to_be_visible()
            expect(page.locator('#quest-finish')).to_be_visible()
        assert before == mutations(page)
        compact_setup(review)
        calibration_instruction(review)
        personal_volume_preparation(review)
        calibration_audio_stall(review)
        session_volume(review)
        session_volume_layout(review)
        assert not review.errors, review.errors
        browser.close()
    print('PASS real-app scenario Arm, calibration narration and cancellation, receipt recovery, recording status, Drive dialog, draft and history')


if __name__ == '__main__':
    run()
