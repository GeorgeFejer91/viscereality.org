"""Focused real-app local-route regressions; no hardware or account requests."""
from pathlib import Path
import importlib.util
import tempfile
from playwright.sync_api import sync_playwright, expect
from layout_render_fixtures import CURRENT_APK, initial, mutations

spec = importlib.util.spec_from_file_location('capture_phone', Path(__file__).with_name('capture-phone-screens.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def compact_setup(review):
    for width, height in ((320, 640), (390, 844), (844, 390)):
        page = review.page(initial(c, {'apk': CURRENT_APK}), viewport={'width': width, 'height': height})
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
                page: [root.scrollWidth, root.scrollHeight],
                viewport: [innerWidth, innerHeight],
                dialogFits: !open || (open.scrollHeight <= open.clientHeight && open.scrollWidth <= open.clientWidth
                  && open.getBoundingClientRect().top >= document.querySelector('#operator-toolbar').getBoundingClientRect().bottom),
                textFits: ![...(open || scope).querySelectorAll('[data-text-fit="reflow"]')]
                  .some(node => node.checkVisibility())
              };
            }''')
            assert result['measured'] == 'ready' and result['page'] == result['viewport'], result
            assert result['dialogFits'] and result['textFits'], result

        assert_fits()
        header = page.get_by_role('banner', name='Geräte und Aktionen')
        settings = header.get_by_role('button', name='Einstellungen')
        recordings = header.get_by_role('button', name='Aufnahmen')
        expect(settings).to_be_visible(timeout=1500)
        expect(recordings).to_be_visible()
        controls = header.locator('button:visible').evaluate_all(
            '(buttons) => buttons.map(button => button.getBoundingClientRect().y)')
        assert len(controls) >= 6 and max(controls) - min(controls) < 2, controls
        settings.focus()
        page.keyboard.press('Enter')
        expect(page.locator('#settings-screen-title')).to_be_focused()
        expect(recordings).to_be_visible()
        page.locator('#settings-screen [data-operator-return]').click()
        expect(settings).to_be_focused()
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
        (legacy, ['track-ready', 'track-ready'], [], None),
        (None, ['track-ready', 'track-ready'], [], None),
        (['probe-5'], ['missing', 'track-ready'], ['probe-5'], None),
    ):
        page = review.page(initial(c, {'scenarios': offered, 'tracks': tracks}),
                           viewport={'width': 320, 'height': 568})
        c.enter_participant(page)
        page.locator('#preview-controller').click()
        page.locator('#preview-continue').click()
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
        else:
            expect(page.locator('#quest-play')).to_be_disabled()
            expect(page.locator('#quest-scenario-unavailable')).to_be_visible()
            assert mutations(page)['session'] == []
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


def completed_session_history(review):
    page = review.page(initial(c, {'scenarios': ['haupt-15'], 'startDelayMs': 2400}))
    c.enter_participant(page)
    page.locator('#preview-controller').click()
    page.locator('#preview-continue').click()
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
        optional_participant_navigation(review)
        completed_session_history(review)
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
        expect(page.locator('#running-screen')).to_be_visible()
        expect(page.locator('#running-screen #quest-data')).to_be_visible()
        expect(page.locator('#quest-play')).to_be_visible()
        expect(page.locator('#quest-finish')).to_be_visible()
        page.locator('[data-operator-go="settings"]:visible').click()
        expect(page.locator('#settings-screen')).to_be_visible()
        expect(page.locator('#settings-screen #quest-play')).to_be_visible()
        expect(page.locator('#settings-screen #quest-finish')).to_be_visible()
        expect(page.locator('#settings-screen #quest-export-status')).to_be_visible()
        expect(page.locator('#quest-control-release')).to_be_visible()
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
        assert not review.errors, review.errors
        browser.close()
    print('PASS real-app scenario Arm, receipt recovery, recording status, Drive dialog, draft and history')


if __name__ == '__main__':
    run()
