"""Focused real-app local-route regressions; no hardware or account requests."""
from pathlib import Path
import importlib.util
import tempfile
from playwright.sync_api import sync_playwright, expect
from layout_render_fixtures import mutations

spec = importlib.util.spec_from_file_location('capture_phone', Path(__file__).with_name('capture-phone-screens.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def run():
    init = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({c.PAIRED}));" + """
window.reviewInitialQuest={polar:'ready',access:{v:1,role:'active',active_phone:'Fixture',revision:1}};
window.reviewInitialSetup={stage:'idle',playing:false,polar_modes:['calibrated','fixed','adaptive']};
"""
    with sync_playwright() as p, tempfile.TemporaryDirectory() as output:
        browser = p.chromium.launch()
        review = c.Review(browser, Path(output))
        page = review.page(init)
        page.clock.run_for(1500)
        expect(page.locator('#participant-screen')).to_be_visible()
        page.locator('#patient-surname').fill('Tes')
        page.locator('#patient-age').fill('30')
        page.get_by_role('radio', name='Keine Angabe').check()
        page.locator('#patient-next').click()
        expect(page.locator('#signals-screen')).to_be_visible()
        page.locator('[data-operator-go="input"]:visible').click()
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
        page.locator('[data-operator-go="settings"]:visible').click()
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
        c.wait_state(page, phase='RUNNING', host='vr-session', clock=18000)
        expect(page.locator('#running-screen')).to_be_visible()
        expect(page.locator('#running-screen #quest-data')).to_be_visible()
        expect(page.locator('#quest-play')).to_be_visible()
        expect(page.locator('#quest-finish')).to_be_visible()
        page.locator('[data-operator-go="settings"]:visible').click()
        expect(page.locator('#settings-screen')).to_be_visible()
        expect(page.locator('#settings-screen #quest-play')).to_be_visible()
        expect(page.locator('#settings-screen #quest-finish')).to_be_visible()
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
        assert not review.errors, review.errors
        browser.close()
    print('PASS real-app legacy history, local return, draft persistence and zero-command navigation')


if __name__ == '__main__':
    run()
