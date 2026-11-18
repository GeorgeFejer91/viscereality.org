"""Production PWA buttons with synthetic radio/native transport, no readiness bypass."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from playwright.sync_api import expect, sync_playwright
from layout_render_fixtures import module

RTC = """
import {pairPhone} from './quest-pairing.js';
export async function openQuestWebRtc(discovery,enrollment,signal,pairing) {
  window.reviewRtcCalls=(window.reviewRtcCalls??0)+1;
  if(window.reviewRtcUnsupported)throw Error('Older native RTC transport');
  if(enrollment&&window.reviewRejectSavedRtc)throw Object.assign(Error('Revoked saved member'),{status:403});
  if(!enrollment) {
    try {
      const ticket=await pairPhone({owner:discovery.owner,signal,onCode:pairing.onCode});
      enrollment={controller:ticket.controller,key:ticket.key};
      await pairing.onEnrollment(enrollment);
      window.reviewSavedBeforeRtcReturn=Boolean(JSON.parse(localStorage.getItem('atmungsraum-quest-pair'))?.enrollment);
    }catch(error){error.pairingAttempted=true;throw error;}
  }
  return {enrollment,transport:{close(){window.reviewRtcClosed=true;}}};
}
"""


def run():
    fixture = module()
    original = "return { owner, endpoints: ['synthetic'], ...(window.reviewPairingEnabled ? { pairing: 1 } : {}) };"
    assert fixture.GATEWAY.count(original) == 1
    fixture.GATEWAY = fixture.GATEWAY.replace(original,
        "return {owner,endpoints:window.reviewLan?['synthetic']:[],pairing:1,rtc:{r:'a'.repeat(32),p:'b'.repeat(64)}};")
    with TemporaryDirectory(prefix='atmungsraum-button-pairing-') as output, sync_playwright() as p:
        browser = p.chromium.launch()
        review = fixture.Review(browser, Path(output))

        def page(extra=''):
            result = review.page(fixture.NEW_PHONE + extra)
            result.context.route('**/quest-webrtc.js*', lambda route: route.fulfill(content_type='text/javascript', body=RTC))
            return result

        def begin(result, previous=None):
            result.locator('#quest-find').click()
            expect(result.locator('#quest-pairing')).to_be_visible()
            expect(result.locator('#quest-pairing input, #quest-pairing textarea, #quest-pairing select')).to_have_count(0)
            assert result.evaluate("JSON.parse(localStorage.getItem('atmungsraum-quest-pair'))") == previous

        for order in ['phone-first', 'headset-first']:
            current = page()
            begin(current)
            if order == 'headset-first':
                current.evaluate("reviewPairing.finish('approved')")
                assert current.evaluate("localStorage.getItem('atmungsraum-quest-pair')") is None
            current.locator('#quest-pairing-confirm').click()
            if order == 'phone-first':
                assert current.evaluate("localStorage.getItem('atmungsraum-quest-pair')") is None
                current.evaluate("reviewPairing.finish('approved')")
            expect(current.locator('#quest-pairing')).to_be_hidden()
            current.clock.run_for(2100)
            expect(current.locator('#check-control')).to_have_attribute('data-check', 'ok')
            assert current.evaluate('reviewRtcCalls') == 1
            assert current.evaluate('reviewSavedBeforeRtcReturn')
            assert current.evaluate('reviewPairingCalls') == ['begin', 'phone-confirmed', 'approved']
            current.context.close()

        for outcome in ['reject', 'cancel']:
            current = page('window.reviewLan=true;')
            begin(current)
            if outcome == 'reject':
                current.evaluate("reviewPairing.finish('rejected')")
            else:
                current.locator('#quest-pairing-cancel').click()
            expect(current.locator('#quest-pairing')).to_be_hidden()
            current.clock.run_for(1000)
            assert current.evaluate('reviewPairingCalls.length') == 2
            assert current.evaluate('reviewRtcCalls') == 1
            assert current.evaluate("localStorage.getItem('atmungsraum-quest-pair')") is None
            current.context.close()

        current = page('window.reviewLan=true;window.reviewRtcUnsupported=true;')
        begin(current)
        current.evaluate("reviewPairing.finish('approved')")
        current.locator('#quest-pairing-confirm').click()
        expect(current.locator('#check-control')).to_have_attribute('data-check', 'ok')
        assert current.evaluate('reviewPairingCalls.filter(x=>x==="begin").length') == 1
        current.context.close()

        current = page("window.reviewInitialQuest.failJoinAfterPairing=true;")
        begin(current)
        current.evaluate("reviewPairing.finish('approved')")
        current.locator('#quest-pairing-confirm').click()
        expect(current.locator('#quest-pairing')).to_be_hidden()
        assert current.evaluate("Boolean(JSON.parse(localStorage.getItem('atmungsraum-quest-pair'))?.enrollment)")
        assert current.evaluate('reviewPairingCalls.filter(x=>x==="begin").length') == 1
        current.context.close()

        for same_device in (False, True):
            previous = {
                'device': ('d' if same_device else 'e') * 32, 'serial': '',
                'identity_kind': 'app-installation', 'endpoint': 'unreachable',
                'enrollment': {'controller': 'c' * 32, 'key': 'e' * 64},
            }
            current = page(
                "localStorage.setItem('atmungsraum-quest-pair',JSON.stringify("
                + json.dumps(previous) + "));window.reviewRejectSavedRtc=true;"
            )
            begin(current, previous)
            current.evaluate("reviewPairing.finish('approved')")
            current.locator('#quest-pairing-confirm').click()
            current.clock.run_for(2100)
            expect(current.locator('#check-control')).to_have_attribute('data-check', 'ok')
            saved = current.evaluate("JSON.parse(localStorage.getItem('atmungsraum-quest-pair'))")
            assert saved['device'] == 'd' * 32
            assert 'endpoint' not in saved, saved
            assert saved['enrollment'] != previous['enrollment']
            assert current.evaluate('reviewRtcCalls') == (2 if same_device else 1)
            assert current.evaluate('reviewPairingCalls') == ['begin', 'phone-confirmed', 'approved']
            current.context.close()
        assert not review.errors, review.errors
        browser.close()
    print('PASS mounted production UI: 8 cases covering both button orders, zero code inputs, reject/cancel, '
          'legacy LAN fallback, lost status, headset switching and revoked membership')


if __name__ == '__main__':
    run()
