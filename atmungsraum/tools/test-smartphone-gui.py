"""Render the production PWA under the playground's strict no-scroll rules."""
import sys
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

from playwright.sync_api import expect, sync_playwright

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from layout_render_fixtures import capture_layout, go, initial, module

AUDIT = (TOOLS.parent / "for-ai" / "scripts" / "gui-layout.js").read_text(encoding="utf-8")
CASES = (
    ("320", {"width": 320, "height": 640}, ""),
    ("389", {"width": 389, "height": 640}, ""),
    ("390-short", {"width": 390, "height": 640}, ""),
    ("390", {"width": 390, "height": 844}, ""),
    ("landscape", {"width": 844, "height": 390}, ""),
    ("enlarged", {"width": 390, "height": 844}, "200%"),
    ("spacing", {"width": 390, "height": 844}, ""),
    ("desktop", {"width": 1440, "height": 900}, ""),
)
SPACING = "*{line-height:1.5!important;letter-spacing:.12em!important;word-spacing:.16em!important}p{margin-bottom:2em!important}"
c = module()
HOLD_RESIZE = """const RealResizeObserver = window.ResizeObserver;
    window.ResizeObserver = class extends RealResizeObserver {
      constructor(callback) {
        super((...args) => { if (!window.reviewHoldResize) callback(...args); });
      }
    };"""


class GuiReview(c.Review):
    def __init__(self, browser, output):
        super().__init__(browser, output)
        self.samples = {}

    def page(self, *args, **kwargs):
        page = super().page(*args, **kwargs)
        if self.text_spacing:
            page.add_style_tag(content=SPACING)
        return page

    def shot(self, page, slug, title, expected, section):
        expect(expected).to_be_visible()
        expect(page.locator("html")).to_have_attribute("data-measurement", "ready")
        page.clock.run_for(250)
        self.samples[slug] = page.evaluate(AUDIT, page.viewport_size)


def capture_dialogs(review, tag, viewport, font):
    kiosk_status = {"v": 1, "g": 2, "requested": True, "effective": True, "state": "active",
                    "service": "connected", "blocked": "", "detail": "Quest-App und Kiosk-Dienst verbunden.",
                    "retry_available": False}
    page = review.page(initial(c, {"kiosk": kiosk_status}), viewport=viewport)
    page.evaluate("size => document.documentElement.style.fontSize = size", font)
    page.clock.run_for(4000)
    for check in ("bluetooth", "bridge", "control", "polar"):
        expect(page.locator("#check-" + check)).to_have_attribute("data-check", "ok")
    for button, dialog in (("toolbar-actions", "toolbar-dialog"),
                           ("install-app", "install-dialog"), ("drive-open", "drive-dialog")):
        trigger = page.locator("#" + button)
        if not trigger.is_visible() and button == "toolbar-actions":
            continue
        trigger.focus()
        page.keyboard.press("Enter")
        review.shot(page, f"{tag}-{dialog}", dialog, page.locator("#" + dialog), "Dialogs")
        page.keyboard.press("Escape")
        expect(trigger).to_be_focused()
    go(page, "settings")
    kiosk = page.locator("#quest-settings > summary")
    kiosk.click()
    review.shot(page, f"{tag}-kiosk", "Kiosk", page.locator("#kiosk-dialog"), "Dialogs")
    page.locator("#kiosk-diagnostics > summary").click()
    review.shot(page, f"{tag}-kiosk-diagnostics", "Kiosk diagnostics", page.locator("#kiosk-detail"), "Dialogs")
    page.keyboard.press("Escape")
    expect(kiosk).to_be_focused()
    page.locator("#settings-screen [data-operator-return]").click()
    page.locator("#operator-connect-next").click()
    field = page.locator("#patient-surname")
    field.focus()
    page.keyboard.type("Tes")
    page.set_viewport_size({"width": viewport["width"], "height": round(viewport["height"] * .7)})
    review.shot(page, f"{tag}-focused-participant", "Focused participant", field, "Keyboard")
    expect(field).to_be_focused()
    expect(field).to_have_value("Tes")
    page.context.close()


def capture_recordings(review, tag, viewport, font):
    page = review.page(initial(c), viewport=viewport)
    page.evaluate("size => document.documentElement.style.fontSize = size", font)
    records = page.evaluate("""() => new Promise((resolve, reject) => {
      const request = indexedDB.open('atmungsraum-session-data', 1);
      request.onupgradeneeded = () => request.result.createObjectStore('exports', { keyPath: 'filename' });
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        const transaction = db.transaction('exports', 'readwrite');
        const store = transaction.objectStore('exports');
        const rows = Array.from({length: 60}, (_, index) => ({
          filename: `Tes_30_N_2026-11-17_haupt-15_r${index + 1}.csv`,
          csv: `row\n${index}\n`, created_at: new Date(index * 1000).toISOString(),
        }));
        for (const row of rows) store.put(row);
        transaction.oncomplete = () => { db.close(); resolve(rows); };
        transaction.onerror = () => { db.close(); reject(transaction.error); };
      };
    })""")
    go(page, "recordings")
    page.locator('a[href="#daten"]').click()
    selector = page.locator("#browser-archives")
    expect(selector.locator("option")).to_have_count(len(records))
    name = page.locator("#browser-archive-name")
    expect(name).to_have_text(records[-1]["filename"])
    selector.select_option(value=str(len(records) - 1))
    expect(name).to_have_text(records[0]["filename"])
    download = page.locator("#browser-export-download")
    review.shot(page, f"{tag}-full-recording-list", "Browser recordings", download, "Recordings")
    with page.expect_download() as result:
        download.click()
    assert result.value.suggested_filename == records[0]["filename"]
    assert result.value.path().read_bytes() == records[0]["csv"].encode()
    page.context.close()


def capture_diagnostics(review, tag, viewport, font):
    failed_join = initial(c) + "window.reviewJoin='fail';"
    failures = (
        ("bluetooth", failed_join + "window.reviewDiscoveryState='bluetooth-off';"),
        ("bridge", failed_join + "window.reviewDiscoveryState='ready';"),
        ("control", initial(c, {"access": {"v": 1, "role": "waiting", "active_phone": "Zweites Smartphone", "revision": 1}})),
        ("polar", initial(c, {"polar": "stream-wait", "ecg": False, "acc": False})),
    )
    for check, fixture in failures:
        page = review.page(fixture, viewport=viewport)
        page.evaluate("size => document.documentElement.style.fontSize = size", font)
        page.wait_for_function("Boolean(document.documentElement.dataset.operatorStep)")
        page.clock.run_for(8000)
        trigger = page.locator("#check-" + check)
        expect(trigger).to_have_attribute("data-check", "unconfirmed")
        trigger.focus()
        page.keyboard.press("Enter")
        dialog = page.locator("#connection-diagnostic")
        expect(page.locator("dialog:visible")).to_have_count(1)
        review.shot(page, f"{tag}-diagnose-{check}", check, dialog, "Diagnostics")
        page.keyboard.press("Escape")
        expect(trigger).to_be_focused()
        page.context.close()


def capture_apk_detail_change(review):
    # Keep the real ResizeObserver installed and delay its callbacks while
    # switching topics, so the visibility change must schedule measurement.
    page = review.page(initial(c) + HOLD_RESIZE, viewport={"width": 390, "height": 844})
    page.clock.run_for(1500)
    go(page, "update")
    page.locator("#apk-identity summary").click()
    page.clock.run_for(250)
    expect(page.locator("#apk-update-status")).to_have_attribute("data-text-fit", "fit")
    identity = page.locator("#apk-full-identity")
    assert identity.get_attribute("data-text-fit") is None
    topic = page.locator("#apk-detail-view")
    topic.focus()
    page.evaluate("window.reviewHoldResize = true")
    topic.select_option("identity")
    page.clock.run_for(50)
    expect(identity).to_be_visible()
    expect(topic).to_be_focused()
    assert identity.get_attribute("data-text-fit") == "fit", "Visible APK identity must be measured before a resize callback"
    review.samples["apk-details-topic-change"] = page.evaluate(AUDIT, page.viewport_size)
    page.context.close()


def capture_connect_visibility_change(review):
    page = review.page(initial(c) + HOLD_RESIZE + "window.reviewJoin='fail';",
                       viewport={"width": 320, "height": 640})
    page.clock.run_for(2000)
    expect(page.locator("html")).to_have_attribute("data-measurement", "ready")
    summary = page.locator("#operator-connection-summary")
    expect(summary).to_be_hidden()
    page.clock.pause_at(page.evaluate("Date.now()") + 1000)
    page.clock.run_for(250)
    assert summary.get_attribute("data-text-fit") is None
    page.evaluate("""() => {
      window.reviewHoldResize = true;
      document.getElementById('operator-connection-summary').hidden = false;
    }""")
    page.clock.run_for(50)
    expect(summary).to_be_visible()
    assert summary.get_attribute("data-text-fit") == "fit", "Visible connection summary needs a fit check before resize"
    review.samples["connect-summary-visibility-change"] = page.evaluate(AUDIT, page.viewport_size)
    page.context.close()


def check_stretch(samples):
    findings = []
    for name, intermediate in samples.items():
        if not name.startswith("layout-390-short-"):
            continue
        step = name.removeprefix("layout-390-short-")
        narrow, tall = samples.get("layout-320-" + step), samples.get("layout-390-" + step)
        if not narrow or not tall or any(sample["problems"] or "width" not in sample
                                         for sample in (narrow, intermediate, tall)):
            continue
        if intermediate["width"] <= narrow["width"] + 1 or tall["height"] <= intermediate["height"] + 1:
            findings.append(f"{step}: body must stretch with added width and height")
        if intermediate["rows"] > 1 and tall["contents"] == intermediate["contents"] and tall["gap"] <= intermediate["gap"] + 1:
            findings.append(f"{step}: distribute added height between groups")
    return findings


def verify():
    failures = []
    with TemporaryDirectory(prefix="atmungsraum-gui-") as output, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        review = GuiReview(browser, Path(output))
        for tag, viewport, font in CASES:
            review.text_spacing = tag == "spacing"
            count_before = len(review.samples)
            capture_layout(review, ((tag, viewport, font),))
            capture_dialogs(review, tag, viewport, font)
            capture_diagnostics(review, tag, viewport, font)
            capture_recordings(review, tag, viewport, font)
            print(f"Checked {tag}: {len(review.samples) - count_before} screens and dialogs", flush=True)
        review.text_spacing = False
        capture_apk_detail_change(review)
        capture_connect_visibility_change(review)
        failures.extend(review.errors)
        failures.extend(check_stretch(review.samples))
        for name, sample in review.samples.items():
            if sample["problems"]:
                failures.append(f"{name}: " + "; ".join(sample["problems"][:5]))
        count = len(review.samples)
        browser.close()
    if failures:
        print(f"FAIL smartphone GUI gate ({len(failures)} cases)")
        print("\n".join(failures))
        return 1
    run_path(str(TOOLS / "test-operator-layout.py"))["run"]()
    print(f"PASS smartphone GUI gate ({count} states)")
    return 0


if __name__ == "__main__":
    raise SystemExit(verify())
