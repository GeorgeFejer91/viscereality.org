"""Render the production PWA under the playground's strict no-scroll rules."""
import json
import sys
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

from playwright.sync_api import expect, sync_playwright

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from layout_render_fixtures import capture_companion_permission, capture_layout, go, initial, module, mutations
from release_fixtures import apk_fixture

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


QUESTIONNAIRE_RECORDINGS = """
window.reviewRecordings = async (query, owner) => {
  const id = 'utc-ns-01763460000000000000';
  const session_id = '01234567-89ab-4def-8abc-0123456789ab';
  const scenario = reviewQuest.recordingScenario ?? 'haupt-15';
  const name = 'Tes_30_N_2026-11-18_' + scenario + '_r2.csv';
  const prefix = new TextEncoder().encode('schema_version,participant_id,session_id\\r\\n4,Tes_30_N,' + session_id + '\\r\\n');
  const outcome = reviewQuest.recordingAborted ? 'aborted' : 'completed';
  const terminal = ['4','Tes_30_N',session_id,'2',scenario,'end','2026-11-18T12:00:00Z',
    '900000000','900000000','ended','ended',...Array(10).fill(''),outcome,outcome,''].join(',');
  const full = new Uint8Array([...prefix, ...new TextEncoder().encode(terminal + '\\r\\n')]);
  const sha = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
    byte => byte.toString(16).padStart(2, '0')).join('');
  const names = ['manifest.json','final.json','events.jsonl','polar-acc.jsonl','polar-ecg.jsonl',
    'polar-hr-rr.jsonl','breath.jsonl','radius.jsonl'];
  const descriptor = async bytes => ({ v:1, id, session_id, name, schema_version:4, size:bytes.length, sha256:await sha(bytes) });
  const manifest_id = 'f'.repeat(64);
  switch (query.query) {
    case 'live-describe': return { ...await descriptor(prefix), state:'live', checkpoint:1 };
    case 'list': return { v:1, state:'ready', entries:reviewQuest.recordingSaved
      ? [{id, exportable:true, participant_csv:true}] : [], next_before:null };
    case 'describe': {
      const emptyHash = await sha(new Uint8Array());
      return { state:'ready', manifest:{ id, manifest_id, schema:'rusty.quest.recording_export.v1', saved:true,
        file_set_policy:'core8-original-no-checkpoints-or-index', device:owner.device, serial:owner.serial,
        total_bytes:0, files:names.map(name => ({name, size:0, sha256:emptyHash})), participant_csv:await descriptor(full) } };
    }
    default: {
      const live = query.query === 'live-read';
      const source = live ? prefix : full;
      const chunk = source.slice(query.offset, query.offset + query.length);
      return { ...query, v:1, state:live ? 'live-chunk' : 'chunk',
        sha256:await sha(live ? chunk : source), data:btoa(String.fromCharCode(...chunk)) };
    }
  }
};
"""


def capture_questionnaire(review, tag, viewport, font, *, outcome="completed", scenario="haupt-15", live=True):
    quest = {"phase": "RUNNING", "host": "vr-session", "generation": 3, "condition": "condition-a",
             "clock": 895000, "duration_ms": 900000, "liveRecording": live,
             "recordingAborted": outcome == "aborted", "recordingScenario": scenario}
    setup = {"stage": "main", "input": "controller-state", "playing": True, "ready": True}
    enrollment = "localStorage.setItem('atmungsraum-quest-pair'"
    fixture = initial(c, quest, setup).replace(enrollment, "if (!localStorage.getItem('atmungsraum-quest-pair')) " + enrollment, 1)
    page = review.page(fixture + QUESTIONNAIRE_RECORDINGS, viewport=viewport)
    try:
        page.evaluate("font => document.documentElement.style.fontSize = font", font)
        expect(page.locator("#running-screen")).to_be_visible()
        if live:
            expect(page.locator("#quest-export-status")).to_contain_text("Laufende Quest-Aufnahme")
        c.wait_state(page, phase="SAVING")
        expect(page.locator("#questionnaire-screen")).to_be_hidden()
        c.wait_state(page, phase="IDLE", host="vr-preparation", recordingSaved=True)
        form = page.locator("#questionnaire-screen")
        if outcome != "completed" or scenario != "haupt-15":
            expect(page.locator("#quest-export-download")).to_be_visible()
            expect(form).to_be_hidden()
            expect(page.locator("#questionnaire-open")).to_be_hidden()
            return
        if not live:
            expect(form).to_be_hidden()
            page.locator("#questionnaire-open").click()
        expect(form).to_be_visible()
        if tag == "390" and live:
            # The pending form and its run survive a fresh document without the Quest.
            page.add_init_script("window.reviewJoin='fail';")
            page.reload(wait_until="load")
            expect(form).to_be_visible()
        answer = page.locator("#questionnaire-answer")
        heading = page.locator("#questionnaire-title")
        submit = page.locator("#questionnaire-next")
        submit.click()
        expect(heading).to_have_text("Frage 1 von 5")
        expect(answer).to_be_focused()
        answer.fill("   ")
        answer.press("Enter")
        expect(heading).to_have_text("Frage 1 von 5")
        review.shot(page, f"{tag}-questionnaire-first", "Question 1", answer, "Questionnaire")
        answer.fill("Antwort 1")
        answer.press("Enter")
        expect(heading).to_be_focused()
        page.locator("#questionnaire-back").click()
        expect(answer).to_have_value("Antwort 1")
        answer.press("Enter")
        for index in range(2, 6):
            expect(heading).to_have_text(f"Frage {index} von 5")
            answer.fill(f"Antwort {index}")
            if index == 5:
                answer.focus()
                if tag == "390":
                    page.set_viewport_size({**viewport, "height": 428})
                review.shot(page, f"{tag}-questionnaire-last", "Question 5", answer, "Questionnaire")
                expect(answer).to_be_focused()
                expect(answer).to_have_value("Antwort 5")
            answer.press("Enter")
        link = page.locator("#questionnaire-download")
        expect(link).to_be_visible()
        if tag == "390" and live:
            page.set_viewport_size(viewport)
        review.shot(page, f"{tag}-questionnaire-saved", "Saved answers", link, "Questionnaire")
        assert mutations(page)["session"] == []
        with page.expect_download() as download:
            link.click()
        assert download.value.suggested_filename == "Tes_30_N_2026-11-18_haupt-15_r2_questionnaire.json"
        record = json.loads(download.value.path().read_bytes())
        assert record["participant_id"] == "Tes_30_N"
        assert record["session_id"] == "01234567-89ab-4def-8abc-0123456789ab"
        assert record["recording_filename"] == "Tes_30_N_2026-11-18_haupt-15_r2.csv"
        assert [item["answer"] for item in record["answers"]] == [f"Antwort {n}" for n in range(1, 6)]
        if tag == "390" and live:
            page.reload(wait_until="load")
            expect(link).to_be_visible()
            expect(form).to_be_hidden()
    finally:
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


def capture_focused_viewport_change(review):
    hold_resize = """const holdResize = event => {
      if (window.reviewHoldViewportResize) event.stopImmediatePropagation();
    };
    window.addEventListener('resize', holdResize, true);
    window.visualViewport?.addEventListener('resize', holdResize, true);"""
    page = review.page(initial(c) + hold_resize, viewport={"width": 390, "height": 844})
    try:
        page.clock.run_for(2000)
        page.locator("#operator-connect-next").click()
        field = page.locator("#patient-surname")
        field.fill("Tes")
        field.evaluate("element => element.setSelectionRange(1, 1)")
        expect(field).to_be_focused()
        page.evaluate("window.reviewHoldViewportResize = true")
        page.set_viewport_size({"width": 390, "height": 591})
        state = page.evaluate("""() => ({
          height: innerHeight,
          measured: document.documentElement.style.getPropertyValue('--usable-height'),
          scroll: document.documentElement.scrollHeight,
        })""")
        assert state["measured"] == "844px", state
        assert state["height"] == 591 and state["scroll"] <= 591, state
        review.shot(page, "focused-participant-delayed-viewport", "Focused participant", field, "Keyboard")
        expect(field).to_be_focused()
        expect(field).to_have_value("Tes")
        assert field.evaluate("element => element.selectionStart") == 1
    finally:
        page.context.close()


def capture_participant_keyboard(review):
    apk = {**apk_fixture(), "package_name": "org.viscereality.atmungsraum.webrtcbench"}
    full_viewport = {"width": 411, "height": 844}
    page = review.page(initial(c, {"apk": apk, "polar": "unconfirmed", "advanceRenderer": False}),
                       viewport=full_viewport)
    try:
        page.clock.run_for(2200)
        page.locator("#operator-connect-next").click()
        next_step = page.locator("#operator-update-next")
        if next_step.is_visible():
            next_step.click()
        surname = page.locator("#patient-surname:invalid")
        page.locator("#patient-next").click()
        expect(surname).to_be_focused()

        page.set_viewport_size({**full_viewport, "height": 428})
        review.shot(page, "participant-keyboard", "Empty participant form", surname, "Keyboard")
        geometry = page.evaluate("""() => ({
          height: visualViewport.height,
          scroll: document.documentElement.scrollHeight,
          buttons: [...document.querySelectorAll('#participant-screen footer button')]
            .map(button => button.getBoundingClientRect().toJSON()),
        })""")
        assert geometry["scroll"] <= geometry["height"], geometry
        assert all(button["bottom"] <= geometry["height"] and button["height"] >= 48
                   and button["width"] >= 48 for button in geometry["buttons"]), geometry

        page.set_viewport_size(full_viewport)
        page.clock.run_for(250)
        expect(surname).to_be_focused()
        expect(surname).to_have_value("")
    finally:
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
            capture_questionnaire(review, tag, viewport, font)
            print(f"Checked {tag}: {len(review.samples) - count_before} screens and dialogs", flush=True)
        review.text_spacing = False
        viewport = {"width": 390, "height": 844}
        capture_questionnaire(review, "aborted", viewport, "", outcome="aborted")
        capture_questionnaire(review, "probe", viewport, "", scenario="probe-5")
        capture_questionnaire(review, "recordings", viewport, "", live=False)
        capture_apk_detail_change(review)
        capture_connect_visibility_change(review)
        capture_companion_permission(review)
        capture_focused_viewport_change(review)
        capture_participant_keyboard(review)
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
