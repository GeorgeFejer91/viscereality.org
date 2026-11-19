"""Render the production PWA under the playground's strict no-scroll rules."""
import hashlib
import json
import re
import sys
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit

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
    page = review.page(initial(c, {"kiosk": kiosk_status, "music": {"v": 1, "state": "off", "volume": 35, "revision": 0}}), viewport=viewport)
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
    music = page.get_by_role("button", name="Hintergrundmusik")
    music.focus()
    page.keyboard.press("Enter")
    music_dialog = page.get_by_role("dialog", name="Musik")
    review.shot(page, f"{tag}-music", "Background music", music_dialog, "Dialogs")
    play = music_dialog.get_by_role("button", name="Abspielen")
    volume = music_dialog.get_by_role("slider", name="Lautstärke")
    expect(volume).to_have_value("35")
    assert volume.bounding_box()["height"] >= 48
    play.click()
    expect(music_dialog.get_by_role("button", name="Stoppen")).to_be_visible()
    expect(music_dialog.get_by_role("status")).to_have_text("Die Quest lädt die Musik …")
    page.evaluate("Object.assign(window.reviewQuest.music, {state: 'playing', revision: 2})")
    page.clock.run_for(1100)
    expect(music_dialog.get_by_role("status")).to_be_hidden()
    volume.focus()
    page.keyboard.press("ArrowRight")
    expect(music_dialog.locator("output")).to_have_text("36 %")
    music_dialog.get_by_role("button", name="Stoppen").click()
    expect(play).to_be_visible()
    assert page.evaluate("window.reviewMusicCommands") == [
        {"v": 1, "playing": True}, {"v": 1, "volume": 36}, {"v": 1, "playing": False}]
    page.keyboard.press("Escape")
    expect(music).to_be_focused()
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


def capture_music_keyboard_burst(review):
    viewport = {"width": 320, "height": 640}
    music = {"v": 1, "state": "playing", "volume": 35, "revision": 1}
    page = review.page(initial(c, {"music": music}), viewport=viewport)
    try:
        page.clock.run_for(4000)
        expect(page.locator("#check-control")).to_have_attribute("data-check", "ok")
        go(page, "settings")
        page.locator("#music-open").click()
        dialog = page.get_by_role("dialog", name="Musik")
        slider = dialog.get_by_role("slider", name="Lautstärke")
        page.evaluate("""() => {
          window.reviewHoldMusicVolume = true;
          window.reviewPhoneAudioStarts = 0;
          const originalPlay = HTMLMediaElement.prototype.play;
          HTMLMediaElement.prototype.play = function (...args) {
            window.reviewPhoneAudioStarts++;
            return originalPlay.apply(this, args);
          };
        }""")
        slider.focus()
        page.keyboard.press("ArrowRight")
        page.wait_for_function("Boolean(window.reviewReleaseMusicVolume)", timeout=4000)
        page.keyboard.press("ArrowRight")
        page.keyboard.press("ArrowRight")
        snapshot = page.evaluate("""() => ({
            value: document.querySelector('#music-volume').value,
            commands: window.reviewMusicCommands,
        })""")
        assert snapshot == {"value": "38", "commands": [{"v": 1, "volume": 36}]}, snapshot
        page.evaluate("window.reviewHoldMusicVolume = false; window.reviewReleaseMusicVolume()")
        page.wait_for_function("window.reviewMusicCommands.length === 2", timeout=3000)
        expect(slider).to_be_focused()
        expect(dialog.locator("output")).to_have_text("38 %")
        assert page.evaluate("window.reviewMusicCommands") == [{"v": 1, "volume": 36}, {"v": 1, "volume": 38}]
        page.evaluate("window.reviewHoldMusicVolume = true; window.reviewReleaseMusicVolume = undefined")
        page.keyboard.press("ArrowRight")
        page.wait_for_function("Boolean(window.reviewReleaseMusicVolume)", timeout=4000)
        page.keyboard.press("ArrowRight")
        expect(slider).to_have_value("40")
        page.evaluate("window.reviewQuest.access.revision++")
        page.clock.run_for(1100)
        page.evaluate("window.reviewHoldMusicVolume = false; window.reviewReleaseMusicVolume()")
        expect(slider).to_have_value("39")
        assert len(page.evaluate("window.reviewMusicCommands")) == 3, "Old lease must retire the unsent volume."
        dialog.get_by_role("button", name="Stoppen").click()
        expect(dialog.get_by_role("button", name="Abspielen")).to_be_visible()
        dialog.get_by_role("button", name="Abspielen").click()
        expect(dialog.get_by_role("button", name="Stoppen")).to_be_visible()
        assert page.evaluate("window.reviewPhoneAudioStarts") == 0
        assert not page.evaluate("performance.getEntriesByType('resource').some(r => r.name.includes('ji3ji3.mp3'))")
        review.shot(page, "music-keyboard-burst", "Music keyboard controls", dialog, "Dialogs")
    finally:
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
  const prefix = new TextEncoder().encode('schema_version,participant_id,session_id,run_index,scenario_id,record_type,timestamp_utc,elapsed_us\\r\\n4,Tes_30_N,' +
    session_id + ',2,' + scenario + ',official-start,2026-11-18T11:45:00.012345Z,0\\r\\n');
  const outcome = reviewQuest.recordingAborted ? 'aborted' : 'completed';
  const audioElapsed = reviewQuest.recordingAborted ? '830000000' : '899902362';
  const terminal = ['4','Tes_30_N',session_id,'2',scenario,'end','2026-11-18T12:00:00Z',
    '900000000',audioElapsed,'ended','ended',...Array(10).fill(''),outcome,outcome,''].join(',');
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


def connect_fixture_drive(page):
    folder = "fixture-folder"
    email = "research@example.test"
    remote = {}
    sessions = {}
    payloads = {}
    reservations = []
    lost_answers = []
    headers = {
        "Access-Control-Allow-Origin": "https://viscereality.org",
        "Access-Control-Allow-Methods": "GET, POST, PUT, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Expose-Headers": "Location",
    }

    def google(route):
        url = urlsplit(route.request.url)
        method = route.request.method
        def reply(body=None, status=200, **extra):
            route.fulfill(status=status, headers={**headers, **extra}, json=body)
        if method == "OPTIONS":
            reply()
        elif url.path == "/drive/v3/about":
            reply({"user": {"emailAddress": email, "permissionId": "fixture-owner"}})
        elif url.path == "/drive/v3/files" and method == "GET":
            query = parse_qs(url.query)["q"][0]
            parent = re.match(r"^'([^']+)' in parents", query)[1]
            name = re.search(r"\bname = '([^']+)'", query)
            recording = re.search(r"appProperties has \{ key='questRecordingId' and value='([^']+)' \}", query)
            mime = re.search(r"mimeType = '([^']+)'", query)
            reply({"files": [{"id": entry["id"]} for entry in remote.values()
                             if entry["parents"] == [parent] and not entry["trashed"]
                             and (not name or entry["name"] == name[1])
                             and (not recording or entry.get("appProperties", {}).get("questRecordingId") == recording[1])
                             and (not mime or entry["mimeType"] == mime[1])]})
        elif url.path.endswith("/generateIds"):
            reservations.append(len(reservations) + 1)
            reply({"ids": [folder if len(reservations) == 1 else f"file-{len(reservations)}"]})
        elif url.path == f"/drive/v3/files/{folder}":
            reply({"id": folder, "mimeType": "application/vnd.google-apps.folder",
                   "ownedByMe": True, "shared": False, "trashed": False})
        elif url.path.startswith("/drive/v3/files/"):
            found = remote.get(url.path.rsplit("/", 1)[1])
            reply(found, 200 if found else 404)
        elif method == "POST":
            metadata = route.request.post_data_json
            parent, = metadata["parents"]
            if metadata["mimeType"] == "application/vnd.google-apps.folder":
                if metadata["name"].startswith("Pair_"):
                    assert parent == folder
                else:
                    assert remote[parent]["name"].startswith("Pair_")
                    assert metadata["name"].startswith("2026-11-18T11-45-00.012345Z__")
                    assert metadata["appProperties"]["phoneClockSource"] == "timeapi.io"
                remote[metadata["id"]] = {**metadata, "ownedByMe": True, "shared": False, "trashed": False}
                reply({"id": metadata["id"]})
            else:
                assert remote[parent]["appProperties"]["questRecordingId"].startswith("utc-ns-")
                assert metadata["appProperties"]["clockSource"] == (
                    "timeapi.io" if metadata["name"].endswith("_questionnaire.json") else "quest-official-start"
                )
                sessions[metadata["id"]] = metadata
                reply(Location=f"https://www.googleapis.com/upload/drive/v3/files?upload_id={metadata['id']}")
        else:
            assert method == "PUT"
            file_id = parse_qs(url.query)["upload_id"][0]
            metadata = sessions[file_id]
            body = route.request.post_data_buffer
            payloads[metadata["name"]] = body
            remote[file_id] = {**metadata, "ownedByMe": True, "shared": False, "trashed": False,
                               "size": str(len(body)), "sha256Checksum": hashlib.sha256(body).hexdigest()}
            if metadata["name"].endswith("_questionnaire.json") and not lost_answers:
                assert metadata["mimeType"] == "application/json"
                lost_answers.append(file_id)
                reply(status=503)
            else:
                reply({"id": file_id})

    page.context.route("https://www.googleapis.com/**", google)
    page.context.route("https://accounts.google.com/gsi/client", lambda route: route.fulfill(
        content_type="text/javascript", body="""window.google={accounts:{oauth2:{
        initTokenClient: options => ({requestAccessToken: () => options.callback({
          access_token:'fixture-token', scope:'https://www.googleapis.com/auth/drive.file', expires_in:3600
        })})
    }}};"""))
    def authorize():
        page.locator("#drive-open").click()
        page.locator("#quest-drive-client").fill("12345-fixture.apps.googleusercontent.com")
        page.locator("#quest-drive-account").fill(email)
        page.locator("#quest-drive-connect").click()

    authorize()
    expect(page.locator("#quest-drive-status")).to_have_text("In Google Drive bestätigt: 1")
    assert len(payloads) == 1 and len(reservations) == 4
    return payloads, reservations, lost_answers, remote, authorize


def rebuild_drive_journal(page, authorize, questionnaire_id):
    page.evaluate("""() => navigator.locks.request('atmungsraum-quest-drive', () => new Promise((resolve, reject) => {
        const request = indexedDB.open('atmungsraum-quest-recordings', 4);
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
            const db = request.result;
            const tx = db.transaction('drive', 'readwrite');
            tx.oncomplete = () => { db.close(); resolve(); };
            tx.onabort = () => { db.close(); reject(tx.error); };
            const cursor = tx.objectStore('drive').openCursor();
            cursor.onsuccess = () => {
                const entry = cursor.result;
                if (!entry) return;
                if (entry.key !== 'destination') entry.delete();
                entry.continue();
            };
        };
    }))""")
    go(page, "recordings")
    with page.expect_response(lambda response: urlsplit(response.url).path == f"/drive/v3/files/{questionnaire_id}"):
        authorize()
    expect(page.locator("#quest-drive-status")).to_have_text("In Google Drive bestätigt: 2")
    return page.evaluate("""() => new Promise((resolve, reject) => {
        const request = indexedDB.open('atmungsraum-quest-recordings', 4);
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
            const db = request.result;
            const receipts = db.transaction('drive').objectStore('drive').getAll();
            receipts.onsuccess = () => {
                db.close();
                resolve(receipts.result.filter(row => row.verified).map(row => row.driveId));
            };
            receipts.onerror = () => { db.close(); reject(receipts.error); };
        };
    })""")


def expect_questionnaire_focus(control):
    expect(control).to_be_focused()
    bounds = control.evaluate("""element => {
        const target = element.getBoundingClientRect();
        const form = element.form.getBoundingClientRect();
        return { above: Math.round(form.top - target.top), below: Math.round(target.bottom - form.bottom),
                 pageScroll: document.scrollingElement.scrollTop };
    }""")
    assert bounds["above"] <= 0 and bounds["below"] <= 0 and bounds["pageScroll"] == 0, bounds


def resize_recordings_fixture(page, viewport):
    page.set_viewport_size(viewport)
    # The CSS media query updates before the app receives visualViewport.resize.
    expect(page.locator("html")).to_have_attribute("style", re.compile(rf"--usable-height:\s*{viewport['height']}px\b"))


def capture_short_recordings(review, page):
    initial_viewport = page.viewport_size
    start = mutations(page)
    topic = page.locator("#recordings-view")
    answer = page.locator("#questionnaire-download")
    answer.focus()
    for width, height in ((320, 428), (640, 320), (594, 272)):
        resize_recordings_fixture(page, {"width": width, "height": height})
        expect(topic).to_be_visible()
        expect(answer).to_be_focused()
        page.clock.run_for(250)
        prefix = f"short-recordings-{width}"
        review.shot(page, prefix, "Recordings after saving", page.locator("#recordings-screen"), "Recordings")
        assert not review.samples[prefix]["problems"], review.samples[prefix]["problems"]
        for view, targets in (
            ("archive", ("#quest-archives",)),
            ("files", ("#quest-export-download", "#questionnaire-download")),
            ("sync", ("#quest-export-status", "#quest-export-retry")),
            ("storage", ("#drive-open", '#recordings-screen a[href="#daten"]')),
        ):
            topic.select_option(view)
            for target in targets:
                locator = page.locator(target)
                expect(locator).to_be_visible()
                if target != "#quest-export-status":
                    assert min(locator.bounding_box()[edge] for edge in ("width", "height")) >= 47.99
            slug = f"{prefix}-{view}"
            review.shot(page, slug, view, topic, "Recordings")
            assert not review.samples[slug]["problems"], review.samples[slug]["problems"]
            if view == "archive":
                topic.focus()
                page.keyboard.press("Tab")
                expect(page.locator("#quest-archives")).to_be_focused()
            if view == "files":
                with page.expect_download() as exported:
                    page.locator("#quest-export-download").click()
                assert exported.value.suggested_filename == "Tes_30_N_2026-11-18_haupt-15_r2.csv"
                with page.expect_download() as exported:
                    answer.click()
                assert exported.value.suggested_filename == "Tes_30_N_2026-11-18_haupt-15_r2_questionnaire.json"
            if view == "sync":
                page.evaluate("""() => {
                    window.readFixtureRecordings = window.reviewRecordings;
                    window.reviewRecordings = (query, owner) => query.query === 'list'
                        ? Promise.reject(Error('Fixture link loss')) : window.readFixtureRecordings(query, owner);
                }""")
                retry = page.locator("#quest-export-retry")
                status = page.locator("#quest-export-status")
                retry.click()
                expect(status).to_contain_text("Abgleich fehlgeschlagen")
                review.shot(page, f"{slug}-failed", "Abgleich fehlgeschlagen", retry, "Recordings")
                assert not review.samples[f"{slug}-failed"]["problems"], review.samples[f"{slug}-failed"]["problems"]
                page.evaluate("""() => {
                    window.reviewRecordings = window.readFixtureRecordings;
                    delete window.readFixtureRecordings;
                }""")
                retry.click()
                expect(status).to_have_text("Aufnahmen gespeichert.")
            if view == "storage":
                drive = page.locator("#drive-open")
                drive.click()
                expect(page.locator("#drive-dialog")).to_be_visible()
                page.keyboard.press("Escape")
                expect(drive).to_be_focused()
                page.locator('#recordings-screen a[href="#daten"]').click()
                expect(page.locator("#daten")).to_be_visible()
                page.go_back()
                expect(topic).to_have_value("storage")
        topic.select_option("files")
        answer.focus()
    assert mutations(page) == start
    resize_recordings_fixture(page, initial_viewport)
    expect(topic).to_be_hidden()


def capture_disconnected_recordings(review, font, *, cached=False):
    page = review.page(initial(c) + """
      const pair = JSON.parse(localStorage.getItem('atmungsraum-quest-pair'));
      pair.endpoint = 'unreachable';
      pair.build = 'c'.repeat(64);
      localStorage.setItem('atmungsraum-quest-pair', JSON.stringify(pair));
      window.reviewDiscoveryState = 'searching';
    """, viewport={"width": 390, "height": 844})
    if cached:
        page.evaluate("""async () => {
            const { openQuestRecordingStore } = await import('./quest-recording-store.js?v=5');
            const pair = JSON.parse(localStorage.getItem('atmungsraum-quest-pair'));
            const store = await openQuestRecordingStore(pair);
            try {
                await store.put('utc-ns-01763115400000000000',
                    new File(['synthetic,sample\\n1,0\\n'], 'Tes_30_N_2025-11-14_haupt-15_r1.csv', {type:'text/csv'}));
            } finally { store.close(); }
        }""")
        page.reload()
    page.evaluate("size => document.documentElement.style.fontSize = size", font)
    page.clock.run_for(2100)
    status = page.locator("#quest-status")
    expect(status).to_have_text("Suche die gespeicherte Quest über Bluetooth …")
    initial_mutations = mutations(page)
    go(page, "recordings")
    resize_recordings_fixture(page, {"width": 594, "height": 272})
    topic = page.locator("#recordings-view")
    back = page.locator("#recordings-screen [data-operator-return]")
    data = "cached" if cached else "empty"
    for view, target in (
        ("archive", "#quest-archives" if cached else "#recordings-archive-empty"),
        ("files", "#quest-export-download" if cached else "#recordings-files-empty"),
        ("sync", "#recordings-sync-empty"),
        ("storage", '#recordings-screen a[href="#daten"]'),
    ):
        topic.select_option(view)
        expect(status).to_have_text("Quest wird gesucht …")
        expect(page.locator("#drive-open")).to_be_hidden()
        slug = f"short-recordings-disconnected-{data}-{font or 'default'}-{view}"
        control = page.locator(target)
        review.shot(page, slug, view, control, "Recordings")
        assert not review.samples[slug]["problems"], review.samples[slug]["problems"]
        if view == "storage" or cached and view in ("archive", "files"):
            assert min(control.bounding_box()[edge] for edge in ("height", "width")) >= 47.99
        bounds = back.bounding_box()
        assert bounds["height"] >= 47.99 and bounds["y"] + bounds["height"] <= 272.5, bounds
    back.click()
    expect(page.locator("#quest")).to_be_visible()
    assert mutations(page) == initial_mutations
    page.context.close()


def capture_questionnaire(review, tag, viewport, font, *, outcome="completed", scenario="haupt-15", live=True,
                          drive=False, focus_viewport=None, short_recordings=False):
    quest = {"phase": "RUNNING", "host": "vr-session", "generation": 3, "condition": "condition-a",
             "clock": 895000, "duration_ms": 900000, "liveRecording": live,
             "recordingAborted": outcome == "aborted", "recordingScenario": scenario}
    setup = {"stage": "main", "input": "controller-state", "playing": True, "ready": True}
    enrollment = "localStorage.setItem('atmungsraum-quest-pair'"
    fixture = initial(c, quest, setup).replace(enrollment, "if (!localStorage.getItem('atmungsraum-quest-pair')) " + enrollment, 1)
    fixture += """
      const captureFetch = window.fetch;
      window.fetch = (url, options) => String(url) === 'https://www.timeapi.io/api/Time/current/zone?timeZone=Etc%2FUTC'
        ? Promise.resolve(Response.json({ timeZone:'Etc/UTC', dateTime:'2026-11-18T12:00:00.1234567' }))
        : captureFetch(url, options);
    """
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
            expect(page.locator("#questionnaire-open")).to_be_visible()
            if drive:
                payloads, reservations, lost_answers, remote, authorize = connect_fixture_drive(page)
            page.locator("#questionnaire-open").click()
        expect(form).to_be_visible()
        if tag == "390" and live:
            # The pending form and its run survive a fresh document without the Quest.
            page.add_init_script("window.reviewJoin='fail';")
            page.reload(wait_until="load")
            expect(form).to_be_visible()
            # The recording and its unfinished form belong to this pairing.
            page.locator("#questionnaire-q1 input").press("Home")
            page.evaluate('''() => {
                const key = 'atmungsraum-quest-pair';
                window.reviewSavedQuestPair = localStorage.getItem(key);
                localStorage.removeItem(key);
                dispatchEvent(new StorageEvent('storage', {key}));
            }''')
            expect(form).to_be_hidden()
            expect(page.locator("#questionnaire-q1 input")).to_have_attribute("aria-valuetext", "Noch keine Auswahl")
            page.evaluate('''() => {
                const key = 'atmungsraum-quest-pair';
                localStorage.setItem(key, window.reviewSavedQuestPair);
                dispatchEvent(new StorageEvent('storage', {key}));
            }''')
            expect(form).to_be_visible()
        if focus_viewport:
            page.set_viewport_size(focus_viewport)
        fields = page.locator("#questionnaire-form")
        first = page.locator("#questionnaire-q1 input")
        last = page.locator("#questionnaire-q5 input")
        submit = page.locator("#questionnaire-submit")
        expect(fields.locator("fieldset")).to_have_count(5)
        expect(fields.locator('input[type="radio"]')).to_have_count(15)
        expect(fields.locator('input[type="range"]')).to_have_count(2)
        submit.click()
        expect_questionnaire_focus(first)
        expect(first).to_have_attribute("aria-invalid", "true")
        expect(page.locator("#questionnaire-status")).to_have_text("Bitte beantworte alle fünf Fragen.")
        review.shot(page, f"{tag}-questionnaire-first", "Scroll form at the first item", first, "Questionnaire")
        first.press("Home")
        expect(first).to_have_attribute("aria-valuetext", "0")
        if focus_viewport:
            submit.click()
            expect_questionnaire_focus(fields.locator("#questionnaire-q2 input").first)
        for question, value in ((2, 1), (3, 5), (4, 3)):
            choice = fields.locator(f'#questionnaire-q{question} input[value="{value}"]')
            choice.check()
            expect(choice).to_be_checked()
        submit.click()
        expect_questionnaire_focus(last)
        expect(last).to_have_attribute("aria-invalid", "true")
        expect(last).to_have_attribute("aria-valuetext", "Noch keine Auswahl")
        if tag in ("320", "landscape", "enlarged"):
            assert fields.evaluate("element => element.scrollTop") > 0
            assert page.evaluate("document.scrollingElement.scrollTop") == 0
        last.press("End")
        expect(last).to_have_attribute("aria-valuetext", "100")
        if tag == "390":
            page.set_viewport_size({**viewport, "height": 428})
        last.scroll_into_view_if_needed()
        review.shot(page, f"{tag}-questionnaire-last", "Scroll form at the last item", last, "Questionnaire")
        expect(last).to_be_focused()
        submit.click()
        link = page.locator("#questionnaire-download")
        expect(link).to_be_visible()
        if (tag == "390" and live) or focus_viewport:
            resize_recordings_fixture(page, viewport)
        review.shot(page, f"{tag}-questionnaire-saved", "Saved answers", link, "Questionnaire")
        if short_recordings:
            capture_short_recordings(review, page)
        assert mutations(page)["session"] == []
        with page.expect_download() as download:
            link.click()
        assert download.value.suggested_filename == "Tes_30_N_2026-11-18_haupt-15_r2_questionnaire.json"
        file_bytes = download.value.path().read_bytes()
        if drive:
            page.clock.run_for(25000)
            expect(page.locator("#quest-drive-status")).to_have_text("In Google Drive bestätigt: 2")
            assert payloads[download.value.suggested_filename] == file_bytes
            assert len(payloads) == 2 and len(reservations) == 5 and len(lost_answers) == 1
            assert json.loads(file_bytes)["clock_source"] == "timeapi.io"
            file_ids = {entry["id"] for entry in remote.values() if entry["name"] in payloads}
            assert set(rebuild_drive_journal(page, authorize, lost_answers[0])) == file_ids
            assert len(remote) == 4 and len(reservations) == 5
        record = json.loads(file_bytes)
        assert record["participant_id"] == "Tes_30_N"
        assert record["session_id"] == "01234567-89ab-4def-8abc-0123456789ab"
        assert record["recording_filename"] == "Tes_30_N_2026-11-18_haupt-15_r2.csv"
        assert record["schema_version"] == 2
        assert [item["answer"] for item in record["answers"]] == [0, 1, 5, 3, 100]
        assert [item["type"] for item in record["answers"]] == ["slider", "likert", "likert", "likert", "slider"]
        if tag == "390" and live:
            page.reload(wait_until="load")
            expect(link).to_be_visible()
            expect(form).to_be_hidden()
    finally:
        page.context.close()


def capture_error_recording_recovery(review, tag, viewport, font):
    page = review.page(initial(c, {"phase": "ERROR", "host": "vr-session", "generation": 3,
                                  "condition": "condition-a", "clock": 155647,
                                  "errorRecording": True}), suffix="#session", viewport=viewport)
    try:
        page.evaluate("size => document.documentElement.style.fontSize = size", font)
        page.clock.run_for(2000)
        expect(page.locator("#running-screen")).to_be_visible()
        expect(page.locator("#quest-play")).to_be_disabled()
        expect(page.locator("#quest-finish")).to_be_enabled()
        review.shot(page, f"{tag}-error-active-writer", "Save failed session",
                    page.locator("#quest-finish"), "Recovery")
        page.locator("#quest-finish").click()
        page.clock.run_for(1000)
        assert mutations(page)["session"] == ["save-next"], "error recovery may only save the existing writer"
    finally:
        page.context.close()


def capture_error_recording_update_save(review):
    meta = {"v": 2, "state": "current", "reason": "installed-version-is-current",
            "installed_code": apk_fixture()["version_code"], "available_code": None,
            "downloaded_bytes": 0, "total_bytes": None,
            "session_allowed": True, "required_code": None, "cycle": "1", "revision": "1",
            "attempt": None, "can_check": False, "busy_reason": "session-active"}
    page = review.page(initial(c, {"phase": "ERROR", "host": "vr-session", "generation": 3,
                                  "errorRecording": True, "metaUpdate": meta}), suffix="#settings")
    try:
        page.clock.run_for(2000)
        go(page, "update")
        button = page.locator("#meta-update-finish")
        expect(button).to_be_enabled()
        button.click()
        page.clock.run_for(1000)
        assert mutations(page)["session"] == ["save-next"], "Update recovery must finish the existing writer first"
    finally:
        page.context.close()


def capture_polar_signal_pause(review, tag, viewport, font):
    for recovered in (False, True):
        health = {"required": True, "pause_latched": True, "invalid_timeout_ms": 5000,
                  "usable_fresh": recovered, "last_pause": {"reason": "breathing-data-unusable"}}
        page = review.page(initial(c, {"phase": "PAUSED", "host": "vr-session", "generation": 3,
                                      "condition": "condition-a", "polarInput": health}), viewport=viewport)
        try:
            page.evaluate("size => document.documentElement.style.fontSize = size", font)
            page.clock.run_for(2000)
            status = page.locator("#quest-status")
            expect(status).to_contain_text("Polar-Signal wieder bereit" if recovered else "Gurtposition, Hautkontakt und Batterie")
            button = page.locator("#quest-play")
            expect(button).to_have_text("Fortsetzen")
            assert mutations(page)["session"] == [], "Polar recovery never resumes automatically"
            if recovered:
                expect(button).to_be_enabled()
            else:
                expect(button).to_be_disabled()
            review.shot(page, f"{tag}-polar-pause-{'ready' if recovered else 'waiting'}",
                        "Polar signal pause", status, "Recovery")
            if recovered:
                button.click()
                page.clock.run_for(500)
                assert mutations(page)["session"] == ["resume"]
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
            capture_error_recording_recovery(review, tag, viewport, font)
            capture_polar_signal_pause(review, tag, viewport, font)
            capture_recordings(review, tag, viewport, font)
            capture_questionnaire(review, tag, viewport, font)
            print(f"Checked {tag}: {len(review.samples) - count_before} screens and dialogs", flush=True)
        review.text_spacing = False
        viewport = {"width": 390, "height": 844}
        capture_questionnaire(review, "aborted", viewport, "", outcome="aborted")
        capture_questionnaire(review, "probe", viewport, "", scenario="probe-5")
        capture_questionnaire(review, "recordings", viewport, "", live=False, drive=True)
        capture_questionnaire(review, "compact-enlarged", viewport, "200%", focus_viewport={"width": 320, "height": 428})
        capture_questionnaire(review, "landscape-enlarged", viewport, "200%", focus_viewport={"width": 640, "height": 320})
        capture_questionnaire(review, "recordings-enlarged", viewport, "200%", short_recordings=True)
        for font in ("", "200%"):
            capture_disconnected_recordings(review, font)
            capture_disconnected_recordings(review, font, cached=True)
        capture_music_keyboard_burst(review)
        capture_error_recording_update_save(review)
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
