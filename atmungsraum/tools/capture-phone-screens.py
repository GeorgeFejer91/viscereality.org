from argparse import ArgumentParser
from html import escape
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit
import csv
import shutil
import subprocess
import zipfile

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
URL = "https://viscereality.org/atmungsraum/"
ARCHIVE = "atmungsraum-smartphone-review.zip"
VIEWPORT = {"width": 390, "height": 844}

GATEWAY = """
export const owner = { serial: 'DEMO-QUEST', device: 'demo-device' };
const candidate = 'b'.repeat(12);
const settings = {
  phase: 'IDLE', host: 'vr-preparation', polar: 'off', fresh: true, generation: 0,
  condition: 'condition-b', clock: 0, pending: false, strapCount: 0, selected: ''
};
window.reviewQuest = settings;
const scenarios = ['probe-anleitung', 'probe-5', 'probe-10', 'probe-15', 'haupt-15'];
function snapshot() {
  const { phase, polar, generation, condition, clock, host, fresh, pending } = settings;
  const connected = ['stream-wait', 'ready'].includes(polar);
  return {
    build: { packaging_fingerprint: 'demo' }, p: phase, h: host, g: generation,
    rec: phase !== 'IDLE' && phase !== 'ERROR', c: condition, ms: clock,
    pending, participant_arm: 1, ecg_preview: 1, recenter: 1,
    pi: { r: true, f: fresh }, a: ['track-ready', 'track-ready'], o: { e: '1' },
    scenarios, y: polar === 'unsupported' ? undefined : {
      s: 'a'.repeat(32), g: 1, r: 1, p: polar !== 'permission', b: polar !== 'bluetooth',
      c: connected, a: polar === 'ready', e: polar === 'ready' || Boolean(settings.ecg), sc: polar === 'scan',
      x: connected ? candidate : settings.selected,
      d: [candidate, 'c'.repeat(12)].slice(0, settings.strapCount)
    },
    q: { v: 2, s: 'a'.repeat(32), g: generation,
      o: [{ ARMED: 'start', RUNNING: 'pause', PAUSED: 'resume' }[phase]] }
  };
}
function command({ id, op, condition, d }) {
  const states = { arm: 'ARMED', start: 'RUNNING', pause: 'PAUSED', resume: 'RUNNING', 'save-next': 'IDLE' };
  if (op === 'arm') {
    settings.generation++;
    settings.condition = condition;
    settings.host = 'vr-session';
  }
  if (op === 'start') settings.clock = 37000;
  if (op === 'select') settings.selected = d;
  if (op === 'connect') settings.polar = 'stream-wait';
  if (op === 'start-acc') settings.polar = 'ready';
  if (op === 'save-next') {
    settings.clock = 0;
    settings.host = 'vr-preparation';
  }
  if (states[op]) settings.phase = states[op];
  if (op === 'start-ecg') settings.ecg = true;
  return { v: 1, id, state: 'confirmed' };
}
export function sameQuest(a, b) { return a?.serial === b?.serial; }
export async function selectQuest() {
  if (window.reviewBluetooth === 'wait') await new Promise(() => {});
  return { id: 'demo-bluetooth' };
}
export async function readQuest(_peer, _signal, setup) {
  if (setup.forceHotspot) {
    setup.onApproval();
    await new Promise(() => {});
  }
  return { owner, endpoints: ['synthetic'] };
}
export async function joinQuest() {
  if (window.reviewJoin === 'wait') await new Promise(() => {});
  if (window.reviewJoin === 'fail') throw Error('Simulated connection failure');
  return {
    owner, initial: snapshot(), status: snapshot, command, participantArm: command, scenarioArm: command,
    setup: command, receipt: async () => null, close() {},
    recordings: async () => ({ v: 1, state: 'ready', entries: [], next_before: null }),
    ecg: async () => settings.polar !== 'ready' ? null : ({
      v: 1, unit: 'uV', rate: 130, age_ms: 0,
      samples: Array.from({ length: 260 }, (_, i) => {
        const cycle = i % 95;
        const beat = 310 * Math.exp(-((cycle - 36) ** 2) / 2)
          - 120 * Math.exp(-((cycle - 40) ** 2) / 3)
          + 40 * Math.exp(-((cycle - 52) ** 2) / 36);
        return [String(1000000000 + i * 7692308), Math.round(beat + 12 * Math.sin(i / 11))];
      })
    })
  };
}
"""

PAIRED = ("{endpoint:'synthetic',serial:'DEMO-QUEST',device:'demo-device',"
          "enrollment:{controller:'" + "a" * 32 + "',key:'" + "b" * 64 + "'}}")


class Review:
    def __init__(self, browser, output):
        self.browser = browser
        self.output = output
        self.rows = []

    def page(self, init="", suffix=""):
        context = self.browser.new_context(
            viewport=VIEWPORT, device_scale_factor=2, is_mobile=True, has_touch=True,
            locale="de-DE", timezone_id="Europe/Berlin", reduced_motion="reduce",
            service_workers="block",
        )

        def route(request_route):
            url = urlsplit(request_route.request.url)
            if url.netloc == "accounts.google.com" and url.path == "/gsi/client":
                request_route.fulfill(content_type="text/javascript",
                                      body="window.google={accounts:{oauth2:{}}};")
                return
            path = url.path.removeprefix("/atmungsraum/")
            if url.netloc != "viscereality.org" or not url.path.startswith("/atmungsraum/"):
                request_route.abort()
                return
            if path == "quest-gateway.js" and url.query == "v=7":
                request_route.fulfill(content_type="text/javascript", body=GATEWAY)
                return
            file = (ROOT / (path or "index.html")).resolve()
            if file.is_relative_to(ROOT) and file.is_file():
                request_route.fulfill(path=file)
            else:
                request_route.fulfill(status=404)

        context.route("**/*", route)
        page = context.new_page()
        page.clock.install()
        if init:
            page.add_init_script(init)
        page.goto(URL + suffix, wait_until="load")
        return page

    def shot(self, page, slug, title, expected, group):
        expect(expected).to_be_visible(timeout=9000)
        expect(page.locator("html")).to_have_attribute("data-measurement", "ready")
        expect(page.locator(".screen:visible")).to_have_count(1)
        page.clock.run_for(100)
        expect(expected).to_be_visible()
        page.evaluate("window.scrollTo(0, 0)")
        width = page.evaluate("document.documentElement.scrollWidth")
        if width > VIEWPORT["width"] + 1:
            print(f"Layout note: {slug} has {width}px of document width", flush=True)
        number = f"{len(self.rows) + 1:02d}"
        path = f"screens/{number}-{slug}.png"
        for element in (expected, page.locator(".screen:visible")):
            assert element.evaluate("""node => node.getBoundingClientRect().bottom + scrollY
                <= document.documentElement.scrollHeight + 1"""), f"{slug} exceeds capture bounds"
        page.screenshot(path=self.output / path, full_page=True)
        self.rows.append((number, group, title, path))
        print(f"{number} {title}", flush=True)

    def index(self):
        with (self.output / "feedback.csv").open("w", encoding="utf-8-sig", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(("Screen", "Section", "State", "Image", "Feedback"))
            writer.writerows((*row, "") for row in self.rows)
        entries = "\n".join(
            f'<figure><a href="{path}"><img src="{path}" alt="{escape(title)}" loading="lazy"></a>'
            f'<figcaption><strong>{number} · {escape(title)}</strong><br>{escape(group)}</figcaption></figure>'
            for number, group, title, path in self.rows
        )
        (self.output / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Atmungsraum smartphone review</title><style>'
            'body{font:16px/1.5 Arial,sans-serif;background:#f2f1eb;color:#24271f;margin:0;padding:24px}'
            'main{max-width:1400px;margin:auto}h1{line-height:1.2}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,230px),1fr));gap:22px}'
            'figure{margin:0;background:white;border:1px solid #cbcfc3;padding:12px}'
            'img{width:100%;max-height:550px;object-fit:contain;object-position:top;background:#000;display:block}'
            'figcaption{padding-top:10px}a{color:#284a32}</style><main>'
            '<h1>Atmungsraum smartphone review</h1>'
            '<p>Click any screenshot to open the full image. Refer to its number when giving feedback '
            'or add one note per screen in <a href="feedback.csv">feedback.csv</a>.</p>'
            '<p>390 × 844 CSS px, German interface, synthetic participant, sensor, and recording data. '
            'Long screens are captured at full page height.</p>'
            '<p><a href="protocol.md">Screenshot protocol</a></p>'
            f'<div class="grid">{entries}</div></main></html>',
            encoding="utf-8",
        )


def wait_state(page, **state):
    page.evaluate("state => Object.assign(window.reviewQuest, state)", state)
    page.clock.run_for(1100)


def capture_glow(page, control):
    page.emulate_media(reduced_motion="no-preference")
    control.evaluate("""async button => {
      const glow = button.getAnimations().find(animation => animation.animationName === 'preview-glow');
      if (!glow) throw Error('Preview glow is unavailable');
      await glow.ready;
      glow.pause();
      glow.currentTime = 1000;
    }""")


def capture(review):
    shot = review.shot
    page = review.page()
    find = page.locator("#quest-find")
    shot(page, "open", "Open Atmungsraum", find, "Connection")
    page.locator("#install-app").click()
    shot(page, "install-help", "Install help", page.locator("#install-dialog[open]"), "Connection")
    page.locator("#close-install").click()
    expect(page.locator("#quest-join, #quest-controller, #quest-key, #quest-repair")).to_have_count(0)
    find.focus()
    find.press("Enter")
    expect(page.locator("#quest-status")).to_contain_text("noch nicht freigegeben")
    expect(find).to_be_focused()
    shot(page, "unprovisioned", "Phone needs Quest authorization", page.locator("#quest-status"), "Connection")
    page.context.close()

    page = review.page("window.reviewBluetooth='wait';")
    page.locator("#quest-find").click()
    expect(page.locator("#quest-find")).to_have_attribute("aria-disabled", "true")
    expect(page.locator("#quest-status")).to_contain_text("Quest wird verbunden")
    shot(page, "bluetooth", "Waiting for Bluetooth", page.locator("#quest-status"), "Connection")
    page.context.close()

    paired = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({PAIRED}));"
    page = review.page(paired + "window.reviewJoin='fail';")
    expect(page.locator("#quest-status")).to_contain_text("Erlaube den Bluetooth-Zugriff")
    page.evaluate("window.reviewJoin = ''")
    button = page.locator("#quest-find")
    button.focus()
    button.press("Enter")
    expect(page.locator("#quest-polar")).to_be_visible()
    expect(page.locator("#quest-status")).to_be_focused()
    expect(button).to_be_hidden()
    page.context.close()

    for mode, slug, title, locator in (
        ("wait", "reconnecting", "Reconnect to a saved Quest", "#quest-status"),
        ("fail", "retry-bluetooth", "Retry Bluetooth connection", "#quest-find"),
    ):
        page = review.page(paired + f"window.reviewJoin='{mode}';")
        if mode == "fail":
            expect(page.locator("#quest-status")).to_contain_text("Erlaube den Bluetooth-Zugriff")
        if mode == "wait":
            expect(page.locator("#quest-status")).to_contain_text("Quest wird verbunden")
        shot(page, slug, title, page.locator(locator), "Connection")
        page.context.close()

    page = review.page(paired)
    status = page.locator("#quest-status")
    polar = page.locator("#quest-polar-status")
    expect(polar).to_contain_text("Lege den Gurt an")
    wait_state(page, host="panel", polar="unsupported")
    expect(status).to_contain_text("Schließe das Bedienfeld")
    shot(page, "headset-panel", "Headset panel still open", status, "Connection")
    wait_state(page, host="vr-preparation", polar="off")
    page.locator("#quest-participant summary").click()
    shot(page, "participant-empty", "Participant form", page.locator("#patient-form"), "Participant and Polar")
    page.locator("#patient-surname").fill("Tes")
    page.locator("#patient-age").fill("30")
    page.get_by_role("radio", name="Keine Angabe").check()
    expect(page.locator("#patient-id")).to_have_text("Tes_30_N")
    shot(page, "participant-ready", "Completed participant form", page.locator("#patient-id"), "Participant and Polar")
    page.locator("#patient-next").click()
    for state, slug, title, label in (
        ("off", "polar-connect", "Connect Polar H10", "Lege den Gurt an"),
        ("permission", "polar-permission", "Quest Bluetooth permission required", "Geräte in der Nähe"),
        ("bluetooth", "polar-bluetooth", "Quest Bluetooth switched off", "Schalte Bluetooth"),
        ("unsupported", "polar-unavailable", "Polar setup unavailable", "nicht verfügbar"),
        ("scan", "polar-scanning", "Searching for Polar H10", "Quest sucht"),
        ("stream-wait", "polar-streams", "Waiting for ECG and acceleration", "Warte auf beide"),
    ):
        wait_state(page, polar=state)
        expect(polar).to_contain_text(label)
        shot(page, slug, title, polar, "Participant and Polar")

    for count, slug, title, label in (
        (0, "polar-no-strap", "No Polar H10 found", "Kein Polar H10 gefunden"),
        (2, "polar-many-straps", "Multiple Polar straps found", "Mehrere Gurte gefunden"),
    ):
        wait_state(page, polar="off", strapCount=count)
        page.locator("#quest-polar-connect").click()
        expect(status).to_contain_text(label)
        shot(page, slug, title, status, "Participant and Polar")

    wait_state(page, polar="off", strapCount=1)
    page.locator("#quest-polar-connect").click()
    preview = page.locator("#preview-status")
    inhale = page.locator("#preview-inhale")
    exhale = page.locator("#preview-exhale")
    expect(preview).to_contain_text("„Einatmen“ drücken")
    shot(page, "inhale-prompt", "Atemvorschau · inhale prompt", preview, "Breathing preview")
    capture_glow(page, inhale)
    shot(page, "inhale-glow", "Atemvorschau · inhale glow at its peak", inhale, "Breathing preview")
    page.emulate_media(reduced_motion="reduce")
    inhale.click()
    expect(preview).to_contain_text("Einatmen läuft")
    shot(page, "inhale-active", "Atemvorschau · inhale in progress", preview, "Breathing preview")
    page.clock.run_for(4100)
    expect(preview).to_contain_text("„Ausatmen“ drücken")
    shot(page, "exhale-prompt", "Atemvorschau · exhale prompt", preview, "Breathing preview")
    capture_glow(page, exhale)
    shot(page, "exhale-glow", "Atemvorschau · exhale glow at its peak", exhale, "Breathing preview")
    page.emulate_media(reduced_motion="reduce")
    exhale.click()
    expect(preview).to_contain_text("Ausatmen läuft")
    shot(page, "exhale-active", "Atemvorschau · exhale in progress", preview, "Breathing preview")
    page.clock.run_for(4100)
    proceed = page.locator("#preview-continue")
    shot(page, "preview-complete", "Atemvorschau · green continuation", proceed, "Breathing preview")
    proceed.click()

    scenario = page.locator("#quest-scenario")
    condition = page.locator("#quest-condition")
    expect(scenario).to_be_visible()
    page.clock.run_for(1100)
    for value, title in (
        ("probe-anleitung", "Anleitung"),
        ("probe-5", "Probe · 5 Min."),
        ("probe-10", "Probe · 10 Min."),
        ("probe-15", "Probe · 15 Min."),
        ("haupt-15", "Haupt-Runde · 15 Min."),
    ):
        scenario.select_option(value)
        expect(scenario).to_have_value(value)
        shot(page, value, title, scenario, "Scenario selection")
    condition.select_option("condition-b")
    shot(page, "condition-b", "VR-Bedingung B selected", condition, "Scenario selection")

    play = page.locator("#quest-play")
    finish = page.locator("#quest-finish")
    wait_state(page, phase="ARMED", host="vr-session")
    expect(status).to_contain_text("vorbereitet")
    shot(page, "armed", "Run prepared", play, "Playback")
    play.click()
    expect(play).to_have_text("Pause")
    shot(page, "running", "Run active with ECG", page.locator("#quest-ecg"), "Playback")
    play.click()
    expect(play).to_have_text("Fortsetzen")
    shot(page, "paused", "Run paused", play, "Playback")
    wait_state(page, fresh=False, polar="stream-wait")
    expect(play).to_be_disabled()
    shot(page, "polar-hold", "Run held while Polar signal is stale", play, "Playback")
    wait_state(page, fresh=True, polar="ready")
    for phase, slug, title, label in (
        ("FINALIZING", "finalizing", "Run ending", "beendet den Probelauf"),
        ("SAVING", "saving", "Quest saving the run", "speichert den Probelauf"),
    ):
        wait_state(page, phase=phase, host="vr-session", clock=37000)
        expect(status).to_contain_text(label)
        shot(page, slug, title, finish, "Playback")
    wait_state(page, phase="ERROR")
    expect(status).to_contain_text("meldet einen Fehler")
    shot(page, "quest-error", "Quest reported an error", status, "Playback")
    wait_state(page, phase="PAUSED", host="vr-session", fresh=True, polar="ready")
    finish.click()
    expect(status).to_contain_text("Probelauf beendet")
    shot(page, "finished", "Run finished and returned to setup", status, "Playback")

    await_archive = page.locator("#quest-export-download")
    page.evaluate("""async () => {
      const { openQuestRecordingStore } = await import('./quest-recording-store.js?v=4');
      const store = await openQuestRecordingStore({ device: 'demo-device', serial: 'DEMO-QUEST' });
      try {
        await store.put('utc-ns-01763115400000000000',
          new File(['synthetic,sample\\n1,0\\n'], 'Tes_30_N_2025-11-14_haupt-15_r1.csv', {type:'text/csv'}));
      } finally { store.close(); }
    }""")
    wait_state(page, phase="IDLE", host="vr-preparation", clock=0)
    page.locator("#quest-export-retry").click()
    shot(page, "quest-recording", "Saved Quest recording and download", await_archive, "Recordings and Drive")
    page.locator("#quest-drive summary").click()
    shot(page, "drive", "Google Drive setup form", page.locator("#quest-drive-form"), "Recordings and Drive")
    page.context.close()

    recovery = review.page(paired + """sessionStorage.setItem('atmungsraum-quest-pending',
      JSON.stringify({id:'aaaaaaaaaaaaaaaa', op:'pause', generation:0,
        owner:{serial:'DEMO-QUEST',device:'demo-device'}}));""")
    shot(recovery, "unknown-receipt", "Unconfirmed action and manual check",
         recovery.locator("#quest-acknowledge"), "Recovery")
    recovery.context.close()

    recovery = review.page(suffix="#daten")
    expect(recovery.locator("#data-status")).to_contain_text("Keine früheren")
    shot(recovery, "legacy-empty", "Historical browser recordings · empty",
         recovery.locator("#data-status"), "Recordings and Drive")
    recovery.evaluate("""async () => {
      await new Promise((resolve, reject) => {
        const request = indexedDB.open('atmungsraum-session-data', 1);
        request.onupgradeneeded = () => request.result.createObjectStore('exports', {keyPath:'filename'});
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
          const db = request.result;
          const transaction = db.transaction('exports', 'readwrite');
          transaction.objectStore('exports').put({
            filename:'Tes_30_N_probe1.csv', csv:'synthetic,sample\\n1,0\\n',
            created_at:'2025-11-14T12:00:00Z'
          });
          transaction.oncomplete = () => {db.close(); resolve();};
          transaction.onabort = () => {db.close(); reject(transaction.error);};
        };
      });
    }""")
    recovery.reload(wait_until="load")
    shot(recovery, "legacy-saved", "Historical browser recordings · download",
         recovery.get_by_role("link", name="Tes_30_N_probe1.csv"), "Recordings and Drive")
    recovery.context.close()


def main():
    parser = ArgumentParser(description="Capture rendered Atmungsraum phone screens into a review ZIP.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this capture")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / "screens").mkdir()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        chromium = browser.version
        review = Review(browser, output)
        capture(review)
        browser.close()
    review.index()
    shutil.copyfile(ROOT / "for-ai" / "PHONE-SCREENSHOTS.md", output / "protocol.md")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    revision = git("rev-parse", "HEAD")
    changes = git("status", "--porcelain", "--", "*.js", "*.css", "*.html", "assets",
                  "tools/capture-phone-screens.py", "for-ai/PHONE-SCREENSHOTS.md")
    (output / "source.txt").write_text(
        f"App checkout: {revision}\nPlaywright: {version('playwright')}\nChromium: {chromium}\n"
        f"Source working tree:\n{changes or 'clean'}\n", encoding="utf-8"
    )
    with zipfile.ZipFile(output / ARCHIVE, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.rglob("*")):
            if file.is_file() and file.name != ARCHIVE:
                archive.write(file, file.relative_to(output))
    print(f"{len(review.rows)} screenshots → {output / ARCHIVE}")


if __name__ == "__main__":
    main()
