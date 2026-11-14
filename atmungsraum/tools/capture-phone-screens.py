from argparse import ArgumentParser
from html import escape
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit
import csv
import json
import shutil
import subprocess
import zipfile

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
URL = "https://viscereality.org/atmungsraum/"
ARCHIVE = "atmungsraum-smartphone-review.zip"
VIEWPORT = {"width": 390, "height": 844}

GATEWAY = """
// Every cache-versioned import shares one fixture, just as it reaches one Quest.
export const owner = window.reviewOwner ??= {
  serial: 'DEMO-QUEST', device: 'd'.repeat(32), epoch: '7', native_epoch: '8',
  ...window.reviewInitialOwner
};
const candidate = 'b'.repeat(12);
const settings = window.reviewQuest ??= {
  phase: 'IDLE', host: 'vr-preparation', polar: 'off', fresh: true, generation: 0,
  condition: 'condition-b', clock: 0, pending: false, strapCount: 0, selected: '',
  ...window.reviewInitialQuest
};
settings.setup ??= {
  v: 1, g: 1, r: 0, stage: 'testing', input: 'automatic', phase: 'inhale',
  position_frames: 0, cycle_frames: 705600, target_frames: 0, completed_frames: 0,
  attempt: 0, calibration: 'idle', failure: '', ready: false, playing: true,
  experiment_elapsed_ms: 0, input_phase: 'unknown', input_volume01: null,
  ...window.reviewInitialSetup
};
window.reviewSetupCommands ??= [];
const receipts = window.reviewReceipts ??= {};
let witness = 100;
const scenarios = ['probe-anleitung', 'probe-5', 'probe-10', 'probe-15', 'haupt-15'];
function snapshot() {
  const { phase, polar, generation, condition, clock, host, fresh, pending } = settings;
  const connected = ['stream-wait', 'ready'].includes(polar);
  const setup = { ...settings.setup };
  setup.phase = ['inhale', 'hold_in', 'exhale', 'hold_out'][
    Math.floor((setup.position_frames % setup.cycle_frames) / (setup.cycle_frames / 4))];
  return {
    v: 1, wifi_snapshot: 1, control_mode: 'remote',
    build: { v: 1, packaging_fingerprint: 'e'.repeat(64) }, p: phase, h: host, g: generation, r: 0,
    rec: phase !== 'IDLE' && phase !== 'ERROR', c: condition, ms: clock,
    pending, participant_arm: 1, ecg_preview: 1, recenter: 1, recordings: 1,
    breath_setup: settings.oldApk ? undefined : 1, setup: settings.oldApk ? undefined : setup,
    pi: { r: true, f: fresh }, a: ['track-ready', 'track-ready'],
    o: { v: 1, s: 'observed', e: owner.native_epoch, a: 5, i: String(witness++), g: generation, r: 0 },
    scenarios, y: polar === 'unsupported' ? undefined : {
      s: 'a'.repeat(32), g: Number(owner.native_epoch), r: 1, p: polar !== 'permission', b: polar !== 'bluetooth',
      c: connected, a: polar === 'ready', e: polar === 'ready' || Boolean(settings.ecg), sc: polar === 'scan',
      x: connected ? candidate : settings.selected,
      d: [candidate, 'c'.repeat(12)].slice(0, settings.strapCount)
    },
    q: { v: 2, s: 'a'.repeat(32), g: generation,
      o: [{ ARMED: 'start', RUNNING: 'pause', PAUSED: 'resume' }[phase]] }
  };
}
function command({ id, op, condition, d }) {
  settings.commands ??= [];
  settings.commands.push(op);
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
  return receipts[id] = { v: 1, id, state: 'confirmed' };
}
function breathSetup(command) {
  window.reviewSetupCommands.push(structuredClone(command));
  const { v, id, g, r, op } = command;
  const setup = settings.setup;
  if (v !== 1 || !/^[a-f0-9]{16}$/.test(id) || g !== setup.g || r !== setup.r
    || !['start', 'calibrate', 'controller'].includes(op)) {
    return receipts[id] = { v: 1, id, state: 'rejected' };
  }
  setup.r++;
  if (op === 'start') Object.assign(setup, { stage: 'testing', playing: true });
  if (op === 'calibrate') Object.assign(setup, {
    stage: 'waiting', input: 'automatic', ready: false, attempt: setup.attempt + 1,
    calibration: 'waiting', failure: '', target_frames: 1411200, completed_frames: 0
  });
  if (op === 'controller') Object.assign(setup, {
    stage: 'testing', input: 'controller-state', ready: true, calibration: 'idle',
    input_phase: 'hold', input_volume01: .5
  });
  const receipt = receipts[id] = { v: 1, id, state: 'confirmed' };
  if (settings.loseSetupReply) throw Error('Simulated lost setup reply');
  return receipt;
}
export const operationId = () => Array.from(crypto.getRandomValues(new Uint8Array(8)),
  value => value.toString(16).padStart(2, '0')).join('');
export function sameQuest(a, b) {
  return Boolean(a && b && ['device', 'serial', 'epoch', 'native_epoch'].every(key => a[key] === b[key]));
}
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
export async function rediscoverQuest() {
  window.reviewDiscovery?.push('bluetooth');
  return window.reviewDiscoveryState === 'ready'
    ? { state: 'ready', owner, endpoints: ['discovered'] }
    : window.reviewDiscoveryState ? { state: window.reviewDiscoveryState } : null;
}
export async function joinQuest({ endpoint }) {
  window.reviewDiscovery?.push(endpoint);
  if (endpoint === 'unreachable') throw Error('Simulated address change');
  if (window.reviewJoin === 'wait') await new Promise(() => {});
  if (window.reviewJoin === 'fail') throw Error('Simulated connection failure');
  return {
    owner: { ...owner }, initial: snapshot(), status: snapshot, command, participantArm: command, scenarioArm: command,
    breathSetup: settings.oldApk ? null : breathSetup, recenter: command,
    receipt: async id => settings.receiptState ? { v: 1, id, state: settings.receiptState }
      : receipts[id] ?? { v: 1, id, state: 'outcome_unknown' }, close() {},
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

PAIRED = ("{endpoint:'synthetic',serial:'DEMO-QUEST',device:'" + "d" * 32 + "',"
          "enrollment:{controller:'" + "a" * 32 + "',key:'" + "b" * 64 + "'}}")


class Review:
    def __init__(self, browser, output):
        self.browser = browser
        self.output = output
        self.rows = []
        self.checks = []
        self.errors = []

    def page(self, init="", suffix="", viewport=None):
        context = self.browser.new_context(
            viewport=viewport or VIEWPORT, device_scale_factor=2, is_mobile=True, has_touch=True,
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
            if path == "quest-gateway.js":
                request_route.fulfill(content_type="text/javascript", body=GATEWAY)
                return
            file = (ROOT / (path or "index.html")).resolve()
            if file.is_relative_to(ROOT) and file.is_file():
                request_route.fulfill(path=file)
            else:
                request_route.fulfill(status=404)

        context.route("**/*", route)
        page = context.new_page()
        page.on("pageerror", lambda error: self.errors.append(str(error)))
        page.on("console", lambda message: self.errors.append(message.text)
                if message.type == "warning" and "Text measurement unavailable" in message.text else None)
        page.clock.install()
        if init:
            page.add_init_script(init)
        page.goto(URL + suffix, wait_until="load")
        expect(page).to_have_title("Atmungsraum | Viscereality")
        return page

    def shot(self, page, slug, title, expected, group):
        expect(expected).to_be_visible(timeout=9000)
        expect(page.locator("html")).to_have_attribute("data-measurement", "ready")
        expect(page.locator(".screen:visible")).to_have_count(1)
        page.clock.run_for(100)
        expect(expected).to_be_visible()
        page.evaluate("window.scrollTo(0, 0)")
        page.clock.run_for(50)  # Paint the restored scroll position before capture.
        geometry = page.evaluate("""() => {
          const root = document.documentElement;
          const screen = document.querySelector('.screen:not([hidden])');
          const visible = selector => [...screen.querySelectorAll(selector)]
            .filter(node => node.getClientRects().length && node.checkVisibility());
          return {
            width: innerWidth, height: innerHeight, documentWidth: root.scrollWidth,
            documentHeight: root.scrollHeight, layout: root.dataset.layout,
            textOverflow: visible('[data-measure]').filter(node => {
              const box = node.getBoundingClientRect(); const range = document.createRange();
              range.selectNodeContents(node);
              return [...range.getClientRects()].some(rect => rect.left < box.left - 2 || rect.right > box.right + 2
                || rect.top < box.top - 2 || rect.bottom > box.bottom + 2);
            }).map(node => ({id: node.id, text: node.textContent.trim(), fit: node.dataset.textFit})),
            missingMeasurement: visible('[data-measure]').filter(node =>
              !['pretext', 'dom'].includes(node.dataset.textMeasurement)).map(node => node.id || node.textContent),
            pretextNoFit: visible('[data-measure]').filter(node => node.dataset.textFit !== 'fit')
              .map(node => ({id: node.id, text: node.textContent.trim(), fit: node.dataset.textFit})),
            smallTargets: visible('button, select, summary').filter(node => {
              const box = node.getBoundingClientRect(); return box.width < 47.9 || box.height < 47.9;
            }).map(node => node.id || node.tagName),
            panelOverflow: visible('.screen-body, .actions').filter(node =>
              node.scrollWidth > node.clientWidth + 2 || node.scrollHeight > node.clientHeight + 2)
              .map(node => node.className),
          };
        }""")
        assert geometry["documentWidth"] <= geometry["width"] + 1, f"{slug}: horizontal overflow {geometry}"
        assert not geometry["textOverflow"], f"{slug}: incomplete labels {geometry['textOverflow']}"
        assert not geometry["missingMeasurement"], f"{slug}: unmeasured labels {geometry['missingMeasurement']}"
        assert not geometry["smallTargets"], f"{slug}: small controls {geometry['smallTargets']}"
        assert not geometry["panelOverflow"], f"{slug}: internal panel overflow {geometry['panelOverflow']}"
        if geometry["layout"] == "bounded":
            assert geometry["documentHeight"] <= geometry["height"] + 2, f"{slug}: bounded page scrolls"
        assert not self.errors, f"{slug}: app errors {self.errors}"
        number = f"{len(self.rows) + 1:02d}"
        path = f"screens/{number}-{slug}.png"
        for element in (expected, page.locator(".screen:visible")):
            assert element.evaluate("""node => node.getBoundingClientRect().bottom + scrollY
                <= document.documentElement.scrollHeight + 1"""), f"{slug} exceeds capture bounds"
        page.screenshot(path=self.output / path, full_page=True)
        self.rows.append((number, group, title, path))
        self.checks.append({"screen": number, "slug": slug, **geometry})
        print(f"{number} {title}", flush=True)

    def index(self):
        (self.output / "checks.json").write_text(
            json.dumps({"screens": self.checks, "app_errors": self.errors}, indent=2), encoding="utf-8")
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
            '<p>390 × 844, 320 × 568 and 1440 × 900 CSS px, German interface; synthetic '
            'native Quest, participant, sensor, and recording data. '
            'Long screens are captured at full page height.</p>'
            '<p><a href="protocol.md">Screenshot protocol</a></p>'
            f'<div class="grid">{entries}</div></main></html>',
            encoding="utf-8",
        )


def wait_state(page, **state):
    page.evaluate("state => Object.assign(window.reviewQuest, state)", state)
    page.clock.run_for(1100)


def wait_setup(page, **setup):
    page.evaluate("setup => Object.assign(window.reviewQuest.setup, setup)", setup)
    page.clock.run_for(1100)


def enter_participant(page):
    participant = page.locator("#quest-participant")
    if participant.get_attribute("open") is None:
        participant.locator("summary").click()
    page.locator("#patient-surname").fill("Tes")
    page.locator("#patient-age").fill("30")
    page.get_by_role("radio", name="Keine Angabe").check()
    page.locator("#patient-next").focus()
    page.locator("#patient-next").press("Enter")


def assert_automatic_rediscovery(review, paired):
    page = review.page(paired + """
      const pair = JSON.parse(localStorage.getItem('atmungsraum-quest-pair'));
      pair.endpoint = 'unreachable';
      localStorage.setItem('atmungsraum-quest-pair', JSON.stringify(pair));
      window.reviewDiscovery = [];
      window.reviewDiscoveryState = 'bluetooth-off';
    """)
    try:
        expect(page.locator("#quest-status")).to_contain_text("Schalte Bluetooth ein")
        page.evaluate("window.reviewDiscoveryState = 'searching'")
        page.clock.run_for(2100)
        expect(page.locator("#quest-status")).to_contain_text("Suche die gespeicherte Quest", timeout=1000)
        page.evaluate("window.reviewDiscoveryState = 'ready'")
        page.clock.run_for(2100)
        expect(page.locator("#quest-participant")).to_be_visible(timeout=1000)
        expect(page.locator("#quest-find")).to_be_hidden()
        calls = page.evaluate("window.reviewDiscovery")
        assert calls[0] == "unreachable" and calls[-1] == "discovered", calls
        assert len(calls[1:-1]) >= 3 and set(calls[1:-1]) == {"bluetooth"}, calls
    finally:
        page.context.close()


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
    assert_automatic_rediscovery(review, paired)
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
    detail = page.locator("#preview-detail")
    calibrate = page.locator("#preview-calibrate")
    fallback = page.locator("#preview-controller")
    proceed = page.locator("#preview-continue")
    expect(preview).to_have_text("Atemtakt: Einatmen")
    expect(page.locator("#preview-input")).to_contain_text("Automatic")
    expect(proceed).to_be_disabled()
    shot(page, "native-automatic", "Native Atemtakt · Automatic", preview, "Breathing setup")
    radius = page.locator("#preview-circle").get_attribute("style")
    page.clock.run_for(8500)
    expect(preview).to_have_text("Atemtakt: Einatmen")
    expect(page.locator("#preview-circle")).to_have_attribute("style", radius)
    expect(page.locator("#preview-title")).to_be_focused()
    page.keyboard.press("Tab")
    expect(calibrate).to_be_focused()
    page.keyboard.press("Enter")
    expect(detail).to_contain_text("nächsten vollständigen Atemzyklus")
    shot(page, "native-waiting", "Calibration waits for the next native cycle", detail, "Breathing setup")
    wait_setup(page, stage="calibrating", calibration="collecting", position_frames=705600)
    expect(detail).to_contain_text("zwei vollständige Atemzyklen")
    expect(page.locator("#preview-progress")).to_have_attribute("value", "0")
    shot(page, "native-collecting", "Native calibration · collecting", detail, "Breathing setup")
    wait_setup(page, position_frames=1411200, completed_frames=705600)
    expect(page.locator("#preview-progress")).to_have_attribute("value", "1")
    expect(proceed).to_be_disabled()
    shot(page, "native-half", "Native calibration · one of two cycles", detail, "Breathing setup")
    wait_setup(page, r=2, stage="testing", calibration="ready", completed_frames=1411200,
               position_frames=2293200, input="polar-tilt", ready=True,
               input_phase="exhale", input_volume01=.23)
    expect(preview).to_have_text("Atemtakt: Atem halten")
    expect(page.locator("#preview-input")).to_contain_text("Erkannt: Ausatmen · 23 %")
    expect(proceed).to_be_enabled()
    shot(page, "native-ready", "Polar ready · measured input differs from intended hold", proceed, "Breathing setup")
    calibrate.focus()
    calibrate.press("Enter")
    expect(detail).to_contain_text("nächsten vollständigen Atemzyklus")
    wait_setup(page, r=4, stage="testing", calibration="failed", failure="insufficient-range",
               completed_frames=1411200, position_frames=3528000, input_phase="unknown", input_volume01=None)
    expect(detail).to_contain_text("insufficient-range")
    expect(calibrate).to_be_enabled()
    shot(page, "native-failure", "Native calibration failed · retry available", detail, "Breathing setup")
    wait_state(page, fresh=False, polar="stream-wait")
    expect(fallback).to_be_enabled()
    expect(proceed).to_be_disabled()
    shot(page, "native-failure-stale", "Stale Polar data · Controller State remains available", fallback, "Breathing setup")
    before = page.evaluate("({g:reviewQuest.setup.g,position:reviewQuest.setup.position_frames})")
    fallback.focus()
    fallback.press("Enter")
    expect(page.locator("#preview-input")).to_contain_text("Controller State")
    expect(proceed).to_be_disabled()
    assert before == page.evaluate("({g:reviewQuest.setup.g,position:reviewQuest.setup.position_frames})")
    shot(page, "native-controller-stale", "Controller fallback · H10 freshness still gates Continue", detail, "Breathing setup")
    wait_state(page, fresh=True, polar="ready")
    expect(proceed).to_be_enabled()
    shot(page, "native-controller-ready", "Controller State · fixed orientation and continuation", proceed, "Breathing setup")
    proceed.focus()
    proceed.press("Enter")
    expect(page.locator("#quest-condition")).to_be_focused()

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
    expect(play).to_have_text("Starten")
    shot(page, "ready-to-start", "Ready to start the selected run", play, "Playback")
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
    play.click()
    expect(play).to_have_text("Pause")
    expect(status).to_contain_text("Probelauf läuft")
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
    assert page.evaluate("window.reviewQuest.commands.slice(-5)") == ["arm", "start", "pause", "resume", "save-next"]
    shot(page, "finished", "Run finished and returned to native input testing", detail, "Playback")
    expect(proceed).to_be_enabled()
    proceed.click()

    await_archive = page.locator("#quest-export-download")
    page.evaluate("""async () => {
      const { openQuestRecordingStore } = await import('./quest-recording-store.js?v=4');
      const store = await openQuestRecordingStore(window.reviewOwner);
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
        owner:{serial:'DEMO-QUEST',device:'""" + "d" * 32 + """',epoch:'7',native_epoch:'8'}}));""")
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


def capture_setup_recovery(review, paired):
    owner = {"serial": "DEMO-QUEST", "device": "d" * 32, "epoch": "7", "native_epoch": "8"}
    journal = {"owner": owner, "command": {"v": 1, "id": "f" * 16, "g": 1, "r": 0, "op": "calibrate"}}
    initial = paired + "window.reviewInitialQuest={polar:'ready',receiptState:'outcome_unknown'};"
    saved = """if (!sessionStorage.getItem('review-journal-created')) {
      sessionStorage.setItem('atmungsraum-breathing-setup-pending', JSON.stringify(%s));
      sessionStorage.setItem('review-journal-created','1');
    }""" % json.dumps(journal)

    page = review.page(initial + saved)
    detail = page.locator("#preview-detail")
    expect(detail).to_contain_text("Ergebnis unklar")
    expect(page.locator("#patient-surname")).to_have_value("")
    page.reload(wait_until="load")
    expect(detail).to_be_visible()
    expect(detail).to_contain_text("Ergebnis unklar")
    expect(page.locator("#preview-continue")).to_be_disabled()
    expect(page.locator("#preview-acknowledge")).to_be_hidden()
    assert page.evaluate("JSON.parse(sessionStorage.getItem('atmungsraum-breathing-setup-pending')).command.id") == "f" * 16
    assert page.evaluate("window.reviewSetupCommands.length") == 0
    review.shot(page, "setup-reload-pending", "Reload · uncertain setup with blank participant ID", detail, "Setup recovery")
    wait_state(page, receiptState="confirmed")
    wait_setup(page, r=1, stage="waiting", calibration="waiting", attempt=1, target_frames=1411200)
    expect(page.locator("#breathing-preview")).to_be_hidden()
    expect(page.locator("#patient-fields")).to_be_enabled()
    assert page.evaluate("sessionStorage.getItem('atmungsraum-breathing-setup-pending')") is None
    assert page.evaluate("window.reviewSetupCommands.length") == 0
    review.shot(page, "setup-reload-confirmed", "Reload · receipt resolves without resending", page.locator("#quest-participant"), "Setup recovery")
    page.context.close()

    for slug, title, renewal in (
        ("gateway", "Renewed gateway epoch", "window.reviewInitialOwner={epoch:'9'};"),
        ("native", "Replaced native owner", "window.reviewInitialOwner={epoch:'9',native_epoch:'10'};"),
        ("audio", "Replaced native audio generation", "window.reviewInitialSetup={g:2};"),
    ):
        page = review.page(initial + saved + renewal)
        acknowledge = page.locator("#preview-acknowledge")
        expect(page.locator("#patient-surname")).to_have_value("")
        expect(acknowledge).to_be_visible()
        expect(acknowledge).to_be_enabled()
        expect(page.locator("#preview-continue")).to_be_disabled()
        review.shot(page, f"setup-{slug}-retired", f"{title} · explicit recovery before participant entry", acknowledge, "Setup recovery")
        acknowledge.focus()
        acknowledge.press("Enter")
        expect(page.locator("#breathing-preview")).to_be_hidden()
        expect(page.locator("#quest-status")).to_be_focused()
        expect(page.locator("#patient-fields")).to_be_enabled()
        assert page.evaluate("sessionStorage.getItem('atmungsraum-breathing-setup-pending')") is None
        assert page.evaluate("window.reviewSetupCommands.length") == 0
        review.shot(page, f"setup-{slug}-recovered", f"{title} · participant entry restored without replay", page.locator("#quest-participant"), "Setup recovery")
        page.context.close()

    page = review.page(paired + "window.reviewInitialQuest={polar:'ready',oldApk:true};")
    enter_participant(page)
    expect(page.locator("#preview-detail")).to_contain_text("Quest-App aktualisieren")
    expect(page.locator("#preview-calibrate")).to_be_disabled()
    expect(page.locator("#preview-controller")).to_be_disabled()
    expect(page.locator("#preview-continue")).to_be_disabled()
    review.shot(page, "setup-old-apk", "Older APK · explicit update required", page.locator("#preview-detail"), "Setup recovery")
    page.context.close()


def capture_accessibility(review, paired):
    initial = paired + """window.reviewInitialQuest={polar:'ready'};
      window.reviewInitialSetup={r:3,input:'polar-tilt',ready:true,calibration:'ready',attempt:1,
        target_frames:1411200,completed_frames:1411200,position_frames:529200,
        input_phase:'inhale',input_volume01:.62};"""
    for slug, title, viewport, text_size, spacing in (
        ("320-reflow", "320 px · complete native setup labels", {"width": 320, "height": 568}, "", False),
        ("200-text", "320 px · 200% text and document reflow", {"width": 320, "height": 568}, "200%", False),
        ("text-spacing", "390 px · WCAG text spacing", VIEWPORT, "", True),
        ("desktop", "Desktop · native input testing", {"width": 1440, "height": 900}, "", False),
    ):
        page = review.page(initial, viewport=viewport)
        enter_participant(page)
        expect(page.locator("#preview-continue")).to_be_enabled()
        if spacing:
            page.add_style_tag(content="""* {line-height:1.5 !important;letter-spacing:.12em !important;word-spacing:.16em !important;}
              p {margin-bottom:2em !important;}""")
        page.evaluate("value => {document.documentElement.style.fontSize=value; document.documentElement.style.setProperty('--review-check','1');}", text_size)
        page.clock.run_for(200)
        review.shot(page, slug, title, page.locator("#preview-input"), "Accessibility and reflow")
        if spacing:
            expect(page.locator("#preview-input")).to_have_attribute("data-text-measurement", "dom")
        proceed = page.locator("#preview-continue")
        proceed.focus()
        proceed.press("Enter")
        expect(page.locator("#quest-condition")).to_be_visible()
        expect(page.locator("#quest-condition")).to_be_focused()
        page.context.close()

    page = review.page(initial)
    enter_participant(page)
    wait_setup(page, r=4, input="automatic", ready=False, calibration="failed", input_phase="bad-tracking",
               failure="insufficient-range-" + "Sensorbewegung-unzureichend-" * 7)
    expect(page.locator("#preview-detail")).to_contain_text("Einrichtung fehlgeschlagen")
    review.shot(page, "long-failure", "Long native failure reason · readable reflow", page.locator("#preview-detail"), "Accessibility and reflow")
    page.context.close()


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
        paired = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({PAIRED}));"
        capture_setup_recovery(review, paired)
        capture_accessibility(review, paired)
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
    print(f"{len(review.rows)} screenshots -> {output / ARCHIVE}")


if __name__ == "__main__":
    main()
