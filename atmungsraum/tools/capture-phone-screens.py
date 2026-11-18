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
from release_fixtures import apk_fixture


ROOT = Path(__file__).resolve().parents[1]
URL = "https://viscereality.org/atmungsraum/"
ARCHIVE = "atmungsraum-smartphone-review.zip"
VIEWPORT = {"width": 390, "height": 844}
DEFAULT_APK = apk_fixture()

GATEWAY = """
// Keep the actual pure access contract; replace transport, never its UI gates.
import { canControlQuest } from './quest-gateway.js?review-contract=1';
export { canControlQuest, companionAccessDenied, controllerAccessState, readJson } from './quest-gateway.js?review-contract=1';
// Every cache-versioned import shares one fixture, just as it reaches one Quest.
export const owner = window.reviewOwner ??= {
  serial: 'DEMO-QUEST', device: 'd'.repeat(32), epoch: '7', native_epoch: '8',
  ...window.reviewInitialOwner
};
const candidate = 'b'.repeat(12);
const settings = window.reviewQuest ??= {
  phase: 'IDLE', host: 'vr-preparation', polar: 'off', fresh: true, generation: 0,
  condition: 'condition-b', clock: 0, pending: false, strapCount: 0, selected: '',
  healthSupported: true,
  apk: __DEFAULT_APK__,
  ...window.reviewInitialQuest
};
settings.setup ??= {
  v: 1, g: 1, r: 0, stage: window.reviewInitialSetup ? 'testing' : 'idle', input: 'automatic', phase: 'inhale',
  position_frames: 0, cycle_frames: 705600, target_frames: 0, completed_frames: 0,
  attempt: 0, calibration: 'idle', failure: '', ready: false, playing: Boolean(window.reviewInitialSetup),
  audio_state: window.reviewInitialSetup ? 'playing' : 'paused', audio_generation: 1,
  experiment_elapsed_ms: 0, input_phase: 'unknown', input_volume01: null,
  ...(!window.reviewInitialSetup ? {parked:true,input_volume01:0} : {}),
  ...window.reviewInitialSetup
};
window.reviewSetupCommands ??= [];
window.reviewAccessCommands ??= [];
window.reviewRecenterCommands ??= [];
const receipts = window.reviewReceipts ??= {};
let witness = 100;
let lastWitness = '';
const ecgStarted = performance.now();
const scenarios = ['probe-anleitung', 'probe-5', 'probe-10', 'probe-15', 'haupt-15'];
function snapshot() {
  window.reviewSnapshotCount = (window.reviewSnapshotCount ?? 0) + 1;
  const { phase, polar, generation, condition, clock, host, fresh, pending } = settings;
  const connected = ['stream-wait', 'ready'].includes(polar);
  const setup = { ...settings.setup };
  // Synthetic capability fixture with an explicit simulated contact report.
  // The production projection must never derive contact from sample traffic.
  const polarHealth = Object.hasOwn(settings,'polarHealth') ? settings.polarHealth
    : settings.healthSupported && !settings.oldApk ? {
      v:1,link:connected,connection_generation:String(settings.connectionGeneration??1),
      observed_at_ns:settings.freezeHealth && window.reviewHealthWitness ? window.reviewHealthWitness
        : (window.reviewHealthWitness=String(Math.max(1,Math.floor(performance.now()*1000000)))),
      ecg:{age_ms:settings.ecgAge??0,fresh:settings.ecg??polar==='ready'},
      acc:{age_ms:settings.accAge??0,fresh:settings.acc??polar==='ready'},
      contact:settings.contactHealth??{state:'detected',supported:true,detected:true,age_ms:0},
      battery:{percent:settings.polarBattery??64,age_ms:settings.polarBatteryAge??0}
    } : undefined;
  if(settings.advanceRenderer && setup.renderer) {
    const frame=Math.floor(performance.now()/100)+1;
    setup.renderer={...setup.renderer,frame,observed_ns:String(frame*100000000),age_ns:0};
  }
  setup.phase = ['inhale', 'hold_in', 'exhale', 'hold_out'][
    Math.floor((setup.position_frames % setup.cycle_frames) / (setup.cycle_frames / 4))];
  return {
    v: 1, wifi_snapshot: 1, control_mode: 'remote',
    build: { v: 1, packaging_fingerprint: 'e'.repeat(64) }, p: phase, h: host, g: generation, r: 0,
    rec: phase !== 'IDLE' && phase !== 'ERROR', c: condition, ms: clock,
    ...(Number.isSafeInteger(settings.duration_ms) ? {duration_ms: settings.duration_ms} : {}),
    ...(settings.probeInstructionPlaying ? {pre_session_instruction: 'probe-5'} : {}),
    pending, participant_arm: settings.participantArm === false ? undefined : 1, ecg_preview: 1, recenter: 1, recordings: 1,
    ...(settings.liveRecording ? {live_recording: 1} : {}),
    controller_access: settings.access,
    polar_health: polarHealth,
    power:Object.hasOwn(settings,'power') ? settings.power
      : {v:1,quest:{percent:82,charging:false},controllers:[]},
    ...(Object.hasOwn(settings,'apk') ? {apk:settings.apk} : {}),
    ...(Object.hasOwn(window,'reviewMetaUpdate') ? {meta_update:window.reviewMetaUpdate}
      : settings.metaUpdate ? {meta_update:settings.metaUpdate} : {}),
    ...(settings.metaUpdate ? {meta_update_control:1} : {}),
    ...(settings.kiosk ? {kiosk:settings.kiosk} : {}),
    breath_setup: settings.oldApk ? undefined : 1, setup: settings.oldApk ? undefined : setup,
    pi: { r: settings.participantArm !== false, f: fresh }, a: settings.tracks ?? ['track-ready', 'track-ready'],
    o: { v: 1, s: 'observed', e: owner.native_epoch, a: 5,
      i: settings.freezeObservation && lastWitness ? lastWitness : (lastWitness = String(witness++)), g: generation, r: 0 },
    scenarios: Object.hasOwn(settings, 'scenarios') ? settings.scenarios ?? undefined : scenarios,
    y: polar === 'unsupported' ? undefined : {
      s: 'a'.repeat(32), g: Number(owner.native_epoch), r: 1, p: polar !== 'permission', b: polar !== 'bluetooth',
      c: connected, a: settings.acc ?? polar === 'ready', e: settings.ecg ?? polar === 'ready', sc: polar === 'scan',
      ...(settings.location === undefined ? {} : { l: settings.location }),
      ...(settings.autoRecovery === undefined ? {} : { ar: settings.autoRecovery }),
      ...(settings.automation === undefined ? {} : { auto: settings.automation }),
      x: connected ? candidate : settings.selected,
      d: [candidate, 'c'.repeat(12)].slice(0, settings.strapCount)
    },
    q: { v: 2, s: 'a'.repeat(32), g: generation,
      o: [{ ARMED: 'start', RUNNING: 'pause', PAUSED: 'resume' }[phase]] }
  };
}
function ecgPacket() {
  window.reviewEcgCalls = (window.reviewEcgCalls ?? 0) + 1;
  if (!(settings.ecg ?? settings.polar === 'ready')) return null;
  if (settings.ecgReplay && window.reviewLastEcg) return window.reviewLastEcg;
  // Absolute native sample indices preserve every overlapping timestamp/value.
  // The virtual browser clock advances the delivered 130 Hz window at 1 Hz.
  const first = Math.floor((performance.now() - ecgStarted) * 130 / 1000);
  if (settings.ecgGap === 'first-packet') settings.ecgGap = [first + 90, first + 110];
  const packet = window.reviewLastEcg = {
    v: 1, unit: 'uV', rate: 130, age_ms: 0,
    samples: Array.from({ length: 260 }, (_, offset) => first + offset)
      .filter(index => !settings.ecgGap || index < settings.ecgGap[0] || index >= settings.ecgGap[1])
      .map(index => {
        const cycle = index % 95;
        const beat = 310 * Math.exp(-((cycle - 36) ** 2) / 2)
          - 120 * Math.exp(-((cycle - 40) ** 2) / 3)
          + 40 * Math.exp(-((cycle - 52) ** 2) / 36);
        return [String(1000000000 + Math.floor(index * 1000000000 / 130)),
          Math.round(beat + 12 * Math.sin(index / 11))];
      })
  };
  window.reviewEcgPackets = [...(window.reviewEcgPackets ?? []), packet].slice(-3);
  return packet;
}
async function observedStatus() {
  window.reviewStatusReads=(window.reviewStatusReads??0)+1;
  window.reviewStatusInFlight=(window.reviewStatusInFlight??0)+1;
  window.reviewStatusMaxInFlight=Math.max(window.reviewStatusMaxInFlight??0,window.reviewStatusInFlight);
  try {
    const current = snapshot();
    if (settings.statusDrop) throw Error('Simulated transient status loss');
    if (settings.statusDelay) await new Promise(resolve => setTimeout(resolve, settings.statusDelay));
    return current;
  } finally { window.reviewStatusInFlight--; }
}
function command({ id, op, condition, d, scenario }) {
  settings.commands ??= [];
  settings.commands.push(op);
  const states = { arm: 'ARMED', start: 'RUNNING', pause: 'PAUSED', resume: 'RUNNING', 'save-next': 'IDLE' };
  if (op === 'arm') {
    settings.generation++;
    settings.condition = condition;
    settings.armedScenario = scenario;
    settings.host = 'vr-session';
  }
  const guidedStart = op === 'start' && settings.armedScenario === 'probe-5' && settings.probeInstructionMs;
  if (op === 'start' && (guidedStart || settings.startDelayMs)) {
    settings.pending = true;
    if (guidedStart) {
      settings.probeInstructionPlaying = true;
      settings.duration_ms = 300000;
      settings.clock = 0;
    }
    setTimeout(() => {
      settings.pending = false;
      settings.probeInstructionPlaying = false;
      settings.phase = 'RUNNING';
      settings.clock = guidedStart ? 0 : 37000;
      receipts[id] = { v: 1, id, state: 'confirmed' };
    }, guidedStart || settings.startDelayMs);
    return receipts[id] = { v: 1, id, state: 'pending' };
  }
  if (op === 'start') settings.clock = 37000;
  if (op === 'select') settings.selected = d;
  if (op === 'connect' && settings.connectTimeoutOnce) {
    settings.connectTimeoutOnce = false;
    settings.automation = {v:1,state:'connecting',retry_ms:0,reason:''};
    setTimeout(() => {
      receipts[id] = {v:1,id,state:'rejected',reason:'connection-timeout'};
      settings.automation = {v:1,state:'backoff',retry_ms:2000,reason:'connection-timeout'};
      setTimeout(() => {
        settings.polar='stream-wait';
        settings.automation={v:1,state:'streams',retry_ms:0,reason:''};
        setTimeout(() => {settings.polar='ready';settings.automation={v:1,state:'ready',retry_ms:0,reason:''};},1000);
      },2000);
    },15000);
    return receipts[id] = {v:1,id,state:'pending'};
  }
  if (op === 'connect') settings.polar = 'stream-wait';
  if (op === 'start-acc') { settings.acc = true; settings.polar = 'ready'; }
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
  if (settings.calibrationInstruction && op === 'cancel-instruction') {
    const target = command.target;
    const matches = Boolean(target && target === setup.instruction_id);
    if (matches) {
      delete setup.instruction_id;
      settings.pending = false;
      receipts[target] = {v:1,id:target,state:'rejected',accepted:false,reason:'calibration-instruction-cancelled'};
    }
    return receipts[id] = {v:1,id,state:matches ? 'confirmed' : 'rejected',accepted:matches,
      reason:matches ? 'calibration-instruction-cancelled' : 'calibration-instruction-not-pending'};
  }
  if (settings.staleSetupRejects > 0) {
    settings.staleSetupRejects--;
    setup.r++;
    if (settings.changeInputOnStale) setup.input = settings.changeInputOnStale;
    const rejected = receipts[id] = {v:1,id,state:'rejected',accepted:false,reason:'stale-revision'};
    if (settings.loseStaleSetupReply) throw Error('Lost explicit rejection');
    return rejected;
  }
  if (v !== 1 || !/^[a-f0-9]{16}$/.test(id) || g !== setup.g || r !== setup.r
    || !['start', 'calibrate', 'controller', 'polar-fixed', 'polar-adaptive'].includes(op)) {
    return receipts[id] = { v: 1, id, state: 'rejected' };
  }
  if (settings.audioRecovery) {
    Object.assign(setup, { stage:'recovering',ready:false,clock_valid:false,audio_recovery_available:false,
      recovery:{id,state:'pending',deadline_ns:2000000000,target_generation:null} });
    return receipts[id] = {v:1,id,state:'pending',accepted:true,reason:'audio-recovery-pending',recovery_generation:null};
  }
  if (settings.deferSetup) return receipts[id] = {v:1,id,state:'pending'};
  if (settings.calibrationInstruction && op === 'calibrate' && setup.attempt === 0) {
    setup.instruction_id = id;
    settings.pending = true;
    return receipts[id] = {v:1,id,state:'pending',accepted:true,reason:'calibration-instruction-pending'};
  }
  setup.r++;
  const parked = setup.stage === 'idle' && !setup.playing;
  if (op === 'start') Object.assign(setup, { stage: 'testing', playing: true, audio_state:'playing', parked:false });
  if (op === 'calibrate') Object.assign(setup, {
    stage: parked ? 'calibrating' : 'waiting', input: 'automatic', ready: false, attempt: setup.attempt + 1,
    playing:true,audio_state:'playing',parked:false,
    calibration: parked ? 'collecting' : 'waiting', failure: '', target_frames: 1411200, completed_frames: 0
  });
  if (op === 'controller') Object.assign(setup, {
    stage: 'testing', input: 'controller-state', ready: true, calibration: 'idle', playing:true,audio_state:'playing',parked:false,
    input_phase: 'hold', input_volume01: .5
  });
  if (['polar-fixed','polar-adaptive'].includes(op)) Object.assign(setup, {
    stage:'testing',input:op === 'polar-fixed' ? 'polar-tilt-fixed' : 'polar-tilt-adaptive',
    ready:true,calibration:'idle',playing:true,audio_state:'playing',parked:false,
    input_phase:'hold',input_volume01:.5
  });
  const receipt = receipts[id] = { v: 1, id, state: 'confirmed' };
  if (settings.loseSetupReply) throw Error('Simulated lost setup reply');
  return receipt;
}
export const operationId = () => {
  window.reviewOperationCounter = (window.reviewOperationCounter ?? 0) + 1;
  return window.reviewOperationCounter.toString(16).padStart(16, '0');
};
export function questEndpoint(input) {
  if (input !== 'http://192.168.2.48:8766/v1/gateway') throw Error('Unexpected fixture endpoint');
  return input;
}
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
  return { owner, endpoints: ['synthetic'], ...(window.reviewPairingEnabled ? { pairing: 1 } : {}) };
}
export async function rediscoverQuest() {
  window.reviewDiscovery?.push('bluetooth');
  return window.reviewDiscoveryState === 'ready'
    ? { state: 'ready', owner, endpoints: ['discovered'], ...(window.reviewPairingEnabled ? { pairing: 1 } : {}) }
    : window.reviewDiscoveryState ? { state: window.reviewDiscoveryState } : null;
}
export async function joinQuest({ endpoint }) {
  window.reviewDiscovery?.push(endpoint);
  if (endpoint === 'unreachable') throw Error('Simulated address change');
  if (settings.failJoinAfterPairing && window.reviewPairing?.state === 'approved' && !settings.pairJoinFailed) {
    settings.pairJoinFailed = true;
    throw Error('Simulated lost reply after pairing approval');
  }
  if (window.reviewJoin === 'wait') await new Promise(() => {});
  if (window.reviewJoin === 'fail') throw Error('Simulated connection failure');
  if (settings.joinDelay) await new Promise(resolve => setTimeout(resolve, settings.joinDelay));
  const controlled = run => value => {
    if (!canControlQuest(snapshot())) throw Error('Fixture controller is read-only while waiting');
    return run(value);
  };
  const controllerAccess = value => {
    window.reviewAccessCommands.push(structuredClone(value));
    const access = settings.access;
    if (value.v !== 1 || !/^[a-f0-9]{16}$/.test(value.id)
      || value.expected_control_revision !== access.revision
      || !['claim', 'release'].includes(value.op)
      || value.op === 'claim' && (access.role !== 'waiting' || access.active_phone)
      || value.op === 'release' && access.role !== 'active') {
      throw Error('Fixture controller-access rejected');
    }
    if (settings.controlNoGrant && value.op === 'claim') {
      if (settings.loseControlReply) throw Error('Fixture lost control response');
      return structuredClone(access);
    }
    Object.assign(access, { role: value.op === 'claim' ? 'active' : 'waiting',
      active_phone: value.op === 'claim' ? 'Fixture-Smartphone' : '', revision: access.revision + 1 });
    if (settings.loseControlReply) throw Error('Fixture lost control response');
    return structuredClone(access);
  };
  return {
    owner: { ...owner }, initial: snapshot(), status: observedStatus, command: controlled(command),
    participantArm: controlled(command), scenarioArm: controlled(command),
    breathSetup: settings.oldApk ? null : controlled(breathSetup),
    breathSetupReceipt: settings.oldApk ? null : controlled(id => {
      (window.reviewSetupReceiptQueries ??= []).push({v:1,id,op:'receipt'});
      return settings.setupReceiptPatch ? {v:1,id,...settings.setupReceiptPatch}
        : settings.receiptState ? {v:1,id,state:settings.receiptState}
          : receipts[id] ?? {v:1,id,state:'outcome_unknown'};
    }),
    kiosk: settings.kiosk ? controlled(value => {
      (window.reviewKioskCommands ??= []).push(structuredClone(value));
      if(value.op === 'receipt') return settings.kioskReceipt ?? {id:value.id,state:'pending',detail:'Waiting'};
      return {id:value.id,state:settings.kioskReply ?? 'pending',detail:'Service restart pending'};
    }) : null,
    metaUpdate: settings.metaUpdate ? controlled(value => {
      (window.reviewMetaUpdateCommands ??= []).push(structuredClone(value));
      if(value.op === 'receipt') return receipts[value.id] ?? {v:1,id:value.id,state:'rejected',cycle:settings.metaUpdate.cycle,
        revision:settings.metaUpdate.revision,attempt:null,reason:'unknown-id'};
      const current=settings.metaUpdate;
      if(value.cycle !== current.cycle || value.revision !== current.revision || !current.can_check) {
        return {v:1,id:value.id,state:'stale',cycle:current.cycle,revision:current.revision,attempt:null,reason:'changed'};
      }
      Object.assign(current,{state:'checking',revision:String(BigInt(current.revision)+1n),attempt:'777',can_check:false,session_allowed:false});
      const result=receipts[value.id]={v:1,id:value.id,state:'accepted',cycle:current.cycle,revision:current.revision,attempt:current.attempt,reason:''};
      if(settings.lostUpdateReply) throw Error('fixture lost reply after native admission');
      return result;
    }) : null,
    recenter: controlled(value => {
      window.reviewRecenterCommands.push(structuredClone(value));
      return receipts[value.id] = { v: 1, id: value.id, state: 'confirmed' };
    }),
    controllerAccess: settings.access ? controllerAccess : null,
    receipt: async id => settings.receiptState ? { v: 1, id, state: settings.receiptState }
      : receipts[id] ?? { v: 1, id, state: 'outcome_unknown' }, close() {},
    recordings: async () => ({ v: 1, state: 'ready', entries: [], next_before: null }),
    ecg: async () => ecgPacket()
  };
}
"""

GATEWAY = GATEWAY.replace('__DEFAULT_APK__', json.dumps(DEFAULT_APK))

PAIRING = """
// Synthetic transport only: no key generation, derivation or HTTP; tickets are fixed dummy data.
export async function pairPhone({ endpoint, owner, signal, onCode }) {
  signal?.throwIfAborted();
  window.reviewPairingCalls ??= [];
  window.reviewPairingCalls.push('begin');
  return new Promise((resolve, reject) => {
    let complete = false;
    const fixture = window.reviewPairing = {
      code: '314159', state: 'pending', phoneConfirmed: false, headsetConfirmed: false,
      finish(state) {
        if (complete) return;
        if (state === 'rejected') {
          complete = true; fixture.state = state; cleanup();
          window.reviewPairingCalls.push('rejected');
          reject(Error('Die Verbindung wurde im Headset abgelehnt.'));
        } else if (state === 'approved') {
          fixture.headsetConfirmed = true;
          settle();
        }
      },
    };
    const cleanup = () => signal?.removeEventListener('abort', aborted);
    const aborted = () => {
      if (complete) return;
      complete = true; fixture.state = 'cancelled'; cleanup();
      window.reviewPairingCalls.push('cancelled');
      reject(signal.reason ?? new DOMException('Abgebrochen', 'AbortError'));
    };
    const settle = () => {
      if (complete || !fixture.phoneConfirmed || !fixture.headsetConfirmed) return;
      complete = true; fixture.state = 'approved'; cleanup();
      window.reviewPairingCalls.push('approved');
      resolve({ ...owner, v: 1, controller: 'a'.repeat(32), key: 'b'.repeat(64) });
    };
    signal?.addEventListener('abort', aborted, { once: true });
    Promise.resolve().then(() => onCode({ code: fixture.code, signal, pairingId: 'c'.repeat(32) }))
      .then(confirmed => {
        if (complete) return;
        if (confirmed !== true) { aborted(); return; }
        fixture.phoneConfirmed = true;
        window.reviewPairingCalls.push('phone-confirmed');
        settle();
      }, error => { if (!complete) { complete = true; cleanup(); reject(error); } });
  });
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
        self.regressions = []
        self.errors = []
        self.tiles = {}
        self.release_policy_failure = False

    def capture_document(self,page,path):
        # Mobile full-page capture can paint temporary bounded layout with a
        # paused clock. Actual viewport tiles keep device metrics unchanged.
        path=Path(path)
        size=page.evaluate('({height:innerHeight,total:document.documentElement.scrollHeight,scroll:scrollY})')
        offsets=list(range(0,max(1,size['total']-size['height']+1),max(1,size['height']-40)))
        if offsets[-1] < size['total']-size['height']:
            offsets.append(size['total']-size['height'])
        files=[]
        tile_checks=[]
        for i,offset in enumerate(offsets):
            page.evaluate('y=>scrollTo(0,y)',offset)
            page.clock.run_for(50)
            hit_test=page.evaluate('''() => {
              const modal=document.querySelector('dialog[open]');
              const root=modal||document.querySelector('.screen:not([hidden])');
              const visible=n=>n.getClientRects().length&&n.checkVisibility();
              const corners=modal?[]:[...document.querySelectorAll('#recenter-vr,#corner-actions')].filter(visible);
              const targets=[...root.querySelectorAll('button,input,select,summary,a[href],.gender-options label')].filter(visible);
              const occluded=[],missed=[];
              for(const target of targets){
                const r=target.getBoundingClientRect();
                for(const corner of corners){
                  if(corner.contains(target))continue;
                  const c=corner.getBoundingClientRect();
                  const x1=Math.max(r.left,c.left,0),x2=Math.min(r.right,c.right,innerWidth);
                  const y1=Math.max(r.top,c.top,0),y2=Math.min(r.bottom,c.bottom,innerHeight);
                  if(x2-x1>1&&y2-y1>1)occluded.push({target:target.id||target.textContent.trim(),corner:corner.id});
                }
                const x=(r.left+r.right)/2,y=(r.top+r.bottom)/2;
                if(x<0||x>=innerWidth||y<0||y>=innerHeight||target.disabled)continue;
                const hit=document.elementFromPoint(x,y);
                if(hit&&!target.contains(hit)&&hit!==target&&!hit.contains(target))missed.push({target:target.id||target.tagName,hit:hit.id||hit.tagName});
              }
              return {scroll:scrollY,width:innerWidth,occluded,missed};
            }''')
            target=path if i==0 else path.with_name(f'{path.stem}-tile-{i+1:02d}.png')
            page.screenshot(path=target)
            files.append(target.relative_to(self.output).as_posix())
            tile_checks.append(hit_test)
            assert not hit_test['occluded'], f'{path.stem}: corner covers controls at tile {i+1}: {hit_test}'
            assert not hit_test['missed'], f'{path.stem}: control hit test failed at tile {i+1}: {hit_test}'
        page.evaluate('y=>scrollTo(0,y)',size['scroll'])
        page.clock.run_for(50)
        self.tiles[path.relative_to(self.output).as_posix()]=files
        (self.output / f'{path.stem}-tile-checks.json').write_text(json.dumps(tile_checks,indent=2),encoding='utf-8')

    def page(self, init="", suffix="", viewport=None, reduced_motion="reduce"):
        context = self.browser.new_context(
            viewport=viewport or VIEWPORT, device_scale_factor=2, is_mobile=True, has_touch=True,
            locale="de-DE", timezone_id="Europe/Berlin", reduced_motion=reduced_motion,
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
            if path == 'quest-release.json' and self.release_policy_failure:
                request_route.fulfill(status=503, content_type='application/json', body='{}')
                return
            if path == "quest-gateway.js" and url.query != "review-contract=1":
                request_route.fulfill(content_type="text/javascript", body=GATEWAY)
                return
            if path == "quest-pairing.js":
                request_route.fulfill(content_type="text/javascript", body=PAIRING)
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
        # Existing state fixtures often inspect both input controls and tracking.
        # They now live on separate pages: follow the real local navigation;
        # never unhide a control or replace a readiness/receipt assertion.
        if expected.count()==1 and not expected.is_visible():
            owner=expected.evaluate("node=>node.closest('.screen')?.id")
            step={'input-screen':'input','breathing-preview':'tracking','signals-screen':'signals',
                  'settings-screen':'settings','recordings-screen':'recordings','participant-screen':'participant'}.get(owner)
            if step:
                show_operator(page,step)
            disclosures=expected.evaluate("node=>{const ids=[];for(let p=node.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS'&&!p.open)ids.push(p.id);return ids.reverse()}")
            for disclosure in disclosures:
                page.locator('#'+disclosure+' > summary').click()
        expect(expected).to_be_visible(timeout=9000)
        expect(page.locator("html")).to_have_attribute("data-measurement", "ready")
        expect(page.locator(".screen:visible")).to_have_count(1)
        page.clock.run_for(100)
        expect(expected).to_be_visible()
        page.evaluate("window.scrollTo(0, 0)")
        page.clock.run_for(50)  # Paint the restored scroll position before capture.
        page.wait_for_function('''() => [...document.querySelectorAll('.screen:not([hidden]) [data-measure], dialog[open] [data-measure]')]
          .filter(node => node.checkVisibility() && node.getClientRects().length)
          .every(node => ['pretext','dom'].includes(node.dataset.textMeasurement))''')
        geometry = page.evaluate("""() => {
          const root = document.documentElement;
          const screen = document.querySelector('.screen:not([hidden])');
          const visible = selector => [...screen.querySelectorAll(selector)]
            .filter(node => node.getClientRects().length && node.checkVisibility());
          return {
            width: innerWidth, height: innerHeight, documentWidth: root.scrollWidth,
            documentHeight: root.scrollHeight, layout: root.dataset.layout,
            regions: [...screen.children].filter(node=>node.checkVisibility()).map(node=>({cls:node.className,rect:node.getBoundingClientRect().toJSON()})),
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
            smallTargets: visible('button, select, summary, input:not([type="radio"]):not([type="checkbox"]), a[href], .gender-options label').filter(node => {
              const box = node.getBoundingClientRect(); return box.width < 47.9 || box.height < 47.9;
            }).map(node => node.id || node.tagName),
            panelOverflow: visible('.screen-body, .actions, .version-panel, details').filter(node =>
              node.scrollWidth > node.clientWidth + 2 || node.scrollHeight > node.clientHeight + 2)
              .map(node => node.className),
            panels: visible('.screen-body, .actions, .version-panel, details, #operator-live, .preview-radius, .preview-figure').map(node=>({
              id:node.id,cls:node.className,client:[node.clientWidth,node.clientHeight],scroll:[node.scrollWidth,node.scrollHeight],
              bounds:node.getBoundingClientRect().toJSON()})),
          };
        }""")
        # Preserve the actual failed render as actionable evidence as well.
        (self.output / 'screens').mkdir(exist_ok=True)
        (self.output / f'{slug}-geometry.json').write_text(json.dumps(geometry,indent=2),encoding='utf-8')
        page.screenshot(path=self.output / 'screens' / f'{slug}-viewport.png')
        assert geometry['width'] <= page.viewport_size['width']+1, f'{slug}: mobile layout viewport expanded beyond requested width {geometry}'
        assert geometry["documentWidth"] <= geometry["width"] + 1, f"{slug}: horizontal overflow {geometry}"
        assert not geometry["textOverflow"], f"{slug}: incomplete labels {geometry['textOverflow']}"
        assert not geometry["missingMeasurement"], f"{slug}: unmeasured labels {geometry['missingMeasurement']}"
        assert not geometry['pretextNoFit'] or geometry['layout']=='reflow', f"{slug}: no-fit labels did not use readable reflow {geometry['pretextNoFit']}"
        assert not geometry["smallTargets"], f"{slug}: small controls {geometry['smallTargets']}"
        assert not geometry["panelOverflow"], f"{slug}: internal panel overflow {geometry['panelOverflow']}"
        regions=geometry['regions']
        for left,right in zip(regions,regions[1:]):
            assert left['rect']['bottom'] <= right['rect']['top']+1, f'{slug}: page regions overlap {regions}'
        if geometry["layout"] == "bounded":
            assert geometry["documentHeight"] <= geometry["height"] + 2, f"{slug}: bounded page scrolls"
        assert not self.errors, f"{slug}: app errors {self.errors}"
        number = f"{len(self.rows) + 1:02d}"
        path = f"screens/{number}-{slug}.png"
        for element in (expected, page.locator(".screen:visible")):
            assert element.evaluate("""node => node.getBoundingClientRect().bottom + scrollY
                <= document.documentElement.scrollHeight + 1"""), f"{slug} exceeds capture bounds"
        self.capture_document(page,self.output / path)
        after=page.evaluate('''() => {
          const s=document.querySelector('.screen:not([hidden])');
          return {width:innerWidth,layout:document.documentElement.dataset.layout,
            regions:[...s.children].filter(n=>n.checkVisibility()).map(n=>n.getBoundingClientRect().toJSON())};
        }''')
        assert after['width'] <= page.viewport_size['width']+1, f'{slug}: capture expanded viewport {after}'
        for left,right in zip(after['regions'],after['regions'][1:]):
            assert left['bottom'] <= right['top']+1, f'{slug}: capture caused page region overlap {after}'
        geometry['afterCapture']=after
        geometry['captureFiles']=self.tiles[path]
        self.rows.append((number, group, title, path))
        self.checks.append({"screen": number, "slug": slug, **geometry})
        print(f"{number} {title}", flush=True)

    def index(self):
        (self.output / "checks.json").write_text(
            json.dumps({"screens": self.checks, "regressions": self.regressions,
                        "app_errors": self.errors}, indent=2), encoding="utf-8")
        with (self.output / "feedback.csv").open("w", encoding="utf-8-sig", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(("Screen", "Section", "State", "Image", "Feedback"))
            writer.writerows((*row, "") for row in self.rows)
        entries = "\n".join(
            f'<figure><a href="{path}"><img src="{path}" alt="{escape(title)}" loading="lazy"></a>'
            f'<figcaption><strong>{number} · {escape(title)}</strong><br>{escape(group)}'
            + ''.join(f' <a href="{tile}">Tile {i+1}</a>' for i,tile in enumerate(self.tiles.get(path,[])))
            + '</figcaption></figure>'
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
            '<p>390 × 844, 320 × 568, 844 × 390 and 1440 × 900 CSS px, German interface; synthetic '
            'native Quest, participant, sensor, and recording data. '
            'Long screens use overlapping actual viewport tiles; open every tile to review the complete document.</p>'
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


def show_operator(page, step):
    """Follow real local navigation, without changing a domain model or gateway."""
    page.wait_for_function('Boolean(window.reviewQuest)')
    ids = {'connect':'quest','input':'input-screen','tracking':'breathing-preview','signals':'signals-screen',
           'battery':'battery-screen','session':'session-screen','calibration':'calibration-screen',
           'settings':'settings-screen','update':'update-screen','recordings':'recordings-screen','participant':'participant-screen'}
    target=page.locator('#'+ids[step])
    if target.is_visible():
        return
    before=page.evaluate('({commands:reviewQuest.commands??[],setup:reviewSetupCommands,access:reviewAccessCommands})')
    for _ in range(6):
        toolbar = page.locator('#toolbar-actions:visible')
        direct=page.locator(f'[data-operator-go="{step}"]:visible')
        if step in ('settings', 'recordings') and toolbar.count():
            toolbar.click()
            page.locator(f'#toolbar-dialog [data-operator-go="{step}"]').click()
        elif direct.count():
            direct.first.click()
        elif step == 'update' and not page.locator('#settings-screen').is_visible():
            show_operator(page, 'settings')
        elif step == 'session' and page.locator('#battery-next:visible').count():
            page.locator('#battery-next').click()
        elif step == 'update' and page.locator('[data-operator-go="settings"]:visible').count():
            page.locator('[data-operator-go="settings"]:visible').first.click()
        elif page.locator('[data-operator-return]:visible').count():
            page.locator('[data-operator-return]:visible').first.click()
        elif page.locator('[data-operator-go="input"]:visible').count():
            page.locator('[data-operator-go="input"]:visible').first.click()
        elif step == 'signals':
            page.evaluate("location.hash = 'signals'")
        elif step != 'connect' and page.locator('#operator-connect-next:visible').count():
            page.locator('#operator-connect-next').click()
        elif step not in ('connect', 'participant') and page.locator('#patient-next:visible').count() and page.locator('#patient-id').text_content() != '—':
            page.locator('#patient-next').click()
        elif page.locator('[data-operator-back]:visible').count():
            page.locator('[data-operator-back]:visible').first.click()
        else:
            break
        if target.is_visible():
            break
    expect(target).to_be_visible()
    assert before==page.evaluate('({commands:reviewQuest.commands??[],setup:reviewSetupCommands,access:reviewAccessCommands})'), 'Local navigation mutated native state'


def review_battery(page):
    """Use the new local review action, never bypass its route or native guards."""
    expect(page.locator('#battery-screen')).to_be_visible()
    before = page.evaluate('({session:reviewQuest.commands??[],setup:reviewSetupCommands,access:reviewAccessCommands})')
    expect(page.locator('#battery-next')).to_be_enabled()
    page.locator('#battery-next').click()
    expect(page.locator('#session-screen')).to_be_visible()
    assert before == page.evaluate('({session:reviewQuest.commands??[],setup:reviewSetupCommands,access:reviewAccessCommands})')


def enter_participant(page, then='input'):
    show_operator(page,'participant')
    participant = page.locator("#quest-participant")
    if participant.get_attribute("open") is None:
        participant.locator("summary").click()
    page.locator("#patient-surname").fill("Tes")
    page.locator("#patient-age").fill("30")
    page.locator("#patient-gender").select_option("prefer-not-to-say")
    page.locator("#patient-next").focus()
    page.locator("#patient-next").press("Enter")
    if then!='input':
        show_operator(page,then)


def capture_scroll_regression(review):
    """Poll the real UI while a phone user edits and scrolls the participant form.

    Live setup telemetry also updates the hidden preview. It must not shrink the
    visible document for one frame, clamp scroll, or replace the focused editor.
    Screenshots force real paint before the baseline and after each native poll;
    measuring only the final reflow height missed the original transient clamp.
    """
    initial = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({PAIRED}));" + """
      window.reviewInitialQuest={polar:'ready',access:{v:1,role:'active',active_phone:'Fixture',revision:1}};
      window.reviewInitialSetup={stage:'calibrating',calibration:'collecting',attempt:1,
        target_frames:1411200,input_phase:'inhale',input_volume01:0,playing:true,position_frames:0};"""
    for slug, viewport, text_size in (
        ("390", VIEWPORT, ""),
        ("320", {"width": 320, "height": 568}, ""),
        ("200-text", {"width": 320, "height": 568}, "200%"),
    ):
        page = review.page(initial, viewport=viewport)
        show_operator(page, 'participant')
        participant = page.locator("#quest-participant")
        expect(participant).to_be_visible()
        if participant.get_attribute('open') is None:
            participant.locator("summary").click()
        page.locator("#patient-age").fill("31")
        page.locator("#patient-gender").select_option("prefer-not-to-say")
        editor = page.locator("#patient-surname")
        editor.fill("Tes")
        page.evaluate("size => document.documentElement.style.fontSize=size", text_size)
        expect(page.locator("html")).to_have_attribute("data-measurement", "ready")
        page.clock.run_for(2200)
        page.screenshot()  # Let initial readable reflow reach the renderer.
        page.evaluate("""() => {
          window.reviewEditedNode = document.querySelector('#patient-surname');
          window.reviewEditedForm = document.querySelector('#patient-form');
          window.reviewPageOrigin = performance.timeOrigin;
        }""")
        samples = []
        for i, letter in enumerate(("a", "o", "e", "a", "o", "e"), 1):
            # A real edit on every poll, with an insertion point to preserve.
            editor.evaluate("node => {node.focus({preventScroll:true}); node.setSelectionRange(1,2);}")
            page.keyboard.type(letter)
            value = editor.input_value()
            page.clock.run_for(50)
            page.screenshot()
            scroll_range = page.evaluate("document.documentElement.scrollHeight - innerHeight")
            page.evaluate("height => scrollTo(0, Math.min(300, height))", scroll_range)
            page.screenshot()
            before = page.evaluate("scrollY")
            if text_size:
                assert before >= 20, f"{slug}: enlarged-text reflow was not exercised: {before}"
            else:
                assert before == min(300, scroll_range), f"{slug}: rendered scroll range did not match: {before}"
            wait_setup(page, position_frames=i * 88200, completed_frames=i * 88200,
                       input_volume01=i / 10, input_phase="inhale" if i % 2 else "exhale")
            page.screenshot()
            state = page.evaluate("""() => ({
              scroll:scrollY, documentHeight:document.documentElement.scrollHeight,
              sameOrigin:performance.timeOrigin===reviewPageOrigin,
              sameEditor:document.querySelector('#patient-surname')===reviewEditedNode,
              sameForm:document.querySelector('#patient-form')===reviewEditedForm,
              focus:document.activeElement.id, value:reviewEditedNode.value,
              selection:[reviewEditedNode.selectionStart,reviewEditedNode.selectionEnd],
              age:document.querySelector('#patient-age').value,
              gender:document.querySelector('#patient-gender').value,
              previewInput:document.querySelector('#preview-input').textContent,
              nativeVolume:window.reviewQuest.setup.input_volume01,
              progress:document.querySelector('#preview-progress').value,
            })""")
            samples.append({"before": before, **state})
            (review.output / f'scroll-{slug}-samples.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
            page.screenshot(path=review.output / f'scroll-{slug}-poll-{i}.png')
            assert state["sameOrigin"], f"{slug}: telemetry navigated or reloaded the page"
            assert state["sameEditor"] and state["sameForm"], f"{slug}: telemetry replaced the live form"
            assert state["focus"] == "patient-surname", f"{slug}: telemetry moved focus: {state}"
            assert state["value"] == value and state["selection"] == [2, 2], f"{slug}: edited text changed: {state}"
            assert state["age"] == "31" and state["gender"] == "prefer-not-to-say", f"{slug}: form values changed: {state}"
            assert ("Einatmen" if i % 2 else "Ausatmen") in state["previewInput"], f"{slug}: native phase did not render: {state}"
            assert state["nativeVolume"] == i / 10
            assert abs(state["progress"] - i / 8) < .001, f"{slug}: native calibration progress froze: {state}"
            assert abs(state["scroll"] - before) <= 2, f"{slug}: native poll reset document scroll: {samples[-1]}"
        assert page.evaluate("window.reviewQuest.commands ?? []") == []
        assert page.evaluate("window.reviewSetupCommands") == []
        review.regressions.append({"kind": "participant-scroll-edit", "viewport": viewport,
                                   "textSize": text_size, "samples": samples})
        review.shot(page, f"scroll-edit-{slug}", f"Participant {slug} · edits and scroll survive six live polls",
                    editor, "Live update stability")
        # A genuine user transition must still change screens and focus.
        enter_participant(page)
        expect(page.locator("#input-screen")).to_be_visible()
        expect(page.locator("#input-screen-title")).to_be_focused()
        page.context.close()


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
        expect(page.locator("#operator-connect-next")).to_be_enabled(timeout=1000)
        expect(page.locator("#check-bluetooth")).to_have_attribute("data-check","ok")
        expect(page.locator("#quest-find")).to_be_hidden()
        calls = page.evaluate("window.reviewDiscovery")
        assert calls[:2] == ["bluetooth", "unreachable"] and calls[-1] == "discovered", calls
        assert len(calls[2:-1]) >= 2 and set(calls[2:-1]) == {"bluetooth"}, calls
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
    show_operator(page, 'participant')
    expect(page.locator("#quest-participant")).to_be_visible()
    expect(page.locator("#participant-screen-title")).to_be_focused()
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
    show_operator(page, "participant")
    if page.locator('#quest-participant').get_attribute('open') is None:
        page.locator("#quest-participant summary").click()
    shot(page, "participant-empty", "Participant form", page.locator("#patient-form"), "Participant and Polar")
    page.locator("#patient-surname").fill("Tes")
    page.locator("#patient-age").fill("30")
    page.locator("#patient-gender").select_option("prefer-not-to-say")
    expect(page.locator("#patient-id")).to_have_text("Tes_30_N")
    shot(page, "participant-ready", "Completed participant form", page.locator("#patient-id"), "Participant and Polar")
    page.locator("#patient-next").click()
    show_operator(page, "connect")
    for state, slug, title, label in (
        ("off", "polar-connect", "Connect Polar H10", "Lege den Gurt an"),
        ("permission", "polar-permission", "Quest Bluetooth permission required", "Geräte in der Nähe"),
        ("bluetooth", "polar-bluetooth", "Quest Bluetooth switched off", "Schalte Bluetooth"),
        ("unsupported", "polar-unavailable", "Polar setup unavailable", "nicht verfügbar"),
        ("scan", "polar-scanning", "Searching for Polar H10", "Quest sucht"),
        ("stream-wait", "polar-streams", "Waiting for ECG and acceleration", "EKG und Bewegungsdaten fehlen"),
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
    expect(preview).to_have_text("Automatic ist bereit.")
    expect(page.locator("#preview-input")).to_contain_text("Automatic")
    expect(proceed).to_be_disabled()
    shot(page, "native-automatic", "Native Atemtakt · Automatic", preview, "Breathing setup")
    radius = page.locator("#preview-circle").get_attribute("style")
    page.clock.run_for(8500)
    expect(preview).to_have_text("Automatic ist bereit.")
    expect(page.locator("#preview-circle")).to_have_attribute("style", radius)
    expect(page.locator("#preview-title")).to_be_focused()
    page.keyboard.press("Tab")
    expect(page.locator(':focus')).to_be_visible()
    show_operator(page,'input')
    expect(page.locator('#input-screen-title')).to_be_focused()
    calibrate.focus()
    expect(calibrate).to_be_focused()
    page.keyboard.press("Enter")
    expect(detail).to_contain_text("zwei vollständige Atemzyklen")
    shot(page, "native-waiting", "First calibration starts the native clock", detail, "Breathing setup")
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
    expect(page.locator("#preview-input")).to_contain_text("Erkannt: Ausatmen · Rohwert der Eingabe: 23 %")
    expect(proceed).to_be_enabled()
    shot(page, "native-ready", "Polar ready · measured input differs from intended hold", proceed, "Breathing setup")
    show_operator(page,'input')
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
    review_battery(page)
    expect(page.locator("#session-screen-title")).to_be_focused()

    scenario = page.locator("#quest-scenario")
    expect(scenario).to_be_visible()
    expect(scenario.get_by_role("radio")).to_have_count(3)
    expect(page.locator("#quest-condition")).to_have_count(0)
    instruction = scenario.locator('input[value="probe-anleitung"]')
    practice = scenario.locator('input[value="probe-5"]')
    main = scenario.locator('input[value="haupt-15"]')
    instruction.check()
    instruction.press("ArrowRight")
    expect(practice).to_be_checked()
    practice.press("ArrowRight")
    expect(main).to_be_checked()
    for value, title in (
        ("probe-anleitung", "Anleitung"),
        ("probe-5", "Geführte Atmung"),
        ("haupt-15", "Haupt-Runde"),
    ):
        tile = scenario.locator(f'input[value="{value}"]')
        tile.check()
        expect(tile).to_be_checked()
        shot(page, value, title, scenario, "Scenario selection")

    play = page.locator("#quest-play")
    finish = page.locator("#quest-finish")
    expect(play).to_have_text("Starten")
    shot(page, "ready-to-start", "Ready to start the selected run", play, "Playback")
    play.click()
    expect(play).to_have_text("Pause")
    assert page.evaluate("reviewQuest.armedScenario") == "haupt-15"
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
    show_operator(page,'tracking')
    shot(page, "finished", "Run finished and returned to native input testing", page.locator("#preview-review"), "Playback")
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
    show_operator(page,'recordings')
    page.locator("#quest-export-retry").click()
    shot(page, "quest-recording", "Saved Quest recording and download", await_archive, "Recordings and Drive")
    show_operator(page, 'connect')
    page.locator("#drive-open").click()
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
    detail = page.locator("#preview-recovery")
    expect(detail).to_contain_text("Ergebnis der Anfrage unklar")
    expect(page.locator("#patient-surname")).to_have_value("")
    page.reload(wait_until="load")
    expect(detail).to_be_visible()
    expect(detail).to_contain_text("Ergebnis der Anfrage unklar")
    expect(page.locator("#preview-continue")).to_be_disabled()
    expect(page.locator("#preview-acknowledge")).to_be_hidden()
    assert page.evaluate("JSON.parse(sessionStorage.getItem('atmungsraum-breathing-setup-pending')).command.id") == "f" * 16
    assert page.evaluate("window.reviewSetupCommands.length") == 0
    review.shot(page, "setup-reload-pending", "Reload · uncertain setup with blank participant ID", detail, "Setup recovery")
    wait_state(page, receiptState="confirmed")
    wait_setup(page, r=1, stage="waiting", calibration="waiting", attempt=1, target_frames=1411200)
    expect(page.locator("#breathing-preview")).to_be_hidden()
    show_operator(page, "participant")
    expect(page.locator("#patient-fields")).to_be_enabled()
    assert page.evaluate("sessionStorage.getItem('atmungsraum-breathing-setup-pending')") is None
    assert page.evaluate("window.reviewSetupCommands.length") == 0
    review.shot(page, "setup-reload-confirmed", "Reload · receipt resolves without resending", page.locator("#quest-participant"), "Setup recovery")
    page.context.close()

    for slug, title, renewal in (
        ("gateway", "Renewed gateway epoch", "window.reviewInitialOwner={epoch:'9'};"),
        ("native", "Replaced native owner", "window.reviewInitialOwner={epoch:'9',native_epoch:'10'};"),
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
        show_operator(page, "participant")
        expect(page.locator("#participant-screen-title")).to_be_focused()
        expect(page.locator("#patient-fields")).to_be_enabled()
        assert page.evaluate("sessionStorage.getItem('atmungsraum-breathing-setup-pending')") is None
        assert page.evaluate("window.reviewSetupCommands.length") == 0
        review.shot(page, f"setup-{slug}-recovered", f"{title} · participant entry restored without replay", page.locator("#quest-participant"), "Setup recovery")
        page.context.close()

    page = review.page(paired + "window.reviewInitialQuest={polar:'ready',oldApk:true};")
    enter_participant(page,then='input')
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
        enter_participant(page,then='input')
        if spacing:
            page.add_style_tag(content="""* {line-height:1.5 !important;letter-spacing:.12em !important;word-spacing:.16em !important;}
              p {margin-bottom:2em !important;}""")
        page.evaluate("value => {document.documentElement.style.fontSize=value; document.documentElement.style.setProperty('--review-check','1');}", text_size)
        page.clock.run_for(200)
        review.shot(page, slug, title, page.locator("#preview-input"), "Accessibility and reflow")
        expect(page.locator("#preview-input-volume")).to_have_text("62 %")
        if spacing:
            expect(page.locator("#preview-input-label")).to_have_attribute("data-text-measurement", "dom")
            expect(page.locator("#preview-input-volume")).to_have_attribute("data-text-measurement", "dom")
        show_operator(page, 'tracking')
        proceed = page.locator("#preview-continue")
        expect(proceed).to_be_enabled()
        proceed.focus()
        proceed.press("Enter")
        review_battery(page)
        expect(page.locator("#quest-scenario")).to_be_visible()
        expect(page.locator("#session-screen-title")).to_be_focused()
        page.context.close()

    page = review.page(initial)
    enter_participant(page,then='input')
    wait_setup(page, r=4, input="automatic", ready=False, calibration="failed", input_phase="bad-tracking",
               failure="insufficient-range-" + "Sensorbewegung-unzureichend-" * 7)
    expect(page.locator("#preview-detail")).to_contain_text("Einrichtung fehlgeschlagen")
    review.shot(page, "long-failure", "Long native failure reason · readable reflow", page.locator("#preview-detail"), "Accessibility and reflow")
    page.context.close()


INSTALLATION = "window.reviewInitialOwner={identity_kind:'app-installation',serial:''};"
NEW_PHONE = INSTALLATION + """window.reviewPairingEnabled=true;
  window.reviewInitialQuest={access:{v:1,role:'waiting',active_phone:'',revision:0}};"""


def begin_pairing(page):
    find = page.locator("#quest-find")
    expect(find).to_be_visible()
    find.focus()
    find.press("Enter")
    expect(page.locator("#quest-pairing")).to_be_visible()
    expect(page.locator("#quest-pairing-code")).to_have_text("314159")
    expect(page.locator("#quest-pairing-confirm")).to_be_focused()
    expect(page.locator("#quest-pairing input, #quest-pairing select, #quest-pairing textarea")).to_have_count(0)
    assert page.evaluate("localStorage.getItem('atmungsraum-quest-pair')") is None


def capture_phone_pairing(review):
    page = review.page(NEW_PHONE)
    begin_pairing(page)
    review.shot(page, "pairing-code", "New phone · compare the six-digit code without typing", page.locator("#quest-pairing-code"), "Phone pairing")
    page.evaluate("window.reviewPairing.finish('approved')")
    assert page.evaluate("window.reviewPairing.state") == "pending"
    assert page.evaluate("localStorage.getItem('atmungsraum-quest-pair')") is None
    page.keyboard.press("Tab")
    expect(page.locator("#quest-pairing-cancel")).to_be_focused()
    page.keyboard.press("Shift+Tab")
    expect(page.locator("#quest-pairing-confirm")).to_be_focused()
    page.keyboard.press("Enter")
    expect(page.locator("#quest-pairing")).to_be_hidden()
    expect(page.locator("#quest-pairing-code")).to_have_text("")
    page.clock.run_for(2100)
    expect(page.locator("#check-control")).to_have_attribute("data-check", "ok")
    expect(page.locator("#quest-control-release")).to_be_hidden()
    stored = page.evaluate("JSON.parse(localStorage.getItem('atmungsraum-quest-pair'))")
    assert stored["identity_kind"] == "app-installation" and stored["serial"] == ""
    assert page.evaluate("window.reviewPairingCalls") == ["begin", "phone-confirmed", "approved"]
    assert page.evaluate("window.reviewAccessCommands.map(command=>command.op)") == ["claim"]
    review.shot(page, "pairing-approved", "Both buttons confirmed · native control acquired automatically", page.locator("#operator-connect-next"), "Phone pairing")
    expect(page.locator("#operator-connect-next")).to_be_enabled()
    page.locator("#operator-connect-next").click()
    expect(page.locator("#quest-participant")).to_be_visible()
    show_operator(page,'settings')
    page.locator('#settings-connection summary').click()
    expect(page.locator("#quest-control-release")).to_be_visible()
    assert page.evaluate("window.reviewAccessCommands.map(command=>command.op)") == ["claim"]
    page.context.close()

    page = review.page(NEW_PHONE)
    begin_pairing(page)
    page.keyboard.press("Enter")
    expect(page.locator("#quest-pairing-confirm")).to_be_disabled()
    expect(page.locator("#quest-status")).to_contain_text("Warte auf die Freigabe im Headset")
    assert page.evaluate("localStorage.getItem('atmungsraum-quest-pair')") is None
    review.shot(page, "pairing-headset-wait", "Phone confirmed · headset approval still required", page.locator("#quest-pairing-code"), "Phone pairing")
    page.locator("#quest-pairing-cancel").focus()
    page.keyboard.press("Enter")
    expect(page.locator("#quest-pairing")).to_be_hidden()
    expect(page.locator("#quest-pairing-code")).to_have_text("")
    expect(page.locator("#quest-find")).to_be_visible()
    expect(page.locator("#quest-find")).to_be_focused()
    assert page.evaluate("window.reviewPairing.state") == "cancelled"
    assert page.evaluate("localStorage.getItem('atmungsraum-quest-pair')") is None
    review.shot(page, "pairing-cancelled", "Phone cancels · no membership saved", page.locator("#quest-find"), "Phone pairing")
    page.context.close()

    page = review.page(NEW_PHONE)
    begin_pairing(page)
    page.evaluate("window.reviewPairing.finish('rejected')")
    expect(page.locator("#quest-pairing")).to_be_hidden()
    expect(page.locator("#quest-status")).to_contain_text("im Headset abgelehnt")
    expect(page.locator("#quest-find")).to_be_focused()
    assert page.evaluate("localStorage.getItem('atmungsraum-quest-pair')") is None
    review.shot(page, "pairing-rejected", "Headset rejects · code cleared and retry available", page.locator("#quest-status"), "Phone pairing")
    page.context.close()

    page = review.page(NEW_PHONE + "window.reviewInitialQuest.failJoinAfterPairing=true;")
    begin_pairing(page)
    page.evaluate("window.reviewPairing.finish('approved')")
    page.keyboard.press("Enter")
    expect(page.locator("#quest-status")).to_contain_text("lost reply after pairing approval")
    saved = page.evaluate("JSON.parse(localStorage.getItem('atmungsraum-quest-pair'))")
    assert saved["enrollment"] == {"controller": "a" * 32, "key": "b" * 64}
    review.shot(page, "pairing-join-retry", "Approved membership survives a lost first connection reply", page.locator("#quest-find"), "Phone pairing")
    page.locator("#quest-find").focus()
    page.keyboard.press("Enter")
    expect(page.locator("#quest-find")).to_be_hidden()
    page.clock.run_for(2100)
    expect(page.locator("#check-control")).to_have_attribute("data-check", "ok")
    assert page.evaluate("JSON.parse(localStorage.getItem('atmungsraum-quest-pair')).enrollment") == saved["enrollment"]
    assert page.evaluate("window.reviewPairingCalls") == ["begin", "phone-confirmed", "approved"]
    assert page.evaluate("window.reviewAccessCommands.map(command=>command.op)") == ["claim"]
    review.shot(page, "pairing-join-recovered", "Retry reuses approved membership and acquires free native control", page.locator("#operator-connect-next"), "Phone pairing")
    page.context.close()


def capture_controller_access(review):
    paired = """localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({
      endpoint:'synthetic',device:'d'.repeat(32),serial:'',identity_kind:'app-installation',
      enrollment:{controller:'a'.repeat(32),key:'b'.repeat(64)}}));"""
    initial = INSTALLATION + paired + """window.reviewInitialQuest={polar:'ready',
      access:{v:1,role:'waiting',active_phone:'Anderes Smartphone',revision:4}};
      window.reviewInitialSetup={r:2,input:'polar-tilt',ready:true,calibration:'ready',attempt:1,
        target_frames:1411200,completed_frames:1411200,input_phase:'hold',input_volume01:.5};"""
    page = review.page(initial + "window.reviewInitialQuest.access.active_phone='';")
    expect(page.locator("#check-bridge")).to_have_attribute("data-check", "ok")
    page.clock.run_for(2100)
    expect(page.locator("#quest-control-claim")).to_be_hidden()
    expect(page.locator("#quest-participant")).to_be_hidden()
    assert page.evaluate("window.reviewAccessCommands.map(command=>command.op)") == ["claim"]
    assert page.evaluate("window.reviewQuest.access.role") == "active"
    review.shot(page, "control-saved-free", "Saved phone reconnects · native control acquired automatically", page.locator("#operator-connect-next"), "Controller access")
    page.context.close()
    page = review.page(initial)
    status = page.locator("#quest-control-status")
    claim = page.locator("#quest-control-claim")
    release = page.locator("#quest-control-release")
    expect(status).to_contain_text("Anderes Smartphone steuert die Quest")
    expect(page.locator("#quest-participant")).to_be_hidden()
    expect(page.locator("#quest-polar")).to_be_hidden()
    expect(page.locator("#recenter-vr")).to_have_attribute("aria-disabled", "true")
    expect(release).to_be_hidden()
    expect(claim).to_be_disabled()
    review.shot(page, "control-waiting", "Trusted phone waits · experiment controls stay unavailable", status, "Controller access")
    assert page.evaluate("window.reviewAccessCommands") == []
    assert page.evaluate("window.reviewQuest.access.revision") == 4
    wait_state(page, access={"v": 1, "role": "waiting", "active_phone": "", "revision": 5})
    page.clock.run_for(2100)
    expect(page.locator("#check-control")).to_have_attribute("data-check", "ok")
    expect(status).to_contain_text("Dieses Smartphone steuert die Quest.")
    expect(page.locator("#patient-fields")).to_be_enabled()
    show_operator(page,'settings')
    page.locator('#settings-connection summary').click()
    expect(release).to_be_visible()
    page.keyboard.press("Escape")
    recenter = page.locator("#recenter-vr")
    expect(recenter).to_have_attribute("aria-disabled", "false")
    recenter.focus()
    recenter.press("Enter")
    operator_status = page.locator("#quest-status")
    expect(operator_status).to_have_text("VR-Szene zentriert.")
    page.locator('#settings-connection summary').click()
    review.shot(page, "control-recentered", "Active phone · recenter confirmation", operator_status, "Controller access")
    release.focus()
    release.press("Enter")
    expect(status).to_contain_text("Steuerung freigegeben")
    expect(operator_status).to_contain_text("Steuerung freigegeben")
    expect(operator_status).not_to_contain_text("VR-Szene zentriert")
    expect(page.locator("#quest-polar")).to_be_hidden()
    expect(claim).to_be_enabled()
    expect(recenter).to_have_attribute("aria-disabled", "true")
    review.shot(page, "control-free", "Recenter then release · Claim prompt replaces the old confirmation", operator_status, "Controller access")
    claim.focus()
    claim.press("Enter")
    show_operator(page,'settings')
    expect(release).to_be_visible()
    enter_participant(page,then='tracking')
    expect(page.locator("#breathing-preview")).to_be_visible()
    expect(page.locator("#preview-continue")).to_be_enabled()
    show_operator(page,'settings')
    expect(page.locator("#settings-screen #quest-control-access")).to_be_visible()
    review.shot(page, "control-preview-host", "Controller release remains reachable on the breathing preview", release, "Controller access")
    before = page.evaluate("({g:reviewQuest.setup.g,position:reviewQuest.setup.position_frames})")
    release.focus()
    release.press("Enter")
    expect(claim).to_be_visible()
    expect(page.locator("#preview-continue")).to_be_disabled()
    expect(page.locator("#preview-calibrate")).to_be_disabled()
    expect(page.locator("#preview-controller")).to_be_disabled()
    review.shot(page, "control-preview-released", "Preview releases control · native clock and input stay intact", claim, "Controller access")
    claim.focus()
    claim.press("Enter")
    expect(release).to_be_visible()
    expect(page.locator("#preview-continue")).to_be_enabled()
    assert before == page.evaluate("({g:reviewQuest.setup.g,position:reviewQuest.setup.position_frames})")
    assert page.evaluate("window.reviewQuest.commands ?? []") == []
    assert page.evaluate("window.reviewSetupCommands") == []
    assert page.evaluate("window.reviewRecenterCommands.length") == 1
    assert page.evaluate("window.reviewAccessCommands.map(command=>command.op)") == ["claim", "release", "claim", "release", "claim"]
    page.context.close()

    for slug, viewport, text_size in (
        ("390", VIEWPORT, ""),
        ("320", {"width": 320, "height": 568}, ""),
        ("200-text", {"width": 320, "height": 568}, "200%"),
    ):
        page = review.page(NEW_PHONE, viewport=viewport)
        begin_pairing(page)
        page.locator("#quest-pairing-confirm [data-measure]").evaluate(
            "node => node.textContent='Der sechsstellige Code stimmt mit dem angezeigten Code im Headset überein'")
        page.evaluate("value => document.documentElement.style.fontSize=value", text_size)
        page.clock.run_for(200)
        review.shot(page, f"pairing-{slug}", f"Pairing {slug} · full code and long confirmation label", page.locator("#quest-pairing-code"), "Pairing accessibility")
        page.locator("#quest-pairing-cancel").focus()
        page.keyboard.press("Enter")
        expect(page.locator("#quest-find")).to_be_focused()
        page.context.close()

        page = review.page(initial, viewport=viewport)
        expect(status := page.locator("#quest-control-status")).to_be_visible()
        wait_state(page, access={"v": 1, "role": "waiting", "revision": 4,
                                "active_phone": "Studienleitung mit sehr langem Gerätenamen für diese Übergabe"})
        page.evaluate("value => document.documentElement.style.fontSize=value", text_size)
        page.clock.run_for(200)
        review.shot(page, f"control-{slug}", f"Waiting controller {slug} · long active-phone label", status, "Pairing accessibility")
        expect(page.locator("#patient-fields")).to_have_js_property("disabled", True)
        expect(page.locator("#patient-surname")).to_be_disabled()
        expect(page.locator("#recenter-vr")).to_have_attribute("aria-disabled", "true")
        page.context.close()


def capture_polar_location(review):
    initial = INSTALLATION + """localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({
      endpoint:'synthetic',device:'d'.repeat(32),serial:'',identity_kind:'app-installation',
      enrollment:{controller:'a'.repeat(32),key:'b'.repeat(64)}}));
      window.reviewInitialQuest={polar:'off',access:{v:1,role:'active',active_phone:'Fixture-Smartphone',revision:4}};"""
    for slug, viewport, text_size in (
        ("390", VIEWPORT, ""),
        ("320", {"width": 320, "height": 568}, ""),
        ("200-text", {"width": 320, "height": 568}, "200%"),
    ):
        page = review.page(initial, viewport=viewport)
        enter_participant(page, then='signals')
        status = page.locator("#quest-polar-status")
        connect = page.locator("#quest-polar-connect")
        expect(status).to_contain_text("Lege den Gurt an")
        expect(connect).to_be_enabled()  # Older observations omit optional y.l.
        page.evaluate("value => document.documentElement.style.fontSize=value", text_size)
        wait_state(page, location="disabled")
        expect(status).to_have_text("Aktiviere die Standortdienste in den Quest-Einstellungen, damit die Quest den H10 finden kann.")
        expect(connect).to_be_disabled()
        connect.click(force=True)
        page.clock.run_for(1100)
        assert page.evaluate("window.reviewQuest.commands ?? []") == []
        review.shot(page, f"location-off-{slug}", f"Location disabled {slug} · enable Standortdienste before H10 discovery", status, "Polar Location services")

        wait_state(page, polar="stream-wait")
        expect(status).to_contain_text("EKG und Bewegungsdaten fehlen")
        expect(status).not_to_contain_text("Standortdienste")
        expect(connect).to_be_enabled()
        connect.focus()
        connect.press("Enter")
        expect(status).to_have_text("Polar H10 verbunden. Quest empfängt EKG und Beschleunigungsdaten.")
        expect(connect).to_be_disabled()  # Both streams are already running.
        assert page.evaluate("window.reviewQuest.location") == "disabled"
        assert page.evaluate("window.reviewQuest.commands") == ["start-ecg", "start-acc"]
        review.shot(page, f"location-connected-{slug}", f"Connected H10 {slug} · ECG and acceleration work while Location is off", status, "Polar Location services")
        show_operator(page,'input')
        expect(page.locator("#input-screen")).to_be_visible()
        expect(page.locator("#preview-calibrate")).to_be_enabled()
        expect(page.locator("#preview-controller")).to_be_enabled()
        page.context.close()


def capture_polar_stream_recovery(review):
    paired = INSTALLATION + f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({PAIRED}));"
    for slug, viewport, text_size in (
        ("390", VIEWPORT, ""),
        ("320", {"width": 320, "height": 568}, ""),
        ("200-text", {"width": 320, "height": 568}, "200%"),
    ):
        for missing, ecg, acc, label, commands in (
            ("ecg", False, True, "Bewegungsdaten bereit; EKG fehlt.", ["start-ecg"]),
            ("acc", True, False, "EKG bereit; Bewegungsdaten fehlen.", ["start-acc"]),
            ("both", False, False, "EKG und Bewegungsdaten fehlen.", ["start-ecg", "start-acc"]),
        ):
            initial = paired + "window.reviewInitialQuest=" + json.dumps({
                "polar": "stream-wait", "ecg": ecg, "acc": acc, "location": "disabled",
                "access": {"v": 1, "role": "active", "active_phone": "Fixture", "revision": 1},
            }) + ";"
            page = review.page(initial, viewport=viewport)
            page.evaluate("size => document.documentElement.style.fontSize=size", text_size)
            status = page.locator("#quest-polar-status")
            connect = page.locator("#quest-polar-connect")
            expect(status).to_have_text("Polar H10 per Bluetooth verbunden. " + label)
            expect(connect).to_have_text("Fehlende Messsignale starten")
            expect(connect).to_be_enabled()  # Connected streams do not need Location discovery.
            page.clock.run_for(3100)
            assert page.evaluate("window.reviewQuest.commands ?? []") == [], "Polling issued a stream command"
            expect(status).to_have_text("Polar H10 per Bluetooth verbunden. " + label)
            review.shot(page, f"polar-missing-{missing}-{slug}", f"H10 {slug} · missing {missing}, explicit recovery",
                        status, "Polar stream recovery")
            connect.focus()
            connect.press("Enter")
            expect(status).to_have_text("Polar H10 verbunden. Quest empfängt EKG und Beschleunigungsdaten.")
            expect(connect).to_have_text("Messsignale bereit")
            expect(connect).to_be_hidden()
            ready_check = page.locator("#check-polar")
            expect(ready_check).to_have_attribute("data-check", "ok")
            assert page.evaluate("window.reviewQuest.commands") == commands
            page.clock.run_for(3100)
            assert page.evaluate("window.reviewQuest.commands") == commands, "Polling retried confirmed stream commands"
            assert page.evaluate("window.reviewQuest.location") == "disabled"
            review.shot(page, f"polar-recovered-{missing}-{slug}", f"H10 {slug} · {missing} recovery confirmed by both streams",
                        ready_check, "Polar stream recovery")
            review.regressions.append({"kind": "polar-stream-recovery", "viewport": viewport,
                                       "textSize": text_size, "missing": missing, "commands": commands})
            page.context.close()


def capture_live_display(review):
    """Render presentation frames between native polls without granting authority."""
    paired = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({PAIRED}));"
    initial = paired + """
      window.reviewInitialQuest={polar:'ready',access:{v:1,role:'active',active_phone:'Fixture',revision:1}};
      window.reviewInitialSetup={stage:'calibrating',calibration:'collecting',attempt:1,
        target_frames:1411200,completed_frames:176400,position_frames:88200,
        audio_state:'playing',audio_generation:1,input_phase:'inhale',input_volume01:.3};
      Object.defineProperty(document,'hidden',{configurable:true,get:()=>Boolean(window.reviewHidden)});
    """
    read = """() => ({
      circle:Number(document.querySelector('#preview-circle').style.transform.match(/scale\\(([^)]+)\\)/)?.[1]),
      fill:Number(document.querySelector('#preview-progress-fill').style.transform.match(/scaleX\\(([^)]+)\\)/)?.[1]),
      progress:document.querySelector('#preview-progress').value,
      volume:Number(document.querySelector('#preview-input-volume').textContent.match(/(\\d+)/)?.[1]),
      measurementWrites:window.reviewMeasurementWrites??0,
      continueDisabled:document.querySelector('#preview-continue').disabled,
      calibrateDisabled:document.querySelector('#preview-calibrate').disabled,
      controllerDisabled:document.querySelector('#preview-controller').disabled,
      status:document.querySelector('#preview-status').textContent,
      polls:window.reviewSnapshotCount, ecgCalls:window.reviewEcgCalls,
      setupCommands:window.reviewSetupCommands.length, commands:(window.reviewQuest.commands??[]).length,
    })"""
    page = review.page(initial, reduced_motion="no-preference")
    enter_participant(page,then='tracking')
    page.clock.run_for(1100)
    page.evaluate("""() => {
      window.reviewMeasurementWrites=0;
      new MutationObserver(records=>window.reviewMeasurementWrites+=records.length)
        .observe(document.querySelector('main'),{subtree:true,attributes:true,attributeFilter:['data-text-measurement']});
    }""")
    page.evaluate('window.reviewQuest.freezeObservation=true')
    first = page.evaluate(read)
    page.clock.run_for(200)
    later = page.evaluate(read)
    assert first["circle"] < later["circle"] <= 1, f"Native-anchored circle did not advance: {first}, {later}"
    assert first["volume"] < later["volume"], f"Automatic percent did not move with the circle: {first}, {later}"
    assert abs(later["volume"] - (later["circle"] - .46) / .54 * 100) <= .6
    assert later["measurementWrites"] == first["measurementWrites"], "Numeric presentation remeasured the whole page between polls"
    assert 0 <= later["polls"]-first["polls"] <= 3 and 0 <= later["ecgCalls"]-first["ecgCalls"] <= 1, "Animation exceeded scheduled transport cadence"
    assert first["progress"] == later["progress"] == .25, "Presentation invented native calibration progress"
    assert first["continueDisabled"] and later["continueDisabled"], "Animation admitted unfinished calibration"
    assert first["calibrateDisabled"] == later["calibrateDisabled"] and first["controllerDisabled"] == later["controllerDisabled"]

    page.evaluate('Object.assign(reviewQuest.setup,{completed_frames:352800,position_frames:132300});reviewQuest.freezeObservation=false')
    page.clock.run_for(110)
    page.evaluate('window.reviewQuest.freezeObservation=true')
    progressing = page.evaluate(read)
    page.clock.run_for(200)
    progressed = page.evaluate(read)
    assert progressing["progress"] == progressed["progress"] == .5
    assert .125 <= progressing["fill"] < progressed["fill"] <= .25, "Visible fill must stop at the confirmed native count"
    assert 0 <= progressed["polls"]-progressing["polls"] <= 3 and 0 <= progressed["ecgCalls"]-progressing["ecgCalls"] <= 1, "Fill animation exceeded scheduled transport cadence"
    assert progressed["continueDisabled"], "Interpolated fill admitted unfinished calibration"
    review.shot(page, "live-breath-between-polls", "Breathing display · smooth frames, native progress and admission",
                page.locator("#preview-status"), "Live presentation")

    page.evaluate('window.reviewQuest.freezeObservation=false')
    wait_setup(page, playing=False, audio_state="paused")
    paused = page.evaluate(read)
    page.clock.run_for(150)
    assert page.evaluate(read)["circle"] == paused["circle"], "Paused native audio kept moving"
    wait_setup(page, playing=True, audio_state="playing")
    page.evaluate("window.reviewQuest.freezeObservation=true")
    page.clock.run_for(3100)  # Polls replay one observation; they cannot renew its age.
    stale = page.evaluate(read)
    page.clock.run_for(200)
    assert page.evaluate(read)["circle"] == stale["circle"], "Stale native audio kept moving"
    expect(page.locator("#preview-status")).to_have_text("Warte auf den aktuellen Atemtakt der Quest.")
    review.shot(page, "live-breath-stale", "Breathing display · replayed observation expires and stops",
                page.locator("#preview-status"), "Live presentation")

    page.evaluate("window.reviewHidden=true; document.dispatchEvent(new Event('visibilitychange'))")
    hidden = page.evaluate(read)
    page.clock.run_for(2100)
    suspended = page.evaluate(read)
    assert suspended["polls"] == hidden["polls"] and suspended["ecgCalls"] == hidden["ecgCalls"], "Hidden document kept polling"
    assert suspended["circle"] == hidden["circle"], "Hidden document kept animating"
    expect(page.locator("#quest-ecg")).to_be_hidden()
    page.evaluate("window.reviewQuest.freezeObservation=false; window.reviewHidden=false; document.dispatchEvent(new Event('visibilitychange'))")
    page.clock.run_for(100)
    page.evaluate('window.reviewQuest.freezeObservation=true')
    resumed = page.evaluate(read)
    page.clock.run_for(200)
    assert page.evaluate(read)["circle"] > resumed["circle"], "A fresh visible observation did not resume the circle"
    assert not resumed["commands"] and not resumed["setupCommands"]
    review.regressions.append({"kind": "live-breath-presentation", "first": first, "later": later,
                               "progressing": progressing, "progressed": progressed,
                               "paused": paused, "stale": stale, "suspended": suspended, "resumed": resumed})
    page.context.close()

    page = review.page(initial)  # Existing default remains reduced motion.
    enter_participant(page,then='tracking')
    page.clock.run_for(1100)
    reduced = page.evaluate(read)
    page.clock.run_for(200)
    reduced_later = page.evaluate(read)
    assert reduced_later["circle"] == reduced["circle"] == .73
    assert reduced_later["volume"] == reduced["volume"] == 50
    assert reduced_later["fill"] == reduced["fill"] == .125
    assert reduced_later["progress"] == .25 and reduced_later["continueDisabled"]
    assert 0 <= reduced_later["polls"]-reduced["polls"] <= 3
    review.shot(page, "live-reduced-motion", "Reduced motion · discrete native breathing display",
                page.locator("#preview-status"), "Live presentation")
    review.regressions.append({"kind": "live-reduced-motion", "first": reduced, "later": reduced_later})
    page.evaluate("window.reviewQuest.freezeObservation=true")
    page.clock.run_for(2700)
    expect(page.locator("#preview-status")).to_have_text("Warte auf den aktuellen Atemtakt der Quest.")
    review.shot(page, "live-reduced-expiry", "Reduced motion · stale native observation still expires",
                page.locator("#preview-status"), "Live presentation")
    show_operator(page,'signals')
    # Synchronize just after an ordinary delivery. Tile painting changes timer
    # phase; a random 200ms interval may straddle the next permitted 1Hz poll.
    ecg_before=page.evaluate('window.reviewEcgCalls??0')
    for _ in range(120):
        page.clock.run_for(10)
        if page.evaluate('window.reviewEcgCalls??0')>ecg_before:
            break
    page.clock.run_for(20)
    expect(page.locator('#quest-ecg')).to_be_visible()
    page.evaluate('reviewQuest.ecgReplay=true')
    def reduced_ecg_state():
        return page.evaluate("({path:document.querySelector('#quest-ecg-trace').getAttribute('d'),ecg:reviewEcgCalls,polls:reviewSnapshotCount,latest:reviewLastEcg.samples.at(-1)[0],now:performance.now()})")
    reduced_ecg=reduced_ecg_state()
    page.clock.run_for(200)
    reduced_ecg_later=reduced_ecg_state()
    assert reduced_ecg['ecg']==reduced_ecg_later['ecg'] and reduced_ecg['polls']==reduced_ecg_later['polls'],(reduced_ecg,reduced_ecg_later)
    assert reduced_ecg['path'] and reduced_ecg_later['path']==reduced_ecg['path'],('Reduced motion kept ECG animating',reduced_ecg,reduced_ecg_later)
    page.evaluate('reviewQuest.ecgReplay=false')
    page.clock.run_for(1100)
    reduced_ecg_fresh=reduced_ecg_state()
    assert reduced_ecg_fresh['latest']>reduced_ecg['latest'] and reduced_ecg_fresh['path']!=reduced_ecg['path'],(reduced_ecg,reduced_ecg_fresh)
    review.regressions.append({'kind':'reduced-ecg-native-delivery','quietBefore':reduced_ecg,'quietAfter':reduced_ecg_later,'fresh':reduced_ecg_fresh})
    page.context.close()

    page = review.page(initial + """Object.assign(window.reviewInitialSetup,{stage:'testing',calibration:'ready',
      input:'polar-tilt',ready:true,completed_frames:1411200,input_volume01:.2});""", reduced_motion="no-preference")
    enter_participant(page,then='tracking')
    page.clock.run_for(1100)
    expect(page.locator("#preview-input-volume")).to_have_text("20 %")
    page.evaluate('reviewQuest.setup.input_volume01=.8')
    page.clock.run_for(110)
    page.evaluate('window.reviewQuest.freezeObservation=true')
    measured_first = page.evaluate(read)
    page.clock.run_for(200)
    measured_later = page.evaluate(read)
    assert 20 <= measured_first["volume"] < measured_later["volume"] <= 80
    assert 0 <= measured_later["polls"]-measured_first["polls"] <= 3
    page.evaluate("window.reviewQuest.freezeObservation=true")
    page.clock.run_for(2700)
    measured_stale = page.evaluate(read)
    page.clock.run_for(200)
    assert page.evaluate(read)["volume"] == measured_stale["volume"] <= 80
    review.shot(page, "live-measured-percent", "Measured input · percent blends only between received samples",
                page.locator("#preview-status"), "Live presentation")
    review.regressions.append({"kind":"live-measured-percent", "first":measured_first,
                               "later":measured_later,"stale":measured_stale})
    page.context.close()

    page = review.page(initial + "window.reviewInitialQuest.statusDelay=1700;", reduced_motion="no-preference")
    enter_participant(page,then='tracking')
    page.clock.run_for(2100)
    delayed = page.evaluate(read)
    page.clock.run_for(200)
    delayed_later = page.evaluate(read)
    assert delayed["circle"] == delayed_later["circle"] == .73, "Slow transport renewed an expired native audio anchor"
    expect(page.locator("#preview-status")).to_have_text("Warte auf den aktuellen Atemtakt der Quest.")
    assert delayed_later["continueDisabled"] and delayed_later["progress"] == .25
    assert not delayed_later["commands"] and not delayed_later["setupCommands"]
    review.shot(page, "live-slow-status", "Native observation · 1700 ms transport cannot animate stale audio",
                page.locator("#preview-status"), "Live presentation")
    review.regressions.append({"kind": "live-slow-native-observation", "first": delayed, "later": delayed_later})
    page.context.close()

    page = review.page(paired + "window.reviewInitialQuest={polar:'ready',ecgGap:'first-packet'};",
                       reduced_motion="no-preference")
    expect(page.locator("#quest-polar-status")).to_contain_text("Quest empfängt EKG und Beschleunigungsdaten")
    page.clock.run_for(1100)
    for _ in range(20):
        if page.locator("#quest-ecg").is_visible():
            break
        page.clock.run_for(100)
    assert page.locator("#quest-ecg").is_visible(), page.evaluate("""() => ({
      now:performance.now(),calls:window.reviewEcgCalls,polls:window.reviewSnapshotCount,
      count:window.reviewLastEcg?.samples.length,
      hiddenAttribute:document.querySelector('#quest-ecg').hasAttribute('hidden'),
      display:getComputedStyle(document.querySelector('#quest-ecg')).display,
      status:document.querySelector('#quest-ecg-status').textContent})""")
    ecg_read = """() => {
      const path=document.querySelector('#quest-ecg-trace').getAttribute('d');
      const range=Number(document.querySelector('#quest-ecg-scale').textContent.match(/±(\\d+)/)?.[1]);
      const raw=new Set((window.reviewEcgPackets??[]).flatMap(packet=>packet.samples.map(([,v])=>(46-v/range*43).toFixed(1))));
      const packets=window.reviewEcgPackets??[];
      const values=new Map(); let consistent=true;
      for(const packet of packets) for(const [timestamp,value] of packet.samples) {
        if(values.has(timestamp)&&values.get(timestamp)!==value) consistent=false;
        values.set(timestamp,value);
      }
      return {path,polls:window.reviewSnapshotCount,calls:window.reviewEcgCalls,segments:(path.match(/M/g)??[]).length,
        rawPointsOnly:[...path.matchAll(/[ML][\\d.]+ ([\\d.]+)/g)].every(point=>raw.has(point[1])),
        overlapConsistent:consistent, latest:packets.at(-1)?.samples.at(-1)?.[0],
        commands:(window.reviewQuest.commands??[]).length,setupCommands:window.reviewSetupCommands.length};
    }"""
    ecg_first = page.evaluate(ecg_read)
    page.clock.run_for(200)
    ecg_later = page.evaluate(ecg_read)
    assert ecg_first["path"] and ecg_first["path"] != ecg_later["path"], "Buffered ECG did not move between deliveries"
    assert ecg_first["polls"] == ecg_later["polls"] and ecg_first["calls"] == ecg_later["calls"], "ECG animation added requests"
    assert ecg_first["segments"] >= 2 and ecg_later["segments"] >= 2, "ECG connected across missing native samples"
    assert ecg_first["rawPointsOnly"] and ecg_later["rawPointsOnly"], "ECG invented a voltage between raw samples"
    assert ecg_later["overlapConsistent"] and not ecg_later["commands"] and not ecg_later["setupCommands"]
    review.shot(page, "live-ecg-buffered-gap", "Live ECG · buffered raw points move while the native gap stays open",
                page.locator("#quest-ecg"), "Live presentation")
    review.regressions.append({"kind": "live-ecg-presentation", "first": ecg_first, "later": ecg_later})
    page.evaluate("""() => {
      Object.assign(window.reviewQuest,{statusDrop:true,joinDelay:400,ecgReplay:true});
      dispatchEvent(new Event('online'));
    }""")
    expect(page.locator("#quest-ecg")).to_be_hidden()
    page.evaluate("window.reviewQuest.statusDrop=false")
    page.clock.run_for(1600)
    expect(page.locator("#quest-polar-status")).to_contain_text("Quest empfängt EKG und Beschleunigungsdaten")
    expect(page.locator("#quest-ecg")).to_be_hidden()
    assert page.locator("#quest-ecg-trace").get_attribute("d") == "", "Duplicate ECG packet revived a cleared trace"
    page.evaluate("window.reviewQuest.ecgReplay=false; window.reviewQuest.joinDelay=0")
    page.clock.run_for(1100)
    expect(page.locator("#quest-ecg")).to_be_visible()
    recovered = page.evaluate(ecg_read)
    assert int(recovered["latest"]) > int(ecg_later["latest"]), "ECG did not recover on a fresh native packet"
    assert not recovered["commands"] and not recovered["setupCommands"]
    review.shot(page, "live-ecg-status-recovery", "Transient status loss · old ECG stays cleared until a new native packet",
                page.locator("#quest-ecg"), "Live presentation")
    review.regressions.append({"kind": "live-ecg-replay-recovery", "latestBefore": ecg_later["latest"],
                               "latestAfter": recovered["latest"]})
    page.context.close()


def capture_parked_calibration(review):
    initial = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({PAIRED}));" + """
      window.reviewInitialQuest={polar:'ready',access:{v:1,role:'active',active_phone:'Fixture',revision:1}};"""
    for slug, viewport, text_size in (
        ("390", VIEWPORT, ""), ("320", {"width":320,"height":568}, ""),
        ("200-text", {"width":320,"height":568}, "200%"),
    ):
        page = review.page(initial, viewport=viewport, reduced_motion="no-preference")
        enter_participant(page,then='tracking')
        page.evaluate("size=>document.documentElement.style.fontSize=size", text_size)
        page.clock.run_for(2700)
        assert page.evaluate("window.reviewSetupCommands") == [], "Opening the app started native audio"
        assert page.evaluate("window.reviewQuest.setup.playing") is False
        expect(page.locator("#preview-status")).to_have_text("Automatic ist bereit.")
        expect(page.locator("#preview-input-volume")).to_have_text("0 %")
        expect(page.locator("#preview-calibrate")).to_be_enabled()
        expect(page.locator("#preview-controller")).to_be_enabled()
        expect(page.locator("#preview-start")).to_be_hidden()
        assert page.locator("#preview-circle").evaluate("node=>node.style.transform") == "scale(0.46)"
        review.shot(page, f"parked-{slug}", f"Automatic {slug} · parked until the calibration button",
                    page.locator("#preview-calibrate"), "Parked startup")
        page.locator("#preview-calibrate").focus()
        page.keyboard.press("Enter")
        expect(page.locator("#preview-detail")).to_contain_text("zwei vollständige Atemzyklen")
        assert page.evaluate("window.reviewSetupCommands.map(command=>command.op)") == ["calibrate"]
        assert page.evaluate("window.reviewQuest.setup.playing") is True
        quality = {"accepted":150,"masked":8,"motion":5,"timing":2,"invalid":1,"usable_fraction":.9,"warning":"motion"}
        wait_setup(page, position_frames=705600, completed_frames=705600, quality=quality)
        expect(page.locator("#preview-progress")).to_have_js_property("value", 1)
        expect(page.locator("#preview-detail")).to_contain_text("Oberkörper ruhig")
        expect(page.locator("#preview-detail")).not_to_contain_text("fehlgeschlagen")
        expect(page.locator("#preview-continue")).to_be_disabled()
        review.shot(page, f"quality-masked-{slug}", f"Calibration {slug} · quality mask preserves the native 32-second attempt",
                    page.locator("#preview-detail"), "Parked startup")
        wait_setup(page, stage="testing", calibration="failed", completed_frames=1411200,
                   position_frames=1411200, failure="excessive-motion")
        show_operator(page,'input')
        expect(page.locator("#preview-calibrate")).to_be_enabled()
        page.locator("#preview-controller").focus()
        page.keyboard.press("Enter")
        expect(page.locator("#preview-continue")).to_be_enabled()
        assert page.evaluate("window.reviewSetupCommands.map(command=>command.op)") == ["calibrate","controller"]
        review.shot(page, f"parked-fallback-{slug}", f"Controller {slug} · one native alternative after calibration failure",
                    page.locator("#preview-review"), "Parked startup")
        review.regressions.append({"kind":"parked-startup","viewport":viewport,"textSize":text_size,
                                   "commands":["calibrate","controller"]})
        page.context.close()

    page = review.page(initial + "window.reviewInitialSetup={stage:'idle',playing:false,audio_state:'paused',source_clock_ready:false,input_volume01:0};")
    enter_participant(page,then='input')
    expect(page.locator("#preview-calibrate")).to_be_disabled()
    expect(page.locator("#preview-controller")).to_be_enabled()
    expect(page.locator("#preview-detail")).to_contain_text("Bewegungsdaten")
    review.shot(page, "parked-clock-warming", "Fresh streams · warming clock leaves Controller State available",
                page.locator("#preview-controller"), "Parked startup")
    page.locator("#preview-controller").focus()
    page.keyboard.press("Enter")
    expect(page.locator("#preview-continue")).to_be_enabled()
    assert page.evaluate("window.reviewSetupCommands.map(command=>command.op)") == ["controller"]
    page.context.close()


def capture_automatic_stream_recovery(review):
    paired = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({PAIRED}));"
    for slug, viewport, text_size in (
        ("390",VIEWPORT,""),("320",{"width":320,"height":568},""),
        ("200-text",{"width":320,"height":568},"200%"),
    ):
        for missing, ecg, acc, label in (
            ("ecg",False,True,"Bewegungsdaten bereit; EKG fehlt."),
            ("acc",True,False,"EKG bereit; Bewegungsdaten fehlen."),
            ("both",False,False,"EKG und Bewegungsdaten fehlen."),
        ):
            page = review.page(paired+"window.reviewInitialQuest="+json.dumps({"polar":"stream-wait",
                "ecg":ecg,"acc":acc,"autoRecovery":True})+";",viewport=viewport)
            page.evaluate("size=>document.documentElement.style.fontSize=size",text_size)
            expect(page.locator("#quest-polar-status")).to_have_text("Polar H10 per Bluetooth verbunden. "+label+
                " Die Quest startet fehlende Messsignale automatisch erneut.")
            expect(page.locator("#quest-polar-connect")).to_be_hidden()
            page.clock.run_for(3100)
            assert page.evaluate("window.reviewQuest.commands??[]") == [], "The phone wrote automatic stream commands"
            review.shot(page,f"automatic-{missing}-{slug}",f"Native recovery {slug} · missing {missing}, no phone activation",
                        page.locator("#quest-polar-status"),"Automatic Polar recovery")
            wait_state(page,autoRecovery=False)
            expect(page.locator("#quest-polar-connect")).to_be_visible()
            expect(page.locator("#quest-polar-status")).not_to_contain_text("automatisch")
            assert page.evaluate("window.reviewQuest.commands??[]") == []
            wait_state(page,autoRecovery=True,ecg=True,acc=True)
            expect(page.locator("#quest-polar-status")).to_have_text("Polar H10 verbunden. Quest empfängt EKG und Beschleunigungsdaten.")
            expect(page.locator("#quest-polar-connect")).to_be_disabled()
            review.regressions.append({"kind":"native-stream-recovery","viewport":viewport,"missing":missing,"commands":[]})
            page.context.close()


def capture_audio_recovery(review):
    paired = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({PAIRED}));"
    for slug, viewport, text_size in (
        ("390",VIEWPORT,""),("320",{"width":320,"height":568},""),
        ("200-text",{"width":320,"height":568},"200%"),
    ):
        page = review.page(paired+"window.reviewInitialQuest={polar:'ready',audioRecovery:true};"
            "window.reviewInitialSetup={stage:'testing',playing:true,calibration:'failed',failure:'clock_stalled',"
            "clock_valid:false,source_clock_ready:false,audio_recovery_available:true};", viewport=viewport)
        page.evaluate("size=>document.documentElement.style.fontSize=size",text_size)
        enter_participant(page,then='input')
        page.clock.run_for(1100)
        calibrate, fallback = page.locator("#preview-calibrate"), page.locator("#preview-controller")
        detail, proceed = page.locator("#preview-detail:visible, #preview-recovery:visible"), page.locator("#preview-continue")
        expect(page.locator("#preview-circle")).to_be_hidden()
        expect(page.locator("#preview-input-volume")).to_be_hidden()
        expect(page.locator("#preview-input-label")).not_to_contain_text("Rohwert der Eingabe")
        assert page.evaluate("window.reviewQuest.setup.playing") is True
        expect(calibrate).to_be_enabled()
        calibrate.click()
        page.clock.run_for(1100)
        expect(detail).to_contain_text("Atemschleife")
        expect(detail).to_contain_text("Bitte warten")
        expect(calibrate).to_be_disabled()
        expect(fallback).to_be_disabled()
        expect(proceed).to_be_disabled()
        expect(page.locator("#preview-input-volume")).to_be_hidden()
        expect(page.locator("#preview-circle")).to_be_hidden()
        assert page.evaluate("window.reviewQuest.setup.playing") is True
        assert page.locator("#preview-progress").evaluate("node=>node.value") == 0
        assert len(page.evaluate("window.reviewSetupCommands")) == 1
        operation_id = page.evaluate("window.reviewSetupCommands[0].id")
        review.shot(page,f"audio-recovery-pending-{slug}",f"Audio recovery {slug} · admitted, no completed calibration",detail,"Audio recovery")
        page.clock.run_for(14000)
        recheck = page.locator("#preview-acknowledge")
        expect(recheck).to_have_text("Bestätigung erneut abfragen")
        expect(recheck).to_be_enabled()
        queries = page.evaluate("window.reviewSetupReceiptQueries")
        assert 1 <= len(queries) <= 12
        assert all(query == {"v":1,"id":operation_id,"op":"receipt"} for query in queries)
        before = len(queries)
        recheck.click()
        page.clock.run_for(1100)
        assert len(page.evaluate("window.reviewSetupReceiptQueries")) > before
        assert len(page.evaluate("window.reviewSetupCommands")) == 1
        wait_state(page,setupReceiptPatch={"state":"confirmed","accepted":True,"reason":"accepted","recovery_generation":2})
        wait_setup(page,g=2,r=1,stage="waiting",calibration="waiting",attempt=1,completed_frames=0,
            target_frames=1411200,recovery=None,clock_valid=True,source_clock_ready=True,audio_generation=2)
        page.clock.run_for(1100)
        expect(detail).to_contain_text("nächsten vollständigen Atemzyklus")
        expect(proceed).to_be_disabled()
        assert page.evaluate("sessionStorage.getItem('atmungsraum-breathing-setup-pending')") is None
        assert len(page.evaluate("window.reviewSetupCommands")) == 1
        review.shot(page,f"audio-recovery-confirmed-{slug}",f"Audio recovery {slug} · confirmed, collection still pending",detail,"Audio recovery")
        wait_setup(page,stage="testing",calibration="failed",failure="clock_stalled",clock_valid=False,
            source_clock_ready=False,audio_recovery_available=True)
        show_operator(page,'input')
        calibrate.click()
        page.clock.run_for(1100)
        wait_state(page,setupReceiptPatch={"state":"rejected","accepted":False,"reason":"audio-recovery-timeout","recovery_generation":None})
        wait_setup(page,stage="testing",recovery=None,audio_recovery_available=True)
        page.clock.run_for(1100)
        expect(detail).to_contain_text("Atemschleife konnte nicht")
        expect(calibrate).to_be_enabled()
        expect(fallback).to_be_enabled()
        expect(proceed).to_be_disabled()
        assert len(page.evaluate("window.reviewSetupCommands")) == 2
        review.shot(page,f"audio-recovery-rejected-{slug}",f"Audio recovery {slug} · terminal failure allows explicit retry",detail,"Audio recovery")
        wait_setup(page,audio_recovery_available=False,source_clock_ready=True)
        expect(detail).to_contain_text("Atmungsraum-App im Headset")
        expect(calibrate).to_be_disabled()
        expect(fallback).to_be_disabled()
        expect(proceed).to_be_disabled()
        review.shot(page,f"audio-recovery-unavailable-{slug}",f"Audio recovery {slug} · unavailable owner requires reopening Quest app",detail,"Audio recovery")
        review.regressions.append({"kind":"audio-recovery","viewport":viewport,"text_size":text_size,
            "original_id":operation_id,"queries":queries,"mutation_count":2,"fabricated_progress":False})
        page.context.close()


def capture_setup_receipt_fence(review):
    paired = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({PAIRED}));"
    operation_id = "f" * 16
    journal = {"owner":{"serial":"DEMO-QUEST","device":"d" * 32,"epoch":"7","native_epoch":"8"},
        "command":{"v":1,"id":operation_id,"g":1,"r":0,"op":"calibrate"}}
    initial = paired + "window.reviewInitialQuest={polar:'ready',receiptState:'outcome_unknown'};" + \
        "window.reviewInitialSetup={g:2,r:1,stage:'waiting',calibration:'waiting',attempt:1,recovery:null,target_frames:1411200};" + \
        "sessionStorage.setItem('atmungsraum-breathing-setup-pending',JSON.stringify(" + json.dumps(journal) + "));"
    for slug, viewport, text_size in (
        ("390",VIEWPORT,""),("320",{"width":320,"height":568},""),
        ("200-text",{"width":320,"height":568},"200%"),
    ):
        page = review.page(initial,viewport=viewport)
        page.evaluate("size=>document.documentElement.style.fontSize=size",text_size)
        recheck = page.locator("#preview-acknowledge")
        expect(recheck).to_have_text("Bestätigung erneut abfragen")
        expect(recheck).to_be_enabled()
        assert page.evaluate("JSON.parse(sessionStorage.getItem('atmungsraum-breathing-setup-pending')).recovery??null") is None
        before = page.evaluate("(window.reviewSetupReceiptQueries??[]).length")
        recheck.click()
        page.clock.run_for(1100)
        assert page.evaluate("window.reviewSetupReceiptQueries.length") > before
        page.clock.run_for(14000)
        expect(recheck).to_have_text("Bestätigung erneut abfragen")
        expect(recheck).to_be_enabled()
        for patch in ({"state":"confirmed"},
            {"state":"confirmed","accepted":True,"reason":"accepted","recovery_generation":3}):
            page.evaluate("patch=>window.reviewQuest.setupReceiptPatch=patch",patch)
            recheck.click()
            page.clock.run_for(1100)
            expect(recheck).to_have_text("Bestätigung erneut abfragen")
            expect(page.locator("#preview-calibrate")).to_be_disabled()
            expect(page.locator("#preview-controller")).to_be_disabled()
            expect(page.locator("#preview-continue")).to_be_disabled()
            assert page.evaluate("JSON.parse(sessionStorage.getItem('atmungsraum-breathing-setup-pending')).command.id") == operation_id
            assert page.evaluate("window.reviewSetupCommands") == []
        queries = page.evaluate("window.reviewSetupReceiptQueries")
        assert all(query == {"v":1,"id":operation_id,"op":"receipt"} for query in queries)
        review.shot(page,f"receipt-fence-unknown-{slug}",f"Lost recovery admission {slug} · recheck preserves original ID",recheck,"Setup receipt fence")
        recheck.focus()
        page.evaluate("window.reviewQuest.setupReceiptPatch={state:'confirmed',accepted:true,reason:'accepted',recovery_generation:2}")
        page.keyboard.press('Enter')
        page.clock.run_for(1100)
        assert page.evaluate("sessionStorage.getItem('atmungsraum-breathing-setup-pending')") is None
        assert page.evaluate("window.reviewSetupCommands") == []
        enter_participant(page,then='input')
        expect(page.locator("#preview-detail")).to_contain_text("nächsten vollständigen Atemzyklus")
        expect(page.locator("#preview-continue")).to_be_disabled()
        assert page.locator("#preview-progress").evaluate("node=>node.value") == 0
        review.shot(page,f"receipt-fence-confirmed-{slug}",f"Lost recovery admission {slug} · matching receipt, collection remains native",page.locator("#preview-detail"),"Setup receipt fence")
        review.regressions.append({"kind":"lost-admission-receipt-fence","viewport":viewport,"text_size":text_size,
            "original_id":operation_id,"queries":queries,"mutations":[],"same_actor_audio_change_acknowledged":False})
        page.context.close()


def main():
    parser = ArgumentParser(description="Capture rendered Atmungsraum phone screens into a review ZIP.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this capture")
    parser.add_argument("--scope", choices=("all", "layout", "pairing", "control", "location", "scroll", "polar", "live", "parked", "automatic", "audio", "receipts"), default="all",
                        help="Full suite or focused pairing, control, Location, scroll, Polar, or live presentation")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / "screens").mkdir()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        chromium = browser.version
        review = Review(browser, output)
        if args.scope in ("all", "layout"):
            from layout_render_fixtures import capture_companion_permission, capture_layout
            capture_layout(review)
            capture_companion_permission(review)
        if args.scope == "all":
            capture(review)
            paired = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({PAIRED}));"
            capture_setup_recovery(review, paired)
            capture_accessibility(review, paired)
        if args.scope in ("all", "pairing"):
            capture_phone_pairing(review)
        if args.scope in ("all", "control"):
            capture_controller_access(review)
        if args.scope in ("all", "control", "location"):
            capture_polar_location(review)
        if args.scope in ("all", "scroll"):
            capture_scroll_regression(review)
        if args.scope in ("all", "polar"):
            capture_polar_stream_recovery(review)
        if args.scope in ("all", "live"):
            capture_live_display(review)
        if args.scope in ("all", "parked"):
            capture_parked_calibration(review)
        if args.scope in ("all", "automatic"):
            capture_automatic_stream_recovery(review)
        if args.scope in ("all", "audio"):
            capture_audio_recovery(review)
        if args.scope == "receipts":
            paired = f"localStorage.setItem('atmungsraum-quest-pair',JSON.stringify({PAIRED}));"
            capture_setup_recovery(review,paired)
        if args.scope in ("all", "receipts"):
            capture_setup_receipt_fence(review)
        browser.close()
    review.index()
    shutil.copyfile(ROOT / "for-ai" / "PHONE-SCREENSHOTS.md", output / "protocol.md")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    revision = git("rev-parse", "HEAD")
    changes = git("status", "--porcelain", "--", "*.js", "*.css", "*.html", "assets",
                  "tools/capture-phone-screens.py", "tools/layout_render_fixtures.py",
                  "tools/test-operator-layout.py", "for-ai/PHONE-SCREENSHOTS.md")
    (output / "source.txt").write_text(
        f"App checkout: {revision}\nPlaywright: {version('playwright')}\nChromium: {chromium}\n"
        f"Source working tree:\n{changes or 'clean'}\n", encoding="utf-8"
    )
    with zipfile.ZipFile(output / ARCHIVE, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.rglob("*")):
            if file.is_file() and file.name != ARCHIVE:
                archive.write(file, file.relative_to(output))
    print(f"{len(review.rows)} rendered states with viewport tiles -> {output / ARCHIVE}")


if __name__ == "__main__":
    main()
