import { BoxBreathingEngine } from './box-breathing.js?v=20261106-placeholders';
import { state, start } from '../ui-logic.js?v=20261106-polar-timeout';
import { participantIdentity } from '../data-logging.js?v=20261105-data';

// This controller owns the final-stage audio; routing and the demo ECG stay in app.js.
export function mountSessionAudio({ data } = {}) {
  const stage = document.getElementById('session-start');
  const play = document.getElementById('play');
  const back = document.getElementById('session-back');
  const phase = document.getElementById('session-phase');
  const clock = document.getElementById('session-time');
  const duration = document.getElementById('session-duration');
  const progress = document.getElementById('session-progress');
  const status = document.getElementById('start-status');
  const labels = { instruction_1: 'Vorbereitung', calibration: 'Zwei tiefe Atemzüge',
    instruction_2: 'Anleitung zur Atmung', instruction: 'Willkommen', inhale: 'Einatmen', hold_in: 'Atempause',
    exhale: 'Ausatmen', hold_out: 'Atempause', rest: 'Pause · natürlich atmen',
    outro: 'Ausklang', ending: 'Ausklang', recovery: 'Natürlich atmen' };
  const text = (element, value) => { if (element.textContent !== value) element.textContent = value; };
  const format = (seconds) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
  let engine, preparation, key = null, gate = false, mode = 'idle', message = '', token = 0, frame = 0;
  let active = null;
  let totalSeconds = 0;
  let tick = () => {};
  let drawPosition = () => {};

  function showProgress(seconds = 0) {
    const elapsed = Math.max(0, Math.min(totalSeconds, seconds));
    const elapsedLabel = format(Math.floor(elapsed));
    const totalLabel = format(totalSeconds);
    text(clock, elapsedLabel);
    text(duration, totalLabel);
    progress.max = totalSeconds || 1;
    progress.value = elapsed;
    progress.setAttribute('aria-valuetext', `${elapsedLabel} von ${totalLabel}`);
  }

  function paint() {
    const busy = ['playing', 'starting', 'pausing', 'paused', 'resuming'].includes(mode);
    stage.dataset.audioState = mode;
    stage.dataset.sessionActive = String(busy);
    stage.dataset.protocolMinutes = state.selection?.value ?? '';
    play.disabled = !key || (mode !== 'playing' && (!gate || state.sensor !== 'ecg'))
      || ['loading', 'starting', 'pausing', 'resuming'].includes(mode)
      || Boolean(engine?.starting || engine?.pausing || engine?.resuming);
    const label = mode === 'error' && !engine?.buffers ? 'Audio erneut laden'
      : mode === 'playing' ? 'Pause' : mode === 'paused' ? 'Fortsetzen'
      : mode === 'pausing' ? 'Pausieren …' : ['starting', 'resuming'].includes(mode) ? 'Startet …'
      : mode === 'completed' ? 'Erneut starten' : 'Starten';
    text(play.querySelector('[data-measure]'), label);
    play.setAttribute('aria-label', label);
    play.dataset.action = mode === 'playing' ? 'pause' : mode === 'paused' ? 'resume' : 'start';
    play.querySelector('path').setAttribute('d', mode === 'playing'
      ? 'M16 10H28V54H16ZM36 10H48V54H36Z' : 'M18 10 54 32 18 54Z');
    text(back.querySelector('[data-measure]'), busy ? 'Beenden' : '← Zurück');
    back.setAttribute('aria-label', busy ? 'Session beenden und zurück' : 'Zurück');
    text(status, message);
    status.hidden = !message;
  }

  // Participant-linked receipts follow the same identity as the session CSV.
  function save(receipt, attempt) {
    if (!receipt || attempt.saved) return true;
    attempt.recording?.stop();
    if (attempt.recordingPending) { attempt.pendingReceipt = receipt; return true; }
    data?.finish(receipt, attempt);
    try {
      const stored = JSON.parse(localStorage.getItem('atmungsraum.audio-receipts') ?? '[]');
      const records = Array.isArray(stored) ? stored.slice(-9) : [];
      records.push({ ...receipt, run_id: attempt.id, selection: attempt.selection,
        schema_version: 1, session_id: attempt.id,
        participant_id: attempt.participant.id, name_prefix: attempt.participant.name,
        age_years: attempt.participant.age, gender_code: attempt.participant.genderCode,
        condition: attempt.selection.branch === 'haupt' ? 'main' : 'probe', run_index: attempt.recording?.runIndex ?? null,
        finished_at: new Date().toISOString() });
      localStorage.setItem('atmungsraum.audio-receipts', JSON.stringify(records));
      attempt.saved = true;
      return true;
    } catch (error) {
      console.warn('Audio receipt could not be saved.', error);
      return false;
    }
  }

  function end(reason = 'user-abort') {
    ++token;
    cancelAnimationFrame(frame);
    engine?.abort(reason);
    // A cancelled resume() may leave the preceding run's receipt in the engine.
    // Never attach that old protocol to a new attempt that has not started.
    const receipt = engine?.receipt();
    if (active?.startAt != null && receipt?.start_at_audio_seconds === active.startAt) save(receipt, active);
    state.started = false;
    if (['playing', 'starting', 'pausing', 'paused', 'resuming'].includes(mode)) {
      mode = 'stopped';
      message = 'Session beendet.';
      text(phase, 'Beendet');
    }
    paint();
  }

  function prepare() {
    if (preparation) return preparation;
    engine ??= new BoxBreathingEngine();
    mode = 'loading'; message = 'Audio wird geladen …'; paint();
    preparation = Promise.all([engine.prepare(), data?.prepare()]).then(() => {
      mode = 'ready'; message = '';
      resetProgress(state.selection);
      if (key) paint();
    }).catch((error) => {
      preparation = null;
      mode = 'error'; message = 'Audio konnte nicht geladen werden. Versuche es erneut.';
      console.warn('Audio preparation failed.', error);
      if (key) paint();
    });
    return preparation;
  }

  play.addEventListener('click', async () => {
    if (mode === 'playing') { await pause(); return; }
    if (mode === 'paused') { await resume(); return; }
    if (!key || !gate) return;
    if (!engine?.buffers) { prepare(); return; }
    if (!start()) return;
    const ownToken = ++token;
    const attempt = { id: crypto.randomUUID(), patient: { ...state.patient }, participant: participantIdentity(state.patient),
      selection: { ...state.selection }, saved: false, recordingPending: Boolean(data) };
    active = attempt;
    mode = 'starting'; message = 'Audio startet …'; paint();
    try {
      // resume() is called from this user action; assets were prepared beforehand.
      const session = await engine.start(attempt.selection.minutes ?? 'anleitung',
        { mainRound: attempt.selection.branch === 'haupt' });
      attempt.startAt = session.startAt;
      const calibration = engine.receipt().calibration;
      if (calibration) window.dispatchEvent(new CustomEvent('atmungsraum:calibration-scheduled', {
        detail: { session_id: attempt.id, scenario_id: session.protocol.scenario_id,
          sample_rate: session.protocol.sample_rate, clock: 'AudioContext.currentTime', ...calibration },
      }));
      attempt.recording = data?.begin({ patient: attempt.patient, selection: attempt.selection, sessionId: attempt.id,
        protocol: { ...session.protocol, scenario_sha256: engine.receipt().scenario_sha256 },
        audioStartDelayMs: Math.max(0, (session.startAt - engine.context.currentTime) * 1000) });
      await attempt.recording?.ready;
      attempt.recordingPending = false;
      if (attempt.pendingReceipt) save(attempt.pendingReceipt, attempt);
      if (ownToken !== token) { save(await session.done, attempt); return; }
      mode = 'playing'; message = '';
      totalSeconds = session.protocol.total_frames / session.protocol.sample_rate;
      stage.dataset.audioStartTime = String(session.startAt);
      stage.dataset.audioEndTime = String(session.endAt);
      stage.dataset.audioSampleRate = String(engine.context.sampleRate);
      drawPosition = () => {
        const position = engine.position();
        const event = position?.event;
        const beat = event?.cycle ? Math.min(4, 1 + Math.floor((position.elapsed_seconds
          - event.offset_frames / session.protocol.sample_rate))) : null;
        text(phase, `${labels[event?.id] ?? 'Bereit'}${beat ? ` · ${beat}` : ''}`);
        showProgress(position?.elapsed_seconds ?? 0);
        stage.dataset.audioPhase = event?.id ?? 'waiting';
        stage.dataset.audioVariant = event?.asset ?? '';
        stage.dataset.audioCycle = String(event?.cycle ?? '');
      };
      tick = () => {
        if (ownToken !== token || mode !== 'playing') return;
        drawPosition();
        frame = requestAnimationFrame(tick); // Display only; it never starts audio.
      };
      paint(); tick();
      const receipt = await session.done;
      const saved = save(receipt, attempt);
      if (ownToken !== token) return;
      cancelAnimationFrame(frame);
      state.started = false;
      if (receipt.status === 'completed') {
        mode = 'completed'; text(phase, 'Fertig'); showProgress(totalSeconds);
        message = attempt.selection.minutes === null
          ? 'Anleitung beendet. Du kannst sie erneut anhören.' : 'Session abgeschlossen.';
      } else {
        mode = 'stopped'; text(phase, 'Unterbrochen');
        message = 'Wiedergabe unterbrochen. Starte eine neue Session.';
      }
      if (!saved) message += ' Das Audio-Protokoll konnte nicht gespeichert werden.';
      paint();
    } catch (error) {
      attempt.recordingPending = false;
      // A failed old reservation must never abort or relabel a newer run.
      const ownsRun = attempt.startAt != null && engine.receipt()?.start_at_audio_seconds === attempt.startAt;
      if (ownToken === token && active === attempt) engine?.abort('start-or-data-failed');
      if (attempt.pendingReceipt) save(attempt.pendingReceipt, attempt);
      else if (ownsRun) save(engine.receipt(), attempt);
      if (ownToken === token) {
        state.started = false; mode = 'error';
        message = 'Session konnte nicht gestartet oder gespeichert werden. Versuche es erneut.';
        console.warn('Audio start failed.', error);
      }
    } finally { paint(); }
  });
  async function pause() {
    if (mode !== 'playing') return;
    const ownToken = token;
    mode = 'pausing';
    cancelAnimationFrame(frame);
    paint();
    try {
      const paused = await engine.pause();
      if (ownToken !== token || mode !== 'pausing' || !paused) return;
      mode = 'paused'; message = '';
      active?.recording?.mark('paused', engine.context.currentTime - active.startAt);
      drawPosition(); paint();
    } catch (error) {
      if (ownToken === token) {
        end('pause-failed'); mode = 'error'; message = 'Pause nicht möglich. Starte neu.';
        console.warn('Audio pause failed.', error);
      }
    } finally { paint(); }
  }
  async function resume() {
    if (mode !== 'paused' || !gate || state.sensor !== 'ecg') return;
    const ownToken = token;
    mode = 'resuming'; paint();
    try {
      const resumed = await engine.resume();
      if (ownToken !== token || mode !== 'resuming' || !resumed) return;
      mode = 'playing'; message = '';
      active?.recording?.mark('playing', engine.context.currentTime - active.startAt);
      paint(); tick();
    } catch (error) {
      if (ownToken === token) {
        end('resume-failed'); mode = 'error'; message = 'Fortsetzen nicht möglich. Starte neu.';
        console.warn('Audio resume failed.', error);
      }
    } finally { paint(); }
  }
  back.addEventListener('click', () => end('navigation'));
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && ['playing', 'starting', 'pausing', 'paused', 'resuming'].includes(mode)) { event.preventDefault(); end('escape'); }
  });
  window.addEventListener('pagehide', () => end('page-hidden'));

  function resetProgress(selection) {
    totalSeconds = selection?.minutes ? selection.minutes * 60 : 0;
    if (selection && engine?.buffers) {
      const protocol = engine.protocol(selection.minutes ?? 'anleitung',
        { mainRound: selection.branch === 'haupt' });
      totalSeconds = protocol.total_frames / protocol.sample_rate;
    }
    showProgress();
  }

  return {
    update(screen, selection, hasSignal) {
      active?.recording?.markSensor(state.sensor);
      gate = hasSignal;
      const next = screen === 'session-start' && hasSignal && selection
        ? `${selection.branch}:${selection.value}` : null;
      if (next !== key) {
        end('navigation-or-signal-changed');
        key = next;
        active = null;
        mode = engine?.buffers ? 'ready' : preparation ? 'loading' : 'idle';
        message = '';
        text(phase, 'Bereit');
        resetProgress(selection);
        delete stage.dataset.audioPhase;
      }
      if (key && !engine?.buffers && !preparation) prepare();
      paint();
    },
  };
}
