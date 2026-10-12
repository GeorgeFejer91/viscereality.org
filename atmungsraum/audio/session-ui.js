import { BoxBreathingEngine } from './box-breathing.js?v=20261012-pause';
import { state, start } from '../ui-logic.js';

// This controller owns the final-stage audio; routing and the demo ECG stay in app.js.
export function mountSessionAudio() {
  const stage = document.getElementById('session-start');
  const play = document.getElementById('play');
  const stop = document.getElementById('stop');
  const controls = document.getElementById('session-controls');
  const resume = document.getElementById('session-resume');
  const pause = document.getElementById('session-pause');
  const phase = document.getElementById('session-phase');
  const clock = document.getElementById('session-time');
  const status = document.getElementById('start-status');
  const labels = { instruction: 'Willkommen', inhale: 'Einatmen', hold_in: 'Atempause',
    exhale: 'Ausatmen', hold_out: 'Atempause', rest: 'Pause · natürlich atmen',
    ending: 'Ausklang', recovery: 'Natürlich atmen' };
  const text = (element, value) => { if (element.textContent !== value) element.textContent = value; };
  const format = (seconds) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
  let engine, preparation, key = null, gate = false, mode = 'idle', message = '', token = 0, frame = 0;
  let active = null;
  let tick = () => {};
  let drawPosition = () => {};

  function paint() {
    const busy = ['playing', 'starting', 'pausing', 'paused', 'resuming'].includes(mode);
    stage.dataset.audioState = mode;
    stage.dataset.sessionActive = String(busy);
    stage.dataset.protocolMinutes = state.selection?.value ?? '';
    play.disabled = !key || !gate || state.sensor !== 'ecg' || busy || mode === 'loading'
      || Boolean(engine?.starting || engine?.pausing || engine?.resuming);
    controls.hidden = !busy;
    pause.disabled = mode !== 'playing';
    resume.disabled = mode !== 'paused' || !gate || state.sensor !== 'ecg';
    stop.disabled = !busy;
    const label = mode === 'error' && !engine?.buffers ? 'Audio erneut laden'
      : state.selection?.minutes === null ? 'Anleitung starten' : 'Session starten';
    text(play.querySelector('[data-measure]'), label);
    play.setAttribute('aria-label', label);
    text(status, message);
    clock.hidden = state.selection?.minutes === null;
  }

  // Receipts contain timing and selected scenario, never participant data or credentials.
  function save(receipt, attempt) {
    if (!receipt || attempt.saved) return true;
    try {
      const stored = JSON.parse(localStorage.getItem('atmungsraum.audio-receipts') ?? '[]');
      const records = Array.isArray(stored) ? stored.slice(-9) : [];
      records.push({ ...receipt, run_id: attempt.id, selection: attempt.selection,
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
    preparation = engine.prepare().then(() => {
      mode = 'ready'; message = 'Audio bereit.';
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
    if (!key || !gate) return;
    if (!engine?.buffers) { prepare(); return; }
    if (!start()) return;
    const ownToken = ++token;
    const attempt = { id: crypto.randomUUID(), selection: { ...state.selection }, saved: false };
    active = attempt;
    mode = 'starting'; message = 'Audio startet …'; paint();
    try {
      // resume() is called from this user action; assets were prepared beforehand.
      const session = await engine.start(attempt.selection.minutes ?? 'anleitung');
      attempt.startAt = session.startAt;
      if (ownToken !== token) return;
      mode = 'playing'; message = 'Folge den Ansagen.';
      stage.dataset.audioStartTime = String(session.startAt);
      stage.dataset.audioEndTime = String(session.endAt);
      stage.dataset.audioSampleRate = String(engine.context.sampleRate);
      drawPosition = () => {
        const position = engine.position();
        const event = position?.event;
        const beat = event?.cycle ? Math.min(4, 1 + Math.floor((position.elapsed_seconds
          - event.offset_frames / session.protocol.sample_rate))) : null;
        text(phase, `${labels[event?.id] ?? 'Bereit'}${beat ? ` · ${beat}` : ''}`);
        const remaining = Math.max(0, Math.min(session.protocol.total_frames / session.protocol.sample_rate,
          Math.ceil(session.endAt - engine.context.currentTime)));
        text(clock, format(remaining));
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
        mode = 'completed'; text(phase, 'Fertig'); text(clock, '00:00');
        message = attempt.selection.minutes === null
          ? 'Anleitung beendet. Du kannst sie erneut anhören.' : 'Session abgeschlossen.';
      } else {
        mode = 'stopped'; text(phase, 'Unterbrochen');
        message = 'Wiedergabe unterbrochen. Starte eine neue Session.';
      }
      if (!saved) message += ' Das Audio-Protokoll konnte nicht gespeichert werden.';
      paint();
    } catch (error) {
      if (ownToken === token) {
        state.started = false; mode = 'error';
        message = 'Audio konnte nicht starten. Versuche es erneut.';
        console.warn('Audio start failed.', error);
      }
    } finally { paint(); }
  });
  pause.addEventListener('click', async () => {
    if (mode !== 'playing') return;
    const ownToken = token;
    const moveFocus = document.activeElement === pause;
    mode = 'pausing';
    cancelAnimationFrame(frame);
    paint();
    try {
      const paused = await engine.pause();
      if (ownToken !== token || mode !== 'pausing' || !paused) return;
      mode = 'paused'; message = 'Pausiert.';
      drawPosition(); paint();
      if (moveFocus) resume.focus?.({ preventScroll: true });
    } catch (error) {
      if (ownToken === token) {
        end('pause-failed'); mode = 'error'; message = 'Pause nicht möglich. Starte neu.';
        console.warn('Audio pause failed.', error);
      }
    } finally { paint(); }
  });
  resume.addEventListener('click', async () => {
    if (mode !== 'paused' || !gate || state.sensor !== 'ecg') return;
    const ownToken = token;
    const moveFocus = document.activeElement === resume;
    mode = 'resuming'; paint();
    try {
      const resumed = await engine.resume();
      if (ownToken !== token || mode !== 'resuming' || !resumed) return;
      mode = 'playing'; message = 'Folge den Ansagen.';
      paint(); tick();
      if (moveFocus) pause.focus?.({ preventScroll: true });
    } catch (error) {
      if (ownToken === token) {
        end('resume-failed'); mode = 'error'; message = 'Fortsetzen nicht möglich. Starte neu.';
        console.warn('Audio resume failed.', error);
      }
    } finally { paint(); }
  });
  stop.addEventListener('click', () => end());
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && ['playing', 'starting', 'pausing', 'paused', 'resuming'].includes(mode)) { event.preventDefault(); end('escape'); }
  });
  window.addEventListener('pagehide', () => end('page-hidden'));

  return {
    update(screen, selection, hasSignal) {
      gate = hasSignal;
      const next = screen === 'session-start' && hasSignal && selection
        ? `${selection.branch}:${selection.value}` : null;
      if (next !== key) {
        end('navigation-or-signal-changed');
        key = next;
        active = null;
        mode = engine?.buffers ? 'ready' : preparation ? 'loading' : 'idle';
        message = engine?.buffers ? 'Audio bereit.' : '';
        text(phase, 'Bereit');
        text(clock, selection?.minutes ? format(selection.minutes * 60) : '');
        delete stage.dataset.audioPhase;
      }
      if (key && !engine?.buffers && !preparation) prepare();
      paint();
    },
  };
}
