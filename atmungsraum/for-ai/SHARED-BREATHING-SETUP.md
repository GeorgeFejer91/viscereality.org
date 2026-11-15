# Shared breathing scheduler and setup calibration

Approved user design, 2026-11-14. PROJECT.md remains the lifecycle owner.

## Accepted behavior

Automatic is the Atmungsraum startup input. One native audio-frame schedule owns
cue playback, intended breathing phase, automatic visual input, calibration
labels, and experiment boundaries. Inhale, hold-in, exhale and hold-out each last
four seconds. Preparation loops indefinitely. The phone projects native state;
browser timers never advance the schedule.

Polar stays connected and acquiring. Calibration selects Automatic and collects
from the next inhale boundary through two full cycles (32 seconds). Progress
uses actual audio frames and accepted samples. Missing, stale, malformed,
reordered, motion-contaminated or insufficient-range data fails the attempt.
Elapsed time alone never proves success. A successful model is activated as
Polar tilt atomically while the loop continues. Retry repeats this process.
Controller State with fixed orientation cancels an outstanding calibration,
keeps the same audio clock and uses the existing controller assessment owner.
Fallback selection stays available after Polar tracking becomes unreliable.
Starting the participant run still requires the study's fresh H10 streams;
changing the breathing driver does not remove ECG/ACC capture requirements.

Start experiment retains the chosen input/model and begins at the next full
cycle boundary. The main experiment lasts exactly 900 seconds (user confirmed).
Its recording clock starts at that boundary. Preparation is not experiment data.
The old embedded 15-second calibration and introductory restart are retired.
Other Rusty Quest products keep their existing defaults and feature activation.

## Buffered Polar timing — 2026-11-15

Idle Automatic does not validate tilt evidence that no attempt or model
consumes. Raw acquisition and source ownership remain active; real source
changes still invalidate old command revisions. Repeated idle diagnostics must
not churn control revisions or overwrite a terminal calibration failure.

Map original sample timestamps into actual healthy, playing, same-generation
audio observations. Retain at most 32 observations covering a 250 ms history
window plus a bracket; never extrapolate a missing frame. At most 64 future
training labels may wait for the next actual clock bracket. Pause, retirement,
controller cancellation and source changes clear evidence at their relevant
barriers. Original acquisition timestamps remain unchanged.

Sensor timestamp gaps and driver readiness stay at 200 ms; sensor/host drift
tolerance stays at 150 ms. A separate 250 ms accepted-ingress deadline admits
normal 200 ms delivery with modest jitter while failing missing packets.
Temporary unready output cannot fabricate missing sensor history. Fit only
after the full two-cycle audio window and an accepted sample watermark within
40 ms of its end, one 25 Hz interval. Wait only within the bounded delivery
allowance for the final packet. Elapsed time alone cannot fit.

Host tests cover valid 200 Hz and 25 Hz batches, 20 ms jitter, missing packets,
genuine sensor gaps, stale clocks, pauses, source changes, history bounds and
queued experiment boundaries. Physical H10 callback timing and wearer fit
remain a separate acceptance gate.

## Design

One JSON defines the loop's sample rate, phases, frame counts and cue identities.
The Android owner loads that cycle into one looping PCM AudioTrack. Playback-head
frames drive the native scheduler; all consumers read the same phase. Pause and
resume preserve the player. Sensor observations retain their own timestamps,
aligned through Android monotonic time with bounded clock freshness.
Polar's original Java nano-time timestamps remain on raw records. A measured,
fresh nano-time/elapsed-realtime pair maps samples onto the audio clock for
calibration; labeling the two clocks identically is invalid after device sleep.

A participant tilt model learns a directed movement axis and robust range from
normalized three-axis gravity direction. Scheduled labels describe instructions,
not proof of compliance: both cycles must contain valid directional motion.
The existing scalar phase classifier supplies confirmation/hold/dwell behavior.
The model belongs to the Polar source owner and calibration generation. Source
replacement invalidates it. Collection uses unbiased samples and is bounded.

Closed setup commands and bounded status use the existing enrolled LAN gateway.
Preserve Origin, identity, epoch, authentication, replay and command-ID checks.
Guest/relay routes cannot mutate calibration. The phone shows native progress,
failure/retry and input testing through existing Pretext helpers. Older APKs
report the required update rather than simulating setup success.
The selected app offers `haupt-15` after unlimited preparation. The archived
timed practice scenarios and local menu prompts remain available to unselected
builds; they cannot start another guidance player over the selected loop.

The calibrated tilt path currently supports zero guidance bias. A nonzero arm
or live-bias request is rejected as `setup-guidance-bias-unsupported`, and no
live-bias capability is advertised. Shared intended phases are available, but
the older tuned XZ classifier's private bias policy is not applied to the new
participant model. Existing unselected builds retain that policy.

The separate Meta distribution chat is 01a1db90-5317-7191-acb5-db6700ef5fb4.
It owns account/channel setup only, not these feature sources.

## Implementation plan

Goal: synchronized preparation and main execution with repeatable tilt
calibration and controller fallback. Existing For-AI state owns continuation.
Tech: Rust, Android Java/OpenXR, plain JavaScript, Pretext 0.0.9 and existing
Rust/Android/Node checks. Implement inline with tests before production changes.

### Native scheduler and model

- [ ] Add `experiment_breathing_setup.rs`: typed `SetupInput`, `SetupPhase`,
  `SetupCommand`, `SetupStatus`, `TiltModel`, `BreathingSetup`.
- [ ] First test exact phase boundaries: frames 0/176400/352800/529200/705600
  produce inhale/hold-in/exhale/hold-out/inhale. Calibration starts on the next
  cycle and consumes exactly 1,411,200 frames. Main duration is 39,690,000 frames.
- [ ] Test valid guided motion, no motion, missing phases, stale/reordered data,
  excessive motion, source replacement, replay/conflict, retry, fallback and
  next-cycle experiment admission. Failure never activates a model.
- [ ] Implement bounded collection, tilt projection and native lifecycle.

### Audio and renderer integration

- [ ] Package one loop JSON and the existing four-second cue assets.
- [ ] Add `BreathingSetupAudio.java` with one process-owned looping AudioTrack,
  validated PCM inventory and playback-frame/monotonic-clock observations.
- [ ] Feed original Polar samples and existing controller assessments into the
  native setup owner; apply its output through the existing driver slot.
- [ ] Select this behavior explicitly for the Atmungsraum build. Test startup,
  source/clock loss, pause/resume and input changes without restarting audio.

### Gateway and phone

- [ ] Add closed enrolled setup/calibration/fallback/main-start commands and
  native status. Test unauthorized, replayed, stale and active-run mutations.
- [ ] Add a pure phone setup-state adapter with behavioral tests.
- [ ] Replace preview timers with native phase/input/progress/failure readback.
  Preserve participant identity, H10/ECG, recording, Drive and reconnect behavior.
- [ ] Render-check phone/desktop, 320 px reflow, enlarged text, keyboard,
  reconnect and older APK compatibility.

### Build and evidence

- [ ] Regenerate/review the cumulative patch against the pinned native upstream;
  update locked source closure, build assets, version, docs and APK release record.
- [ ] Run focused Rust/Android/Node suites, context and diff checks. Compile a
  fresh APK; verify its package, signer, assets, libraries and source binding.
- [ ] Coordinate exact headset/build resources and qualify real audio/visual
  phase, worn-Polar calibration/retry/fallback and a full 900-second run.
- [ ] Review all changes, preserve concurrent remote work, and follow the
  authorized Git/Pages/APK workflow. Record exact evidence and remaining blockers.

Optional named skills `for-ai`, `ponytail` and `uncodixfy-pretext` are absent.
Use the documented bounded fallback: existing For-AI lifecycle, Superpowers,
locked Pretext helpers and rendered verification. Install no remote skill code.
