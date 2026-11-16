# Shared breathing scheduler and setup calibration

Approved user design, 2026-11-14. PROJECT.md remains the lifecycle owner.

## Smooth actual display and optional pretrained Polar modes — 2026-11-16

The yellow actual-radius circle and percentage use one animated physical
progress value. During the visible preparation preview, read-only status may
refresh at 100ms with one request in flight. Ordinary receipt reconciliation,
ECG, recording sync, connection attempts and commands retain their existing
cadence. Stop fast reads outside the preview or while backgrounded. The native
500ms total renderer freshness limit stays. A bounded history of distinct GPU
observations paints adjacent source-time samples with 280ms playout delay;
incoming samples append rather than restarting a composite blend. Each endpoint
keeps its original absolute expiry. Where no fresh bracket exists, clamp to a
supported sample or freeze/snap honestly. Never extrapolate the desired phase
into an actual radius measurement. This is presentation delay, not a clock,
recording, classifier or command-authority change.

At equal physical progress, the yellow outer stroke touches the blue inner
stroke. Keep the same center and physical scale; subtract the blue's two border
widths from the yellow's unscaled border-box diameter. Geometry does not change
the measured progress or percentage and must hold at minimum/maximum sizes.

Keep the existing participant-calibrated Polar mode and add explicit choices
for the existing pretrained fixed-axis and adaptively adjusting tilt-state
classifiers. The latter adjusts the classifier's motion frame; it is distinct
from the generic principal-axis projection. Preserve the shipped classifier
weights and the accepted participant learner. The new presets do not claim a
participant calibration or require the two-cycle fit. Display the confirmed
input separately from a pending selection, and show participant calibration
controls only in its applicable path. Controller State remains the fixed-axis
fallback. Older APKs must not imply support for the new choices.

Native setup owns selection, capability, source/clock/quality readiness and
original-ID receipts. Preparation mode switches keep a healthy shared box loop;
they cannot change an active or queued main experiment. Classified inhale,
exhale and hold feed the existing retained-state radius integration. Record
truthful selected-preset provenance, not a fabricated participant model. Startup
remains Automatic, parked and silent, and the main remains exactly 900 seconds.

The selected native setup advertises `polar_modes: ["calibrated", "fixed",
"adaptive"]`. Existing `polar-tilt` remains the participant fit. New closed
operations `polar-fixed` and `polar-adaptive` confirm inputs `polar-tilt-fixed`
and `polar-tilt-adaptive`. They reuse the shipped fitted tilt-state classifier,
respectively with its fixed sensor frame or adaptive compensation. They do not
run a participant fit or substitute the principal-axis DynamicAxis algorithm.
The phone distinguishes the requested mode from confirmed input and hides
personal calibration controls for those presets. Missing/malformed capability
cannot authorize either operation. Existing receipt/CAS, active-controller,
preparation-only and source/clock/readiness guards remain.

Preset selection preserves healthy PCM position/generation and retires old
classifier/driver confirmations. Source/audio/projection changes still fence
state; ordinary masked samples retain the last confirmed breath state. Preset
tracking provenance names its exact classifier and packaged model, never an
invented participant axis/range or continuous volume measurement. Main remains
900 seconds with current required streams. Automatic stays parked on load.

The preview requests status-only observations every 100ms while visible and
connected; requests coalesce. The ordinary one-second receipt/ECG/export/Drive
loop is unchanged. Actual playout uses unique renderer observations with exact relative
`observed_ns` spacing. The first accepted arrival anchors the source timeline
using its full charged age; the bounded 32-sample history adds no source data or
future prediction. The fixed 280ms playhead interpolates only a supported pair
whose original absolute 500ms expiries both remain valid. Expiry, replay, source/
settings/owner/audio replacement and visibility barriers remain. Unsupported
intervals clamp or freeze; reduced motion selects discrete supported samples. Radius and
percentage share that same actual progress. Gold's border box is four CSS pixels
narrower before the shared physical scale, placing its outer edge at blue's
inner edge at equal progress, including responsive minimum/maximum sizes.

## Operator controls and continuous connection — 2026-11-15

The participant accepts the current tilt calibration as good enough. Keep its
learner, frame-clock authority and retained breath-state integration unchanged.

Quest owns H10 discovery, connection and both required streams. On normal app
load it must begin discovery and keep making bounded, delayed attempts if the
strap is absent or a connection fails. Reuse the native retry owner; phone
polling must not create connection effects. A manual connection must not silently
disable that recovery path after failure. Preserve explicit Stop and ambiguous
multi-strap selection barriers.

The phone names the confirmed active breath input and any pending switch.
Controller State has no calibration: hide Polar progress, quality and failure
controls after confirmed Controller selection, with an explicit route back to
Polar setup. The same box loop continues throughout. Successful clicks alone
cannot assert a confirmed input change.

Overlay desired expansion in blue with fresh headset-applied radius progress in
gold, with text labels and distinct stroke styles. Sensor/model coordinates are
not applied radius. Use the actual renderer hook and its live generation/settings
and age fences; unavailable/stale data must not become an invented actual trace.
Smooth animation remains presentation only and preserves raw observations.

Connection uncertainty needs a prominent explanation and a clearly named next
action. Read-only reconciliation, local acknowledgement and an actual headset
or app reset must be distinguished. Never silently replay an uncertain mutation.
If a reset is genuinely required, state what is being reset and why.

Expose effective soft-kiosk state and prerequisites in PWA headset settings.
Distinguish requested, armed, running, blocked and unknown observations. Offer
only supported authenticated recovery actions under the existing native owner;
request admission does not prove foreground protection. Preserve platform prompts,
pairing approval and permission boundaries. No silent system-shell suppression.

### Installed build and update indication

Authenticated private status exposes optional `apk`: null or
`{v:1,package_name,version_code,version_name,build_id}`. Values come from the
installed own package and its generated build-identity asset, cached once.
`build_id` must match authenticated `build.packaging_fingerprint`; it is not the
APK-byte hash or certificate fingerprint. Missing, malformed, cross-build or
retired-actor metadata is unknown.

The same-origin public `quest-release.json` declares minimum/latest integer
version codes per package. The phone shows installed and required versions,
with required/recommended/compatible/unknown/unavailable states. Policy fetches
are read-only and uncached; failed checks must not retain a false up-to-date
claim. No install, Meta API access or experiment admission effect is introduced.
Update the policy with release evidence, and verify it at the deployed URL.

## Smooth phone presentation — 2026-11-15

Live telemetry remeasures text without briefly returning the current page to
a shorter bounded layout. It preserves the same form, focus, caret, draft values
and scroll position. Real screen navigation may choose a new layout; viewport
and font changes preserve an active scroll or editor. Retain actual Pretext
measurement and readable document reflow rather than rebuilding controls on polls.

The phone animates the breathing circle from a fresh accepted native audio
position in the existing 16-second schedule. Its finite 1,500ms display budget
charges native age, the full status transport and receipt-to-paint delay.
Duplicate observations cannot renew it. Reduced motion uses discrete native
samples and still expires; backgrounding suspends the display and shows a
waiting state until a fresh observation. This never admits a command, trains
a model, switches input or advances the main experiment.

Automatic's percentage uses the same fresh native phase fraction as its circle.
For Polar or Controller input, the displayed percentage interpolates only between
already observed measurements. It cannot predict a future sample. Both displays
share expiry, pause, replay, reduced-motion and visibility barriers. The numeric
field reserves measured space for `100 %`; high-rate numeric changes do not
trigger whole-page text remeasurement or reset document scroll.

Calibration's accessible progress value and text remain the actual native
completed-frame count. Only the visible fill interpolates between received
counts, and it cannot exceed the last confirmed count or finish a calibration.

The ECG preview retains bounded validated raw samples for a two-second rolling
window with about one second of playout reserve. It changes position and scale,
not microvolt values or sensor timestamps. Gaps remain open, starvation stops
at the last received sample, and only newer timestamps extend age-adjusted
freshness. A status outage clears the image while preserving the replay fence.
Confirmed source/owner/generation changes reset the display. Recording, Polar
acquisition and calibration retain their native paths unchanged.

## Accepted behavior

Latest user ruling, 2026-11-15: calibration assumes the participant follows the
guidance. Scheduled phases supply training labels for the person's tilt axis,
direction and range. Reject obvious unusable data or technical failure; do not
require an ideal waveform, flat holds or cross-cycle compliance/recurrence.
After learning, switch to Polar tilt and let the participant assess tracking
while the same loop continues. The experimenter may repeat calibration or choose
Controller State. The fixed-tilt system's filtering and mapping lessons can inform
the learner without becoming a pre-calibration breathing classifier.

Calibrated tilt classifies inhale, exhale or hold for the existing frame-time
radius integrator; its normalized measured coordinate is not the rendered radius.
Preserve smooth state-driven expansion/contraction. The user's explicit correction
is to retain the last valid breath state until valid data warrants a new state:
inhale continues expanding, exhale continues contracting, and hold keeps its radius.
Within the same source/model/audio lineage, unavailable or invalid samples cannot
change that state. Continue integrating it once per rendered frame with the
existing rates and limits, while reporting input readiness honestly. No packet
catch-up or radius reset is permitted for a transient sample lapse.
Actual source, projection, audio, model, input-selection and configuration barriers
still retire the appropriate history. Code 7 exposed an undifferentiated Waiting
reset during the participant preview. Reviewed code-8 source corrects this;
fresh participant tracking acceptance remains open.

The calibrated path uses the common normalized-tilt phase classifier, while the
older fixed-tilt path uses its 25 Hz feature classifier. Their phase timing is
not identical. A normalized box-wave probe at 25/50/100/200 Hz produces the same
states for individual and batched delivery, without chatter on that fixture;
Hold transition latency differs by up to 160ms because the common derivative
filter is per observation. This is a qualification limit, not wearer tracking
acceptance. The radius continuity repair does not change that classifier.

The app must select controller-only input. Hand tracking and simultaneous
hands/controllers must remain unselected for this product, including manifest
capabilities and runtime activation, so automatic switching cannot interrupt the
Controller State breathing driver. Other products keep their existing selection.

Automatic is the Atmungsraum startup input, parked at minimum expansion with no
audio playback. Loading the app, opening the phone setup and lifecycle resume
cannot start it. An accepted explicit calibration or Controller selection starts
the loop once. One native audio-frame schedule owns
cue playback, intended breathing phase, automatic visual input, calibration
labels, and experiment boundaries. Inhale, hold-in, exhale and hold-out each last
four seconds. Preparation loops indefinitely. The phone projects native state;
browser timers never advance the schedule.

Polar stays connected and automatically establishes/retries both ECG and motion
streams while the selected native owner requires them. Explicit Stop and source
retirement fence recovery; a connected Bluetooth link alone is not readiness.

Calibration selects Automatic and collects two full cycles (32 seconds). The
first parked start begins with the first actual playing observation; retries use
the next inhale boundary without restarting the PCM track. Progress uses actual
audio frames. Bounded invalid, motion-contaminated or mistimed intervals are
masked throughout collection, with usable phase coverage and range evaluated
only at the end. A brief bad interval alone must not end collection immediately.
Source/clock retirement and an actual pause during active collection remain
fatal. Missing coverage, sustained motion or insufficient range fails the fit.
Elapsed time alone never proves success. A successful model is activated as
Polar tilt atomically while the loop continues. Retry repeats this process.
Controller State with fixed orientation is available from parked startup and
cancels an outstanding calibration,
keeps the same audio clock and uses the existing controller assessment owner.
Fallback selection stays available after Polar tracking becomes unreliable.
Starting the participant run still requires the study's fresh H10 streams;
changing the breathing driver does not remove ECG/ACC capture requirements.

Start experiment retains the chosen input/model and begins at the next full
cycle boundary. The main experiment lasts exactly 900 seconds (user confirmed).
Its recording clock starts at that boundary. Preparation is not experiment data.
The old embedded 15-second calibration and introductory restart are retired.
Other Rusty Quest products keep their existing defaults and feature activation.

## Explicit unhealthy-audio recovery — 2026-11-15

An explicit Calibrate or Controller action may reserve recovery only in owned
preparation with no unretired main effects. Native validates the original ID/CAS
before Java claims one transport replacement. The failed attempt/model/training
and callback state are retired. A fresh owned generation, actual forward audio
frames and fresh post-claim input must precede the new setup. Calibration then
uses the next full boundary and a complete new two-cycle window. Healthy retries
keep the same loop. Recovery never repairs or restarts a main recording.

Pending recovery is not completed setup. The authenticated private-LAN/current-
controller gateway retains native receipts. The existing three-field read-only
lookup `{v:1,id,op:"receipt"}` needs no old CAS and cannot replay the operation.
Cross-generation settlement requires the native `recovery_generation` proof for
the original ID and current actor. A lost reply preserves uncertainty and the
original ID; same-actor audio-generation changes cannot acknowledge it away.
Actual actor retirement remains a separate boundary.

Native `clock_valid` and `audio_recovery_available` are eligibility/health hints;
native admission remains authoritative. Unhealthy/recovering display does not
extrapolate a circle or percentage. Bounded poll/head/dispatch diagnostics are
inert, with platform measurements tagged to their sampled generation. Intent
playing and a retained sensor anchor do not prove advancing audio. The initial
code-8 physical stall trigger remains unproven; code 9 requires fresh wearer QA.

## Buffered Polar timing — 2026-11-15

Idle Automatic does not validate tilt evidence that no attempt or model
consumes. Raw acquisition and source ownership remain active; real source
changes still invalidate old command revisions. Repeated idle diagnostics must
not churn control revisions or overwrite a terminal calibration failure.

Keep original raw sample and callback timestamps. The selected repair separately
warms a source-bound fixed sensor/receipt-anchor estimate before calibration;
its provenance declares estimated acquisition timing and a 250 ms receipt bound.
The offset does not move per packet or retry. See the
[recording clock contract](DATA-LOGGING.md#shared-scheduler-recording-boundary).

Map the derived estimate through a fresh measured Java-nano/elapsed pair into
actual healthy, playing, same-generation audio observations. Retain at most
32 observations covering a 250 ms history
window plus a bracket; never extrapolate a missing frame. At most 64 future
training labels may wait for the next actual clock bracket. Pause, retirement,
controller cancellation and source changes clear evidence at their relevant
barriers. Original acquisition timestamps remain unchanged.

Sensor timestamp gaps and driver readiness stay at 200 ms; sensor/host drift
tolerance stays at 150 ms. A separate 250 ms accepted-ingress deadline admits
normal 200 ms delivery with modest jitter while failing missing packets.
Temporary unready output cannot fabricate missing sensor history. Fit only
after the full two-cycle audio window and the bounded 250 ms final delivery
allowance. Missing final or interior segments may be masked if the window still
has sufficient usable evidence. Elapsed time alone cannot fit.

The selected quality policy uses at most 800 uniformly timed 25 Hz observations.
At least 600 must be usable overall, 280 in each cycle and 60 in each instructed
four-second phase. Each inhale/exhale needs six usable observations in both its
first and last second. Finite raw acceleration and 0.85–1.15 g magnitude remain
required. A causal 100 ms gravity estimate feeds the 0.60 normalized-gravity
chord/second motion guard, with 200 ms settling after bad intervals. Existing
engineering range limits still apply. Code 7 removes code 6's ideal per-cycle
fit test, following its attended rejection and the latest user ruling. The
retained measured span 0.02–0.50 is an engineering gain/noise limit and can reject
smaller real motion; it is not a validated physiological cutoff. These thresholds
awaiting wearer qualification. Optional quality counts are bounded 0–800 and
do not change control revisions or prove success.

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
