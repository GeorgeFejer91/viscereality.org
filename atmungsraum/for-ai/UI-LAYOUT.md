# Operator layout contract — 2026-11-16

The user requests clearer layout and sequential pages, without new experiment
features. Use the actual root website code as the visual reference. This is a
PWA-only change; native behavior, audio and recordings remain unchanged.

## Visual source and design

Reference `index.html` and `assets/css/site.css` in Git (the root files are
outside this sparse checkout). Retain Arial, black/white, blue `#00639c`, gold
`#a26f19`, the existing authentic VR icon and readable telemetry colors. Reuse
the site's restrained framed groups, thin left rules, 5–8px corners, and clear
20–30px spacing. Scope app styles; do not import the promotional root CSS.
Controls retain 48px minimum touch targets. Reserve the top toolbar for Bluetooth,
WLAN, native control and Polar checks, the connected recenter control, and the grey
Einstellungen and Aufnahmen icons. Put the row before every page, including
historical recordings, and above device, installation and Drive dialogs.
Settings and recordings open their existing pages; return restores the toolbar
button. Disable recordings while native playback or receipt recovery needs the
operator's attention. Installation stays on the connection and setup page.
Keep the checks and grey action buttons in one row with equal 48 px minimum
touch targets. Through 389 px the grey actions button opens Einstellungen and
Aufnahmen below the row. Wider screens show both buttons directly. Keep the
four checks and the connected recenter control in the row. Reserve one 48 px
toolbar track plus the safe top inset. APK status remains on its update page.

Use real HTML/CSS prototypes and actual rendered fixture screens as the visual
design artifact: these are forms, navigation and existing code-native SVG/CSS
diagrams, not raster illustrations. No generated imagery or new brand assets
are appropriate. The existing real Pretext implementation remains mandatory.

The page heading uses its natural content height. Keep the actual-radius
playout, physical percentage semantics, and exact touching inset geometry.

## Page sequence

Connection and setup → APK update when required → participant ID → input choice → personal calibration
when selected → tracking check → battery review → scenario tiles and Start → running/finalizing
→ recordings. Settings and recovery remain available as before. Keep the saved
signals route and the earlier browser-recordings route usable. Pairing codes
belong only to an actual incoming pairing request.

After saving the session and answering any required questionnaire, **Nächste
Sitzung** clears the previous participant form and the tracking and battery
confirmations. It opens participant entry for the next run. The action requires
a confirmed idle Quest and settled requests. Returning to stopped preparation
through browser history or after reconnect opens the method chooser. Live
calibration and recovery still take priority over local navigation.

The connection page groups Bluetooth pairing, live ECG, installation and Drive
setup beside the shared status toolbar. Require an
authenticated WLAN connection and eligible control state for **Weiter**. Polar
readiness remains a separate native fact; the controller alternative must
stay accessible. Checking or unnecessary checks are neutral, confirmed checks
are green, and failed checks are red with a named diagnostic popup. Use a
visible symbol as well as color. The Bluetooth-shaped toolbar control represents Quest connection once an
actual authenticated link is established, including a saved WLAN rejoin. It
then turns green and is labelled Quest-Verbindung. This is not proof of a live
Bluetooth transport; discovery/pairing uses Bluetooth and commands use WLAN.
Keep required recovery actions visible on the page.

APK code comparison and Meta update controls live on their own preparation
page when the public code floor or native maintenance status requires action.
An already current/allowed APK skips that page; Settings can open it explicitly.
An offline allowed native failure remains a warning, not an added admission
block. Keep the connection page available for reconnection. Once opened, retain
the update page and diagnostic focus through terminal polls until the operator
continues. Only the validated native status can release native admission.

The input chooser and tracking review retain a visible confirmed-method label.
Use the same gold as the actual VR-radius circle for every active method, with
aria-pressed reflecting only validated native input. Inactive choices stay
neutral. The active Controller button remains in the list. Keep a pending
request separate from the confirmed method; never optimistically highlight it.
Readiness changes must not toggle the entire status block above the circles.
The tracking action reads Atmungsmessmethode bestätigen for every method.
Use Atmungsmessmethode in participant-facing labels instead of the generic
Eingabe; explain that it chooses Polar- or Controller-driven breathing movement.

The live ECG view subtracts a rolling median baseline only when drawing and
scales its deviations about the existing middle line. Label this centering in
the scale. Preserve raw samples, timestamps, gaps, freshness and stored data.

Before scenario selection/Start, show one focused battery review: Quest,
Polar and available controllers. Keep unknown values explicit. A missing
controller or unreported battery is informational. Continue confirms this
review for the current participant/Quest and preparation; percentage changes
must not cause navigation. Active sessions retain finish/recovery controls.

Enter participant details on a dedicated, aligned form. Preserve field values
and the existing identity generator. During personal calibration the Quest
still owns the two-cycle cue, measurement and completion. Leave the circles
visible at review and ask the operator to confirm the appropriate movement.
Retry invokes the existing native calibration command. Expose the direct
Controller shortcut at review after two Quest-reported attempts in this
participant setup; the existing input chooser still offers Controller and
supported pretrained Polar modes without this shortcut gate.

Show three choices on the scenario page. Group **Anleitung**
(`probe-anleitung`) and **Probe-Runde** (`probe-5`, five minutes) under
**Vorbereitung**. Place **Haupt-Runde** (`haupt-15`, 900 seconds) in a
separate **Hauptstudie** section. Show illustrated square tiles where the
labels and available height fit. Reflow by available width and text size,
keeping preparation and the main study separate. The operator selects a scenario. Quest
receives the established condition value with the exact scenario ID. Arm,
recording metadata, filenames and per-participant counters retain their
existing `probe` and `main` rules. The running screen shows the authenticated
Quest elapsed clock. Render determinate progress
only if the authenticated Quest snapshot reports a positive duration or
confirms the shared-loop main-only build's fixed 900-second protocol and start
frame. Other snapshots without a duration show elapsed time alone.
The running view draws a simple particle sphere from the authenticated
Quest renderer's applied radius, using the same source-time buffer and
500 ms freshness window as tracking. Pause holds the visible radius.
Missing or stale observations clear the running preview. Reduced motion
leaves the numeric radius visible. The preview does not issue commands
or drive the session clock.

The user confirmed that the 10- and 15-minute practice rounds, the 20-minute
box-breathing scenario, and the condition selector are intentionally absent
from the operator UI. Keep their existing native contracts. The PWA offers
the intersection of its three choices and the authenticated Quest catalog,
using condition A. Start requires an offered choice and a ready A track.
When either is unavailable, show the diagnostic. A main-only Quest presents
Haupt-Runde alone.

## Smartphone GUI rules

Apply the [playground rules](https://github.com/GeorgeFejer91/atmungsraum-ui/blob/58906318e1f407c9c1e2874180d926c3fdee540e/for-ai/PROJECT.md#smartphone-gui-design-rules)
to the real operator and recovery flow.

1. Fit every input step, form, dialog and overlay into one usable visual
   viewport with all its labels, statuses, actions and focus indicators. Use
   the space available with safe areas, browser chrome and the keyboard.
2. The document and every app-authored region must have zero scroll range in
   either axis during navigation, editing, dialogs and focus changes. Keep
   browser zoom and text enlargement available. The Haupt-Runde
   questionnaire is the sole exception. Its form can scroll vertically to
   expose all five items while its title and save action remain visible.
3. Allocate the shell first using the outside-in accordion stretch layout.
   Reserve heading and action tracks. Use balanced insets and shrinkable
   Grid or Flex tracks, preserve reading and focus order, and keep label-field
   pairs together. Align peer controls and grow their shared label track when
   a label wraps. Distribute space between ordered groups and peer cells as
   each dimension grows. Derive readable type and spacing from both dimensions
   and keep visible focus and 48 px minimum touch targets.
4. Use the locked Pretext 0.0.9 measurement in the actual loaded typography,
   inner width, inner height, spacing and locale. Confirm the painted DOM.
   Use task-specific views for optional controls and bounded selection for
   lists. Each view must expose its full readable text and actions.
5. Show neutral black-and-white SVG checks while checking, green checks on
   confirmation and red checks on failure. Pair color with a symbol and an
   accessible label. Each red check opens its own diagnostic content in one
   popup. Keep those details in the popup, retain the visible single toolbar
   row above it, and use named task actions instead of generic page arrows.

## Behavior invariants

- Local navigation, Back/Next and browser history send zero native commands.
  They preserve drafts, participant identity, acceptance tokens and journals
  unless the operator explicitly starts the next participant.
- Existing domain helpers alone own readiness. Do not gate access to the
  Controller fallback on ECG/ACC readiness or add a stricter wizard policy.
- Original controls have one persistent DOM instance and their original IDs.
  Move shared nodes only on page transitions, never on each telemetry update.
- Selecting a mode/calibrating remains an explicit existing command. A local
  change-input link must never reuse the calibrate command handler.
- Saved reconnect/foreground resume automatically requests a free native lease.
  Only fresh native active-role readback confirms control; connection, membership
  and Polar/input readiness remain separate. Other live owners block acquisition,
  unknown requests are never replayed, and manual Release pauses auto-claims.
- The exact two-cycle native progress owns calibration completion. No local
  timers classify success. Confirmed input and requested input stay distinct.
- Tracking acceptance still binds owner/epoch/generation/attempt/input.
- Start retains Arm→confirmed Start in the same foreground click. Running,
  pause/resume, finish/finalization and recording sync retain current guards.
- Actual native recording/recovery takes precedence over obsolete local steps;
  settings may expose diagnostics without hiding required recovery controls.
- After verified storage of a completed Haupt-Runde staged live on this phone,
  show all five required questions on a single scrollable form before
  returning to recordings. Use five-point Likert choices and 0–100 sliders.
  Preserve responses during scrolling and keep native running/recovery state
  higher priority. See [the questionnaire data contract](DATA-LOGGING.md#haupt-runde-questionnaire).
- Native completion opens recordings. Enable “Nächste Sitzung” after confirmed
  idle, reconciliation of the current recording generation and storage of any
  required questionnaire, once active calibration has settled. The action clears
  the previous participant and local tracking and battery confirmations and opens
  the participant form.
- A Quest that does not require a participant ID skips that page in both
  directions. Native completion replaces the current history entry with
  Recordings so returning from the earlier browser exports opens Recordings.
- The earlier browser-recordings link works without a Quest. Return to the
  same history entry when coming from Aufnahmen. Keep the Drive dialog usable
  on both setup and recordings, and show live transfer status during a run.
  Short Recordings viewports use one native section selector for the Quest
  recording, files, Quest sync and storage. The controls keep their IDs and
  saved selection. After questionnaire save, open the files section directly.
- A disconnected controller can reconnect with an unresolved receipt. An
  enrolled waiting phone can claim free control to recover the original journal.
- Navigation never starts scenarios, changes duration (main 900 s), changes files,
  Drive authorization, native source state or kiosk policy.
- Telemetry preserves page, focus, caret, unfinished edits and scroll. View
  transitions focus the visible heading; polling does not steal focus.
- Only change a screen's `hidden` flag when its visibility changes. Even an
  unchanged assignment triggers the measurement observer and can reset the
  bounded layout, transiently clamping document scroll before readable reflow.
- Measure nested aside/details bounds as well as the outer page. A bounded
  grid must not shrink an APK/recovery panel over its neighbors. Only lock a
  panel's measured minimum height while its ancestor page is actually visible.
- On participant entry, let the update warning take its current content
  height, including while the phone keyboard is open. Shorter warnings give
  the form back its space without changing the focused input or its caret.
- Browser history stores/restores each local utility return target. Back must
  not retain a later settings page's return target or return to itself.
- Paused, finalizing and saving are native operational states, not routine
  connection messages: keep them visible in the current heading/status.
- Preview-only100ms status reads belong to visible calibration/tracking;
  ordinary ECG/receipt/export cadence and singleflight remain unchanged.
  Input diagnostics show the raw breathing value on older snapshots that
  supply it without a renderer. This view uses the ordinary status cadence.

## Required verification

Pure navigation tests plus rendered real-app fixtures must prove zero-command
Back/Next/settings, explicit mode/calibration, recovery priority, recording
controls, persistent participant drafts and unchanged acceptance invalidation.
Run the [smartphone GUI release gate](VERIFICATION.md#smartphone-gui-release-gate)
for 320 and 390 px portrait, landscape, desktop, 200% text, text spacing, and
long localized statuses. Check actual Pretext and DOM bounds, 48 px controls,
zero document and nested overflow, keyboard and stable focus. Run the complete
phone suite with its real native/Java interoperability prerequisites. Review
the whole change independently. Publish a coherent cache graph, exact Pages
deployment and affected live URLs. Verify the installed phone through the
[autonomous device procedure](VERIFICATION.md#autonomous-device-validation)
and record pending wearer checks in the [active checklist](PROJECT.md).

## Attended readiness presentation — 2026-11-17

Do not show healthy Meta completion on ordinary pages. The explicit update
page retains its terminal status and geometry. During method review, retain
one stable instruction and the confirmation button's appearance through transient
sample-level readiness changes; its disabled property still follows native
readiness. Once the operator accepts a method, keep that local judgement only
while actor/audio/attempt/method/sensor binding and the testing loop remain
valid. Keep Session visible through a transient unknown phase, with Start
explicitly disabled until the existing native readiness gate is true again.
Retire that judgement when native leaves IDLE, even if an older snapshot keeps
setup.stage at testing after recording. No local acceptance authorizes Arm,
recording or native readiness.

A stale-revision retry is permitted only for the original valid bound explicit
rejected/accepted:false result, never an unknown/pending/accepted result. Read
fresh status and recheck unchanged intent/action gates before each new ID; cap
at3 total requests. Preserve a named error and actual confirmed method on failure.

## Concise APK update states — 2026-11-17

Connect and Update share one short update status plus the relevant action.
Keep installed/required/channel codes and diagnostics in Details; avoid repeating
the same requirement in connection, compatibility and update notices. Keep a
separate genuine failure visible when it adds information. An enabled update
button must not be accompanied by a generic instruction to wait for preparation.
When Meta does not yet offer the required build, show that fact and “Erneut
prüfen”. Preserve explicit recording/save actions and original unknown receipts.
