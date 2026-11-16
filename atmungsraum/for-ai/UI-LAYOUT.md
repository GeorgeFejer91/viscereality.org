# Operator layout contract — 2026-11-16

The user requests clearer layout and sequential pages, without new experiment
features. Use the actual root website code as the visual reference. This is a
PWA-only change; APK11, native behavior, audio and recordings remain unchanged.

## Visual source and design

Reference `index.html` and `assets/css/site.css` in Git (the root files are
outside this sparse checkout). Retain Arial, black/white, blue `#00639c`, gold
`#a26f19`, the existing authentic VR icon and readable telemetry colors. Reuse
the site's restrained framed groups, thin left rules, 5–8px corners, and clear
20–30px spacing. Scope app styles; do not import the promotional root CSS.
Controls retain 48px touch targets. Do not copy tiny website navigation.
Install belongs to the connection and setup page. Reserve document space for
recenter and keep it clear of form choices and mode labels. Check corner-control intersections at actual scroll tiles.

Use real HTML/CSS prototypes and actual rendered fixture screens as the visual
design artifact: these are forms, navigation and existing code-native SVG/CSS
diagrams, not raster illustrations. No generated imagery or new brand assets
are appropriate. The existing real Pretext implementation remains mandatory.

The header has its natural content height; remove the old fixed 48px track
that clashes with its brand/title/separator. Prefer readable document reflow
over clipping, smaller fonts or nested scrolling. Keep the actual-radius
playout, physical percentage semantics, and exact touching inset geometry.

## Page sequence

Connection and setup → participant ID → input choice → personal calibration
when selected → tracking check → scenario tiles and Start → running/finalizing
→ recordings. Settings and recovery remain available as before. Keep the saved
signals route and the earlier browser-recordings route usable. Pairing codes
belong only to an actual incoming pairing request.

The first page groups Bluetooth, authenticated WLAN, APK and Polar checks, live
ECG, installation, Drive setup and the existing device utilities. Require an
authenticated WLAN connection and eligible control state for **Weiter**. Polar
readiness remains a separate native fact; the controller alternative must
stay accessible. Checking or unnecessary checks are neutral, confirmed checks
are green, and failed checks are red with a named diagnostic popup. Use a
visible symbol as well as color. A saved WLAN rejoin does not prove Bluetooth
was checked. Keep required recovery actions visible on the page.

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
separate **Hauptstudie** section. The operator selects a scenario. Quest
receives the established condition value with the exact scenario ID. Arm,
recording metadata, filenames and per-participant counters retain their
existing `probe` and `main` rules. Playback uses the authenticated Quest elapsed
clock. Show a determinate bar when that status also reports a valid total
duration. The elapsed time remains available on older builds.
The running screen shows the native elapsed clock. Render percentage progress
only if the authenticated Quest snapshot reports a positive duration; a missing
duration still shows elapsed time, without estimating the total.

Older Quest catalogs may advertise the 10- and 15-minute practice rounds or
the 20-minute box-breathing scenario. Keep those exact IDs and eligible VR
conditions in the secondary options. A condition-only snapshot with no catalog
keeps the existing condition Arm. An explicitly unavailable or invalid catalog
blocks Start and shows the operator a diagnostic. A main-only Quest presents
Haupt-Runde alone. Haupt-Runde requires condition A even on a mixed catalog.

## Smartphone GUI rules

Adapt the [playground design and release rules](https://github.com/GeorgeFejer91/atmungsraum-ui/blob/main/for-ai/PROJECT.md#smartphone-gui-design-rules)
to the real operator and recovery flow. With ordinary phone text, every input
step, popup and its actions must fit together in one usable viewport. Allocate
the available shell first, with reserved heading and action rows, ordered
groups, balanced safe-area insets and 48px targets. Use shrinkable tracks and
shared label/control rows. Use surplus space on both axes between the groups.
Provide optional diagnostics and settings in their own views rather than
stacking them on the primary task. No generic arrow controls, clipped content
or scrollbar hiding.

Keep real local Pretext 0.0.9 measurement of bounded labels using their actual
font and inner geometry. Verify painted DOM and focus as well as measurement.
Keep text enlargement and browser zoom enabled. An unbounded recording list,
extreme text/keyboard constraints or lengthy native recovery information may
require readable document reflow while preserving full text and actions; treat
unexpected scrolling in ordinary input states as a layout failure. An
exception must be evidenced, with its state and viewport recorded rather than
counted as a strict fit.

## Behavior invariants

- Local navigation, Back/Next and browser history send zero native commands;
  do not clear drafts, participant identity, acceptance tokens or journals.
- Existing domain helpers alone own readiness. Do not gate access to the
  Controller fallback on ECG/ACC readiness or add a stricter wizard policy.
- Original controls have one persistent DOM instance and their original IDs.
  Move shared nodes only on page transitions, never on each telemetry update.
- Selecting a mode/calibrating remains an explicit existing command. A local
  change-input link must never reuse the calibrate command handler.
- Saved reconnect does not acquire control; claim/release authority is native.
- The exact two-cycle native progress owns calibration completion. No local
  timers classify success. Confirmed input and requested input stay distinct.
- Tracking acceptance still binds owner/epoch/generation/attempt/input.
- Start retains Arm→confirmed Start in the same foreground click. Running,
  pause/resume, finish/finalization and recording sync retain current guards.
- Actual native recording/recovery takes precedence over obsolete local steps;
  settings may expose diagnostics without hiding required recovery controls.
- On native completion, show recordings. “Nächste Sitzung” is local navigation:
  retain the participant and return through tracking if the native generation
  invalidated the previous acceptance; never fabricate a new accepted token.
- The earlier browser-recordings link works without a Quest. Return to the
  same history entry when coming from Aufnahmen. Keep the Drive dialog usable
  on both setup and recordings, and show live transfer status during a run.
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
- Browser history stores/restores each local utility return target. Back must
  not retain a later settings page's return target or return to itself.
- Paused, finalizing and saving are native operational states, not routine
  connection messages: keep them visible in the current heading/status.
- Preview-only100ms status reads belong to visible calibration/tracking;
  ordinary ECG/receipt/export cadence and singleflight remain unchanged.

## Required verification

Pure navigation tests plus rendered real-app fixtures must prove zero-command
Back/Next/settings, explicit mode/calibration, recovery priority, recording
controls, persistent participant drafts and unchanged acceptance invalidation.
Inspect 390px,320px,200% text, long localized status, landscape and desktop.
Check actual Pretext/DOM bounds,48px controls, no overlap/horizontal or nested
overflow, keyboard/focus and multi-poll scroll stability. Run the complete
phone suite with its real native/Java interoperability prerequisites. Review
the whole change independently. Publish a coherent cache graph, exact Pages
deployment and affected live URLs. Physical installed-phone qualification is
pending phone availability; new-mode wearer testing stays deferred.
