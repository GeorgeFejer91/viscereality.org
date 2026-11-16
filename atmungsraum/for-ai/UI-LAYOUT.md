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
Install/recenter tools occupy the reserved document header space and scroll
away with it; a floating recenter target must not cover form choices or mode
labels on long pages. Check corner-control intersections at actual scroll tiles.

Use real HTML/CSS prototypes and actual rendered fixture screens as the visual
design artifact: these are forms, navigation and existing code-native SVG/CSS
diagrams, not raster illustrations. No generated imagery or new brand assets
are appropriate. The existing real Pretext implementation remains mandatory.

The header has its natural content height; remove the old fixed 48px track
that clashes with its brand/title/separator. Prefer readable document reflow
over clipping, smaller fonts or nested scrolling. Keep the actual-radius
playout, physical percentage semantics, and exact touching inset geometry.

## Page sequence

One main job per page: connect/claim → participant → signals → input choice →
personal calibration when selected → tracking check → session selection →
running/finalizing → recordings. Settings/version/kiosk are a separate utility
view with a clear return. Preserve the legacy browser-recordings route.
Native recovery and pending operations must remain visible and actionable.
Pairing codes belong only to actual incoming pairing, not routine startup.

Use short headings, a small preparation step indicator, left-aligned status
groups, and one clear primary action. Keep advanced identity/diagnostic values
accessible in settings. A compact compatibility warning remains visible in
context, without creating a new experiment gate. The full tracking display
belongs on calibration/tracking pages, not every setup page.

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
- Navigation never changes scenarios/defaults, duration (main900s), files,
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
