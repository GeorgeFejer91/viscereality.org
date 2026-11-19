# Optional manual Quest update help — 2026-11-19

Settings and the existing Update page open `#update-help`. The six numbered
steps are presentation only: save a running session, use the confirmed kiosk
escape (three separate Meta-menu openings in five seconds with the stimulus
returning between them), find Atmungsraum's Developer Store page, check the
Developer release channel when necessary, wait for completion/reopen, reconnect
the PWA and verify the expected APK. Exact Store navigation is intentionally
not specified until the attended captures confirm it.

`index.html` owns the steps via `data-manual-update-step="0"` through `"5"`.
The existing `quest-ui.js` navigation owns entry, local step selection and focus;
`operator-layout.js` retains the Settings/Update return chain. Help remains
accessible from Settings when disconnected and during native recording or
recovery. It does not open the Store or dispatch update, kiosk or experiment
commands. Native session/recovery guards still govern the underlying flow.

Background-music integration exposed24px overflow at390×844/200% when a prior
archive visit left its status visible in active Settings. The update-link group
now spans the existing active Settings grid and uses two internal columns.
The mounted regression covers RUNNING, PAUSED, FINALIZING and SAVING Settings
and their ECG dialogs, preserving48px targets, full Pretext labels, music
controls, native authority and return focus. Qualification passes all eight
affected states,30 guide renders and the combined429-state smartphone GUI gate;
the separate35-render Update suite also passes.

## Authentic screenshot follow-up

No screenshots or placeholder image boxes ship in this increment. On the
user's return, capture the real headset route together; do not access the
personally used phone or headset for this preparation task.

- Capture the separate Meta-menu escape and stimulus return without a run.
- Capture the actual Store entry route, the Developer channel identification,
  and Atmungsraum's Update action; record the observed labels and Quest version.
- Capture real download/install completion, app reopen and PWA APK readback.
- Check the captures for account, pairing, participant and recording information;
  publish only approved cropped/redacted images, retaining originals privately.
- Bind each approved image to its existing step with descriptive German alt
  text, intrinsic dimensions and local paths. Add no empty image containers or
  screenshot notices before authentic assets exist. Verify image visibility and
  text together at320px/200%, landscape, keyboard navigation and Back/focus.

`python -B tools/test-manual-update-help.py` mounts the real app with inert
transport and actual Pretext, covering all six steps across the five established
viewport/text cases, disconnected entry and preserved storage, draft, recording,
pending journal and focus. The existing repository Playwright fixture provides
inert rendered verification. No test server/device port is needed. Runtime
update detection and native SDK logic are unchanged.
