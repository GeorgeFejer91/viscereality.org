# Operator layout implementation plan

> Execute using Superpowers subagent-driven-development. PROJECT.md remains the
> lifecycle owner; this is its bounded layout implementation checklist.

**Goal:** focused sequential operator pages matching actual Viscereality code.
**Architecture:** persistent static DOM with a local view resolver; existing
domain models and native command handlers stay authoritative.
**Stack:** existing plain JS/HTML/CSS and real Pretext0.0.9.
**Spec:** [UI-LAYOUT.md](UI-LAYOUT.md).

## Global constraints

No native/APK/audio changes, new features, stricter readiness gates, automatic
commands on navigation, discarded drafts/journals, or new dependencies.
Keep actual-radius500ms freshness/280ms playout,100ms visible-preview reads,
48px controls, main900s, existing input modes and exact command receipts.
Standing user publication authority applies; no redundant approval loop.

### Task 1: Implement persistent sequential pages

Own index.html,quest-ui.js,style.css,app.js, new operator-layout.js and focused
layout tests. Read the spec plus private flow-code-audit.md/site-code-reference.md
under the dated UI validation folder. Preserve domain files unchanged.

- [x] Add failing pure view/navigation tests: local Back/Next/settings emit no
  commands; native recording/recovery override obsolete preparation; missing
  signals cannot block entering input choice/Controller fallback.
- [x] Add static screens and one local view resolver, retain original IDs and
  action handlers; route only local step names, never identity in URLs.
- [x] Split participant/signals/input/calibration/tracking/session/recordings
  and settings, preserving explicit commands and shared status/recovery.
- [x] Apply actual root-site primitives, natural header geometry, responsive
  readable reflow and existing full Pretext measurement.
- [x] Run focused tests and current model suites; record known fixture updates
  needed. Self-review the complete diff. Do not publish independently.

### Task 2: Qualify actual rendered flows and integrate

Own tools/capture-phone-screens.py, integration tests, cache versions/live
verification helper and any evidence-driven corrections coordinated with root.
Consumes Task1 static screens/local resolver. Use actual mocked gateway, not
static HTML screenshots.

- [x] Adapt existing fixtures to the new local sequence, keeping original
  command, receipt, acceptance and interoperability assertions.
- [x] Render/inspect all main pages and relevant error/recovery states at the
  contract sizes. Check zero-command local navigation, explicit input changes,
  running/finalizing controls and six-poll draft/scroll/focus preservation.
- [x] Run full zero-skip Node gate; inspect independent task and whole-change
  reviews, correct actionable findings, rerun affected checks.
- [x] Version the complete changed module graph, update PROJECT/For-AI evidence,
  commit/push/ordinary PR merge, exact Pages and live-file/render checks.
- [x] Record physical phone verification pending, untouched APK11 and deferred
  wearer tests. Keep the overall hardware milestone open.
