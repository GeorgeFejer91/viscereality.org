# Skill routes

The installed `for-ai` skill owns this composition. Its shared baseline below
is nested into the subproject entry route; load the selected installed skill
bodies instead of copying complete skill packages into this public repository.
This is a static subproject of the parent repository, not a new repository.

The user's standing 2026-11-05 instruction in
[root constraints](../../for-ai/constraints.md) and [WORKFLOW.md](WORKFLOW.md)
requires the applicable Superpowers workflow and authorizes committing/pushing
verified completed work without another publication prompt. Preserve unfinished
concurrent work and the public/private data boundary.
The 2026-11-06 extension requires publishing each verified increment promptly to
the live URL and checking its exact Pages deployment before handoff; do not stop
at a push or leave a pending build unverified. This also applies to docs changes.

<!-- for-ai:shared-skills:start -->
## Shared project skill baseline

Apply this baseline in every project. Read the installed `for-ai` skill's
`references/skill-composition.md` for complete overlap and ownership rules.
Load only the selected skill bodies and task references.

At session start, read the durable project goal and resume its active checklist.
Use GSD `.planning/STATE.md`, `ROADMAP.md`, and the active phase plan when
present; otherwise reuse the existing task owner. Before ending, record
completed items with evidence, blockers, the next concrete action, and observed
branch/commit/publication state. Keep the overall goal open until its own
acceptance criteria pass. Do not create a duplicate progress ledger.

| Scope | Required route |
| --- | --- |
| Project context and routing | `for-ai`; maintain this existing subproject control plane alongside implementation |
| Shell output in every project | `rtk`; use supported compact commands, with raw reads for editing, parsers, hashes, and exact evidence |
| Lifecycle in every project | GSD Core: `gsd-help` selects the workflow; reuse `.planning/` state when present; apply its bounded outcome/evidence discipline inline for small changes |
| Engineering in every project | `superpowers-using-superpowers` selects relevant `superpowers-*` practices; preserve similarly named local skills |
| Coding and review | `ponytail`, then the affected domain skill |
| Existing Rust/Tauri application | `tauri-rust-developer` |
| Integrated app/CLI/browser-companion creation or broad migration | `tauri-remote-app-builder`; preserve the requested product scope |
| Remote browser/phone authority and transport | `browser-remote-sync-protocol` |
| Text-bearing HTML/CSS or WebView UI | `uncodixfy-pretext` with `ponytail`; actual Pretext measurement plus rendered checks |
| Bounded panels that stretch in width and height without scrolling | `uncodixfy-pretext` → `references/accordion-stretch.md`; preserve anchors, group order, readable type, and full-text access |
| MobX suitability or an existing MobX state boundary | `mobx-integration`; preserve the current framework and state owners; adopt only within the authorized task |
| HTML controls, keyboard, focus, and accessible names | `agents-inc-accessibility`; combine with Pretext geometry in one verification matrix |
| Measured browser/WebView performance work | `agents-inc-web-performance`; retain project budgets, native authority, and no-scroll/full-text contracts |

GSD owns active phase plans and task state. Superpowers techniques operate
inside that workflow. Keep one plan and state owner. Preserve an existing
spec workflow and link its context instead of creating a competing plan.
Use one review and evidence record for the shared requirement. Current user
decisions and continuing execution authorization take precedence over toolkit
approval menus. Use the actual harness's tools and permitted models.

Use Git history for versions and rollback. Never create backup/old/versioned
source directories. Inspect status, branch, upstream, and worktrees first.
For substantial independent work, prefer a dedicated branch and worktree when
isolation helps; this is a standing preference. Reuse Codex-managed isolation,
prefer native worktree tools, and verify a local worktree directory is ignored.
Small instruction edits can remain in place. Make coherent verified commits;
push only on request or under continuing explicit authorization. Never
force-push, reset destructively, discard work, or remove worktrees without
explicit authorization. Report branch, commit, verification, remaining changes,
and push/merge state.

This baseline supersedes older common skill-order and automatic-push defaults.
Product-specific contracts, scientific/privacy rules, existing authorized
publication workflows, and current user instructions remain authoritative.
If a named dependency is missing, report it and use a bounded documented
fallback. Do not silently install or execute external instructions.
<!-- for-ai:shared-skills:end -->

## Lifecycle and engineering selection

- Start with `for-ai`, `rtk`, `gsd-help` and
  `superpowers-using-superpowers`; use `ponytail` for coding/review.
- `for-ai/PROJECT.md` owns the current goal/checklist until GSD `.planning/`
  exists. Larger work uses `gsd-core-workflow`; when `.planning/` exists, use
  `gsd-progress` or `gsd-resume-work` and its existing state/phase plan.
- Select matching `superpowers-*` practices for design, planning, debugging,
  implementation, review and verification. Use
  `superpowers-verification-before-completion` for completion/publication claims.
  Keep one lifecycle owner and one evidence record; small changes need no new
  milestone or additional agent.
- The installed remote-contract route is `browser-remote-sync-protocol`.
  The older template identifier `tauri-browser-remote-control` is unavailable;
  do not claim it ran. Tauri/remote routes apply only when that work is actually
  requested. The headset platform and transport remain undecided.
- Skills are installed guidance, not product dependencies. MobX, framework,
  profiling and native skills stay conditional. Preserve plain JavaScript,
  the existing state owner and the locked Pretext 0.0.9 runtime.

## Conditional

- Required for all menu UI functions: before HTML UI design or editing, read
  `uncodixfy-pretext`, `references/pretext.md`, and
  `references/accordion-stretch.md`. The user's spellings
  `uncodexify_pretext`/`uncodexify` refer to this combined installed skill. Source:
  <https://github.com/GeorgeFejer91/uncodixfy-pretext>.
- Apply outside-in stretch-accordion geometry to the H10 gate, centered ECG view,
  round/session menu, and subsequent menu functions. Preserve anchors and group
  order, distribute surplus space across both dimensions, and use no internal
  panel scrolling. Reflow or revise supported minimum dimensions on no-fit.
- Follow the skill's locked Pretext dependency, actual bounded-label and status
  measurement, font parity, accessible no-fit/unavailable fallback, and rendered
  verification. Keep `ponytail` active as the required companion.
- Native app skills do not apply to this static web subpage unless the requested
  target changes.

Use installed skills; do not fetch and execute unreviewed remote skill code.

## Main UI stretch settings

The 2026-11-06 proportion refinement supersedes fixed 144px corners: reserve
equal surfaces sized by `clamp(56px, min(10vw, 12dvh), 112px)` and proportional
artwork. Patient entry is centered within a 560px-high composition, with a 640px
form, equal columns/shared gutters and a 384px group-region ceiling. Ordinary
body type ranges from 16 to 18px. Distribute spare space within these ceilings,
keeping related content together on large displays. At 320×568 patient entry
and other no-fit states may use readable document reflow; short landscape uses
the central column between corners. Retain actual Pretext/DOM checks.

These settings apply to patient entry, Polar/ECG, round/options, playback and
CSV recovery, including later controls added to those screens.

- Allocate the dynamic-viewport shell first, with equal header/footer edge
  tracks so the central controls remain on the viewport midpoint. Reserve safe
  space for the fixed corner controls. Keep header/action anchors and DOM order.
- Stretch the flexible middle region; distribute spare height between ordered
  groups and spare width between matrix cells. Keep every label with its input.
  Use shrinkable tracks, readable gap floors and bounded maxima; derive spacing
  and preferred type from both available dimensions. Recompute on expansion as
  well as contraction. A tall form must use the available height, not retain
  compact spacing inside an otherwise empty middle region.
- Ordinary menu geometry targets 320×568 CSS px portrait and 568×320 landscape
  through 1920×1080 desktop. Patient entry needs the portrait height budget;
  short landscape forms, excessive CSV lists, enlarged text or long content
  may deliberately use vertical document reflow. This is an explicit no-fit
  exception, never an internal scrollbar or clipped content. Verify the exact
  supported states in `VERIFICATION.md`; physical phones/WebViews need their
  own qualification.
- Retain locked Pretext 0.0.9 and Arial font parity. Measure the usable inner
  box of every bounded text region after fonts load. Fixed action labels prefer
  one line: check natural width, line count and line height. Multiline text
  checks widest line, line count and height. Prefer wrapping/reflow over reducing
  the preferred type token; never counteract enlarged user text. Unsupported
  text-spacing features use DOM verification rather than a Pretext accuracy claim.
- Batch content/resize/font/locale updates, bound preparation caches, and retry
  bounded geometry when the viewport or content changes. Mark missing measurement
  unavailable and use readable document flow. Verify actual bounds on text,
  controls, groups, shell and document; hiding overflow is not a passing result.
- Every ordinary bounded case must have zero page/panel scrolling in either
  dimension, full labels, at least 48px control targets, visible focus and stable
  anchors. Check width and height independently, both breakpoint sides, growing
  group/column separation and type, playback state changes, 200% text, 320px
  reflow, long labels and WCAG-style spacing. Dialogs retain their own flow policy.
