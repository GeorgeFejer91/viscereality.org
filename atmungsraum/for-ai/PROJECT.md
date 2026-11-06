# Project

## Collected-data ZIP and Wormhole QR sharing — 2026-11-06

Requested outcome: package all saved browser exports, upload to Wormhole and
share its full download URL as a QR code. Reuse `data-files.js` and its ZIP
writer; do not change participant data, recording, identity or retention.
Wormhole `/api/room` currently answers cross-origin upload preflight with405
and no CORS permission. This static Pages app has no upload backend. A preference
question is pending for a Wormhole handoff versus hosted automatic upload or
another provider. The bounded static-site increment implements the explicit
handoff; do not present it as fulfilling automatic upload.

- [x] Inspect existing ZIP/storage flow and verify Wormhole's official security,
  FAQ/roadmap and the actual API preflight. No participant files uploaded.
- [x] Add a local-only QR generator with strict complete Wormhole-link validation;
  test key preservation and reject unrelated/unsafe/incomplete URLs.
- [x] Integrate a clearly labelled ZIP → Wormhole → QR flow in the existing
  accessible file dialog, preserving downloads and native sharing.
- [x] Verify rendered layouts, focus, failure/recovery and QR decoding. All24
  Node tests, UI/context, seven QR layouts and ten existing files-dialog cases
  pass; see VERIFICATION.md. Independent review found no actionable issues
  and separately passed the data-file regressions and whitespace check.
- [x] Review, commit/push and check exact Pages deployment/live assets.
- [ ] Fully automatic upload: resolve backend hosting/provider choice before
  claiming the original one-click outcome is complete.

Entry: canonical checkout on `main`, synchronized at `a216168`; audio sidecar
publication is separate work. RTK is unavailable at the configured path; use
bounded native output. Keep the overall sensor/headset and wording goals open.
Release: implementation `e11a926531e72f6ef6de804777c0463f4c1dd2b9` is on
GitHub. Concurrent audio evidence commits superseded its initial Pages run;
the containing `e5f58b49de54502fe96564ecfce45008a35fe964` release completed
successfully in [Pages run42247056025](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/42247056025).
Six live runtime/license files match Git SHA-256 exactly, both URL variants
return200, and the live seven QR/layout cases plus ten existing files-dialog
cases pass. Independent QR decoding preserves the full link/key; zero external
requests occur during QR creation. No participant data was uploaded.

Observed `main`/`origin/main` synchronized at `e5f58b4`, clean before this
evidence-only update. This note's publication keeps runtime bytes unchanged;
verify its exact deployment and published note before handoff. Next product
action: resolve hosting/provider choice for fully automatic upload; that
original goal remains open. Participant exports remain local and untracked.

## Matching numbered audio / CSV / TXT groups — 2026-11-06

The current request requires scenario-owned audio folders and matching numerical
prefixes for each WAV, CSV and TXT. Reuse `assets/audio/scenarios/haupt-15/`
and the four `probe-*` folders; `box-20/` remains an engine-only diagnostic.
The latest remote already implements independent numbered WAV/JSON/TXT cues.
Complete that structure with one same-basename CSV per cue, listing every
occurrence in the authored timeline. Keep first-use numbering and first-onset
timestamps, with the complete playlist/timeline specifying repeated cycles.
No wording, audio bytes, runtime paths or protocol timings need to change.

- [x] Reconcile with the newer scenario-placeholder release; preserve the older
  local draft snapshot in Git stash `codex/pre-scenario-folder-sync-2026-11-06`.
- [x] Baseline: 22 Node tests, six JSON/CSV pairs and 53 packages pass.
- [x] New sidecar regression fails on all 53 missing per-cue CSV files.
- [x] Extend the existing packager, generate matching CSVs, document the order,
  and verify each occurrence, transcript, status and frame time.
- [x] Review, commit/push, verify exact Pages deployment and live sidecars,
  then open the current scenario library for the user.

Entry: main synchronized to `eafec8a`; clean after integrating the newer remote.
Previous Pages run `42241115947` timed out; recovery run `42243359695` is queued.
Current evidence: all 53 matching sidecars cover the 890 authored events; the
Python regression, generator check, all 22 Node tests, six scenario JSON/CSV
pairs, UI state/context and whitespace checks pass. WAVs, authoritative JSON,
playlists and runtime source are unchanged. Independent review found no issues
and separately confirmed that the 53 CSVs partition all 890 events exactly once.
Release evidence: sidecars committed/pushed as
`a216168e92b22f09fef1fb6453fc5da3156d7ba9`. Its exact Pages deployment API
reports `succeed`; Actions run `42243976605` attempt 2 was cancelled after the
backend published successfully when the concurrent `e11a926` release started.
All 67 affected/public context files match local bytes: 53 cue CSVs, six
scenario README files, six protocol JSONs, the scenario index and app HTML.
The active D: checkout has the verified sidecars and its scenario library was
opened in Explorer. Existing recordings and silent-placeholder status are
unchanged. The older draft snapshot remains in the original checkout's Git
stash. Concurrent unfinished publication notes and unrelated QR work are
preserved; they are not part of this sidecar change.

This asset-organization increment is complete. Next product action: user
approves or edits the draft wording before final speech generation. Overall
real sensor/headset/listening acceptance goals remain open. This evidence-only
follow-up must also finish its exact Pages deployment/live checks.

## Scenario folders and working silent-placeholder UI — 2026-11-06

The latest user requests authorize silent WAV placeholders now so the UI works;
final narration and wording remain pending approval. This supersedes the prior
text-only hold on runtime/publication, without authorizing new synthesized speech.
Each of six scenarios owns its independent numbered unique cues in
`assets/audio/scenarios/<scenario>/`: 53 WAVs, matching JSON/TXT, asset CSV,
timeline CSV, manifest and playlist. Numbers encode first-use order and the
filename includes first onset; repeated cycles reuse the unique cue.
TXT is the intended draft script, while `spoken_text` is empty and metadata
explicitly says `silent-placeholder` / `awaiting-wording-approval` (or no-speech).
All original shared recordings remain byte-for-byte unchanged.

The UI loads these scenario assets, caching identical PCM by hash within one
context. The saved scenario JSON remains runtime authority and pins the shared
protocol manifest. Main goes directly to its sole 15-minute sequence; all slots
still total 900 seconds, with preparation26s, measurement15s at26–41s,
Instruction2 40s, fifty16-second cycles and Outro19s. Participant labels read
“Vorbereitung” and “Zwei tiefe Atemzüge”. Timing events remain available; the
future H10 accelerometer measurement remains not-integrated with null value.
Receipts record the scenario hash and `audio_asset_mode: silent-placeholders`.

- [x] Copy the complete sparse checkout, Git metadata, drafts and ignored assets
  to the user-selected secondary drive; verify all 1,489 files (97,320,133 bytes)
  by SHA-256 and recreate the internal dependency junction at its new target.
  Serve and edit the relocated copy. The original was left for manual cleanup
  after automatic review blocked recursive removal; Explorer selects it.
- [x] Prepare 43 individual original-library/sample TXT drafts and a review index;
  Instruction1 proposes “Wir beginnen mit zwei ruhigen, tiefen Atemzügen.”
- [x] Generate independent scenario placeholders and connect runtime/CSV/export
  to those assets; preserve original speech and all protocol timings.
- [x] Observe missing-registry regression fail, and reproduce native decode input
  detachment in a test before fixing it. Pass all22 Node tests, six scenario
  CSV pairs,53 placeholder packages, UI/context checks and12 native offline
  renders at44.1/48kHz with zero sample error. Verify silent preview export.
- [x] Pass five actual browser playback routes,26/41-second transitions,
  pause/resume, replay, CSV export and missing-audio retry;18 enlarged/localized
  text cases pass, screenshot inspected, no page errors. Independent review
  found no issues and audited all53 asset groups/manifests/playlists.
- [x] Commit/push and verify exact Pages deployment and affected live assets/UI:
  feature6f097cf is included in eafec8a, successful Pages run42243359695
  attempt2. All265 changed public files match Git bytes; both entry URLs return
  200. Five live playback routes,26/41-second transitions, pause/resume/export,
  replay/retry and18 enlarged/localized text cases pass without page errors.
- [ ] User approves draft wording before final speech creation/replacement;
  after approval, align exact TXT/JSON/CSV/recordings and reverify timing.

Entry: main/origin at `990172adeaadf3bfcdf1a1a78c79badcef18363b`;
previous exact Pages release `42226744863`. The verified placeholder change is pushed as
`6f097cf91d52036da2781c12578e2d0ecacfbb4f`, with a clean main/origin. Pages
run `42241115947` built successfully, but attempt1 timed out in updating_pages
after10minutes; failed-job attempt2 was cancelled immediately. Its existing
site artifact is2,146,337,782bytes with the standard1GB warning; that warning
alone does not establish the cause. Recovery run42243359695 initially failed
because the timed-out6f097cf deployment still held the Pages lock. Explicitly
cancelled only that stale deployment through the Pages API, verified
deployment_cancelled, then reran the eafec8a failed job. Attempt2 succeeded;
live file and browser checks above pass. The concurrent a216168 increment adds
matching per-cue CSVs and preserves runtime/audio; its publication has its own
top-level checklist. Preserve it and its active deployment. The local server
serves the relocated checkout; machine-specific paths and migration hashes
stay outside published source. This placeholder UI increment is complete.
Final narration awaits wording approval. Overall real sensor/headset,
physical-phone and human listening acceptance remain open.

## Main-round calibration and one 15-minute scenario — 2026-11-06

The latest user instructions supersede the earlier two-main-option/silent-baseline
contract: Haupt-Runde goes directly to one final 15-minute Start screen. There is
no main duration menu or five-minute baseline. All assets together total exactly
900 seconds: Instruction 1 (26 s), Calibration (15 s), Instruction 2 (40 s), fifty
complete 16-second inhale/pause/exhale/pause cycles (800 s), and Outro (19 s,
including seven seconds of quiet settling before the unchanged ending speech).
No additional rest cue interrupts the main cycles. Practice remains unchanged.

The shared manifest names eight main asset classes; scenario JSON drives playback
and its CSV contains exact times, asset paths and transcript_de. Instruction 1
uses the user-requested Secret Tunnel existing-instructor clone. The original
nine WAVs remain unchanged. Calibration is a separate silent measurement window
at source frames 1,146,600–1,808,100 (26–41 s). Receipts and the advance
`atmungsraum:calibration-scheduled` event expose its audio-clock start/end.
The later breathing input will be the ECG sensor's accelerometer. Value, unit
and method remain unset with status not-integrated; no physiological result is
fabricated from synthetic ECG. See audio/calibration.md.

- [x] Verify Respyra source; distinguish its comfortable natural-range protocol
  from our requested two-breath adaptation and original German narration.
- [x] Package reusable Instruction 1 / silent Calibration / Outro assets and
  transcript JSON/CSV, using cached checked Secret Tunnel speech without retiming.
- [x] Replace main duration choice with direct 15-minute entry, remove haupt-10,
  author six scenario JSON/CSV pairs, and add explicit calibration clock metadata.
- [x] Observe the changed main-choice regression fail, then pass all 21 Node tests,
  six scenario pairs, UI state and context checks. Preserve pause/resume/identity.
- [x] Verify current native offline renders, responsive/Pretext layout, actual
  browser calibration transitions/CSV and timing hook, then final independent review.
- [x] Commit/push and verify exact Pages deployment and affected live assets/flow.

Entry: clean canonical main at 64341d9; continuing publication authority applies.
RTK is unavailable, so bounded native commands are used. No prior feature push
occurred before the user's later corrections. The earlier two-option/48-second
composite implementation was locally verified but is superseded, not a release.
Current verification: 21 Node tests; six scenario pairs; UI/context; full PCM
decode/integrity; 12 native Chrome offline renders at 44.1/48 kHz with zero sample
error; 137 responsive/Pretext cases and 18 main/root enlarged/localized text cases;
five actual playback routes, advance timing hook, CSV phases/duration, calibration
entry/exit at 26/41 s, pause/resume/replay and missing-audio retry. Independent
review found no runtime issues and confirmed original nine WAVs unchanged.
Publication verified: feature commit 23b5f2821eb5d0fd13bdc71af7075268fd3b2e30,
Pages run 42225450192 completed successfully; Pages API reports built for that
exact commit. All 54 changed public files match Git bytes and both entry URLs
return 200. Removed main10 JSON/CSV/review URLs return 404. Live Chrome passes all
five playback routes and 18 enlarged/localized text cases, including direct main,
actual 26/41-second calibration transitions, pause/resume, CSV phases/duration,
replay and missing-asset recovery, with no page errors. Live screenshot inspected.
At this evidence write main/origin both match 23b5f28 and source is clean; these
closing notes are the only pending changes. Publish/check this documentation
increment under the standing rule. No runtime blocker remains. Next product
work is real H10 accelerometer/ECG and headset integration, physical-phone
qualification and human listening acceptance; those overall goals remain open.

## Responsive proportions — 2026-11-06

Rebalance the patient screen and permanent controls for smartphone and desktop.
Use a centered patient composition capped at 560px high, a 640px-wide form,
aligned equal columns and shared column gutters. Form groups still distribute
space within a 384px ceiling. Body type tops out at 18px; user enlargement keeps
readable document reflow. The three equal corner surfaces now scale from 56px
to 112px with both viewport dimensions; this supersedes fixed 144px sizing.
Preserve the concurrently completed belt-artwork glow and all interaction/data
owners. No participant writer or audio behavior changes belong to this task.

- [x] Apply scoped CSS geometry and refresh the entry cache tokens.
- [x] Verify 149 responsive/Pretext/text/focus cases, ten folder/download cases,
  two unavailable-measurement fallbacks, all 19 Node tests, seven scenario pairs
  and UI/context checks. Independent review has no actionable findings.
- [x] Commit/push and verify the exact Pages deployment and live assets:
  layout commit `f2d683c74bf1c32d56f36f183d27b31dc1938829` is included in
  `b164c805c1df1b43f565d3ee69aeffda53eb7307`, successful Pages run `42213877838`.
  All seven changed public files match Git bytes and both entry URLs return 200.
  All 149 live layout cases and ten live folder cases pass with no page errors.

Entry: canonical main at `a792da7`, with existing belt-glow changes preserved;
their owner committed them separately as `7e07b9c` during this task. RTK is absent;
bounded native commands are used. A concurrent documentation update superseded
the first Pages run; its successful descendant above contains the exact layout.
At live verification main/origin both match `b164c80`, with a clean working tree
before this documentation-only closing evidence. This UI increment is complete
with no blocker; publish/check the closing evidence under the standing rule.
Next overall acceptance actions remain real H10/headset, physical-phone sharing
and listening checks; those product goals stay open.

## Coherent Polar belt and frame glow — 2026-11-06

Remove the separate status dot. The original belt artwork and its frame
pulse together: green with ECG (four-second cycle), red after signal loss
(two-second cycle), neutral while waiting, and steady with reduced motion.
The PNG is used as a CSS luminance mask so the glow follows the belt artwork
rather than its opaque rectangular background; source bytes remain unchanged.

- [x] Replace the dot and implement synchronized symbol/frame glow using the
  existing state owner, accessible status text and shared color/cycle tokens.
- [x] Verify 24 Chrome state/size/reduced-motion cases, actual pulse variation,
  source-mask loading and folder open/close; inspect green/red screenshots.
  UI-state/context and whitespace checks pass; review has no actionable findings.
- [x] Commit/push and verify the exact Pages deployment and affected live behavior:
  `7e07b9c6b0abd9c8d3e22d54abd339bec130e297`, successful Pages run `42213036555`.
  All six published files match Git bytes; both entry URLs return 200. All 24
  live glow cases pass without page errors, including both cyclic border states.

Entry: clean main/origin at `a792da7`. The glow release is complete with no blocker.
During live verification the independent proportions owner advanced main/origin
to `f2d683c`, preserving the glow and updating corner dimensions under its own
task. Preserve that task's publication state. This closing evidence changes
documentation only; publish/check it under the standing rule. Overall real
H10/headset, physical-phone sharing and listening acceptance remain open.

## Larger corners and recovered Polar artwork — 2026-11-06

The three permanent corner surfaces are 144×144 CSS px: three times the previous
48px recenter/folder dimensions. Their SVG artwork grows threefold too. The Polar
status keeps its accessible status semantics and green/lost/reduced-motion states.
It embeds the original dissertation Polar H10 belt illustration. The recovered
source is PNG, not SVG: Dissertation-scaffold-overleaf, Study 4/figures/PolarH10.png,
Git blob `71fef4c5acd3cb20122be459675963d476d5bc89`; shipped bytes match exactly.
No standalone belt SVG was located in the searched publication/dissertation trees.

- [x] Enlarge all three surfaces and reuse the existing belt illustration.
- [x] Reserve corner space and verify readable portrait/landscape/zoom layouts,
  fixed positions, all menus, signal states and folder/playback behavior.
  All 19 Node tests, seven scenario pairs, UI/context/whitespace, 85 new rendered
  cases and ten folder cases pass. Review has no actionable findings.
- [x] Commit/push and confirm the exact Pages deployment and affected live assets:
  `0a3c99a83301c3650d79dca1f88a52389494b2c8`, successful Pages run `42208617785`.
  All eight published files match Git bytes; both entry URLs return 200. The 85
  live corner cases and ten live folder cases pass with no page errors.

Entry: clean main/origin at `c962c20`. At live verification, local/remote main
match `0a3c99a` with a clean tree before this closing documentation-only update.
The requested increment is complete with no blocker; publish/check the closing
evidence under the standing rule. See VERIFICATION.md for the revised reflow
boundary. Next product acceptance actions remain real H10/headset,
physical-phone sharing and listening checks; those goals stay open.

Closing-evidence publication recovery: `463c842`, Pages run `42209351688`, built
successfully but attempt 1 timed out in `syncing_files` after ten minutes; the
failed-job-only retry was immediately reported cancelled. The artifact warning
is 2,143,894,180 bytes against a stated 1 GB allowance; this is the whole existing
site, not a diagnosis of the timeout. The verified `0a3c99a` UI stays live.
Publish this updated evidence as a fresh commit/run, then verify its exact status
and public bytes. No runtime or unrelated media changes are part of recovery.
Resolved: `a792da7` deployed successfully in run `42211204098`; all eight changed
public files match Git bytes and both entry URLs return 200. Local/remote main
matched with a clean tree at handoff.

## Folder-only data entry — 2026-11-06

Remove both CSV-Dateien footer links. The permanent bottom-right folder is the
sole visible entry to collected data, including retained exports after reload.
Keep the legacy recovery URL and the shared download links used by the dialog.

- [x] Remove the redundant links and their obsolete visibility update.
- [x] Verify saved exports, folder/download/reload behavior, responsive layout
  and existing checks; review the complete diff.
- [x] Commit/push and confirm exact Pages deployment and affected live URLs:
  `0895f63213792e8a5994712b137620f1dd91ef70`, Pages run `42203147921` succeeded.
  All nine published files match Git bytes; both entry URLs return 200.
  Ten live folder cases plus download/reload/playback checks pass, no page errors.

Local evidence: all 19 Node tests, seven scenario pairs, UI/context/syntax and
whitespace checks pass. Chrome passes ten folder cases plus downloads/reload/
playback and 161 stretch cases, with no page errors. The remaining patient link
has a 240px width budget after its old 180px singleton width caused no-fit at
768×1024. Focused review has no findings; see VERIFICATION.md.

Entry: clean canonical main at `ba7efdb`, matching refreshed origin/main.
The concurrent instruction-evidence owner advanced main/origin to `7bbd231`
without including these edits. At live verification, main/origin both point to
`0895f63` and the tree is clean before this documentation-only evidence update.
This task is complete with no blocker; publish/check this closing evidence
under the standing rule. The next product acceptance actions remain real
H10/headset, physical-phone sharing and listening checks; those goals stay open.

## Prompt live publication is standard — 2026-11-06

The user requires future agents to push completed verified changes to the live
site as soon as ready. The root standing rule now explicitly covers prompt
incremental publication, no repeat permission request, exact Pages tracking,
affected live-URL verification and docs-only updates. Root/local entrypoints,
skill indexes and WORKFLOW.md route to that single authority.

- [x] Refresh origin/main: clean local/remote main both `3f62dec` at entry.
  Its Pages run `42080146068` is confirmed successful.
- [x] Update the existing root/subproject instructions and decision owner.
- [x] Run scoped context and Git whitespace checks; review the eight-file
  instruction-only diff. No runtime regression suite is needed for prose changes.
- [x] Push `ba7efdbfe225f1cd2d3d88cc5a5398069dd2269e`; Pages run `42202202945`
  succeeded. All eight public changed files match Git bytes and both site entry
  URLs return 200. Context/remote equality checks pass.

This instruction-only release is verified live. No blocker remains for the rule.
Observed main and origin/main both `ba7efdb` before this closing evidence update;
the separate folder-only task has since added unfinished changes in shared files.
Preserve those changes and stage only this task's evidence. Push this evidence
follow-up and verify its deployment before handoff. The overall H10/headset/
physical-phone/listening goals remain open; their acceptance is the next product
action. Live receipts remain ignored in `output/playwright/live-release-assets-ba7efdb.json`.

## Publish folder/sharing and standing workflow instruction — 2026-11-05

The user explicitly requests pushing everything currently completed and making
Superpowers plus GitHub synchronization a standing instruction for future agents.
Root constraints own that continuing authorization; root/local AGENTS.md, skill
routes and WORKFLOW.md make it discoverable. This release includes the folder,
ZIP/sharing feature, final Pretext refinements and all current project evidence.
Keep participant data, voice caches and ignored QA output local.

- [x] Fetch remote main and review all current source/assets/tests/notes;
  starting local and remote HEAD both `7233468`.
- [x] Record the standing workflow/publication instruction in the existing
  root and subproject control planes.
- [x] Run fresh complete tests, context/scenario checks and rendered acceptance:
  19 Node tests, seven scenario pairs, UI/context/syntax/whitespace, cold-open
  regression, ten popup cases and all 161 stretch cases pass with no page errors.
- [x] Commit/push all 20 reviewed intended files: `a52d56a875cc6c30f149f1c4831e098ac12c8711`;
  local main and remote main match, with a clean tree before this evidence update.
- [x] Verify Pages run `42079217920` succeeded for that exact commit; all 20
  published files match Git bytes and both entry URLs return 200. All ten live
  popup cases plus download/share-boundary/reload/playback checks pass, no errors.
  This closing evidence accompanies a final documentation-only commit/push.

The folder release and standing instruction are complete. No release blocker
remains. The overall real H10/headset, physical-phone share sheet and listening
acceptance goals remain open; those are the next product acceptance actions.
Git observation immediately before this evidence update: main and origin/main
both `a52d56a`, no modified/untracked source. This evidence-only follow-up must
also be pushed, with final SHA/synchronization reported at handoff. The standing
authorization is durable and requires no repeat publication permission.

## Saved-files folder and sharing — 2026-11-05

The user requests a permanent bottom-right folder button with an authored SVG,
a scrollable popup listing downloaded session data, and all-file ZIP/email/app
sharing. The native dialog lists locally retained CSV exports, including records
recovered after reload, without navigating or stopping audio. It cannot enumerate
the operating system's Downloads folder. ZIP download and the device share sheet
use one archive of every saved CSV from this browser. Email/WhatsApp destinations
depend on installed apps and browser file-share support; unsupported browsers
retain ZIP and individual CSV downloads.

- [x] Add the fixed safe-area-aware folder SVG/button and accessible modal list.
- [x] Reuse the shared writer's retained exports; preserve CSV bytes and names,
  with session UUID directories only for colliding filenames.
- [x] Verify ZIP interoperability, sharing/cancel/error fallback, keyboard focus,
  reload recovery, active playback, responsive/enlarged layouts and regressions.

Evidence: 19 Node tests including independent ZIP extraction/CRC/Unicode checks;
10 new Chrome overlay layout cases plus functional sharing/download and 61-record
recovery checks; the existing 161-case stretch matrix; UI/context and seven
scenario-pair checks all pass. See VERIFICATION.md. The platform share API alone
was simulated; no email/WhatsApp message was sent.
Focused review is complete: a hydration-triggered read loop was fixed, and the
cold-open regression now lists 60 files with two full export-store reads.

Publication is now verified by the folder release above: `a52d56a` is pushed and
live, including the final Pretext refinements and preceding release notes.
Native Chrome passed the ten overlay cases and functional file/download/reload/
playback sequence against the public URL, with zero page errors. The next feature
acceptance action is real native sharing on supported phones.
The overall real H10/headset and listening acceptance goals remain open.

## Publish patient/ECG and recenter changes — 2026-11-05

The user explicitly requests pushing all current changes to the live site.
Include patient entry, continuous background mock ECG, session CSV recording,
the persistent recenter placeholder, tests and their For-AI notes. Keep local
participant exports, diagnostic output and voice/source caches ignored.
Clarification: Start records a segment of an independently running ECG stream;
it must never restart the source. Audio pauses leave ECG recording running.
DATA-LOGGING.md owns this contract; real H10 acquisition is still pending.

- [x] Refresh remote main and review all changed/new paths; starting HEAD and
  origin/main both `ac39ec4`.
- [x] Fresh local UI/context/scenario checks, 50 recenter layout/no-op cases,
  25 patient/integration cases and four fallback cases pass. Full integration
  includes actual >30-second signal loss and natural 68-second completion.
- [x] Verify the explicit background-stream regression with the complete suite:
  all 18 Node tests pass; separate recordings start at advancing source indices
  and exclude intervening idle time without changing the shared source epoch.
- [x] Review the three additional app/HTML/CSS stretch-layout edits that arrived
  during deployment. Their 50-case Chrome check passes with no page errors;
  two real browser recordings retain exact source values and advance across idle
  time, including 102 ECG samples recorded while audio time is paused.
- [x] Run the final stretch acceptance matrix: all 156 native Chrome cases pass
  with no page errors, including growth in both dimensions, short landscape,
  all six playback routes, reflow/spacing and missing-Pretext fallback.
- [x] Commit all reviewed changes, push main and verify the exact Pages build:
  `72334681b5e447f7298aacf58b255fc5543b99b6`, run `42060990832` attempt 3 succeeded.
- [x] Verify deployed assets and browser behavior: all 25 public changed files
  match committed bytes, both entry URLs return 200, and 50 final live Chrome
  cases pass with exact continuous-source CSV values and no page errors.

The Pages connector returned an internal error; use the existing authenticated
GitHub CLI/API for exact-commit build evidence. No hosting settings change is
needed: Pages is already configured from main, root `/`, on viscereality.org.
Initial release `e2b6e32` was pushed with local and remote SHA equal and a clean
tree at that instant; Pages run `42059584563` succeeded for that exact commit.
All 25 public changed files matched committed bytes, and 50 live Chrome cases
passed, including successive ECG recording segments and audio-pause capture.
The later three layout files are the only additional uncommitted product changes at the
next observation and are included in this same all-changes publication task.
The overall headset/H10 and listening acceptance goals remain open.

Release boundary: follow-up `7233468` contains the reviewed stretch source and
initial live evidence; it is pushed with remote SHA verified. The next shared
working-tree edits (CSV/ZIP export work and an additional Pretext capability
guard) arrived after this verified release snapshot and belong to their active
tasks. Preserve them; do not sweep them into publication while incomplete.
Pages run `42060990832` built successfully but attempt 1 failed creating the
deployment with GitHub HTTP 500. Retry only failed jobs, retaining the successful
build artifact; verify the deployment job and live bytes before declaring it live.

Final observation: main and origin/main both point to `7233468`; the exact
Pages build/deployment succeeded on attempt 3. This publication task is complete.
The final live recordings start at samples 7395 and 7689, with the first ending
at 7523; 103 paused ECG samples retain one audio time while values change.
Newer CSV/ZIP source/assets/tests/notes and the Pretext guard remain uncommitted
in the shared checkout, along with this closing evidence. Preserve that active
work and validate/publish it under its own task. Do not claim the working tree
is clean or that the newer feature is live. See VERIFICATION.md for receipts
and network/deployment recovery. Overall hardware/listening goals stay open.

## Stretch-accordion settings applied — 2026-11-05

Apply the installed `uncodixfy-pretext` accordion-stretch settings to the entire
main menu, including patient entry and CSV recovery. `SKILLS.md` owns the
geometry/typography rules; `VERIFICATION.md` owns the rendered acceptance matrix.
Keep the existing dark static UI, centered controls, patient/audio state owners,
and fixed recenter/signal controls. Stretch means distributing space between
ordered groups, not adding collapsible sections.

- [x] Make the durable layout settings and supported-size policy explicit.
- [x] Apply dimension-responsive group spacing and complete Pretext/no-fit checks.
- [x] Verify independent width/height resizing, anchors, full text, keyboard flow,
  enlarged text and unavailable measurement; record fresh evidence and Git state.

Evidence: 161 native Chrome rendered cases and two measurement-unavailable
fallbacks passed, including all six Start/Pause/Resume/End routes, independent
dimension growth, both responsive breakpoints, 200% text/spacing, long text/IDs,
locale changes and bounded-layout recovery. All 19 current Node tests, seven
scenario pairs, UI/context/syntax and whitespace checks pass. See VERIFICATION.md.
No blocker remains for this layout task. Physical phones/Safari/WebViews, real
H10/headset integration and listening acceptance remain open product checks.

During this task the concurrent publication owner committed the initial guidance
in `e2b6e32` and the core layout in `7233468` on `main`. Final capability/reflow
polish and these evidence notes are local at this observation, alongside separate
CSV-folder/dialog work in shared runtime files. Preserve that work; the publication
task above owns deployment. Next action: integrate the final reviewed refinements
through that release and verify its exact deployed bytes. No push was performed
by this layout task; no live qualification of these final refinements is claimed.

Starting state: shared canonical checkout on `main`, HEAD `ac39ec4`, with existing
uncommitted patient/ECG/logging and recenter work. Preserve those changes in place.
RTK is absent on PATH and at its documented path; use bounded native commands.
The overall real headset/H10 product goal remains open.

## Persistent VR recenter placeholder — 2026-11-05

Keep the icon-only `VR zentrieren` button fixed at the top left across every
screen, playback state and document scroll, including patient entry and CSV
recovery. Respect the device safe area and reserve heading space below it.
The local vector asset `assets/icons/recenter-vr.svg` shows a VR headset with
four arrows pointing inward from above, below, left and right toward the
central view target. It is editable SVG with no raster dependency.

The button is a UI placeholder: `type="button"`, `aria-disabled="true"`,
keyboard-focusable, with a German accessible name and unavailable description.
Do not attach a click handler, navigation, transport event or simulated success.
The later intended action is to recenter the headset's view to its central
field of view. Headset authority, reference frame, command/acknowledgment and
transport remain undecided; implement them only as separately requested work.

- [x] Add the persistent inert button and four-direction headset SVG.
- [x] Verify visibility, keyboard focus, no-op activation and responsive layout:
  50 native Chrome cases passed with real Pretext and no page errors; UI state,
  context, SVG XML/vector-only and whitespace checks passed. See VERIFICATION.md.

Git observation: canonical shared checkout on `main` at `ac39ec4`; patient,
mock-ECG and participant-logging changes were already uncommitted at entry.
This placeholder remains local alongside that work; no commit, push or deployment
was performed for the original UI task. The publication task above now owns its
release state. Its own UI scope is complete. The overall headset/H10
product goal remains open; the next recenter work needs an explicitly chosen
headset reference frame and command/acknowledgment contract before integration.

## Patient entry, mock ECG and data logging — 2026-11-05

Add a German patient-details screen **before** the Polar/ECG gate. Collect
exactly three surname letters (confirmed by the user), age in whole years, and one
explicit gender choice (male, female, other, prefer not to say). Reuse the
existing state owner, bounded Pretext layout and native accessible controls.
Show the derived participant ID. The corner ambient indicator breathes green
with signal, flashes red on loss (steady under reduced motion), and loss beyond
30 seconds returns to Polar while retaining patient details. Use a seeded
NeuroKit2-generated mock ECG continuously throughout the demo. Export a CSV for
every started run; keep identity consistent in all participant data logs.
[DATA-LOGGING.md](DATA-LOGGING.md) owns the enforced naming/schema contract:
`Fej_35_M_probe1.csv` / `Fej_35_M_main1.csv`, with condition-specific run indices.
Patient form state is in memory; saved CSVs, counters and linked audio receipts
persist in this browser. No participant values enter navigation URLs.

- [x] Gate all exercise routes/start until valid patient details are submitted.
- [x] Add the form before connection and an edit route; preserve reconnects.
- [x] Add the ambient signal state and the actual NeuroKit source.
- [x] Enforce CSV names/schema, persistent run indices and download recovery.
- [x] Verify validation, keyboard flow, rendered phone/desktop/reflow layouts,
  existing audio checks and context; record evidence and Git state here.

Evidence: 17 Node tests, UI-state/context gates, seven scenario pairs and the
25-case Chrome integration passed. Real >30-second loss and natural 68-second
completion produced the expected CSVs. Targeted fallback/review checks passed
keyboard gender/ID updates, no-JS form protection, missing-source retry,
unavailable Pretext, delayed-counter abort/failure ownership, signed lead-time
pause, offline checkpoint recovery and cross-tab live-session exclusion. See
VERIFICATION.md; local scripts/screenshots/receipts are ignored in output/.

This task is implemented locally. At last observation the canonical checkout
is on `main`, HEAD `ac39ec4`, with uncommitted source/docs and concurrent VR
recenter-placeholder changes in shared files. Preserve that independent work;
no push or live deployment check is part of this request. The recorded origin
ref matched the starting HEAD; no fresh remote verification is claimed. Next
product action remains listening acceptance and real H10/headset integration,
including actual acquisition timestamps and a shared study storage owner.
Those overall goals remain open. Recheck fresh Git state before integration.
The subsequent publication task above now owns release of this implementation.

## Combined publication and skill baseline — 2026-11-05

The user explicitly authorizes pushing all current repository changes,
including the playback controls, simplified transparent round SVGs, scenario
audio protocol and remaining verification notes. Update this subproject's
`for-ai/` with the installed For-AI baseline and preserve its existing product
contracts; the parent repository remains the Git/publication owner.

- [x] Nest the full shared baseline in `for-ai/SKILLS.md` and route to it from
  local `AGENTS.md` and `for-ai/README.md`; keep project-specific UI rules.
- [x] Run current context, scenario, audio and rendered-browser checks.
- [x] Commit all intended changes, push `main`, and verify remote SHA and Pages.
- [x] Verify deployed SVGs, scenario files and browser playback/layout.

Starting Git observation: canonical checkout on `main` at `0f5d05a`, three
commits ahead of refreshed `origin/main`; only the existing alignment-publication
notes remained uncommitted. Publication is authorized by this request. Keep the
overall H10/headset/video-integration and listening acceptance goals open.
Fresh checks passed 13 Node audio tests, seven JSON/CSV pairs, UI/context gates,
13 round-layout cases, SVG transparency, 24 player-layout cases and all six
scenario playback/retry routes. No browser errors occurred. See
`VERIFICATION.md`. Release `5a0e3a9` was pushed to `main`; Pages run
`42048199569` succeeded and the Pages API reported that exact commit built.
All 26 checked public files matched the release, including both SVGs, runtime
modules, shared manifest, fourteen JSON/CSV files and the baseline entry chain.
Live Chrome passed phone/desktop round layout and both branches, all six
scenario Start/Pause/Resume/End routes, exact receipt hashes and missing-file
retry recovery, with no page errors. The live desktop screenshot was reviewed.

The push request and nested-baseline task are complete. This final evidence is
saved in the following documentation-only commit; product files remain those
verified at `5a0e3a9`. At live verification, local/remote `main` matched and the
tree was clean. Next product actions: listening acceptance and real headset/H10
transport/video synchronization, with target-device evidence. Those overall
product goals remain open; recheck fresh Git state at the next session.

## Playback controls task — 2026-11-05

The user requests clearer buttons and a video-player-style progress display.
Keep the centered primary Starten/Pause/Fortsetzen control, style Beenden as
a bordered button, and group the phase, progress bar and elapsed/total clocks
below playback. The timeline is visible before starting and remains visible
after stopping or finishing. It follows active Web Audio time and freezes on
pause; replay resets it. The Anleitung example shows its real 68-second audio
duration even though the menu choice has no imposed exploration time limit.
Use a semantic read-only progress element; seeking would alter the authored
breathing protocol and is outside this change.

- [x] Implement audio-clock progress, elapsed/total time and clearer buttons.
- [x] Extend the controller regression to cover reset, pause, exact totals,
  completion and replay; all eight audio tests pass.
- [x] Verify 24 Chrome cases: seven viewports in ready/playing/paused states,
  plus 200% text, long German labels/text spacing, and missing Pretext.
  See `VERIFICATION.md` for evidence and device limits.

The overall headset/H10 product goal and listening review remain open.
Existing round artwork and viewport-centering work are preserved. This task
is committed on `main` under `Add playback buttons and audio-clock progress`;
its predecessor was `a79275f`, matching the locally recorded `origin/main`.
The original playback task left publication to the combined release above,
which now owns its current Git/deployment state. Screenshots and the 24-case
executable check remain ignored in `output/playwright/`. Real phone/Safari,
headset/H10 and listening acceptance remain separate open checks.

## Scenario audio protocol — 2026-11-05

Every final sequence must play predefined assets from the shared `assets/audio/`
library. Each scenario owns a JSON timeline and a matching CSV under
`audio/scenarios/`; JSON is the runtime authority and CSV is its checked,
readable timing sheet. Preserve the chosen runtime samples and existing exact
durations, silent main-round baseline, pause/resume and abort behavior. The
current browser player is the integration target; real video/headset execution
remains pending. No new voice generation or asset selection is authorized here.

- [x] Add all six menu scenarios plus the existing engine-only 20-minute
  diagnostic as JSON/CSV pairs; bind each to the shared library manifest.
- [x] Load and validate those files before playback; retain the single Web Audio
  clock and record scenario identity/hash in scheduling receipts.
- [x] Verify file/CSV consistency, invalid-input rejection, all existing audio
  behavior and native offline renders; update the audio workflow and evidence.

Implementation uses the existing plain JavaScript engine and Node checks with
no new dependencies. Work stays in the shared checkout to preserve the pending
UI/player edits already present in `app.js`, `index.html`, `style.css`,
`audio/session-ui.js` and `tests/box-breathing.test.mjs`. Initial Git observation:
`main` at `a79275f`, matching the locally recorded `origin/main`; no publication
of this change has been observed. Overall headset/H10 acceptance remains open.

Evidence: 13 Node audio tests, seven JSON/CSV consistency checks, fourteen native
offline renders (all seven scenarios at 44.1/48 kHz with zero sample error), all
six real-browser playback routes and missing-JSON/retry behavior passed. All
886 participant text-score ranges/scripts/clips match the new JSONs. Hash-bound
JSON/CSV uses pinned LF bytes so Windows and GitHub Pages agree. See
`VERIFICATION.md`. Concurrent tasks committed the existing player/artwork edits
as `2e58f6f` and `7c8cadc` during this work. This change is prepared as the local
commit `Load final-sequence audio from scenario JSON timelines` on `main`;
no push/merge/deployment is part of this task. Unrelated alignment-publication
notes remain separate working-tree changes. Next product action: listen/review
the chosen samples and implement the actual headset/video transport with a
shared timing anchor; acceptance requires target-device audiovisual evidence.

## Round-button infographic task — 2026-11-05

- [x] Generate two matching transparent infographics and convert them into
  actual SVG paths. Originals, SVGs and exact prompts/settings are retained
  under `assets/rounds/`.
- [x] Embed the SVGs above the existing German round labels, preserving the
  routes, accessible link names and real Pretext measurement.
- [x] Verify 13 rendered cases, both branch routes, keyboard activation and
  readable unavailable-measurement fallback. See `VERIFICATION.md`.

The final simplified practice graphic shows lungs with a repeat arrow. The main
graphic shows a clipboard with a simple three-point data chart, identifying the
research/data-collection role requested by the user. Both SVG backgrounds and
the clipboard interior are transparent. This replaces the first breathing-loop
artwork and supersedes the earlier text-only round-choice rule. The symbols do
not establish implemented research data collection; the overall headset/H10
product goal and pending audio listening review remain open.

Git observation: the shared checkout is on `main`. Concurrent UI alignment work
committed the two image references and their CSS in `e4f76d9` while this task
was tracing/verifying the assets; player/audio edits remain owned by the other
task. The initial assets were published in `a79275f` (Pages run `42044006553`
succeeded). The simplified revision is saved locally under
`Simplify round SVGs for practice and research`; its new PNGs, real path-only
SVGs, prompts and transparent-alpha checks are retained with the existing
13-case rendered QA. It was initially kept local alongside the separate player
and scenario changes. The combined release above now publishes and verifies
this simplified artwork, the player and scenario audio together. Preserve the
remaining product boundaries; the overall headset/H10 goal remains open.

## Current UI alignment task — 2026-11-05

The user requires symmetric, centered controls on the main Atmungsraum UI.
Keep headings and single footer actions on the horizontal midpoint. Center
the sensor-selection button, ECG, choice grids and primary playback button
against the viewport itself, with equal top/bottom space budgets. Starting,
pausing or showing the countdown must not displace the playback button.
Use the dynamic viewport height with the existing small-viewport fallback.
Enlarged text retains the deliberate document-reflow escape hatch.

- [x] Correct the shared shell, sensor action, footer and playback geometry.
- [x] Verify 79 rendered cases at seven ordinary viewport sizes, plus 18
  enlarged-text/spacing/fallback cases and the keyboard interaction sequence.
- [x] Publish and verify the exact UI files and centering on the live site.

Local Chromium checks, UI state, all eight audio tests, context and whitespace
checks passed; screenshots and geometry receipts are ignored in
`output/playwright/`. See `VERIFICATION.md`. Branch is `main`; the refreshed
remote sample-page changes were merged while retaining the earlier local audio
review notes. Publishing the navigation UI is covered by the existing approved
prototype deployment workflow. The overall headset/H10 product goal and
listening acceptance remain open.

Alignment feature `e4f76d9` is published as an ancestor of `a79275f`; Pages run
`42044006553` completed successfully. All three live entry files matched the
feature's committed bytes, and all 79 live Chromium layout cases passed.
Concurrent scenario/progress/image UI work is present in the shared checkout;
preserve it. The alignment request is complete. Any subsequent UI revision must
retain this geometry and rerun the centered-layout checks. The live verification
notes are updated locally alongside that concurrent work.

## Voice sample subpage — 2026-10-28

The user requested publishing the exact phrase “Einatmen … zwei … drei … vier.”
from all twenty newly saved Atmungsraum profiles at `/atmungsraum/samples/`.
This separate listening page uses ten LANG reference variants (two actors) and
ten EmoDB actors. LANG ordering uses constituent-source valence means; EmoDB
neutral-valence estimates are tied and source validation breaks the tie.
Neither source ratings nor exact-word ASR establish the synthesized delivery's
calmness. Samples preserve their original generated bytes and timing.

Scope: the new `samples/` directory, its publication metadata, and these owning
notes. Reuse the existing locked Pretext runtime and static GitHub Pages hosting.
The existing session scheduler, chosen ElevenLabs assets, and H10 UI keep their
current behavior. Original references and full machine-specific receipts stay
private. The user's request explicitly authorizes publishing these new auditions.

- [x] Generate and fully decode twenty German MP3s; verify their exact words.
- [x] Check the static players, downloads, text measurement and responsive reflow.
- [x] Publish to `main` and verify the public sample page and all MP3 URLs.

Publication verified on 2026-10-28: feature commit `1f3f5db` completed GitHub
Pages build/deployment run `40730446610`. The public page and all twenty MP3s
returned HTTP 200; every downloaded MP3 matched its recorded SHA-256 hash.
Live Chromium verified the twenty players, text measurements, keyboard
play/pause and MP3 download. URL: <https://viscereality.org/atmungsraum/samples/>.

- Purpose: provide a smartphone-compatible browser interface that controls actions
  on a VR headset screen through a corresponding Atmungsraum app.
- Intended user: someone operating the VR experience from a smartphone browser.
- Repository: <https://github.com/GeorgeFejer91/viscereality.org> (public).
- Default and publishing branch: `main`.
- Public URL: <https://viscereality.org/atmungsraum/>.
- Entrypoint: `atmungsraum/index.html`, relative to the repository root.
- Current website stack: static HTML, `style.css`, and plain JavaScript; local
  Pretext runtime pinned to `@chenglou/pretext` 0.0.9.
- Hosting: the existing GitHub Pages branch deployment from `main`, root `/`,
  with the existing `viscereality.org` custom domain.

## Current task: audio protocol review — 2026-10-27

- [x] Inventory the checked-in generation, audition, packaging and sequencing
  scripts, separating runtime and local listening revisions. Evidence:
  `audio-review/01-PROTOCOL-INVENTORY.txt`.
- [x] Provide participant-readable German speech and unspoken sound/silence
  notes in a separate review folder. Evidence: `audio-review/02-PARTICIPANT-EXPERIENCE.txt`
  and seven full text timelines, derived from production `createProtocol()`.
- [x] Verify source hashes, all nine runtime WAV hashes/byte counts, timeline
  continuity/totals/overlay counts, eight audio tests, UI state, context and
  whitespace. See `VERIFICATION.md`.
- [ ] Review wording and delivery by listening; decide whether to revise the
  welcome's counting promise, silent main-round opening or hold wording.

The current request is fulfilled as a local text review; the overall product
goal remains open. No speech, scheduler or UI behavior changed. The documented
`reference-audio/` cache and generated ElevenLabs listening package are absent
from this checkout; restoring them is required for audition comparison, not
for reading the current runtime timelines. No fresh listening or hardware
verification is claimed. Next action: collect timestamped review notes using
the template in `audio-review/00-START-HERE.txt` before changing audio assets.

Git observation at task start: canonical repository `GeorgeFejer91/viscereality.org`,
primary checkout on `main`, HEAD `b365411`, clean and aligned with the locally
recorded `origin/main`. Task changes are limited to `audio-review/` and this
subproject's guidance. No remote ref refresh or live deployment check was made;
this review is not a publication request. Recheck Git state on next entry.
Local review work is recorded under the commit subject
`docs(atmungsraum): inventory audio protocols and participant review`.
No push or deployment is part of this task; generated audio remains untouched.

## Intended product contract — 2026-10-12

The user approved this prototype for publication on 2026-10-12 as the main
navigation logic for the UI controlling the VR setup. It remains a prototype.
The bottom-level Start action is intended to play an audio sequence and initiate
a corresponding sequence of VR events. Browser audio already works; the VR event
set, command transport, timing, and synchronization still need to be defined.

The website is the smartphone-accessible control surface for the VR headset app.
The planned app should mirror the website's UI, with corresponding screens,
controls, labels, and actions. Adapt layout to the phone and headset while keeping
the meaning of those controls consistent.

Actions initiated in the phone browser should trigger the corresponding actions
on the VR headset screen. The intended interaction runs from phone browser to
headset app; controls must ultimately affect the headset experience, not only
the browser's local UI.

UI mirroring refers to corresponding interface elements. Whether live screen or
video streaming is needed is Undecided. Do not infer a breathing exercise or a
specific action set from the page name.

## Patient entry and Polar H10 verification

The first page state collects patient details and shows the generated participant
ID. Weiter then opens the Polar connection gate. The gate precedes the round/session
menu; session actions must not appear ready before details and ECG are verified.

When the H10 is connected, display its live ECG waveform in a box in the middle
of the interface. Use actual ECG samples from that sensor. A selected device,
heart-rate reading, or simulated waveform does not establish a working ECG feed.
Represent connecting, disconnected, and connected-but-no-ECG states accurately;
do not leave a stale trace presented as live after the connection or data stops.

The intended browser checks the headset and receives ECG from it. The H10
acquisition path within the headset app and the transport remain Undecided.
Choose the transport against the actual supported smartphones and browsers. The
verified ECG view enables a Weiter button to the round menu.

## Menu layout contract — 2026-10-12

Use the installed `uncodixfy-pretext` skill and its outside-in stretch-accordion
layout for every UI function in this menu, including the H10 gate and ECG view.
Allocate the outer panel first, retain header and action/footer anchors, preserve
ordered groups and label/control pairs, and distribute extra width and height
between groups and cells. Use shrinkable Grid/Flex tracks with `min-width: 0` and
`min-height: 0`; derive spacing and preferred type from both dimensions.

The bounded menu panel has no internal scrollbar. Reflow controls or revise the
supported minimum when they do not fit. Preserve readable type, touch targets,
keyboard order, visible focus, and user-enlarged text. This stretch geometry does
not require collapsible sections. Use actual locked Pretext measurements for all
bounded labels and status text, with font parity and a readable unavailable/no-fit
fallback. Verify the rendered layout; CSS controls geometry and Pretext measures
text. See [SKILLS.md](SKILLS.md) and [VERIFICATION.md](VERIFICATION.md).

## Round menu and UI prototype — 2026-10-12

Use easy, simple German throughout the interface. The browser prototype implements
this browser-local decision tree:

1. Emulate the headset searching for Polar, finding it, selecting/connecting it,
   then waiting for ECG. One compact status line shows the current step, with
   Polar auswählen available when found. The ECG box appears only during
   recording. Weiter is red/disabled until recording, then green/enabled.
   Erneut suchen appears only when disconnected or waiting for ECG, where retry
   aids recovery. The status and canvas identify the demo.
   Patient entry precedes this step. Every started run follows DATA-LOGGING.md;
   CSV-Dateien provides local download recovery. The corner ambient signal is
   visible across screens without replacing the connection gate.
2. Show "Übung wählen" with Probe-Runde and Haupt-Runde together. Use the two
   generated SVG infographics above HTML labels, soft dark surfaces, muted sage
   and cream, gentle corners, and
   restrained borders. Choices are square where the measured labels fit.
3. Show the clickable boxes for the chosen round:
   - Probe-Runde: Anleitung (guided free exploration, no time limit), 5, 10,
     or 15 Minuten. Anleitung and Frei ausprobieren are the same option.
   - Haupt-Runde: direct entry to one 15-minute sequence. Instruction 1,
     Calibration, Instruction 2, fifty complete breathing cycles and Outro
     together last exactly 900 seconds. There is no main duration menu/baseline.
4. Every box goes directly to a final screen with a **Starten** play
   button. The selected round/option is shown for orientation. Starting is
   browser-local; the separately integrated audio controller owns playback.

There are no intermediate radio confirmations or separate instruction pages.
The initial final screen shows the play action. During playback that same button
becomes Pause, or Fortsetzen while paused, and the back link becomes Beenden.
The phase and elapsed/total progress remain visible in every player state.
Remove redundant descriptors, decoration,
duplicate buttons, and routine ready/playback messages; keep the choice timing
breakdown and actionable loading/completion/error feedback. Real headset session
execution remains pending.
Hash navigation supports Zurück and browser history; cold deep links still pass
through the connection gate. After Weiter there is no ECG return button. Signal
loss after this verified transition keeps the current screen and existing audio
while requesting reconnection every five seconds. Return to ECG only after more
than 30 seconds without recovery and at least one retry; clear the old waveform
and lock Weiter. Recovery before expiry cancels the deadline and retry count.
New sessions cannot start without current ECG. JavaScript is required for the demo; without it the
patient form stays disabled with an explanatory message and cannot submit
participant values into a URL.

Runtime assets remain local; Pretext 0.0.9 measures bounded text with the painted
Arial font. The header and footer anchor a bounded panel, and the middle controls
use shrinkable container-sized tracks. At enlarged text/no-fit the page explicitly
reflows into document flow, with no inner scrollbar and no font shrinking.

The actual product should receive verified connection state and live H10 ECG
from the headset. This UI demo implements neither real sensor acquisition nor
browser-to-headset transport. Do not present its synthetic waveform or local
playback as hardware verification or a command delivered to a headset.

The future headset transport can report `atmungsraum:sensor-state` CustomEvents
with `{ state: 'ecg' | 'no-ecg' | 'disconnected' }`. The browser emits
`atmungsraum:reconnect-request` with the retry number. These hooks implement the
UI's recovery behavior; an actual transport and verified ECG sample source are
still required. Do not mistake events or synthetic traces for hardware validation.

## German audio cues — 2026-10-12

Five separate German speech cues are saved in `assets/audio/breathwork/de/`:
Ankommen, Einatmen, Ausatmen, natürlich atmen, and Abschluss. Their exact text,
filenames, generation settings, durations, and SHA-256 hashes live in `cues.json`
beside the MP3 files. ElevenLabs returned the saved voice named
`British Breathing Instructor` (ID `rt9KyyQmwqa1uhHGJkMa`); use this voice for
these Atmungsraum cues. The generated set uses `eleven_multilingual_v2`, a speed
of 0.85, and 44.1 kHz / 128 kbps MP3 output.

These are local audio assets. Playback, breathing-cycle timing, and headset
integration are not implemented by their generation. All files passed full MP3
decode and integrity checks; German pronunciation and subjective voice review
remain pending.

## Reusable box-breathing engine — 2026-10-12

`audio/box-breathing.js` loads twelve reusable German PCM WAV samples from
`assets/audio/box-breathing/de/soft-v2/manifest.json`. The phase samples are exactly
four seconds at 44.1 kHz. Repeat the four separate files `inhale.wav`, `hold_in.wav`,
`exhale.wav`, and `hold_out.wav` in that order. The instruction and ending samples
play once (40 and 12 seconds). The two hold files are independent roles with
identical Und halten text followed by quiet. Inhale/exhale say Einatmen/Ausatmen
with counts two/three/four. Welcome, ending, and rest use du. Delivery is softer
and slower; the exact phase wording follows the user's latest instruction.
Do not generate or concatenate new speech for each cycle.

The user selected exact 5/10/15/20-minute total durations including welcome,
ending, and rests. Complete 16-second cycles form blocks of at most 18 cycles
(288 seconds, the nearest complete-cycle duration below five minutes). A 20-second
natural-breathing rest separates blocks when another full cycle fits. Session
cycle counts are 15/33/50/68, with 0/1/2/3 rests and 8/0/8/0 seconds quiet recovery.
Every third complete cycle overlays both inhale and exhale with the supplied
breath sounds. Use the Web Audio clock and preload
all samples; JavaScript timers and ended callbacks must not trigger phase audio.
Haupt-Runde is the single 900-second class-based scenario described above.
Instruction 1 prepares two comfortable deep breaths; the separate silent
Calibration window runs at t0 + 26 to t0 + 41 audio-clock seconds. Instruction 2
then explains paced breathing, which begins at t0 + 81. Fifty complete cycles
end at t0 + 881; Outro lasts 19 seconds and ends at t0 + 900. Main has no baseline
or between-block rest. Pause freezes this clock and resumes the same nodes;
receipts preserve planned calibration timing with its measurement still unset.
Hidden pages, interrupted audio clocks, failed preparation, or missed start
deadlines invalidate/abort an attempt. Host-side saving of receipts and physical
audio/sensor synchronization remain integration work. The browser controller
saves the last ten participant-linked scheduling receipts locally; the CSV
writer retains each exported run in IndexedDB. These are not acoustic onset
measurements or an integrated study data store. See DATA-LOGGING.md for identity,
clock, mock-source and persistence limits.

The original German/English reference audio and native-caption transcripts are
downloaded under ignored `reference-audio/`. Newly synthesized German samples
use British Breathing Instructor. See [audio/README.md](../audio/README.md) for
the source analysis, scripts, timing contract, API, verification, and limits.
The round menu's final Play button is wired to browser playback through
`audio/session-ui.js`; Stop/Escape/navigation and interrupted clocks abort the run.
The main button changes between Starten, Pause, and Fortsetzen. Pause deliberately
suspends the Web Audio context and freezes the countdown; Fortsetzen resumes the
existing schedule from the same position. Beenden returns to the choice menu and
aborts the run; Escape stops in place. No separate Stop or Start/Pause row is shown.
Timed totals count active audio-clock time, so deliberate pauses extend wall-clock
duration. Pause/resume records are included in the local scheduling receipt.
No H10 or headset behavior is established by the audio checks.

Alternative voice sourcing is local audition work: two neutral, two positive,
and two restrained EmoDB profiles are saved in Secret Tunnel. The user rejected
the neutral takes as robotic and the positive takes as too energetic. Restrained
16-second counted cycles await listening. Archive CC0 and publisher CC-BY-4.0
metadata conflict; preserve attribution. LANG norms contain no low-arousal/high-positive-valence
match at the current thresholds. OmniVoice's pretrained model is CC-BY-NC and
speaker clone consent is not established by corpus metadata. Do not replace or
publish runtime samples as unrestricted clinic voices on this evidence. See
[voice-sourcing.md](../audio/voice-sourcing.md).

## Current implementation

The initial 2026-10-09 page was a blank black placeholder. The working tree now
contains the functional German UI prototype described above, including synthetic
ECG gating, four practice/direct main choice, and final play-button routes. Real H10 data,
headset connection, and headset session execution remain unimplemented. Browser
audio is wired separately through audio/session-ui.js. Keep the black visual
foundation and preserve work in progress.

The AI guidance lives at `atmungsraum/for-ai/`; it is not page content. This
subproject shares the parent Git repository and deployment, with no nested repo.

## Undecided

- H10 connection owner, transport, ECG acquisition, and supported phone browsers.
- Hardware verification of the headset-delivered ECG gate.
- Headset implementation of the selected session start and unlimited Anleitung.
- Headset platform, app stack, and app source location.
- Browser-to-headset transport, hosting requirements, and supported networks.
- Device discovery, pairing, access control, and headset-state feedback.
- Whether live screen or video streaming is part of the experience.
- The VR event sequence initiated by Start, including audio/event timing and
  synchronization between the browser and headset.
