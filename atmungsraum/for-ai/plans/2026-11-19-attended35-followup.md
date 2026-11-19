# Atmungsraum testing handoff — 19 November 2026

## Which build to test

| Surface | Baseline |
| --- | --- |
| Private Meta Developer channel | Code 33, version 0.2.32-dev. The website requires at least code 27. Check the code actually installed before testing. |
| Public phone PWA | 20261119-polar-pause at [Atmungsraum](https://viscereality.org/atmungsraum/). Refresh or reopen the installed PWA before testing. |
| Lab candidate used in the completed run | Developer 35 r2, version 0.2.34-dev. The reproduction source is merged in [PR 167](https://github.com/GeorgeFejer91/viscereality.org/pull/167). The signed candidate is a lab artifact and has yet to qualify for Meta distribution. |

Start with the installed Meta build for an independent baseline. If the test
requires Developer 35 r2, use the [assembly and provenance notes](../../quest/apks/README.md#developer35-main-signal-recovery-candidate--2026-11-19)
or arrange access to the signed lab candidate through the maintainers' private
channel. The Meta build and the lab candidate have different behavior. Preserve
the installed app's data and trusted phone pairing when changing versions.

For another headset and phone, follow the [device setup guide](../COLLABORATOR-QUEST-SETUP.md)
and [current release policy](https://viscereality.org/atmungsraum/quest-release.json).
Use your own paired phone and headset, a test participant ID, and a Polar
strap free of other raw-motion clients.

## What the attended run established

On Developer 35 r2, the personal Tilt fit completed, the main session ran the
full 900 seconds, the Quest reached the native audio completion threshold, and
the phone downloaded the saved CSV. The wearer agreed to proceed for the
continuity test and found the tracking unsatisfactory.

The raw ECG audit found out-of-order samples and missing interior samples even
though the writer reported zero drops. This is open recording-integrity work.
The phone briefly returned to connection recovery after automatic completion,
then reconnected. The next-session action reused the previous participant and
showed a stopped calibration preview. The next Arm was rejected even as public
status showed Idle. A separate attempt after an idle app restart also failed.
Start readiness sometimes flickered. The second run and the physical
five-second signal-loss recovery remain unverified.

## What we are working on

- **Quest recording and lifecycle.** Local host tests reproduce and repair raw
  ECG retry ordering, silent clock-contention loss, and the stale command
  coordinator that rejected the second Arm. A separate host reproduction
  shows that failed recording preparation can retire a valid calibration.
  The initiating error in the attended cold Arm and the post-completion
  connection interruption remain under investigation. These changes still
  need a signed integrated build and physical tests.
- **Phone new-participant flow.** A local PWA change clears the previous
  participant and follows participant details, breathing method/calibration,
  tracking check, batteries, and Start. The rendered smartphone gate passed.
  This change awaits Pixel validation and deployment. Start readiness feedback
  is still being investigated.
- **Polar calibration.** The local browser tuner keeps a model fitted on two
  guided cycles and replays it on separate recordings. Timestamp-based
  derivative smoothing and a wider hold threshold improved agreement with the
  cues in offline tests, with empty holds still variable. The installed
  personal Tilt route uses zero guidance bias. We compare the older Quest bias
  formula only as a separate replay because matching a cue-informed classifier
  against that same cue can overstate tracking. Headset acceptance requires
  a fresh wearer check across inhale, both holds and exhale.

## Suggested independent test

1. Record the headset model, installed APK code, PWA version and breathing
   method. Confirm phone control and live ECG and motion before preparation.
2. Fit personal Polar Tilt on two guided cycles. In the subsequent preview,
   check whether the sphere rises during inhale, settles during the full hold,
   falls during exhale and stays low during the empty hold. Record which
   transitions fail, including any noticeable delay. Controller State remains
   available as a fallback.
3. Run a full main and verify that it reaches completion and can be saved.
   If the phone briefly loses connection, record when and where it returns.
   Start the next participant using the normal flow, check that the form is
   blank, and report how far the second run gets. Note any disabled or flickering
   Start state and the on-screen reason.
4. Test signal-loss pause and manual Continue separately on an eligible build.
   Use the five-second policy only when the artifact is known to have it.
   Note whether calibration and the current recording survive the pause.

Share timestamps, versions, user-visible symptoms and sanitized logs through the
private project channel. Keep raw physiological samples and participant exports
out of GitHub. Drive setup is outside this checkpoint.
