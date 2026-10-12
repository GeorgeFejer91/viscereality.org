# Browser playback completion audit — 2026-10-12

Scope: actual local browser playback from the final Play screen for selected
5/10/15/20-minute sessions. The final source uses four separate reusable German
four-second phase WAVs, a welcome/ending once, quiet holds, 20-second rests,
third-cycle inhale/exhale breath variants, and exact total durations.

| Requirement | Current evidence | Result |
| --- | --- | --- |
| All four durations reach the final Play action and start browser audio | Actual main-round menu actions observed in the in-app Chromium browser; DOM audio clock spans were 300/600/900/1200 seconds; `app.js` mounts `audio/session-ui.js` | Passed |
| Four separate four-second phase assets repeat | Current manifest/PCM headers, SHA-256/byte checks and Node tests; `scheduleProtocol()` assigns the predecoded phase buffers to every cycle event | Passed |
| Requested wording and pauses | Inhale/exhale say Einatmen/Ausatmen plus counts; each independent hold says Und halten followed by silence; PCM hold tails are silent after two seconds | Passed technically; subjective delivery remains reviewable |
| Welcome and ending play once | Source protocol adds one instruction and one ending; native full renders compare every sample against those events | Passed |
| Exact 5/10/15/20-minute totals | Native OfflineAudioContext output lengths: 13,230,000 / 26,460,000 / 39,690,000 / 52,920,000 frames at 44.1 kHz; all maximum sample differences zero, after the quiet-hold update | Passed |
| Resampled playback | Five-minute native render at 48 kHz: 14,400,000 frames, zero maximum sample difference | Passed |
| Complete breathing cycles and rests | Tested counts 15/33/50/68; rests 0/1/2/3; recovery 8/0/8/0 seconds; blocks ≤18 complete cycles; rests 20 seconds; third cycles use both breath variants | Passed |
| Stable cue scheduling | All sources queued against one AudioContext anchor; each start/stop derives from integer source frames. Audio phase changes never depend on timers, RAF or ended callbacks | Passed by source, scheduling tests, and native sample comparison |
| Play, Stop, Escape and back navigation | Actual browser starts/stops observed; 20-minute double click produced one active run, disabled Play and enabled Beenden; Stop/Escape restored start availability; back navigation cancelled playback | Passed |
| Native completion and restart | A complete 68-second instruction example reached `completed`, Fertig, hidden unlimited-mode clock and an enabled replay action in the real browser | Passed |
| Interrupted clocks and cancelled starts | Engine tests prove suspension abort, corruption rejection, stale-callback protection and deadline guards. Controller regression proves Stop during unlock cannot attach a previous receipt to a new scenario | Passed |
| Correct phase/count/time status | Actual 20-minute run progressed through repeated phases on the shared audio clock; e.g. cycle 14 hold-out at 15:37 remaining matches the 40-second welcome and elapsed protocol position | Passed |
| Narrow preview usability | Playing final screen observed at 320×568 CSS px: measured text fit, document exactly 320×568, no overflow, readable phase/time/Beenden; screenshot visually inspected | Passed for this viewport; broader device/accessibility qualification separate |
| Ready preview available | Current source reloaded, sample integrity preparation succeeded, current 20-minute start/Stop observed, and final 20-minute screen restored to Audio bereit with Play enabled | Passed |

`pnpm run test:audio` passes six tests, `pnpm run check:ui` passes, the local
context checker passes, and `git diff --check` passes. No publishing is implied.
The final ready-screen screenshot is saved locally as
`reference-audio/playback-proof/ready-320.jpg` (320×568 JPEG, ignored).
The latest UI retains the current 30-second sensor-reconnection grace: a transient
loss disables new starts while retaining the current flow; expiry returns to the
gate and aborts the run. Audio-clock or hidden-page interruption still aborts.

The audio goal establishes browser scheduling, not measured acoustic output,
clinical suitability, real Polar H10 acquisition, or headset synchronization.
Alternative OmniVoice profiles remain local auditions because their model terms
and speaker permissions do not establish unrestricted clinical deployment.
See [voice sourcing](../audio/voice-sourcing.md). The browser currently uses the
saved ElevenLabs instructor voice with the requested phase wording.
