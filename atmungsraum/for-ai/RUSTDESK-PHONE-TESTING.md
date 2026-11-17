# Smartphone UI validation with RustDeskMCP

Use this runbook for the installed Atmungsraum PWA on the lab Android phone.
RustDeskMCP supplies physical phone frames and taps. The
[smartphone GUI release gate](VERIFICATION.md#smartphone-gui-release-gate)
supplies repeatable Pretext and painted DOM measurements. The authenticated
Quest status and receipts supply native behavior evidence.

## Prepare and connect

1. Record the Git commit, exact Pages deployment and live PWA release, then
   verify the installed standalone PWA loaded that release. Bind Quest actions
   to the selected headset and its installed APK. Record Android, Chrome and
   RustDesk versions. Check the actual device before each new session.
2. On Android, open RustDesk's **Share screen** tab. Keep **Screen capture**
   active and **Input control** enabled, with **RustDesk Input** bound in
   Android Accessibility. For operator-owned permanent authentication, use
   **Share screen → ⋮ → Set permanent password**. The operator enters and
   confirms it on the phone. Use the operator's connection approval or a
   locally protected credential at runtime. Keep credential entry out of
   screen capture, transcripts, command history, and evidence.
3. Open **RustDeskMCP** from the Windows Start menu. Use **Settings → Security
   → Enable MCP server** and confirm the endpoint at
   `http://127.0.0.1:59940/mcp`. Keep it on loopback. For the existing lab USB
   route, confirm **Direct IP access** and the Android listener in RustDesk,
   select the exact USB phone, list existing forwards, then reserve an unused
   loopback port with `adb -s <phone> forward --no-rebind tcp:0 tcp:<rustdesk-port>`.
   Record the returned port in private evidence. Preserve other operators'
   mappings and sessions.
4. Follow the [MCP tool reference](skills/rustdesk-mcp-operator/references/mcp-workflows.md).
   Send JSON-RPC `initialize`, inspect `tools/list`, and call
   `open_desktop_session` with `127.0.0.1:<returned-port>`. Use its `session`
   handle for subsequent calls. Resolve authenticating or waiting states and
   select the intended display if needed. A fresh `get_desktop_frame` matching
   the Android screen establishes capture. Connection or password
   acknowledgement alone leaves capture unverified.
5. Confirm input with a reversible PWA action. Open **Einstellungen**, check
   the rendered heading, and return to the original page. At narrow widths,
   open the grey actions menu first. For a series of taps, coordinate one
   driver, lock local RustDeskMCP viewer input for that session while driving,
   and release the lock before returning control.

## Use the lab credential file

The handset owner picks the permanent password. Store the path to their existing
password file in `%LOCALAPPDATA%\Atmungsraum-Validation\rustdesk-password-file.txt`
on the Windows host. The locator contains one absolute path. Enter the password
on the handset first. For a new MCP connection that reports `needsPassword`, use
the locator to supply the saved credential directly to `input_password`.

```powershell
$rustDeskLocator = Join-Path $env:LOCALAPPDATA 'Atmungsraum-Validation/rustdesk-password-file.txt'
$rustDeskPasswordFile = [IO.File]::ReadAllText($rustDeskLocator).TrimEnd([char]13, [char]10)
$rustDeskPassword = [IO.File]::ReadAllText($rustDeskPasswordFile).TrimEnd([char]13, [char]10)
if (!$rustDeskPassword -or $rustDeskPassword.Contains("`n") -or $rustDeskPassword.Contains("`r")) {
    throw 'Expected a single nonempty RustDesk password line.'
}
try {
    $rustDeskBody = @{
        jsonrpc = '2.0'
        id = 3
        method = 'tools/call'
        params = @{name = 'input_password'; arguments = @{session = $session; password = $rustDeskPassword}}
    } | ConvertTo-Json -Depth 8 -Compress
    $rustDeskAuth = Invoke-RestMethod -Uri 'http://127.0.0.1:59940/mcp' -Method Post `
        -ContentType 'application/json' -Body $rustDeskBody -TimeoutSec 20
    if ($rustDeskAuth.result.structuredContent.connected -ne $true) {
        throw 'RustDeskMCP did not confirm authentication.'
    }
} finally {
    Remove-Variable rustDeskPassword, rustDeskBody -ErrorAction SilentlyContinue
}
```

Set `$session` to the handle returned by `open_desktop_session`. Keep the
password file and the machine-specific locator outside the checkout. Keep
request bodies out of captured logs. After authentication, prove capture and
input using the checks above.

## Run and evaluate cases

- Record a case ID, installed release, phone/Android configuration, driver,
  orientation and viewport, initial screen, visible target, expected result,
  and any expected Quest request. Fetch the current frame, identify the
  interactive hit area, send one gesture by MCP, then fetch a new frame and
  check the result. Recalculate coordinates after rotation, keyboard changes,
  scrolling of OS surfaces, or page transitions. Use raw remote frame
  coordinates for `mouse_click`. Use `keyboard_input` only after verifying
  the intended field is focused.
- Exercise the installed app's launcher, safe areas, keyboard, rotation, and
  connection, participant, input, tracking, battery, session, recording,
  settings, update, and recovery views. Before each mutation, check eligible
  native state. Assert one intended applied request and resulting native state,
  or zero native mutations for local navigation. For a missing receipt, read the
  original request state before retrying. Use synthetic participant details.
- Apply the current [UI rules](UI-LAYOUT.md#smartphone-gui-rules) to every view.
  Check visible labels and focus, 48 px hit targets, the single toolbar row
  and 389/390 px actions breakpoint, and fit in the usable viewport with the
  keyboard. The release gate verifies real Pretext, full painted text, zero
  document and nested scroll range on app-authored surfaces, and the viewport,
  text enlargement and spacing matrix. Capture the actual phone to verify
  physical safe areas, native dialogs, keyboard and hit targets.
- If a frame is unavailable, check projection, session authentication, and
  display selection. If clicks report dispatch but the screen does not change,
  confirm the actual Android input service is bound and release any
  UIAutomator helper that owns the phone. Keep the helper and RustDesk input
  checks sequential. If a floating RustDesk overlay blocks an Android or
  browser consent prompt, pause the run-owned screen share, complete the
  attended prompt, then reconnect and capture a new frame. Record the actual
  driver when direct touch or Mobile MCP is needed.
- Fix confirmed UI defects in the same validation workflow. Preserve the
  original failed case, add a targeted regression where practical, apply a
  scoped fix, run the GUI release gate and affected workflows, then repeat the
  same action through RustDeskMCP on the latest installed release. Keep the
  case open until the corrected flow and applicable native evidence pass.
  Use the [VR input runbook](VR-UI-TESTING.md) for actual Touch and immersive
  behavior on the Quest release.

## Record evidence and release control

Retain redacted before/after evidence, case outcomes, and the associated
receipt where applicable. Store raw captures, logs, device selectors,
addresses, credentials, and recording data outside Git. Report frame capture,
a visible phone transition, and a phone-driven native receipt as separate
observations. Record results and the next action in [PROJECT.md](PROJECT.md).

Release the viewer input lock and close the run-owned RustDesk session.
Remove only the forward created by this run with
`adb -s <phone> forward --remove tcp:<returned-port>`, then verify its absence.
Restore bench settings changed temporarily for the run and leave the phone
available to the operator.
