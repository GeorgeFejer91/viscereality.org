# VR UI testing with Meta XR Operator

Meta XR Operator is an experimental OpenXR API layer. Its MCP proxy lets an
agent inspect a running XR session and drive supported input. Use it in a
development build with synthetic data. Record the Operator version and discover
the tools available in the live session each time. [Meta XR Operator overview](https://developers.meta.com/vr/documentation/unity/meta-xr-operator/)

## Choose the target

| Target | Test route |
| --- | --- |
| Native Atmungsraum rehearsal | Qualify Meta's standalone OpenXR layer in a dedicated Quest 2 test APK built from the [native rehearsal recipe](../quest/README.md). Use Meta XR Simulator first when a compatible desktop build exists. |
| Unity OpenXR test scene | Use Unity 6000.0.x or later, Meta XR Core SDK v207 or later, and OpenXR Plugin 1.17.0 or later. Enable OpenXR and configure the layer and proxy through AI Tools and Project Setup Tool. Use Simulator Play mode for iteration and a Development Build for Quest. [Unity setup](https://developers.meta.com/vr/documentation/unity/meta-xr-operator/getting-started/) |
| Phone PWA and Quest 2 release | Drive VR-Probe from the installed Android PWA and follow the [physical acceptance checklist](PROJECT.md#quest-2-control-media-and-ui-acceptance--2026-11-12). The phone issues operator requests and Quest owns the scene, clock, H10 and recordings. XR Operator can observe the headset side during a test. |

## Connect to a live session

1. **Standalone desktop.** In the launch shell, set `XR_API_LAYER_PATH` to
   the standalone bundle's platform directory and `XR_ENABLE_API_LAYERS` to
   `XR_APILAYER_METAX_operator`. Launch the compatible OpenXR app from that
   shell. Register the bundled `meta-xr-operator-mcp-proxy` as a local stdio
   MCP server using its absolute path. Use a fixed starting pose and scene
   for repeatable Simulator runs. [Standalone setup](https://developers.meta.com/vr/documentation/unity/meta-xr-operator/connecting-ai-agents/) ·
   [Simulator workflow](https://developers.meta.com/vr/documentation/unity/meta-xr-operator/xr-simulator/)
2. **Native Quest 2 test APK.** Package Meta's Android ARM64 Operator
   library under `lib/arm64-v8a/` and its layer manifest under
   `assets/openxr/1/api_layers/implicit.d/`. Verify both entries and the
   intended OpenXR loader in the signed test APK. The MCP server needs
   `android.permission.INTERNET`. For full tools on Horizon OS v205 or
   later, declare `com.oculus.experimental.enabled` as an optional manifest
   feature and set `debug.oculus.experimentalEnabled` to `1` on the selected
   headset for the run. Read the property's prior value and verify the change.
   [Standalone Quest setup](https://developers.meta.com/vr/documentation/unity/meta-xr-operator/connecting-ai-agents/) ·
   [Quest platform requirements](https://developers.meta.com/vr/documentation/unity/meta-xr-operator/quest/)
3. **Quest 2 transport and capture.** Follow the [awake runbook](QUEST-AWAKE-TESTING.md)
   for device selection, install and launch ownership, display state, and
   protected prompts. Use its selected awake provider throughout the run.
   With the exact serial in `$selector`, inspect existing forwards and bind a
   free host port without replacing an existing mapping. Close your desktop XR
   session first if it owns port `8720`. Connect the host MCP proxy to the
   forwarded server. Capture uses MediaProjection and needs in-headset consent.
   If capture reports permission pending, wait for the wearer to approve it.
   [Meta's Quest setup and capture guide](https://developers.meta.com/vr/documentation/unity/meta-xr-operator/quest/) ·
   [ADB forwarding syntax](https://android.googlesource.com/platform/packages/modules/adb/+/refs/heads/main/docs/user/adb.1.md)

   ```powershell
   adb forward --list
   adb -s $selector forward --no-rebind tcp:8720 tcp:8720
   ```

4. **Readiness.** Confirm the target app is foregrounded, its OpenXR session
   is `FOCUSED`, and a fresh frame or frame counter advances. Query the live
   session through MCP to confirm the proxy is reaching the intended app.
   If the proxy lists tools but returns no live state, recheck the selected
   build, active XR runtime, focus and forwarding before sending input.
   [Meta's connection check](https://developers.meta.com/vr/documentation/unity/meta-xr-operator/getting-started/)

## Exercise the UI and record evidence

1. Write the starting scene, control, input modality and expected visible
   outcome. Inspect a fresh capture and, when the Unity scene tools are
   available, the matching canvas or control state. For the native rehearsal,
   use the frame and the app's state or receipts.
2. For a wearer-facing control enabled by the test build, use its normal
   input path. Aim the supported
   controller or hand at the target, check hover or focus, then press and
   release or pinch and release. Check the resulting frame and the app state
   or receipt for exactly one transition. Repeat for the affected focus,
   disabled or recovery state where applicable. Use Simulator head-pose
   overrides for repeatable positioning. On Quest 2, ask the wearer to move
   their head and use the actual controller or hands when qualifying physical
   bindings and comfort. [Simulator inputs](https://developers.meta.com/vr/documentation/unity/meta-xr-operator/xr-simulator/) ·
   [Quest pose limits](https://developers.meta.com/vr/documentation/unity/meta-xr-operator/quest/)
3. For a phone-driven change such as recenter, send the action from VR-Probe.
   Compare the pre/post Quest frame and the recenter receipt for that request.
   For playback changes, pair phone and headset captures with the same native
   run generation and player position receipts. Use the existing H10,
   recording and audio checks for their own acceptance.
4. Record build and runtime versions, test target, initial state, action,
   expected and observed results, fresh frames or receipts, and cleanup. Label
   each result as simulated, injected on Quest 2, or performed by a wearer.
   Record unavailable tools or missing evidence as blocked. Keep raw
   captures, passthrough images, logs, serials and synthetic run files private.

## Finish the run

Release any held inputs while the session is live. End the test app and
capture session. If this run created the mapping, remove it with
`adb -s $selector forward --remove tcp:8720` and verify its absence. Restore
any temporary XR Operator device properties you changed and read them back.
Use the [awake runbook cleanup](QUEST-AWAKE-TESTING.md#recover-and-finish)
for power and proximity. Use the regular rehearsal build for final acceptance.
