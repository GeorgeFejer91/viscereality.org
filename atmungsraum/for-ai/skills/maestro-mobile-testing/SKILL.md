---
name: maestro-mobile-testing
description: Author and run repeatable Maestro UI tests on Android and iOS, including a USB-connected Android phone. Use for smoke tests, regression flows, and pass/fail evidence after app builds.
---

# Maestro mobile testing

Use [Maestro](https://github.com/mobile-dev-inc/maestro) for flows with explicit assertions. Use `android-cli` or `mobile-automation` for exploratory device work before turning a stable flow into a regression.

1. Run `maestro --version` and `adb devices -l` on Android. Choose the exact authorized device serial and confirm the app package is installed on that device. For an installed Android PWA, identify its actual launch package from the phone. For a browser-tab test, target the browser package and start from an explicit URL.
2. Use the existing flow directory if one exists. Put the installed package in the YAML `appId` and give each flow a known starting state. Prefer stable accessibility labels or resource IDs. Add `assertVisible` or `assertNotVisible` for the outcome that matters. Use synthetic data. Omit state clearing on a personal or shared phone unless resetting that app is part of the requested test.
3. Run the flow on the selected phone. Keep screenshots, reports, and failure logs in a local artifact directory outside public or tracked paths.

   ```text
   maestro --device <serial> test --format junit --output <report.xml> --debug-output <local-dir> <flow.yaml>
   ```

4. Report the device, build or URL, command, exit status, passed or failed assertions, and evidence paths. Verify the foreground app or PWA before drawing conclusions from failures. A scripted assertion only covers the named outcome. Check hardware-dependent behavior with the actual hardware.

See [Windows setup](https://docs.maestro.dev/maestro-cli/how-to-install-maestro-cli), [Android physical devices](https://docs.maestro.dev/get-started/supported-platform/android), and [targeting a device](https://docs.maestro.dev/maestro-flows/flow-control-and-logic/specify-and-start-devices).
