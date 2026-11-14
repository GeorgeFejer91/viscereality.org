# Atmungsraum smartphone screenshot protocol

Capture the phone operator's screens for review. The script renders the
checked-out app in mobile Chromium at 390 × 844 CSS pixels with a 2× device
scale, German locale, and Europe/Berlin time zone. It uses fresh browser
contexts and synthetic Quest, ECG, participant, and recording fixtures.
The browser renders the real app HTML, CSS, and controller logic. All requests
are intercepted, including the simulated Google sign-in library. The capture
does not connect to a phone, headset, Polar sensor, or Google account.

## Create the review pack

From `atmungsraum/`, with Python 3.10 or later:

```powershell
python -m pip install playwright==1.58.0
python -m playwright install chromium
python tools/capture-phone-screens.py --output output/playwright/phone-review-2026-11-14
```

Choose a new output directory for each run. The command stops if a required
screen is absent. The resulting ZIP contains the numbered PNGs, an offline
gallery, `feedback.csv`, this protocol, and checkout and browser versions in
`source.txt`. The repository ignores
`output/playwright/`.

## Review one screen at a time

Unzip the pack, open `index.html`, and click a numbered thumbnail to see
the full PNG. Long forms include their full scrollable content. Put comments
next to the corresponding numbers in `feedback.csv` or refer to the numbers
when sending feedback.

The pack covers Bluetooth connection and retry, participant entry, Polar setup,
native breathing setup, each participant scenario selection, playback,
recordings, Drive setup, and recovery. All cache-versioned gateway imports
share one fixture. Its owner has a 32-character hexadecimal device ID,
positive gateway/native epoch strings, bounded native observations, and closed
setup commands with matching confirmed, rejected or unknown receipts.

The breathing captures show Automatic, waiting for the next native inhale,
collecting, one of two completed cycles, Polar ready, a failed attempt with
retry, and Controller State with fixed orientation. Explicit native frame
positions supply the four-second phase schedule and 32-second collection
progress. Advancing the browser clock alone must leave the phase and radius
unchanged. The ready example deliberately shows an intended hold alongside
measured exhalation and 23% input to distinguish instructions from sensing.
Stale Polar data must permit controller fallback while keeping Continue gated.

Recovery captures reload a journaled operation with a blank participant ID,
resolve it from its receipt without resending, and explicitly acknowledge
renewed gateway epochs, replaced native owners and replaced audio generations.
The recovery action must be reachable before participant entry, restore the
enabled participant form and leave the command journal empty. An older APK
must show the update requirement and disable setup actions.

The matrix also captures 320 × 568 reflow, 200% text at that width, WCAG-style
text spacing, long native failure reasons and 1440 × 900 desktop. Keyboard
checks start calibration with Tab/Enter, activate controller fallback and
recovery with Enter, and verify focus after Continue and acknowledgement.
Every capture checks page identity, one meaningful visible screen, actual
Pretext/DOM measurement, painted text bounds, horizontal overflow, 48px
controls, internal panel overflow and uncaught app errors. Vertical document
reflow is allowed; internal scrolling and clipped labels fail. `checks.json`
records these results and any Pretext no-fit indications separately from
actual painted overflow. Closed details are excluded from painted checks.

Main and practice runs share the same
playback controls, so they have separate selection screenshots and a shared
set of playback screenshots. The browser's Bluetooth chooser, phone
permission dialogs, Google account chooser, and headset display belong to
their respective platforms and need separate device captures. The current
participant gateway does not offer the Box-Atmung diagnostic timeline.

## Latest local verification — 2026-11-14

The updated native fixtures passed all 59 captures against app commit
`deb617b9a8a099603895f5e4c75c857dd421f9bf` with Python Playwright 1.58.0 and
Chromium 145.0.7632.6. Local evidence is under
`output/playwright/native-setup-review-20261114-04/`: gallery, ZIP, source
provenance and `checks.json`. There were no uncaught app errors, horizontal
overflow, painted text overflow, unmeasured labels, undersized controls or
internal panel overflow. Thirty-four captures used readable document reflow.
Conservative Pretext no-fit flags on some existing content-width controls are
recorded separately; their actual painted bounds passed. Native-ready,
renewed-owner recovery and enlarged-text screenshots were visually inspected.
The run used only intercepted synthetic data and did not upload artifacts or
qualify physical headset audio, worn-sensor calibration or an installed PWA.

## Share via Wormhole

Open [Wormhole](https://wormhole.app/), select the generated ZIP, wait for the
upload to finish, and copy the complete download URL including its fragment.
Open that URL in a fresh browser context and confirm the ZIP is available.
Keep the local archive for later feedback because the Wormhole link expires.
