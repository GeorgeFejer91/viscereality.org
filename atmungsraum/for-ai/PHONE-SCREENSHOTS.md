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
python -m pip install playwright==1.60.0
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

The pack covers connection and enrollment, participant entry, Polar setup,
every Atemvorschau step, each participant scenario selection, playback,
recordings, Drive setup, and recovery. It includes the peak of each button
glow with motion enabled and steady frames for the rest of the sequence.
Main and practice runs share the same
playback controls, so they have separate selection screenshots and a shared
set of playback screenshots. The browser's Bluetooth chooser, phone
permission dialogs, Google account chooser, and headset display belong to
their respective platforms and need separate device captures. The current
participant gateway does not offer the Box-Atmung diagnostic timeline.

## Share via Wormhole

Open [Wormhole](https://wormhole.app/), select the generated ZIP, wait for the
upload to finish, and copy the complete download URL including its fragment.
Open that URL in a fresh browser context and confirm the ZIP is available.
Keep the local archive for later feedback because the Wormhole link expires.
