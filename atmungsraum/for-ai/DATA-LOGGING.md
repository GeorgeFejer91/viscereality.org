# Participant data and file naming

This is the required contract for **all Atmungsraum participant data logging**,
including ECG, future sensor streams, events, derived exports and audio receipts.
Read it before adding or changing a writer. `data-logging.js` owns participant
normalization, condition codes, run reservation and the primary CSV filename.
Reuse those functions; do not construct participant filenames in another writer.

## Identity and filenames

The primary session CSV has exactly this filename:

```text
<name3>_<age>_<gender>_<condition><run>.csv
Fej_35_M_probe1.csv
Fej_35_M_main1.csv
```

The order is **name, age, gender, condition/run**, following the user's examples.
The user confirmed three surname letters rather than the full surname.

| Field | Required meaning |
| --- | --- |
| `name3` | Exactly three Unicode letters from the surname; trim and normalize NFC. Use title case when doing so preserves three letters (e.g. `FEJ` → `Fej`, `MÜL` → `Mül`); preserve the original NFC letters when case conversion expands them. Reject digits, separators, path characters and punctuation. |
| `age` | Nonnegative whole years, canonical decimal integer; no leading zeroes. The form and writer reject missing, fractional, nonfinite and unsafe integers. |
| `gender` | Exactly `M` male, `F` female, `O` other/Divers, or `N` prefer not to say/Keine Angabe. No default choice. |
| `condition` | `probe` for every Probe-Runde option, including Anleitung; `main` for Haupt-Runde. Internal route `haupt` maps to `main` only at this shared boundary. |
| `run` | Positive, one-based attempt index, independently allocated per participant ID and condition. Increment on replay; do not reuse an allocated index after abort or failure. |

The participant ID is `<name3>_<age>_<gender>`, e.g. `Fej_35_M`.
The session stem is `<participant_id>_<condition><run>`.
Keep demographics unchanged for a participant across conditions. Editing them
creates a different ID; it does not rename previous files. Three surname letters
plus demographics are not guaranteed to distinguish different people: the UUID
`session_id` is the unique identity of an individual run.

The exact exercise is in `scenario_id`: `probe-anleitung`, `probe-5`, `probe-10`,
`probe-15`, or the single `haupt-15`. Preserve this field and planned duration;
the filename's `probe`/`main` does not encode duration. Anleitung is a practice
run; its current audio example is 68 seconds, with no menu exploration limit.

Additional participant files must start with the same session stem and may add
a registered stream suffix, e.g. `Fej_35_M_main1_events.csv`. Register new
conditions/streams here and in the shared writer before use. Every data file
must retain participant ID, name prefix, age, gender code, condition, run index
and session UUID as metadata; participant CSV rows repeat them. Participant-linked
JSON/audio receipts must retain the same identity too. Shared scenario CSVs,
audio manifests and synthetic source fixtures contain no participant data and
are not renamed by this protocol.

## CSV schema v1

Use UTF-8, comma separators, one header row, CRLF records, decimal points and
RFC-style quoting. Quote commas, quotes and newlines; neutralize spreadsheet
formula prefixes in nonnumeric text. Retain these current fields:

| Group | Columns |
| --- | --- |
| Identity | `schema_version`, `participant_id`, `name_prefix`, `age_years`, `gender_code`, `condition`, `run_index`, `session_id` |
| Authored session | `scenario_id`, `scenario_sha256`, `planned_duration_seconds`, `session_started_utc` |
| Sample and clocks | `sample_index`, `elapsed_ms`, `timestamp_utc`, `audio_elapsed_ms`, `playback_state`, `phase` |
| ECG provenance | `ecg_value`, `ecg_unit`, `signal_source`, `signal_state`, `signal_sha256`, `sample_rate_hz`, `source_sample_index`, `source_loop` |
| Outcome | `session_status` (`completed` or `aborted`), `end_reason` |

Sample indices start at zero. UTC timestamps use ISO 8601 with `Z`; elapsed
times derive from `performance.now()`. The ECG wall clock continues through
deliberate audio pauses; `audio_elapsed_ms` stays fixed while paused. `phase`
follows the selected authored audio protocol, including the distinct `instruction_1`, `calibration`, `instruction_2` and
`outro` main classes. Main `planned_duration_seconds` is exactly 900, including
all instructions and calibration. No main baseline is present. Calibration
receipts expose planned audio-clock start/end; the future H10 accelerometer
value remains null/not-integrated until a real measurement source is wired.
Missing signal produces a blank ECG value with explicit `signal_state`, rather
than a stale or fabricated available measurement. A signal timeout, navigation,
Escape, hidden page, interrupted audio clock or manual End aborts the attempt;
retain its partial CSV and the reason. A start that never produces a run has no
successful-session CSV. Reserve indices transactionally; gaps are acceptable.

## Current mock source

The ECG source runs continuously in the background, independently of the
session recorder. Once connected/prepared, its clock continues through menus,
idle time, audio playback and audio pauses. Start begins recording the current
segment of that already-running stream; it must not generate a new source,
restart its epoch or return to sample zero. End/completion closes only the
recording, and a later Start opens another segment at the current stream position.
Audio Pause freezes audio time only: ECG acquisition and its recording continue.
The demo keeps one `MockECG` instance/epoch for the page and snapshots
`signal.indexAt(recordingStart)` per run. A page reload starts a new demo clock.
The future real H10 receiver must preserve this acquisition/recording separation.

`tools/generate-mock-ecg.py` calls NeuroKit2 `ecg_simulate(method="ecgsyn")`
with seed 42, 70 bpm, 1 bpm standard deviation, noise 0.01 and 256 Hz.
The shipped trace is 1,800 seconds / 460,800 float32 little-endian samples.
`assets/ecg/mock-ecg.json` records generator/library versions, settings and the
binary SHA-256. The browser checks length, hash and finite values before ECG-ready.
The generator currently uses NeuroKit2 0.2.13, NumPy 2.5.2 and SciPy 1.18.0.
Python is a local asset-generation dependency, not a browser/runtime service.

The static site continuously replays this NeuroKit-generated source against a
page-clock epoch; it does not run Python in the browser. The source wraps after
30 minutes, explicitly recorded by `source_sample_index` and `source_loop`.
Samples are generated independently of participants and are not physiological
responses to the exercise. Values use arbitrary units (`a.u.`), not calibrated
H10 units. Export reconstructs the deterministic mock sample timeline, with
availability and playback anchors recorded during the run. Its timestamps are
simulated sample times, not hardware arrival or acoustic-onset measurements.
The trace is still generated when its visualization is hidden or audio is paused.

See the official [NeuroKit ECG simulator source](https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/ecg/ecg_simulate.html).
Future real H10 logging must ingest actual samples, units, timestamps and loss
events. It must not relabel this replay/reconstruction as hardware data.

## Saving and recovery

Each successfully started run snapshots the patient and selection. Its unique
UUID is shared by its CSV and audio receipt. IndexedDB
`atmungsraum-session-data` reserves counters with a read/write transaction, so
same-origin tabs cannot allocate the same next index. Counters and completed
CSV exports survive reloads in that browser profile. Counter uniqueness is
scoped to that origin/profile; clearing storage or using a different device
does not share a study registry. Preserve session UUIDs and reconcile counters
with a shared registry before combining real study data across devices.

The initial active checkpoint and shared source cache are committed before the
recording reports ready. Further checkpoints occur every second and on playback,
signal and stop changes. On reload, reconstruct an interrupted run only through
its last durable checkpoint; mark it `aborted` / `page-reload-or-interrupted`.
The uncommitted tail can be absent. Cached source samples allow recovery without
network ECG assets. A per-session Web Lock is held by the live recording and
released after its final export transaction (or automatically at teardown).
Recovery takes that lock only if available and re-reads the checkpoint, so it
cannot finalize another tab's live run. This browser recorder requires IndexedDB,
Web Locks, Web Crypto and a secure context (HTTPS or localhost).

Completion or abort triggers a CSV download and stores the export locally.
The bottom-right folder lists retained exports for repeat download, including
after reload and without a live signal. It is the sole visible data entry point;
the legacy `#daten` recovery URL still works, with no separate CSV-Dateien button.
The CSV database is loaded independently
of ECG/audio assets so a missing source cannot block recovery. Do not silently
delete exports or reset counters. Browser storage and downloaded participant
files are local data, not repository assets; never commit them or upload them
as part of publishing the static application.

The permanent bottom-right folder button opens a native dialog over any screen.
It lists the same saved exports, refreshes from IndexedDB on opening, and keeps
in-memory exports available if final browser storage failed. It does not inspect
the OS Downloads directory or include an unfinished live recording. Every entry
remains individually downloadable. `Atmungsraum-Daten.zip` bundles all retained
CSV files with their original bytes/names; a colliding name is placed inside its
session UUID directory. This is a packaging name, not a new participant writer.
The ZIP uses uncompressed ZIP32 entries, rejecting unsupported counts/sizes.
The explicit Teilen action passes that ZIP to the native file share sheet when
supported; recipients/apps are chosen by the user. Unsupported/rejected sharing
retains a ZIP download for manual attachment. No mailto attachment workaround,
automatic transmission or server upload is involved. Cancel leaves files intact.

The explicit **Per QR-Code teilen** action offers automatic Wormhole upload
through a restricted authenticated upload relay, followed by local QR generation and
password-gated forwarding directly from the browser to Discord. The user's
2026-11-07 choice hosts only an encrypted webhook envelope in public Git/Pages;
a send-time password decrypts it locally. The raw webhook and password must never
enter public source. Cancel or a wrong password sends nothing to Discord.
The password field clears on close. The app writes neither password nor unlocked
webhook to web storage/IndexedDB. On explicit opt-in and confirmed delivery,
use only the browser's native password manager to save/autofill the password.
The relay is used only for optional Wormhole upload. See
[sharing/README.md](../sharing/README.md).
A manual fallback remains: download the ZIP, upload it at `https://wormhole.app/`,
paste the complete URL including its `#` key, then choose **QR-Code anzeigen**.
Wormhole rejects site-origin upload CORS preflight; Pages alone cannot upload.

Automatic upload reuses the exact same ZIP writer. A fresh per-upload key and
salt encrypt its immutable Blob with official pinned `wormhole-crypto@0.3.1`.
A first pass hashes encrypted 8 MiB pieces; a second pass uploads identical
ciphertext through narrowly allowed relay routes to Wormhole/Backblaze. The
file key remains on the device during upload. No plaintext CSV is sent to the
relay or storage provider. Only after upload completion acknowledgement does
the app show the URL/QR and request the Discord password. After successful
local decryption it sends the full keyed URL directly to Discord; no notification
passes through Cloudflare. The Discord destination has access to that share.
Provider acknowledgement is not proof of recipient access.

Relay URL/access key are entered privately per dialog and cleared on close or
collapse; neither is persisted in browser storage. A confirmed uploaded URL is
retained only in page memory for the same exact saved records, restored after
dialog reopening, and invalidated when those records change. Reload loses this
receipt. Notification retry prompts for the password and reuses the existing
URL without uploading again;
successful sends are remembered in this page to suppress accidental repeats.
No durable exactly-once delivery is promised. Check the destination after an
uncertain timeout before retrying. Cancellation cannot undo a provider request
already accepted. Existing uploads expire under the provider's rules; invalidating
local UI does not delete or revoke the share.

QR generation uses the local MIT `qrcode-generator@2.0.4` encoder under
`vendor/qrcode/`, loaded only when needed. No remote QR service, automatic
clipboard read or analytics is used. Accept only complete HTTPS links on the
exact `wormhole.app` origin with a room and 128-bit base64url fragment; reject
credentials, extra paths, queries and missing keys. Pasted-link validation is
format validation, not proof of the room's existence, contents or availability.
Copy/open actions use the whole URL. Editing, closing/collapsing or refreshing
clears displayed links/QRs; an acknowledged automatic-upload receipt may restore
the same share as described above. A failed QR module leaves the complete link,
copy/open actions and downloads available. Clipboard denial selects the input
for manual copying. No saved export is deleted. Anyone with the QR/link can
access that Wormhole share while it is valid. Wormhole controls expiration,
storage and download limits; no availability promise is fabricated. See its
[security design](https://wormhole.app/security) and [FAQ](https://wormhole.app/faq).

The browser controls the download destination and can block downloads; a click
does not prove a file reached disk. Keep repeat-download links and surface
storage errors. Current [download behavior](https://developer.mozilla.org/en-US/docs/Web/API/HTMLAnchorElement/download)
does not establish filesystem persistence. End sessions normally before closing
the page for a full export. Reload/forced-close recovery uses the last durable
checkpoint, not an assumed successful asynchronous unload write. Clearing browser
storage, storage exhaustion or device failure can still destroy local records.
Real sensor acquisition and cross-device study storage remain open integration
work; do not claim this prototype is a qualified study recorder.
