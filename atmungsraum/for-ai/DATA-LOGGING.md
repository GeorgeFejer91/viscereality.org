# Participant data and file naming

This is the required contract for **all Atmungsraum participant data logging**,
including ECG, future sensor streams, events, derived exports and audio receipts.
Read it before adding or changing a writer. `data-logging.js` implements
participant normalization, condition codes, run reservation and the primary CSV
filename for the browser demo. The Quest writer must follow the same contract
for native exports.

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
Quest owns the real H10 BLE connection, sample clock, recording and artifact
generation. Preserve the same acquisition/recording separation in the native
path. During integrated runs, the browser serves only as the external controller
and viewer of authorized native state. An export request delegates artifact
generation to Quest under the shared naming contract. Browser-local recording
and CSV generation belong to the standalone synthetic demo.

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

### Native recording synchronization

The enrolled controller uses the RQG1 LAN gateway's idle-only recording
inventory, manifest and read operations. The manifest authenticates the Quest
and recording identity. The browser checks each original file's length and
SHA-256 before committing the complete ZIP to `atmungsraum-quest-recordings`
in IndexedDB. The key binds the device, serial and immutable native recording
ID. Rejoining reconciles old and new eligible runs, skips saved recordings
and retries interrupted transfers. A Web Lock avoids parallel sync of the
same Quest across tabs. Active, incomplete and unvalidated recordings stay
on Quest. The page must be visible and connected for automatic transfer.

The operator can select a stored archive and explicitly download it. This
store is separate from the synthetic CSV list and its upload/share workflow.
Legacy runs use the original eight-file archive. The participant CSV
path keeps the original bytes and Quest filename in browser storage.
Its real sensor rows add `observed_monotonic_ns`, `source_clock`,
`source_time_ns` and `source_frame_sequence`. UTC is the anchored Quest
observation time. Sensor time stays in the Polar clock domain. Gap rows
have blank voltage, and unavailable rehearsal scenario and audio timing
fields stay blank. See PROJECT.md for the open encrypted Arm and physical
phone checks.

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
The legacy `#daten` recovery URL lists retained CSVs for repeat download,
including after reload and without a live signal. The CSV database is loaded
independently of ECG/audio assets. Do not silently delete exports or reset
counters. Browser storage and downloaded participant files are local data;
never commit them or upload them when publishing the static application.

The permanent bottom-right folder opens a dialog on every screen. It counts
this browser's completed/recovered session exports and those still pending. It
reads IndexedDB on opening and after returning to the tab, and retains in-memory
exports if final storage failed. It cannot inspect the OS Downloads directory
or include an unfinished live recording.

**Neue Dateien senden** selects exports without a matching delivery receipt.
**Alle Dateien senden** resends all saved exports. Each action creates
`Atmungsraum-Daten.zip` with the selected CSV bytes and original names; name
collisions use the session UUID as a directory. This remains ZIP32 packaging
using the same writer. An updated CSV under the same UUID counts as new.

After Discord acknowledges a send, the browser saves a separate checkout
receipt under `atmungsraum-file-delivery-v1:receipt:<UUID>` in localStorage.
Each JSON object has a `version`, an `acceptedAt` timestamp and a `files`
array of SHA-256 fingerprints of session ID, filename and exact CSV bytes.
The receipt contains no CSVs, filenames, participant IDs, passwords or URLs.
**Neue Dateien senden** checks the combined fingerprints from all receipts and
the earlier `atmungsraum-file-delivery-v1` history. **Alle Dateien senden**
creates its own receipt after acknowledgment. A Web Lock prevents parallel
sends in tabs of the same origin. Attempts that fail before acknowledgment
leave no checkout receipt. Clearing browser data resets the history. After
a timeout or an unsaved receipt, check Discord before resending. Export files
remain available independently of the checkout receipts.

The masked password decrypts the public webhook ciphertext locally before any
file transfer. On opt-in and confirmed delivery, the browser's own credential
manager can save it. The app does not save the password or unlocked webhook in
web storage. The authenticated upload relay (Render Free setup) receives only
encrypted chunks. A fresh per-upload key and salt encrypt the immutable ZIP
with the pinned Wormhole crypto; two passes hash then upload identical 8 MiB
ciphertext pieces to Wormhole/Backblaze. Only after the provider acknowledges
the upload does the browser send the full keyed URL directly to Discord.
Relay URL and access key clear when the folder closes. See [sharing setup](../sharing/README.md).

Anyone with the full URL can access the share while valid. Wormhole controls
expiry, storage and download limits. Cancellation cannot undo a provider
request already accepted and can leave an unused room. Discord acknowledgement
does not prove a recipient downloaded the archive.

The browser controls the download destination and can block downloads; a click
does not prove a file reached disk. Keep repeat-download links and surface
storage errors. Current [download behavior](https://developer.mozilla.org/en-US/docs/Web/API/HTMLAnchorElement/download)
does not establish filesystem persistence. End sessions normally before closing
the page for a full export. Reload/forced-close recovery uses the last durable
checkpoint, not an assumed successful asynchronous unload write. Clearing browser
storage, storage exhaustion or device failure can still destroy local records.
Real sensor acquisition and cross-device study storage remain open integration
work; do not claim this prototype is a qualified study recorder.
