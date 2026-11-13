# Participant data and file naming

This is the required contract for **all Atmungsraum participant data logging**,
including ECG, future sensor streams, events, derived exports and audio receipts.
Read it before adding or changing a writer. `data-logging.js` implements
participant normalization, condition codes, run reservation and the primary CSV
filename for the browser demo. The Quest writer must follow the same contract
for native exports.

## Study collection on Quest and Pixel

Quest 2 records the original Polar H10 streams and finalizes the participant
CSV on the headset. The installed Pixel PWA pulls verified finalized files
while Quest is idle before or after a session. While the PWA is open and Drive
is authorized, it automatically delivers the retained bundle to the private
research folder. Use [the collection and sync procedure](QUEST-PHONE-SYNC.md)
for enrollment, recovery and verification. The filename and identity contract
below applies throughout the transfer.

## Identity and filenames

The primary session CSV has exactly this filename:

```text
<participant_id>_<YYYY-MM-DD>_<scenario_id>_r<run>.csv
Fej_35_M_2026-11-13_probe-5_r1.csv
Fej_35_M_2026-11-13_haupt-15_r1.csv
```

The filename contains the participant ID, the UTC calendar date of official
Start, the authored session name and the attempt index. The form uses three
surname letters. Quest anchors the date to the same Start event used for the
CSV's `session_started_utc`, including after checkpoint recovery.

| Field | Required meaning |
| --- | --- |
| `name3` | Exactly three Unicode letters from the surname; trim and normalize NFC. Use title case when doing so preserves three letters (e.g. `FEJ` → `Fej`, `MÜL` → `Mül`); preserve the original NFC letters when case conversion expands them. Reject digits, separators, path characters and punctuation. |
| `age` | Nonnegative whole years, canonical decimal integer; no leading zeroes. The form and writer reject missing, fractional, nonfinite and unsafe integers. |
| `gender` | Exactly `M` male, `F` female, `O` other/Divers, or `N` prefer not to say/Keine Angabe. No default choice. |
| `scenario_id` | The exact authored session name. The condition remains `probe` or `main` in metadata and determines the attempt counter. |
| `run` | Positive, one-based attempt index, independently allocated per participant ID and condition. Increment on replay; do not reuse an allocated index after abort or failure. |

The participant ID is `<name3>_<age>_<gender>`, e.g. `Fej_35_M`.
The session stem is `<participant_id>_<YYYY-MM-DD>_<scenario_id>_r<run>`.
Keep demographics unchanged for a participant across conditions. Editing them
creates a different ID; it does not rename previous files. Three surname letters
plus demographics are not guaranteed to distinguish different people: the UUID
`session_id` is the unique identity of an individual run.

The exact exercise is in `scenario_id`: `probe-anleitung`, `probe-5`, `probe-10`,
`probe-15`, or `haupt-15`. The standalone browser also registers `box-20`.
Preserve the exact scenario and planned duration in the CSV. Anleitung is a practice
run; its current audio example is 68 seconds, with no menu exploration limit.

Additional participant files must start with the same session stem and may add
a registered stream suffix, e.g. `Fej_35_M_2026-11-13_haupt-15_r1_events.csv`. Register new
conditions/streams here and in the shared writer before use. Every data file
must retain participant ID, name prefix, age, gender code, condition, run index
and session UUID as metadata; participant CSV rows repeat them. Participant-linked
JSON/audio receipts must retain the same identity too. Shared scenario CSVs,
audio manifests and synthetic source fixtures contain no participant data and
are not renamed by this protocol.

## Participant CSV v2

Quest produces one UTF-8, comma-separated CSV per started attempt. The Pixel
downloads and sends this same file to the existing private Drive destination.
The signed Quest export descriptor carries `schema_version: 2` and binds the
filename, bytes, SHA-256 and session UUID. Quest keeps its checkpointed native
streams for recovery. Older exports retain their original format and ZIP
transfer. Earlier browser CSVs remain downloadable at `#daten`.

| Columns | Meaning |
| --- | --- |
| `schema_version`, `participant_id`, `session_id`, `run_index`, `scenario_id` | Format 2, normalized participant ID, attempt UUID, one-based index, authored scenario |
| `record_type` | `ecg`, `accelerometer`, `radius`, `breath` or `end` |
| `timestamp_utc`, `elapsed_us` | Quest observation time projected onto UTC and whole microseconds since official Start |
| `audio_elapsed_us`, `playback_state`, `audio_phase` | Authored audio position and phase at observation time, with position held through a pause |
| `sensor_time_ns`, `sensor_frame_sequence`, `sensor_sample_index` | Polar sample time on the sensor clock, frame counter and index within that frame, on ECG and accelerometer rows |
| `ecg_uv` | Every retained decoded Polar H10 ECG reading in microvolts |
| `acc_x_mg`, `acc_y_mg`, `acc_z_mg` | Every retained decoded three-axis accelerometer sample in milligravity |
| `render_frame`, `radius_m` | Renderer frame number and the applied world-anchor radius in metres from the particle renderer |
| `breathing_phase` | Polar accelerometer classifier output before guidance bias, including `unknown` |
| `session_status`, `end_reason` | Final outcome and reason on the `end` row |

Streams retain their own cadence and measurement rows. Empty measurement cells
mean that the column does not apply to that row. No values are filled or
resampled to another stream's cadence. For within-sensor analyses use
`sensor_time_ns` and frame/sample positions. Quest observation time groups
measurements by host processing and can show BLE batching or jitter. The
renderer value is the radius parameter used for the frame, so it does not
describe the outermost extent of an individual particle. The audio phase and
the classified breathing phase are separate measurements of different things.

## Historical CSV schema v1

The field list below describes files saved by earlier browser and Quest
releases. The live browser controller no longer starts local recordings.

## Historical browser mock source

The ECG source runs continuously in the background, independently of the
session recorder. Once connected/prepared, its clock continues through menus,
idle time, audio playback and audio pauses. Start begins recording the current
segment of that already-running stream; it must not generate a new source,
restart its epoch or return to sample zero. End/completion closes only the
recording, and a later Start opens another segment at the current stream position.
Audio Pause freezes audio time only: ECG acquisition and its recording continue.
The former demo kept one `MockECG` instance/epoch for the page and snapshot
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
Legacy runs use the original eight-file archive. The participant CSV path keeps
the original bytes and Quest filename in browser storage. One transaction also
stores the same-session sensor ZIP with the original ECG and ACC JSONL bytes,
the participant CSV and the authenticated export manifest. The ZIP, ECG,
ACC and exported manifest filenames all use the participant session stem.
The signed manifest retains the original Quest stream keys and digests.
Those keys identify app-private writer and recovery files. Existing CSVs and
archives keep their saved legacy names and remain readable.
Google Drive delivery reads this committed bundle during authorized idle time.
Participant export validates up to 1 GiB of Quest core files. The combined CSV,
ECG and ACC export remains bounded to 128 MiB for the phone. Legacy full
archives retain their 128 MiB limit.
The browser records the reserved Drive ID and SHA-256 before upload, then
confirms the destination and bytes before marking delivery complete. Keep
delivery receipts separate from Quest and phone recording custody.
The archived JSONL streams preserve separate sensor and Quest clocks. A
Quest v1 participant CSV keeps the original 28-column header and leaves ECG
voltage blank on signal-gap markers. Quest v2 carries Polar sample time
explicitly and uses sample positions to expose gaps.

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

**Ungesendete Daten senden** selects exports without a matching delivery receipt.
**Alle Daten senden** resends all saved exports. Each action creates
`Atmungsraum-Daten.zip` with the selected CSV bytes and original names; name
collisions use the session UUID as a directory. This remains ZIP32 packaging
using the same writer. An updated CSV under the same UUID counts as new.

After Discord acknowledges a send, the browser saves a separate checkout
receipt under `atmungsraum-file-delivery-v1:receipt:<UUID>` in localStorage.
Each JSON object has a `version`, an `acceptedAt` timestamp and a `files`
array of SHA-256 fingerprints of session ID, filename and exact CSV bytes.
The receipt contains no CSVs, filenames, participant IDs, passwords or URLs.
**Ungesendete Daten senden** checks the combined fingerprints from all receipts and
the earlier `atmungsraum-file-delivery-v1` history. **Alle Daten senden**
creates its own receipt after acknowledgment. A Web Lock prevents parallel
sends in tabs of the same origin. Attempts that fail before acknowledgment
leave no checkout receipt. Clearing browser data resets the history. After
a timeout or an unsaved receipt, check Discord before resending. Export files
remain available independently of the checkout receipts.

The startup password decrypts the public delivery envelope locally and opens
the app interface. The browser can remember it on opt-in. The app holds the
decrypted webhook in tab memory. The GitHub Pages-only release does not have
an automatic upload route because Wormhole rejects the app origin.

The tested relay flow encrypts the immutable ZIP on the phone with a fresh
per-upload key and salt and the pinned Wormhole crypto. Two passes hash and
upload identical ciphertext pieces of up to 8 MiB. After Wormhole acknowledges
the upload, the browser sends the complete keyed URL directly to Discord.
See [sharing](../sharing/README.md).

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
