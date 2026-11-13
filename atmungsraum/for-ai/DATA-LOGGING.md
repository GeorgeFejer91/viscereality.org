# Participant data and file naming

This is the required contract for **all Atmungsraum participant data logging**,
including ECG, future sensor streams, events, derived exports and audio receipts.
Quest owns acquisition, the run counter, CSV generation, and recovery.
The browser validates participant input and Quest filenames in `data-logging.js`.

## Study collection on Quest and Pixel

Quest 2 records the original Polar H10 streams and finalizes the participant
CSV on the headset. The installed Pixel PWA pulls verified finalized files
while Quest is idle before or after a session. While the PWA is open and Drive
is authorized, it automatically delivers the verified CSV to the private
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
CSV's elapsed-time origin, including after checkpoint recovery.

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
`probe-15`, or `haupt-15`. The native recording metadata retains the planned duration
and scenario hash. Anleitung is a practice run with a 68-second audio example.

Quest keeps its native streams in the session directory with the recording
manifest. The CSV repeats the participant ID, run index, UUID and scenario
on each row. The participant ID embeds the surname prefix, age and gender code.
Historical exports can also use `<participant_id>_<condition><run>.csv`.
Shared scenario CSVs and audio manifests contain no participant data.

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
measurements by host processing and can show BLE batching or jitter. Rows can
arrive slightly out of observation-time order. Sort by the relevant clock
for analysis. The
renderer value is the radius parameter used for the frame, so it does not
describe the outermost extent of an individual particle. The audio phase and
the classified breathing phase are separate measurements of different things.

## Historical CSV schema v1

Earlier browser and Quest releases wrote 28-column ECG-only participant
CSVs. The live browser controller no longer starts local recordings.

## Saving and recovery

### Native recording synchronization

The enrolled controller uses the RQG1 LAN gateway's idle-only recording
inventory, manifest and read operations. The manifest authenticates the Quest
and recording identity. The browser checks each original file's length and
SHA-256 before committing the v2 CSV or historical ZIP to `atmungsraum-quest-recordings`
in IndexedDB. The key binds the device, serial and immutable native recording
ID. Rejoining reconciles old and new eligible runs, skips saved recordings
and retries interrupted transfers. A Web Lock avoids parallel sync of the
same Quest across tabs. Active, incomplete and unvalidated recordings stay
on Quest. The page must be visible and connected for automatic transfer.

The operator can select a stored archive and explicitly download it. This
store is separate from earlier browser exports.
The v2 path stores the exact Quest CSV and filename as both the local download
and Drive payload. Older participant exports retain the ZIP with the original
ECG and ACC JSONL bytes, CSV and signed manifest. Dated ZIP entries use the
same participant session stem. Legacy runs without a participant CSV retain
the original eight-file archive. The signed manifest retains the Quest stream
keys and digests for the app-private writer and recovery files.
Google Drive delivery reads the committed payload during authorized idle time.
Participant export validates up to 1 GiB of Quest core files. The v2 CSV may
use up to 128 MiB. The older participant ZIP route applies that limit to the
combined CSV, ECG and ACC. Legacy full archives also use the 128 MiB limit.
The browser records the reserved Drive ID and SHA-256 before upload, then
confirms the destination and bytes before marking delivery complete. Keep
delivery receipts separate from Quest and phone recording custody.
The archived JSONL streams preserve separate sensor and Quest clocks. A
Quest v1 participant CSV keeps the original 28-column header and leaves ECG
voltage blank on signal-gap markers. Quest v2 carries Polar sample time
explicitly and uses sample positions to expose gaps.

### Earlier browser exports

The `#daten` route reads completed exports from the existing
`atmungsraum-session-data` IndexedDB database and offers their original bytes
for download. Opening this route in a fresh profile leaves the database absent.
The retired recorder, synthetic ECG trace, and generator are available in Git
history. Quest recordings and earlier downloads stay private.
