# Quest, Pixel and Google Drive recording sync

Quest owns acquisition, timing, participant-run identity, and the original
recording. Atmungsraum on the phone pulls finished files from the enrolled
Quest over trusted Wi-Fi. An idle pull can start before the first rehearsal or
after any rehearsal and catches up on files created while the controller was
closed. The installed Pixel PWA is the study sync controller. Its verified local
copy is the input to automatic delivery to the research Google Drive.
After submission, the Haupt-Runde questionnaire JSON uses the same idle
delivery to the same private folder, with its own file ID and checksum
receipt. See the [questionnaire data contract](DATA-LOGGING.md#haupt-runde-questionnaire).

## Live sync goal

During a session the phone polls a short ECG waveform for the operator. With
a live-capable Quest build, the phone also pulls committed ranges of the
full-resolution CSV and uploads them in resumable batches while Quest runs
biofeedback locally. The Drive file becomes available after finalization.
The installed Pixel verified this with a synthetic 256 KiB prefix and a
two-byte final tail in the private research folder. Both the chunk response
and a separate status query exposed the acknowledged range. Drive reported
the expected owner, parent, size and SHA-256 after finalization. The synthetic
object was then moved to trash.

The CSV transport reserves one Drive file ID and sends batches of at most
1 MiB. During a live upload it retains at least one byte until the final
length is known. The PWA persists the acknowledged offset and the end of
an in-flight request in IndexedDB. After an ambiguous response it queries
Drive before sending more data. An expired upload session can restart at byte
zero with the same file ID. The browser uses Google's upload-specific status
override to read partial acknowledgements while rejecting HTTP redirects.
The installed app must be in the foreground and Drive must be authorized for
live upload. Quest keeps recording if either connection drops. Pixel resumes
from its committed offset after rejoin. If the browser was reloaded, reconnect
Drive from the existing account control to renew the in-memory authorization.
The final reconciliation pulls any remaining bytes and verifies the complete
signed SHA-256 before finalizing the same Drive file ID.
If staged bytes are missing or their assembled checksum differs, the Pixel
fetches the verified file again from byte zero. It opens a fresh resumable
session under the existing Drive ID before sending any corrected bytes.

For the paired performance trial, open the controller with
`?live-sync=off` for the baseline, then use the ordinary URL for the live
run. Both URLs preserve local recording and the existing post-session sync.
The baseline setting disables active phone transfer and Drive upload on that
controller. The Quest CSV projection is active in both conditions.

See the [active checklist](PROJECT.md#live-recording-sync-to-google-drive)
and [Drive resumable upload documentation](https://developers.google.com/workspace/drive/api/guides/manage-uploads).

### Pixel Wi-Fi and reload recovery

The installed Pixel PWA release `20261113-live-repair` staged 524,289
synthetic bytes in IndexedDB. Drive accepted the first 524,288 bytes. The
fixture deliberately dropped that acknowledgement after the real response
arrived, leaving the pending range on disk.

Pixel Wi-Fi was disabled at the device. A signed Quest gateway attempt
failed. The phone rejoined its original network, reached Quest, and reloaded
the installed PWA. After reconnecting the authorized Drive account, staging
resumed at byte 524,289. A Drive status query recovered the accepted range
and the existing upload session resumed at byte 524,288. Drive acknowledged
1,048,576 bytes before finalization, then verified the 1,310,725-byte file
with the expected SHA-256 under the same file ID. The fixture trashed the
synthetic file and cleared its local entries.

This test used real Wi-Fi, the real enrolled Quest connection and the
designated private Drive. Quest stayed idle and the data ranges were
synthetic. Repeat the recovery during a physical recording to measure
the effect on live Polar acquisition, VR biofeedback and phone controls.

### Paired physical trial measurements

Use a synthetic participant ID and the full `haupt-15` audio on the same APK,
Pixel release, Wi-Fi link, Polar presentation policy and renderer settings.
Run the sync-disabled and sync-enabled conditions with the same planned phone
Pause, Resume and Finish actions. Collect a comparable uninterrupted interval
in both runs. In the enabled run, separately mark the fault windows for Wi-Fi
loss, a failed Drive request and a Pixel reload. Verify the UI rejoined the
same running generation, the local and Drive offsets resumed, and Finish
produced one object with the original file ID.

Record Quest's OpenXR frame timing from the renderer scorecard and radius
timestamps. Report the median, p95 and p99 frame interval and the sample
count for each comparison window. For Polar freshness, use the native
`accepted_age_ms` and `ingress_to_commit_ms` diagnostics and the host
timestamps in `breath.jsonl` provenance. For the low-latency policy, the
same-clock age is the Java midpoint minus the selected sample's host receipt
time. Report the presentation policy and the fraction of valid Polar
assessments. Check that the applied radius varies with valid Polar breath
volume during playback. Compare the phone tap to the signed control receipt
and visible state change. Report recording drops from the terminal receipt
and corroborate stream-specific interruptions and resumptions from the CSV.

Measure live sync by byte offset. Pair a Quest prefix observation with the
first Drive acknowledgement that covers it. Use the response `Range` and
the end of the browser request as the acknowledgement witness. The first
phone observation of the prefix gives an observed delay. For a bound on the
earlier Quest publication, capture the checkpoint sidecar change on Quest.
Measure the device wall-clock offset before and after each run if comparing
timestamps across devices. A bench clock check found Quest about 0.95 seconds
ahead of Pixel. Keep same-device elapsed times for frame timing and controls.
Report live and fault windows with sample counts and any clock uncertainty.
Retain only timing, status, byte counts and hashes in shared evidence.

### Sync-enabled physical trial on 2026-11-14

Quest 2 APK `0.1.29` and the installed Pixel 10 PWA release
`20261113-live-repair` ran all 900 seconds of `haupt-15` with a real H10
and a synthetic participant ID. The operator used the Pixel for playback
and Finish. The first Drive range acknowledgement arrived about 7.5 seconds
after the first accepted Start request. In the uninterrupted window at
165–235 seconds, the median, p95 and p99 delay from Quest prefix observation
to Drive acknowledgement were 3.41, 5.35 and 6.13 seconds over 69 sidecar
samples. Using the preceding Quest read to bound publication gave 4.45,
6.37 and 7.11 seconds at the same percentiles. These delays use one host
clock. Device clock checks put Quest about two seconds ahead of Pixel.

- An injected Drive request reset at byte 17,301,504 stopped the upload.
  About ten seconds later, a Drive status query confirmed the last accepted
  range and the next request resumed at byte 17,301,504.
- With Pixel Wi-Fi disabled for about 8.5 seconds, the Quest sidecar kept
  advancing. The Pixel rejoined and continued upload from its saved offset.
- After a deliberate WebAPK reload, the PWA rejoined the running session.
  Google required another account selection. The first resumed Drive write
  began at byte 24,379,392. The Pixel later displayed Chrome's crash screen.
  After a second reload and Drive reauthorization, the PWA caught up and
  kept using the same reserved Drive file ID. The renderer exit cause was
  unconfirmed. This trial ran with DevTools diagnostics attached.
- Separate ECG and accelerometer stop/restart commands produced one
  interruption/resumption pair per stream in CSV v4. The other stream
  continued during each deliberate stop. Manual playback pauses produced
  playback events. The accelerometer loss triggered the expected Quest
  playback hold, and the phone resumed it after both streams recovered.

Quest saved the run with zero writer drops and one terminal row marking
completion after 900 audio seconds. The final participant CSV was 74,967,884
bytes. Drive had acknowledged 74,711,040 bytes before Finish. After Finish,
the final request sent the remaining 256,844 bytes. An independent Drive
metadata and folder query confirmed one private CSV with the same name, byte
count and SHA-256 as Quest and the installed Pixel. The verified receipt
retained the original Drive file ID. SHA-256
`adfb17cf038c08ce43cdfb8e01d3fb109b9ac4c4a8aed23da522d5bfd03365c1`.
The sync-disabled baseline and paired performance comparison are tracked
in the [active checklist](PROJECT.md#live-recording-sync-to-google-drive).

### Paired physical trial results on 2026-11-14

Both sessions used Quest 2 APK `0.1.29`, the installed Pixel 10 PWA
`20261113-live-repair`, condition A, the same H10 and the low-latency
Polar presentation policy. Both played the complete 900-second audio and
ended via the Pixel. The normal Pause and Resume actions were exercised
around 04:23 and 12:09. The enabled run included the fault windows above.
The baseline used `?live-sync=off`. Browser network observation saw no
live prefix requests or Drive upload during baseline playback.

The table uses the uninterrupted 165–235 second playback window.
Values in the timing rows are median, p95 and p99 in milliseconds.

| Measurement | Sync disabled | Sync enabled |
| --- | --- | --- |
| Radius observation intervals (5,027 per run) | 13.925 / 14.324 / 14.602 | 13.924 / 14.375 / 15.056 |
| Scorecard mean period between frame numbers (41 intervals per run) | 13.925 / 13.950 / 13.958 | 13.925 / 13.933 / 13.942 |
| CPU time per submitted OpenXR frame (42 scorecards per run) | 1.308 / 1.430 / 1.647 | 1.545 / 1.831 / 3.571 |
| Native accepted Polar sample age | 112 / 313 / 356 (69) | 95 / 223 / 387 (70) |
| Native ingress-to-commit | 0 / 0 / 0 (69) | 0 / 0 / 0 (70) |
| Selected Polar sample age at the Java midpoint | 107.747 / 331.992 / 427.572 (4,979) | 91.957 / 216.979 / 287.418 (5,027) |
| Valid Polar breath assessments | 4,979 of 4,979 | 5,027 of 5,027 |

The scorecard period divides wall time between scorecards by the change in
submitted frame number. Quest and Pixel clocks were checked at the bench.
The Quest wall clock shifted by about two seconds during the baseline.
The per-frame radius and source-age measurements use device monotonic clocks.
Using either wall-clock offset to select the baseline scorecards gives the
same p95 CPU time and frame period.

The renderer used Polar phase mapping. In the enabled comparison window,
314 matched inhale observations raised the radius, 246 matched exhale
observations lowered it, and 4,322 hold observations kept it steady.
The baseline comparison window contained only hold assessments, and its
radius stayed steady. The first 445 seconds of baseline include 406
matched inhale observations that raised the radius and 367 exhale
observations that lowered it. Both recordings retained the applied radius.

At the matched 12:09 phone actions, the enabled run showed Pause and Resume
changes after 1.34 and 1.58 seconds. The baseline showed them after 1.56 and
0.62 seconds. The confirmed signed receipts arrived after 1.22 and 1.44
seconds with sync enabled, and after 1.47 and 0.47 seconds with sync disabled.
The baseline also showed the expected state changes after 1.40 and 1.52
seconds at the earlier pause and resume. This pair shows stable frame
cadence, fresh biofeedback and an increase in submitted-frame CPU time
with live sync. Phone response varied between actions. A transient Quest
status rejection occurred during Finish in both runs, then the UI recovered.
The enabled run had the additional Chrome renderer exit reported above.
These observations describe one pair of runs with DevTools attached.

Quest finalized both recordings with zero writer drops. The baseline
recording was 67,609,282 bytes, with 13 naturally occurring ECG interruption
and resumption pairs and no accelerometer interruption marker. Each reported
ECG gap exceeded the one-second marker threshold. The enabled run retained
the separate deliberate ECG and accelerometer gaps above. The baseline CSV
arrived in Drive about 7 minutes 24 seconds after the phone Finish tap.
Independent metadata and folder queries confirmed one private CSV with the
same name, size and SHA-256 as Quest and Pixel. Its SHA-256 was
`9556af6101e599589c0fdfa4b11f87449f6e5259fb6e06804e50dfa53b1c0b79`.

### Pixel live upload renderer recovery

During a repeat with the original PWA, Chrome exited during live upload
with a bad Blob reference in the Android system log. It did so again after
reload without continuous DevTools observation. Quest retained its active
recording and continued to publish committed prefixes.

The `20261114-recording-bytes` PWA stores verified live ranges as byte
arrays and assembles bounded Drive writes directly from bytes. IndexedDB
can still read ranges stored as Blobs by the earlier release. It reads
the old Blob payload directly before combining it with neighboring bytes.
The final CSV retains its signed hash and file naming checks. On the
installed Pixel, a fixture read 65,534 exact bytes across one old Blob and
two new ranges, rejected a missing suffix, and verified the saved offset.
The fixture cleared its own staging entries.

After [Pages run 43595036518](https://github.com/GeorgeFejer91/viscereality.org/actions/runs/43595036518),
the installed Pixel loaded the corrected release during the same Quest
recording. Drive resumed from the pending range above 13 MB and accepted
78,118,912 bytes before Finish. The Pixel kept the original Drive file ID.
It remained responsive through the final seven minutes of playback and
the Android log showed no further renderer exit during that interval.
Quest, Pixel and Drive matched at 78,340,857 bytes with SHA-256
`f6e73ff937a3d16202dbcd80ed08b3253d9f18e782b7b9c758841ee353349c92`.
The Drive folder contained one private copy. Quest saved all 900 audio
seconds with zero writer drops. This repeat had intermittent H10 gaps and
the applied radius stayed at its minimum, so use the first paired runs
for the biofeedback comparison.

### Full trial on the updated PWA

The installed Pixel ran `20261114-recording-bytes` from Start through
Finish with Quest 2 APK `0.1.29`. Quest completed the authored 15-minute
audio with zero writer drops. Android logged no Chrome renderer exit
during this run. Drive reconciled a failed write at byte 20,709,376,
and the PWA caught up after about 8.5 seconds of Pixel Wi-Fi loss.
After a planned reload, Quest rejoined during the same run and Drive
authorization was renewed. The acknowledged offset advanced from
24,379,392 to 25,427,968 under the original file ID. Fresh ECG and
accelerometer samples resumed after deliberate separate stops. CSV v4
retained both stop/resume pairs and 11 additional ECG gap pairs.

Drive had acknowledged at least 66,322,432 bytes before Finish.
Quest, Pixel and the single private Drive CSV matched at 67,400,882 bytes
with SHA-256
`3d9112d6fb8a66c10d09765eece17c16af32bbaaec9db01b37d41da519e68167`.
Both streams were fresh in all 70 signed status observations during the
165–235 second comparison window. Native accepted sample age was
116 ms median and 185 ms p95. Radius observation intervals were
13.929 ms median and 14.338 ms p95 across 5,027 intervals. The two phone
Pause/Resume pairs showed the expected new controls in 0.80/0.53 and
1.36/1.49 seconds. Finish returned to the completed screen after 16.12
seconds, including the native save.

Quest received both streams but classified the breath motion as rejected
in this run. The rendered radius stayed at its minimum. The earlier
paired qualification demonstrates live directional scene response and
provides the comparison with the baseline. The updated run verifies
the data path, renderer cadence, input freshness and recovery.
With live sync disabled on the same release, a 38-second physical probe
also showed fresh ECG, `rejected-motion` in all 450 sampled breath
observations, and a fixed radius. The probe did not establish the cause
of motion rejection. Keep the native classifier's diagnostic state
separate from live upload and raw signal availability when interpreting
the result.

### Fresh Polar response on the updated PWA

A subsequent full `haupt-15` run on `20261114-recording-bytes`
restored directional Polar biofeedback during live Drive uploads.
The uninterrupted 165–235 second playback window contained 5,028
valid Polar breath assessments and no rejected assessments. Quest
recorded 222 rising radius steps with classified inhale and 368 falling
steps with classified exhale. The applied radius ranged from 1.00 to
1.65 m. All 70 signed phone status observations in this window reported
fresh ECG and accelerometer streams. The selected Polar sample age
at the Java midpoint was 92.1 ms at the median and 225.2 ms at p95.
Median and p95 radius observation intervals were 13.924 and 14.378 ms
over 5,027 intervals. Pixel continued to obtain live Drive acknowledgements
during the same window.

The full run completed the authored 15-minute audio and saved with zero
writer drops. The final CSV contained 70,280 renderer observations with a
frame number and an applied radius between 1.00 and 2.00 m. It also contained
126,491 ECG readings, 195,416 three-axis accelerometer readings, and 69,718
breathing phase observations. The export recorded six ECG and four
accelerometer interruption/resumption pairs, including the deliberate
stop of each stream.

During this run, Drive reconciled the failed request at byte 20,709,376.
Quest continued recording through the 8.6-second Pixel Wi-Fi interruption
and the subsequent WebAPK reload. The Pixel rejoined the same recording
and, after Drive reauthorization, resumed at byte 22,020,096 under the same
file ID. Drive accepted 69,730,304 bytes before Finish. The final single
private CSV was 69,906,729 bytes and matched the Quest and Pixel copies
by SHA-256 `7c3830f15a402aea54427c5a723d4a8d1576f1a14db8b5509234d74b94fd3aae`.

After transport recovery, 63 Quest recording checkpoints during playback
seconds 575–640 were matched to the first Drive acknowledgement covering
their bytes. The median delay was 3.20 seconds, p95 was 4.94 seconds, and
p99 was 5.35 seconds. Both observations used the bench computer clock.

### Repeat comparison on the updated PWA

Both full sessions used the installed Pixel release
`20261114-recording-bytes` and Quest 2 APK `0.1.29`. The baseline used
`?live-sync=off`. The Pixel network observer saw zero Drive writes during
baseline playback. The scorecard and signed status measurements use
playback seconds 165–235. Radius intervals and Java midpoint sample age
use seconds 165–235 on the native observation clock, which advances
through a playback pause. Timing values are median, p95 and p99
in milliseconds.

| Measurement | Sync disabled | Sync enabled |
| --- | --- | --- |
| Radius observation intervals (5,027 per run) | 13.928 / 14.297 / 14.597 | 13.924 / 14.378 / 14.808 |
| Scorecard mean period between frame numbers (41 intervals per run) | 13.925 / 13.942 / 13.950 | 13.925 / 13.942 / 13.950 |
| CPU time per submitted OpenXR frame (42 scorecards per run) | 1.326 / 1.610 / 2.265 | 1.375 / 1.708 / 1.787 |
| Native accepted Polar sample age | 127 / 513 / 793 (72) | 98 / 222 / 424 (70) |
| Selected Polar sample age at the Java midpoint | 112.9 / 377.8 / 466.8 (4,840) | 92.1 / 225.2 / 314.6 (5,028) |

At the matched phone probe around 12:09, Pause and Resume each updated
the visible controls in 0.74 and 0.54 seconds with sync disabled, and in
1.26 and 0.71 seconds with sync enabled. Signed command receipts arrived in
0.61 and 0.43 seconds with sync disabled, and in 1.20 and 0.51 seconds with
sync enabled. A Polar hold coincided with the earlier baseline Pause and
Quest rejected that phone command, so the 04:23 action is excluded from
the comparison. Finish reached the completed screen after 18.46 seconds
with sync disabled and 15.78 seconds with sync enabled.

Both runs reached the end of the authored audio and Quest saved with
zero writer drops. The baseline recorded 79 ECG and 22 accelerometer gap
pairs and 19 playback pauses. Quest kept the radius at 1.00 m and
exported all baseline breathing phases as unknown.
The repeat compares frame timing, raw sample age, phone controls and
recording reliability. The baseline classification state prevents a
comparison of directional biofeedback in this pair. The
[earlier pair](#paired-physical-trial-results-on-2026-11-14) observed
directional biofeedback in both modes.

The baseline produced a 77,448,407-byte CSV. Quest, Pixel and a single
private Drive object matched by name, size and SHA-256
`155801cc7f8e66d096ab30f72e9b56c627f2d3f7c33f39d2c687ea5257ffa182`.
Drive acknowledged the final bytes 21 minutes 21 seconds after the
Finish tap. That interval includes an aborted export, a rejected Quest
read, Google reauthorization and interruptions to USB observation. The
phone eventually finished the transfer over Wi-Fi and verified delivery.

## Operator procedure

1. Open the installed Atmungsraum icon on the enrolled Pixel 10. The app
   tries the saved Quest connection first. If needed, tap **Mit Bluetooth
   verbinden** and select the Quest in Chrome. Grant Local Network Access if
   prompted. A configured hotspot is reused when the shared LAN is
   unreachable. Close the headset's maintenance panel to return to the
   immersive scene before starting. The PWA expects prior trusted enrollment
   from lab provisioning. See the [connection checklist](PROJECT.md#automatic-quest-connection--2026-11-14)
   for fresh-phone provisioning and hotspot automation.
2. Before a session, let the idle controller reconcile any finalized runs.
   In **Google Drive**, authorize the designated research account as described
   below. The PWA creates and remembers **Atmungsraum – Quest-Aufnahmen** as
   its private Drive destination. The PWA stages verified bundles in
   IndexedDB. The automatic queue writes those bundles to the Drive folder.
3. Confirm fresh Polar ECG and accelerometer data, enter the participant ID using the shared form,
   choose the scenario and start the session. Quest writes the original
   streams and the participant CSV and finalizes the run locally.
4. After the session, keep the Pixel PWA visible and the Quest idle until the
   run appears in the recording selector. Wait for the Drive confirmation,
   then verify that the new CSV is in the designated folder. For qualification,
   compare the SHA-256 and manifest with the copies on the Pixel and Quest.
5. After an interrupted connection, reopen the app while Quest is in Remote
   mode. The saved pairing opens VR-Probe and tries the last known address.
   If it changed, supported browsers can rediscover the Quest using its
   saved Bluetooth grant. Use **Mit Bluetooth verbinden** if Chrome needs the chooser
   again or Local Network Access has to be granted.
   Use **Jetzt abgleichen** if needed. If Google authorization expired,
   reconnect Drive. Confirm delivery before relying on the Drive copy.

## Bench verification on Quest 2 and Pixel 10

The signed Quest rehearsals start in Remote. The installed Pixel Chrome
157.0.8103.50 WebAPK discovers it over Bluetooth, then opens its authenticated
Wi-Fi gateway. Close any floating overlays if Chrome blocks Local Network
Access. A Quest APK update that changes the trusted build fingerprint
can be accepted using **Mit Bluetooth verbinden** with the same trusted enrollment. A restart on the same
build can rejoin the saved address.

The first two bench recordings used synthetic participant details and live
H10 signals. The first exercised the 68-second instruction timeline, manual
Pause and Resume, and recovery after ECG interruption. The second finished
early after a Quest restart and Pixel PWA reload. Quest saved both attempts
with distinct UUIDs, successive attempt indexes and zero dropped writer
records. SHA-256 matched for the CSV, ECG and ACC across Quest and the
Pixel's IndexedDB. The PWA verified the ZIP checksums in the private Drive
folder, and independent Drive inventory reported one object for each run.
The first ZIP was 22,301,743 bytes and the second was 2,865,249 bytes.
A third 2,842,132-byte ZIP finalized with the Pixel controller in the
background. Drive still contained only two objects while the phone was
away. Reopening the installed PWA and renewing authorization caught up
automatically. The Android Drive app showed all three objects in the private
folder. Its independent download matched the Pixel ZIP SHA-256. Each
downloaded CSV, ECG and ACC entry matched the original on Quest. Android
Files opened the first downloaded CSV's file details, and the file hash
matched the Quest original. The bench Pixel has no CSV viewer.

The five-minute `probe-5` run reached its completion threshold on Quest 2
and saved with zero dropped writer records after signal pauses. Its internal
core exceeded the original 128 MiB export limit. The CSV, ECG and ACC
together totaled 131,155,067 bytes. The updated export gate validates a
larger bounded native core and applies the phone limit to the files delivered
to the Pixel. The installed Pixel committed a 131,158,666-byte ZIP and
its three data-file hashes matched Quest. A PWA reload during delivery
preserved the local ZIP and reserved Drive ID. After account renewal, the
queue confirmed the exact ZIP hash and the private folder contained one copy.

Build 0.1.26 saved a new dated v2 CSV with live H10 data and changing
renderer radius after Start, Pause, Resume and Finish. The CSV contained
41,820 data rows. The Pixel verified the Quest hash and persisted the single
7,167,072-byte CSV. A one-shot transport interruption on the Pixel blocked
the first media PUT before Drive received it. The PWA retained its pending
receipt and retried the same Drive ID automatically. Drive confirmed the
matching hash, and an independent inventory showed one private CSV.
The signed 0.1.27 release repeated the complete transfer with another
dated CSV and matching Quest, Pixel and Drive hashes. Cold H10 setup
required an operator ECG restart before the phone could start the test.
An independent download of the latest dated CSV in Android Drive matched
the Quest SHA-256 and is available in Android Downloads.

## Run identity and files

1. The operator enters participant details in Atmungsraum. Use the existing
   normalization in [data-logging.js](../data-logging.js) and the filename
   contract in [DATA-LOGGING.md](DATA-LOGGING.md).
2. For participant recording mode, bind the normalized ID to the authorized
   Arm request. Signed status advertises `participant_arm: 1` when Remote-only
   and recording export are selected. The enrolled LAN `participant-arm`
   action carries the existing v1 Arm fields, `participant_id`, and the
   observed native generation `g` in AES-GCM. Derive the request key using
   HMAC-SHA-256 over `RQG1-ARM-KEY|controller|device|epoch` with the enrolled
   key. Additional data is `RQG1-ARM|device|epoch|nonce`. Quest admits it
   against current idle VR preparation and live ECG, validates the ID and
   reserves the attempt index durably
   per participant and condition. It allocates a unique session UUID before
   writing samples. A participant edit applies to a later Arm.
3. Quest writes and finalizes `<participant_id>_<YYYY-MM-DD>_<scenario_id>_r<run>.csv`.
   For example, `Tes_30_N_2026-11-13_probe-5_r7.csv` contains the participant ID,
   the UTC Start date, the authored session name and attempt seven.
   Preserve actual H10 sample times, units,
   pauses and signal gaps. A finalized partial run retains its abort reason
   and reserved index. After an app restart, the next attempt gets a new UUID
   and an incremented index. A filename is for people. The native run ID is
   the sync key.
4. Expose the file only after its bytes and terminal metadata are durable.
   Keep its name, size, hash and run identity in the authenticated export
   manifest. Unfinished or invalid runs stay outside the eligible inventory.
   Keep the originals on Quest after transfer.

## Transfer

Quest CSV v2 is the only participant file the controller pulls for new runs.
The signed descriptor advertises `schema_version: 2`. After checking its
length and SHA-256, the controller saves the CSV and sends those same bytes
to Drive. Quest keeps its original checkpoint streams. The legacy ZIP
procedure below applies to earlier recordings.

Reuse the existing RQG1 LAN gateway. The enrolled controller verifies the
Quest identity and signed observations. Requests and replies bind the action,
nonce and gateway epoch with HMAC-SHA-256. Private recording responses use the
existing AES-GCM envelope. Bluetooth discovers the address and can provision
a Pixel Wi-Fi hotspot to Quest 2 through the enrolled setup characteristic.
The controller verifies the selected serial, then sends the hotspot credentials
using AES-GCM with a one-use GATT challenge and both Quest owner epochs.
Android owns the headset’s network request and reports its actual local address.
Wi-Fi carries
the data.

| Operation | Purpose |
| --- | --- |
| `recordings {query:"list", before:"", limit:16}` | Enumerate exportable finalized runs, newest first. Follow `next_before` until it is null. |
| `recordings {query:"describe", id}` | Pin the run and its manifest. Retry preparation of the same query within the existing deadline. |
| `recordings {query:"read", manifest_id, file, offset, length}` | Read up to 32 KiB. Verify the returned manifest, file, offset, length and final SHA-256 before saving. |
| `recordings {query:"live-describe"}` | Discover the current started participant run and its fsynced CSV prefix. |
| `recordings {query:"live-read", id, session_id, offset, length}` | Read up to 32 KiB within that committed prefix. The encrypted reply binds the run, offset, length and range SHA-256. |

Run one transfer per Quest from the browser profile. The finalized inventory
and read operations begin on enrolled join,
after the run returns to idle, and on an idle retry. Stop requesting recording
bytes as soon as the controller leaves idle or the operator starts a command.
Quest independently enforces idle Remote admission around recording I/O.
The live operations require the enrolled private LAN controller and the same
active generation and session before and after each read. The PWA commits up
to four sequential 32 KiB ranges per transaction. When behind, it drains the
current checkpoint in successive transactions, yielding between them so
controls can cancel the pass. It then asks for a new checkpoint.
Keep status and acquisition independent from sync.

Key each committed browser copy by the stable Quest identity and recording ID.
Skip committed IDs on the next inventory. Commit only complete verified files
in one IndexedDB transaction. Retry failed, interrupted or uncommitted files
on the next idle check. After browser restart, the controller tries to rejoin
the enrolled gateway. The same browser profile retains its copies across browser
and Quest app restarts. The operator can explicitly download a saved copy to
the phone filesystem.

## Implementation and acceptance

Legacy rehearsals sync as an eight-file JSON/JSONL archive. A participant
run advertises `participant_csv: true` in the list entry and includes a
`participant_csv` descriptor in the signed manifest. The descriptor binds
the name, size, SHA-256 and session UUID. For these runs the browser stores the
original CSV and a ZIP with the CSV, original ECG and ACC streams, and signed
export manifest. New participant files use a stem such as
`Tes_30_N_2026-11-13_probe-5_r4` for the CSV, ZIP and each exported sensor
stream and manifest when exporting a historical sensor ZIP. The date comes
from official Start in UTC. The signed v2 descriptor identifies the single
CSV, which is both the ordinary download and the Drive payload. Keep the
original native streams on Quest. The next eligible idle reconciliation
upgrades an older incomplete browser entry for the same recording ID.
Probe-Runde and Haupt-Runde have independent counters for each participant.
The authored scenario ID in each filename names the selected session. The
VR condition remains in the recording metadata. The app accepts previously
saved names and keeps their original hashes and delivery receipts.

On startup after an abrupt stop, Quest verifies the last durable checkpoint.
If the run reached official Start, Quest can publish an aborted CSV and the
durable prefixes of the original streams using the same recording ID and
session UUID. The signed `core8-recovered-checkpoint` export records the
checkpoint ordinal and digest, and the six stream digests. The original files,
including bytes after the checkpoint, remain on Quest. Recordings armed before
official Start produce no recovered CSV. Older participant manifests with no
authored offsets retain their ECG and playback times with an unknown phase.

The Chrome fixtures passed restart and catch-up with synthetic native data.
Physical acceptance requires the installed Android PWA, Meta Quest 2 and a
worn H10. Run the same synthetic participant and condition twice across
full app restarts. Check that the two CSVs have distinct session UUIDs and
increasing run indexes, observe a run finalized while the browser is absent,
then rejoin and confirm one copy per ID. Verify a partial run, the idle gate
and a CSV reopened from phone Files. Track evidence and open work in
[PROJECT.md](PROJECT.md) and [VERIFICATION.md](VERIFICATION.md).

## Google Drive delivery acceptance

### Bind the exact USB hardware pair

Pair the installed Atmungsraum PWA with the intended Quest and open it in
standalone mode on the Pixel. Connect both devices over USB and use
`adb devices -l` to identify their serials. From `atmungsraum/` run:

```powershell
node tools/configure-drive-pair.mjs <Quest-USB-serial> <Pixel-USB-serial> --check
node tools/configure-drive-pair.mjs <Quest-USB-serial> <Pixel-USB-serial>
```

For a serial-less commissioned Quest, add `--quest-device <verified-installation-id>`
from the headset's commissioning record. The tool compares that ID with the
selected phone's saved Quest pairing. It stores the same 24-character fingerprint
whenever the same two hardware serials are supplied. Repeat with the second
headset and phone. Keep the diagnostic output local. The default for a phone
without USB provisioning uses the authenticated Quest identity and a random
PWA installation ID.

Authorize the same OAuth client and research account when both pairs belong in
one Drive. A new phone reuses one matching app-visible private research root
and creates or reuses its own `Pair_<id>` child. If the account already has
several matching roots, resolve their identity before authorizing a new phone.
For the acceptance check, send two new synthetic or approved datasets, check
their distinct UTC folders, and compare the downloaded files to the phone.
See [the timestamp contract](DATA-LOGGING.md#dataset-folders-and-utc).

### Authorize the research account

The phone must automatically upload each newly saved Quest recording's
original `polar-ecg.jsonl` and `polar-acc.jsonl` bytes to the designated private
Drive folder. Retain the signed export manifest and the shared participant
identity and filename contract. Use one immutable recording identity to bind
the local copy, queued upload and Drive receipt. Keep Quest originals and the
phone's durable copy for recovery. Accept uploads only after Drive readback
matches the folder, file identity, size and content hash. Opening the Drive app
or saving a download on Android supplies no upload receipt.

Use per-file Google Drive authorization for the configured account and the
production Pages origin. Keep access tokens in memory. Google requires a user
gesture to renew an expired browser token. Persist pending work before network
access and resume on the next authorized foreground session. Report pending and
failed uploads separately from verified Drive copies. Keep the destination
private and prevent a different signed-in Google account from receiving a run.

Enable Drive API and register a Web application OAuth client in Google Cloud.
Register `https://viscereality.org` as an authorized JavaScript origin and
authorize the designated account for the `drive.file` scope. In VR-Probe,
join the enrolled Quest, open **Google Drive**, enter the public OAuth client ID
and intended account email, then choose **Google Drive verbinden** and confirm
the intended Google account. The app verifies the account and creates its
private **Atmungsraum – Quest-Aufnahmen** folder. It remembers the account,
client ID and folder ID in IndexedDB. Keep client secrets out of the browser.
See [Google's web client setup](https://developers.google.com/identity/oauth2/web/guides/get-google-api-clientid)
and [Drive scope reference](https://developers.google.com/workspace/drive/api/guides/api-specific-auth).

The Drive queue reads already-committed bundles while the controller is idle.
An IndexedDB receipt reserves a Drive ID before upload and records the exact
file hash. On an uncertain response, retry that ID and verify the existing
object instead of creating another one. The controller checks the destination
is still private, verifies the Drive object's parent, owner, name, size and
SHA-256, then stores the confirmed receipt. CSV v2 through v4 upload the original
CSV. Earlier participant uploads keep the ZIP
with the CSV, ECG, ACC and export manifest. Legacy local downloads retain the
eight-file archive. Start, close,
and background transitions stop upload work. A later authorized idle pass
reconciles pending files. Google Drive on Android can display the resulting
folder independently of the browser.

Run the following with synthetic participant identities before study use:

1. On the installed Pixel PWA, join the Quest 2 and connect the worn H10. Start, pause,
   resume and finish. Verify the native receipt, nonempty ECG and ACC samples
   with native timestamps and units, then local save and matching Drive bytes.
2. Restart both apps and repeat the same condition. Check independent run UUIDs,
   increasing attempt indices and one Drive object per run. Finish an additional
   run with the phone absent, then rejoin and verify automatic catch-up.
3. Repeat with a finalized partial run and with larger recordings spanning
   multiple transfer chunks. Corrupt an export chunk in the fixture and verify
   rejection before local commit or upload.
4. Interrupt the network before and after Drive accepts an upload. Test process
   close, concurrent tabs, expired authorization, quota and permission failures,
   wrong account, and a recording starting during catch-up. Verify pending data
   survives, accepted uploads are reconciled without duplicates, and the Quest
   control path stays responsive.
5. Reopen the actual Drive folder on the Pixel and download a saved bundle for
   independent hash and contents verification. Record the tested build, elapsed
   durations, run counts and failures without participant data or credentials.

## Check one Haupt-Runde evidence bundle

Collect a private copy of the native run directory after the writer has
settled. Keep the authenticated recording export manifest, the committed
Pixel CSV and a separately downloaded Drive CSV. For a recovered run, retain
the native recovery receipt and its checkpoint file along with the original
streams. Use the exact run directory name that the Quest returned.

Get an independent Drive listing filtered by the exact CSV name and its
designated private parent, with trashed files excluded. Keep the full listing
response, including any continuation token, and the verified Pixel delivery
receipt. Verify the folder identity and privacy before running the check.
Then run the read-only verifier from `atmungsraum` with Python 3.12 or newer.

```powershell
python -B tools/check-recording-evidence.py `
  --expect complete --native-run '<native run directory>' `
  --export-manifest '<authenticated export manifest JSON>' `
  --pixel-csv '<committed Pixel CSV>' --drive-csv '<downloaded Drive CSV>' `
  --drive-list '<independent Drive listing JSON>' `
  --drive-receipt '<verified Pixel delivery receipt JSON>' `
  --drive-folder '<designated private Drive folder ID>'
```

Use `--expect partial` for a separately planned early finish or checkpoint
recovery. The checker compares the committed ECG, accelerometer, radius and
breath observations with their CSV rows and clock fields. It also checks
the CSV playback transitions and terminal playback fields against the
committed native event order, pause intervals and retained phase offsets.
It verifies native file and checkpoint hashes, the terminal outcome,
writer drops, matching CSV bytes from all three copies, and one verified
Drive object under the reserved file ID. A completed run must have exactly
one committed native completion
event between official Start and finalization, on the same clock anchor.
A partial run has no committed completion event. The checker reports counts,
hashes, and native elapsed time to the durable completion marker when present.

Keep the actual AudioTrack 39,690,000-frame receipt at 44,100 Hz, wearer and
input evidence, signal/fault window notes, and the headset/phone release
receipts with the run. The terminal CSV audio value is a projection. Verify
the actual audio clock and measured sample admission at the main boundaries
on the headset. For a clean run, inspect the interruption row counts and
correlate any gaps with the native sensor and clock traces.
