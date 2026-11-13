# Quest, Pixel and Google Drive recording sync

Quest owns acquisition, timing, participant-run identity, and the original
recording. Atmungsraum on the phone pulls finished files from the enrolled
Quest over trusted Wi-Fi. An idle pull can start before the first rehearsal or
after any rehearsal and catches up on files created while the controller was
closed. The installed Pixel PWA is the study sync controller. Its verified local
copy is the input to automatic delivery to the research Google Drive.

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

For the paired performance trial, open the controller with
`?live-sync=off` for the baseline, then use the ordinary URL for the live
run. Both URLs preserve local recording and the existing post-session sync.
The baseline setting disables active phone transfer and Drive upload on that
controller. The Quest CSV projection is active in both conditions.

See the [active checklist](PROJECT.md#live-recording-sync-to-google-drive)
and [Drive resumable upload documentation](https://developers.google.com/workspace/drive/api/guides/manage-uploads).

## Operator procedure

1. Open the installed Atmungsraum icon on the Pixel 10 and go to **VR-Probe**.
   On a fresh Quest profile, press **B** three times without holding the grip
   in Remote mode to open the headset's maintenance panel. Under **Network
   control**, select **Approve trusted LAN controller** on the trusted test
   network. Tap **Quest verbinden**, select the Quest in Chrome and enter the
   Controller ID and Quest key from the headset once. If using the Pixel
   hotspot, turn it on and enter its name and password under
   **Smartphone-Hotspot verwenden**. Approve Android Nearby devices, hotspot
   join on Quest, and Chrome Local Network Access when prompted. Close the
   maintenance panel to return to the immersive scene before starting.
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
   then verify that the new ZIP is in the designated folder. For qualification,
   compare the SHA-256 and manifest with the copies on the Pixel and Quest.
5. After an interrupted connection, reopen the app while Quest is in Remote
   mode. The saved pairing opens VR-Probe and tries the last known address.
   If it changed, supported browsers can rediscover the Quest using its
   saved Bluetooth grant. Use **Erneut verbinden** if Chrome needs the chooser
   again or Local Network Access has to be granted.
   Use **Jetzt abgleichen** if needed. If Google authorization expired,
   reconnect Drive. Confirm delivery before relying on the Drive copy.

## Bench verification on Quest 2 and Pixel 10

The signed Quest rehearsals start in Remote. The installed Pixel Chrome
157.0.8103.50 WebAPK discovers it over Bluetooth, then opens its authenticated
Wi-Fi gateway. Close any floating overlays if Chrome blocks Local Network
Access. A Quest APK update that changes the trusted build fingerprint
requires the operator to set up the connection again. A restart on the same
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
active generation and session before and after each read. The PWA issues at
most four sequential 32 KiB ranges per live pass. Controls can abort a pass.
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
SHA-256, then stores the confirmed receipt. CSV v2 and v3 upload the original
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
