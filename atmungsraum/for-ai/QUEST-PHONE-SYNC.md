# Quest-to-phone recording sync

Quest owns acquisition, timing, participant-run identity, and the original
recording. Atmungsraum on the phone pulls finished files from the enrolled
Quest over trusted Wi-Fi. An idle pull can start before the first rehearsal or
after any rehearsal and catches up on files created while the controller was
closed.

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
3. Quest writes and finalizes `<participant_id>_<condition><run>.csv`, for
   example `Tes_30_N_probe2.csv`. Preserve actual H10 sample times, units,
   pauses and signal gaps. A finalized partial run retains its abort reason
   and reserved index. After an app restart, the next attempt gets a new UUID
   and an incremented index. A filename is for people. The native run ID is
   the sync key.
4. Expose the file only after its bytes and terminal metadata are durable.
   Keep its name, size, hash and run identity in the authenticated export
   manifest. Unfinished or invalid runs stay outside the eligible inventory.
   Keep the originals on Quest after transfer.

## Transfer

Reuse the existing RQG1 LAN gateway. The enrolled controller verifies the
Quest identity and signed observations. Requests and replies bind the action,
nonce and gateway epoch with HMAC-SHA-256. Private recording responses use the
existing AES-GCM envelope. Bluetooth discovers the address. Wi-Fi carries
the data.

| Operation | Purpose |
| --- | --- |
| `recordings {query:"list", before:"", limit:16}` | Enumerate exportable finalized runs, newest first. Follow `next_before` until it is null. |
| `recordings {query:"describe", id}` | Pin the run and its manifest. Retry preparation of the same query within the existing deadline. |
| `recordings {query:"read", manifest_id, file, offset, length}` | Read up to 32 KiB. Verify the returned manifest, file, offset, length and final SHA-256 before saving. |

Run one transfer per Quest from the browser profile. Begin on enrolled join,
after the run returns to idle, and on an idle retry. Stop requesting recording
bytes as soon as the controller leaves idle or the operator starts a command.
Quest independently enforces idle Remote admission around recording I/O.
Keep status and acquisition independent from sync.

Key each committed browser copy by the stable Quest identity and recording ID.
Skip committed IDs on the next inventory. Commit only complete verified files
in one IndexedDB transaction. Retry failed, interrupted or uncommitted files
on the next idle check. A browser restart requires rejoining the enrolled
gateway. The same browser profile retains its committed copies across browser
and Quest app restarts. The operator can explicitly download a saved copy to
the phone filesystem.

## Implementation and acceptance

Legacy rehearsals sync as an eight-file JSON/JSONL archive. A participant
run advertises `participant_csv: true` in the list entry and includes a
`participant_csv` descriptor in the signed manifest. The descriptor binds
the name, size, SHA-256 and session UUID. For these runs the browser stores the
original CSV and a ZIP with the CSV, original ECG and ACC streams, and signed
export manifest. The ordinary download still yields the participant CSV. The
next eligible idle reconciliation upgrades an older CSV-only entry for the
same recording ID. Keep the original streams on Quest.
The writer, export and encrypted Arm from the operator form have host
evidence. Both current rehearsal tracks use `probe` for file indexing.
Qualification of the matched APK and physical phone remains open.

The Chrome fixtures passed restart and catch-up with synthetic native data.
Physical acceptance requires an installed Android controller and a Quest with
a worn H10. Run the same synthetic participant and condition twice across
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

Run the following with synthetic participant identities before study use:

1. On the installed Pixel PWA, join the worn H10 and actual Quest. Start, pause,
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
