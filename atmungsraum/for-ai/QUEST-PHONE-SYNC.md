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
   Arm request. Send it in an encrypted enrolled-only, generation-bound
   payload. Quest validates the ID and reserves the attempt index durably
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
the name, size, SHA-256 and session UUID. For these runs the browser
fetches only the CSV, replacing a stored legacy archive for the same native
recording ID after verification. The original streams remain on Quest.
The writer and export have host evidence. Encrypted Arm identity from
the operator form and qualification of the matched APK remain open.

The Chrome fixtures passed restart and catch-up with synthetic native data.
Physical acceptance requires an installed Android controller and a Quest with
a worn H10. Run the same synthetic participant and condition twice across
full app restarts. Check that the two CSVs have distinct session UUIDs and
increasing run indexes, observe a run finalized while the browser is absent,
then rejoin and confirm one copy per ID. Verify a partial run, the idle gate
and a CSV reopened from phone Files. Track evidence and open work in
[PROJECT.md](PROJECT.md) and [VERIFICATION.md](VERIFICATION.md).
