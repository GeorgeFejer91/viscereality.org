"""Exercise the collection verifier at its CLI with temporary synthetic evidence."""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC as UTC_ZONE, datetime, timedelta
from pathlib import Path


TOOL = Path(__file__).with_name("check-recording-evidence.py")
START = 2_000_000_000
UTC = 1_700_000_000_000_000_000
RUN = "utc-ns-01700000000000000000"
SESSION = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
NAME = "Tes_30_N_2023-11-14_haupt-15_r1.csv"
NATIVE_SCHEMA = "rusty.quest.native_renderer.session_recording_"
HEADER = """schema_version,participant_id,session_id,run_index,scenario_id,record_type,timestamp_utc,elapsed_us,audio_elapsed_us,playback_state,audio_phase,sensor_time_ns,sensor_frame_sequence,sensor_sample_index,ecg_uv,acc_x_mg,acc_y_mg,acc_z_mg,render_frame,radius_m,breathing_phase,session_status,end_reason,signal_reason""".split(",")


def digest(data):
    return hashlib.sha256(data).hexdigest()


class CollectionEvidenceTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.evidence = Path(scratch.name)
        self.native = self.evidence / RUN
        self.native.mkdir()
        self.rows = [
            dict(record_type="official-start", audio_elapsed_us=0, playback_state="waiting", audio_phase="waiting"),
            dict(record_type="ecg", sensor_time_ns=5000, sensor_frame_sequence=1, sensor_sample_index=0, ecg_uv=-37),
            dict(record_type="accelerometer", sensor_time_ns=5000, sensor_frame_sequence=1, sensor_sample_index=0,
                 acc_x_mg=1.5, acc_y_mg=-2, acc_z_mg=1000),
            dict(record_type="radius", render_frame=6, radius_m=0.8),
            dict(record_type="breath", breathing_phase="hold"),
            dict(record_type="end", elapsed_us=900000000, timestamp_utc="2023-11-14T22:28:20.000000Z",
                 audio_elapsed_us=900000000, playback_state="ended", audio_phase="ended",
                 session_status="completed", end_reason="completed"),
        ]
        self.drive = dict(id="drive-file", name=NAME, parents=["private-folder"], ownedByMe=True,
                          shared=False, trashed=False, mimeType="text/csv")
        self.publish()

    def json_file(self, path, body):
        path.write_text(json.dumps(body), encoding="utf-8")

    def streams(self, file, kind, records):
        target = self.native / file
        with target.open("w", encoding="utf-8", newline="\n") as out:
            for n, row in enumerate(records, 1):
                base = dict(schema=NATIVE_SCHEMA + "row.v1", record_sequence=n, kind=kind,
                            session_started_at_utc=RUN, condition="condition-a",
                            observed_monotonic_ns=START, observed_utc_ns=UTC)
                print(json.dumps({**base, **row}), file=out)
        return target.stat().st_size

    def publish(self, partial=False, duplicate=False, old_projection=False, polar_setup=None,
                include_completion=True, completion_offset_ns=0, csv_audio_start=None, pause=False, extra_events=(),
                continuous_signals=False):
        identity = dict(participant_id="Tes_30_N", session_id=SESSION, scenario_id="haupt-15", run_index=1)
        phases = [[0, "instruction_1"], [1_000_000, "instruction_2"]] if pause else [[0, "instruction_1"]]
        participant = dict(identity, planned_duration_seconds=900, phase_offsets_ns=phases)
        end_ns = START + (3_000_000 if partial else 900_000_000_000)
        rows = [row.copy() for row in self.rows if not (old_projection and row["record_type"] == "breath")]
        clocks = dict(source_clock="polar-sensor", source_time_ns=5000, sample_index=0, source_sequence=1,
                      frame_sequence=1, received_source_clock="java-nano-time", received_source_time_ns=7000)
        ecg = [dict(clocks, microvolts=-37)]
        acc = [dict(clocks, xyz_mg=[1.5, -2, 1000])]
        if continuous_signals:
            start_utc = datetime.fromtimestamp(UTC // 1_000_000_000, UTC_ZONE)
            for seconds in range(1, 900):
                delta = seconds * 1_000_000_000
                stamp = dict(observed_monotonic_ns=START + delta, observed_utc_ns=UTC + delta)
                frame = dict(clocks, frame_sequence=seconds + 1, source_sequence=seconds + 1,
                             source_time_ns=delta + 5_000, received_source_time_ns=delta + 7_000)
                ecg.append(dict(frame, **stamp, microvolts=-37))
                acc.append(dict(frame, **stamp, xyz_mg=[1.5, -2, 1000]))
                when = (start_utc + timedelta(seconds=seconds)).isoformat(timespec="microseconds").replace("+00:00", "Z")
                for row in self.rows[1:3]:
                    rows.insert(-1, dict(row, elapsed_us=seconds * 1_000_000, timestamp_utc=when,
                                         sensor_time_ns=delta + 5_000, sensor_frame_sequence=seconds + 1))
        if partial:
            rows[-1].update(elapsed_us=3000, timestamp_utc="2023-11-14T22:13:20.003000Z",
                            audio_elapsed_us=3000, playback_state="playing", audio_phase="instruction_1",
                            session_status="aborted", end_reason="interrupted-recovery")
        events = [dict(event="official-start"), dict(event="audio-started")]
        if csv_audio_start is not False:
            rows.insert(1, dict(record_type="audio-started", audio_elapsed_us=0,
                                playback_state="playing", audio_phase="instruction_1") | (csv_audio_start or {}))
        if pause:
            for event, elapsed, audio, state, phase, stamp in (
                ("experiment-paused", 1000, 1000, "paused", "instruction_2", "2023-11-14T22:13:20.001000Z"),
                ("experiment-resumed", 2000, 1000, "playing", "instruction_2", "2023-11-14T22:13:20.002000Z"),
                ("audio-ended", 900_000_000, 899_999_000, "ended", "ended", "2023-11-14T22:28:20.000000Z"),
            ):
                events.append(dict(event=event, observed_monotonic_ns=START + elapsed * 1000,
                                   observed_utc_ns=UTC + elapsed * 1000))
                rows.insert(-1, dict(record_type=event, elapsed_us=elapsed, timestamp_utc=stamp,
                                     audio_elapsed_us=audio, playback_state=state, audio_phase=phase))
            events.insert(3, dict(event="audio-started", observed_monotonic_ns=START + 1_500_000,
                                  observed_utc_ns=UTC + 1_500_000))
            rows[-1]["audio_elapsed_us"] = 899_999_000
        filename = ".recovered-" + NAME if partial else NAME
        with (self.native / filename).open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, HEADER)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(schema_version=4, **identity) |
                                dict(elapsed_us=0, timestamp_utc="2023-11-14T22:13:20.000000Z") | row)
        original = (self.native / filename).read_bytes()
        for copy in ("pixel.csv", "drive.csv"):
            (self.evidence / copy).write_bytes(original)
        desc = dict(schema_version=4, name=NAME, session_id=SESSION, size=len(original), sha256=digest(original))
        clock = "java-nano-time" if polar_setup is not None else "openxr-time"
        if not partial and include_completion:
            completed_at = end_ns + completion_offset_ns
            events.append(dict(event="completion-reached", observed_monotonic_ns=completed_at,
                               observed_utc_ns=UTC + completed_at - START))
        events.extend(extra_events)
        lengths = {
            "events.jsonl": self.streams("events.jsonl", "event", events),
            "polar-ecg.jsonl": self.streams("polar-ecg.jsonl", "polar-ecg", ecg),
            "polar-acc.jsonl": self.streams("polar-acc.jsonl", "polar-acc", acc),
            "polar-hr-rr.jsonl": self.streams("polar-hr-rr.jsonl", "polar-hr-rr", []),
            "radius.jsonl": self.streams("radius.jsonl", "effective-radius-snapshot",
                                        [dict(source_frame=6, resulting_radius_m=0.8)]),
            "breath.jsonl": self.streams("breath.jsonl", "breath", [dict(
                assessment_source="controller" if polar_setup is None else "polar-acc",
                sampled_clock=clock, sampled_time_ns=8, observed_source_clock=clock,
                observed_source_time_ns=9, unbiased_phase="hold", phase="inhale")]),
        }
        manifest = dict(schema=NATIVE_SCHEMA + "manifest.v1", participant=participant, started_at_utc=RUN,
                        participant_csv_schema=4, clock_anchor=dict(monotonic_clock="android-elapsed-realtime",
                        monotonic_ns=START, utc_epoch_ns=UTC), completion_threshold_ns=900_000_000_000)
        if not old_projection:
            manifest["participant_csv_controller_breath"] = True
        if polar_setup:
            manifest["breath_tracking_configuration"] = dict(
                schema="rusty.quest.breathing_setup_tracking_configuration.v1", source="polar-acc",
                sampled_clock="java-nano-time", raw_sampled_clock="java-nano-time",
            )
        self.json_file(self.native / "manifest.json", manifest)
        terminal = dict(schema=NATIVE_SCHEMA + "final.v1", participant=participant, started_at_utc=RUN,
                        completed=not partial, saved=True, dropped_records=0, loss=dict(dropped_records=0),
                        durable_stream_lengths=lengths, ended_monotonic_ns=end_ns, participant_csv=desc,
                        completion_threshold_ns=900_000_000_000, stop_reason="interrupted-recovery" if partial else "stop")
        terminal_name = "participant-recovered.json" if partial else "final.json"
        recovery = {}
        if partial:
            del terminal["completion_threshold_ns"]
            checkpoint = self.native / "checkpoint-00000000000000000001.json"
            self.json_file(checkpoint, dict(schema=NATIVE_SCHEMA + "checkpoint.v1", checkpoint_ordinal=1,
                                           durable_stream_lengths=lengths))
            recovery = dict(v=1, checkpoint_ordinal=1, checkpoint_sha256=digest(checkpoint.read_bytes()),
                            stream_sha256={name: digest((self.native / name).read_bytes()) for name in lengths})
            terminal["recovery"] = recovery
        self.json_file(self.native / terminal_name, terminal)
        entries = []
        for name in ("manifest.json", "final.json", *lengths):
            data = (self.native / (terminal_name if name == "final.json" else name)).read_bytes()
            entries.append(dict(name=name, size=len(data), sha256=digest(data)))
        if partial:
            with (self.native / "polar-ecg.jsonl").open("ab") as stream:
                stream.write(b"uncommitted-checkpoint-tail")
        export = dict(v=1, schema="rusty.quest.recording_export.v1", id=RUN, saved=True, files=entries,
                      participant_csv=desc, disposition="interrupted-early" if partial else "finalized-completed",
                      file_set_policy="core8-recovered-checkpoint" if partial else "core8-original-no-checkpoints-or-index")
        if partial:
            export["recovery"] = recovery
        self.json_file(self.evidence / "export.json", export)
        drive = dict(self.drive, size=str(desc["size"]), sha256Checksum=desc["sha256"])
        drive_rows = [drive, dict(drive, id="new-reservation")] if duplicate else [drive]
        self.json_file(self.evidence / "drive-list.json", {"files": drive_rows})
        self.json_file(self.evidence / "delivery.json", dict(folderId="private-folder", driveId="drive-file",
                                                           sha256=desc["sha256"], verified=True))

    def invoke(self, mode="complete", fail=None, *, uninterrupted=False):
        result = subprocess.run([
            sys.executable, "-B", str(TOOL), "--native-run", str(self.native),
            "--export-manifest", str(self.evidence / "export.json"),
            "--pixel-csv", str(self.evidence / "pixel.csv"), "--drive-csv", str(self.evidence / "drive.csv"),
            "--drive-list", str(self.evidence / "drive-list.json"), "--drive-receipt", str(self.evidence / "delivery.json"),
            "--drive-folder", "private-folder", "--expect", mode,
            *(["--require-uninterrupted"] if uninterrupted else []),
        ], capture_output=True, text=True)
        if fail:
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(fail, result.stderr)
            self.assertNotIn(NAME, result.stderr)
            return
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(SESSION, result.stdout)
        result = json.loads(result.stdout)
        self.assertGreater(result["csv_rows"]["ecg"], 0)
        self.assertGreater(result["csv_rows"]["accelerometer"], 0)
        self.assertEqual(result["writer_drops"], 0)
        return result

    def test_original_and_recovered_exports_match(self):
        complete = self.invoke()
        self.assertEqual(complete["csv_rows"]["breath"], 1)
        self.assertEqual(complete["native_start_to_completion_us"], 900_000_000)
        self.publish(partial=True, old_projection=True)
        result = self.invoke("partial")
        self.assertIsNone(result["native_start_to_completion_us"])
        self.assertEqual(result["ignored_post_checkpoint_bytes"], len(b"uncommitted-checkpoint-tail"))
        self.assertEqual(result["csv_rows"].get("breath", 0), 0)

    def test_complete_requires_a_durable_completion_event(self):
        self.publish(include_completion=False)
        self.invoke(fail="native completion event disagrees with the terminal outcome")

    def test_completion_event_must_be_inside_the_run(self):
        self.publish(completion_offset_ns=1)
        self.invoke(fail="native completion event is outside the run or has the wrong clock")

    def test_missing_playback_transition_fails_even_when_all_copies_match(self):
        self.publish(csv_audio_start=False)
        self.invoke(fail="CSV playback transition or clock mismatch")

    def test_corrupted_playback_transition_fields_fail(self):
        for update in (
            {"record_type": "experiment-paused"},
            {"timestamp_utc": "2023-11-14T22:13:20.001000Z"},
            {"audio_elapsed_us": 1000},
            {"playback_state": "paused"},
            {"audio_phase": "ended"},
        ):
            with self.subTest(update=update):
                self.publish(csv_audio_start=update)
                self.invoke(fail="CSV playback transition or clock mismatch")

    def test_paused_time_is_excluded_from_the_playback_projection(self):
        self.publish(pause=True)
        self.assertEqual(self.invoke()["terminal_csv_audio_elapsed_us"], 899_999_000)
        self.rows[-1]["audio_elapsed_us"] = 0
        self.publish()
        self.invoke(fail="CSV playback transition or clock mismatch")

    def test_hashes_do_not_excuse_a_missing_or_changed_projection(self):
        self.rows[1]["ecg_uv"] = -36
        self.publish()
        self.invoke(fail="ecg projection mismatch")
        self.rows[1]["ecg_uv"] = -37
        self.rows.pop(4)
        self.publish()
        self.invoke(fail="missing breath row")

    def test_projection_clock_must_agree_even_when_all_copies_match(self):
        self.rows[1]["elapsed_us"] = 1
        self.publish()
        self.invoke(fail="ecg projected observation clock mismatch")

    def test_duplicate_drive_objects_and_changed_pixel_copy_fail(self):
        self.publish(duplicate=True)
        self.invoke(fail="expected one Drive object")
        self.publish()
        (self.evidence / "pixel.csv").write_bytes(b"changed")
        self.invoke(fail="Pixel CSV: byte count or SHA-256 mismatch")

    def test_incomplete_drive_search_never_proves_uniqueness(self):
        inventory_path = self.evidence / "drive-list.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        self.json_file(inventory_path, dict(inventory, incompleteSearch=True))
        self.invoke(fail="expected one Drive object in the complete inventory")

    def test_uninterrupted_mode_requires_no_pause_or_missing_breath_projection(self):
        self.invoke(fail="uninterrupted run contains an ECG receipt gap", uninterrupted=True)
        self.publish(continuous_signals=True)
        self.assertTrue(self.invoke(uninterrupted=True)["uninterrupted_checks"])
        self.publish(pause=True)
        self.invoke()
        self.invoke(fail="uninterrupted run contains pauses or faults", uninterrupted=True)
        self.publish(old_projection=True)
        self.invoke()
        self.invoke(fail="uninterrupted run has no breathing observations", uninterrupted=True)

    def test_signal_interruption_requires_native_clock_and_fails_uninterrupted_mode(self):
        reason = "no-recorded-receipt-for-one-second"
        self.rows.insert(-1, dict(record_type="ecg-interrupted", signal_reason=reason))
        event = dict(event="ecg-interrupted", signal_reason=reason)
        self.publish(extra_events=[event])
        self.invoke()
        self.invoke(fail="uninterrupted run contains pauses or faults", uninterrupted=True)
        self.publish(extra_events=[dict(event, observed_utc_ns=UTC + 1)])
        self.invoke(fail="native signal event clock mismatch")

    def test_polar_java_clock_requires_retained_setup_configuration(self):
        self.publish(polar_setup=True)
        self.assertEqual(self.invoke()["csv_rows"]["breath"], 1)
        self.publish(polar_setup=False)
        self.invoke(fail="native breath clock provenance mismatch")


if __name__ == "__main__":
    unittest.main()
