"""Exercise the collection verifier at its CLI with temporary synthetic evidence."""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
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
                include_completion=True, completion_offset_ns=0):
        identity = dict(participant_id="Tes_30_N", session_id=SESSION, scenario_id="haupt-15", run_index=1)
        end_ns = START + (3_000_000 if partial else 900_000_000_000)
        rows = [row.copy() for row in self.rows if not (old_projection and row["record_type"] == "breath")]
        if partial:
            rows[-1].update(elapsed_us=3000, timestamp_utc="2023-11-14T22:13:20.003000Z",
                            audio_elapsed_us=3000, session_status="aborted", end_reason="interrupted-recovery")
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
        clocks = dict(source_clock="polar-sensor", source_time_ns=5000, sample_index=0, source_sequence=1,
                      frame_sequence=1, received_source_clock="java-nano-time", received_source_time_ns=7000)
        clock = "java-nano-time" if polar_setup is not None else "openxr-time"
        events = [dict(event="official-start")]
        if not partial and include_completion:
            completed_at = end_ns + completion_offset_ns
            events.append(dict(event="completion-reached", observed_monotonic_ns=completed_at,
                               observed_utc_ns=UTC + completed_at - START))
        lengths = {
            "events.jsonl": self.streams("events.jsonl", "event", events),
            "polar-ecg.jsonl": self.streams("polar-ecg.jsonl", "polar-ecg", [dict(clocks, microvolts=-37)]),
            "polar-acc.jsonl": self.streams("polar-acc.jsonl", "polar-acc", [dict(clocks, xyz_mg=[1.5, -2, 1000])]),
            "polar-hr-rr.jsonl": self.streams("polar-hr-rr.jsonl", "polar-hr-rr", []),
            "radius.jsonl": self.streams("radius.jsonl", "effective-radius-snapshot",
                                        [dict(source_frame=6, resulting_radius_m=0.8)]),
            "breath.jsonl": self.streams("breath.jsonl", "breath", [dict(
                assessment_source="controller" if polar_setup is None else "polar-acc",
                sampled_clock=clock, sampled_time_ns=8, observed_source_clock=clock,
                observed_source_time_ns=9, unbiased_phase="hold", phase="inhale")]),
        }
        manifest = dict(schema=NATIVE_SCHEMA + "manifest.v1", participant=identity, started_at_utc=RUN,
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
        terminal = dict(schema=NATIVE_SCHEMA + "final.v1", participant=identity, started_at_utc=RUN,
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

    def invoke(self, mode="complete", fail=None):
        result = subprocess.run([
            sys.executable, "-B", str(TOOL), "--native-run", str(self.native),
            "--export-manifest", str(self.evidence / "export.json"),
            "--pixel-csv", str(self.evidence / "pixel.csv"), "--drive-csv", str(self.evidence / "drive.csv"),
            "--drive-list", str(self.evidence / "drive-list.json"), "--drive-receipt", str(self.evidence / "delivery.json"),
            "--drive-folder", "private-folder", "--expect", mode,
        ], capture_output=True, text=True)
        if fail:
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(fail, result.stderr)
            self.assertNotIn(NAME, result.stderr)
            return
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(SESSION, result.stdout)
        result = json.loads(result.stdout)
        self.assertEqual(result["csv_rows"]["ecg"], 1)
        self.assertEqual(result["csv_rows"]["accelerometer"], 1)
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

    def test_polar_java_clock_requires_retained_setup_configuration(self):
        self.publish(polar_setup=True)
        self.assertEqual(self.invoke()["csv_rows"]["breath"], 1)
        self.publish(polar_setup=False)
        self.invoke(fail="native breath clock provenance mismatch")


if __name__ == "__main__":
    unittest.main()
