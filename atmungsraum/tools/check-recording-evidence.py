"""Verify one saved Haupt-Runde evidence bundle without printing participant data."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import UUID


NATIVE_SCHEMA = "rusty.quest.native_renderer.session_recording_"
SCENARIO = "haupt-15"
END_US = 900_000_000
STREAMS = {
    "events.jsonl", "polar-hr-rr.jsonl", "polar-ecg.jsonl", "polar-acc.jsonl", "breath.jsonl", "radius.jsonl",
}
PROJECTED = {
    "ecg": ("polar-ecg.jsonl", "polar-ecg", ("ecg_uv", "microvolts")),
    "accelerometer": ("polar-acc.jsonl", "polar-acc", ()),
    "radius": ("radius.jsonl", "effective-radius-snapshot", ("render_frame", "source_frame")),
    "breath": ("breath.jsonl", "breath", ("breathing_phase", "unbiased_phase")),
}
SENSOR_FIELDS = dict(sensor_time_ns="source_time_ns", sensor_frame_sequence="source_sequence",
                     sensor_sample_index="sample_index")
EVENTS = {
    "official-start", "audio-started", "experiment-paused", "experiment-resumed", "audio-ended", "audio-error",
    "ecg-interrupted", "ecg-resumed", "acc-interrupted", "acc-resumed",
}
SIGNAL_EVENTS = {"ecg-interrupted", "ecg-resumed", "acc-interrupted", "acc-resumed"}
HEADER = """schema_version,participant_id,session_id,run_index,scenario_id,record_type,timestamp_utc,elapsed_us,audio_elapsed_us,playback_state,audio_phase,sensor_time_ns,sensor_frame_sequence,sensor_sample_index,ecg_uv,acc_x_mg,acc_y_mg,acc_z_mg,render_frame,radius_m,breathing_phase,session_status,end_reason,signal_reason""".split(",")


class EvidenceError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def read_json(path):
    require(path.stat().st_size <= 1024 * 1024, "evidence JSON exceeds size limit")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_bytes(path, descriptor, label, allow_tail=False):
    length = descriptor["size"]
    actual = path.stat().st_size
    error = f"{label}: byte count or SHA-256 mismatch"
    require(type(length) is int and 0 <= length <= actual and (allow_tail or length == actual), error)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        remaining = length
        while remaining:
            block = source.read(min(remaining, 1024 * 1024))
            require(block, error)
            digest.update(block)
            remaining -= len(block)
    require(digest.hexdigest() == descriptor["sha256"], error)
    return actual - length


def native_rows(path, bound, run):
    with path.open("rb") as source:
        while bound:
            line = source.readline(bound)
            require(line.endswith(b"\n"), "committed native stream ends inside a row")
            bound -= len(line)
            row = json.loads(line, parse_float=Decimal)
            require(row["schema"] == NATIVE_SCHEMA + "row.v1" and row["session_started_at_utc"] == run,
                    "native row identity mismatch")
            yield row


def iso_utc(microseconds):
    return (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=microseconds)).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def projected_rows(path, bound, run, wanted_kind, start, end, allow_controller):
    for row in native_rows(path, bound, run):
        require(row["kind"] == wanted_kind, "native stream kind mismatch")
        at = row["observed_monotonic_ns"]
        if not start <= at <= end:
            continue
        if wanted_kind == "breath":
            allowed = {"polar-acc", "controller"} if allow_controller else {"polar-acc"}
            if row.get("assessment_source") not in allowed:
                continue
        yield row


def check_measurement(kind, actual, native, setup_polar_clock):
    require(native is not None, f"unexpected {kind} row")
    _, _, primary = PROJECTED[kind]
    fields = dict([primary]) if primary else {}
    precision = {}
    if kind in {"ecg", "accelerometer"}:
        require(native["source_clock"] == "polar-sensor" and native["received_source_clock"] == "java-nano-time"
                and native["received_source_time_ns"] > 0 and native["frame_sequence"] > 0,
                f"{kind} native clock provenance mismatch")
        fields.update(SENSOR_FIELDS)
    if kind == "accelerometer":
        axes = native["xyz_mg"]
        require(len(axes) == 3, "native ACC axis count mismatch")
        precision.update(zip(("acc_x_mg", "acc_y_mg", "acc_z_mg"), axes))
    elif kind == "radius":
        require(native["resulting_radius_m"] > 0, "native radius is invalid")
        precision["radius_m"] = native["resulting_radius_m"]
    elif kind == "breath":
        allowed_clock = {"openxr-time"}
        if setup_polar_clock and native["assessment_source"] == "polar-acc":
            allowed_clock.add("java-nano-time")
        require(native["sampled_clock"] in allowed_clock and native["sampled_clock"] == native["observed_source_clock"]
                and native["sampled_time_ns"] > 0 and native["observed_source_time_ns"] >= native["sampled_time_ns"],
                "native breath clock provenance mismatch")
    require(all(actual[key] == str(native[value]) for key, value in fields.items())
            and all(Decimal(actual[key]) == value for key, value in precision.items()),
            f"{kind} projection mismatch")
    for key in HEADER[11:21]:
        require(key in fields or key in precision or actual[key] == "", f"unexpected {kind} measurement column")


def verify_csv(path, native_directory, run, manifest, terminal, expected):
    identity = manifest["participant"]
    require(identity == terminal["participant"] and identity["scenario_id"] == SCENARIO
            and UUID(identity["session_id"]).version == 4 and identity["run_index"] >= 1,
            "participant run identity mismatch")
    clock = manifest["clock_anchor"]
    require(clock["monotonic_clock"] == "android-elapsed-realtime", "native clock anchor mismatch")
    lengths = terminal["durable_stream_lengths"]
    events = list(native_rows(native_directory / "events.jsonl", lengths["events.jsonl"], run))
    starts = [row["observed_monotonic_ns"] for row in events if row["event"] == "official-start"]
    require(len(starts) == 1, "expected one native official start")
    start, end = starts[0], terminal["ended_monotonic_ns"]
    require(clock["monotonic_ns"] <= start < end, "native measurement interval mismatch")
    offset = clock["utc_epoch_ns"] - clock["monotonic_ns"]
    signal_events = iter(
        (row["event"], row["observed_monotonic_ns"], row["signal_reason"])
        for row in events if row["event"] in SIGNAL_EVENTS and start <= row["observed_monotonic_ns"] <= end
    )
    tracking = manifest.get("breath_tracking_configuration") or {}
    setup_polar_clock = (tracking.get("schema") == "rusty.quest.breathing_setup_tracking_configuration.v1"
                         and tracking.get("source") == "polar-acc"
                         and tracking.get("sampled_clock") == tracking.get("raw_sampled_clock") == "java-nano-time")
    date = iso_utc((start + offset) // 1000)[:10]
    expected_name = f"{identity['participant_id']}_{date}_{SCENARIO}_r{identity['run_index']}.csv"
    require(expected_name == terminal["participant_csv"]["name"], "official start filename mismatch")
    common = {key: str(identity[key]) for key in ("participant_id", "session_id", "run_index", "scenario_id")}
    counts = Counter()
    allow_controller = manifest.get("participant_csv_controller_breath") is True
    last = None
    with ExitStack() as resources:
        projected = {}
        for kind, (file, native_kind, _) in PROJECTED.items():
            iterator = projected_rows(native_directory / file, lengths[file], run, native_kind, start, end, allow_controller)
            resources.callback(iterator.close)
            projected[kind] = iterator
        source = resources.enter_context(path.open(encoding="utf-8", newline=""))
        rows = csv.DictReader(source, strict=True)
        require(rows.fieldnames == HEADER, "CSV v4 header mismatch")
        for row in rows:
            require(None not in row and None not in row.values(), "CSV row width mismatch")
            require(row["schema_version"] == "4" and all(row[key] == value for key, value in common.items()),
                    "CSV row identity mismatch")
            kind = row["record_type"]
            require(kind in (EVENTS | PROJECTED.keys() | {"end"}) and counts["end"] == 0, "CSV row beyond terminal or unknown kind")
            require(0 <= int(row["elapsed_us"]) <= (end - start) // 1000, "CSV row outside start/end clock bounds")
            if kind in PROJECTED:
                native = next(projected[kind], None)
                check_measurement(kind, row, native, setup_polar_clock)
                at = native["observed_monotonic_ns"]
                require(native["observed_utc_ns"] == at + offset
                        and row["timestamp_utc"] == iso_utc((at + offset) // 1000)
                        and row["elapsed_us"] == str((at - start) // 1000)
                        and not any(row[key] for key in HEADER[8:11]),
                        f"{kind} projected observation clock mismatch")
            if kind in {"official-start", "end"}:
                at = start if kind == "official-start" else end
                require(row["timestamp_utc"] == iso_utc((at + offset) // 1000)
                        and row["elapsed_us"] == str((at - start) // 1000), f"{kind} UTC binding mismatch")
            if kind in SIGNAL_EVENTS:
                event = next(signal_events, None)
                require(event is not None and event[0] == kind and event[2] == row["signal_reason"]
                        and row["elapsed_us"] == str((event[1] - start) // 1000)
                        and row["timestamp_utc"] == iso_utc((event[1] + offset) // 1000),
                        "signal event projection mismatch")
            counts[kind] += 1
            last = row
        for kind, iterator in projected.items():
            require(next(iterator, None) is None, f"missing {kind} row")
    require(next(signal_events, None) is None, "missing committed signal event")
    require(counts["official-start"] == counts["end"] == 1 and all(counts[kind] for kind in ("ecg", "accelerometer", "radius")),
            "CSV start, terminal or main measurement stream missing")
    require(last["session_status"] == expected and last["end_reason"] and 0 <= int(last["audio_elapsed_us"]) <= END_US,
            "CSV terminal outcome or projected audio duration mismatch")
    return dict(counts), int(last["audio_elapsed_us"])


def verify_drive(args, descriptor):
    inventory = read_json(args.drive_list)
    entries = inventory["files"]
    require(not inventory.get("nextPageToken") and len(entries) == 1, "expected one Drive object in the complete inventory")
    file = entries[0]
    delivery = read_json(args.drive_receipt)
    require(file["name"] == descriptor["name"] and file["parents"] == [args.drive_folder]
            and file["ownedByMe"] is True and file["shared"] is False and file["trashed"] is False
            and file["mimeType"] == "text/csv"
            and int(file["size"]) == descriptor["size"]
            and file.get("sha256Checksum", descriptor["sha256"]) == descriptor["sha256"],
            "Drive object metadata mismatch")
    require(delivery["verified"] is True and delivery["folderId"] == args.drive_folder
            and delivery["driveId"] == file["id"] and delivery["sha256"] == descriptor["sha256"],
            "Drive receipt disagrees with the independently listed object")


def verify(args):
    exported = read_json(args.export_manifest)
    run = args.native_run.name
    require(re.fullmatch(r"utc-ns-[0-9]{20}", run) and exported["id"] == run
            and exported["schema"] == "rusty.quest.recording_export.v1" and exported["saved"] is True,
            "authenticated export identity or saved flag mismatch")
    policy = exported["file_set_policy"]
    recovered = policy == "core8-recovered-checkpoint"
    require(recovered or policy == "core8-original-no-checkpoints-or-index", "unknown export file set")
    terminal_path = args.native_run / ("participant-recovered.json" if recovered else "final.json")
    manifest = read_json(args.native_run / "manifest.json")
    terminal = read_json(terminal_path)
    require(manifest["schema"] == NATIVE_SCHEMA + "manifest.v1" and terminal["schema"] == NATIVE_SCHEMA + "final.v1"
            and manifest["started_at_utc"] == terminal["started_at_utc"] == run
            and manifest["participant_csv_schema"] == 4,
            "native manifest and terminal identity mismatch")
    complete = args.expect == "complete"
    require(terminal["saved"] is True and terminal["completed"] is complete
            and terminal["dropped_records"] == terminal["loss"]["dropped_records"] == 0
            and manifest["completion_threshold_ns"] == END_US * 1000
            and terminal.get("completion_threshold_ns", manifest["completion_threshold_ns"]) == END_US * 1000
            and (recovered or "completion_threshold_ns" in terminal)
            and exported["disposition"] == ("finalized-completed" if complete else
                                            "interrupted-early" if recovered else "finalized-early"),
            "native terminal outcome, writer drops or 900-second profile mismatch")

    descriptor = exported["participant_csv"]
    name = descriptor["name"]
    require(isinstance(name, str) and re.fullmatch(r"[^/\\:]+_[0-9]{4}-[0-9]{2}-[0-9]{2}_haupt-15_r[1-9][0-9]*\.csv", name)
            and descriptor == terminal["participant_csv"] and descriptor["schema_version"] == 4
            and descriptor["session_id"] == manifest["participant"]["session_id"], "CSV descriptor mismatch")
    path = args.native_run / (".recovered-" + name if recovered else name)
    verify_bytes(path, descriptor, "Quest CSV")
    verify_bytes(args.pixel_csv, descriptor, "Pixel CSV")
    verify_bytes(args.drive_csv, descriptor, "Drive CSV")
    verify_drive(args, descriptor)

    entries = {entry["name"]: entry for entry in exported["files"]}
    require(len(exported["files"]) == len(entries) == 8 and entries.keys() == STREAMS | {"manifest.json", "final.json"}
            and terminal["durable_stream_lengths"].keys() == STREAMS, "native file inventory mismatch")
    if recovered:
        checkpoint = terminal["recovery"]
        require(checkpoint == exported["recovery"] and checkpoint["v"] == 1
                and type(checkpoint["checkpoint_ordinal"]) is int and checkpoint["checkpoint_ordinal"] > 0
                and checkpoint["stream_sha256"] == {name: entries[name]["sha256"] for name in STREAMS},
                "native recovery witness mismatch")
        checkpoint_path = args.native_run / f"checkpoint-{checkpoint['checkpoint_ordinal']:020d}.json"
        verify_bytes(checkpoint_path, dict(size=checkpoint_path.stat().st_size, sha256=checkpoint["checkpoint_sha256"]),
                     "native checkpoint")
        require(read_json(checkpoint_path)["durable_stream_lengths"] == terminal["durable_stream_lengths"],
                "native checkpoint lengths disagree")
    tails = 0
    for name, entry in entries.items():
        if name in STREAMS:
            require(entry["size"] == terminal["durable_stream_lengths"][name], "native committed length disagrees")
        source = terminal_path if name == "final.json" else args.native_run / name
        tails += verify_bytes(source, entry, name, allow_tail=recovered and name in STREAMS)

    counts, projected_audio_us = verify_csv(path, args.native_run, run, manifest, terminal, "completed" if complete else "aborted")
    return {"pass": True, "mode": args.expect, "bytes": descriptor["size"], "sha256": descriptor["sha256"],
            "csv_rows": counts, "writer_drops": 0, "drive_objects": 1, "native_checkpoint_recovery": recovered,
            "ignored_post_checkpoint_bytes": tails, "terminal_csv_audio_elapsed_us": projected_audio_us}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("native-run", "export-manifest", "pixel-csv", "drive-csv", "drive-list", "drive-receipt"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--drive-folder", required=True)
    parser.add_argument("--expect", choices=("complete", "partial"), required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args), sort_keys=True))
    except EvidenceError as error:
        parser.exit(1, f"Evidence failed: {error}\n")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, csv.Error, InvalidOperation) as error:
        parser.exit(1, f"Evidence input is incomplete or invalid ({type(error).__name__}).\n")


if __name__ == "__main__":
    main()
