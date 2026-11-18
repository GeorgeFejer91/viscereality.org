"""Verify one saved Haupt-Runde evidence bundle without printing participant data."""
import argparse
import csv
import hashlib
import json
import re
from bisect import bisect_right
from collections import Counter
from contextlib import ExitStack, closing
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from uuid import UUID


NATIVE_SCHEMA = "rusty.quest.native_renderer.session_recording_"
SCENARIO = "haupt-15"
END_US = 900_000_000
POLAR_RECEIPT_TIMEOUT_NS = 1_000_000_000
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
PLAYBACK_EVENTS = {
    "official-start", "audio-started", "experiment-paused", "experiment-resumed", "audio-ended", "audio-error",
}
SIGNAL_EVENTS = {"ecg-interrupted", "ecg-resumed", "acc-interrupted", "acc-resumed"}
INTERRUPTION_EVENTS = SIGNAL_EVENTS | {"experiment-paused", "experiment-resumed", "audio-error"}
CSV_RECORD_TYPES = PLAYBACK_EVENTS | SIGNAL_EVENTS | PROJECTED.keys() | {"end"}
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


def playback_rows(events, start, end, offset, identity):
    duration = END_US * 1000
    require(identity["planned_duration_seconds"] == END_US // 1_000_000,
            "native playback profile has the wrong duration")
    offsets, phases = zip(*identity["phase_offsets_ns"])
    require(offsets[0] == 0 and all(type(value) is int and 0 <= value < duration for value in offsets)
            and all(left < right for left, right in zip(offsets, offsets[1:]))
            and all(isinstance(phase, str) and phase for phase in phases),
            "native playback phase offsets are invalid")

    def projection(kind, at, elapsed, state):
        elapsed = min(elapsed, duration)
        if elapsed == duration:
            state = "ended"
        phase = state if state in {"waiting", "ended"} else phases[bisect_right(offsets, elapsed) - 1]
        return kind, iso_utc((at + offset) // 1000), str((at - start) // 1000), str(elapsed // 1000), state, phase

    state = None
    media_started = False
    last = start
    elapsed = 0
    for event in events:
        at, kind = event["observed_monotonic_ns"], event["event"]
        if kind not in PLAYBACK_EVENTS or not start <= at <= end:
            continue
        if kind == "official-start":
            next_state = "waiting"
        elif state in {None, "ended"}:
            continue
        elif kind == "audio-started":
            if media_started:
                continue
            media_started = True
            next_state = "paused" if state == "paused" else "playing"
        elif kind == "experiment-paused":
            next_state = "paused"
        elif kind == "experiment-resumed":
            next_state = "playing" if media_started else "waiting"
        else:
            next_state = "ended"
        require(last <= at and event["observed_utc_ns"] == at + offset,
                "native playback transition order or clock mismatch")
        if state == "playing":
            elapsed += at - last
        last, state = at, next_state
        yield projection(kind, at, elapsed, state)
    yield projection("end", end, elapsed + (end - last if state == "playing" else 0), state)


def verify_polar_receipt_coverage(native_directory, lengths, run, start, completion):
    for label, filename in (("ECG", "polar-ecg.jsonl"), ("ACC", "polar-acc.jsonl")):
        latest = start
        gap = f"uninterrupted run contains an {label} receipt gap"
        with closing(native_rows(native_directory / filename, lengths[filename], run)) as rows:
            for _, batch in groupby(rows, key=itemgetter("record_sequence")):
                # The native writer observes a nonempty batch at its last
                # sample's observation timestamp. Older batches may arrive late.
                for sample in batch:
                    observed = sample["observed_monotonic_ns"]
                if start <= observed <= completion:
                    require(observed - latest <= POLAR_RECEIPT_TIMEOUT_NS, gap)
                    latest = max(latest, observed)
        require(completion - latest <= POLAR_RECEIPT_TIMEOUT_NS, gap)


def verify_csv(path, native_directory, run, manifest, terminal, expected, require_uninterrupted):
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
    completion = [row for row in events if row["event"] == "completion-reached"]
    require(len(completion) == (1 if expected == "completed" else 0),
            "native completion event disagrees with the terminal outcome")
    completion_elapsed_us = None
    if completion:
        at = completion[0]["observed_monotonic_ns"]
        require(start < at <= end and completion[0]["observed_utc_ns"] == at + offset,
                "native completion event is outside the run or has the wrong clock")
        completion_elapsed_us = (at - start) // 1000
    playback = playback_rows(events, start, end, offset, identity)
    signal_events = (
        row
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
            require(kind in CSV_RECORD_TYPES and counts["end"] == 0, "CSV row beyond terminal or unknown kind")
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
            if kind in PLAYBACK_EVENTS or kind == "end":
                require(tuple(row[key] for key in HEADER[5:11]) == next(playback, None)
                        and not any(row[key] for key in HEADER[11:21]),
                        "CSV playback transition or clock mismatch")
            if kind in SIGNAL_EVENTS:
                event = next(signal_events, None)
                require(event is not None, "unexpected signal event projection")
                at = event["observed_monotonic_ns"]
                require(event["observed_utc_ns"] == at + offset, "native signal event clock mismatch")
                require(event["event"] == kind and event["signal_reason"] == row["signal_reason"]
                        and row["elapsed_us"] == str((at - start) // 1000)
                        and row["timestamp_utc"] == iso_utc((at + offset) // 1000),
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
    if require_uninterrupted:
        require(not INTERRUPTION_EVENTS.intersection(counts), "uninterrupted run contains pauses or faults")
        require(counts.get("breath", 0) > 0, "uninterrupted run has no breathing observations")
        verify_polar_receipt_coverage(native_directory, lengths, run, start, completion[0]["observed_monotonic_ns"])
    return dict(counts), int(last["audio_elapsed_us"]), completion_elapsed_us


def verify_drive(args, descriptor):
    inventory = read_json(args.drive_list)
    entries = inventory["files"]
    require(not inventory.get("nextPageToken") and inventory.get("incompleteSearch", False) is False
            and len(entries) == 1, "expected one Drive object in the complete inventory")
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

    counts, projected_audio_us, native_completion_us = verify_csv(
        path, args.native_run, run, manifest, terminal, "completed" if complete else "aborted", args.require_uninterrupted)
    return {"pass": True, "mode": args.expect, "bytes": descriptor["size"], "sha256": descriptor["sha256"],
            "csv_rows": counts, "writer_drops": 0, "drive_objects": 1, "native_checkpoint_recovery": recovered,
            "uninterrupted_checks": args.require_uninterrupted,
            "ignored_post_checkpoint_bytes": tails, "terminal_csv_audio_elapsed_us": projected_audio_us,
            "native_start_to_completion_us": native_completion_us}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("native-run", "export-manifest", "pixel-csv", "drive-csv", "drive-list", "drive-receipt"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--drive-folder", required=True)
    parser.add_argument("--expect", choices=("complete", "partial"), required=True)
    parser.add_argument("--require-uninterrupted", action="store_true",
                        help="require breathing observations and continuous Polar receipts in a completed run without pauses or faults")
    args = parser.parse_args()
    if args.require_uninterrupted and args.expect != "complete":
        parser.error("--require-uninterrupted requires --expect complete")
    try:
        print(json.dumps(verify(args), sort_keys=True))
    except EvidenceError as error:
        parser.exit(1, f"Evidence failed: {error}\n")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, csv.Error, InvalidOperation) as error:
        parser.exit(1, f"Evidence input is incomplete or invalid ({type(error).__name__}).\n")


if __name__ == "__main__":
    main()
