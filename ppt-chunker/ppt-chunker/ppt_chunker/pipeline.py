from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .dependencies import discover_ffmpeg_tools
from .exceptions import PipelineError
from .manifest import build_manifest
from .media import (
    cut_chunk_frame_exact,
    decode_check,
    enforce_chunk_size,
    extract_frame_png,
    ffprobe_duration,
    ffprobe_video_stream_info,
    image_mismatch_score,
    normalize_video,
    rename_with_hash,
)
from .models import Hiccup, ResolvedSegment, SlideFeature, TimingDecision
from .ppt_export import export_ppt_to_video
from .slide_media_scan import extract_slide_features
from .timing import (
    TINY_TRANSITION_SEC,
    build_timing_config_payload,
    load_overrides,
    parse_pptx_timing_xml,
    probe_timing_com,
    resolve_segments,
    segments_from_config,
)
from .utils import read_json, utc_now_iso, write_json

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "balanced1080": {
        "fps": 30,
        "height": 1080,
        "quality": 85,
        "default_static_sec": 4.0,
        "default_transition_sec": 0.5,
        "duration_tolerance_sec": 0.25,
        "max_chunk_mb": 95.0,
    }
}


def inspect_command(args: argparse.Namespace) -> None:
    pptx_path = Path(args.pptx).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ffprobe_bin = discover_ffmpeg_tools(None, getattr(args, "ffprobe_bin", None))[1]

    features = extract_slide_features(pptx_path, str(ffprobe_bin))
    com_rows = probe_timing_com(pptx_path)
    _merge_com_into_features(features, com_rows)

    profile = _profile_settings(getattr(args, "profile", "balanced1080"))
    report = {
        "version": 3,
        "created_at_utc": utc_now_iso(),
        "pptx": str(pptx_path),
        "profile": getattr(args, "profile", "balanced1080"),
        "defaults": {
            "default_static_sec": profile["default_static_sec"],
            "default_transition_sec": profile["default_transition_sec"],
        },
        "slides": [row.to_dict() for row in features],
        "summary": {
            "slide_count": len(features),
            "media_slides": len([r for r in features if r.static_classification == "media"]),
            "static_slides": len([r for r in features if r.static_classification == "static"]),
            "unresolved_visible_media_slides": len(
                [r for r in features if r.unresolved_visible_media_count > 0]
            ),
        },
    }
    write_json(output_dir / "feature_report.json", report)
    print(f"Feature report written: {output_dir / 'feature_report.json'}")


def build_command(args: argparse.Namespace) -> None:
    pptx_path = Path(args.pptx).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir = output_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    hiccup_report = output_dir / "hiccup_report.json"
    if hiccup_report.exists():
        hiccup_report.unlink()

    profile_name = getattr(args, "profile", "balanced1080")
    profile = _profile_settings(profile_name)
    strict = bool(getattr(args, "strict", True))
    presentation_id = sanitize_id(getattr(args, "presentation_id", None) or pptx_path.stem)
    title = getattr(args, "title", None) or pptx_path.stem
    ffmpeg_bin, ffprobe_bin = discover_ffmpeg_tools(
        getattr(args, "ffmpeg_bin", None), getattr(args, "ffprobe_bin", None)
    )

    overrides = _load_pipeline_overrides(getattr(args, "overrides_file", None))
    defaults = overrides.get("defaults", {})
    default_static_sec = float(defaults.get("slide_sec", profile["default_static_sec"]))
    default_transition_sec = float(
        defaults.get("transition_sec", profile["default_transition_sec"])
    )
    max_chunk_mb = float(getattr(args, "max_chunk_mb", profile["max_chunk_mb"]))
    mute_output = bool(getattr(args, "mute_output", True))

    features = extract_slide_features(pptx_path, str(ffprobe_bin))
    com_rows = probe_timing_com(pptx_path)
    _merge_com_into_features(features, com_rows)
    xml_rows = {row.slide_number: row for row in parse_pptx_timing_xml(pptx_path)}
    intrinsic_slide_map = _collect_intrinsic_slide_map(xml_rows, com_rows)
    write_json(
        output_dir / "feature_report.json",
        {
            "version": 3,
            "created_at_utc": utc_now_iso(),
            "pptx": str(pptx_path),
            "profile": profile_name,
            "slides": [row.to_dict() for row in features],
        },
    )

    decisions, timing_hiccups = _resolve_timing_decisions(
        features=features,
        overrides=overrides,
        default_static_sec=default_static_sec,
        default_transition_sec=default_transition_sec,
        intrinsic_slide_map=intrinsic_slide_map,
        prefer_intrinsic_when_no_rewrite=bool(getattr(args, "no_rewrite_timings", False)),
    )
    write_json(
        output_dir / "timing_decisions.json",
        {
            "version": 3,
            "created_at_utc": utc_now_iso(),
            "defaults": {
                "default_static_sec": default_static_sec,
                "default_transition_sec": default_transition_sec,
            },
            "slides": [row.to_dict() for row in decisions],
        },
    )
    _append_hiccups(output_dir, "timing", timing_hiccups)

    segments = [row.to_segment() for row in decisions]
    if strict and _has_error_hiccups(timing_hiccups):
        raise PipelineError("Strict build blocked: unresolved timing/media hiccups detected.")

    raw_mp4 = build_dir / f"{presentation_id}_raw.mp4"
    master_mp4 = output_dir / f"{presentation_id}_master.mp4"
    rewrite_timings = not bool(getattr(args, "no_rewrite_timings", False))
    reuse_master = bool(getattr(args, "reuse_master", False))

    if reuse_master and master_mp4.exists():
        print(f"Reusing existing master video: {master_mp4}")
    else:
        print("Exporting PowerPoint video...")
        export_ppt_to_video(
            pptx_path=pptx_path,
            output_mp4=raw_mp4,
            segments=segments,
            rewrite_timings=rewrite_timings,
            fps=int(profile["fps"]),
            vert_resolution=int(profile["height"]),
            quality=int(profile["quality"]),
            default_slide_duration_s=float(default_static_sec),
        )
        normalize_video(
            ffmpeg_bin=ffmpeg_bin,
            input_mp4=raw_mp4,
            output_mp4=master_mp4,
            fps=int(profile["fps"]),
            height=int(profile["height"]),
            mute_output=mute_output,
        )

    source_gap_frames_by_slide: dict[int, int] = {}
    if bool(getattr(args, "no_rewrite_timings", False)):
        decisions, source_gap_frames_by_slide, compat_hiccups = _apply_no_rewrite_export_compat(
            decisions=decisions,
            features=features,
            master_mp4=master_mp4,
            ffprobe_bin=ffprobe_bin,
            fps=int(profile["fps"]),
            default_transition_sec=default_transition_sec,
        )
        _append_hiccups(output_dir, "timing_compat", compat_hiccups)
        if strict and _has_error_hiccups(compat_hiccups):
            raise PipelineError("Strict build blocked: no-rewrite compatibility failed.")
        segments = [row.to_segment() for row in decisions]
        write_json(
            output_dir / "timing_decisions.json",
            {
                "version": 3,
                "created_at_utc": utc_now_iso(),
                "defaults": {
                    "default_static_sec": default_static_sec,
                    "default_transition_sec": default_transition_sec,
                },
                "slides": [row.to_dict() for row in decisions],
            },
        )

    (
        chunk_entries_legacy,
        chunk_entries_extended,
        chunk_hiccups,
        segment_meta,
    ) = _segment_mixed_assets(
        output_dir=output_dir,
        master_mp4=master_mp4,
        segments=segments,
        png_by_slide={},
        source_gap_frames_by_slide=source_gap_frames_by_slide,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        fps=int(profile["fps"]),
        max_chunk_mb=max_chunk_mb,
        mute_output=mute_output,
    )
    _append_hiccups(output_dir, "chunking", chunk_hiccups)
    if strict and _has_error_hiccups(chunk_hiccups):
        raise PipelineError("Strict build blocked: chunking hiccups detected.")

    manifest = build_manifest(
        presentation_id=presentation_id,
        source_ppt=pptx_path,
        title=title,
        segments=segments,
        chunk_entries_legacy=chunk_entries_legacy,
        chunk_entries_extended=chunk_entries_extended,
        encoding={
            "fps": int(profile["fps"]),
            "height": int(profile["height"]),
            "quality": int(profile["quality"]),
            "mute": mute_output,
            "profile": profile_name,
        },
        master_file=master_mp4.name,
        deck_meta=overrides.get("deck_meta", {}),
        timeline_skip_sec=float(segment_meta.get("timeline_skip_sec", 0.0)),
    )
    write_json(output_dir / "manifest.json", manifest)
    _copy_player_template(output_dir)

    build_report = {
        "version": 3,
        "created_at_utc": utc_now_iso(),
        "strict": strict,
        "presentation_id": presentation_id,
        "title": title,
        "profile": profile_name,
        "output_dir": str(output_dir),
        "master_file": master_mp4.name,
        "slide_count": len(segments),
        "timeline_skip_sec": float(segment_meta.get("timeline_skip_sec", 0.0)),
    }
    write_json(output_dir / "build_report.json", build_report)
    print(f"Build completed: {output_dir}")


def validate_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    strict = bool(getattr(args, "strict", True))
    ffmpeg_bin, ffprobe_bin = discover_ffmpeg_tools(
        getattr(args, "ffmpeg_bin", None), getattr(args, "ffprobe_bin", None)
    )
    if not (output_dir / "manifest.json").exists():
        raise PipelineError(f"Missing manifest.json in {output_dir}")

    manifest = read_json(output_dir / "manifest.json")
    hiccups: list[Hiccup] = []
    static_scores: list[dict[str, Any]] = []

    chunks = manifest.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        hiccups.append(Hiccup("manifest_no_chunks", "error", "Manifest has no chunks."))
    else:
        total_duration = 0.0
        for chunk in chunks:
            file_rel = str(chunk.get("file", "")).strip()
            if not file_rel:
                hiccups.append(Hiccup("manifest_file_missing", "error", "Chunk has no file field."))
                continue
            fpath = output_dir / file_rel
            if not fpath.exists() or fpath.stat().st_size == 0:
                hiccups.append(
                    Hiccup(
                        "chunk_missing_or_empty",
                        "error",
                        f"Missing or empty chunk asset: {file_rel}",
                    )
                )
                continue
            expected_dur = float(chunk.get("duration", 0.0) or 0.0)
            total_duration += expected_dur
            if fpath.suffix.lower() == ".mp4":
                try:
                    decode_check(ffprobe_bin, fpath)
                    actual_dur = ffprobe_duration(ffprobe_bin, fpath)
                    if abs(actual_dur - expected_dur) > 0.25:
                        hiccups.append(
                            Hiccup(
                                "chunk_duration_mismatch",
                                "error",
                                (
                                    f"Chunk duration mismatch for {file_rel}: "
                                    f"expected={expected_dur:.3f}s actual={actual_dur:.3f}s"
                                ),
                            )
                        )
                except Exception as exc:
                    hiccups.append(
                        Hiccup(
                            "chunk_decode_failure",
                            "error",
                            f"Chunk decode failed for {file_rel}: {exc}",
                        )
                    )
        master_file = manifest.get("master_file")
        if master_file:
            master_path = output_dir / str(master_file)
            if master_path.exists():
                master_duration = ffprobe_duration(ffprobe_bin, master_path)
                timeline_skip = float(manifest.get("timeline_skip_sec", 0.0) or 0.0)
                expected_master = total_duration + max(0.0, timeline_skip)
                if abs(master_duration - expected_master) > 0.4:
                    hiccups.append(
                        Hiccup(
                            "timeline_drift",
                            "error",
                            (
                                "Master duration and chunk timeline diverge: "
                                f"master={master_duration:.3f}s "
                                f"chunks={total_duration:.3f}s "
                                f"skip={timeline_skip:.3f}s"
                            ),
                        )
                    )

    feature_report_path = output_dir / "feature_report.json"
    if feature_report_path.exists():
        feature_report = read_json(feature_report_path)
        for row in feature_report.get("slides", []):
            unresolved = int(row.get("unresolved_visible_media_count", 0) or 0)
            visible = int(row.get("visible_media_count", 0) or 0)
            if visible > 0 and unresolved > 0:
                slide_num = int(row.get("slide_number", 0))
                hiccups.append(
                    Hiccup(
                        "visible_media_duration_unresolved",
                        "error",
                        f"Slide {slide_num} has unresolved visible media durations.",
                        slide_number=slide_num,
                    )
                )

    if any((c.get("asset_kind") == "image") for c in chunks):
        player_html = output_dir / "index.html"
        if not player_html.exists():
            hiccups.append(
                Hiccup("player_missing", "error", "Player file missing while image assets are used.")
            )
        else:
            text = player_html.read_text(encoding="utf-8", errors="ignore")
            if "asset_kind" not in text or "img-layer" not in text:
                hiccups.append(
                    Hiccup(
                        "player_asset_type_incompatibility",
                        "error",
                        "Manifest uses image assets but player lacks image runtime support.",
                    )
                )

    segments = manifest.get("segments", [])
    master_file = manifest.get("master_file")
    if isinstance(segments, list) and master_file:
        master_path = output_dir / str(master_file)
        if master_path.exists():
            for seg in segments:
                if seg.get("type") != "slide" or seg.get("asset_kind") != "image":
                    continue
                image_rel = str(seg.get("file", ""))
                image_path = output_dir / image_rel
                if not image_path.exists():
                    continue
                source_start_sec = float(
                    seg.get("source_start_sec", seg.get("start_sec", 0.0)) or 0.0
                )
                start_sec = source_start_sec + (1.0 / 30.0)
                sample_png = output_dir / "build" / f"{seg.get('id', 'slide')}_sample.png"
                try:
                    extract_frame_png(ffmpeg_bin, master_path, sample_png, start_sec)
                    score = image_mismatch_score(image_path, sample_png)
                    static_scores.append({"segment_id": seg.get("id"), "mismatch": score})
                    if score > 0.02:
                        hiccups.append(
                            Hiccup(
                                "static_visual_mismatch",
                                "error",
                                (
                                    f"Static slide visual mismatch for {seg.get('id')}: "
                                    f"score={score:.4f}"
                                ),
                                slide_number=int(seg.get("slide_index", 0) or 0),
                            )
                        )
                except Exception as exc:
                    hiccups.append(
                        Hiccup(
                            "static_visual_check_failed",
                            "error",
                            f"Could not validate static slide {seg.get('id')}: {exc}",
                        )
                    )

    _append_hiccups(output_dir, "validate", hiccups)
    report = {
        "version": 3,
        "created_at_utc": utc_now_iso(),
        "strict": strict,
        "status": "fail" if _has_error_hiccups(hiccups) else "pass",
        "error_count": len([h for h in hiccups if h.severity == "error"]),
        "warning_count": len([h for h in hiccups if h.severity != "error"]),
        "static_visual_scores": static_scores,
    }
    write_json(output_dir / "validation_report.json", report)
    marker = output_dir / "validation.ok"
    if report["status"] == "pass":
        marker.write_text("ok\n", encoding="utf-8")
    elif marker.exists():
        marker.unlink()

    if strict and report["status"] != "pass":
        raise PipelineError("Validation failed in strict mode.")
    print(f"Validation status: {report['status']}")


def publish_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    deck = str(args.deck).strip()
    if not deck:
        raise PipelineError("Deck name is required for publish.")
    report_path = output_dir / "validation_report.json"
    if not report_path.exists():
        raise PipelineError("Cannot publish without validation_report.json")
    report = read_json(report_path)
    if str(report.get("status")) != "pass":
        raise PipelineError("Validation must pass before publish.")

    repo_root = Path(__file__).resolve().parents[3]
    target_dir = repo_root / "presentations" / deck
    target_dir.mkdir(parents=True, exist_ok=True)

    for name in ("chunks", "assets"):
        src = output_dir / name
        dst = target_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        if src.exists():
            shutil.copytree(src, dst)

    for name in ("manifest.json", "index.html"):
        src_file = output_dir / name
        if src_file.exists():
            shutil.copy2(src_file, target_dir / name)
        else:
            raise PipelineError(f"Publish source missing required file: {src_file}")

    print(f"Published artifacts to: {target_dir}")

    if bool(getattr(args, "git_push", False)):
        rel = str(target_dir.relative_to(repo_root)).replace("\\", "/")
        _git_add_commit_push(
            repo_root=repo_root,
            pathspec=rel,
            remote=str(getattr(args, "git_remote", "origin")),
            branch=str(getattr(args, "git_branch", "main")),
            message=f"Publish {deck} presentation artifacts",
        )


def analyze_command(args: argparse.Namespace) -> None:
    inspect_args = argparse.Namespace(
        pptx=args.pptx,
        output_dir=args.output_dir,
        profile=getattr(args, "profile", "balanced1080"),
        ffprobe_bin=getattr(args, "ffprobe_bin", None),
    )
    inspect_command(inspect_args)

    pptx_path = Path(args.pptx).expanduser().resolve()
    xml_sources = parse_pptx_timing_xml(pptx_path)
    com_sources = probe_timing_com(pptx_path) if bool(getattr(args, "com_probe", False)) else {}
    overrides = (
        load_overrides(Path(args.overrides_file))
        if getattr(args, "overrides_file", None)
        else None
    )
    segments = resolve_segments(
        xml_sources=xml_sources,
        com_sources=com_sources,
        timing_mode=getattr(args, "timing_mode", "ppt"),
        default_slide_sec=float(getattr(args, "default_slide_sec", 4.0)),
        default_transition_sec=float(getattr(args, "default_transition_sec", 0.5)),
        overrides=overrides,
    )
    config = build_timing_config_payload(
        segments=segments,
        default_slide_sec=float(getattr(args, "default_slide_sec", 4.0)),
        default_transition_sec=float(getattr(args, "default_transition_sec", 0.5)),
        timing_mode=getattr(args, "timing_mode", "ppt"),
    )
    write_json(Path(args.output_dir).expanduser().resolve() / "timing_config.json", config)
    print("Legacy timing config written.")


def chunk_command(args: argparse.Namespace) -> None:
    mp4_path = Path(args.mp4).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin, ffprobe_bin = discover_ffmpeg_tools(
        getattr(args, "ffmpeg_bin", None), getattr(args, "ffprobe_bin", None)
    )
    config = read_json(Path(args.config).expanduser().resolve())
    segments = segments_from_config(config)

    chunk_entries_legacy, chunk_entries_extended, hiccups, segment_meta = _segment_mixed_assets(
        output_dir=output_dir,
        master_mp4=mp4_path,
        segments=segments,
        png_by_slide={},
        source_gap_frames_by_slide={},
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        fps=30,
        max_chunk_mb=float(getattr(args, "max_chunk_mb", 95.0)),
        mute_output=bool(getattr(args, "mute_output", True)),
    )
    _append_hiccups(output_dir, "chunk", hiccups)
    if _has_error_hiccups(hiccups):
        raise PipelineError("Chunking failed with errors.")

    manifest = build_manifest(
        presentation_id=sanitize_id(config.get("presentation_id", mp4_path.stem)),
        source_ppt=None,
        title=str(config.get("title") or mp4_path.stem),
        segments=segments,
        chunk_entries_legacy=chunk_entries_legacy,
        chunk_entries_extended=chunk_entries_extended,
        encoding={"timing_mode": "config"},
        master_file=mp4_path.name,
        timeline_skip_sec=float(segment_meta.get("timeline_skip_sec", 0.0)),
    )
    write_json(output_dir / "manifest.json", manifest)
    _copy_player_template(output_dir)
    print(f"Wrote manifest: {output_dir / 'manifest.json'}")


def run_command(args: argparse.Namespace) -> None:
    build_args = argparse.Namespace(
        pptx=args.pptx,
        output_dir=args.output_dir,
        presentation_id=getattr(args, "presentation_id", None),
        title=getattr(args, "title", None),
        profile=getattr(args, "profile", "balanced1080"),
        overrides_file=getattr(args, "overrides_file", None),
        strict=bool(getattr(args, "strict", True)),
        ffmpeg_bin=getattr(args, "ffmpeg_bin", None),
        ffprobe_bin=getattr(args, "ffprobe_bin", None),
        max_chunk_mb=float(
            getattr(args, "max_chunk_mb", PROFILE_PRESETS["balanced1080"]["max_chunk_mb"])
        ),
        mute_output=bool(getattr(args, "mute_output", True)),
        no_rewrite_timings=False,
        reuse_master=bool(getattr(args, "reuse_master", False)),
    )
    build_command(build_args)
    validate_args = argparse.Namespace(
        output_dir=args.output_dir,
        strict=bool(getattr(args, "strict", True)),
        ffmpeg_bin=getattr(args, "ffmpeg_bin", None),
        ffprobe_bin=getattr(args, "ffprobe_bin", None),
    )
    validate_command(validate_args)


def _segment_mixed_assets(
    *,
    output_dir: Path,
    master_mp4: Path,
    segments: list[ResolvedSegment],
    png_by_slide: dict[int, Path],
    source_gap_frames_by_slide: dict[int, int] | None,
    ffmpeg_bin: Path,
    ffprobe_bin: Path,
    fps: int,
    max_chunk_mb: float,
    mute_output: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Hiccup], dict[str, Any]]:
    chunks_dir = output_dir / "chunks"
    assets_dir = output_dir / "assets"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    hiccups: list[Hiccup] = []
    stream_info = ffprobe_video_stream_info(ffprobe_bin, master_mp4)
    source_fps = float(stream_info["fps"])
    source_frames = int(round(float(stream_info["nb_frames"])))
    if abs(source_fps - fps) > 0.01:
        hiccups.append(
            Hiccup(
                "fps_mismatch",
                "warning",
                f"Normalized master FPS={source_fps:.3f} differs from expected {fps}.",
            )
        )

    source_frame_cursor = 0
    timeline_frame_cursor = 0
    skipped_source_frames = 0
    legacy: list[dict[str, Any]] = []
    extended: list[dict[str, Any]] = []
    gap_map = source_gap_frames_by_slide or {}

    for seg in segments:
        gap_frames = int(gap_map.get(seg.slide_number, 0) or 0)
        if gap_frames > 0:
            if source_frame_cursor + gap_frames > source_frames + 1:
                hiccups.append(
                    Hiccup(
                        "source_gap_out_of_bounds",
                        "error",
                        (
                            f"Source gap before slide {seg.slide_number} exceeds master timeline "
                            f"(gap_frames={gap_frames})."
                        ),
                        slide_number=seg.slide_number,
                    )
                )
            source_frame_cursor += gap_frames
            skipped_source_frames += gap_frames

        transition_frames = (
            int(round(seg.transition_duration_s * fps))
            if seg.transition_type != "none" and seg.transition_duration_s > 0
            else 0
        )
        slide_frames = max(1, int(round(seg.slide_duration_s * fps)))

        if transition_frames > 0:
            if source_frame_cursor + transition_frames > source_frames + 1:
                hiccups.append(
                    Hiccup(
                        "timeline_out_of_bounds",
                        "error",
                        f"Transition for slide {seg.slide_number} exceeds master timeline.",
                        slide_number=seg.slide_number,
                    )
                )
            raw_transition = chunks_dir / f"trans_{seg.slide_number:02d}.mp4"
            cut_chunk_frame_exact(
                ffmpeg_bin=ffmpeg_bin,
                input_mp4=master_mp4,
                output_mp4=raw_transition,
                start_frame=source_frame_cursor,
                frame_count=transition_frames,
                fps=fps,
                mute_output=mute_output,
            )
            enforce_chunk_size(ffmpeg_bin, raw_transition, max_chunk_mb, mute_output=mute_output)
            hashed_transition = rename_with_hash(raw_transition)
            decode_check(ffprobe_bin, hashed_transition)
            transition_duration = transition_frames / fps
            t_start = timeline_frame_cursor / fps
            t_end = (timeline_frame_cursor + transition_frames) / fps
            source_t_start = source_frame_cursor / fps
            source_t_end = (source_frame_cursor + transition_frames) / fps
            transition_rel = f"chunks/{hashed_transition.name}"
            legacy.append(
                {
                    "type": "transition",
                    "file": transition_rel,
                    "duration": round(transition_duration, 3),
                    "slide_from": seg.slide_number - 1 if seg.slide_number > 1 else 0,
                    "slide_to": seg.slide_number,
                    "asset_kind": "video",
                }
            )
            extended.append(
                {
                    "id": f"trans_{seg.slide_number:02d}",
                    "type": "transition",
                    "slide_index": seg.slide_number,
                    "start_sec": round(t_start, 3),
                    "end_sec": round(t_end, 3),
                    "source_start_sec": round(source_t_start, 3),
                    "source_end_sec": round(source_t_end, 3),
                    "duration_sec": round(transition_duration, 3),
                    "file": transition_rel,
                    "loop_default": False,
                    "asset_kind": "video",
                }
            )
            source_frame_cursor += transition_frames
            timeline_frame_cursor += transition_frames

        slide_start = timeline_frame_cursor / fps
        slide_end = (timeline_frame_cursor + slide_frames) / fps
        source_slide_start = source_frame_cursor / fps
        source_slide_end = (source_frame_cursor + slide_frames) / fps
        if seg.asset_kind == "image":
            raw_png = assets_dir / f"slide_{seg.slide_number:02d}.png"
            try:
                extract_frame_png(ffmpeg_bin, master_mp4, raw_png, source_slide_start + (1.0 / fps))
                hashed_png = rename_with_hash(raw_png)
                png_rel = f"assets/{hashed_png.name}"
                legacy.append(
                    {
                        "type": "slide",
                        "file": png_rel,
                        "duration": round(slide_frames / fps, 3),
                        "slide_number": seg.slide_number,
                        "label": seg.label,
                        "loop": True,
                        "asset_kind": "image",
                    }
                )
                extended.append(
                    {
                        "id": f"slide_{seg.slide_number:02d}",
                        "type": "slide",
                        "slide_index": seg.slide_number,
                        "start_sec": round(slide_start, 3),
                        "end_sec": round(slide_end, 3),
                        "source_start_sec": round(source_slide_start, 3),
                        "source_end_sec": round(source_slide_end, 3),
                        "duration_sec": round(slide_frames / fps, 3),
                        "file": png_rel,
                        "loop_default": True,
                        "asset_kind": "image",
                    }
                )
                seg.static_image_file = png_rel
            except Exception as exc:
                hiccups.append(
                    Hiccup(
                        "static_png_extract_failed",
                        "error",
                        f"Failed extracting static slide image for slide {seg.slide_number}: {exc}",
                        slide_number=seg.slide_number,
                    )
                )
        else:
            if source_frame_cursor + slide_frames > source_frames + 1:
                hiccups.append(
                    Hiccup(
                        "timeline_out_of_bounds",
                        "error",
                        f"Slide {seg.slide_number} exceeds master timeline.",
                        slide_number=seg.slide_number,
                    )
                )
            raw_slide = chunks_dir / f"slide_{seg.slide_number:02d}.mp4"
            cut_chunk_frame_exact(
                ffmpeg_bin=ffmpeg_bin,
                input_mp4=master_mp4,
                output_mp4=raw_slide,
                start_frame=source_frame_cursor,
                frame_count=slide_frames,
                fps=fps,
                mute_output=mute_output,
            )
            enforce_chunk_size(ffmpeg_bin, raw_slide, max_chunk_mb, mute_output=mute_output)
            hashed_slide = rename_with_hash(raw_slide)
            decode_check(ffprobe_bin, hashed_slide)
            slide_rel = f"chunks/{hashed_slide.name}"
            legacy.append(
                {
                    "type": "slide",
                    "file": slide_rel,
                    "duration": round(slide_frames / fps, 3),
                    "slide_number": seg.slide_number,
                    "label": seg.label,
                    "loop": True,
                    "asset_kind": "video",
                }
            )
            extended.append(
                {
                    "id": f"slide_{seg.slide_number:02d}",
                    "type": "slide",
                    "slide_index": seg.slide_number,
                    "start_sec": round(slide_start, 3),
                    "end_sec": round(slide_end, 3),
                    "source_start_sec": round(source_slide_start, 3),
                    "source_end_sec": round(source_slide_end, 3),
                    "duration_sec": round(slide_frames / fps, 3),
                    "file": slide_rel,
                    "loop_default": True,
                    "asset_kind": "video",
                }
            )
        source_frame_cursor += slide_frames
        timeline_frame_cursor += slide_frames

    if abs(source_frames - source_frame_cursor) > 1:
        hiccups.append(
            Hiccup(
                "timeline_drift_frames_source",
                "error",
                (
                    "Source frame cursor mismatch after segmentation: "
                    f"source={source_frames} consumed={source_frame_cursor}"
                ),
            )
        )
    expected_timeline_frames = 0
    for seg in segments:
        if seg.transition_type != "none" and seg.transition_duration_s > 0:
            expected_timeline_frames += int(round(seg.transition_duration_s * fps))
        expected_timeline_frames += max(1, int(round(seg.slide_duration_s * fps)))
    if abs(expected_timeline_frames - timeline_frame_cursor) > 1:
        hiccups.append(
            Hiccup(
                "timeline_drift_frames_timeline",
                "error",
                (
                    "Timeline frame cursor mismatch after segmentation: "
                    f"expected={expected_timeline_frames} got={timeline_frame_cursor}"
                ),
            )
        )
    meta = {
        "timeline_skip_frames": skipped_source_frames,
        "timeline_skip_sec": round(skipped_source_frames / float(fps), 3),
        "source_frames": source_frames,
        "timeline_frames": timeline_frame_cursor,
    }
    return legacy, extended, hiccups, meta


def _resolve_timing_decisions(
    *,
    features: list[SlideFeature],
    overrides: dict[str, Any],
    default_static_sec: float,
    default_transition_sec: float,
    intrinsic_slide_map: dict[int, float | None] | None = None,
    prefer_intrinsic_when_no_rewrite: bool = False,
) -> tuple[list[TimingDecision], list[Hiccup]]:
    decisions: list[TimingDecision] = []
    hiccups: list[Hiccup] = []
    slide_overrides = overrides.get("slides", {}) if isinstance(overrides, dict) else {}

    for feature in features:
        override = slide_overrides.get(str(feature.slide_number), {})
        slide_reason = "static_default"
        transition_reason = "authored"

        intrinsic_slide = (
            intrinsic_slide_map.get(feature.slide_number)
            if isinstance(intrinsic_slide_map, dict)
            else None
        )

        if override.get("slide_sec") is not None:
            slide_duration = float(override["slide_sec"])
            slide_reason = "override"
        elif prefer_intrinsic_when_no_rewrite:
            if intrinsic_slide is not None:
                slide_duration = float(intrinsic_slide)
                slide_reason = "intrinsic_no_rewrite"
            else:
                slide_duration = float(default_static_sec)
                slide_reason = "static_default_no_rewrite"
        elif feature.visible_media_count > 0:
            if feature.max_visible_media_duration_s is not None:
                slide_duration = float(feature.max_visible_media_duration_s)
                slide_reason = "media_max"
            else:
                slide_duration = float(default_static_sec)
                slide_reason = "media_unresolved_fallback"
        else:
            slide_duration = float(default_static_sec)
            slide_reason = "static_default"

        transition_type = str(feature.transition_type or "none")
        transition_duration = float(feature.transition_duration_s or 0.0)
        if override.get("transition_sec") is not None:
            transition_duration = float(override["transition_sec"])
            transition_reason = "override"

        if transition_type == "none":
            transition_duration = 0.0
            transition_reason = "none"
        elif transition_duration <= 0:
            transition_duration = float(default_transition_sec)
            transition_reason = "default_transition"
        if 0.0 < transition_duration <= TINY_TRANSITION_SEC:
            transition_duration = 0.0
            transition_type = "none"
            transition_reason = "tiny_as_none"
        if transition_type == "none":
            transition_duration = 0.0

        asset_kind = "image" if feature.static_classification == "static" else "video"
        if override.get("asset_kind") in ("video", "image"):
            asset_kind = str(override["asset_kind"])
            slide_reason = "override"

        if feature.visible_media_count > 0 and feature.unresolved_visible_media_count > 0:
            hiccups.append(
                Hiccup(
                    "visible_media_duration_unresolved",
                    "error",
                    f"Slide {feature.slide_number} has unresolved visible media durations.",
                    slide_number=feature.slide_number,
                )
            )
        if feature.unsupported_media_count > 0:
            hiccups.append(
                Hiccup(
                    "unsupported_media_reference",
                    "warning",
                    f"Slide {feature.slide_number} has unsupported visible media references.",
                    slide_number=feature.slide_number,
                )
            )

        decisions.append(
            TimingDecision(
                slide_number=feature.slide_number,
                label=feature.label,
                slide_duration_s=round(max(0.1, slide_duration), 3),
                transition_duration_s=round(max(0.0, transition_duration), 3),
                transition_type=transition_type,
                slide_reason=slide_reason,
                transition_reason=transition_reason,
                asset_kind=asset_kind,
                static_image_file=f"assets/slide_{feature.slide_number:02d}.png"
                if asset_kind == "image"
                else None,
            )
        )
    return decisions, hiccups


def _collect_intrinsic_slide_map(
    xml_rows: dict[int, Any], com_rows: dict[int, Any]
) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    slide_numbers = sorted(set(list(xml_rows.keys()) + list(com_rows.keys())))
    for slide in slide_numbers:
        com = com_rows.get(slide)
        xml = xml_rows.get(slide)
        com_adv = getattr(com, "advance_after_s", None) if com is not None else None
        xml_adv = getattr(xml, "advance_after_s", None) if xml is not None else None
        xml_media = getattr(xml, "media_timing_s", None) if xml is not None else None
        chosen = com_adv if com_adv is not None else xml_adv
        if chosen is None:
            chosen = xml_media
        out[slide] = round(float(chosen), 3) if chosen is not None else None
    return out


def _apply_no_rewrite_export_compat(
    *,
    decisions: list[TimingDecision],
    features: list[SlideFeature],
    master_mp4: Path,
    ffprobe_bin: Path,
    fps: int,
    default_transition_sec: float,
) -> tuple[list[TimingDecision], dict[int, int], list[Hiccup]]:
    hiccups: list[Hiccup] = []
    source_gap_frames_by_slide: dict[int, int] = {}
    source_frames = int(round(float(ffprobe_video_stream_info(ffprobe_bin, master_mp4)["nb_frames"])))
    expected_frames = _expected_frames(decisions, fps)
    delta = source_frames - expected_frames
    if abs(delta) <= 1:
        return decisions, source_gap_frames_by_slide, hiccups
    if delta < 0:
        hiccups.append(
            Hiccup(
                "no_rewrite_frame_underflow",
                "error",
                (
                    "No-rewrite export produced fewer frames than timing decisions "
                    f"(source={source_frames} expected={expected_frames} delta={delta})."
                ),
            )
        )
        return decisions, source_gap_frames_by_slide, hiccups

    tiny_map = {
        f.slide_number: f.transition_type
        for f in features
        if f.transition_type != "none" and 0.0 < float(f.transition_duration_s or 0.0) <= TINY_TRANSITION_SEC
    }
    if not tiny_map:
        hiccups.append(
            Hiccup(
                "no_rewrite_frame_drift",
                "error",
                (
                    "No-rewrite export drift detected but no authored tiny transitions available "
                    f"for compatibility adjustment (delta_frames={delta})."
                ),
            )
        )
        return decisions, source_gap_frames_by_slide, hiccups

    compat_transition = max(2.0, float(default_transition_sec))
    compat_frames = int(round(compat_transition * fps))
    remaining = delta

    for decision in decisions:
        if decision.slide_number not in tiny_map:
            continue
        if decision.transition_duration_s > 0:
            continue
        if remaining <= 0:
            continue
        take = min(compat_frames, remaining)
        if take <= 0:
            continue
        source_gap_frames_by_slide[decision.slide_number] = take
        remaining -= take
        if abs(remaining) <= 1:
            break

    if abs(remaining) > 1:
        hiccups.append(
            Hiccup(
                "no_rewrite_frame_drift_unresolved",
                "error",
                (
                    "No-rewrite compatibility could not resolve export-only frame drift: "
                    f"source={source_frames} expected={expected_frames} remaining_delta={remaining}"
                ),
            )
        )
        return decisions, source_gap_frames_by_slide, hiccups

    hiccups.append(
        Hiccup(
            "tiny_transition_export_gap_skip_applied",
            "warning",
            (
                "Applied no-rewrite export compatibility by skipping source-only tiny-transition "
                f"frames while preserving authored transition semantics (delta_frames={delta})."
            ),
            details={
                "source_gap_frames_by_slide": source_gap_frames_by_slide,
                "compat_frames_per_tiny_transition": compat_frames,
            },
        )
    )
    return decisions, source_gap_frames_by_slide, hiccups


def _expected_frames(decisions: list[TimingDecision], fps: int) -> int:
    total = 0
    for row in decisions:
        total += int(round(float(row.slide_duration_s) * fps))
        total += int(round(float(row.transition_duration_s) * fps))
    return total


def _merge_com_into_features(
    features: list[SlideFeature], com_rows: dict[int, Any]
) -> None:
    for feature in features:
        com = com_rows.get(feature.slide_number)
        if com is None:
            continue
        com_transition_dur = getattr(com, "transition_duration_s", None)
        if com_transition_dur is not None:
            feature.transition_duration_s = round(max(0.0, float(com_transition_dur)), 3)
        if feature.transition_type == "none" and feature.transition_duration_s > 0:
            feature.transition_type = "custom"


def _load_pipeline_overrides(path: str | None) -> dict[str, Any]:
    if not path:
        return {"defaults": {}, "slides": {}, "deck_meta": {}, "strict": {}}
    p = Path(path).expanduser().resolve()
    compat = load_overrides(p)
    raw = read_json(p)
    compat["deck_meta"] = raw.get("deck_meta", {})
    compat["strict"] = raw.get("strict", {})
    return compat


def _profile_settings(profile: str) -> dict[str, Any]:
    if profile not in PROFILE_PRESETS:
        allowed = ", ".join(sorted(PROFILE_PRESETS.keys()))
        raise PipelineError(f"Unknown profile '{profile}'. Allowed: {allowed}")
    return dict(PROFILE_PRESETS[profile])


def _append_hiccups(output_dir: Path, stage: str, hiccups: list[Hiccup]) -> None:
    report_path = output_dir / "hiccup_report.json"
    if report_path.exists():
        report = read_json(report_path)
    else:
        report = {"version": 3, "created_at_utc": utc_now_iso(), "hiccups": []}

    rows = report.get("hiccups", [])
    if not isinstance(rows, list):
        rows = []
    for hiccup in hiccups:
        payload = hiccup.to_dict()
        payload["stage"] = stage
        rows.append(payload)
    report["hiccups"] = rows
    report["error_count"] = len([r for r in rows if str(r.get("severity")) == "error"])
    report["warning_count"] = len([r for r in rows if str(r.get("severity")) != "error"])
    report["updated_at_utc"] = utc_now_iso()
    write_json(report_path, report)


def _has_error_hiccups(hiccups: list[Hiccup]) -> bool:
    return any(h.severity == "error" for h in hiccups)


def _copy_player_template(output_dir: Path) -> None:
    src = Path(__file__).resolve().parent.parent / "player" / "index.html"
    dst = output_dir / "index.html"
    if not src.exists():
        raise PipelineError(f"Player template not found: {src}")
    shutil.copy2(src, dst)


def _git_add_commit_push(
    *,
    repo_root: Path,
    pathspec: str,
    remote: str,
    branch: str,
    message: str,
) -> None:
    subprocess.run(["git", "add", pathspec], cwd=repo_root, check=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", pathspec],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    if not status.stdout.strip():
        print("No git changes to commit after publish.")
        return
    subprocess.run(["git", "commit", "-m", message], cwd=repo_root, check=True)
    subprocess.run(["git", "push", remote, branch], cwd=repo_root, check=True)
    print("Publish commit pushed.")


def sanitize_id(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "presentation"
