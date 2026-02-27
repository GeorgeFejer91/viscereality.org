from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

from .dependencies import discover_ffmpeg_tools
from .exceptions import PipelineError
from .manifest import build_manifest
from .media import (
    cut_chunk_precise,
    enforce_chunk_size,
    ffprobe_duration,
    fit_video_duration,
    normalize_video,
)
from .models import ResolvedSegment
from .ppt_export import export_ppt_to_video
from .timing import (
    build_timing_config_payload,
    load_overrides,
    parse_pptx_timing_xml,
    probe_timing_com,
    resolve_segments,
    segments_from_config,
)
from .utils import read_json, write_json


def analyze_command(args: argparse.Namespace) -> None:
    pptx_path = Path(args.pptx).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_sources = parse_pptx_timing_xml(pptx_path)
    com_sources = probe_timing_com(pptx_path) if args.com_probe else {}
    if args.com_probe and not com_sources:
        print("WARNING: COM probe requested but unavailable; using XML timings only.")
    overrides = load_overrides(Path(args.overrides_file)) if args.overrides_file else None
    segments = resolve_segments(
        xml_sources=xml_sources,
        com_sources=com_sources,
        timing_mode=args.timing_mode,
        default_slide_sec=args.default_slide_sec,
        default_transition_sec=args.default_transition_sec,
        overrides=overrides,
    )

    config = build_timing_config_payload(
        segments=segments,
        default_slide_sec=args.default_slide_sec,
        default_transition_sec=args.default_transition_sec,
        timing_mode=args.timing_mode,
    )
    config["analysis"] = {
        "com_probe_used": bool(com_sources),
        "slide_count": len(segments),
    }
    config_path = output_dir / "timing_config.json"
    write_json(config_path, config)

    print(f"Slides analyzed: {len(segments)}")
    print(f"Timing config written to: {config_path}")


def chunk_command(args: argparse.Namespace) -> None:
    mp4_path = Path(args.mp4).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(Path(args.config).expanduser().resolve())
    segments = segments_from_config(config)
    ffmpeg_bin, ffprobe_bin = discover_ffmpeg_tools(args.ffmpeg_bin, args.ffprobe_bin)

    title = str(config.get("title") or config.get("presentation_id") or "Presentation")
    presentation_id = sanitize_id(config.get("presentation_id") or mp4_path.stem)
    manifest = _chunk_from_segments(
        mp4_path=mp4_path,
        output_dir=output_dir,
        segments=segments,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        title=title,
        presentation_id=presentation_id,
        source_ppt=None,
        duration_tolerance=float(args.duration_tolerance),
        max_chunk_mb=float(args.max_chunk_mb),
        mute_output=bool(args.mute_output),
        encoding={
            "fps": None,
            "height": None,
            "mute": bool(args.mute_output),
            "timing_mode": "config",
        },
        master_file=mp4_path.name,
    )
    if args.generate_player:
        _copy_player_template(output_dir)
    print(f"Wrote manifest: {output_dir / 'manifest.json'}")
    print(f"Created chunks: {len(manifest['chunks'])}")


def run_command(args: argparse.Namespace) -> None:
    pptx_path = Path(args.pptx).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir = output_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    presentation_id = sanitize_id(args.presentation_id or pptx_path.stem)
    ffmpeg_bin, ffprobe_bin = discover_ffmpeg_tools(args.ffmpeg_bin, args.ffprobe_bin)

    xml_sources = parse_pptx_timing_xml(pptx_path)
    com_sources: dict[int, Any] = {}
    if args.timing_mode in ("ppt", "hybrid"):
        com_sources = probe_timing_com(pptx_path)
        if not com_sources:
            print("WARNING: COM probe unavailable, falling back to XML timings for resolution.")

    overrides = load_overrides(Path(args.overrides_file)) if args.overrides_file else None
    segments = resolve_segments(
        xml_sources=xml_sources,
        com_sources=com_sources,
        timing_mode=args.timing_mode,
        default_slide_sec=float(args.default_slide_sec),
        default_transition_sec=float(args.default_transition_sec),
        overrides=overrides,
    )

    config_payload = build_timing_config_payload(
        segments=segments,
        default_slide_sec=float(args.default_slide_sec),
        default_transition_sec=float(args.default_transition_sec),
        timing_mode=args.timing_mode,
    )
    config_payload["presentation_id"] = presentation_id
    config_payload["title"] = args.title or pptx_path.stem
    config_payload["export"] = {"fps": int(args.fps), "height": int(args.height), "mute": bool(args.mute_output)}
    config_payload["analysis"] = {
        "com_probe_used": bool(com_sources),
        "slide_count": len(segments),
    }
    write_json(output_dir / "timing_config.json", config_payload)

    raw_mp4 = build_dir / f"{presentation_id}_raw.mp4"
    master_mp4 = output_dir / f"{presentation_id}_master.mp4"
    chunk_source_mp4 = master_mp4

    print("Exporting PowerPoint to video (COM)...")
    export_ppt_to_video(
        pptx_path=pptx_path,
        output_mp4=raw_mp4,
        segments=segments,
        rewrite_timings=bool(args.rewrite_timings),
        fps=int(args.fps),
        vert_resolution=int(args.height),
        quality=int(args.quality),
        default_slide_duration_s=float(args.default_slide_sec),
    )
    print("Normalizing exported video...")
    normalize_video(
        ffmpeg_bin=ffmpeg_bin,
        input_mp4=raw_mp4,
        output_mp4=master_mp4,
        fps=int(args.fps),
        height=int(args.height),
        mute_output=bool(args.mute_output),
    )

    expected_duration = _expected_duration(segments)
    normalized_duration = ffprobe_duration(ffprobe_bin, master_mp4)
    normalized_drift = abs(normalized_duration - expected_duration)
    if bool(args.fit_duration) and normalized_drift > float(args.duration_tolerance):
        max_fit_ratio = float(args.max_fit_ratio)
        if expected_duration <= 0:
            raise PipelineError("Expected duration is non-positive; cannot fit duration.")
        ratio = normalized_duration / expected_duration
        if ratio < 1 / max_fit_ratio or ratio > max_fit_ratio:
            raise PipelineError(
                f"Duration fit rejected. Ratio {ratio:.4f} is outside max range "
                f"[{1/max_fit_ratio:.4f}, {max_fit_ratio:.4f}]"
            )
        print(
            f"Fitting master duration to expected timeline "
            f"({normalized_duration:.3f}s -> {expected_duration:.3f}s)..."
        )
        fitted_mp4 = output_dir / f"{presentation_id}_master_fitted.mp4"
        speed_factor = fit_video_duration(
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            input_mp4=master_mp4,
            output_mp4=fitted_mp4,
            target_duration_s=expected_duration,
            fps=int(args.fps),
            height=int(args.height),
            mute_output=bool(args.mute_output),
        )
        print(f"Applied duration fit speed factor: {speed_factor:.6f}")
        chunk_source_mp4 = fitted_mp4

    _chunk_from_segments(
        mp4_path=chunk_source_mp4,
        output_dir=output_dir,
        segments=segments,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        title=args.title or pptx_path.stem,
        presentation_id=presentation_id,
        source_ppt=pptx_path,
        duration_tolerance=float(args.duration_tolerance),
        max_chunk_mb=float(args.max_chunk_mb),
        mute_output=bool(args.mute_output),
        encoding={
            "fps": int(args.fps),
            "height": int(args.height),
            "mute": bool(args.mute_output),
            "timing_mode": args.timing_mode,
        },
        master_file=chunk_source_mp4.name,
    )

    if args.generate_player:
        _copy_player_template(output_dir)

    print("\nRun completed.")
    print(f"Output directory: {output_dir}")
    print(f"Master MP4: {master_mp4}")
    print(f"Manifest: {output_dir / 'manifest.json'}")


def _chunk_from_segments(
    *,
    mp4_path: Path,
    output_dir: Path,
    segments: list[ResolvedSegment],
    ffmpeg_bin: Path,
    ffprobe_bin: Path,
    title: str,
    presentation_id: str,
    source_ppt: Path | None,
    duration_tolerance: float,
    max_chunk_mb: float,
    mute_output: bool,
    encoding: dict[str, Any],
    master_file: str | None,
) -> dict[str, Any]:
    if not mp4_path.exists():
        raise PipelineError(f"Video file not found: {mp4_path}")
    if not segments:
        raise PipelineError("No segments available for chunking.")

    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    video_duration = ffprobe_duration(ffprobe_bin, mp4_path)
    expected_duration = _expected_duration(segments)
    drift = abs(video_duration - expected_duration)
    if drift > duration_tolerance:
        raise PipelineError(
            f"Duration mismatch. Video={video_duration:.3f}s Config={expected_duration:.3f}s "
            f"Drift={drift:.3f}s exceeds tolerance={duration_tolerance:.3f}s."
        )

    cursor = 0.0
    legacy_chunks: list[dict[str, Any]] = []
    extended_segments: list[dict[str, Any]] = []

    for seg in segments:
        if seg.transition_type != "none" and seg.transition_duration_s > 0:
            start = cursor
            end = round(cursor + seg.transition_duration_s, 3)
            fname = f"trans_{seg.slide_number:02d}.mp4"
            fpath = chunks_dir / fname
            cut_chunk_precise(
                ffmpeg_bin=ffmpeg_bin,
                input_mp4=mp4_path,
                output_mp4=fpath,
                start_s=start,
                duration_s=seg.transition_duration_s,
                mute_output=mute_output,
            )
            enforce_chunk_size(ffmpeg_bin, fpath, max_chunk_mb, mute_output=mute_output)
            actual = ffprobe_duration(ffprobe_bin, fpath)
            if abs(actual - seg.transition_duration_s) > max(0.25, duration_tolerance):
                raise PipelineError(
                    f"Transition chunk duration mismatch for {fname}: expected "
                    f"{seg.transition_duration_s:.3f}s got {actual:.3f}s"
                )
            legacy_chunks.append(
                {
                    "type": "transition",
                    "file": f"chunks/{fname}",
                    "duration": round(seg.transition_duration_s, 3),
                    "slide_from": seg.slide_number - 1 if seg.slide_number > 1 else 0,
                    "slide_to": seg.slide_number,
                }
            )
            extended_segments.append(
                {
                    "id": f"trans_{seg.slide_number:02d}",
                    "type": "transition",
                    "slide_index": seg.slide_number,
                    "start_sec": round(start, 3),
                    "end_sec": end,
                    "duration_sec": round(seg.transition_duration_s, 3),
                    "file": f"chunks/{fname}",
                    "loop_default": False,
                }
            )
            cursor = end

        start = cursor
        end = round(cursor + seg.slide_duration_s, 3)
        fname = f"slide_{seg.slide_number:02d}.mp4"
        fpath = chunks_dir / fname
        cut_chunk_precise(
            ffmpeg_bin=ffmpeg_bin,
            input_mp4=mp4_path,
            output_mp4=fpath,
            start_s=start,
            duration_s=seg.slide_duration_s,
            mute_output=mute_output,
        )
        enforce_chunk_size(ffmpeg_bin, fpath, max_chunk_mb, mute_output=mute_output)
        actual = ffprobe_duration(ffprobe_bin, fpath)
        if abs(actual - seg.slide_duration_s) > max(0.25, duration_tolerance):
            raise PipelineError(
                f"Slide chunk duration mismatch for {fname}: expected "
                f"{seg.slide_duration_s:.3f}s got {actual:.3f}s"
            )

        legacy_chunks.append(
            {
                "type": "slide",
                "file": f"chunks/{fname}",
                "duration": round(seg.slide_duration_s, 3),
                "slide_number": seg.slide_number,
                "label": seg.label,
                "loop": True,
            }
        )
        extended_segments.append(
            {
                "id": f"slide_{seg.slide_number:02d}",
                "type": "slide",
                "slide_index": seg.slide_number,
                "start_sec": round(start, 3),
                "end_sec": end,
                "duration_sec": round(seg.slide_duration_s, 3),
                "file": f"chunks/{fname}",
                "loop_default": True,
            }
        )
        cursor = end

    final_drift = abs(video_duration - cursor)
    if final_drift > duration_tolerance:
        raise PipelineError(
            f"Timeline cursor mismatch. Cursor={cursor:.3f}s Video={video_duration:.3f}s "
            f"Drift={final_drift:.3f}s exceeds tolerance={duration_tolerance:.3f}s."
        )

    manifest = build_manifest(
        presentation_id=presentation_id,
        source_ppt=source_ppt,
        title=title,
        segments=segments,
        chunk_entries_legacy=legacy_chunks,
        chunk_entries_extended=extended_segments,
        encoding=encoding,
        master_file=master_file,
    )
    write_json(output_dir / "manifest.json", manifest)

    _validate_manifest_files(output_dir, manifest)
    return manifest


def _validate_manifest_files(output_dir: Path, manifest: dict[str, Any]) -> None:
    missing = []
    for chunk in manifest.get("chunks", []):
        file_rel = chunk.get("file")
        if not file_rel:
            continue
        file_path = output_dir / file_rel
        if not file_path.exists() or file_path.stat().st_size == 0:
            missing.append(str(file_rel))
    if missing:
        raise PipelineError(f"Manifest references missing/empty chunk files: {missing}")


def _expected_duration(segments: list[ResolvedSegment]) -> float:
    total = 0.0
    for seg in segments:
        total += seg.slide_duration_s
        if seg.transition_type != "none":
            total += seg.transition_duration_s
    return round(total, 3)


def _copy_player_template(output_dir: Path) -> None:
    src = Path(__file__).resolve().parent.parent / "player" / "index.html"
    dst = output_dir / "index.html"
    if not src.exists():
        raise PipelineError(f"Player template not found: {src}")
    shutil.copy2(src, dst)


def sanitize_id(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "presentation"
