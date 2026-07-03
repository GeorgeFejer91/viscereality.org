from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .build import build_presentation, inspect_pptx
from .config import PROFILE_PRESETS, PresenterConfig, load_config
from .errors import PresenterError
from .family import build_family, inspect_family, oracle_qa_family, publish_family, visual_audit_family
from .publish import publish_build
from .qa import (
    run_candidate_sweep,
    run_media_phase_calibration,
    run_morph_progress_calibration,
    run_qa,
    run_static_fallback_generation,
    run_track_progress_calibration,
    run_transition_time_calibration,
    run_visual_audit,
)
from .reference import export_reference_mp4
from .utils import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pptx-html-presenter",
        description="Compile PPTX decks into static HTML scene presentations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Inspect a PPTX and write a reusable report.")
    inspect.add_argument("pptx")
    inspect.add_argument("-o", "--output-dir")

    build = sub.add_parser("build", help="Build static HTML presentation output.")
    build.add_argument("pptx")
    build.add_argument("--out", required=True)
    build.add_argument("--config")
    build.add_argument("--profile", choices=sorted(PROFILE_PRESETS), default=None)
    build.add_argument("--title")
    build.add_argument("--slug")
    build.add_argument("--asset-mode", choices=["copy", "source-only", "manifest-only"], default=None)
    build.add_argument("--soft-max-mb", type=float)
    build.add_argument("--hard-max-mb", type=float)
    build.add_argument("--gif-transcode", action=argparse.BooleanOptionalAction, default=None)
    build.add_argument("--video-transcode", action=argparse.BooleanOptionalAction, default=None)
    build.add_argument("--image-optimize", action=argparse.BooleanOptionalAction, default=None)
    build.add_argument("--allow-oversize-assets", action=argparse.BooleanOptionalAction, default=None)
    build.add_argument("--qa", action="store_true")
    build.add_argument("--visual-audit", action="store_true")
    build.add_argument("--reference-mp4")
    build.add_argument("--ffmpeg-bin")
    build.add_argument("--node-bin")
    build.add_argument("--playwright-dir")
    build.add_argument("--calibrate-qa", action="store_true")
    build.add_argument("--qa-slide-hold-sec", type=float)
    build.add_argument("--qa-settled-offset-sec", type=float)
    build.add_argument("--qa-transition-reference-lead-fraction", type=float)

    qa = sub.add_parser("qa", help="Create QA sample plan and optional reference frames.")
    qa.add_argument("build_dir")
    qa.add_argument("--reference", dest="reference_mp4")
    qa.add_argument("--ffmpeg-bin")
    qa.add_argument("--node-bin")
    qa.add_argument("--playwright-dir")
    qa.add_argument("--reuse-html", action="store_true")
    qa.add_argument("--calibrate", action="store_true")
    qa.add_argument("--visual-audit", action="store_true")
    qa.add_argument("--slide-hold-sec", type=float)
    qa.add_argument("--settled-offset-sec", type=float)
    qa.add_argument("--transition-reference-lead-fraction", type=float)
    qa.add_argument("--slides", help="Comma-separated slide numbers to sample.")

    visual_audit = sub.add_parser("visual-audit", help="Capture every slide and transition for visual overlap/layer review.")
    visual_audit.add_argument("build_dir")
    visual_audit.add_argument("--node-bin")
    visual_audit.add_argument("--playwright-dir")

    candidate_sweep = sub.add_parser(
        "candidate-sweep",
        help="Render and score alternate HTML candidates for one QA sample.",
    )
    candidate_sweep.add_argument("build_dir")
    candidate_sweep.add_argument("--sample", required=True, dest="sample_id")
    candidate_sweep.add_argument(
        "--vary",
        required=True,
        choices=["progress", "track-progress", "phase", "media-phase", "media-clock", "phase-offset", "media-phase-offset"],
        help="Field to vary: global Morph progress, one track's Morph progress, one track's media clock, or a media-clock offset.",
    )
    candidate_sweep.add_argument("--values", required=True, help="Comma list or start:end:step range.")
    candidate_sweep.add_argument("--track-id", help="Track id, comma track list, or all. Required for phase/media-clock sweeps.")
    candidate_sweep.add_argument("--reference-frame", help="Reference PNG for the source sample.")
    candidate_sweep.add_argument("--reference", dest="reference_mp4", help="Reference MP4 if the frame is missing.")
    candidate_sweep.add_argument("--ffmpeg-bin")
    candidate_sweep.add_argument("--node-bin")
    candidate_sweep.add_argument("--playwright-dir")
    candidate_sweep.add_argument("--reuse-html", action="store_true")

    media_phase = sub.add_parser("media-phase", help="Estimate per-object media phase offsets against a reference MP4.")
    media_phase.add_argument("build_dir")
    media_phase.add_argument("--reference", required=True, dest="reference_mp4")
    media_phase.add_argument("--ffmpeg-bin")
    media_phase.add_argument("--slides", help="Comma-separated slide numbers to inspect.")
    media_phase.add_argument("--step-sec", type=float, default=0.5)
    media_phase.add_argument("--search-sec", type=float, default=12.0)
    media_phase.add_argument("--min-score", type=float, default=0.70)
    media_phase.add_argument("--include-transitions", action="store_true")
    media_phase.add_argument("--apply", action="store_true")
    media_phase.add_argument("--overrides-out", help="Write config-ready media_phase_overrides JSON.")

    transition_time = sub.add_parser(
        "transition-time",
        help="Estimate per-transition PowerPoint MP4 timing offsets against captured HTML frames.",
    )
    transition_time.add_argument("build_dir")
    transition_time.add_argument("--reference", required=True, dest="reference_mp4")
    transition_time.add_argument("--ffmpeg-bin")
    transition_time.add_argument("--node-bin")
    transition_time.add_argument("--playwright-dir")
    transition_time.add_argument("--slides", help="Comma-separated source slide numbers to inspect.")
    transition_time.add_argument("--fps", type=int, default=8)
    transition_time.add_argument("--window-sec", type=float, default=1.0)
    transition_time.add_argument("--min-score", type=float, default=0.55)
    transition_time.add_argument("--apply", action="store_true")
    transition_time.add_argument("--reuse-html", action=argparse.BooleanOptionalAction, default=True)
    transition_time.add_argument("--overrides-out", help="Write config-ready transition_time_overrides JSON.")

    morph_progress = sub.add_parser(
        "morph-progress",
        help="Estimate per-transition Morph progress maps against captured HTML candidates.",
    )
    morph_progress.add_argument("build_dir")
    morph_progress.add_argument("--reference", required=True, dest="reference_mp4")
    morph_progress.add_argument("--ffmpeg-bin")
    morph_progress.add_argument("--node-bin")
    morph_progress.add_argument("--playwright-dir")
    morph_progress.add_argument("--slides", help="Comma-separated source slide numbers to inspect.")
    morph_progress.add_argument("--candidate-step", type=float, default=0.05)
    morph_progress.add_argument("--min-score", type=float, default=0.55)
    morph_progress.add_argument(
        "--compare-mode",
        choices=["auto", "anchors", "full"],
        default="auto",
        help="Compare full frames or crop around inferred panel/anchor tracks.",
    )
    morph_progress.add_argument("--reuse-html", action="store_true")
    morph_progress.add_argument("--overrides-out", help="Write config-ready transition_progress_overrides JSON.")

    track_progress = sub.add_parser(
        "track-progress",
        help="Estimate per-track Morph progress maps against captured HTML candidates.",
    )
    track_progress.add_argument("build_dir")
    track_progress.add_argument("--reference", required=True, dest="reference_mp4")
    track_progress.add_argument("--ffmpeg-bin")
    track_progress.add_argument("--node-bin")
    track_progress.add_argument("--playwright-dir")
    track_progress.add_argument("--slides", help="Comma-separated source slide numbers to inspect.")
    track_progress.add_argument("--tracks", help="Comma-separated track ids. Defaults to panel/container tracks.")
    track_progress.add_argument("--progresses", help="Comma-separated raw Morph progress values to inspect.")
    track_progress.add_argument("--candidate-step", type=float, default=0.05)
    track_progress.add_argument("--min-score", type=float, default=0.0)
    track_progress.add_argument("--min-improvement", type=float, default=0.002)
    track_progress.add_argument("--stability-weight", type=float, default=0.02)
    track_progress.add_argument("--reuse-html", action="store_true")
    track_progress.add_argument("--overrides-out", help="Write config-ready transition_track_progress_overrides JSON.")

    static_fallback = sub.add_parser(
        "static-fallback",
        help="Generate PowerPoint-rendered static overlay fallbacks with live-media holes.",
    )
    static_fallback.add_argument("build_dir")
    static_fallback.add_argument("--reference", required=True, dest="reference_mp4")
    static_fallback.add_argument("--ffmpeg-bin")
    static_fallback.add_argument("--slides", help="Comma-separated slide numbers to inspect.")
    static_fallback.add_argument("--hole-padding-px", type=int, default=2)
    static_fallback.add_argument("--settled-only", action=argparse.BooleanOptionalAction, default=True)
    static_fallback.add_argument("--overrides-out", help="Write config-ready raster_fallback_overrides JSON.")

    reference = sub.add_parser("reference", help="Export a PowerPoint MP4 reference through COM.")
    reference.add_argument("pptx")
    reference.add_argument("--out", required=True)
    reference.add_argument("--scene", help="Path to deck.scene.json or a build directory for normalized timing export.")
    reference.add_argument("--fps", type=int, default=30)
    reference.add_argument("--height", type=int, default=1080)
    reference.add_argument("--quality", type=int, default=100)
    reference.add_argument("--default-slide-sec", type=float, default=None)
    reference.add_argument("--use-timings", action=argparse.BooleanOptionalAction, default=None)
    reference.add_argument("--clamp-media", action=argparse.BooleanOptionalAction, default=None)
    reference.add_argument("--ffmpeg-bin")

    publish = sub.add_parser("publish", help="Copy a validated build into presentations/<deck>.")
    publish.add_argument("build_dir")
    publish.add_argument("--deck", required=True)
    publish.add_argument("--repo-root")
    publish.add_argument("--force", action="store_true")
    publish.add_argument("--update-shared-decks", action=argparse.BooleanOptionalAction, default=True)

    family = sub.add_parser("family", help="Build, audit, and publish a multi-deck shared-asset family.")
    family_sub = family.add_subparsers(dest="family_command", required=True)

    family_inspect = family_sub.add_parser("inspect", help="Inspect all decks in a family config.")
    family_inspect.add_argument("family_config")

    family_build = family_sub.add_parser("build", help="Build all family decks and hoist shared assets.")
    family_build.add_argument("family_config")
    family_build.add_argument("--ffmpeg-bin")
    family_build.add_argument("--force", action="store_true")

    family_audit = family_sub.add_parser("visual-audit", help="Run full visual audit for all family staging builds.")
    family_audit.add_argument("family_config")
    family_audit.add_argument("--node-bin")
    family_audit.add_argument("--playwright-dir")

    family_oracle = family_sub.add_parser("oracle-qa", help="Run PowerPoint MP4 oracle QA for all family decks.")
    family_oracle.add_argument("family_config")
    family_oracle.add_argument("--ffmpeg-bin")
    family_oracle.add_argument("--node-bin")
    family_oracle.add_argument("--playwright-dir")
    family_oracle.add_argument("--target", choices=["public", "staging"], default="public")
    family_oracle.add_argument("--keep-reference", action="store_true")
    family_oracle.add_argument("--force", action="store_true")
    family_oracle.add_argument("--min-free-gb", type=float)
    family_oracle.add_argument(
        "--transition-reference-lead-fraction",
        type=float,
        help="Override QA reference transition lead fraction for timing calibration.",
    )
    family_oracle.add_argument("--slides", help="Comma-separated slide numbers to sample.")
    family_oracle.add_argument("--decks", help="Comma-separated family deck IDs to run.")

    family_publish = family_sub.add_parser("publish", help="Publish family staging builds to public deck folders.")
    family_publish.add_argument("family_config")
    family_publish.add_argument("--force", action="store_true")
    family_publish.add_argument("--archive-chunked", action=argparse.BooleanOptionalAction, default=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            report = inspect_pptx(
                Path(args.pptx),
                Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
            )
            if not args.output_dir:
                write_json(Path("inspect-report.json"), report)
            print(f"slides={report['slideCount']} assets={report['assetCount']}")
            return 0
        if args.command == "build":
            config = _config_from_args(args)
            report = build_presentation(
                Path(args.pptx),
                Path(args.out).expanduser().resolve(),
                config,
                title=args.title,
                slug=args.slug,
                ffmpeg_bin=args.ffmpeg_bin,
            )
            if args.qa:
                run_qa(
                    Path(args.out),
                    reference_mp4=Path(args.reference_mp4) if args.reference_mp4 else None,
                    ffmpeg_bin=args.ffmpeg_bin,
                    node_bin=args.node_bin,
                    playwright_dir=Path(args.playwright_dir) if args.playwright_dir else None,
                    reuse_html=False,
                calibrate=args.calibrate_qa,
                slide_hold_sec=args.qa_slide_hold_sec,
                settled_offset_sec=args.qa_settled_offset_sec,
                transition_reference_lead_fraction=args.qa_transition_reference_lead_fraction,
                visual_audit=args.visual_audit,
            )
            print(f"built={report['outputDir']} status={report['status']}")
            return 0
        if args.command == "qa":
            report = run_qa(
                Path(args.build_dir),
                reference_mp4=Path(args.reference_mp4) if args.reference_mp4 else None,
                ffmpeg_bin=args.ffmpeg_bin,
                node_bin=args.node_bin,
                playwright_dir=Path(args.playwright_dir) if args.playwright_dir else None,
                reuse_html=args.reuse_html,
                calibrate=args.calibrate,
                slide_hold_sec=args.slide_hold_sec,
                settled_offset_sec=args.settled_offset_sec,
                transition_reference_lead_fraction=args.transition_reference_lead_fraction,
                slides=_parse_slide_filter(args.slides),
                visual_audit=args.visual_audit,
            )
            print(f"qa={report['status']} samples={len(report['samples'])}")
            return 0
        if args.command == "visual-audit":
            report = run_visual_audit(
                Path(args.build_dir),
                node_bin=args.node_bin,
                playwright_dir=Path(args.playwright_dir) if args.playwright_dir else None,
            )
            print(f"visual-audit={report['status']} samples={report['summary']['sampleCount']}")
            return 0
        if args.command == "candidate-sweep":
            report = run_candidate_sweep(
                Path(args.build_dir),
                sample_id=args.sample_id,
                vary=args.vary,
                values=_parse_float_list(args.values),
                track_id=args.track_id,
                reference_frame=Path(args.reference_frame) if args.reference_frame else None,
                reference_mp4=Path(args.reference_mp4) if args.reference_mp4 else None,
                ffmpeg_bin=args.ffmpeg_bin,
                node_bin=args.node_bin,
                playwright_dir=Path(args.playwright_dir) if args.playwright_dir else None,
                reuse_html=args.reuse_html,
            )
            best = report.get("best") or {}
            best_value = best.get("value", "none")
            best_score = best.get("ssim", "none")
            print(
                f"candidate-sweep sample={report['sampleId']} candidates={report['summary']['scoredCount']} best={best_value} ssim={best_score}"
            )
            return 0
        if args.command == "media-phase":
            report = run_media_phase_calibration(
                Path(args.build_dir),
                reference_mp4=Path(args.reference_mp4),
                ffmpeg_bin=args.ffmpeg_bin,
                slides=_parse_slide_filter(args.slides),
                step_sec=args.step_sec,
                search_sec=args.search_sec,
                min_score=args.min_score,
                include_transitions=args.include_transitions,
                apply=args.apply,
            )
            if args.overrides_out:
                payload = {"media_phase_overrides": report.get("configOverrides", [])}
                transition_overrides = report.get("transitionConfigOverrides", [])
                if transition_overrides:
                    payload["transition_media_phase_overrides"] = transition_overrides
                write_json(
                    Path(args.overrides_out).expanduser().resolve(),
                    payload,
                )
            print(
                f"media-phase rows={len(report['rows'])} applied={report['summary']['appliedCount']}"
            )
            return 0
        if args.command == "transition-time":
            report = run_transition_time_calibration(
                Path(args.build_dir),
                reference_mp4=Path(args.reference_mp4),
                ffmpeg_bin=args.ffmpeg_bin,
                node_bin=args.node_bin,
                playwright_dir=Path(args.playwright_dir) if args.playwright_dir else None,
                slides=_parse_slide_filter(args.slides),
                fps=args.fps,
                window_sec=args.window_sec,
                min_score=args.min_score,
                apply=args.apply,
                reuse_html=args.reuse_html,
            )
            if args.overrides_out:
                write_json(
                    Path(args.overrides_out).expanduser().resolve(),
                    {"transition_time_overrides": report.get("configOverrides", [])},
                )
            print(
                f"transition-time samples={report['summary']['sampleCount']} overrides={report['summary']['overrideCount']}"
            )
            return 0
        if args.command == "morph-progress":
            report = run_morph_progress_calibration(
                Path(args.build_dir),
                reference_mp4=Path(args.reference_mp4),
                ffmpeg_bin=args.ffmpeg_bin,
                node_bin=args.node_bin,
                playwright_dir=Path(args.playwright_dir) if args.playwright_dir else None,
                slides=_parse_slide_filter(args.slides),
                candidate_step=args.candidate_step,
                min_score=args.min_score,
                compare_mode=args.compare_mode,
                reuse_html=args.reuse_html,
            )
            if args.overrides_out:
                write_json(
                    Path(args.overrides_out).expanduser().resolve(),
                    {"transition_progress_overrides": report.get("configOverrides", [])},
                )
            print(
                f"morph-progress samples={report['summary']['sampleCount']} overrides={report['summary']['overrideCount']}"
            )
            return 0
        if args.command == "track-progress":
            report = run_track_progress_calibration(
                Path(args.build_dir),
                reference_mp4=Path(args.reference_mp4),
                ffmpeg_bin=args.ffmpeg_bin,
                node_bin=args.node_bin,
                playwright_dir=Path(args.playwright_dir) if args.playwright_dir else None,
                slides=_parse_slide_filter(args.slides),
                tracks=_parse_track_filter(args.tracks),
                progresses=set(_parse_float_list(args.progresses)) if args.progresses else None,
                candidate_step=args.candidate_step,
                min_score=args.min_score,
                min_improvement=args.min_improvement,
                stability_weight=args.stability_weight,
                reuse_html=args.reuse_html,
            )
            if args.overrides_out:
                write_json(
                    Path(args.overrides_out).expanduser().resolve(),
                    {"transition_track_progress_overrides": report.get("configOverrides", [])},
                )
            print(
                f"track-progress samples={report['summary']['sampleCount']} overrides={report['summary']['overrideCount']}"
            )
            return 0
        if args.command == "static-fallback":
            report = run_static_fallback_generation(
                Path(args.build_dir),
                reference_mp4=Path(args.reference_mp4),
                ffmpeg_bin=args.ffmpeg_bin,
                slides=_parse_slide_filter(args.slides),
                hole_padding_px=args.hole_padding_px,
                settled_only=args.settled_only,
            )
            if args.overrides_out:
                write_json(
                    Path(args.overrides_out).expanduser().resolve(),
                    {"raster_fallback_overrides": report.get("configOverrides", [])},
                )
            print(
                f"static-fallback slides={report['summary']['count']} bytes={report['summary']['totalBytes']}"
            )
            return 0
        if args.command == "reference":
            out = export_reference_mp4(
                Path(args.pptx),
                Path(args.out),
                scene_path=Path(args.scene) if args.scene else None,
                use_timings=args.use_timings,
                clamp_media=args.clamp_media,
                ffmpeg_bin=args.ffmpeg_bin,
                fps=args.fps,
                height=args.height,
                quality=args.quality,
                default_slide_sec=args.default_slide_sec,
            )
            print(f"reference={out}")
            return 0
        if args.command == "publish":
            report = publish_build(
                Path(args.build_dir),
                deck_id=args.deck,
                repo_root=Path(args.repo_root).expanduser().resolve() if args.repo_root else None,
                force=args.force,
                update_shared_decks=args.update_shared_decks,
            )
            print(f"published={report['target']}")
            return 0
        if args.command == "family":
            config_path = Path(args.family_config).expanduser().resolve()
            if args.family_command == "inspect":
                report = inspect_family(config_path)
                print(
                    f"family-inspect={report['status']} decks={len(report['preflight']['decks'])} "
                    f"unique-media-mb={report['preflight']['estimatedUniqueSourceMediaMb']}"
                )
                return 0
            if args.family_command == "build":
                report = build_family(
                    config_path,
                    ffmpeg_bin=args.ffmpeg_bin,
                    force=args.force,
                )
                print(f"family-build={report['status']} decks={len(report['decks'])}")
                return 0
            if args.family_command == "visual-audit":
                report = visual_audit_family(
                    config_path,
                    node_bin=args.node_bin,
                    playwright_dir=Path(args.playwright_dir) if args.playwright_dir else None,
                )
                print(f"family-visual-audit={report['status']} decks={len(report['decks'])}")
                return 0
            if args.family_command == "oracle-qa":
                report = oracle_qa_family(
                    config_path,
                    ffmpeg_bin=args.ffmpeg_bin,
                    node_bin=args.node_bin,
                    playwright_dir=Path(args.playwright_dir) if args.playwright_dir else None,
                    target=args.target,
                    keep_reference=args.keep_reference,
                    force=args.force,
                    min_free_gb=args.min_free_gb,
                    slides=_parse_slide_filter(args.slides),
                    deck_ids=_parse_deck_filter(args.decks),
                    transition_reference_lead_fraction=args.transition_reference_lead_fraction,
                )
                print(f"family-oracle-qa={report['status']} decks={len(report['decks'])}")
                return 0
            if args.family_command == "publish":
                report = publish_family(
                    config_path,
                    force=args.force,
                    archive_chunked=args.archive_chunked,
                )
                print(f"family-publish={report['status']} decks={len(report['decks'])}")
                return 0
    except PresenterError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _config_from_args(args: argparse.Namespace) -> PresenterConfig:
    config = load_config(Path(args.config).expanduser().resolve() if args.config else None)
    profile = PROFILE_PRESETS[args.profile] if args.profile else config.profile
    asset_policy = config.asset_policy
    overrides = {}
    if args.asset_mode is not None:
        overrides["mode"] = args.asset_mode
    if args.soft_max_mb is not None:
        overrides["soft_max_mb"] = args.soft_max_mb
    if args.hard_max_mb is not None:
        overrides["hard_max_mb"] = args.hard_max_mb
    if args.gif_transcode is not None:
        overrides["transcode_gif"] = args.gif_transcode
    if args.video_transcode is not None:
        overrides["transcode_video"] = args.video_transcode
    if args.image_optimize is not None:
        overrides["optimize_static_images"] = args.image_optimize
    if args.allow_oversize_assets is not None:
        overrides["allow_oversize_assets"] = args.allow_oversize_assets
    if overrides:
        asset_policy = replace(asset_policy, **overrides)
    qa_overrides = {}
    if getattr(args, "qa_slide_hold_sec", None) is not None:
        qa_overrides["slide_hold_sec"] = args.qa_slide_hold_sec
    if getattr(args, "qa_settled_offset_sec", None) is not None:
        qa_overrides["settled_offset_sec"] = args.qa_settled_offset_sec
    if getattr(args, "qa_transition_reference_lead_fraction", None) is not None:
        qa_overrides["transition_reference_lead_fraction"] = args.qa_transition_reference_lead_fraction
    qa_policy = replace(config.qa_policy, **qa_overrides) if qa_overrides else config.qa_policy
    return replace(config, profile=profile, asset_policy=asset_policy, qa_policy=qa_policy)


def _parse_slide_filter(value: str | None) -> set[int] | None:
    if not value:
        return None
    slides: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            slides.update(range(min(start, end), max(start, end) + 1))
        else:
            slides.add(int(part))
    return slides


def _parse_float_list(value: str) -> list[float]:
    raw = value.strip()
    if not raw:
        return []
    if "," not in raw and ":" in raw:
        parts = [float(part.strip()) for part in raw.split(":") if part.strip()]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Range values must use start:end:step.")
        start, end, step = parts
        if step == 0:
            raise argparse.ArgumentTypeError("Range step must not be zero.")
        if (end - start) * step < 0:
            raise argparse.ArgumentTypeError("Range step moves away from end.")
        values: list[float] = []
        current = start
        epsilon = abs(step) / 1000.0
        if step > 0:
            while current <= end + epsilon:
                values.append(round(current, 6))
                current += step
        else:
            while current >= end - epsilon:
                values.append(round(current, 6))
                current += step
        return values
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_track_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def _parse_deck_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}
