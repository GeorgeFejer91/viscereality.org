#!/usr/bin/env python3
"""
Generalized PPT Chunker v3
--------------------------
Primary staged workflow:
  inspect -> build -> validate -> publish

Legacy wrappers are kept for compatibility:
  analyze, chunk, run, upload
"""

from __future__ import annotations

import argparse
import sys

from ppt_chunker.exceptions import PipelineError
from ppt_chunker.pipeline import (
    analyze_command,
    build_command,
    chunk_command,
    fast_publish_command,
    inspect_command,
    publish_command,
    run_command,
    validate_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature-driven PPT pipeline for GitHub Pages playback."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # v3 commands
    p_inspect = sub.add_parser("inspect", help="Extract slide/media features from PPTX")
    p_inspect.add_argument("pptx", help="Path to .pptx")
    p_inspect.add_argument("-o", "--output-dir", default="output")
    p_inspect.add_argument("--profile", default="balanced1080")
    p_inspect.add_argument("--ffprobe-bin")

    p_build = sub.add_parser("build", help="Build artifacts from PPTX")
    p_build.add_argument("pptx", help="Path to .pptx")
    p_build.add_argument("-o", "--output-dir", default="output")
    p_build.add_argument("--profile", default="balanced1080")
    p_build.add_argument("--presentation-id")
    p_build.add_argument("--title")
    p_build.add_argument("--overrides-file")
    p_build.add_argument("--ffmpeg-bin")
    p_build.add_argument("--ffprobe-bin")
    p_build.add_argument("--max-chunk-mb", type=float, default=95.0)
    p_build.add_argument(
        "--transition-fit-policy",
        choices=["target-preserve-frames", "measured"],
        default="target-preserve-frames",
        help=(
            "For no-rewrite PowerPoint exports, either preserve measured rendered "
            "transition durations or retime all rendered frames to the requested duration."
        ),
    )
    p_build.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p_build.add_argument("--mute-output", action=argparse.BooleanOptionalAction, default=True)
    p_build.add_argument("--no-rewrite-timings", action="store_true")
    p_build.add_argument("--reuse-master", action="store_true")

    p_validate = sub.add_parser("validate", help="Validate output artifacts")
    p_validate.add_argument("output_dir", help="Build output directory")
    p_validate.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p_validate.add_argument("--ffmpeg-bin")
    p_validate.add_argument("--ffprobe-bin")

    p_publish = sub.add_parser("publish", help="Publish validated artifacts to presentations/<deck>")
    p_publish.add_argument("output_dir", help="Build output directory")
    p_publish.add_argument("--deck", required=True, help="Deck name (e.g., alpCHI)")
    p_publish.add_argument("--git-push", action="store_true")
    p_publish.add_argument("--git-remote", default="origin")
    p_publish.add_argument("--git-branch", default="main")

    p_fast = sub.add_parser(
        "fast-publish",
        help="Prepare timed PPTX, export master MP4, chunk exact slide loops/transitions, validate, publish",
    )
    p_fast.add_argument("pptx", help="Path to .pptx")
    p_fast.add_argument("-o", "--output-dir", default="output_fast_publish")
    p_fast.add_argument("--deck", required=True, help="Deck name under presentations/<deck>")
    p_fast.add_argument("--presentation-id")
    p_fast.add_argument("--title")
    p_fast.add_argument("--profile", default="balanced1080")
    p_fast.add_argument("--transition-sec", type=float, default=2.0)
    p_fast.add_argument("--default-static-sec", type=float, default=1.0)
    p_fast.add_argument(
        "--media-scope",
        choices=["central", "center", "visible"],
        default="central",
        help="Use central/on-stage media for slide loop timing by default; visible keeps legacy behavior.",
    )
    p_fast.add_argument(
        "--transition-renderer",
        choices=["ppt", "synthetic"],
        default="ppt",
        help="Use authored PowerPoint transitions in the export, or synthesize fades after export.",
    )
    p_fast.add_argument(
        "--transition-fit-policy",
        choices=["target-preserve-frames", "measured"],
        default="target-preserve-frames",
        help=(
            "For authored PowerPoint transitions, cut the real rendered frame span and "
            "either retime all frames to --transition-sec or keep measured duration."
        ),
    )
    p_fast.add_argument(
        "--normalize-gif-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Retiming pass for embedded GIFs: preserve pixels, extend frame delays onto the "
            "smallest grid shared by GIF timing and the export fps."
        ),
    )
    p_fast.add_argument(
        "--gif-duration-policy",
        choices=["export-grid", "ceil-second", "nearest-second"],
        default="export-grid",
        help=(
            "How normalized GIF loop totals are snapped: the finest shared export grid, "
            "a whole-second ceiling for maximum robustness, or the nearest whole second."
        ),
    )
    p_fast.add_argument("--ffmpeg-bin")
    p_fast.add_argument("--ffprobe-bin")
    p_fast.add_argument("--max-chunk-mb", type=float, default=95.0)
    p_fast.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p_fast.add_argument("--mute-output", action=argparse.BooleanOptionalAction, default=True)
    p_fast.add_argument("--reuse-master", action="store_true")
    p_fast.add_argument("--git-remote", default="origin")
    p_fast.add_argument("--git-branch", default="main")

    # legacy wrappers
    p_analyze = sub.add_parser("analyze", help="Legacy wrapper -> inspect + timing_config")
    p_analyze.add_argument("pptx", help="Path to .pptx")
    p_analyze.add_argument("-o", "--output-dir", default="output")
    p_analyze.add_argument(
        "--timing-mode",
        choices=["ppt", "uniform", "hybrid"],
        default="ppt",
    )
    p_analyze.add_argument("--default-slide-sec", type=float, default=4.0)
    p_analyze.add_argument("--default-transition-sec", type=float, default=0.5)
    p_analyze.add_argument("--overrides-file")
    p_analyze.add_argument("--com-probe", action="store_true")
    p_analyze.add_argument("--ffprobe-bin")
    p_analyze.add_argument("--profile", default="balanced1080")

    p_chunk = sub.add_parser("chunk", help="Legacy wrapper for chunking an existing MP4")
    p_chunk.add_argument("mp4", help="Path to master MP4")
    p_chunk.add_argument("-c", "--config", default="output/timing_config.json")
    p_chunk.add_argument("-o", "--output-dir", default="output")
    p_chunk.add_argument("--ffmpeg-bin")
    p_chunk.add_argument("--ffprobe-bin")
    p_chunk.add_argument("--max-chunk-mb", type=float, default=95.0)
    p_chunk.add_argument("--mute-output", action=argparse.BooleanOptionalAction, default=True)

    p_run = sub.add_parser("run", help="Legacy wrapper -> build + validate")
    p_run.add_argument("pptx", help="Path to .pptx")
    p_run.add_argument("-o", "--output-dir", default="output")
    p_run.add_argument("--profile", default="balanced1080")
    p_run.add_argument("--presentation-id")
    p_run.add_argument("--title")
    p_run.add_argument("--overrides-file")
    p_run.add_argument("--ffmpeg-bin")
    p_run.add_argument("--ffprobe-bin")
    p_run.add_argument("--max-chunk-mb", type=float, default=95.0)
    p_run.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument("--mute-output", action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument("--reuse-master", action="store_true")

    p_upload = sub.add_parser("upload", help="Legacy wrapper -> publish")
    p_upload.add_argument("output_dir")
    p_upload.add_argument("--deck", required=True)
    p_upload.add_argument("--git-push", action="store_true")
    p_upload.add_argument("--git-remote", default="origin")
    p_upload.add_argument("--git-branch", default="main")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            inspect_command(args)
        elif args.command == "build":
            build_command(args)
        elif args.command == "validate":
            validate_command(args)
        elif args.command == "publish":
            publish_command(args)
        elif args.command == "fast-publish":
            fast_publish_command(args)
        elif args.command == "analyze":
            analyze_command(args)
        elif args.command == "chunk":
            chunk_command(args)
        elif args.command == "run":
            run_command(args)
        elif args.command == "upload":
            publish_command(args)
        else:
            parser.error(f"Unknown command: {args.command}")
        return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
