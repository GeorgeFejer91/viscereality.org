#!/usr/bin/env python3
"""
Hybrid PPT Chunker v2
---------------------
Backwards-compatible CLI with `analyze` and `chunk`, plus end-to-end `run`.
"""

from __future__ import annotations

import argparse
import sys

from ppt_chunker.exceptions import PipelineError
from ppt_chunker.pipeline import analyze_command, chunk_command, run_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid PPT Chunker v2: analyze/chunk/run pipeline for GitHub Pages playback."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Extract and resolve timings from PPTX")
    p_analyze.add_argument("pptx", help="Path to .pptx file")
    p_analyze.add_argument("-o", "--output-dir", default="output", help="Output directory")
    p_analyze.add_argument(
        "--timing-mode",
        choices=["ppt", "uniform", "hybrid"],
        default="ppt",
        help="Timing mode for resolved output config",
    )
    p_analyze.add_argument("--default-slide-sec", type=float, default=5.0)
    p_analyze.add_argument("--default-transition-sec", type=float, default=2.0)
    p_analyze.add_argument("--overrides-file", help="Optional JSON config to merge as overrides")
    p_analyze.add_argument(
        "--com-probe",
        action="store_true",
        help="Use PowerPoint COM probe for higher fidelity timing extraction",
    )

    p_chunk = sub.add_parser("chunk", help="Chunk MP4 based on timing config")
    p_chunk.add_argument("mp4", help="Path to exported .mp4 file")
    p_chunk.add_argument("-c", "--config", default="output/timing_config.json")
    p_chunk.add_argument("-o", "--output-dir", default="output")
    p_chunk.add_argument("--ffmpeg-bin", help="Optional explicit ffmpeg binary path")
    p_chunk.add_argument("--ffprobe-bin", help="Optional explicit ffprobe binary path")
    p_chunk.add_argument("--duration-tolerance", type=float, default=1.0)
    p_chunk.add_argument("--max-chunk-mb", type=float, default=95.0)
    chunk_audio_group = p_chunk.add_mutually_exclusive_group()
    chunk_audio_group.add_argument(
        "--mute-output", action="store_true", default=True, help="Mute chunk audio (default)"
    )
    chunk_audio_group.add_argument("--keep-audio", action="store_false", dest="mute_output")
    p_chunk.add_argument("--generate-player", action="store_true")

    p_run = sub.add_parser("run", help="Run full pipeline: analyze + export + normalize + chunk")
    p_run.add_argument("pptx", help="Path to .pptx file")
    p_run.add_argument("-o", "--output-dir", default="output")
    p_run.add_argument("--presentation-id", help="Override ID used in output names/manifests")
    p_run.add_argument("--title", help="Presentation title for manifest/player")
    p_run.add_argument(
        "--timing-mode",
        choices=["ppt", "uniform", "hybrid"],
        default="ppt",
        help="Timing resolution mode",
    )
    p_run.add_argument("--default-slide-sec", type=float, default=5.0)
    p_run.add_argument("--default-transition-sec", type=float, default=2.0)
    p_run.add_argument("--overrides-file", help="Optional JSON timing override file")
    p_run.add_argument("--rewrite-timings", action="store_true")
    p_run.add_argument("--fps", type=int, default=30)
    p_run.add_argument("--height", type=int, default=1080)
    p_run.add_argument("--quality", type=int, default=85)
    p_run.add_argument("--ffmpeg-bin", help="Optional explicit ffmpeg binary path")
    p_run.add_argument("--ffprobe-bin", help="Optional explicit ffprobe binary path")
    p_run.add_argument("--duration-tolerance", type=float, default=1.0)
    p_run.add_argument(
        "--fit-duration",
        action="store_true",
        help="If post-normalization drift exceeds tolerance, time-warp master video to expected duration.",
    )
    p_run.add_argument(
        "--max-fit-ratio",
        type=float,
        default=1.1,
        help="Maximum allowed duration fit ratio (actual/expected) before failing (default: 1.1).",
    )
    p_run.add_argument("--max-chunk-mb", type=float, default=95.0)
    run_audio_group = p_run.add_mutually_exclusive_group()
    run_audio_group.add_argument(
        "--mute-output", action="store_true", default=True, help="Mute output audio (default)"
    )
    run_audio_group.add_argument("--keep-audio", action="store_false", dest="mute_output")
    p_run.add_argument("--generate-player", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "analyze":
            analyze_command(args)
        elif args.command == "chunk":
            chunk_command(args)
        elif args.command == "run":
            run_command(args)
        else:
            parser.error(f"Unknown command: {args.command}")
        return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
