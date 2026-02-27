from __future__ import annotations

import json
import os
from pathlib import Path

from .exceptions import PipelineError
from .utils import run_subprocess


def ffprobe_duration(ffprobe_bin: Path, media_path: Path) -> float:
    cmd = [
        str(ffprobe_bin),
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(media_path),
    ]
    result = run_subprocess(cmd, f"ffprobe duration check for {media_path}")
    payload = json.loads(result.stdout or "{}")
    duration = payload.get("format", {}).get("duration")
    if duration is None:
        raise PipelineError(f"ffprobe did not return duration for {media_path}")
    return float(duration)


def normalize_video(
    ffmpeg_bin: Path,
    input_mp4: Path,
    output_mp4: Path,
    fps: int,
    height: int,
    mute_output: bool = True,
) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-i",
        str(input_mp4),
        "-vf",
        f"scale=-2:{height}",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if mute_output:
        cmd.extend(["-an"])
    else:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    cmd.append(str(output_mp4))
    run_subprocess(cmd, "normalize master mp4")


def fit_video_duration(
    ffmpeg_bin: Path,
    ffprobe_bin: Path,
    input_mp4: Path,
    output_mp4: Path,
    target_duration_s: float,
    fps: int,
    height: int,
    mute_output: bool = True,
) -> float:
    if target_duration_s <= 0:
        raise PipelineError(f"Target duration must be > 0, got {target_duration_s}")

    # A preliminary ffprobe call is required to compute the timing warp factor.
    actual_duration_s = ffprobe_duration(ffprobe_bin, input_mp4)
    if actual_duration_s <= 0:
        raise PipelineError(f"Invalid input duration for fit step: {actual_duration_s}")

    speed_factor = actual_duration_s / target_duration_s
    if speed_factor <= 0:
        raise PipelineError(f"Computed invalid speed factor: {speed_factor}")

    # setpts uses reciprocal scale relative to desired playback speed.
    video_filter = f"setpts=PTS/{speed_factor:.10f},fps={fps},scale=-2:{height}"
    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-i",
        str(input_mp4),
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if mute_output:
        cmd.extend(["-an"])
    else:
        audio_speed = target_duration_s / actual_duration_s
        cmd.extend(["-filter:a", _atempo_chain(audio_speed), "-c:a", "aac", "-b:a", "128k"])
    cmd.append(str(output_mp4))
    run_subprocess(cmd, "fit master duration")
    return speed_factor


def cut_chunk_precise(
    ffmpeg_bin: Path,
    input_mp4: Path,
    output_mp4: Path,
    start_s: float,
    duration_s: float,
    mute_output: bool = True,
    crf: int = 18,
) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(input_mp4),
        "-t",
        f"{duration_s:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(crf),
        "-g",
        "15",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if mute_output:
        cmd.extend(["-an"])
    else:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    cmd.append(str(output_mp4))
    run_subprocess(cmd, f"cut chunk {output_mp4.name}")


def enforce_chunk_size(
    ffmpeg_bin: Path, chunk_path: Path, max_chunk_mb: float, mute_output: bool = True
) -> None:
    if max_chunk_mb <= 0:
        return
    max_bytes = int(max_chunk_mb * 1024 * 1024)
    if not chunk_path.exists():
        raise PipelineError(f"Chunk missing after encode: {chunk_path}")
    if chunk_path.stat().st_size <= max_bytes:
        return

    temp_out = chunk_path.with_suffix(".reduced.mp4")
    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-i",
        str(chunk_path),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if mute_output:
        cmd.extend(["-an"])
    else:
        cmd.extend(["-c:a", "aac", "-b:a", "96k"])
    cmd.append(str(temp_out))
    run_subprocess(cmd, f"reduce chunk size for {chunk_path.name}")
    temp_out.replace(chunk_path)

    if chunk_path.stat().st_size > max_bytes:
        actual_mb = chunk_path.stat().st_size / (1024 * 1024)
        raise PipelineError(
            f"Chunk {chunk_path.name} is {actual_mb:.2f}MB, above max {max_chunk_mb:.2f}MB"
        )


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def _atempo_chain(speed: float) -> str:
    if speed <= 0:
        raise PipelineError(f"Invalid audio speed factor: {speed}")
    factors = []
    remaining = speed
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)
    return ",".join(f"atempo={min(2.0, max(0.5, f)):.8f}" for f in factors)
