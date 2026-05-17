from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

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


def ffprobe_video_stream_info(ffprobe_bin: Path, media_path: Path) -> dict[str, float]:
    cmd = [
        str(ffprobe_bin),
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-count_frames",
        "-show_streams",
        str(media_path),
    ]
    result = run_subprocess(cmd, f"ffprobe stream info for {media_path}")
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    for stream in streams:
        if stream.get("codec_type") != "video":
            continue
        avg = stream.get("avg_frame_rate") or "0/1"
        try:
            num, den = avg.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        except Exception:
            fps = 0.0
        duration = float(stream.get("duration") or 0.0)
        if duration <= 0:
            duration = ffprobe_duration(ffprobe_bin, media_path)
        nb_frames_raw = stream.get("nb_frames") or stream.get("nb_read_frames")
        if nb_frames_raw is None:
            nb_frames = int(round(duration * fps)) if fps > 0 else 0
        else:
            nb_frames = int(nb_frames_raw)
        return {"fps": fps, "duration_s": duration, "nb_frames": float(nb_frames)}
    raise PipelineError(f"No video stream found for {media_path}")


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
    # Use accurate seek (after input) so chunk boundaries match timeline precisely.
    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-i",
        str(input_mp4),
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-avoid_negative_ts",
        "make_zero",
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


def cut_chunk_frame_exact(
    ffmpeg_bin: Path,
    input_mp4: Path,
    output_mp4: Path,
    start_frame: int,
    frame_count: int,
    fps: int,
    mute_output: bool = True,
    crf: int = 18,
    output_frame_count: int | None = None,
) -> None:
    if frame_count <= 0:
        raise PipelineError(f"Frame count must be > 0 for {output_mp4.name}")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    end_frame = start_frame + frame_count
    if output_frame_count is not None and output_frame_count <= 0:
        raise PipelineError(f"Output frame count must be > 0 for {output_mp4.name}")
    if output_frame_count is not None and output_frame_count != frame_count:
        setpts = f"{(float(output_frame_count) / float(frame_count)):.10f}*(PTS-STARTPTS)"
    else:
        setpts = "PTS-STARTPTS"
    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-i",
        str(input_mp4),
        "-vf",
        (
            f"trim=start_frame={start_frame}:end_frame={end_frame},"
            f"setpts={setpts}"
        ),
        "-r",
        str(fps),
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
    run_subprocess(cmd, f"cut frame-exact chunk {output_mp4.name}")


def cut_chunk_retimed_preserve_frames(
    ffmpeg_bin: Path,
    input_mp4: Path,
    output_mp4: Path,
    start_frame: int,
    frame_count: int,
    target_duration_s: float,
    source_fps: float | None = None,
    mute_output: bool = True,
    crf: int = 16,
) -> None:
    if frame_count <= 0:
        raise PipelineError(f"Frame count must be > 0 for {output_mp4.name}")
    if target_duration_s <= 0:
        raise PipelineError(f"Target duration must be > 0 for {output_mp4.name}")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    end_frame = start_frame + frame_count
    # Preserve every decoded source frame and rewrite timestamps onto an evenly
    # spaced output timeline. Do not force a CFR output rate here: if PowerPoint
    # rendered 93 frames and the target is 2s, the transition becomes ~46.5 fps
    # rather than dropping 33 frames to fit 30 fps.
    setpts = _frame_preserving_setpts(
        target_duration_s=target_duration_s,
        frame_count=frame_count,
        source_fps=source_fps,
    )
    filter_expr = (
        f"trim=start_frame={start_frame}:end_frame={end_frame},"
        f"setpts={setpts}"
    )
    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-i",
        str(input_mp4),
        "-vf",
        filter_expr,
        "-fps_mode",
        "passthrough",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-g",
        str(max(15, min(frame_count, 60))),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if mute_output:
        cmd.extend(["-an"])
    else:
        cmd.extend(["-an"])
    cmd.append(str(output_mp4))
    run_subprocess(cmd, f"cut retimed frame-preserving chunk {output_mp4.name}")


def reconcile_video_duration(
    ffmpeg_bin: Path,
    ffprobe_bin: Path,
    media_path: Path,
    target_duration_s: float,
    tolerance_s: float,
    mute_output: bool = True,
    crf: int = 16,
) -> dict[str, float | bool | str]:
    """Cross-check a rendered MP4 duration and retime it if needed.

    The correction pass deliberately preserves every decoded video frame.  If a
    chunk has the right visual frames but the container/playback duration is off,
    we rewrite timestamps onto the desired timeline instead of dropping or
    duplicating frames to satisfy a CFR grid.
    """

    if target_duration_s <= 0:
        raise PipelineError(f"Target duration must be > 0 for {media_path.name}")
    if tolerance_s < 0:
        raise PipelineError(f"Duration tolerance must be >= 0, got {tolerance_s}")
    if not media_path.exists() or media_path.stat().st_size == 0:
        raise PipelineError(f"Cannot duration-check missing/empty file: {media_path}")

    before_s = ffprobe_duration(ffprobe_bin, media_path)
    result: dict[str, float | bool | str] = {
        "target_duration_s": float(target_duration_s),
        "actual_duration_before_s": float(before_s),
        "actual_duration_after_s": float(before_s),
        "delta_before_s": float(before_s - target_duration_s),
        "delta_after_s": float(before_s - target_duration_s),
        "duration_corrected": False,
        "correction_method": "none",
    }
    if abs(before_s - target_duration_s) <= tolerance_s:
        return result

    stream_info = ffprobe_video_stream_info(ffprobe_bin, media_path)
    frame_count = int(round(float(stream_info.get("nb_frames", 0.0) or 0.0)))
    if frame_count <= 0:
        raise PipelineError(f"Cannot duration-correct {media_path.name}: no video frames found")

    temp_out = media_path.with_name(f"{media_path.stem}.durationfix{media_path.suffix}")
    if temp_out.exists():
        temp_out.unlink()
    setpts = _frame_preserving_setpts(
        target_duration_s=target_duration_s,
        frame_count=frame_count,
        source_fps=float(stream_info.get("fps", 0.0) or 0.0),
    )
    filter_expr = f"setpts={setpts}"
    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-i",
        str(media_path),
        "-vf",
        filter_expr,
        "-fps_mode",
        "passthrough",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-g",
        str(max(15, min(frame_count, 60))),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if mute_output:
        cmd.extend(["-an"])
    else:
        audio_speed = target_duration_s / before_s if before_s > 0 else 1.0
        cmd.extend(["-filter:a", _atempo_chain(audio_speed), "-c:a", "aac", "-b:a", "128k"])
    cmd.append(str(temp_out))
    run_subprocess(cmd, f"duration-correct chunk {media_path.name}")
    temp_out.replace(media_path)

    after_s = ffprobe_duration(ffprobe_bin, media_path)
    result.update(
        {
            "actual_duration_after_s": float(after_s),
            "delta_after_s": float(after_s - target_duration_s),
            "duration_corrected": True,
            "correction_method": "ffmpeg_setpts_preserve_frames",
        }
    )
    if abs(after_s - target_duration_s) > tolerance_s:
        raise PipelineError(
            f"Duration correction failed for {media_path.name}: "
            f"target={target_duration_s:.6f}s actual={after_s:.6f}s "
            f"tolerance={tolerance_s:.6f}s"
        )
    return result


def synthesize_frame_fade_transition(
    ffmpeg_bin: Path,
    input_mp4: Path,
    output_mp4: Path,
    previous_frame: int,
    next_frame: int,
    frame_count: int,
    fps: int,
    mute_output: bool = True,
    crf: int = 18,
) -> None:
    if frame_count <= 0:
        raise PipelineError(f"Frame count must be > 0 for {output_mp4.name}")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    duration_s = frame_count / float(fps)
    with tempfile.TemporaryDirectory(prefix="ppt_chunker_fade_") as td:
        prev_png = Path(td) / "previous.png"
        next_png = Path(td) / "next.png"
        extract_frame_png(ffmpeg_bin, input_mp4, prev_png, max(0.0, previous_frame / float(fps)))
        extract_frame_png(ffmpeg_bin, input_mp4, next_png, max(0.0, next_frame / float(fps)))
        cmd = [
            str(ffmpeg_bin),
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-t",
            f"{duration_s:.6f}",
            "-i",
            str(prev_png),
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-t",
            f"{duration_s:.6f}",
            "-i",
            str(next_png),
            "-filter_complex",
            (
                f"[0:v][1:v]xfade=transition=fade:duration={duration_s:.6f}:offset=0,"
                f"format=yuv420p,fps={fps}"
            ),
            "-frames:v",
            str(frame_count),
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
            cmd.extend(["-an"])
        cmd.append(str(output_mp4))
        run_subprocess(cmd, f"synthesize fade transition {output_mp4.name}")


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


def content_hash(path: Path, digest_size: int = 8) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:digest_size]


def rename_with_hash(path: Path) -> Path:
    digest = content_hash(path)
    stem = path.stem
    suffix = path.suffix
    hashed = path.with_name(f"{stem}.{digest}{suffix}")
    path.replace(hashed)
    return hashed


def decode_check(ffprobe_bin: Path, media_path: Path) -> None:
    cmd = [
        str(ffprobe_bin),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(media_path),
    ]
    run_subprocess(cmd, f"decode check for {media_path.name}")


def extract_frame_png(
    ffmpeg_bin: Path, input_mp4: Path, output_png: Path, at_sec: float
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-ss",
        f"{max(0.0, at_sec):.6f}",
        "-i",
        str(input_mp4),
        "-frames:v",
        "1",
        str(output_png),
    ]
    run_subprocess(cmd, f"extract frame for {input_mp4.name}")


def image_mismatch_score(image_a: Path, image_b: Path) -> float:
    with Image.open(image_a) as a, Image.open(image_b) as b:
        if a.size != b.size:
            b = b.resize(a.size)
        arr_a = np.asarray(a.convert("RGB"), dtype=np.float32)
        arr_b = np.asarray(b.convert("RGB"), dtype=np.float32)
    mse = float(np.mean((arr_a - arr_b) ** 2))
    # Normalize to [0, 1] approximately.
    norm = mse / (255.0**2)
    return round(norm, 6)


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


def _frame_preserving_setpts(
    *, target_duration_s: float, frame_count: int, source_fps: float | None
) -> str:
    if frame_count <= 0:
        raise PipelineError(f"Frame count must be > 0, got {frame_count}")
    if target_duration_s <= 0:
        raise PipelineError(f"Target duration must be > 0, got {target_duration_s}")
    if frame_count == 1:
        return "0"
    source_frame_duration_s = (
        1.0 / float(source_fps) if source_fps is not None and float(source_fps) > 0 else 0.0
    )
    # MP4 duration is effectively the final frame PTS plus its sample duration.
    # Preserve all frames, but place the last frame so that this final sample
    # closes exactly on the target duration.
    if source_frame_duration_s <= 0 or source_frame_duration_s >= target_duration_s:
        frame_interval_s = target_duration_s / float(frame_count)
    else:
        frame_interval_s = (target_duration_s - source_frame_duration_s) / float(
            frame_count - 1
        )
    return f"N*{frame_interval_s:.10f}/TB"
