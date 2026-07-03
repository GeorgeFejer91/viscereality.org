from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from typing import Any

from .config import AssetPolicy
from .models import AssetRef, PptxDeck
from .utils import ensure_dir, find_binary, format_mb, write_json


def prepare_assets(
    deck: PptxDeck,
    out_dir: Path,
    policy: AssetPolicy,
    *,
    ffmpeg_bin: str | None = None,
) -> dict[str, Any]:
    assets_dir = ensure_dir(out_dir / "assets")
    source_dir = ensure_dir(assets_dir / "source")
    optimized_dir = ensure_dir(assets_dir / "optimized")
    ffmpeg = find_binary("ffmpeg.exe", ffmpeg_bin) or find_binary("ffmpeg", ffmpeg_bin)
    by_hash: dict[str, dict[str, str]] = {}
    report_assets: list[dict[str, Any]] = []
    github_safe = True

    with zipfile.ZipFile(deck.source_path) as zf:
        for asset in sorted(deck.assets.values(), key=lambda item: item.source_path):
            ext = "." + asset.extension.lower().lstrip(".")
            if policy.mode == "manifest-only":
                asset.output_file = None
                asset.source_file = asset.source_path
                asset.warnings.append("manifest-only-asset-not-copied")
                if format_mb(asset.size_bytes) > policy.soft_max_mb:
                    asset.warnings.append("github-soft-limit-warning")
                if format_mb(asset.size_bytes) > policy.hard_max_mb:
                    asset.warnings.append("github-hard-limit-blocker")
                    github_safe = False
                report_assets.append(
                    {
                        "id": asset.id,
                        "sourcePath": asset.source_path,
                        "sourceFile": asset.source_file,
                        "outputFile": asset.output_file,
                        "kind": asset.kind,
                        "extension": asset.extension,
                        "sourceSizeMb": format_mb(asset.size_bytes),
                        "outputSizeMb": None,
                        "sha256": asset.sha256,
                        "width": asset.width,
                        "height": asset.height,
                        "durationSec": asset.duration_sec,
                        "animated": asset.animated,
                        "alpha": asset.alpha,
                        "warnings": asset.warnings,
                    }
                )
                continue
            raw = zf.read(asset.source_path)
            source_name = f"{asset.sha256[:16]}{ext}"
            source_path = source_dir / source_name
            if asset.sha256 not in by_hash:
                source_path.write_bytes(raw)
                by_hash[asset.sha256] = {"source": f"assets/source/{source_name}"}
            asset.source_file = by_hash[asset.sha256]["source"]
            _probe_image_metadata(asset, source_path)

            output_rel = by_hash[asset.sha256].get("output")
            if output_rel is None:
                output_path = source_path
                if _should_transcode_gif(asset, policy):
                    converted = _try_convert_gif_for_publish(
                        source_path,
                        optimized_dir,
                        asset,
                        policy,
                        ffmpeg=ffmpeg,
                    )
                    if converted is not None:
                        output_path = converted
                        if converted.suffix.lower() in {".mp4", ".webm"}:
                            asset.kind = "video"
                            asset.extension = converted.suffix.lower().lstrip(".")
                        else:
                            asset.kind = "image"
                            asset.extension = "webp"
                    else:
                        asset.warnings.append("gif-transcode-unavailable")
                elif _should_transcode_video(asset, policy):
                    converted = _try_optimize_video_with_ffmpeg(
                        source_path,
                        optimized_dir,
                        asset,
                        policy,
                        ffmpeg=ffmpeg,
                    )
                    if converted is not None:
                        output_path = converted
                        asset.kind = "video"
                        asset.extension = "mp4"
                output_rel = output_path.relative_to(out_dir).as_posix()
                by_hash[asset.sha256]["output"] = output_rel
            asset.output_file = output_rel

            output_abs = out_dir / output_rel
            output_size = output_abs.stat().st_size if output_abs.exists() else asset.size_bytes
            if format_mb(output_size) > policy.soft_max_mb:
                asset.warnings.append("github-soft-limit-warning")
            if format_mb(output_size) > policy.hard_max_mb:
                asset.warnings.append("github-hard-limit-blocker")
                github_safe = False
            report_assets.append(
                {
                    "id": asset.id,
                    "sourcePath": asset.source_path,
                    "sourceFile": asset.source_file,
                    "outputFile": asset.output_file,
                    "kind": asset.kind,
                    "extension": asset.extension,
                    "sourceSizeMb": format_mb(asset.size_bytes),
                    "outputSizeMb": format_mb(output_size),
                    "sha256": asset.sha256,
                    "width": asset.width,
                    "height": asset.height,
                    "durationSec": asset.duration_sec,
                    "animated": asset.animated,
                    "alpha": asset.alpha,
                    "warnings": asset.warnings,
                }
            )

    pruned_source_assets = {"count": 0, "bytes": 0}
    pruned_optimized_assets = {"count": 0, "bytes": 0}
    if policy.prune_unreferenced_source_assets and policy.mode != "manifest-only":
        referenced_output_files = {
            str(row.get("outputFile"))
            for row in report_assets
            if row.get("outputFile")
        }
        pruned_source_assets = _prune_unreferenced_source_assets(
            source_dir,
            out_dir,
            referenced_output_files,
        )
        pruned_optimized_assets = _prune_unreferenced_asset_files(
            optimized_dir,
            out_dir,
            referenced_output_files,
        )

    report = {
        "githubPagesSafe": github_safe,
        "assetPolicy": {
            "mode": policy.mode,
            "softMaxMb": policy.soft_max_mb,
            "hardMaxMb": policy.hard_max_mb,
            "transcodeGif": policy.transcode_gif,
            "transcodeVideo": policy.transcode_video,
            "videoCrf": policy.video_crf,
            "allowOversizeAssets": policy.allow_oversize_assets,
            "pruneUnreferencedSourceAssets": policy.prune_unreferenced_source_assets,
        },
        "prunedSourceAssets": {
            "count": pruned_source_assets["count"],
            "bytes": pruned_source_assets["bytes"],
            "mb": format_mb(pruned_source_assets["bytes"]),
        },
        "prunedOptimizedAssets": {
            "count": pruned_optimized_assets["count"],
            "bytes": pruned_optimized_assets["bytes"],
            "mb": format_mb(pruned_optimized_assets["bytes"]),
        },
        "assets": report_assets,
    }
    write_json(out_dir / "asset-report.json", report)
    return report


def _prune_unreferenced_source_assets(
    source_dir: Path,
    out_dir: Path,
    referenced_output_files: set[str],
) -> dict[str, int]:
    return _prune_unreferenced_asset_files(source_dir, out_dir, referenced_output_files)


def _prune_unreferenced_asset_files(
    asset_dir: Path,
    out_dir: Path,
    referenced_output_files: set[str],
) -> dict[str, int]:
    if not asset_dir.exists():
        return {"count": 0, "bytes": 0}
    count = 0
    size_bytes = 0
    skipped = 0
    for asset_file in asset_dir.rglob("*"):
        if not asset_file.is_file():
            continue
        rel = asset_file.relative_to(out_dir).as_posix()
        if rel in referenced_output_files:
            continue
        file_size = asset_file.stat().st_size
        try:
            asset_file.unlink()
        except OSError:
            skipped += 1
            continue
        size_bytes += file_size
        count += 1
    return {"count": count, "bytes": size_bytes, "skipped": skipped}


def _should_transcode_gif(asset: AssetRef, policy: AssetPolicy) -> bool:
    return (
        policy.mode != "source-only"
        and policy.transcode_gif
        and asset.extension.lower() == "gif"
        and (asset.animated or format_mb(asset.size_bytes) > policy.soft_max_mb)
    )


def _try_convert_gif_for_publish(
    source: Path,
    output_dir: Path,
    asset: AssetRef,
    policy: AssetPolicy,
    *,
    ffmpeg: Path | None,
) -> Path | None:
    if asset.alpha:
        converted = _try_convert_gif_to_webm_with_ffmpeg(
            source,
            output_dir,
            asset,
            ffmpeg=ffmpeg,
        )
        if converted is None:
            converted = _try_convert_gif_to_webp_with_ffmpeg(
                source,
                output_dir,
                asset,
                ffmpeg=ffmpeg,
                quality=policy.webp_quality,
            )
        if converted is None and format_mb(asset.size_bytes) <= 30:
            converted = _try_convert_gif_to_webp(
                source,
                output_dir / f"{asset.sha256[:16]}.webp",
                quality=policy.webp_quality,
            )
        if converted is None:
            asset.warnings.append("gif-alpha-preserved-original")
        elif converted.suffix.lower() == ".webm":
            asset.warnings.append("gif-alpha-preserved-webm")
        elif converted.suffix.lower() == ".webp":
            asset.warnings.append("gif-alpha-preserved-webp")
        return converted

    converted = _try_convert_gif_with_ffmpeg(
        source,
        output_dir,
        asset,
        ffmpeg=ffmpeg,
        quality=policy.webp_quality,
    )
    if converted is not None and format_mb(converted.stat().st_size) > policy.hard_max_mb:
        asset.warnings.append("gif-mp4-over-hard-limit-trying-smaller-mp4")
        converted.unlink(missing_ok=True)
        converted = _try_convert_opaque_gif_to_smaller_mp4(
            source,
            output_dir,
            asset,
            policy,
            ffmpeg=ffmpeg,
        )
    if converted is not None and format_mb(converted.stat().st_size) > policy.hard_max_mb:
        asset.warnings.append("gif-small-mp4-over-hard-limit-trying-webm")
        converted.unlink(missing_ok=True)
        converted = None
    if converted is None:
        converted = _try_convert_gif_to_webm_with_ffmpeg(
            source,
            output_dir,
            asset,
            ffmpeg=ffmpeg,
        )
    if converted is not None and format_mb(converted.stat().st_size) > policy.hard_max_mb:
        asset.warnings.append("gif-webm-over-hard-limit-trying-webp")
        converted.unlink(missing_ok=True)
        converted = _try_convert_gif_to_webp_with_ffmpeg(
            source,
            output_dir,
            asset,
            ffmpeg=ffmpeg,
            quality=policy.webp_quality,
        )
    if converted is not None and format_mb(converted.stat().st_size) > policy.hard_max_mb:
        asset.warnings.append("gif-transcode-over-hard-limit")
        converted.unlink(missing_ok=True)
        converted = None
    if converted is None and format_mb(asset.size_bytes) <= 30:
        converted = _try_convert_gif_to_webp(
            source,
            output_dir / f"{asset.sha256[:16]}.webp",
            quality=policy.webp_quality,
        )
    if converted is not None and converted.suffix.lower() == ".mp4":
        asset.warnings.append("gif-opaque-converted-to-mp4")
    elif converted is not None and converted.suffix.lower() == ".webp":
        asset.warnings.append("gif-opaque-converted-to-webp")
    return converted


def _try_convert_opaque_gif_to_smaller_mp4(
    source: Path,
    output_dir: Path,
    asset: AssetRef,
    policy: AssetPolicy,
    *,
    ffmpeg: Path | None,
) -> Path | None:
    if asset.alpha or ffmpeg is None:
        return None
    for crf, max_width in ((28, 1280), (32, 1280), (34, 960)):
        converted = _try_convert_gif_with_ffmpeg(
            source,
            output_dir,
            asset,
            ffmpeg=ffmpeg,
            quality=policy.webp_quality,
            crf=crf,
            max_width=max_width,
            suffix=f"-crf{crf}-w{max_width}",
        )
        if converted is None:
            continue
        if format_mb(converted.stat().st_size) <= policy.hard_max_mb:
            asset.warnings.append(f"gif-opaque-converted-to-small-mp4-crf{crf}")
            return converted
        asset.warnings.append(f"gif-small-mp4-crf{crf}-over-hard-limit")
        converted.unlink(missing_ok=True)
    return None


def _should_transcode_video(asset: AssetRef, policy: AssetPolicy) -> bool:
    return (
        policy.mode != "source-only"
        and policy.transcode_video
        and asset.kind == "video"
        and format_mb(asset.size_bytes) > policy.soft_max_mb
    )


def _probe_image_metadata(asset: AssetRef, path: Path) -> None:
    if asset.kind not in {"image", "svg"}:
        return
    if asset.extension.lower() == "svg":
        _probe_svg_metadata(asset, path)
        return
    try:
        from PIL import Image
    except Exception:
        asset.warnings.append("pillow-unavailable")
        return
    try:
        with Image.open(path) as image:
            asset.width, asset.height = image.size
            frame_count = int(getattr(image, "n_frames", 1) or 1)
            asset.animated = frame_count > 1 or asset.extension.lower() == "gif"
            asset.alpha = image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            )
            if asset.animated:
                total_ms = 0
                for frame_index in range(frame_count):
                    image.seek(frame_index)
                    total_ms += int(image.info.get("duration", 0) or 0)
                if total_ms:
                    asset.duration_sec = round(total_ms / 1000.0, 3)
    except Exception as exc:
        asset.warnings.append(f"image-probe-failed:{exc}")


def _probe_svg_metadata(asset: AssetRef, path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    import re

    width = re.search(r'\bwidth="([0-9.]+)', text)
    height = re.search(r'\bheight="([0-9.]+)', text)
    view_box = re.search(r'\bviewBox="[^"]*?([0-9.]+)\s+([0-9.]+)"', text)
    if width and height:
        asset.width = int(float(width.group(1)))
        asset.height = int(float(height.group(1)))
    elif view_box:
        asset.width = int(float(view_box.group(1)))
        asset.height = int(float(view_box.group(2)))


def _try_convert_gif_to_webp(source: Path, output: Path, quality: int) -> Path | None:
    if output.exists() and output.stat().st_size > 0:
        return output
    try:
        from PIL import Image, ImageSequence
    except Exception:
        return None
    try:
        with Image.open(source) as image:
            frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(image)]
            durations = [
                int(frame.info.get("duration", image.info.get("duration", 100)) or 100)
                for frame in ImageSequence.Iterator(image)
            ]
            if not frames:
                return None
            output.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(
                output,
                format="WEBP",
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=int(image.info.get("loop", 0) or 0),
                quality=max(1, min(100, int(quality))),
                method=6,
            )
        return output
    except Exception:
        return None


def _try_convert_gif_to_webp_with_ffmpeg(
    source: Path,
    output_dir: Path,
    asset: AssetRef,
    *,
    ffmpeg: Path | None,
    quality: int,
) -> Path | None:
    if ffmpeg is None:
        asset.warnings.append("ffmpeg-missing")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{asset.sha256[:16]}.webp"
    if output.exists() and output.stat().st_size > 0:
        return output
    cmd = [
        str(ffmpeg),
        "-y",
        "-i",
        str(source),
        "-an",
        "-vf",
        (
            "scale='min(1920,iw)':'min(1080,ih)'"
            ":force_original_aspect_ratio=decrease:force_divisible_by=2,format=yuva420p"
        ),
        "-loop",
        "0",
        "-c:v",
        "libwebp_anim",
        "-q:v",
        str(max(1, min(100, int(quality)))),
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception as exc:
        asset.warnings.append(f"ffmpeg-gif-webp-transcode-failed:{exc}")
        output.unlink(missing_ok=True)
        return None
    if not output.exists() or output.stat().st_size <= 0:
        return None
    return output


def _try_convert_gif_to_webm_with_ffmpeg(
    source: Path,
    output_dir: Path,
    asset: AssetRef,
    *,
    ffmpeg: Path | None,
) -> Path | None:
    if ffmpeg is None:
        asset.warnings.append("ffmpeg-missing")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{asset.sha256[:16]}.webm"
    if output.exists() and output.stat().st_size > 0:
        return output
    cmd = [
        str(ffmpeg),
        "-y",
        "-i",
        str(source),
        "-an",
        "-vf",
        (
            "scale='min(1920,iw)':'min(1080,ih)'"
            ":force_original_aspect_ratio=decrease:force_divisible_by=2,format=yuva420p"
        ),
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuva420p",
        "-auto-alt-ref",
        "0",
        "-row-mt",
        "1",
        "-threads",
        "0",
        "-deadline",
        "good",
        "-cpu-used",
        "4",
        "-b:v",
        "0",
        "-crf",
        "32",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception as exc:
        asset.warnings.append(f"ffmpeg-gif-webm-transcode-failed:{exc}")
        output.unlink(missing_ok=True)
        return None
    if not output.exists() or output.stat().st_size <= 0:
        return None
    return output


def _try_convert_gif_with_ffmpeg(
    source: Path,
    output_dir: Path,
    asset: AssetRef,
    *,
    ffmpeg: Path | None,
    quality: int,
    crf: int = 20,
    max_width: int = 1920,
    suffix: str = "",
) -> Path | None:
    if ffmpeg is None:
        asset.warnings.append("ffmpeg-missing")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{asset.sha256[:16]}{suffix}.mp4"
    if output.exists() and output.stat().st_size > 0:
        return output
    if asset.alpha:
        asset.warnings.append("gif-alpha-not-flattened-to-mp4")
        return None
    cmd = [
        str(ffmpeg),
        "-y",
        "-i",
        str(source),
        "-an",
        "-vf",
        f"scale='min({max_width},iw)':'min(1080,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception as exc:
        asset.warnings.append(f"ffmpeg-gif-transcode-failed:{exc}")
        output.unlink(missing_ok=True)
        return None
    if not output.exists() or output.stat().st_size <= 0:
        return None
    return output


def _try_optimize_video_with_ffmpeg(
    source: Path,
    output_dir: Path,
    asset: AssetRef,
    policy: AssetPolicy,
    *,
    ffmpeg: Path | None,
) -> Path | None:
    if ffmpeg is None:
        asset.warnings.append("ffmpeg-missing")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    crf = max(0, min(51, int(policy.video_crf)))
    output = output_dir / f"{asset.sha256[:16]}-crf{crf}.mp4"
    if output.exists() and output.stat().st_size > 0:
        if output.stat().st_size < source.stat().st_size:
            return output
        output.unlink(missing_ok=True)
    cmd = [
        str(ffmpeg),
        "-y",
        "-i",
        str(source),
        "-an",
        "-vf",
        "scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception as exc:
        asset.warnings.append(f"ffmpeg-video-optimize-failed:{exc}")
        output.unlink(missing_ok=True)
        return None
    if not output.exists() or output.stat().st_size <= 0:
        return None
    if output.stat().st_size >= source.stat().st_size:
        asset.warnings.append("video-optimized-output-not-smaller")
        output.unlink(missing_ok=True)
        return None
    return output
