from __future__ import annotations

import os
import shutil
from pathlib import Path

from .exceptions import PipelineError


def _find_on_path(name: str) -> Path | None:
    found = shutil.which(name)
    if not found:
        return None
    return Path(found).resolve()


def _winget_candidates(binary_name: str) -> list[Path]:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return []

    packages_dir = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    if not packages_dir.exists():
        return []

    hits = []
    for path in packages_dir.rglob(binary_name):
        if path.is_file():
            hits.append(path.resolve())
    return hits


def resolve_binary(binary_name: str, explicit_path: str | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists():
            return path.resolve()
        raise PipelineError(f"{binary_name} not found at explicit path: {explicit_path}")

    path_hit = _find_on_path(binary_name)
    if path_hit:
        return path_hit

    winget_hits = _winget_candidates(binary_name)
    if winget_hits:
        return sorted(winget_hits)[0]

    raise PipelineError(
        f"Could not locate {binary_name}. Install it or pass --{binary_name.replace('.exe', '')}-bin."
    )


def discover_ffmpeg_tools(
    ffmpeg_bin: str | None = None, ffprobe_bin: str | None = None
) -> tuple[Path, Path]:
    ffmpeg = resolve_binary("ffmpeg.exe", ffmpeg_bin)
    ffprobe = resolve_binary("ffprobe.exe", ffprobe_bin)
    return ffmpeg, ffprobe

