#!/usr/bin/env python3
"""Generate static ASCII frame assets for the Viscereality website.

The output is intentionally plain text plus small JSON manifests so GitHub Pages
can serve compressed, non-video assets while the browser renders the animation
with regular HTML text.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


PALETTE = (
    " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"
)


def parse_fraction(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return 0.0
        return float(numerator) / denominator_value
    return float(value)


def probe_video(path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open media file: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frame_count


def auto_rows(cols: int, width: int, height: int) -> int:
    aspect = width / max(height, 1)
    return max(1, round(cols / aspect / 2))


def frame_to_ascii(frame: np.ndarray, cols: int, rows: int) -> str:
    resized = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    indices = np.floor_divide(gray, max(1, 256 // len(PALETTE)))
    np.clip(indices, 0, len(PALETTE) - 1, out=indices)
    lut = np.array(list(PALETTE), dtype="<U1")
    chars = lut[indices]
    return "\n".join("".join(row) for row in chars)


def convert_media(source: Path, out_dir: Path, cols: int, target_fps: float) -> dict:
    width, height, source_fps, source_frames = probe_video(source)
    rows = auto_rows(cols, width, height)
    step = max(1, round(source_fps / target_fps))
    effective_fps = source_fps / step
    safe_name = source.stem.lower().replace(" ", "-")
    manifest_path = out_dir / f"{safe_name}.json"
    frames_path = out_dir / f"{safe_name}.frames.txt"

    cap = cv2.VideoCapture(str(source))
    frames: list[str] = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % step == 0:
            frames.append(frame_to_ascii(frame, cols, rows))

        frame_index += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames were generated for {source}")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path.write_text("\f".join(frames), encoding="utf-8", newline="\n")

    manifest = {
        "version": 1,
        "source": source.as_posix(),
        "frames": frames_path.name,
        "width": width,
        "height": height,
        "fps": round(effective_fps, 3),
        "sourceFps": round(source_fps, 3),
        "frameCount": len(frames),
        "sourceFrameCount": source_frames,
        "cols": cols,
        "rows": rows,
        "palette": PALETTE,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate static ASCII animation assets.")
    parser.add_argument("--out-dir", type=Path, default=Path("assets/ascii-media"))
    parser.add_argument("--fps", type=parse_fraction, default=12.0)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument(
        "--cols",
        action="append",
        default=[],
        help="Per-source columns as filename=cols, for example viscereality-hero-desktop.mp4=180.",
    )
    args = parser.parse_args()

    cols_by_name = {}
    for item in args.cols:
        name, value = item.split("=", 1)
        cols_by_name[name] = int(value)

    for source in args.sources:
        cols = cols_by_name.get(source.name, 120)
        manifest = convert_media(source, args.out_dir, cols, args.fps)
        print(
            f"{source} -> {manifest['frames']} "
            f"({manifest['cols']}x{manifest['rows']} @ {manifest['fps']} fps, "
            f"{manifest['frameCount']} frames)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
