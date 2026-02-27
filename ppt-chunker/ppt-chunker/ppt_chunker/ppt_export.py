from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .exceptions import PipelineError
from .models import ResolvedSegment


def export_ppt_to_video(
    pptx_path: Path,
    output_mp4: Path,
    segments: list[ResolvedSegment] | None = None,
    rewrite_timings: bool = False,
    fps: int = 30,
    vert_resolution: int = 1080,
    quality: int = 85,
    default_slide_duration_s: float = 5.0,
    timeout_sec: int = 7200,
    retries: int = 10,
) -> Path:
    try:
        import pywintypes  # type: ignore
        import win32com.client  # type: ignore
    except Exception as exc:
        raise PipelineError(
            "pywin32 is required for PowerPoint COM export. Install with: py -3 -m pip install pywin32"
        ) from exc

    if not pptx_path.exists():
        raise PipelineError(f"PPTX file not found: {pptx_path}")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ppt_chunker_") as td:
        temp_ppt = Path(td) / pptx_path.name
        shutil.copy2(pptx_path, temp_ppt)

        app = None
        pres = None
        try:
            app = _com_call(lambda: win32com.client.DispatchEx("PowerPoint.Application"), pywintypes, retries)
            pres = _com_call(
                lambda: app.Presentations.Open(str(temp_ppt), False, False, False), pywintypes, retries
            )

            if rewrite_timings and segments:
                _apply_timings_to_presentation(pres, segments, pywintypes, retries)
                _com_call(lambda: pres.Save(), pywintypes, retries)

            _com_call(
                lambda: pres.CreateVideo(
                    str(output_mp4),
                    True,
                    float(default_slide_duration_s),
                    int(vert_resolution),
                    int(fps),
                    int(quality),
                ),
                pywintypes,
                retries,
            )
            _wait_for_video_export(pres, output_mp4, pywintypes, retries, timeout_sec=timeout_sec)
            return output_mp4
        except Exception as exc:
            raise PipelineError(f"PowerPoint export failed: {exc}") from exc
        finally:
            try:
                if pres is not None:
                    pres.Close()
            except Exception:
                pass
            try:
                if app is not None:
                    app.Quit()
            except Exception:
                pass


def _apply_timings_to_presentation(
    pres: Any, segments: list[ResolvedSegment], pywintypes: Any, retries: int
) -> None:
    slide_count = int(_com_call(lambda: pres.Slides.Count, pywintypes, retries))
    mapping = {seg.slide_number: seg for seg in segments}
    for idx in range(1, slide_count + 1):
        if idx not in mapping:
            continue
        seg = mapping[idx]
        slide = _com_call(lambda i=idx: pres.Slides(i), pywintypes, retries)
        transition = _com_call(lambda s=slide: s.SlideShowTransition, pywintypes, retries)
        _com_call(lambda t=transition: setattr(t, "AdvanceOnClick", False), pywintypes, retries)
        _com_call(lambda t=transition: setattr(t, "AdvanceOnTime", True), pywintypes, retries)
        _com_call(lambda t=transition: setattr(t, "AdvanceTime", float(seg.slide_duration_s)), pywintypes, retries)
        _com_call(
            lambda t=transition: setattr(t, "Duration", float(max(0.0, seg.transition_duration_s))),
            pywintypes,
            retries,
        )


def _wait_for_video_export(
    pres: Any, output_mp4: Path, pywintypes: Any, retries: int, timeout_sec: int
) -> None:
    # PpMediaTaskStatus constants:
    # 0 none, 1 in progress, 2 queued, 3 done, 4 failed
    done_status = 3
    failed_status = 4
    started = time.time()
    while time.time() - started < timeout_sec:
        status = int(_com_call(lambda: pres.CreateVideoStatus, pywintypes, retries))
        if status == done_status:
            if not output_mp4.exists() or output_mp4.stat().st_size == 0:
                raise PipelineError(f"Export reported done but output is missing/empty: {output_mp4}")
            return
        if status == failed_status:
            raise PipelineError("PowerPoint export failed (CreateVideoStatus=failed).")
        time.sleep(1.0)
    raise PipelineError(f"Timed out waiting for PowerPoint export after {timeout_sec}s.")


def _com_call(fn, pywintypes: Any, retries: int):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except pywintypes.com_error as exc:
            text = str(exc).upper()
            if "RPC_E_CALL_REJECTED" in text and attempt < retries:
                time.sleep(0.35 * attempt)
                continue
            raise
    raise PipelineError("COM retry budget exhausted.")

