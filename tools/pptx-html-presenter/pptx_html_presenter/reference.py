from __future__ import annotations

import subprocess
from io import BytesIO
import json
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from .errors import PresenterError
from .utils import find_binary


def export_reference_mp4(
    pptx: Path,
    output_mp4: Path,
    *,
    scene_path: Path | None = None,
    use_timings: bool | None = None,
    clamp_media: bool | None = None,
    ffmpeg_bin: str | None = None,
    fps: int = 30,
    height: int = 1080,
    quality: int = 100,
    default_slide_sec: float | None = None,
    timeout_sec: int = 7200,
) -> Path:
    pptx = pptx.expanduser().resolve()
    output_mp4 = output_mp4.expanduser().resolve()
    if not pptx.exists():
        raise PresenterError(f"PPTX not found: {pptx}")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    transition_durations = _transition_durations_from_scene(scene_path) if scene_path else []
    effective_slide_sec = _effective_slide_hold(default_slide_sec, scene_path)
    effective_use_timings = _effective_use_timings(use_timings, scene_path)
    effective_clamp_media = clamp_media if clamp_media is not None else scene_path is not None
    timing_json = json.dumps(transition_durations)
    with tempfile.TemporaryDirectory(prefix="pptx_html_presenter_ref_") as temp_dir:
        temp_pptx = Path(temp_dir) / pptx.name
        shutil.copy2(pptx, temp_pptx)
        try:
            temp_pptx.chmod(temp_pptx.stat().st_mode | stat.S_IWRITE)
        except Exception:
            pass
        if transition_durations:
            _patch_pptx_timing_xml(
                temp_pptx,
                transition_durations,
                effective_slide_sec,
                clamp_media=effective_clamp_media,
                ffmpeg_bin=ffmpeg_bin,
            )
        script = f"""
$ErrorActionPreference = 'Stop'
$pptx = '{_ps_quote(str(temp_pptx))}'
$out = '{_ps_quote(str(output_mp4))}'
$useTimings = ${str(effective_use_timings).lower()}
$app = $null
$pres = $null
try {{
  $app = New-Object -ComObject PowerPoint.Application
  $pres = $app.Presentations.Open($pptx, $false, $false, $false)
  $pres.CreateVideo($out, $useTimings, [double]{effective_slide_sec}, [int]{height}, [int]{fps}, [int]{quality})
  $started = Get-Date
  while ($true) {{
    $status = [int]$pres.CreateVideoStatus
    if ($status -eq 3) {{
      if (-not (Test-Path -LiteralPath $out)) {{ throw 'PowerPoint reported done but MP4 is missing.' }}
      if ((Get-Item -LiteralPath $out).Length -le 0) {{ throw 'PowerPoint reported done but MP4 is empty.' }}
      break
    }}
    if ($status -eq 4) {{ throw 'PowerPoint CreateVideo failed.' }}
    if (((Get-Date) - $started).TotalSeconds -gt {timeout_sec}) {{ throw 'Timed out waiting for PowerPoint CreateVideo.' }}
    Start-Sleep -Seconds 1
  }}
}} finally {{
  if ($pres -ne $null) {{ try {{ $pres.Close() }} catch {{ }} }}
  if ($app -ne $null) {{ try {{ $app.Quit() }} catch {{ }} }}
}}
"""
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            text=True,
            capture_output=True,
            timeout=timeout_sec + 60,
        )
    if result.returncode != 0:
        raise PresenterError(
            "PowerPoint reference export failed: "
            + (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")
        )
    return output_mp4


def _transition_durations_from_scene(scene_path: Path | None) -> list[float]:
    if scene_path is None:
        return []
    path = scene_path.expanduser().resolve()
    if path.is_dir():
        path = path / "deck.scene.json"
    if not path.exists():
        raise PresenterError(f"Scene file not found: {path}")
    scene = json.loads(path.read_text(encoding="utf-8"))
    slide_count = int(scene.get("deck", {}).get("slideCount", 0) or len(scene.get("slides", [])))
    durations = [0.0 for _ in range(slide_count)]
    for transition in scene.get("transitions", []):
        to_index = int(transition.get("to", 0) or 0)
        if 1 <= to_index <= slide_count:
            durations[to_index - 1] = float(transition.get("durationSec", 0.0) or 0.0)
    return durations


def _effective_use_timings(use_timings: bool | None, scene_path: Path | None) -> bool:
    if use_timings is not None:
        return use_timings
    return scene_path is not None


def _effective_slide_hold(default_slide_sec: float | None, scene_path: Path | None) -> float:
    if default_slide_sec is not None:
        return float(default_slide_sec)
    if scene_path is not None:
        path = scene_path.expanduser().resolve()
        if path.is_dir():
            path = path / "deck.scene.json"
        if path.exists():
            scene = json.loads(path.read_text(encoding="utf-8"))
            qa = scene.get("qa", {}) or {}
            if qa.get("slideHoldSec") is not None:
                return float(qa["slideHoldSec"])
    return 1.0


def _patch_pptx_timing_xml(
    pptx_path: Path,
    transition_durations: list[float],
    slide_hold_sec: float,
    *,
    clamp_media: bool,
    ffmpeg_bin: str | None,
) -> None:
    slide_hold_ms = max(100, int(round(slide_hold_sec * 1000.0)))
    source = pptx_path
    patched = pptx_path.with_suffix(".timed.pptx")
    ffmpeg = find_binary("ffmpeg.exe", ffmpeg_bin) or find_binary("ffmpeg", ffmpeg_bin)
    with zipfile.ZipFile(source, "r") as zin, zipfile.ZipFile(
        patched, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            slide_number = _slide_number_from_path(item.filename)
            if slide_number is not None and 1 <= slide_number <= len(transition_durations):
                duration_ms = max(0, int(round(float(transition_durations[slide_number - 1]) * 1000.0)))
                data = _patch_slide_transition_timing(data, slide_hold_ms, duration_ms)
            elif clamp_media and item.filename.startswith("ppt/media/"):
                data = _clamp_media_asset(item.filename, data, pptx_path.parent, ffmpeg, slide_hold_sec)
            zout.writestr(item, data)
    patched.replace(source)


def _slide_number_from_path(path: str) -> int | None:
    match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", path)
    return int(match.group(1)) if match else None


def _patch_slide_transition_timing(data: bytes, slide_hold_ms: int, duration_ms: int) -> bytes:
    text = data.decode("utf-8")
    if "<p:transition" not in text:
        replacement = _transition_tag(slide_hold_ms, duration_ms, self_closing=True)
        insert_at = text.find("<p:timing")
        if insert_at == -1:
            insert_at = text.find("</p:sld>")
        if insert_at == -1:
            return data
        text = text[:insert_at] + replacement + text[insert_at:]
    else:
        def repl(match: re.Match[str]) -> str:
            tag = match.group(0)
            self_closing = tag.endswith("/>")
            body = tag[: -2 if self_closing else -1]
            body = re.sub(r'\s(?:advClick|advTm|dur|spd)="[^"]*"', "", body)
            body = re.sub(r'\s\w+:dur="[^"]*"', "", body)
            if "xmlns:p14=" not in body:
                body += ' xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main"'
            body += f' advClick="0" advTm="{slide_hold_ms}" dur="{duration_ms}" p14:dur="{duration_ms}"'
            return body + ("/>" if self_closing else ">")

        text = re.sub(r"<p:transition\b[^>]*(?:/>|>)", repl, text)
    text = _clamp_media_timing(text, slide_hold_ms)
    return text.encode("utf-8")


def _clamp_media_timing(text: str, slide_hold_ms: int) -> str:
    media_hold_ms = max(1, min(slide_hold_ms, 1000))
    return re.sub(r'\bdur="30000"', f'dur="{media_hold_ms}"', text)


def _clamp_media_asset(
    name: str,
    data: bytes,
    work_dir: Path,
    ffmpeg: Path | None,
    slide_hold_sec: float,
) -> bytes:
    ext = Path(name).suffix.lower()
    if ext == ".gif":
        return _first_frame_gif(data, slide_hold_sec) or data
    if ext == ".mp4":
        return _short_mp4(data, work_dir, ffmpeg, slide_hold_sec) or data
    return data


def _first_frame_gif(data: bytes, slide_hold_sec: float) -> bytes | None:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(BytesIO(data)) as image:
            image.seek(0)
            frame = image.convert("P", palette=getattr(Image, "ADAPTIVE", 1))
            out = BytesIO()
            frame.save(
                out,
                format="GIF",
                duration=max(100, int(round(slide_hold_sec * 1000.0))),
                loop=0,
            )
            return out.getvalue()
    except Exception:
        return None


def _short_mp4(data: bytes, work_dir: Path, ffmpeg: Path | None, slide_hold_sec: float) -> bytes | None:
    if ffmpeg is None:
        return None
    digest = str(abs(hash(data)))
    source = work_dir / f"reference-media-{digest}.mp4"
    output = work_dir / f"reference-media-{digest}.short.mp4"
    try:
        source.write_bytes(data)
        subprocess.run(
            [
                str(ffmpeg),
                "-y",
                "-i",
                str(source),
                "-t",
                f"{max(0.1, slide_hold_sec):.3f}",
                "-an",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if output.exists() and output.stat().st_size > 0:
            return output.read_bytes()
    except Exception:
        return None
    finally:
        source.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
    return None


def _transition_tag(slide_hold_ms: int, duration_ms: int, *, self_closing: bool) -> str:
    close = "/>" if self_closing else ">"
    return (
        '<p:transition xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" '
        f'advClick="0" advTm="{slide_hold_ms}" dur="{duration_ms}" p14:dur="{duration_ms}"{close}'
    )


def _ps_quote(value: str) -> str:
    return value.replace("'", "''")
