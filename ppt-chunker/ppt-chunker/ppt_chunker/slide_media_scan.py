from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image

from .dependencies import discover_ffmpeg_tools
from .exceptions import PipelineError
from .media import ffprobe_duration
from .models import MediaCandidate, SlideFeature
from .timing import P_NS, R_NS, parse_pptx_timing_xml

A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".wmv",
    ".avi",
    ".m4v",
    ".webm",
    ".mpeg",
    ".mpg",
}
GIF_EXTENSION = ".gif"


def extract_slide_features(pptx_path: Path, ffprobe_bin: str | None = None) -> list[SlideFeature]:
    pptx_path = Path(pptx_path).expanduser().resolve()
    if not zipfile.is_zipfile(pptx_path):
        raise PipelineError(f"Invalid PPTX zip: {pptx_path}")

    ffprobe_path = discover_ffmpeg_tools(None, ffprobe_bin)[1]
    timing_rows = {row.slide_number: row for row in parse_pptx_timing_xml(pptx_path)}

    ns = {"p": P_NS, "a": A_NS, "r": R_NS}
    out: list[SlideFeature] = []

    with zipfile.ZipFile(str(pptx_path)) as zf:
        slide_paths = _ordered_slide_paths(zf)
        slide_w, slide_h = _slide_size_emu(zf)
        for index, slide_path in enumerate(slide_paths, start=1):
            root = ET.parse(zf.open(slide_path)).getroot()
            rel_map = _slide_relationships(zf, slide_path)
            candidates: list[MediaCandidate] = []

            for shape in _iter_slide_shapes(root, ns):
                shape_id = _shape_id(shape, ns)
                geom = _shape_geometry(shape, ns)
                off_canvas, visible = _visibility_flags(geom, slide_w, slide_h)
                rel_ids = _shape_rel_ids(shape)
                for rel_id in rel_ids:
                    target = rel_map.get(rel_id)
                    if not target:
                        continue
                    ext = PurePosixPath(target).suffix.lower()
                    is_media = ext in VIDEO_EXTENSIONS or ext == GIF_EXTENSION
                    if not is_media:
                        continue
                    supported = ext in VIDEO_EXTENSIONS or ext == GIF_EXTENSION
                    duration_s = None
                    probe_error = None
                    if supported:
                        try:
                            duration_s = _probe_duration_s(zf, target, ext, ffprobe_path)
                        except Exception as exc:
                            probe_error = str(exc)
                    candidates.append(
                        MediaCandidate(
                            shape_id=shape_id,
                            rel_id=rel_id,
                            target=target,
                            extension=ext,
                            visible=visible,
                            off_canvas=off_canvas,
                            supported=supported,
                            duration_s=duration_s,
                            duration_probe_error=probe_error,
                        )
                    )

            visible_candidates = [c for c in candidates if c.visible]
            visible_media_count = len(visible_candidates)
            resolved_durations = [
                float(c.duration_s) for c in visible_candidates if c.duration_s is not None and c.duration_s > 0
            ]
            unresolved_visible = [
                c for c in visible_candidates if c.supported and (c.duration_s is None or c.duration_s <= 0)
            ]
            unsupported_visible = [c for c in visible_candidates if not c.supported]
            max_visible = round(max(resolved_durations), 3) if resolved_durations else None
            static_classification = "static" if visible_media_count == 0 else "media"

            timing = timing_rows.get(index)
            transition_type = timing.transition_type if timing is not None else "none"
            transition_duration_s = (
                round(float(timing.transition_duration_s), 3) if timing is not None else 0.0
            )
            label = timing.label if timing is not None else f"Slide {index}"

            out.append(
                SlideFeature(
                    slide_number=index,
                    label=label,
                    transition_type=transition_type,
                    transition_duration_s=transition_duration_s,
                    visible_media_count=visible_media_count,
                    max_visible_media_duration_s=max_visible,
                    unresolved_visible_media_count=len(unresolved_visible),
                    off_canvas_media_count=len([c for c in candidates if c.off_canvas]),
                    unsupported_media_count=len(unsupported_visible),
                    media_candidates=candidates,
                    static_classification=static_classification,
                )
            )
    return out


def _ordered_slide_paths(zf: zipfile.ZipFile) -> list[str]:
    pres_xml = ET.parse(zf.open("ppt/presentation.xml")).getroot()
    rels_xml = ET.parse(zf.open("ppt/_rels/presentation.xml.rels")).getroot()
    ns = {"p": P_NS, "r": R_NS}
    rels_ns = {"r": PKG_REL_NS}

    rel_map: dict[str, str] = {}
    for rel in rels_xml.findall(".//r:Relationship", rels_ns):
        rid = rel.get("Id")
        target = rel.get("Target")
        if rid and target:
            rel_map[rid] = target

    out: list[str] = []
    for node in pres_xml.findall(".//p:sldIdLst/p:sldId", ns):
        rid = node.get(f"{{{R_NS}}}id")
        if not rid or rid not in rel_map:
            continue
        out.append(_normalize_ppt_path("ppt/presentation.xml", rel_map[rid]))

    if out:
        return out

    fallback = sorted(
        [n for n in zf.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")],
        key=lambda x: int(PurePosixPath(x).stem.replace("slide", "")),
    )
    return fallback


def _slide_relationships(zf: zipfile.ZipFile, slide_path: str) -> dict[str, str]:
    rel_path = str(PurePosixPath(slide_path).parent / "_rels" / f"{PurePosixPath(slide_path).name}.rels")
    if rel_path not in zf.namelist():
        return {}
    root = ET.parse(zf.open(rel_path)).getroot()
    rels_ns = {"r": PKG_REL_NS}
    out: dict[str, str] = {}
    for rel in root.findall(".//r:Relationship", rels_ns):
        rid = rel.get("Id")
        target = rel.get("Target")
        if not rid or not target:
            continue
        out[rid] = _normalize_ppt_path(slide_path, target)
    return out


def _slide_size_emu(zf: zipfile.ZipFile) -> tuple[int, int]:
    try:
        root = ET.parse(zf.open("ppt/presentation.xml")).getroot()
    except KeyError:
        return (0, 0)
    ns = {"p": P_NS}
    node = root.find(".//p:sldSz", ns)
    if node is None:
        return (0, 0)
    try:
        return (int(node.get("cx", "0")), int(node.get("cy", "0")))
    except ValueError:
        return (0, 0)


def _iter_slide_shapes(root: Any, ns: dict[str, str]) -> list[Any]:
    sp_tree = root.find(".//p:cSld/p:spTree", ns)
    if sp_tree is None:
        return []
    return list(sp_tree)


def _shape_id(shape: Any, ns: dict[str, str]) -> int | None:
    c_nv = shape.find(".//p:cNvPr", ns)
    if c_nv is None:
        return None
    try:
        return int(c_nv.get("id"))
    except Exception:
        return None


def _shape_geometry(shape: Any, ns: dict[str, str]) -> tuple[int, int, int, int] | None:
    xfrm = shape.find(".//a:xfrm", ns)
    if xfrm is None:
        return None
    off = xfrm.find(".//a:off", ns)
    ext = xfrm.find(".//a:ext", ns)
    if off is None or ext is None:
        return None
    try:
        x = int(off.get("x", "0"))
        y = int(off.get("y", "0"))
        cx = int(ext.get("cx", "0"))
        cy = int(ext.get("cy", "0"))
    except ValueError:
        return None
    return (x, y, cx, cy)


def _visibility_flags(
    geom: tuple[int, int, int, int] | None, slide_w: int, slide_h: int
) -> tuple[bool, bool]:
    if geom is None:
        return (False, True)
    x, y, cx, cy = geom
    if cx <= 0 or cy <= 0:
        return (True, False)
    if slide_w <= 0 or slide_h <= 0:
        return (False, True)
    right = x + cx
    bottom = y + cy
    intersects = right > 0 and bottom > 0 and x < slide_w and y < slide_h
    return (not intersects, intersects)


def _shape_rel_ids(shape: Any) -> list[str]:
    out: list[str] = []
    for node in shape.iter():
        for key, val in node.attrib.items():
            local_key = key.rsplit("}", 1)[-1]
            if local_key in ("embed", "link") and val:
                out.append(str(val))
    return sorted(set(out))


def _normalize_ppt_path(base_path: str, target: str) -> str:
    base = PurePosixPath(base_path).parent
    combined = base / target
    parts: list[str] = []
    for part in combined.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _probe_duration_s(
    zf: zipfile.ZipFile, target_path: str, extension: str, ffprobe_bin: Any
) -> float | None:
    norm_target = target_path.lstrip("/")
    if norm_target not in zf.namelist():
        return None
    raw = zf.read(norm_target)
    if extension == GIF_EXTENSION:
        return _gif_duration_s(raw)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    try:
        tmp.write(raw)
        tmp.flush()
        tmp.close()
        return round(float(ffprobe_duration(ffprobe_bin, Path(tmp.name))), 3)
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass


def _gif_duration_s(raw: bytes) -> float | None:
    image = Image.open(io.BytesIO(raw))
    total_ms = 0
    try:
        while True:
            total_ms += int(image.info.get("duration", 0))
            image.seek(image.tell() + 1)
    except EOFError:
        pass
    if total_ms <= 0:
        return None
    return round(total_ms / 1000.0, 3)
