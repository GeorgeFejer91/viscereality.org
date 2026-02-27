from __future__ import annotations

import re
import time
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from .exceptions import PipelineError
from .models import ResolvedSegment, SlideTimingSource

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def parse_pptx_timing_xml(pptx_path: Path) -> list[SlideTimingSource]:
    if not pptx_path.exists():
        raise PipelineError(f"PPTX file not found: {pptx_path}")

    ns = {"p": P_NS, "a": A_NS, "r": R_NS}
    slides_info: list[SlideTimingSource] = []

    with zipfile.ZipFile(pptx_path) as zf:
        try:
            pres_xml = ET.parse(zf.open("ppt/presentation.xml"))
            rels_xml = ET.parse(zf.open("ppt/_rels/presentation.xml.rels"))
        except KeyError as exc:
            raise PipelineError(f"Invalid PPTX structure: {exc}") from exc

        slide_files = _ordered_slide_files(pres_xml, rels_xml, zf.namelist())
        for idx, slide_file in enumerate(slide_files, start=1):
            full_path = slide_file if slide_file.startswith("ppt/") else f"ppt/{slide_file}"
            try:
                slide_xml = ET.parse(zf.open(full_path))
            except KeyError:
                slide_xml = ET.parse(zf.open(slide_file))

            root = slide_xml.getroot()
            transition = root.find(".//p:transition", ns)
            transition_type = "none"
            transition_duration_s = 0.0
            advance_after_s = None

            if transition is not None:
                spd = transition.get("spd")
                dur_ms = transition.get("dur")
                adv_tm = transition.get("advTm")
                for child in transition:
                    tag = _local_name(child.tag)
                    if tag not in ("sndAc", "sndLst"):
                        transition_type = tag
                        break
                if dur_ms:
                    transition_duration_s = int(dur_ms) / 1000.0
                elif spd:
                    transition_duration_s = {"slow": 1.0, "med": 0.75, "fast": 0.5}.get(spd, 0.75)
                if adv_tm:
                    advance_after_s = int(adv_tm) / 1000.0

            label = _extract_slide_title(root, ns, idx)
            slides_info.append(
                SlideTimingSource(
                    slide_number=idx,
                    label=label,
                    transition_type=transition_type,
                    transition_duration_s=round(transition_duration_s, 3),
                    advance_after_s=round(advance_after_s, 3) if advance_after_s is not None else None,
                )
            )

    return slides_info


def probe_timing_com(
    pptx_path: Path, retries: int = 8, retry_sleep_s: float = 1.0
) -> dict[int, SlideTimingSource]:
    try:
        import pywintypes  # type: ignore
        import win32com.client  # type: ignore
    except Exception:
        return {}

    def com_call(fn):
        for attempt in range(1, retries + 1):
            try:
                return fn()
            except pywintypes.com_error as exc:  # type: ignore[attr-defined]
                text = str(exc).upper()
                if "RPC_E_CALL_REJECTED" in text and attempt < retries:
                    time.sleep(retry_sleep_s * attempt)
                    continue
                raise
        raise PipelineError("PowerPoint COM call retry budget exhausted.")

    app = None
    pres = None
    try:
        app = com_call(lambda: win32com.client.DispatchEx("PowerPoint.Application"))
        pres = com_call(lambda: app.Presentations.Open(str(pptx_path), False, True, False))
        count = int(com_call(lambda: pres.Slides.Count))
        out: dict[int, SlideTimingSource] = {}
        for i in range(1, count + 1):
            slide = com_call(lambda idx=i: pres.Slides(idx))
            tr = com_call(lambda s=slide: s.SlideShowTransition)
            advance_on_time = bool(com_call(lambda t=tr: t.AdvanceOnTime))
            advance_after_s = float(com_call(lambda t=tr: t.AdvanceTime)) if advance_on_time else None
            transition_duration_s = float(com_call(lambda t=tr: t.Duration))
            transition_type = "none"
            try:
                effect = int(com_call(lambda t=tr: t.EntryEffect))
                if effect != 0:
                    transition_type = f"effect_{effect}"
            except Exception:
                pass

            out[i] = SlideTimingSource(
                slide_number=i,
                label=f"Slide {i}",
                transition_type=transition_type,
                transition_duration_s=round(max(0.0, transition_duration_s), 3),
                advance_after_s=round(advance_after_s, 3) if advance_after_s is not None else None,
            )
        return out
    except Exception:
        return {}
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


def load_overrides(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"defaults": {}, "slides": {}, "export": {}}

    if not path.exists():
        raise PipelineError(f"Overrides/config file not found: {path}")

    import json

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    defaults = cfg.get("defaults", {})
    slides: dict[str, dict[str, Any]] = {}
    export = cfg.get("export", {})

    if "segments" in cfg and isinstance(cfg["segments"], list):
        default_slide = cfg.get("default_slide_duration_s")
        default_transition = cfg.get("default_transition_duration_s")
        if default_slide is not None:
            defaults.setdefault("slide_sec", default_slide)
        if default_transition is not None:
            defaults.setdefault("transition_sec", default_transition)
        for row in cfg["segments"]:
            sn = row.get("slide_number")
            if sn is None:
                continue
            slides[str(sn)] = {
                "slide_sec": row.get("slide_duration_s"),
                "transition_sec": row.get("transition_duration_s"),
                "label": row.get("label"),
                "transition_type": row.get("transition_type"),
            }

    if isinstance(cfg.get("slides"), dict):
        for key, row in cfg["slides"].items():
            if isinstance(row, dict):
                slides[str(key)] = {
                    "slide_sec": row.get("slide_sec", row.get("slide_duration_s")),
                    "transition_sec": row.get("transition_sec", row.get("transition_duration_s")),
                    "label": row.get("label"),
                    "transition_type": row.get("transition_type"),
                }

    return {"defaults": defaults, "slides": slides, "export": export}


def resolve_segments(
    xml_sources: list[SlideTimingSource],
    com_sources: dict[int, SlideTimingSource],
    timing_mode: str,
    default_slide_sec: float,
    default_transition_sec: float,
    overrides: dict[str, Any] | None = None,
) -> list[ResolvedSegment]:
    overrides = overrides or {"defaults": {}, "slides": {}, "export": {}}
    default_slide = float(overrides.get("defaults", {}).get("slide_sec", default_slide_sec))
    default_transition = float(
        overrides.get("defaults", {}).get("transition_sec", default_transition_sec)
    )

    out: list[ResolvedSegment] = []
    for xml in xml_sources:
        com = com_sources.get(xml.slide_number)
        transition_type = xml.transition_type or "none"
        source = "default"

        if timing_mode == "uniform":
            slide_sec = default_slide
            transition_sec = default_transition if transition_type != "none" else 0.0
            source = "uniform"
        else:
            slide_sec, slide_src = _choose_first(
                [
                    (getattr(com, "advance_after_s", None), "com"),
                    (xml.advance_after_s, "xml"),
                    (default_slide, "default"),
                ]
            )
            if transition_type == "none":
                transition_sec = 0.0
                trans_src = "none"
            else:
                transition_sec, trans_src = _choose_first(
                    [
                        (getattr(com, "transition_duration_s", None), "com"),
                        (xml.transition_duration_s, "xml"),
                        (default_transition, "default"),
                    ]
                )
            source = f"{slide_src}/{trans_src}"

        ov = overrides.get("slides", {}).get(str(xml.slide_number), {})
        if ov.get("slide_sec") is not None:
            slide_sec = float(ov["slide_sec"])
            source = "override"
        if ov.get("transition_sec") is not None:
            transition_sec = float(ov["transition_sec"])
            source = "override"
        if ov.get("transition_type"):
            transition_type = str(ov["transition_type"])
        label = str(ov.get("label") or xml.label or f"Slide {xml.slide_number}")

        slide_sec = round(max(0.1, float(slide_sec)), 3)
        transition_sec = round(max(0.0, float(transition_sec)), 3)
        if transition_type == "none":
            transition_sec = 0.0
        elif transition_sec > 0 and transition_type == "none":
            transition_type = "custom"

        out.append(
            ResolvedSegment(
                slide_number=xml.slide_number,
                label=label,
                transition_type=transition_type,
                transition_duration_s=transition_sec,
                slide_duration_s=slide_sec,
                duration_source=source,
            )
        )
    return out


def build_timing_config_payload(
    segments: list[ResolvedSegment],
    default_slide_sec: float,
    default_transition_sec: float,
    timing_mode: str,
) -> dict[str, Any]:
    slides_block: dict[str, Any] = {}
    legacy_segments: list[dict[str, Any]] = []
    for seg in segments:
        legacy = seg.to_legacy_json()
        legacy_segments.append(legacy)
        slides_block[str(seg.slide_number)] = {
            "label": seg.label,
            "slide_sec": legacy["slide_duration_s"],
            "transition_sec": legacy["transition_duration_s"],
            "transition_type": seg.transition_type,
        }

    return {
        "_comment": (
            "Backwards-compatible config. You can edit `segments` like v1, or edit "
            "`defaults` + `slides` for v2."
        ),
        "default_slide_duration_s": round(float(default_slide_sec), 3),
        "default_transition_duration_s": round(float(default_transition_sec), 3),
        "segments": legacy_segments,
        "defaults": {
            "slide_sec": round(float(default_slide_sec), 3),
            "transition_sec": round(float(default_transition_sec), 3),
            "timing_mode": timing_mode,
        },
        "slides": slides_block,
    }


def segments_from_config(config: dict[str, Any]) -> list[ResolvedSegment]:
    raw = config.get("segments")
    if not isinstance(raw, list) or not raw:
        raise PipelineError("Config is missing a non-empty `segments` array.")

    out: list[ResolvedSegment] = []
    for row in raw:
        out.append(
            ResolvedSegment(
                slide_number=int(row["slide_number"]),
                label=str(row.get("label") or f"Slide {row['slide_number']}"),
                transition_type=str(row.get("transition_type", "none")),
                transition_duration_s=round(float(row.get("transition_duration_s", 0.0)), 3),
                slide_duration_s=round(float(row["slide_duration_s"]), 3),
                duration_source="config",
            )
        )
    return out


def _choose_first(values: list[tuple[float | None, str]]) -> tuple[float, str]:
    for val, source in values:
        if val is not None:
            return float(val), source
    raise PipelineError("No value available in chooser.")


def _extract_slide_title(root: Any, ns: dict[str, str], slide_number: int) -> str:
    for shape in root.findall(".//p:sp", ns):
        placeholder = shape.find(".//p:nvSpPr/p:nvPr/p:ph", ns)
        if placeholder is None:
            continue
        p_type = placeholder.get("type", "")
        if p_type not in ("title", "ctrTitle"):
            continue
        texts = shape.findall(".//a:t", ns)
        text = "".join(t.text or "" for t in texts).strip()
        if text:
            return text
    return f"Slide {slide_number}"


def _ordered_slide_files(pres_xml: Any, rels_xml: Any, zip_names: list[str]) -> list[str]:
    ns = {"p": P_NS, "r": R_NS}
    rels_ns = {"r": PKG_REL_NS}
    sld_id_nodes = pres_xml.findall(".//p:sldIdLst/p:sldId", ns)
    rid_to_target: dict[str, str] = {}

    for rel in rels_xml.findall(".//r:Relationship", rels_ns):
        rid = rel.get("Id")
        target = rel.get("Target")
        if rid and target:
            rid_to_target[rid] = target
    if not rid_to_target:
        for rel in rels_xml.getroot():
            rid = rel.get("Id")
            target = rel.get("Target")
            if rid and target:
                rid_to_target[rid] = target

    out = []
    for node in sld_id_nodes:
        rid = node.get(f"{{{R_NS}}}id")
        if rid and rid in rid_to_target:
            out.append(rid_to_target[rid].lstrip("/"))

    if out:
        return out

    fallback = [
        name
        for name in zip_names
        if name.startswith("ppt/slides/slide") and name.endswith(".xml")
    ]
    fallback.sort(key=lambda n: int(re.search(r"slide(\d+)", n).group(1)))
    return fallback


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag
