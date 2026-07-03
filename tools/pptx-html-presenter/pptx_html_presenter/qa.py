from __future__ import annotations

import copy
import math
import re
import subprocess
import tempfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

from .errors import PresenterError
from .utils import ensure_dir, find_binary, read_json, utc_now_iso, write_json


def run_qa(
    build_dir: Path,
    *,
    reference_mp4: Path | None = None,
    ffmpeg_bin: str | None = None,
    node_bin: str | None = None,
    playwright_dir: Path | None = None,
    reuse_html: bool = False,
    calibrate: bool = False,
    slide_hold_sec: float | None = None,
    settled_offset_sec: float | None = None,
    transition_reference_lead_fraction: float | None = None,
    slides: set[int] | None = None,
    visual_audit: bool | None = None,
) -> dict[str, Any]:
    build_dir = build_dir.expanduser().resolve()
    scene_path = build_dir / "deck.scene.json"
    if not scene_path.exists():
        raise PresenterError(f"Missing deck.scene.json in {build_dir}")
    scene = read_json(scene_path)
    qa_dir = ensure_dir(build_dir / "qa")
    samples = _sample_plan(
        scene,
        slide_hold_sec=slide_hold_sec,
        settled_offset_sec=settled_offset_sec,
        transition_reference_lead_fraction=transition_reference_lead_fraction,
    )
    if slides is not None:
        samples = _filter_samples_for_slides(samples, slides)
    blockers: list[str] = []
    reference_blockers: list[str] = []
    html_blockers: list[str] = []
    ffmpeg = find_binary("ffmpeg.exe", ffmpeg_bin) or find_binary("ffmpeg", ffmpeg_bin)
    node = find_binary("node.exe", node_bin) or find_binary("node", node_bin)
    html_dir = ensure_dir(qa_dir / "html")
    reuse_existing_html = reuse_html and _frames_exist(html_dir, samples)
    if reference_mp4 is None:
        reference_blockers.append("reference-mp4-not-provided")
    elif not reference_mp4.expanduser().exists():
        reference_blockers.append(f"reference-mp4-missing:{reference_mp4}")
    if reference_mp4 is not None and ffmpeg is None:
        reference_blockers.append("ffmpeg-missing")
    if not reuse_existing_html:
        if node is None:
            html_blockers.append("node-missing")
        elif not _node_has_playwright(node, playwright_dir):
            html_blockers.append("playwright-missing")
    blockers = [*reference_blockers, *html_blockers]

    extracted_frames: list[dict[str, Any]] = []
    if not reference_blockers and reference_mp4 is not None and ffmpeg is not None:
        ref_dir = ensure_dir(qa_dir / "reference")
        for sample in samples:
            out = ref_dir / f"{sample['id']}.png"
            _extract_reference_frame(ffmpeg, reference_mp4.expanduser().resolve(), out, sample["referenceSec"])
            extracted_frames.append({"sampleId": sample["id"], "file": out.relative_to(build_dir).as_posix()})
    html_frames: list[dict[str, Any]] = []
    html_capture_report: dict[str, Any] = {}
    samples_path = qa_dir / "samples.json"
    write_json(samples_path, samples)
    if reuse_existing_html or (not html_blockers and node is not None):
        if not reuse_existing_html:
            html_capture_report = _capture_html_frames(node, build_dir, samples_path, html_dir, playwright_dir)
        else:
            html_capture_report = _read_capture_report(html_dir)
        capture_failures = int(html_capture_report.get("failures", 0) or 0)
        if capture_failures:
            html_blockers.append(f"html-capture-failed:{capture_failures}")
        html_frames = [
            {"sampleId": sample["id"], "file": (html_dir / f"{sample['id']}.png").relative_to(build_dir).as_posix()}
            for sample in samples
            if (html_dir / f"{sample['id']}.png").exists()
        ]
    comparisons = []
    if extracted_frames and html_frames:
        comparisons = _compare_frame_sets(build_dir, samples)
    calibrated_comparisons: list[dict[str, Any]] = []
    calibration_summary: dict[str, Any] = {"enabled": calibrate}
    if calibrate and comparisons and reference_mp4 is not None and ffmpeg is not None:
        calibrated_comparisons = _calibrate_reference_alignment(
            ffmpeg,
            reference_mp4.expanduser().resolve(),
            build_dir,
            samples,
            comparisons,
        )
        calibration_summary = _calibration_summary(calibrated_comparisons)
        calibration_summary["enabled"] = True
    blockers = [*reference_blockers, *html_blockers]
    visual_audit_report: dict[str, Any] | None = None
    visual_audit_enabled = (
        bool(visual_audit)
        if visual_audit is not None
        else bool(((scene.get("qa") or {}).get("visualAudit") or {}).get("enabled", False))
    )
    if visual_audit_enabled:
        visual_audit_report = run_visual_audit(
            build_dir,
            node_bin=str(node) if node else node_bin,
            playwright_dir=playwright_dir,
        )
        if visual_audit_report.get("status") != "passed":
            blockers.append(
                f"visual-audit-{visual_audit_report.get('status')}:{visual_audit_report.get('summary', {}).get('failureCount', 0)}"
            )

    failed = [row for row in comparisons if not row.get("passed", False)]
    if blockers:
        status = "blocked"
    elif failed:
        status = "failed"
    elif comparisons:
        status = "passed"
    else:
        status = "partial"

    report = {
        "schema": "pptx-html-presenter.qa.v1",
        "generatedAtUtc": utc_now_iso(),
        "status": status,
        "blockers": blockers,
        "thresholds": {
            "slideSsim": 0.985,
            "morphSsim": 0.965,
            "ssimMethod": "local-uniform-11-luma",
        },
        "samples": samples,
        "referenceFrames": extracted_frames,
        "htmlFrames": html_frames,
        "htmlCapture": html_capture_report,
        "comparisons": comparisons,
        "calibration": calibration_summary,
        "calibratedComparisons": calibrated_comparisons,
        "visualAudit": visual_audit_report,
        "notes": [
            "HTML frame capture is routed through window.PptxHtmlPresenter.captureAt().",
            "Strict pass/fail uses predicted PowerPoint MP4 timestamps; calibratedComparisons are diagnostic only.",
            "Install ffmpeg, Node, and Playwright to run the complete automated comparison.",
        ],
    }
    write_json(qa_dir / "report.json", report)
    _write_contact_sheet_stub(qa_dir, report)
    return report


def run_visual_audit(
    build_dir: Path,
    *,
    node_bin: str | None = None,
    playwright_dir: Path | None = None,
    samples: tuple[float, ...] | None = None,
    fail_on_timeout: bool | None = None,
) -> dict[str, Any]:
    build_dir = build_dir.expanduser().resolve()
    scene_path = build_dir / "deck.scene.json"
    if not scene_path.exists():
        raise PresenterError(f"Missing deck.scene.json in {build_dir}")
    scene = read_json(scene_path)
    audit_config = ((scene.get("qa") or {}).get("visualAudit") or {})
    audit_dir = ensure_dir(build_dir / "qa" / "visual-audit")
    node = find_binary("node.exe", node_bin) or find_binary("node", node_bin)
    blockers: list[str] = []
    if node is None:
        blockers.append("node-missing")
    elif not _node_has_playwright(node, playwright_dir):
        blockers.append("playwright-missing")

    audit_samples = _visual_audit_sample_plan(
        scene,
        samples=tuple(float(v) for v in (samples or audit_config.get("samples") or (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))),
        reverse_midpoints=bool(audit_config.get("reverseMidpoints", True)),
    )
    samples_path = audit_dir / "samples.json"
    write_json(samples_path, audit_samples)
    html_dir = ensure_dir(audit_dir / "html")
    for stale in [
        *html_dir.glob("*.png"),
        html_dir / "capture-report.json",
        audit_dir / "settled-slides-contact-sheet.png",
        audit_dir / "transition-midpoints-contact-sheet.png",
        audit_dir / "failures-contact-sheet.png",
    ]:
        if stale.exists():
            stale.unlink()
    capture_report: dict[str, Any] = {}
    if not blockers and node is not None:
        capture_report = _capture_html_frames(node, build_dir, samples_path, html_dir, playwright_dir)

    frames = _visual_audit_frame_rows(build_dir, html_dir, audit_samples, capture_report)
    failure_count = sum(1 for row in frames if row["status"] == "failed")
    warning_count = sum(1 for row in frames if row["status"] == "warning")
    fail_on_timeout_effective = bool(
        audit_config.get("failOnTimeout", True) if fail_on_timeout is None else fail_on_timeout
    )
    if blockers:
        status = "blocked"
    elif failure_count and fail_on_timeout_effective:
        status = "failed"
    else:
        status = "passed" if not failure_count else "warning"

    contact_sheets = _write_visual_audit_contact_sheets(build_dir, audit_dir, frames)
    report = {
        "schema": "pptx-html-presenter.visual-audit.v1",
        "generatedAtUtc": utc_now_iso(),
        "status": status,
        "blockers": blockers,
        "samples": audit_samples,
        "capture": capture_report,
        "frames": frames,
        "contactSheets": contact_sheets,
        "summary": {
            "sampleCount": len(audit_samples),
            "capturedCount": sum(1 for row in frames if row.get("file")),
            "failureCount": failure_count,
            "warningCount": warning_count,
            "settledSlideCount": sum(1 for row in frames if row.get("kind") == "slide"),
            "forwardTransitionCount": sum(1 for row in frames if row.get("direction") == "forward" and row.get("kind") == "transition"),
            "reverseTransitionCount": sum(1 for row in frames if row.get("direction") == "reverse" and row.get("kind") == "transition"),
        },
        "notes": [
            "Failures include capture timeouts, missing screenshots, and near-uniform blank frames.",
            "Contact sheets are intended for human review of overlap, layering, and object continuity.",
        ],
    }
    write_json(audit_dir / "report.json", report)
    return report


def _visual_audit_sample_plan(
    scene: dict[str, Any],
    *,
    samples: tuple[float, ...],
    reverse_midpoints: bool,
) -> list[dict[str, Any]]:
    audit_scene = copy.deepcopy(scene)
    audit_scene.setdefault("qa", {})["transitionSamples"] = list(samples)
    planned = _sample_plan(audit_scene)
    out: list[dict[str, Any]] = []
    forward_midpoints: dict[tuple[int, int], dict[str, Any]] = {}
    for sample in planned:
        row = copy.deepcopy(sample)
        if row.get("kind") == "slide":
            row["direction"] = "forward"
            row["auditKind"] = "settled-slide"
            row["expectedVisibleObjects"] = _expected_visible_object_floor(
                scene, int(row.get("slide") or 0)
            )
            out.append(row)
            continue
        if row.get("kind") == "transition":
            row["direction"] = "forward"
            row["auditKind"] = "forward-transition"
            row["expectedVisibleObjects"] = _expected_transition_visible_object_floor(
                scene, int(row.get("from") or 0), int(row.get("to") or 0)
            )
            out.append(row)
            if abs(float(row.get("progress", 0.0) or 0.0) - 0.5) <= 0.0001:
                forward_midpoints[(int(row["from"]), int(row["to"]))] = row

    if reverse_midpoints:
        for transition in scene.get("transitions", []) or []:
            try:
                from_slide = int(transition["from"])
                to_slide = int(transition["to"])
            except (TypeError, ValueError, KeyError):
                continue
            source = forward_midpoints.get((from_slide, to_slide), {})
            reverse = copy.deepcopy(source)
            reverse.update(
                {
                    "id": f"reverse-{to_slide:03d}-{from_slide:03d}-050",
                    "kind": "transition",
                    "from": to_slide,
                    "to": from_slide,
                    "progress": 0.5,
                    "direction": "reverse",
                    "auditKind": "reverse-transition",
                    "referenceSec": source.get("referenceSec", 0.0),
                    "mediaSec": source.get("mediaSec", 0.0),
                    "mediaClocks": source.get("mediaClocks", {}),
                    "expectedVisibleObjects": source.get("expectedVisibleObjects"),
                }
            )
            out.append(reverse)
    return out


def _expected_visible_object_floor(scene: dict[str, Any], *slide_numbers: int) -> int:
    counts: list[int] = []
    slides = scene.get("slides", []) or []
    for slide_number in slide_numbers:
        if slide_number <= 0 or slide_number > len(slides):
            continue
        slide = slides[slide_number - 1]
        objects = slide.get("nodes") or slide.get("objects") or []
        visible = [obj for obj in objects if _object_can_intersect_slide(obj)]
        if visible:
            counts.append(len(visible))
    if not counts:
        return 1
    return max(1, min(3, max(counts)))


def _expected_transition_visible_object_floor(
    scene: dict[str, Any],
    from_slide_number: int,
    to_slide_number: int,
) -> int:
    from_tracks = _visible_track_ids(scene, from_slide_number)
    to_tracks = _visible_track_ids(scene, to_slide_number)
    common_tracks = from_tracks & to_tracks
    if len(common_tracks) <= 1:
        return 0
    return max(2, min(3, len(common_tracks)))


def _visible_track_ids(scene: dict[str, Any], slide_number: int) -> set[str]:
    slides = scene.get("slides", []) or []
    if slide_number <= 0 or slide_number > len(slides):
        return set()
    slide = slides[slide_number - 1]
    objects = slide.get("nodes") or slide.get("objects") or []
    return {
        str(obj.get("trackId"))
        for obj in objects
        if obj.get("trackId") and _object_can_intersect_slide(obj)
    }


def _object_can_intersect_slide(obj: dict[str, Any]) -> bool:
    if (obj.get("rasterFallback") or {}).get("settledOnly"):
        return False
    if float(obj.get("opacity", 1.0) or 0.0) <= 0.005:
        return False
    geometry = obj.get("geometry") or {}
    try:
        left = float(geometry.get("leftPct", 0.0) or 0.0)
        top = float(geometry.get("topPct", 0.0) or 0.0)
        width = float(geometry.get("widthPct", 0.0) or 0.0)
        height = float(geometry.get("heightPct", 0.0) or 0.0)
    except (TypeError, ValueError):
        return True
    return width > 0 and height > 0 and left < 1 and left + width > 0 and top < 1 and top + height > 0


def _read_capture_report(html_dir: Path) -> dict[str, Any]:
    report_path = html_dir / "capture-report.json"
    if not report_path.exists():
        return {}
    try:
        return read_json(report_path)
    except Exception:
        return {}


def _visual_audit_frame_rows(
    build_dir: Path,
    html_dir: Path,
    samples: list[dict[str, Any]],
    capture_report: dict[str, Any],
) -> list[dict[str, Any]]:
    capture_by_id = {
        str(row.get("id")): row
        for row in (capture_report.get("samples", []) or [])
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["id"])
        path = html_dir / f"{sample_id}.png"
        capture = capture_by_id.get(sample_id, {})
        diagnostics = capture.get("diagnostics") or {}
        quality = _visual_frame_quality(path) if path.exists() else {"exists": False}
        reasons: list[str] = []
        if not path.exists():
            reasons.append("missing-screenshot")
        if capture.get("status") and capture.get("status") != "ok":
            reasons.append(str(capture.get("error") or "capture-failed"))
        if quality.get("nearUniform"):
            reasons.append("near-uniform-frame")
        visible_objects: int | None = None
        try:
            visible_objects = int(diagnostics.get("objectsVisible", 1))
            if visible_objects <= 0:
                reasons.append("no-visible-objects")
        except (TypeError, ValueError):
            pass
        expected_visible = int(sample.get("expectedVisibleObjects") or 0)
        progress = float(sample.get("progress") or 0.0)
        if (
            sample.get("kind") == "transition"
            and 0.0 < progress < 1.0
            and expected_visible > 1
            and visible_objects is not None
            and visible_objects < expected_visible
        ):
            reasons.append(f"visible-objects-below-floor:{visible_objects}<{expected_visible}")
        status = "failed" if reasons else "ok"
        row = {
            "sampleId": sample_id,
            "kind": sample.get("kind"),
            "auditKind": sample.get("auditKind"),
            "direction": sample.get("direction", "forward"),
            "from": sample.get("from"),
            "to": sample.get("to"),
            "slide": sample.get("slide"),
            "progress": sample.get("progress"),
            "status": status,
            "reasons": reasons,
            "file": _path_for_report(path, build_dir) if path.exists() else None,
            "diagnostics": diagnostics,
            "quality": quality,
        }
        if sample.get("expectedVisibleObjects") is not None:
            row["expectedVisibleObjects"] = sample.get("expectedVisibleObjects")
        if capture.get("pageEvents"):
            row["pageEvents"] = capture.get("pageEvents")
        rows.append(row)
    return rows


def _visual_frame_quality(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return {"exists": path.exists()}
    if not path.exists():
        return {"exists": False}
    with Image.open(path).convert("RGB") as image:
        arr = np.asarray(image, dtype=np.float32)
    luma = arr[:, :, 0] * 0.2126 + arr[:, :, 1] * 0.7152 + arr[:, :, 2] * 0.0722
    std = float(luma.std())
    mean = float(luma.mean())
    return {
        "exists": True,
        "meanLuma": round(mean, 3),
        "stdLuma": round(std, 3),
        "nearUniform": std < 1.5,
    }


def _write_visual_audit_contact_sheets(
    build_dir: Path,
    audit_dir: Path,
    frames: list[dict[str, Any]],
) -> list[str]:
    sheets: list[str] = []
    sheets.extend(
        _write_contact_sheet_png(
            build_dir,
            audit_dir / "settled-slides-contact-sheet.png",
            [row for row in frames if row.get("kind") == "slide"],
        )
    )
    sheets.extend(
        _write_contact_sheet_png(
            build_dir,
            audit_dir / "transition-midpoints-contact-sheet.png",
            [
                row
                for row in frames
                if row.get("kind") == "transition"
                and abs(float(row.get("progress", 0.0) or 0.0) - 0.5) <= 0.0001
            ],
        )
    )
    sheets.extend(
        _write_contact_sheet_png(
            build_dir,
            audit_dir / "failures-contact-sheet.png",
            [row for row in frames if row.get("status") == "failed"],
        )
    )
    return sheets


def _write_contact_sheet_png(
    build_dir: Path,
    out: Path,
    rows: list[dict[str, Any]],
) -> list[str]:
    if not rows:
        return []
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return []
    tile_w = 320
    image_h = 180
    label_h = 34
    gap = 12
    cols = 5
    rows_count = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * tile_w + (cols + 1) * gap, rows_count * (image_h + label_h) + (rows_count + 1) * gap), "#111111")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        col = index % cols
        row_index = index // cols
        x = gap + col * (tile_w + gap)
        y = gap + row_index * (image_h + label_h + gap)
        file_value = row.get("file")
        image_path = build_dir / file_value if file_value else None
        if image_path and image_path.exists():
            with Image.open(image_path).convert("RGB") as thumb:
                thumb.thumbnail((tile_w, image_h), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (tile_w, image_h), "#000000")
                canvas.paste(thumb, ((tile_w - thumb.width) // 2, (image_h - thumb.height) // 2))
                sheet.paste(canvas, (x, y))
        else:
            draw.rectangle([x, y, x + tile_w, y + image_h], fill="#202020", outline="#555555")
            draw.text((x + 10, y + 78), "missing", fill="#dddddd")
        status = str(row.get("status") or "")
        label = f"{row.get('sampleId')}  {status}"
        draw.rectangle([x, y + image_h, x + tile_w, y + image_h + label_h], fill="#181818")
        draw.text((x + 6, y + image_h + 7), label[:56], fill="#ff7777" if status == "failed" else "#e6e6e6")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return [_path_for_report(out, build_dir)]


def _filter_samples_for_slides(samples: list[dict[str, Any]], slides: set[int]) -> list[dict[str, Any]]:
    return [
        sample
        for sample in samples
        if (
            sample.get("kind") == "slide"
            and int(sample.get("slide", 0) or 0) in slides
        )
        or (
            sample.get("kind") == "transition"
            and int(sample.get("from", 0) or 0) in slides
        )
    ]


def run_candidate_sweep(
    build_dir: Path,
    *,
    sample_id: str,
    vary: str,
    values: list[float],
    track_id: str | None = None,
    reference_frame: Path | None = None,
    reference_mp4: Path | None = None,
    ffmpeg_bin: str | None = None,
    node_bin: str | None = None,
    playwright_dir: Path | None = None,
    reuse_html: bool = False,
) -> dict[str, Any]:
    try:
        import numpy as np
        from PIL import Image, ImageChops
    except Exception as exc:
        raise PresenterError(f"Candidate sweep needs Pillow and NumPy: {exc}") from exc

    build_dir = build_dir.expanduser().resolve()
    scene_path = build_dir / "deck.scene.json"
    if not scene_path.exists():
        raise PresenterError(f"Missing deck.scene.json in {build_dir}")
    if not values:
        raise PresenterError("Candidate sweep needs at least one value.")

    scene = read_json(scene_path)
    qa_dir = ensure_dir(build_dir / "qa")
    base_sample = _candidate_sweep_base_sample(build_dir, scene, sample_id)
    sweep_vary = _normalize_candidate_sweep_vary(vary)
    candidate_samples = _candidate_sweep_samples(base_sample, sweep_vary, values, track_id)

    ref_path = _candidate_sweep_reference_frame(
        build_dir,
        qa_dir,
        base_sample,
        reference_frame=reference_frame,
        reference_mp4=reference_mp4,
        ffmpeg_bin=ffmpeg_bin,
    )
    node = find_binary("node.exe", node_bin) or find_binary("node", node_bin)
    if node is None:
        raise PresenterError("node missing for candidate sweep")
    if not _node_has_playwright(node, playwright_dir):
        raise PresenterError("playwright missing for candidate sweep")

    sweep_id = _candidate_sweep_dir_name(str(base_sample["id"]), sweep_vary, track_id)
    sweep_dir = ensure_dir(qa_dir / "candidate-sweep" / sweep_id)
    html_dir = ensure_dir(sweep_dir / "html")
    diff_dir = ensure_dir(sweep_dir / "diff")
    side_by_side_dir = ensure_dir(sweep_dir / "side-by-side")
    samples_path = sweep_dir / "samples.json"
    write_json(samples_path, candidate_samples)
    if not (reuse_html and _frames_exist(html_dir, candidate_samples)):
        _capture_html_frames(node, build_dir, samples_path, html_dir, playwright_dir)

    rows: list[dict[str, Any]] = []
    with Image.open(ref_path).convert("RGB") as ref_img:
        ref_arr = np.asarray(ref_img, dtype=np.float32)
        for sample in candidate_samples:
            html_path = html_dir / f"{sample['id']}.png"
            if not html_path.exists():
                continue
            with Image.open(html_path).convert("RGB") as html_img:
                if html_img.size != ref_img.size:
                    html_img = html_img.resize(ref_img.size)
                diff = ImageChops.difference(ref_img, html_img)
                diff_path = diff_dir / f"{sample['id']}.png"
                side_by_side_path = side_by_side_dir / f"{sample['id']}.png"
                diff.save(diff_path)
                side_by_side = Image.new("RGB", (ref_img.width * 3, ref_img.height))
                side_by_side.paste(ref_img, (0, 0))
                side_by_side.paste(html_img, (ref_img.width, 0))
                side_by_side.paste(diff, (ref_img.width * 2, 0))
                side_by_side.save(side_by_side_path)
                html_arr = np.asarray(html_img, dtype=np.float32)
            delta = np.abs(ref_arr - html_arr)
            score = _global_ssim(ref_arr, html_arr)
            sweep_meta = sample.get("candidateSweep", {}) or {}
            rows.append(
                {
                    "sampleId": base_sample["id"],
                    "candidateId": sample["id"],
                    "kind": base_sample.get("kind"),
                    "vary": sweep_vary,
                    "trackId": track_id,
                    "value": sweep_meta.get("value"),
                    "ssim": round(float(score), 6),
                    "meanDelta": round(float(delta.mean()), 3),
                    "p95Delta": round(float(np.percentile(delta, 95)), 3),
                    "htmlFrame": _path_for_report(html_path, build_dir),
                    "diffFile": _path_for_report(diff_path, build_dir),
                    "sideBySideFile": _path_for_report(side_by_side_path, build_dir),
                }
            )

    best = max(rows, key=lambda row: float(row.get("ssim", 0.0) or 0.0), default=None)
    report = {
        "schema": "pptx-html-presenter.candidate-sweep.v1",
        "generatedAtUtc": utc_now_iso(),
        "sampleId": base_sample["id"],
        "sourceSample": base_sample,
        "vary": sweep_vary,
        "trackId": track_id,
        "values": [round(float(value), 6) for value in values],
        "referenceFrame": _path_for_report(ref_path, build_dir),
        "samples": candidate_samples,
        "rows": rows,
        "best": best,
        "summary": {
            "candidateCount": len(candidate_samples),
            "scoredCount": len(rows),
            "outputDir": _path_for_report(sweep_dir, build_dir),
        },
    }
    write_json(sweep_dir / "report.json", report)
    return report


def run_transition_time_calibration(
    build_dir: Path,
    *,
    reference_mp4: Path,
    ffmpeg_bin: str | None = None,
    node_bin: str | None = None,
    playwright_dir: Path | None = None,
    slides: set[int] | None = None,
    fps: int = 8,
    window_sec: float = 1.0,
    min_score: float = 0.55,
    apply: bool = False,
    reuse_html: bool = True,
) -> dict[str, Any]:
    build_dir = build_dir.expanduser().resolve()
    scene_path = build_dir / "deck.scene.json"
    if not scene_path.exists():
        raise PresenterError(f"Missing deck.scene.json in {build_dir}")
    reference = reference_mp4.expanduser().resolve()
    if not reference.exists():
        raise PresenterError(f"Reference MP4 not found: {reference}")
    ffmpeg = find_binary("ffmpeg.exe", ffmpeg_bin) or find_binary("ffmpeg", ffmpeg_bin)
    if ffmpeg is None:
        raise PresenterError("ffmpeg missing")

    scene = read_json(scene_path)
    qa_dir = ensure_dir(build_dir / "qa")
    samples = [
        sample
        for sample in _sample_plan(scene)
        if sample.get("kind") == "transition"
        and (slides is None or int(sample.get("from", 0) or 0) in slides)
    ]
    if not samples:
        raise PresenterError("No transition samples matched the requested slide filter.")
    html_dir = ensure_dir(qa_dir / "html")
    if not (reuse_html and _frames_exist(html_dir, samples)):
        node = find_binary("node.exe", node_bin) or find_binary("node", node_bin)
        if node is None:
            raise PresenterError("node missing for transition timing calibration")
        if not _node_has_playwright(node, playwright_dir):
            raise PresenterError("playwright missing for transition timing calibration")
        samples_path = qa_dir / "transition-time-samples.json"
        write_json(samples_path, samples)
        _capture_html_frames(node, build_dir, samples_path, html_dir, playwright_dir)

    alignment_rows = _calibrate_reference_alignment(
        ffmpeg,
        reference,
        build_dir,
        samples,
        [],
        fps=fps,
        slide_window_sec=window_sec,
        transition_window_sec=window_sec,
    )
    current_offsets = _transition_time_current_offsets(scene)
    config_overrides = _transition_time_config_overrides(
        alignment_rows,
        samples,
        current_offsets,
        min_score,
    )
    if apply:
        scene.setdefault("qa", {})["transitionTimeOverrides"] = _merge_transition_time_overrides(
            scene.get("qa", {}).get("transitionTimeOverrides", []),
            config_overrides,
        )
        write_json(scene_path, scene)

    report = {
        "schema": "pptx-html-presenter.transition-time.v1",
        "generatedAtUtc": utc_now_iso(),
        "reference": str(reference),
        "fps": fps,
        "windowSec": window_sec,
        "minScore": min_score,
        "applied": apply,
        "rows": alignment_rows,
        "configOverrides": config_overrides,
        "summary": {
            "sampleCount": len(alignment_rows),
            "overrideCount": len(config_overrides),
            "slides": sorted({int(sample["from"]) for sample in samples}),
        },
    }
    write_json(qa_dir / "transition-time-report.json", report)
    return report


def run_morph_progress_calibration(
    build_dir: Path,
    *,
    reference_mp4: Path,
    ffmpeg_bin: str | None = None,
    node_bin: str | None = None,
    playwright_dir: Path | None = None,
    slides: set[int] | None = None,
    candidate_step: float = 0.05,
    min_score: float = 0.55,
    compare_mode: str = "auto",
    reuse_html: bool = False,
) -> dict[str, Any]:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        raise PresenterError(f"Morph progress calibration needs Pillow and NumPy: {exc}") from exc

    build_dir = build_dir.expanduser().resolve()
    scene_path = build_dir / "deck.scene.json"
    if not scene_path.exists():
        raise PresenterError(f"Missing deck.scene.json in {build_dir}")
    reference = reference_mp4.expanduser().resolve()
    if not reference.exists():
        raise PresenterError(f"Reference MP4 not found: {reference}")
    ffmpeg = find_binary("ffmpeg.exe", ffmpeg_bin) or find_binary("ffmpeg", ffmpeg_bin)
    if ffmpeg is None:
        raise PresenterError("ffmpeg missing")
    node = find_binary("node.exe", node_bin) or find_binary("node", node_bin)
    if node is None:
        raise PresenterError("node missing for Morph progress calibration")
    if not _node_has_playwright(node, playwright_dir):
        raise PresenterError("playwright missing for Morph progress calibration")

    scene = read_json(scene_path)
    qa_dir = ensure_dir(build_dir / "qa")
    progress_dir = ensure_dir(qa_dir / "morph-progress")
    ref_dir = ensure_dir(qa_dir / "reference")
    html_dir = ensure_dir(progress_dir / "html")
    samples = [
        sample
        for sample in _sample_plan(scene)
        if sample.get("kind") == "transition"
        and (slides is None or int(sample.get("from", 0) or 0) in slides)
    ]
    if not samples:
        raise PresenterError("No transition samples matched the requested slide filter.")
    candidates = _morph_progress_candidates(candidate_step)
    candidate_samples = _morph_progress_candidate_samples(samples, candidates)
    samples_path = progress_dir / "candidate-samples.json"
    write_json(samples_path, candidate_samples)
    pairs = {
        (int(sample.get("from", 0) or 0), int(sample.get("to", 0) or 0))
        for sample in samples
    }
    if not (reuse_html and _frames_exist(html_dir, candidate_samples)):
        capture_scene = _neutralized_progress_scene(scene, pairs)
        _capture_html_frames_with_scene_override(
            node,
            build_dir,
            scene_path,
            capture_scene,
            samples_path,
            html_dir,
            playwright_dir,
        )

    for sample in samples:
        ref_path = ref_dir / f"{sample['id']}.png"
        if not ref_path.exists():
            _extract_reference_frame(ffmpeg, reference, ref_path, float(sample["referenceSec"]))

    transitions_by_pair = {
        (int(transition.get("from", 0) or 0), int(transition.get("to", 0) or 0)): transition
        for transition in scene.get("transitions", [])
    }
    slides_by_index = {
        int(slide.get("index", 0) or 0): slide
        for slide in scene.get("slides", [])
    }
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["id"])
        from_slide = int(sample.get("from", 0) or 0)
        to_slide = int(sample.get("to", 0) or 0)
        transition = transitions_by_pair.get((from_slide, to_slide), {})
        anchor_tracks = _morph_progress_anchor_tracks(
            slides_by_index.get(from_slide, {}),
            slides_by_index.get(to_slide, {}),
            transition,
        )
        ref_path = ref_dir / f"{sample_id}.png"
        with Image.open(ref_path).convert("RGB") as ref_img:
            crop_bbox = _morph_progress_crop_bbox(
                slides_by_index.get(from_slide, {}),
                slides_by_index.get(to_slide, {}),
                transition,
                anchor_tracks,
                ref_img.size,
            )
            use_crop = crop_bbox is not None and compare_mode != "full"
            if compare_mode == "anchors" and crop_bbox is None:
                use_crop = False
            ref_cmp = ref_img.crop(crop_bbox) if use_crop and crop_bbox else ref_img
            ref_arr = np.asarray(ref_cmp, dtype=np.float32)
            best: dict[str, Any] | None = None
            candidate_rows: list[dict[str, Any]] = []
            for candidate in candidates:
                candidate_id = _morph_progress_candidate_id(sample_id, candidate)
                html_path = html_dir / f"{candidate_id}.png"
                if not html_path.exists():
                    continue
                with Image.open(html_path).convert("RGB") as html_img:
                    if html_img.size != ref_img.size:
                        html_img = html_img.resize(ref_img.size)
                    html_cmp = html_img.crop(crop_bbox) if use_crop and crop_bbox else html_img
                    html_arr = np.asarray(html_cmp, dtype=np.float32)
                score = _global_ssim(ref_arr, html_arr)
                candidate_row = {
                    "value": round(float(candidate), 4),
                    "score": round(float(score), 6),
                }
                candidate_rows.append(candidate_row)
                if best is None or score > float(best["score"]):
                    best = candidate_row
            if best is None:
                continue
            rows.append(
                {
                    "sampleId": sample_id,
                    "from": from_slide,
                    "to": to_slide,
                    "progress": round(float(sample.get("progress", 0.0) or 0.0), 4),
                    "bestProgressValue": best["value"],
                    "score": best["score"],
                    "threshold": min_score,
                    "passed": float(best["score"]) >= min_score,
                    "compareMode": "anchors" if use_crop else "full",
                    "anchorTracks": anchor_tracks,
                    "cropBbox": list(crop_bbox) if use_crop and crop_bbox else None,
                    "topCandidates": sorted(
                        candidate_rows,
                        key=lambda item: float(item["score"]),
                        reverse=True,
                    )[:5],
                }
            )

    config_overrides = _morph_progress_config_overrides(rows, min_score)
    report = {
        "schema": "pptx-html-presenter.morph-progress.v1",
        "generatedAtUtc": utc_now_iso(),
        "reference": str(reference),
        "candidateStep": candidate_step,
        "minScore": min_score,
        "compareMode": compare_mode,
        "rows": rows,
        "configOverrides": config_overrides,
        "summary": {
            "sampleCount": len(rows),
            "overrideCount": len(config_overrides),
            "slides": sorted({int(row["from"]) for row in rows}),
            "candidateCount": len(candidates),
        },
    }
    write_json(progress_dir / "report.json", report)
    return report


def run_track_progress_calibration(
    build_dir: Path,
    *,
    reference_mp4: Path,
    ffmpeg_bin: str | None = None,
    node_bin: str | None = None,
    playwright_dir: Path | None = None,
    slides: set[int] | None = None,
    tracks: set[str] | None = None,
    progresses: set[float] | None = None,
    candidate_step: float = 0.05,
    min_score: float = 0.0,
    min_improvement: float = 0.002,
    stability_weight: float = 0.02,
    reuse_html: bool = False,
) -> dict[str, Any]:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        raise PresenterError(f"Track progress calibration needs Pillow and NumPy: {exc}") from exc

    build_dir = build_dir.expanduser().resolve()
    scene_path = build_dir / "deck.scene.json"
    if not scene_path.exists():
        raise PresenterError(f"Missing deck.scene.json in {build_dir}")
    reference = reference_mp4.expanduser().resolve()
    if not reference.exists():
        raise PresenterError(f"Reference MP4 not found: {reference}")
    ffmpeg = find_binary("ffmpeg.exe", ffmpeg_bin) or find_binary("ffmpeg", ffmpeg_bin)
    if ffmpeg is None:
        raise PresenterError("ffmpeg missing")
    node = find_binary("node.exe", node_bin) or find_binary("node", node_bin)
    if node is None:
        raise PresenterError("node missing for track progress calibration")
    if not _node_has_playwright(node, playwright_dir):
        raise PresenterError("playwright missing for track progress calibration")

    scene = read_json(scene_path)
    qa_dir = ensure_dir(build_dir / "qa")
    progress_dir = ensure_dir(qa_dir / "track-progress")
    ref_dir = ensure_dir(qa_dir / "reference")
    html_dir = ensure_dir(progress_dir / "html")
    samples = [
        sample
        for sample in _sample_plan(scene)
        if sample.get("kind") == "transition"
        and (slides is None or int(sample.get("from", 0) or 0) in slides)
        and _progress_filter_matches(float(sample.get("progress", 0.0) or 0.0), progresses)
    ]
    if not samples:
        raise PresenterError("No transition samples matched the requested slide/progress filter.")

    transitions_by_pair = {
        (int(transition.get("from", 0) or 0), int(transition.get("to", 0) or 0)): transition
        for transition in scene.get("transitions", [])
    }
    slides_by_index = {
        int(slide.get("index", 0) or 0): slide
        for slide in scene.get("slides", [])
    }
    tracks_by_sample: dict[str, list[str]] = {}
    for sample in samples:
        from_slide = int(sample.get("from", 0) or 0)
        to_slide = int(sample.get("to", 0) or 0)
        transition = transitions_by_pair.get((from_slide, to_slide), {})
        if tracks is not None:
            selected_tracks = sorted(tracks)
        else:
            selected_tracks = _track_progress_default_tracks(
                slides_by_index.get(from_slide, {}),
                slides_by_index.get(to_slide, {}),
                transition,
            )
        selected_tracks = [
            track_id
            for track_id in selected_tracks
            if _track_exists_in_transition(track_id, slides_by_index.get(from_slide, {}), slides_by_index.get(to_slide, {}), transition)
        ]
        if selected_tracks:
            tracks_by_sample[str(sample["id"])] = selected_tracks
    if not tracks_by_sample:
        raise PresenterError("No eligible tracks found for track progress calibration.")

    candidates = _morph_progress_candidates(candidate_step)
    candidate_samples = _track_progress_candidate_samples(samples, tracks_by_sample, transitions_by_pair, candidates)
    samples_path = progress_dir / "candidate-samples.json"
    write_json(samples_path, candidate_samples)
    if not (reuse_html and _frames_exist(html_dir, candidate_samples)):
        _capture_html_frames(node, build_dir, samples_path, html_dir, playwright_dir)

    for sample in samples:
        ref_path = ref_dir / f"{sample['id']}.png"
        if not ref_path.exists():
            _extract_reference_frame(ffmpeg, reference, ref_path, float(sample["referenceSec"]))

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in candidate_samples:
        source_sample_id = str(sample.get("sourceSampleId") or sample.get("id"))
        sweep = sample.get("candidateSweep") or {}
        track_id = str(sweep.get("trackId") or "")
        if not track_id:
            continue
        ref_path = ref_dir / f"{source_sample_id}.png"
        html_path = html_dir / f"{sample['id']}.png"
        if not ref_path.exists() or not html_path.exists():
            continue
        with Image.open(ref_path).convert("RGB") as ref_img, Image.open(html_path).convert("RGB") as html_img:
            if html_img.size != ref_img.size:
                html_img = html_img.resize(ref_img.size)
            ref_arr = np.asarray(ref_img, dtype=np.float32)
            html_arr = np.asarray(html_img, dtype=np.float32)
        score = _global_ssim(ref_arr, html_arr)
        source_sample = next(item for item in samples if str(item["id"]) == source_sample_id)
        from_slide = int(source_sample.get("from", 0) or 0)
        to_slide = int(source_sample.get("to", 0) or 0)
        transition = transitions_by_pair.get((from_slide, to_slide), {})
        baseline_value = _track_progress_current_value(
            float(source_sample.get("progress", 0.0) or 0.0),
            track_id,
            transition,
        )
        key = (source_sample_id, track_id)
        row = rows_by_key.setdefault(
            key,
            {
                "sampleId": source_sample_id,
                "from": from_slide,
                "to": to_slide,
                "progress": round(float(source_sample.get("progress", 0.0) or 0.0), 4),
                "trackId": track_id,
                "baselineProgressValue": round(float(baseline_value), 4),
                "baselineScore": None,
                "baselinePoints": _track_progress_baseline_points(track_id, transition),
                "candidateRows": [],
            },
        )
        value = round(float(sweep.get("value", 0.0) or 0.0), 4)
        candidate_row = {"value": value, "score": round(float(score), 6)}
        row["candidateRows"].append(candidate_row)
        if abs(value - baseline_value) <= 0.0001:
            row["baselineScore"] = candidate_row["score"]

    rows: list[dict[str, Any]] = []
    for row in rows_by_key.values():
        candidates_for_row = sorted(
            row["candidateRows"],
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        best = candidates_for_row[0] if candidates_for_row else None
        if row.get("baselineScore") is None:
            baseline = min(
                candidates_for_row,
                key=lambda item: abs(float(item["value"]) - float(row["baselineProgressValue"])),
                default=None,
            )
            row["baselineScore"] = baseline["score"] if baseline else None
        row["bestProgressValue"] = best["value"] if best else None
        row["score"] = best["score"] if best else None
        row["improvement"] = (
            round(float(best["score"]) - float(row["baselineScore"]), 6)
            if best and row.get("baselineScore") is not None
            else None
        )
        row["topCandidates"] = candidates_for_row[:5]
        rows.append(row)
    rows.sort(key=lambda item: (int(item["from"]), int(item["to"]), str(item["trackId"]), float(item["progress"])))

    config_overrides = _track_progress_config_overrides(
        rows,
        min_score=min_score,
        min_improvement=min_improvement,
        stability_weight=stability_weight,
    )
    report = {
        "schema": "pptx-html-presenter.track-progress.v1",
        "generatedAtUtc": utc_now_iso(),
        "reference": str(reference),
        "candidateStep": candidate_step,
        "minScore": min_score,
        "minImprovement": min_improvement,
        "stabilityWeight": stability_weight,
        "tracks": sorted(tracks) if tracks is not None else None,
        "rows": rows,
        "configOverrides": config_overrides,
        "summary": {
            "sampleCount": len(rows),
            "overrideCount": len(config_overrides),
            "candidateCount": len(candidate_samples),
            "slides": sorted({int(row["from"]) for row in rows}),
        },
    }
    write_json(progress_dir / "report.json", report)
    return report


def _frames_exist(frame_dir: Path, samples: list[dict[str, Any]]) -> bool:
    return all((frame_dir / f"{sample['id']}.png").exists() for sample in samples)


def _candidate_sweep_base_sample(
    build_dir: Path,
    scene: dict[str, Any],
    sample_id: str,
) -> dict[str, Any]:
    qa_samples_path = build_dir / "qa" / "samples.json"
    samples = read_json(qa_samples_path) if qa_samples_path.exists() else _sample_plan(scene)
    for sample in samples:
        if str(sample.get("id")) == sample_id:
            return copy.deepcopy(sample)
    raise PresenterError(f"QA sample not found: {sample_id}")


def _normalize_candidate_sweep_vary(vary: str) -> str:
    normalized = vary.strip().lower().replace("_", "-")
    aliases = {
        "morph": "progress",
        "morph-progress": "progress",
        "object-progress": "track-progress",
        "per-track-progress": "track-progress",
        "media-phase": "phase",
        "media-clock": "phase",
        "phase-delta": "phase-offset",
        "media-phase-offset": "phase-offset",
        "media-clock-offset": "phase-offset",
        "clock-offset": "phase-offset",
        "fade-enter-end": "enter-fade-end",
        "unmatched-enter-end": "enter-fade-end",
        "unmatched-fade-enter-end": "enter-fade-end",
        "fade-exit-end": "exit-fade-end",
        "unmatched-exit-end": "exit-fade-end",
        "unmatched-fade-exit-end": "exit-fade-end",
        "clock": "phase",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"progress", "phase", "phase-offset", "track-progress", "enter-fade-end", "exit-fade-end"}:
        raise PresenterError(
            "Candidate sweep --vary must be progress, track-progress, phase, phase-offset, "
            "enter-fade-end, or exit-fade-end."
        )
    return normalized


def _candidate_sweep_samples(
    base_sample: dict[str, Any],
    vary: str,
    values: list[float],
    track_id: str | None = None,
) -> list[dict[str, Any]]:
    normalized = _normalize_candidate_sweep_vary(vary)
    if normalized == "progress" and base_sample.get("kind") != "transition":
        raise PresenterError("Progress sweeps require a transition sample.")
    if normalized == "track-progress" and base_sample.get("kind") != "transition":
        raise PresenterError("Track-progress sweeps require a transition sample.")
    if normalized in {"enter-fade-end", "exit-fade-end"} and base_sample.get("kind") != "transition":
        raise PresenterError(f"{normalized} sweeps require a transition sample.")
    if normalized in {"phase", "track-progress"} and not track_id:
        raise PresenterError(f"{normalized} sweeps require --track-id.")
    if normalized == "phase" and track_id not in (base_sample.get("mediaClocks") or {}):
        raise PresenterError(f"Track {track_id} is not present in sample mediaClocks.")
    if normalized == "phase-offset":
        missing = [
            candidate
            for candidate in _candidate_sweep_track_ids(base_sample, track_id)
            if candidate not in (base_sample.get("mediaClocks") or {})
        ]
        if missing:
            raise PresenterError(f"Tracks are not present in sample mediaClocks: {', '.join(missing)}")

    out: list[dict[str, Any]] = []
    for value in values:
        candidate = copy.deepcopy(base_sample)
        numeric = float(value)
        candidate["id"] = _candidate_sweep_candidate_id(str(base_sample["id"]), normalized, numeric, track_id)
        candidate["sourceSampleId"] = base_sample["id"]
        candidate["candidateSweep"] = {
            "vary": normalized,
            "trackId": track_id,
            "value": round(numeric, 6),
        }
        if normalized == "progress":
            candidate["progress"] = round(_clamp01(numeric), 4)
        elif normalized == "phase":
            clocks = dict(candidate.get("mediaClocks") or {})
            clocks[str(track_id)] = round(numeric, 3)
            candidate["mediaClocks"] = clocks
        elif normalized == "phase-offset":
            clocks = dict(candidate.get("mediaClocks") or {})
            target_tracks = _candidate_sweep_track_ids(base_sample, track_id)
            for target_track in target_tracks:
                clocks[target_track] = round(float(clocks[target_track]) + numeric, 3)
            candidate["mediaClocks"] = clocks
            candidate["candidateSweep"]["trackIds"] = target_tracks
        elif normalized == "enter-fade-end":
            candidate["unmatchedFadeOverride"] = {"enterStart": 0.0, "enterEnd": round(_clamp01(numeric), 4)}
        elif normalized == "exit-fade-end":
            candidate["unmatchedFadeOverride"] = {"exitStart": 0.0, "exitEnd": round(_clamp01(numeric), 4)}
        else:
            candidate["trackProgressOverrides"] = {str(track_id): round(_clamp01(numeric), 4)}
        out.append(candidate)
    return out


def _candidate_sweep_track_ids(base_sample: dict[str, Any], track_id: str | None) -> list[str]:
    clocks = base_sample.get("mediaClocks") or {}
    if not track_id or str(track_id).strip().lower() == "all":
        return sorted(str(track) for track in clocks)
    return [part.strip() for part in str(track_id).split(",") if part.strip()]


def _track_progress_candidate_samples(
    samples: list[dict[str, Any]],
    tracks_by_sample: dict[str, list[str]],
    transitions_by_pair: dict[tuple[int, int], dict[str, Any]],
    candidates: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["id"])
        transition = transitions_by_pair.get(
            (
                int(sample.get("from", 0) or 0),
                int(sample.get("to", 0) or 0),
            ),
            {},
        )
        for track_id in tracks_by_sample.get(sample_id, []):
            baseline = _track_progress_current_value(
                float(sample.get("progress", 0.0) or 0.0),
                track_id,
                transition,
            )
            values = sorted({round(_clamp01(float(value)), 4) for value in [*candidates, baseline]})
            for value in values:
                candidate = copy.deepcopy(sample)
                candidate["id"] = _candidate_sweep_candidate_id(sample_id, "track-progress", value, track_id)
                candidate["sourceSampleId"] = sample_id
                candidate["trackProgressOverrides"] = {track_id: value}
                candidate["candidateSweep"] = {
                    "vary": "track-progress",
                    "trackId": track_id,
                    "value": value,
                    "baselineValue": round(float(baseline), 4),
                }
                out.append(candidate)
    return out


def _track_progress_default_tracks(
    from_slide: dict[str, Any],
    to_slide: dict[str, Any],
    transition: dict[str, Any],
) -> list[str]:
    return _morph_progress_anchor_tracks(from_slide, to_slide, transition, max_tracks=4)


def _track_exists_in_transition(
    track_id: str,
    from_slide: dict[str, Any],
    to_slide: dict[str, Any],
    transition: dict[str, Any],
) -> bool:
    if any(str(obj.get("trackId") or "") == track_id for obj in from_slide.get("objects", []) or []):
        return True
    if any(str(obj.get("trackId") or "") == track_id for obj in to_slide.get("objects", []) or []):
        return True
    if any(str(row.get("trackId") or "") == track_id for row in transition.get("inferredMotions", []) or []):
        return True
    return False


def _progress_filter_matches(progress: float, allowed: set[float] | None) -> bool:
    if allowed is None:
        return True
    rounded = round(float(progress), 4)
    return any(abs(rounded - round(float(value), 4)) <= 0.0001 for value in allowed)


def _track_progress_current_value(
    progress: float,
    track_id: str,
    transition: dict[str, Any],
) -> float:
    for row in transition.get("trackProgressOverrides", []) or []:
        if str(row.get("trackId") or "") != str(track_id):
            continue
        value = _progress_map_value(progress, row.get("points"))
        if value is not None:
            return _clamp01(float(value))
    value = _progress_map_value(progress, transition.get("progressMap"))
    if value is not None:
        return _clamp01(float(value))
    return _clamp01(float(progress))


def _track_progress_baseline_points(
    track_id: str,
    transition: dict[str, Any],
) -> list[dict[str, float]]:
    for row in transition.get("trackProgressOverrides", []) or []:
        if str(row.get("trackId") or "") != str(track_id):
            continue
        points = _normalized_progress_points(row.get("points"))
        if points:
            return points
    points = _normalized_progress_points(transition.get("progressMap"))
    if points:
        return points
    return [{"progress": 0.0, "value": 0.0}, {"progress": 1.0, "value": 1.0}]


def _normalized_progress_points(points: Any) -> list[dict[str, float]]:
    if not isinstance(points, list):
        return []
    out: list[dict[str, float]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            progress = _clamp01(float(_row_value(point, "progress", "raw")))
            value = _clamp01(
                float(
                    _row_value(
                        point,
                        "value",
                        "mappedProgress",
                        "mapped_progress",
                        "interpolationProgress",
                        "interpolation_progress",
                    )
                )
            )
        except (TypeError, ValueError):
            continue
        out.append({"progress": round(progress, 4), "value": round(value, 4)})
    by_progress: dict[float, dict[str, float]] = {}
    for point in out:
        by_progress[point["progress"]] = point
    return [by_progress[key] for key in sorted(by_progress)]


def _track_progress_config_overrides(
    rows: list[dict[str, Any]],
    *,
    min_score: float,
    min_improvement: float,
    stability_weight: float = 0.02,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        try:
            key = (
                int(row["from"]),
                int(row["to"]),
                str(row["trackId"]),
            )
        except (TypeError, ValueError, KeyError):
            continue
        grouped.setdefault(key, []).append(row)

    overrides: list[dict[str, Any]] = []
    for (from_slide, to_slide, track_id), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: float(row.get("progress", 0.0) or 0.0))
        selected = _track_progress_monotonic_selection(group, stability_weight=stability_weight)
        if not selected:
            continue
        baseline_total = sum(float(row.get("baselineScore") or 0.0) for row, _choice in selected)
        selected_total = sum(float(choice.get("score") or 0.0) for _row, choice in selected)
        improved_count = sum(
            1
            for row, choice in selected
            if float(choice.get("score") or 0.0) >= min_score
            and float(choice.get("score") or 0.0) > float(row.get("baselineScore") or 0.0) + min_improvement
        )
        if improved_count <= 0 or selected_total <= baseline_total + min_improvement:
            continue
        points_by_progress: dict[float, float] = {}
        baseline_points = group[0].get("baselinePoints") or []
        for point in baseline_points:
            try:
                points_by_progress[round(_clamp01(float(point["progress"])), 4)] = round(_clamp01(float(point["value"])), 4)
            except (TypeError, ValueError, KeyError):
                continue
        points_by_progress.setdefault(0.0, 0.0)
        points_by_progress.setdefault(1.0, 1.0)
        for row, choice in selected:
            progress = round(_clamp01(float(row.get("progress", 0.0) or 0.0)), 4)
            if progress in (0.0, 1.0):
                continue
            points_by_progress[progress] = round(_clamp01(float(choice.get("value", progress) or 0.0)), 4)
        points = _monotonic_progress_points(points_by_progress)
        overrides.append(
            {
                "from": from_slide,
                "to": to_slide,
                "track_id": track_id,
                "points": points,
                "sample_count": len(group),
                "improved_count": improved_count,
                "baseline_score": round(baseline_total / max(1, len(selected)), 6),
                "score": round(selected_total / max(1, len(selected)), 6),
                "source": "track-progress",
            }
        )
    return overrides


def _track_progress_monotonic_selection(
    rows: list[dict[str, Any]],
    *,
    stability_weight: float = 0.02,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not rows:
        return []
    options_by_row: list[list[dict[str, Any]]] = []
    for row in rows:
        options = []
        seen: set[float] = set()
        for candidate in row.get("candidateRows", []) or []:
            try:
                value = round(_clamp01(float(candidate["value"])), 4)
                score = float(candidate["score"])
            except (TypeError, ValueError, KeyError):
                continue
            if value in seen:
                continue
            seen.add(value)
            baseline_value = float(row.get("baselineProgressValue", value) or value)
            selection_score = score - (max(0.0, stability_weight) * abs(value - baseline_value))
            options.append({"value": value, "score": score, "selectionScore": selection_score})
        if not options:
            return []
        options.sort(key=lambda item: float(item["value"]))
        options_by_row.append(options)

    states: list[list[dict[str, Any]]] = []
    first_states = [
        {"score": float(option["selectionScore"]), "prev": None, "choice": option}
        for option in options_by_row[0]
    ]
    states.append(first_states)
    for row_index in range(1, len(options_by_row)):
        row_states: list[dict[str, Any]] = []
        previous_states = states[row_index - 1]
        for option in options_by_row[row_index]:
            best_prev_index = None
            best_prev_score = None
            for previous_index, previous in enumerate(previous_states):
                if float(previous["choice"]["value"]) > float(option["value"]) + 0.0001:
                    continue
                previous_score = float(previous["score"])
                if best_prev_score is None or previous_score > best_prev_score:
                    best_prev_score = previous_score
                    best_prev_index = previous_index
            if best_prev_index is None:
                row_states.append({"score": float("-inf"), "prev": None, "choice": option})
            else:
                row_states.append(
                    {
                        "score": float(best_prev_score) + float(option["selectionScore"]),
                        "prev": best_prev_index,
                        "choice": option,
                    }
                )
        states.append(row_states)
    final_index, final_state = max(
        enumerate(states[-1]),
        key=lambda item: float(item[1]["score"]),
    )
    if float(final_state["score"]) == float("-inf"):
        return []
    choices: list[dict[str, Any]] = []
    index = final_index
    for row_index in range(len(rows) - 1, -1, -1):
        state = states[row_index][index]
        choices.append(state["choice"])
        index = state["prev"] if state["prev"] is not None else 0
    choices.reverse()
    return list(zip(rows, choices))


def _monotonic_progress_points(points_by_progress: dict[float, float]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    previous = 0.0
    for progress in sorted(points_by_progress):
        progress = round(_clamp01(float(progress)), 4)
        value = _clamp01(float(points_by_progress[progress]))
        if progress == 0.0:
            value = 0.0
        elif progress == 1.0:
            value = 1.0
        else:
            value = max(previous, value)
        points.append({"progress": progress, "value": round(value, 4)})
        previous = value
    if not points or points[0]["progress"] != 0.0:
        points.insert(0, {"progress": 0.0, "value": 0.0})
    if points[-1]["progress"] != 1.0:
        points.append({"progress": 1.0, "value": 1.0})
    return points


def _candidate_sweep_candidate_id(
    sample_id: str,
    vary: str,
    value: float,
    track_id: str | None = None,
) -> str:
    track_part = f"-{_safe_slug(track_id)}" if track_id else ""
    return f"{_safe_slug(sample_id)}-{_safe_slug(vary)}{track_part}-{_candidate_value_label(value)}"


def _candidate_sweep_dir_name(sample_id: str, vary: str, track_id: str | None) -> str:
    track_part = f"-{_safe_slug(track_id)}" if track_id else ""
    return f"{_safe_slug(sample_id)}-{_safe_slug(vary)}{track_part}"


def _candidate_value_label(value: float) -> str:
    rounded = round(float(value), 6)
    sign = "m" if rounded < 0 else "p"
    raw = f"{abs(rounded):.6f}".rstrip("0").rstrip(".")
    return f"{sign}{raw.replace('.', 'p') or '0'}"


def _safe_slug(value: str | None) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip())
    return slug.strip("-") or "value"


def _candidate_sweep_reference_frame(
    build_dir: Path,
    qa_dir: Path,
    base_sample: dict[str, Any],
    *,
    reference_frame: Path | None,
    reference_mp4: Path | None,
    ffmpeg_bin: str | None,
) -> Path:
    if reference_frame is not None:
        path = reference_frame.expanduser().resolve()
        if not path.exists():
            raise PresenterError(f"Reference frame not found: {path}")
        return path

    ref_dir = ensure_dir(qa_dir / "reference")
    path = ref_dir / f"{base_sample['id']}.png"
    if path.exists():
        return path
    if reference_mp4 is not None:
        reference = reference_mp4.expanduser().resolve()
        if not reference.exists():
            raise PresenterError(f"Reference MP4 not found: {reference}")
        ffmpeg = find_binary("ffmpeg.exe", ffmpeg_bin) or find_binary("ffmpeg", ffmpeg_bin)
        if ffmpeg is None:
            raise PresenterError("ffmpeg missing for candidate sweep reference extraction")
        _extract_reference_frame(ffmpeg, reference, path, float(base_sample["referenceSec"]))
        return path
    raise PresenterError(
        "Reference frame missing. Run qa first, pass --reference-frame, or pass --reference with ffmpeg."
    )


def _path_for_report(path: Path, build_dir: Path) -> str:
    try:
        return path.relative_to(build_dir).as_posix()
    except ValueError:
        return str(path)


def _sample_plan(
    scene: dict[str, Any],
    *,
    slide_hold_sec: float | None = None,
    settled_offset_sec: float | None = None,
    transition_reference_lead_fraction: float | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    timeline = 0.0
    qa = scene.get("qa", {}) or {}
    assets = {asset.get("id"): asset for asset in scene.get("assets", [])}
    slides = scene.get("slides", [])
    slides_by_index = {int(slide["index"]): slide for slide in slides}
    active_media: dict[str, dict[str, Any]] = {}
    slide_timed_video_phase = float(qa.get("slideTimedVideoPhaseSec", 0.0) or 0.0)
    transition_time_overrides = _transition_time_override_map(
        qa.get("transitionTimeOverrides", []) or []
    )

    def render_key(obj: dict[str, Any]) -> tuple[str, str]:
        asset_id = str(obj.get("assetId") or "")
        if is_paused_media(obj) and obj.get("posterAssetId"):
            poster_id = str(obj.get("posterAssetId") or "")
            poster_asset = assets.get(poster_id, {}) or {}
            if poster_asset.get("file"):
                return poster_id, "image"
        asset = assets.get(asset_id, {}) or {}
        render_kind = "video" if asset.get("kind") == "video" else str(obj.get("kind") or "")
        return asset_id, render_kind

    def transition_render_object(
        from_obj: dict[str, Any] | None, to_obj: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if from_obj is None or to_obj is None:
            return to_obj or from_obj
        from_asset = assets.get(str(from_obj.get("assetId") or ""), {}) or {}
        to_asset = assets.get(str(to_obj.get("assetId") or ""), {}) or {}
        same_asset = bool(from_obj.get("assetId")) and from_obj.get("assetId") == to_obj.get("assetId")
        if from_asset.get("kind") == "video" and same_asset:
            if (
                not is_paused_media(to_obj)
                and not is_visible(from_obj)
                and has_explicit_media_phase(to_obj)
                and not is_animated_loop_asset(from_asset)
            ):
                return to_obj
            if (
                is_paused_media(from_obj)
                and not is_paused_media(to_obj)
                and not is_visible(from_obj)
                and not is_animated_loop_asset(from_asset)
            ):
                return to_obj
            if not is_paused_media(to_obj) or is_visible(from_obj) or is_animated_loop_asset(from_asset):
                return from_obj
        if from_asset.get("kind") == "video" and (
            is_visible(from_obj)
            or to_asset.get("kind") != "video"
        ):
            return from_obj
        if to_asset.get("kind") == "video":
            return to_obj
        return to_obj

    def is_visible(obj: dict[str, Any] | None) -> bool:
        if obj is None:
            return False
        geometry = obj.get("geometry") or {}
        left = float(geometry.get("leftPct", 0.0))
        top = float(geometry.get("topPct", 0.0))
        width = float(geometry.get("widthPct", 1.0))
        height = float(geometry.get("heightPct", 1.0))
        opacity = float(obj.get("opacity", 1.0) or 0.0)
        return opacity > 0.01 and left < 1.0 and top < 1.0 and (left + width) > 0.0 and (top + height) > 0.0

    def is_slide_timed_video(obj: dict[str, Any], render_kind: str) -> bool:
        return render_kind == "video" and (obj.get("mediaTiming") or {}).get("kind") == "playFrom"

    def is_paused_media(obj: dict[str, Any]) -> bool:
        return bool((obj.get("mediaTiming") or {}).get("paused"))

    def has_explicit_media_phase(obj: dict[str, Any]) -> bool:
        try:
            float((obj.get("mediaTiming") or {}).get("phaseSec"))
        except (TypeError, ValueError):
            return False
        return True

    def is_animated_loop_asset(asset: dict[str, Any]) -> bool:
        source = f"{asset.get('sourceFile', '')} {asset.get('sourcePath', '')} {asset.get('extension', '')}".lower()
        return bool(asset.get("animated")) or ".gif" in source

    def media_offset(obj: dict[str, Any], transition: dict[str, Any] | None = None) -> float:
        timing = obj.get("mediaTiming") or {}
        offset = float(timing.get("startSec", 0.0) or 0.0)
        if timing.get("kind") == "playFrom":
            offset += slide_timed_video_phase
        phase_override = _transition_media_phase_override(obj, transition)
        offset += float(phase_override if phase_override is not None else timing.get("phaseSec", 0.0) or 0.0)
        return offset

    def sync_objects(objects: list[dict[str, Any]], event_sec: float) -> None:
        nonlocal active_media
        next_media: dict[str, dict[str, Any]] = {}
        for obj in objects:
            track_id = str(obj.get("trackId") or "")
            if not track_id:
                continue
            asset_id, render_kind = render_key(obj)
            prior = active_media.get(track_id)
            slide_timed = is_slide_timed_video(obj, render_kind)
            if prior and prior["assetId"] == asset_id and prior["renderKind"] == render_kind and not slide_timed:
                next_media[track_id] = prior
                continue
            next_media[track_id] = {
                "assetId": asset_id,
                "renderKind": render_kind,
                "mediaOffset": media_offset(obj),
                "startedAt": event_sec if render_kind == "video" and not is_paused_media(obj) else None,
            }
        active_media = next_media

    def sync_transition(from_slide: dict[str, Any], to_slide: dict[str, Any], event_sec: float) -> None:
        nonlocal active_media
        from_by_track = {str(obj.get("trackId") or ""): obj for obj in from_slide.get("objects", []) if obj.get("trackId")}
        to_by_track = {str(obj.get("trackId") or ""): obj for obj in to_slide.get("objects", []) if obj.get("trackId")}
        next_media: dict[str, dict[str, Any]] = {}
        for track_id in sorted(set(from_by_track) | set(to_by_track)):
            to_obj = to_by_track.get(track_id)
            from_obj = from_by_track.get(track_id)
            obj = transition_render_object(from_obj, to_obj)
            if obj is None:
                continue
            asset_id, render_kind = render_key(obj)
            if render_kind != "video":
                next_media[track_id] = {
                    "assetId": asset_id,
                    "renderKind": render_kind,
                    "mediaOffset": media_offset(obj, transition),
                    "startedAt": None,
                }
                continue
            if obj is to_obj and to_obj is not None and is_slide_timed_video(to_obj, render_kind):
                next_media[track_id] = {
                    "assetId": asset_id,
                    "renderKind": render_kind,
                    "mediaOffset": media_offset(to_obj, transition),
                    "startedAt": None if is_paused_media(to_obj) else event_sec,
                }
                continue
            prior = active_media.get(track_id)
            if prior and prior["assetId"] == asset_id and prior["renderKind"] == render_kind:
                next_media[track_id] = prior
                continue
            next_media[track_id] = {
                "assetId": asset_id,
                "renderKind": render_kind,
                "mediaOffset": media_offset(obj, transition),
                "startedAt": None if is_paused_media(obj) else event_sec,
            }
        active_media = next_media

    def media_clocks_at(sample_sec: float) -> dict[str, float]:
        clocks: dict[str, float] = {}
        for track_id, state in active_media.items():
            if state.get("renderKind") != "video":
                continue
            offset = float(state.get("mediaOffset", 0.0) or 0.0)
            if state.get("startedAt") is None:
                clocks[track_id] = round(offset, 3)
            else:
                clocks[track_id] = round(offset + max(0.0, sample_sec - float(state["startedAt"])), 3)
        return clocks

    slide_hold = float(slide_hold_sec if slide_hold_sec is not None else qa.get("slideHoldSec", 1.0))
    settled_offset = float(
        settled_offset_sec if settled_offset_sec is not None else qa.get("settledOffsetSec", 0.12)
    )
    lead_fraction = float(
        transition_reference_lead_fraction
        if transition_reference_lead_fraction is not None
        else qa.get("transitionReferenceLeadFraction", 0.0)
    )
    lead_fraction = max(0.0, min(1.0, lead_fraction))
    transition_samples = tuple(float(v) for v in qa.get("transitionSamples", (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)))
    for slide in slides:
        slide_index = int(slide["index"])
        sync_objects(slide.get("objects", []), timeline)
        html_sec = round(timeline + min(settled_offset, slide_hold / 2.0), 3)
        samples.append(
            {
                "id": f"slide-{slide_index:03d}-settled",
                "kind": "slide",
                "slide": slide_index,
                "progress": 1.0,
                "referenceSec": html_sec,
                "mediaSec": html_sec,
                "mediaClocks": media_clocks_at(html_sec),
            }
        )
        transition = next(
            (t for t in scene.get("transitions", []) if int(t["from"]) == slide_index),
            None,
        )
        timeline += slide_hold
        if transition:
            duration = float(transition.get("durationSec", 0.0) or 0.0)
            to_index = int(transition["to"])
            timing_override = transition_time_overrides.get((slide_index, to_index), {})
            transition_lead = float(
                timing_override.get("referenceLeadFraction", lead_fraction)
                if timing_override.get("referenceLeadFraction") is not None
                else lead_fraction
            )
            transition_lead = max(0.0, min(1.0, transition_lead))
            progress_offsets = timing_override.get("progressOffsets", []) or []
            to_slide = slides_by_index.get(int(transition["to"]), {})
            sync_transition(slide, to_slide, timeline)
            for progress in transition_samples:
                progress_offset = _transition_progress_reference_offset(progress_offsets, progress)
                reference_offset = (
                    progress_offset
                    if progress_offset is not None
                    else float(timing_override.get("referenceOffsetSec", 0.0) or 0.0)
                )
                reference_start = timeline - (duration * transition_lead) + reference_offset
                html_sec = round(timeline + (duration * progress), 3)
                reference_sec = round(max(0.0, reference_start + (duration * progress)), 3)
                media_sec = reference_sec
                samples.append(
                    {
                        "id": f"trans-{slide_index:03d}-{int(transition['to']):03d}-{int(progress * 100):03d}",
                        "kind": "transition",
                        "from": slide_index,
                        "to": int(transition["to"]),
                        "progress": progress,
                        "referenceSec": reference_sec,
                        "mediaSec": media_sec,
                        "mediaClocks": media_clocks_at(media_sec),
                    }
                )
            timeline += duration
    return samples


def _transition_time_override_map(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        try:
            from_slide = int(_row_value(row, "from", "fromSlide", "from_slide"))
            to_slide = int(_row_value(row, "to", "toSlide", "to_slide"))
        except (TypeError, ValueError):
            continue
        normalized: dict[str, Any] = {}
        lead = _row_value(row, "reference_lead_fraction", "referenceLeadFraction")
        if lead is not None:
            try:
                normalized["referenceLeadFraction"] = float(lead)
            except (TypeError, ValueError):
                pass
        offset = _row_value(row, "reference_offset_sec", "referenceOffsetSec")
        if offset is not None:
            try:
                normalized["referenceOffsetSec"] = float(offset)
            except (TypeError, ValueError):
                pass
        progress_offsets = _row_value(row, "progress_offsets", "progressOffsets")
        if isinstance(progress_offsets, list):
            normalized["progressOffsets"] = [
                {
                    "progress": float(_row_value(item, "progress")),
                    "referenceOffsetSec": float(
                        _row_value(item, "reference_offset_sec", "referenceOffsetSec")
                    ),
                    **(
                        {"score": float(_row_value(item, "score"))}
                        if _row_value(item, "score") is not None
                        else {}
                    ),
                }
                for item in progress_offsets
                if _row_value(item, "progress") is not None
                and _row_value(item, "reference_offset_sec", "referenceOffsetSec") is not None
            ]
        out[(from_slide, to_slide)] = normalized
    return out


def _transition_progress_reference_offset(rows: list[dict[str, Any]], progress: float) -> float | None:
    if not rows:
        return None
    target = round(float(progress), 3)
    for row in rows:
        try:
            row_progress = round(float(_row_value(row, "progress")), 3)
            if row_progress == target:
                return float(_row_value(row, "referenceOffsetSec", "reference_offset_sec"))
        except (TypeError, ValueError):
            continue
    return None


def _row_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _extract_reference_frame(ffmpeg: Path, reference: Path, out: Path, seconds: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-ss",
            f"{seconds:.3f}",
            "-i",
            str(reference),
            "-frames:v",
            "1",
            str(out),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _node_has_playwright(node: Path, playwright_dir: Path | None = None) -> bool:
    try:
        env = None
        code = "require.resolve('playwright')"
        if playwright_dir is not None:
            import os

            env = dict(os.environ)
            env["PLAYWRIGHT_PACKAGE_DIR"] = str(playwright_dir.expanduser().resolve())
            package_path = playwright_dir.expanduser().resolve() / "node_modules" / "playwright" / "index.mjs"
            code = f"import({str(package_path.as_uri())!r})"
        result = subprocess.run(
            [str(node), "-e", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            env=env,
        )
        return result.returncode == 0
    except Exception:
        return False


def _capture_html_frames(
    node: Path,
    build_dir: Path,
    samples_path: Path,
    html_dir: Path,
    playwright_dir: Path | None,
) -> dict[str, Any]:
    script = Path(__file__).with_name("browser_capture.mjs")
    env = None
    if playwright_dir is not None:
        import os

        env = dict(os.environ)
        env["PLAYWRIGHT_PACKAGE_DIR"] = str(playwright_dir.expanduser().resolve())
    subprocess.run(
        [str(node), str(script), str(build_dir), str(samples_path), str(html_dir)],
        check=True,
        env=env,
    )
    return _read_capture_report(html_dir)


def _capture_html_frames_with_scene_override(
    node: Path,
    build_dir: Path,
    scene_path: Path,
    scene: dict[str, Any],
    samples_path: Path,
    html_dir: Path,
    playwright_dir: Path | None,
) -> None:
    original = scene_path.read_text(encoding="utf-8")
    try:
        write_json(scene_path, scene)
        _capture_html_frames(node, build_dir, samples_path, html_dir, playwright_dir)
    finally:
        scene_path.write_text(original, encoding="utf-8")


def _morph_progress_candidates(candidate_step: float) -> list[float]:
    step = max(0.01, min(0.5, float(candidate_step)))
    values = {0.0, 1.0}
    count = int(1.0 / step) + 1
    for index in range(count + 1):
        values.add(round(_clamp01(index * step), 4))
    return sorted(values)


def _morph_progress_candidate_samples(
    samples: list[dict[str, Any]],
    candidates: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample in samples:
        for candidate in candidates:
            row = dict(sample)
            row["id"] = _morph_progress_candidate_id(str(sample["id"]), candidate)
            row["progress"] = round(float(candidate), 4)
            row["sourceSampleId"] = sample["id"]
            out.append(row)
    return out


def _morph_progress_candidate_id(sample_id: str, candidate: float) -> str:
    return f"{sample_id}-candidate-{int(round(_clamp01(candidate) * 1000)):04d}"


def _neutralized_progress_scene(
    scene: dict[str, Any],
    pairs: set[tuple[int, int]],
) -> dict[str, Any]:
    out = copy.deepcopy(scene)
    identity = [{"progress": 0.0, "value": 0.0}, {"progress": 1.0, "value": 1.0}]
    for transition in out.get("transitions", []):
        key = (
            int(transition.get("from", 0) or 0),
            int(transition.get("to", 0) or 0),
        )
        if key not in pairs:
            continue
        transition["progressMap"] = identity
        transition["easing"] = "linear"
    return out


def _morph_progress_config_overrides(
    rows: list[dict[str, Any]],
    min_score: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        try:
            from_slide = int(row["from"])
            to_slide = int(row["to"])
        except (TypeError, ValueError, KeyError):
            continue
        progress = float(row.get("progress", 0.0) or 0.0)
        if progress not in (0.0, 1.0) and float(row.get("score", 0.0) or 0.0) < min_score:
            continue
        grouped.setdefault((from_slide, to_slide), []).append(row)

    overrides: list[dict[str, Any]] = []
    for (from_slide, to_slide), group in sorted(grouped.items()):
        by_progress: dict[float, list[dict[str, Any]]] = {}
        for row in group:
            progress = round(_clamp01(float(row.get("progress", 0.0) or 0.0)), 4)
            by_progress.setdefault(progress, []).append(row)
        by_progress.setdefault(0.0, [{"bestProgressValue": 0.0, "score": 1.0}])
        by_progress.setdefault(1.0, [{"bestProgressValue": 1.0, "score": 1.0}])
        points: list[dict[str, Any]] = []
        for progress in sorted(by_progress):
            values = sorted(
                _clamp01(float(row.get("bestProgressValue", progress) or 0.0))
                for row in by_progress[progress]
            )
            scores = sorted(float(row.get("score", 0.0) or 0.0) for row in by_progress[progress])
            value = 0.0 if progress == 0.0 else 1.0 if progress == 1.0 else _median(values)
            points.append(
                {
                    "progress": round(progress, 4),
                    "value": round(value, 4),
                    "score": round(_median(scores), 6),
                }
            )
        points = _monotonic_morph_points(points)
        if len(points) < 2:
            continue
        scores = sorted(float(row.get("score", 0.0) or 0.0) for row in group)
        anchor_tracks = sorted(
            {
                str(track)
                for row in group
                for track in (row.get("anchorTracks") or [])
                if str(track)
            }
        )
        overrides.append(
            {
                "from": from_slide,
                "to": to_slide,
                "points": [
                    {"progress": point["progress"], "value": point["value"]}
                    for point in points
                ],
                "sample_count": len(group),
                "median_score": round(_median(scores), 6) if scores else 0.0,
                "anchor_tracks": anchor_tracks,
                "source": "morph-progress",
            }
        )
    return overrides


def _monotonic_morph_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not points:
        return []
    normalized = sorted(points, key=lambda item: float(item["progress"]))
    previous = 0.0
    for point in normalized:
        progress = _clamp01(float(point["progress"]))
        if progress == 0.0:
            value = 0.0
        elif progress == 1.0:
            value = 1.0
        else:
            value = max(previous, _clamp01(float(point["value"])))
        point["progress"] = round(progress, 4)
        point["value"] = round(value, 4)
        previous = value
    normalized[-1]["value"] = 1.0 if normalized[-1]["progress"] == 1.0 else normalized[-1]["value"]
    return normalized


def _morph_progress_anchor_tracks(
    from_slide: dict[str, Any],
    to_slide: dict[str, Any],
    transition: dict[str, Any],
    *,
    max_tracks: int = 5,
) -> list[str]:
    from_by_id = {str(obj.get("id") or ""): obj for obj in from_slide.get("objects", [])}
    to_by_id = {str(obj.get("id") or ""): obj for obj in to_slide.get("objects", [])}
    from_by_track = {
        str(obj.get("trackId") or ""): obj
        for obj in from_slide.get("objects", [])
        if obj.get("trackId")
    }
    to_by_track = {
        str(obj.get("trackId") or ""): obj
        for obj in to_slide.get("objects", [])
        if obj.get("trackId")
    }
    candidates: list[tuple[float, str]] = []
    for match in transition.get("matches", []) or []:
        track_id = str(match.get("trackId") or "")
        if not track_id:
            continue
        left = from_by_id.get(str(match.get("fromObjectId") or "")) or from_by_track.get(track_id)
        right = to_by_id.get(str(match.get("toObjectId") or "")) or to_by_track.get(track_id)
        if left is None or right is None:
            continue
        score = _morph_progress_anchor_score(left, right)
        if score <= 0:
            continue
        candidates.append((score, track_id))
    for row in transition.get("inferredMotions", []) or []:
        track_id = str(row.get("panelTrackId") or row.get("trackId") or "")
        if not track_id:
            continue
        from_geometry = row.get("fromGeometry") or {}
        to_geometry = row.get("toGeometry") or {}
        move = _geometry_motion_score(from_geometry, to_geometry)
        area = max(_geometry_area(from_geometry), _geometry_area(to_geometry))
        if move <= 0.01 or area <= 0.01:
            continue
        candidates.append((move + area, track_id))
    if not candidates:
        return []
    by_track: dict[str, float] = {}
    for score, track_id in candidates:
        by_track[track_id] = max(score, by_track.get(track_id, 0.0))
    panel_tracks = [
        (score, track_id)
        for track_id, score in by_track.items()
        if score >= 1000.0
    ]
    selected = panel_tracks or [(score, track_id) for track_id, score in by_track.items()]
    selected.sort(reverse=True)
    return [track_id for _score, track_id in selected[:max_tracks]]


def _morph_progress_anchor_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_geometry = left.get("geometry") or {}
    right_geometry = right.get("geometry") or {}
    move = _geometry_motion_score(left_geometry, right_geometry)
    area = max(_geometry_area(left_geometry), _geometry_area(right_geometry))
    if move <= 0.005 and area <= 0.02:
        return 0.0
    panel_bonus = 1000.0 if _is_morph_progress_panel_anchor(left) or _is_morph_progress_panel_anchor(right) else 0.0
    return panel_bonus + (area * 100.0) + (move * 20.0)


def _is_morph_progress_panel_anchor(obj: dict[str, Any]) -> bool:
    geometry = obj.get("geometry") or {}
    width = float(geometry.get("widthPct", 0.0) or 0.0)
    height = float(geometry.get("heightPct", 0.0) or 0.0)
    if width < 0.25 or height < 0.25:
        return False
    kind = str(obj.get("kind") or "").lower()
    name = str(obj.get("name") or "").lower()
    shape = str(obj.get("shape") or "").lower()
    raster = obj.get("rasterFallback") or {}
    raster_label = f"{raster.get('source', '')} {raster.get('file', '')}".lower() if isinstance(raster, dict) else ""
    is_panel_skin = kind == "image" and (
        "panel" in raster_label
        or bool(isinstance(raster, dict) and raster.get("replaceTrackIds"))
    )
    is_rounded_shape = kind == "shape" and (
        "roundrect" in shape.replace(" ", "")
        or "rounded" in name
        or "rectangle: rounded" in name
    )
    return is_panel_skin or is_rounded_shape


def _morph_progress_crop_bbox(
    from_slide: dict[str, Any],
    to_slide: dict[str, Any],
    transition: dict[str, Any],
    anchor_tracks: list[str],
    size: tuple[int, int],
    *,
    padding_px: int = 32,
) -> tuple[int, int, int, int] | None:
    if not anchor_tracks:
        return None
    from_by_track = {
        str(obj.get("trackId") or ""): obj
        for obj in from_slide.get("objects", [])
        if obj.get("trackId")
    }
    to_by_track = {
        str(obj.get("trackId") or ""): obj
        for obj in to_slide.get("objects", [])
        if obj.get("trackId")
    }
    inferred_by_track = {
        str(row.get("trackId") or ""): row
        for row in transition.get("inferredMotions", []) or []
        if row.get("trackId")
    }
    boxes: list[tuple[int, int, int, int]] = []
    for track_id in anchor_tracks:
        from_obj = from_by_track.get(track_id)
        to_obj = to_by_track.get(track_id)
        from_geometry, to_geometry = _transition_geometry_pair(
            from_obj,
            to_obj,
            inferred_by_track.get(track_id),
        )
        for geometry in (from_geometry, to_geometry):
            box = _geometry_pixel_bbox(geometry or {}, size)
            if box is not None:
                boxes.append(box)
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return _expand_bbox((x0, y0, x1, y1), size, padding_px)


def _geometry_pixel_bbox(
    geometry: dict[str, Any],
    size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    width, height = size
    left = float(geometry.get("leftPct", 0.0) or 0.0) * width
    top = float(geometry.get("topPct", 0.0) or 0.0) * height
    right = left + (float(geometry.get("widthPct", 0.0) or 0.0) * width)
    bottom = top + (float(geometry.get("heightPct", 0.0) or 0.0) * height)
    x0 = max(0, min(width, int(round(left))))
    y0 = max(0, min(height, int(round(top))))
    x1 = max(0, min(width, int(round(right))))
    y1 = max(0, min(height, int(round(bottom))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _geometry_motion_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_cx = float(left.get("leftPct", 0.0) or 0.0) + (float(left.get("widthPct", 0.0) or 0.0) / 2.0)
    left_cy = float(left.get("topPct", 0.0) or 0.0) + (float(left.get("heightPct", 0.0) or 0.0) / 2.0)
    right_cx = float(right.get("leftPct", 0.0) or 0.0) + (float(right.get("widthPct", 0.0) or 0.0) / 2.0)
    right_cy = float(right.get("topPct", 0.0) or 0.0) + (float(right.get("heightPct", 0.0) or 0.0) / 2.0)
    return abs(right_cx - left_cx) + abs(right_cy - left_cy)


def _geometry_area(geometry: dict[str, Any]) -> float:
    return max(0.0, float(geometry.get("widthPct", 0.0) or 0.0)) * max(
        0.0,
        float(geometry.get("heightPct", 0.0) or 0.0),
    )


def _compare_frame_sets(build_dir: Path, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        import numpy as np
        from PIL import Image, ImageChops
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    diff_dir = ensure_dir(build_dir / "qa" / "diff")
    side_by_side_dir = ensure_dir(build_dir / "qa" / "side-by-side")
    for sample in samples:
        ref = build_dir / "qa" / "reference" / f"{sample['id']}.png"
        html = build_dir / "qa" / "html" / f"{sample['id']}.png"
        if not ref.exists() or not html.exists():
            continue
        with Image.open(ref).convert("RGB") as ref_img, Image.open(html).convert("RGB") as html_img:
            if html_img.size != ref_img.size:
                html_img = html_img.resize(ref_img.size)
            diff = ImageChops.difference(ref_img, html_img)
            diff_path = diff_dir / f"{sample['id']}.png"
            side_by_side_path = side_by_side_dir / f"{sample['id']}.png"
            diff.save(diff_path)
            side_by_side = Image.new("RGB", (ref_img.width * 3, ref_img.height))
            side_by_side.paste(ref_img, (0, 0))
            side_by_side.paste(html_img, (ref_img.width, 0))
            side_by_side.paste(diff, (ref_img.width * 2, 0))
            side_by_side.save(side_by_side_path)
            ref_arr = np.asarray(ref_img, dtype=np.float32)
            html_arr = np.asarray(html_img, dtype=np.float32)
            delta = np.abs(ref_arr - html_arr)
            mean_delta = float(delta.mean())
            p95_delta = float(np.percentile(delta, 95))
            ssim = _global_ssim(ref_arr, html_arr)
            threshold = 0.985 if sample["kind"] == "slide" else 0.965
            rows.append(
                {
                    "sampleId": sample["id"],
                    "kind": sample["kind"],
                    "ssim": round(ssim, 6),
                    "meanDelta": round(mean_delta, 3),
                    "p95Delta": round(p95_delta, 3),
                    "threshold": threshold,
                    "passed": ssim >= threshold,
                    "diffFile": diff_path.relative_to(build_dir).as_posix(),
                    "sideBySideFile": side_by_side_path.relative_to(build_dir).as_posix(),
                }
            )
    return rows


def run_static_fallback_generation(
    build_dir: Path,
    *,
    reference_mp4: Path,
    ffmpeg_bin: str | None = None,
    slides: set[int] | None = None,
    hole_padding_px: int = 2,
    settled_only: bool = True,
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise PresenterError(f"Static fallback generation needs Pillow: {exc}") from exc

    build_dir = build_dir.expanduser().resolve()
    scene_path = build_dir / "deck.scene.json"
    if not scene_path.exists():
        raise PresenterError(f"Missing deck.scene.json in {build_dir}")
    reference = reference_mp4.expanduser().resolve()
    if not reference.exists():
        raise PresenterError(f"Reference MP4 not found: {reference}")
    ffmpeg = find_binary("ffmpeg.exe", ffmpeg_bin) or find_binary("ffmpeg", ffmpeg_bin)
    if ffmpeg is None:
        raise PresenterError("ffmpeg missing")

    scene = read_json(scene_path)
    qa_dir = ensure_dir(build_dir / "qa")
    ref_dir = ensure_dir(qa_dir / "reference")
    fallback_dir = ensure_dir(build_dir / "assets" / "fallback")
    samples = _sample_plan(scene)
    slide_samples = {
        int(sample["slide"]): sample
        for sample in samples
        if sample.get("kind") == "slide" and "slide" in sample
    }
    assets = {str(asset.get("id")): asset for asset in scene.get("assets", [])}
    rows: list[dict[str, Any]] = []
    overrides: list[dict[str, Any]] = []
    for slide in scene.get("slides", []):
        slide_index = int(slide.get("index", 0) or 0)
        if slides is not None and slide_index not in slides:
            continue
        sample = slide_samples.get(slide_index)
        if sample is None:
            continue
        ref_path = ref_dir / f"{sample['id']}.png"
        if not ref_path.exists():
            _extract_reference_frame(ffmpeg, reference, ref_path, float(sample["referenceSec"]))
        with Image.open(ref_path).convert("RGBA") as image:
            draw = ImageDraw.Draw(image)
            holes: list[dict[str, Any]] = []
            for obj in slide.get("objects", []):
                if not _is_live_media_hole(obj, assets):
                    continue
                bbox = _object_pixel_bbox(obj, image.size)
                if bbox is None:
                    continue
                x0, y0, x1, y1 = _expand_bbox(bbox, image.size, int(hole_padding_px))
                draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 0))
                holes.append(
                    {
                        "objectId": obj.get("id"),
                        "trackId": obj.get("trackId"),
                        "assetId": obj.get("assetId"),
                        "bbox": [x0, y0, x1, y1],
                    }
                )
            data_io = BytesIO()
            image.save(data_io, format="PNG", optimize=True)
            data = data_io.getvalue()
        digest = sha256(data).hexdigest()
        fallback_path = fallback_dir / f"static-slide-{slide_index:03d}-{digest[:16]}.png"
        fallback_path.write_bytes(data)
        rel_file = fallback_path.relative_to(build_dir).as_posix()
        override = {
            "slide": slide_index,
            "file": rel_file,
            "object_id": f"s{slide_index}-static-fallback-{digest[:8]}",
            "track_id": f"track-static-fallback-{slide_index:03d}",
            "name": f"PowerPoint static fallback {slide_index}",
            "settled_only": settled_only,
            "z": _max_slide_z(slide) + 1000,
            "source": "static-fallback",
        }
        overrides.append(override)
        rows.append(
            {
                "slide": slide_index,
                "sampleId": sample["id"],
                "file": rel_file,
                "bytes": len(data),
                "sha256": digest,
                "holeCount": len(holes),
                "holes": holes,
                "configOverride": override,
            }
        )

    report = {
        "schema": "pptx-html-presenter.static-fallback.v1",
        "generatedAtUtc": utc_now_iso(),
        "reference": str(reference),
        "holePaddingPx": int(hole_padding_px),
        "settledOnly": bool(settled_only),
        "rows": rows,
        "configOverrides": overrides,
        "summary": {
            "count": len(rows),
            "slides": sorted({row["slide"] for row in rows}),
            "totalBytes": sum(int(row["bytes"]) for row in rows),
        },
    }
    write_json(qa_dir / "static-fallback-report.json", report)
    return report


def _is_live_media_hole(obj: dict[str, Any], assets: dict[str, dict[str, Any]]) -> bool:
    asset = assets.get(str(obj.get("assetId") or ""))
    if not asset or asset.get("kind") != "video":
        return False
    timing = obj.get("mediaTiming") or {}
    if timing.get("paused") and obj.get("posterAssetId"):
        return False
    return True


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    size: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    width, height = size
    x0, y0, x1, y1 = bbox
    pad = max(0, int(padding))
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(width, x1 + pad),
        min(height, y1 + pad),
    )


def _max_slide_z(slide: dict[str, Any]) -> int:
    values = []
    for obj in slide.get("objects", []):
        try:
            values.append(int(float(obj.get("z", 0) or 0)))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0


def run_media_phase_calibration(
    build_dir: Path,
    *,
    reference_mp4: Path,
    ffmpeg_bin: str | None = None,
    slides: set[int] | None = None,
    step_sec: float = 0.5,
    search_sec: float = 12.0,
    min_score: float = 0.70,
    include_transitions: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        raise PresenterError(f"Media phase calibration needs Pillow and NumPy: {exc}") from exc

    build_dir = build_dir.expanduser().resolve()
    scene_path = build_dir / "deck.scene.json"
    if not scene_path.exists():
        raise PresenterError(f"Missing deck.scene.json in {build_dir}")
    reference = reference_mp4.expanduser().resolve()
    if not reference.exists():
        raise PresenterError(f"Reference MP4 not found: {reference}")
    ffmpeg = find_binary("ffmpeg.exe", ffmpeg_bin) or find_binary("ffmpeg", ffmpeg_bin)
    if ffmpeg is None:
        raise PresenterError("ffmpeg missing")

    scene = read_json(scene_path)
    qa_dir = ensure_dir(build_dir / "qa")
    ref_dir = ensure_dir(qa_dir / "reference")
    samples = _sample_plan(scene)
    slide_samples = {
        int(sample["slide"]): sample
        for sample in samples
        if sample.get("kind") == "slide" and "slide" in sample
    }
    assets = {str(asset.get("id")): asset for asset in scene.get("assets", [])}
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="pptx_html_presenter_media_phase_") as temp_dir:
        frame_cache: dict[tuple[str, float], Any] = {}
        temp_path = Path(temp_dir)
        for slide in scene.get("slides", []):
            slide_index = int(slide.get("index", 0) or 0)
            if slides is not None and slide_index not in slides:
                continue
            sample = slide_samples.get(slide_index)
            if sample is None:
                continue
            ref_path = ref_dir / f"{sample['id']}.png"
            if not ref_path.exists():
                _extract_reference_frame(ffmpeg, reference, ref_path, float(sample["referenceSec"]))
            with Image.open(ref_path).convert("RGB") as ref_img:
                ref_size = ref_img.size
                for obj in slide.get("objects", []):
                    asset_id = str(obj.get("assetId") or "")
                    asset = assets.get(asset_id)
                    if not asset or asset.get("kind") != "video" or not asset.get("file"):
                        continue
                    bbox = _object_pixel_bbox(obj, ref_size)
                    if bbox is None:
                        continue
                    x0, y0, x1, y1 = bbox
                    if (x1 - x0) < 64 or (y1 - y0) < 64:
                        continue
                    ref_patch = ref_img.crop(bbox)
                    predicted = float((sample.get("mediaClocks") or {}).get(str(obj.get("trackId")), 0.0) or 0.0)
                    asset_path = build_dir / str(asset["file"])
                    max_phase = _media_phase_search_limit(asset, obj, search_sec)
                    candidates = _phase_candidates(max_phase, step_sec, predicted)
                    best: dict[str, Any] | None = None
                    for phase in candidates:
                        try:
                            frame = _asset_video_frame(
                                ffmpeg,
                                asset_path,
                                phase,
                                temp_path,
                                frame_cache,
                            )
                        except PresenterError:
                            continue
                        patch = frame.resize(ref_patch.size, _resample_lanczos(Image))
                        score = _global_ssim(
                            np.asarray(ref_patch, dtype=np.float32),
                            np.asarray(patch, dtype=np.float32),
                        )
                        if best is None or score > float(best["score"]):
                            best = {"phaseSec": phase, "score": score}
                    if best is None:
                        continue
                    phase_target = _media_phase_target(scene, slide_index, obj, asset_id)
                    target_slide = int(phase_target["slide"])
                    target_obj = phase_target["object"]
                    current_phase = float((target_obj.get("mediaTiming") or {}).get("phaseSec", 0.0) or 0.0)
                    phase_delta = float(best["phaseSec"]) - predicted
                    row = {
                        "slide": slide_index,
                        "sampleId": sample["id"],
                        "objectId": obj.get("id"),
                        "trackId": obj.get("trackId"),
                        "assetId": asset_id,
                        "name": obj.get("name"),
                        "predictedMediaSec": round(predicted, 3),
                        "bestPhaseSec": round(float(best["phaseSec"]), 3),
                        "phaseDeltaSec": round(phase_delta, 3),
                        "currentPhaseSec": round(current_phase, 3),
                        "recommendedPhaseSec": round(current_phase + phase_delta, 3),
                        "score": round(float(best["score"]), 6),
                        "applied": False,
                        "targetSlide": target_slide,
                        "targetObjectId": target_obj.get("id"),
                        "targetTrackId": target_obj.get("trackId"),
                        "targetAssetId": target_obj.get("assetId"),
                        "targetName": target_obj.get("name"),
                        "targetReason": phase_target["reason"],
                        "geometry": obj.get("geometry"),
                        "paused": bool((obj.get("mediaTiming") or {}).get("paused")),
                    }
                    if apply and float(best["score"]) >= min_score:
                        timing = target_obj.setdefault("mediaTiming", {})
                        timing["phaseSec"] = round(current_phase + phase_delta, 3)
                        row["applied"] = True
                    rows.append(row)
        if include_transitions:
            slides_by_index = {int(slide.get("index", 0) or 0): slide for slide in scene.get("slides", [])}
            transitions_by_pair = {
                (int(transition.get("from", 0) or 0), int(transition.get("to", 0) or 0)): transition
                for transition in scene.get("transitions", [])
            }
            for sample in samples:
                if sample.get("kind") != "transition":
                    continue
                from_slide = int(sample.get("from", 0) or 0)
                to_slide = int(sample.get("to", 0) or 0)
                if slides is not None and from_slide not in slides:
                    continue
                transition = transitions_by_pair.get((from_slide, to_slide))
                if not transition:
                    continue
                ref_path = ref_dir / f"{sample['id']}.png"
                if not ref_path.exists():
                    _extract_reference_frame(ffmpeg, reference, ref_path, float(sample["referenceSec"]))
                with Image.open(ref_path).convert("RGB") as ref_img:
                    ref_size = ref_img.size
                    for media in _transition_effective_media_objects(
                        slides_by_index.get(from_slide, {}),
                        slides_by_index.get(to_slide, {}),
                        transition,
                        float(sample.get("progress", 0.0) or 0.0),
                        assets,
                    ):
                        obj = media["object"]
                        track_id = str(obj.get("trackId") or "")
                        if track_id not in (sample.get("mediaClocks") or {}):
                            continue
                        asset_id = str(obj.get("assetId") or "")
                        asset = assets.get(asset_id)
                        if not asset or asset.get("kind") != "video" or not asset.get("file"):
                            continue
                        bbox = _object_pixel_bbox(obj, ref_size)
                        if bbox is None:
                            continue
                        x0, y0, x1, y1 = bbox
                        if (x1 - x0) < 64 or (y1 - y0) < 64:
                            continue
                        ref_patch = ref_img.crop(bbox)
                        predicted = float((sample.get("mediaClocks") or {}).get(track_id, 0.0) or 0.0)
                        asset_path = build_dir / str(asset["file"])
                        source_obj = media["sourceObject"]
                        max_phase = _media_phase_search_limit(asset, source_obj, search_sec)
                        candidates = _phase_candidates(max_phase, step_sec, predicted)
                        best = None
                        for phase in candidates:
                            try:
                                frame = _asset_video_frame(
                                    ffmpeg,
                                    asset_path,
                                    phase,
                                    temp_path,
                                    frame_cache,
                                )
                            except PresenterError:
                                continue
                            patch = _visible_asset_patch(
                                frame,
                                obj,
                                bbox,
                                ref_size,
                            ).resize(ref_patch.size, _resample_lanczos(Image))
                            score = _global_ssim(
                                np.asarray(ref_patch, dtype=np.float32),
                                np.asarray(patch, dtype=np.float32),
                            )
                            if best is None or score > float(best["score"]):
                                best = {"phaseSec": phase, "score": score}
                        if best is None:
                            continue
                        source_slide = int(media["sourceSlide"])
                        phase_target = _media_phase_target(scene, source_slide, source_obj, asset_id)
                        target_slide = int(phase_target["slide"])
                        target_obj = phase_target["object"]
                        current_phase = float((target_obj.get("mediaTiming") or {}).get("phaseSec", 0.0) or 0.0)
                        current_transition_phase = float((obj.get("mediaTiming") or {}).get("phaseSec", current_phase) or 0.0)
                        phase_delta = float(best["phaseSec"]) - predicted
                        row = {
                            "kind": "transition",
                            "slide": from_slide,
                            "from": from_slide,
                            "to": to_slide,
                            "progress": round(float(sample.get("progress", 0.0) or 0.0), 3),
                            "sampleId": sample["id"],
                            "objectId": source_obj.get("id"),
                            "trackId": track_id,
                            "assetId": asset_id,
                            "name": source_obj.get("name"),
                            "predictedMediaSec": round(predicted, 3),
                            "bestPhaseSec": round(float(best["phaseSec"]), 3),
                            "phaseDeltaSec": round(phase_delta, 3),
                            "currentPhaseSec": round(current_phase, 3),
                            "currentTransitionPhaseSec": round(current_transition_phase, 3),
                            "recommendedPhaseSec": round(current_phase + phase_delta, 3),
                            "recommendedTransitionPhaseSec": round(current_transition_phase + phase_delta, 3),
                            "score": round(float(best["score"]), 6),
                            "applied": False,
                            "targetSlide": target_slide,
                            "targetObjectId": target_obj.get("id"),
                            "targetTrackId": target_obj.get("trackId"),
                            "targetAssetId": target_obj.get("assetId"),
                            "targetName": target_obj.get("name"),
                            "targetReason": phase_target["reason"],
                            "geometry": obj.get("geometry"),
                            "visibleBBox": [x0, y0, x1, y1],
                            "sourceSlide": source_slide,
                            "sourceObjectId": source_obj.get("id"),
                            "paused": bool((source_obj.get("mediaTiming") or {}).get("paused")),
                        }
                        if apply and float(best["score"]) >= min_score:
                            timing = target_obj.setdefault("mediaTiming", {})
                            timing["phaseSec"] = round(current_phase + phase_delta, 3)
                            row["applied"] = True
                        rows.append(row)

    report = {
        "schema": "pptx-html-presenter.media-phase.v1",
        "generatedAtUtc": utc_now_iso(),
        "reference": str(reference),
        "stepSec": step_sec,
        "searchSec": search_sec,
        "minScore": min_score,
        "includeTransitions": include_transitions,
        "applied": apply,
        "rows": rows,
        "configOverrides": _media_phase_config_overrides(rows, min_score),
        "transitionConfigOverrides": _transition_media_phase_config_overrides(rows, min_score),
        "summary": {
            "count": len(rows),
            "appliedCount": sum(1 for row in rows if row["applied"]),
            "slides": sorted({row["slide"] for row in rows}),
        },
    }
    write_json(qa_dir / "media-phase-report.json", report)
    if apply:
        write_json(scene_path, scene)
    return report


def _media_phase_config_overrides(rows: list[dict[str, Any]], min_score: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("kind") == "transition":
            continue
        if float(row.get("score", 0.0) or 0.0) < min_score:
            continue
        key = (
            row.get("targetSlide", row.get("slide")),
            row.get("targetObjectId", row.get("objectId")),
            row.get("targetTrackId", row.get("trackId")),
            row.get("targetAssetId", row.get("assetId")),
        )
        grouped.setdefault(key, []).append(row)
    overrides: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(part) for part in item)):
        group = grouped[key]
        representative = max(group, key=lambda item: float(item.get("score", 0.0) or 0.0))
        phases = sorted(float(row.get("recommendedPhaseSec", 0.0) or 0.0) for row in group)
        scores = sorted(float(row.get("score", 0.0) or 0.0) for row in group)
        override = {
            "slide": representative.get("targetSlide", representative.get("slide")),
            "object_id": representative.get("targetObjectId", representative.get("objectId")),
            "track_id": representative.get("targetTrackId", representative.get("trackId")),
            "asset_id": representative.get("targetAssetId", representative.get("assetId")),
            "name": representative.get("targetName", representative.get("name")),
            "phase_sec": round(_median(phases), 3),
            "score": round(_median(scores), 6),
            "sample_count": len(group),
            "source": "media-phase",
        }
        if representative.get("targetReason") and representative.get("targetReason") != "observed-object":
            override.update(
                {
                    "observed_slide": representative.get("slide"),
                    "observed_object_id": representative.get("objectId"),
                    "target_reason": representative.get("targetReason"),
                }
            )
        overrides.append(override)
    return overrides


def _transition_media_phase_config_overrides(rows: list[dict[str, Any]], min_score: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any, Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("kind") != "transition":
            continue
        if float(row.get("score", 0.0) or 0.0) < min_score:
            continue
        key = (
            row.get("from"),
            row.get("to"),
            row.get("targetTrackId", row.get("trackId")),
            row.get("targetObjectId", row.get("objectId")),
            row.get("targetAssetId", row.get("assetId")),
        )
        grouped.setdefault(key, []).append(row)
    overrides: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(part) for part in item)):
        group = grouped[key]
        representative = max(group, key=lambda item: float(item.get("score", 0.0) or 0.0))
        phases = sorted(
            float(
                row.get(
                    "recommendedTransitionPhaseSec",
                    row.get("recommendedPhaseSec", 0.0),
                )
                or 0.0
            )
            for row in group
        )
        if len(phases) > 1 and (phases[-1] - phases[0]) > 1.5:
            continue
        scores = sorted(float(row.get("score", 0.0) or 0.0) for row in group)
        overrides.append(
            {
                "from": representative.get("from"),
                "to": representative.get("to"),
                "object_id": representative.get("targetObjectId", representative.get("objectId")),
                "track_id": representative.get("targetTrackId", representative.get("trackId")),
                "asset_id": representative.get("targetAssetId", representative.get("assetId")),
                "name": representative.get("targetName", representative.get("name")),
                "phase_sec": round(_median(phases), 3),
                "score": round(_median(scores), 6),
                "sample_count": len(group),
                "source": "media-phase-transition",
            }
        )
    return overrides


def _transition_effective_media_objects(
    from_slide: dict[str, Any],
    to_slide: dict[str, Any],
    transition: dict[str, Any],
    progress: float,
    assets: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    from_index = int(from_slide.get("index", transition.get("from", 0)) or 0)
    to_index = int(to_slide.get("index", transition.get("to", 0)) or 0)
    from_by_track = {
        str(obj.get("trackId") or ""): obj
        for obj in from_slide.get("objects", [])
        if obj.get("trackId")
    }
    to_by_track = {
        str(obj.get("trackId") or ""): obj
        for obj in to_slide.get("objects", [])
        if obj.get("trackId")
    }
    inferred_by_track = {
        str(row.get("trackId") or ""): row
        for row in transition.get("inferredMotions", []) or []
        if row.get("trackId")
    }
    eased = _transition_interpolation_progress(progress, transition)
    rows: list[dict[str, Any]] = []
    for track_id in sorted(set(from_by_track) | set(to_by_track)):
        from_obj = from_by_track.get(track_id)
        to_obj = to_by_track.get(track_id)
        source_obj = _transition_render_object_for_assets(from_obj, to_obj, assets)
        if source_obj is None:
            continue
        asset = assets.get(str(source_obj.get("assetId") or ""), {}) or {}
        if asset.get("kind") != "video":
            continue
        source_slide = from_index if source_obj is from_obj else to_index
        from_geometry, to_geometry = _transition_geometry_pair(from_obj, to_obj, inferred_by_track.get(track_id))
        if from_geometry is None and to_geometry is None:
            continue
        if from_geometry is None:
            from_geometry = to_geometry
        if to_geometry is None:
            to_geometry = from_geometry
        obj = dict(source_obj)
        obj["geometry"] = _lerp_scene_geometry(from_geometry or {}, to_geometry or {}, eased)
        _apply_transition_media_phase_override(obj, transition)
        rows.append(
            {
                "trackId": track_id,
                "object": obj,
                "sourceObject": source_obj,
                "sourceSlide": source_slide,
            }
        )
    return rows


def _transition_render_object_for_assets(
    from_obj: dict[str, Any] | None,
    to_obj: dict[str, Any] | None,
    assets: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if from_obj is None or to_obj is None:
        return to_obj or from_obj
    from_asset = assets.get(str(from_obj.get("assetId") or ""), {}) or {}
    to_asset = assets.get(str(to_obj.get("assetId") or ""), {}) or {}
    same_asset = bool(from_obj.get("assetId")) and from_obj.get("assetId") == to_obj.get("assetId")
    if from_asset.get("kind") == "video" and same_asset:
        if (
            not _is_paused_scene_media(to_obj)
            and not _is_scene_object_visible(from_obj)
            and _has_explicit_media_phase_for_scene(to_obj)
            and not _is_animated_loop_asset_for_scene(from_asset)
        ):
            return to_obj
        if (
            _is_paused_scene_media(from_obj)
            and not _is_paused_scene_media(to_obj)
            and not _is_scene_object_visible(from_obj)
            and not _is_animated_loop_asset_for_scene(from_asset)
        ):
            return to_obj
        if (
            not _is_paused_scene_media(to_obj)
            or _is_scene_object_visible(from_obj)
            or _is_animated_loop_asset_for_scene(from_asset)
        ):
            return from_obj
    if from_asset.get("kind") == "video" and (
        _is_scene_object_visible(from_obj)
        or to_asset.get("kind") != "video"
    ):
        return from_obj
    if to_asset.get("kind") == "video":
        return to_obj
    return to_obj


def _has_explicit_media_phase_for_scene(obj: dict[str, Any]) -> bool:
    try:
        float((obj.get("mediaTiming") or {}).get("phaseSec"))
    except (TypeError, ValueError):
        return False
    return True


def _apply_transition_media_phase_override(obj: dict[str, Any], transition: dict[str, Any] | None) -> None:
    phase = _transition_media_phase_override(obj, transition)
    if phase is None:
        return
    timing = dict(obj.get("mediaTiming") or {})
    timing["phaseSec"] = phase
    obj["mediaTiming"] = timing


def _transition_media_phase_override(
    obj: dict[str, Any],
    transition: dict[str, Any] | None,
) -> float | None:
    if not transition:
        return None
    for row in transition.get("mediaPhaseOverrides", []) or []:
        if row.get("trackId") and str(row.get("trackId")) != str(obj.get("trackId") or ""):
            continue
        if row.get("objectId") and str(row.get("objectId")) != str(obj.get("id") or ""):
            continue
        if row.get("assetId") and str(row.get("assetId")) != str(obj.get("assetId") or ""):
            continue
        if row.get("name") and str(row.get("name")) != str(obj.get("name") or ""):
            continue
        try:
            return float(_row_value(row, "phaseSec", "phase_sec"))
        except (TypeError, ValueError):
            continue
    return None


def _transition_geometry_pair(
    from_obj: dict[str, Any] | None,
    to_obj: dict[str, Any] | None,
    inferred: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if from_obj is not None and to_obj is not None:
        return from_obj.get("geometry"), to_obj.get("geometry")
    if from_obj is None and to_obj is not None:
        return (inferred or {}).get("fromGeometry"), to_obj.get("geometry")
    if from_obj is not None and to_obj is None:
        return from_obj.get("geometry"), (inferred or {}).get("toGeometry")
    return None, None


def _transition_interpolation_progress(progress: float, transition: dict[str, Any]) -> float:
    mapped = _progress_map_value(progress, transition.get("progressMap"))
    if mapped is not None:
        return mapped
    easing = transition.get("easing") or "easeInOutQuad"
    value = _clamp01(progress)
    if isinstance(easing, str):
        key = easing.strip().lower()
        if key == "linear":
            return value
        if key.startswith("power"):
            try:
                exponent = float(key.replace("power", "").replace(":", "").strip("() ") or "1")
            except ValueError:
                exponent = 1.0
            return value ** max(0.001, exponent)
    return value * 2 * value if value < 0.5 else 1 - ((-2 * value + 2) ** 2) / 2


def _progress_map_value(progress: float, points: Any) -> float | None:
    if not isinstance(points, list) or len(points) < 2:
        return None
    normalized: list[dict[str, float]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            normalized.append(
                {
                    "progress": _clamp01(float(point.get("progress"))),
                    "value": _clamp01(
                        float(
                            _row_value(
                                point,
                                "value",
                                "mappedProgress",
                                "mapped_progress",
                                "interpolationProgress",
                                "interpolation_progress",
                            )
                        )
                    ),
                }
            )
        except (TypeError, ValueError):
            continue
    if len(normalized) < 2:
        return None
    normalized.sort(key=lambda item: item["progress"])
    x = _clamp01(progress)
    if x <= normalized[0]["progress"]:
        return normalized[0]["value"]
    for left, right in zip(normalized, normalized[1:]):
        if x <= right["progress"]:
            span = max(0.0001, right["progress"] - left["progress"])
            return left["value"] + ((right["value"] - left["value"]) * ((x - left["progress"]) / span))
    return normalized[-1]["value"]


def _lerp_scene_geometry(left: dict[str, Any], right: dict[str, Any], progress: float) -> dict[str, Any]:
    out = dict(right)
    for key in ("leftPct", "topPct", "widthPct", "heightPct", "x", "y", "w", "h", "rotation"):
        if key in left or key in right:
            out[key] = _lerp_float(left.get(key, 0.0), right.get(key, 0.0), progress)
    out["flipH"] = bool(left.get("flipH")) if progress < 0.5 else bool(right.get("flipH"))
    out["flipV"] = bool(left.get("flipV")) if progress < 0.5 else bool(right.get("flipV"))
    return out


def _visible_asset_patch(
    frame: Any,
    obj: dict[str, Any],
    bbox: tuple[int, int, int, int],
    render_size: tuple[int, int],
) -> Any:
    geometry = obj.get("geometry") or {}
    render_w, render_h = render_size
    obj_left = float(geometry.get("leftPct", 0.0) or 0.0) * render_w
    obj_top = float(geometry.get("topPct", 0.0) or 0.0) * render_h
    obj_w = max(0.001, float(geometry.get("widthPct", 0.0) or 0.0) * render_w)
    obj_h = max(0.001, float(geometry.get("heightPct", 0.0) or 0.0) * render_h)
    x0, y0, x1, y1 = bbox
    frac_l = _clamp01((x0 - obj_left) / obj_w)
    frac_r = _clamp01((x1 - obj_left) / obj_w)
    frac_t = _clamp01((y0 - obj_top) / obj_h)
    frac_b = _clamp01((y1 - obj_top) / obj_h)
    if frac_r <= frac_l or frac_b <= frac_t:
        return frame
    source_w, source_h = frame.size
    crop = (
        max(0, min(source_w, int(round(frac_l * source_w)))),
        max(0, min(source_h, int(round(frac_t * source_h)))),
        max(0, min(source_w, int(round(frac_r * source_w)))),
        max(0, min(source_h, int(round(frac_b * source_h)))),
    )
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        return frame
    return frame.crop(crop)


def _is_scene_object_visible(obj: dict[str, Any] | None) -> bool:
    if obj is None:
        return False
    geometry = obj.get("geometry") or {}
    left = float(geometry.get("leftPct", 0.0) or 0.0)
    top = float(geometry.get("topPct", 0.0) or 0.0)
    width = float(geometry.get("widthPct", 0.0) or 0.0)
    height = float(geometry.get("heightPct", 0.0) or 0.0)
    opacity = float(obj.get("opacity", 1.0) or 0.0)
    return opacity > 0.01 and left < 1.0 and top < 1.0 and (left + width) > 0.0 and (top + height) > 0.0


def _is_paused_scene_media(obj: dict[str, Any]) -> bool:
    return bool((obj.get("mediaTiming") or {}).get("paused"))


def _is_animated_loop_asset_for_scene(asset: dict[str, Any]) -> bool:
    source = f"{asset.get('sourceFile', '')} {asset.get('sourcePath', '')} {asset.get('extension', '')}".lower()
    return bool(asset.get("animated")) or ".gif" in source


def _lerp_float(left: Any, right: Any, progress: float) -> float:
    return float(left or 0.0) + ((float(right or 0.0) - float(left or 0.0)) * _clamp01(progress))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _media_phase_target(
    scene: dict[str, Any],
    slide_index: int,
    obj: dict[str, Any],
    asset_id: str,
) -> dict[str, Any]:
    timing = obj.get("mediaTiming") or {}
    if timing.get("kind") == "playFrom" or timing.get("paused"):
        return {"slide": slide_index, "object": obj, "reason": "observed-object"}
    track_id = str(obj.get("trackId") or "")
    if not track_id or not asset_id:
        return {"slide": slide_index, "object": obj, "reason": "observed-object"}

    origin: tuple[int, dict[str, Any]] | None = None
    for slide in sorted(scene.get("slides", []), key=lambda item: int(item.get("index", 0) or 0)):
        current_index = int(slide.get("index", 0) or 0)
        if current_index > slide_index:
            break
        candidate = next(
            (
                candidate
                for candidate in slide.get("objects", [])
                if str(candidate.get("trackId") or "") == track_id
            ),
            None,
        )
        if (
            candidate
            and str(candidate.get("assetId") or "") == asset_id
            and _is_persistent_loop_media(candidate)
        ):
            if origin is None:
                origin = (current_index, candidate)
            continue
        origin = None
    if origin is None:
        return {"slide": slide_index, "object": obj, "reason": "observed-object"}
    origin_slide, origin_obj = origin
    if origin_slide == slide_index and origin_obj is obj:
        return {"slide": slide_index, "object": obj, "reason": "observed-object"}
    return {"slide": origin_slide, "object": origin_obj, "reason": "persistent-loop-origin"}


def _is_persistent_loop_media(obj: dict[str, Any]) -> bool:
    timing = obj.get("mediaTiming") or {}
    return timing.get("kind") != "playFrom" and not bool(timing.get("paused"))


def _transition_time_current_offsets(scene: dict[str, Any]) -> dict[tuple[int, int], float]:
    offsets: dict[tuple[int, int], float] = {}
    for key, row in _transition_time_override_map(
        scene.get("qa", {}).get("transitionTimeOverrides", []) or []
    ).items():
        offsets[key] = float(row.get("referenceOffsetSec", 0.0) or 0.0)
    return offsets


def _transition_time_config_overrides(
    alignment_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    current_offsets: dict[tuple[int, int], float],
    min_score: float,
) -> list[dict[str, Any]]:
    samples_by_id = {sample["id"]: sample for sample in samples}
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in alignment_rows:
        if row.get("kind") != "transition":
            continue
        if float(row.get("alignedSsim", 0.0) or 0.0) < min_score:
            continue
        sample = samples_by_id.get(str(row.get("sampleId")))
        if not sample:
            continue
        key = (int(sample["from"]), int(sample["to"]))
        grouped.setdefault(key, []).append(row)

    overrides: list[dict[str, Any]] = []
    for (from_slide, to_slide), rows in sorted(grouped.items()):
        deltas = sorted(float(row["alignmentDeltaSec"]) for row in rows)
        scores = sorted(float(row["alignedSsim"]) for row in rows)
        offset = current_offsets.get((from_slide, to_slide), 0.0) + _median(deltas)
        progress_offsets = []
        for row in rows:
            sample = samples_by_id.get(str(row.get("sampleId")), {})
            progress_offsets.append(
                {
                    "progress": round(float(sample.get("progress", 0.0) or 0.0), 3),
                    "reference_offset_sec": round(
                        current_offsets.get((from_slide, to_slide), 0.0)
                        + float(row["alignmentDeltaSec"]),
                        3,
                    ),
                    "score": round(float(row.get("alignedSsim", 0.0) or 0.0), 6),
                }
            )
        overrides.append(
            {
                "from": from_slide,
                "to": to_slide,
                "reference_offset_sec": round(offset, 3),
                "progress_offsets": sorted(progress_offsets, key=lambda item: item["progress"]),
                "sample_count": len(rows),
                "median_alignment_delta_sec": round(_median(deltas), 3),
                "median_aligned_ssim": round(_median(scores), 6),
                "source": "transition-time",
            }
        )
    return overrides


def _merge_transition_time_overrides(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for row in existing:
        try:
            key = (int(_row_value(row, "from", "fromSlide", "from_slide")), int(_row_value(row, "to", "toSlide", "to_slide")))
        except (TypeError, ValueError):
            continue
        merged[key] = dict(row)
    for row in updates:
        key = (int(row["from"]), int(row["to"]))
        merged[key] = dict(row)
    return [merged[key] for key in sorted(merged)]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _calibrate_reference_alignment(
    ffmpeg: Path,
    reference: Path,
    build_dir: Path,
    samples: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    *,
    fps: int = 8,
    slide_window_sec: float = 1.5,
    transition_window_sec: float = 1.0,
) -> list[dict[str, Any]]:
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return []

    thumb_size = (480, 270)
    strict_by_id = {row["sampleId"]: row for row in comparisons}
    max_sec = max((float(sample["referenceSec"]) for sample in samples), default=0.0) + 2.0
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pptx_html_presenter_ref_cal_") as temp_dir:
        frame_dir = Path(temp_dir)
        _extract_reference_thumbnails(ffmpeg, reference, frame_dir, fps, max_sec, thumb_size)
        for sample in samples:
            sample_id = sample["id"]
            html_path = build_dir / "qa" / "html" / f"{sample_id}.png"
            if not html_path.exists():
                continue
            with Image.open(html_path).convert("RGB") as html_img:
                html_img = html_img.resize(thumb_size)
                html_arr = np.asarray(html_img, dtype=np.float32)
            reference_sec = float(sample["referenceSec"])
            window = slide_window_sec if sample["kind"] == "slide" else transition_window_sec
            best: tuple[float, float] | None = None
            start_frame = max(1, int((max(0.0, reference_sec - window) * fps)) + 1)
            end_frame = int(((reference_sec + window) * fps)) + 1
            for frame_index in range(start_frame, end_frame + 1):
                frame_path = frame_dir / f"frame_{frame_index:06d}.jpg"
                if not frame_path.exists():
                    continue
                with Image.open(frame_path).convert("RGB") as ref_img:
                    ref_arr = np.asarray(ref_img, dtype=np.float32)
                score = _global_ssim(ref_arr, html_arr)
                frame_sec = (frame_index - 1) / fps
                if best is None or score > best[0]:
                    best = (score, frame_sec)
            if best is None:
                continue
            strict = strict_by_id.get(sample_id, {})
            threshold = float(strict.get("threshold", 0.985 if sample["kind"] == "slide" else 0.965))
            rows.append(
                {
                    "sampleId": sample_id,
                    "kind": sample["kind"],
                    "predictedReferenceSec": round(reference_sec, 3),
                    "alignedReferenceSec": round(best[1], 3),
                    "alignmentDeltaSec": round(best[1] - reference_sec, 3),
                    "strictSsim": strict.get("ssim"),
                    "alignedSsim": round(best[0], 6),
                    "threshold": threshold,
                    "alignedPassed": best[0] >= threshold,
                    "diagnosticOnly": True,
                }
            )
    return rows


def _extract_reference_thumbnails(
    ffmpeg: Path,
    reference: Path,
    output_dir: Path,
    fps: int,
    max_sec: float,
    size: tuple[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-i",
            str(reference),
            "-t",
            f"{max_sec:.3f}",
            "-vf",
            f"fps={fps},scale={size[0]}:{size[1]}",
            "-q:v",
            "4",
            str(output_dir / "frame_%06d.jpg"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _object_pixel_bbox(obj: dict[str, Any], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    geometry = obj.get("geometry") or {}
    width, height = size
    left = float(geometry.get("leftPct", 0.0) or 0.0) * width
    top = float(geometry.get("topPct", 0.0) or 0.0) * height
    right = left + (float(geometry.get("widthPct", 0.0) or 0.0) * width)
    bottom = top + (float(geometry.get("heightPct", 0.0) or 0.0) * height)
    x0 = max(0, min(width, int(round(left))))
    y0 = max(0, min(height, int(round(top))))
    x1 = max(0, min(width, int(round(right))))
    y1 = max(0, min(height, int(round(bottom))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _media_phase_search_limit(asset: dict[str, Any], obj: dict[str, Any], search_sec: float) -> float:
    timing = obj.get("mediaTiming") or {}
    durations = [
        float(value)
        for value in (asset.get("durationSec"), timing.get("durationSec"), search_sec)
        if value is not None and float(value) > 0
    ]
    return max(0.0, min(float(search_sec), min(durations) if durations else float(search_sec)))


def _phase_candidates(max_phase: float, step_sec: float, predicted: float) -> list[float]:
    step = max(0.05, float(step_sec))
    max_safe_phase = max(0.0, max_phase - 0.08)
    values = {0.0, round(max(0.0, min(max_safe_phase, predicted)), 3)}
    count = int(max_safe_phase / step) + 1
    for index in range(count + 1):
        values.add(round(min(max_safe_phase, index * step), 3))
    values.add(round(max_safe_phase, 3))
    return sorted(values)


def _asset_video_frame(
    ffmpeg: Path,
    asset_path: Path,
    phase_sec: float,
    temp_dir: Path,
    frame_cache: dict[tuple[str, float], Any],
) -> Any:
    from PIL import Image

    key = (str(asset_path), round(float(phase_sec), 3))
    cached = frame_cache.get(key)
    if cached is not None:
        return cached.copy()
    out = temp_dir / f"asset_{abs(hash(key))}.png"
    command = [
        str(ffmpeg),
        "-y",
        "-ss",
        f"{phase_sec:.3f}",
        "-i",
        str(asset_path),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(out),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not out.exists():
        subprocess.run(
            [
                str(ffmpeg),
                "-y",
                "-i",
                str(asset_path),
                "-ss",
                f"{phase_sec:.3f}",
                "-frames:v",
                "1",
                "-update",
                "1",
                str(out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if not out.exists():
        raise PresenterError(f"ffmpeg did not emit a frame for {asset_path} at {phase_sec:.3f}s")
    with Image.open(out).convert("RGB") as img:
        frame_cache[key] = img.copy()
    return frame_cache[key].copy()


def _resample_lanczos(image_module: Any) -> Any:
    return getattr(getattr(image_module, "Resampling", image_module), "LANCZOS")


def _calibration_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"enabled": True, "count": 0}
    slide_rows = [row for row in rows if row["kind"] == "slide"]
    transition_rows = [row for row in rows if row["kind"] == "transition"]
    return {
        "enabled": True,
        "count": len(rows),
        "fps": 8,
        "slideSearchWindowSec": 1.5,
        "transitionSearchWindowSec": 1.0,
        "slideAlignedPassed": sum(1 for row in slide_rows if row["alignedPassed"]),
        "transitionAlignedPassed": sum(1 for row in transition_rows if row["alignedPassed"]),
        "largestAbsDeltaSec": max(abs(float(row["alignmentDeltaSec"])) for row in rows),
    }


def _global_ssim(left: Any, right: Any) -> float:
    import numpy as np

    left_gray = (
        (left[..., 0].astype(np.float32) * 0.2126)
        + (left[..., 1].astype(np.float32) * 0.7152)
        + (left[..., 2].astype(np.float32) * 0.0722)
    )
    right_gray = (
        (right[..., 0].astype(np.float32) * 0.2126)
        + (right[..., 1].astype(np.float32) * 0.7152)
        + (right[..., 2].astype(np.float32) * 0.0722)
    )
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_x = _mean_filter(left_gray)
    mu_y = _mean_filter(right_gray)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x = _mean_filter(left_gray * left_gray) - mu_x_sq
    sigma_y = _mean_filter(right_gray * right_gray) - mu_y_sq
    sigma_xy = _mean_filter(left_gray * right_gray) - mu_xy
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    score_map = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator != 0,
    )
    return float(np.clip(np.mean(score_map), -1.0, 1.0))


def _mean_filter(values: Any, radius: int = 5) -> Any:
    import numpy as np

    window = radius * 2 + 1
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="reflect").astype(np.float32)
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    sums = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return sums / float(window * window)


def _write_contact_sheet_stub(qa_dir: Path, report: dict[str, Any]) -> None:
    rows = "\n".join(
        f"<tr><td>{sample['id']}</td><td>{sample['kind']}</td><td>{sample['referenceSec']}</td></tr>"
        for sample in report["samples"]
    )
    html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>QA Samples</title>
<style>
body {{ margin: 24px; background: #101010; color: #f0f0f0; font: 14px/1.4 sans-serif; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border-bottom: 1px solid #333; padding: 6px 8px; text-align: left; }}
</style>
<h1>QA Samples</h1>
<p>Status: {report['status']}</p>
<table><thead><tr><th>Sample</th><th>Kind</th><th>Reference second</th></tr></thead><tbody>{rows}</tbody></table>
</html>
"""
    (qa_dir / "contact-sheet.html").write_text(html, encoding="utf-8")
