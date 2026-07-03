from __future__ import annotations

import math
import re
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import PresenterConfig
from .models import PptxDeck, SceneObject, Slide
from .utils import slugify, utc_now_iso, write_json


def compile_scene(deck: PptxDeck, config: PresenterConfig, out_dir: Path) -> dict[str, Any]:
    _assign_tracks(deck, config.morph_policy.match_threshold)
    transitions = _compile_transitions(deck, config)
    title = config.title or deck.title
    slug = config.slug or slugify(title)
    slides = [slide.to_scene(deck.slide_width, deck.slide_height) for slide in deck.slides]
    assets = [asset.to_scene() for asset in sorted(deck.assets.values(), key=lambda a: a.id)]
    phase_override_report = _apply_media_phase_overrides(slides, config.media_phase_overrides)
    raster_fallback_report = _apply_raster_fallback_overrides(
        slides,
        assets,
        config.raster_fallback_overrides,
        out_dir,
        deck.slide_width,
        deck.slide_height,
        allow_full_slide_static=config.fallback_policy.full_slide_static,
    )
    panel_relationship_report = _annotate_panel_relationships(
        slides,
        deck.slide_width,
        deck.slide_height,
        infer_panels=config.group_policy.infer_panels,
        panel_border_on_top=config.group_policy.panel_border_on_top,
    )
    graph_report = _annotate_scene_graph_v2(
        slides,
        deck.slide_width,
        deck.slide_height,
        explicit_groups=config.group_policy.explicit_groups,
    )
    panel_transition_report = _apply_panel_relationships_to_transitions(
        transitions,
        slides,
        deck.slide_width,
        deck.slide_height,
    )
    scene = {
        "schema": f"pptx-html-presenter.scene.v{config.scene_schema_version}",
        "schemaVersion": config.scene_schema_version,
        "generatedAtUtc": utc_now_iso(),
        "deck": {
            "id": slug,
            "title": title,
            "sourcePath": deck.source_path,
            "sourceSha256": deck.source_sha256,
            "slideCount": len(deck.slides),
            "slideSize": {"width": deck.slide_width, "height": deck.slide_height},
            "renderProfile": {
                "name": config.profile.name,
                "width": config.profile.width or deck.slide_width,
                "height": config.profile.height or deck.slide_height,
                "fps": config.profile.fps,
            },
        },
        "assets": assets,
        "slides": slides,
        "transitions": transitions,
        "runtime": {
            "easing": config.morph_policy.easing,
            "fadeUnmatched": config.morph_policy.fade_unmatched,
            "unmatchedFadeStart": config.morph_policy.unmatched_fade_start,
            "unmatchedFadeEnd": config.morph_policy.unmatched_fade_end,
            "reverse": config.morph_policy.reverse,
            "groupRenderer": config.scene_schema_version >= 2,
            "outlineStyle": {
                "normalizeWhiteOutlines": config.outline_policy.normalize_white_outlines,
                "borderOnTop": config.outline_policy.border_on_top,
                "widthPct": config.outline_policy.width_pct,
                "minPx": config.outline_policy.min_px,
                "maxPx": config.outline_policy.max_px,
            },
            "visualEffects": {
                "glowScale": config.visual_effects.glow_scale,
                "glowAlphaScale": config.visual_effects.glow_alpha_scale,
            },
            "layerPolicy": _runtime_layer_policy(config),
            "autoAdvance": _runtime_auto_advance_rows(config.auto_advance),
            "autoSegments": _runtime_auto_advance_rows(config.auto_segments),
        },
        "qa": {
            "slideHoldSec": config.qa_policy.slide_hold_sec,
            "settledOffsetSec": config.qa_policy.settled_offset_sec,
            "transitionSamples": list(config.qa_policy.transition_samples),
            "transitionReferenceLeadFraction": config.qa_policy.transition_reference_lead_fraction,
            "transitionTimeOverrides": list(config.transition_time_overrides),
            "slideTimedVideoPhaseSec": config.qa_policy.slide_timed_video_phase_sec,
            "mediaPhaseOverridesApplied": phase_override_report["appliedCount"],
            "mediaPhaseOverrideRows": phase_override_report["rows"],
            "rasterFallbacksApplied": raster_fallback_report["appliedCount"],
            "rasterFallbackRows": raster_fallback_report["rows"],
            "panelRelationshipRows": panel_relationship_report["rows"],
            "panelRelationshipsApplied": panel_relationship_report["appliedCount"],
            "groupRows": graph_report["rows"],
            "groupsApplied": graph_report["groupCount"],
            "relationshipsApplied": graph_report["relationshipCount"],
            "panelTransitionRows": panel_transition_report["rows"],
            "panelTransitionRowsApplied": panel_transition_report["appliedCount"],
            "visualAudit": {
                "enabled": config.visual_audit.enabled,
                "samples": list(config.visual_audit.samples),
                "reverseMidpoints": config.visual_audit.reverse_midpoints,
                "failOnTimeout": config.visual_audit.fail_on_timeout,
            },
        },
    }
    write_json(out_dir / "deck.scene.json", scene)
    write_json(out_dir / "provenance.json", _provenance(scene))
    write_json(out_dir / "group-report.json", graph_report)
    return scene


def _apply_media_phase_overrides(
    slides: list[dict[str, Any]], overrides: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not overrides:
        return {"appliedCount": 0, "rows": rows}
    slides_by_index = {int(slide["index"]): slide for slide in slides}
    applied_count = 0
    for index, override in enumerate(overrides):
        row: dict[str, Any] = {"index": index, "status": "skipped"}
        try:
            slide_index = int(_override_value(override, "slide", "slideIndex", "slide_index"))
            phase_sec = float(_override_value(override, "phase_sec", "phaseSec"))
        except (TypeError, ValueError):
            row["status"] = "invalid"
            rows.append(row)
            continue
        row.update({"slide": slide_index, "phaseSec": round(phase_sec, 3)})
        slide = slides_by_index.get(slide_index)
        if slide is None:
            row["status"] = "missing-slide"
            rows.append(row)
            continue
        targets = [
            obj
            for obj in slide.get("objects", [])
            if _media_phase_override_matches(obj, override)
        ]
        if len(targets) != 1:
            row["status"] = "missing-object" if not targets else "ambiguous-object"
            row["matchCount"] = len(targets)
            rows.append(row)
            continue
        target = targets[0]
        timing = target.setdefault("mediaTiming", {})
        timing["phaseSec"] = round(phase_sec, 3)
        target.setdefault("provenance", {})["mediaPhaseOverride"] = {
            "phaseSec": round(phase_sec, 3),
            "source": override.get("source", "config"),
            "score": override.get("score"),
        }
        row.update(
            {
                "status": "applied",
                "objectId": target.get("id"),
                "trackId": target.get("trackId"),
                "assetId": target.get("assetId"),
                "name": target.get("name"),
            }
        )
        applied_count += 1
        rows.append(row)
    return {"appliedCount": applied_count, "rows": rows}


def _runtime_auto_advance_rows(rows: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            from_slide = int(_override_value(row, "from", "from_slide", "fromSlide") or 0)
            to_slide = int(_override_value(row, "to", "to_slide", "toSlide") or from_slide + 1)
        except (TypeError, ValueError):
            continue
        if from_slide <= 0 or to_slide <= 0:
            continue
        out.append(
            {
                "from": from_slide,
                "to": to_slide,
                "delaySec": float(_override_value(row, "delay_sec", "delaySec", "delay") or 0.0),
                "source": str(_override_value(row, "source") or "config"),
            }
        )
    return out


def _runtime_layer_policy(config: PresenterConfig) -> dict[str, Any]:
    overrides: list[dict[str, Any]] = []
    for row in config.layer_policy.transition_layer_overrides:
        try:
            from_slide = int(_override_value(row, "from", "from_slide", "fromSlide") or 0)
            to_slide = int(_override_value(row, "to", "to_slide", "toSlide") or 0)
        except (TypeError, ValueError):
            continue
        if from_slide <= 0 or to_slide <= 0:
            continue
        override: dict[str, Any] = {
            "from": from_slide,
            "to": to_slide,
            "mode": str(_override_value(row, "mode") or "panels-above-decorative"),
        }
        for source_key, output_key in {
            "panel_tracks": "panelTracks",
            "panelTracks": "panelTracks",
            "decorative_tracks": "decorativeTracks",
            "decorativeTracks": "decorativeTracks",
        }.items():
            if source_key in row:
                override[output_key] = list(_string_values(row[source_key]))
        for source_key, output_key in {
            "z_boost": "zBoost",
            "zBoost": "zBoost",
            "decorative_z_drop": "decorativeZDrop",
            "decorativeZDrop": "decorativeZDrop",
        }.items():
            value = row.get(source_key)
            if value is None:
                continue
            try:
                override[output_key] = float(value)
            except (TypeError, ValueError):
                pass
        if row.get("source"):
            override["source"] = row.get("source")
        overrides.append(override)
    return {
        "panelOutlineOnTop": config.layer_policy.panel_outline_on_top,
        "decorativeTracks": list(config.layer_policy.decorative_tracks),
        "transitionLayerOverrides": sorted(
            overrides,
            key=lambda item: (int(item["from"]), int(item["to"]), str(item.get("source", ""))),
        ),
    }


def _string_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def _media_phase_override_matches(obj: dict[str, Any], override: dict[str, Any]) -> bool:
    selectors = [
        ("object_id", "objectId", "id"),
        ("track_id", "trackId", "trackId"),
        ("asset_id", "assetId", "assetId"),
        ("name", "name", "name"),
    ]
    saw_selector = False
    for snake_key, camel_key, obj_key in selectors:
        value = _override_value(override, snake_key, camel_key)
        if value is None:
            continue
        saw_selector = True
        if str(obj.get(obj_key) or "") != str(value):
            return False
    return saw_selector


def _apply_raster_fallback_overrides(
    slides: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    overrides: tuple[dict[str, Any], ...],
    out_dir: Path,
    slide_w: float,
    slide_h: float,
    *,
    allow_full_slide_static: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not overrides:
        return {"appliedCount": 0, "rows": rows}
    slides_by_index = {int(slide["index"]): slide for slide in slides}
    assets_by_id = {str(asset.get("id")): asset for asset in assets}
    applied_count = 0
    for index, override in enumerate(overrides):
        row: dict[str, Any] = {"index": index, "status": "skipped"}
        try:
            slide_index = int(_override_value(override, "slide", "slideIndex", "slide_index"))
        except (TypeError, ValueError):
            row["status"] = "invalid-slide"
            rows.append(row)
            continue
        slide = slides_by_index.get(slide_index)
        if slide is None:
            row.update({"slide": slide_index, "status": "missing-slide"})
            rows.append(row)
            continue
        source_value = _override_value(override, "file", "image_file", "imageFile")
        if not source_value:
            row.update({"slide": slide_index, "status": "missing-file"})
            rows.append(row)
            continue
        if not allow_full_slide_static and _raster_fallback_is_full_slide_static(override, slide_w, slide_h):
            row.update(
                {
                    "slide": slide_index,
                    "status": "skipped-object-mapped-static",
                    "reason": "full-slide settled fallback would replace live object layers",
                }
            )
            rows.append(row)
            continue
        replace_ids = {str(value) for value in (_override_value(override, "replace", "replace_track_ids", "replaceTrackIds") or [])}
        replace_targets = _raster_fallback_replace_targets(slide, replace_ids, override, slide_w, slide_h)
        if replace_ids and _raster_fallback_requires_panel_target(override) and not replace_targets:
            row.update(
                {
                    "slide": slide_index,
                    "status": "stale-replace-target",
                    "replaceIds": sorted(replace_ids),
                }
            )
            rows.append(row)
            continue
        try:
            asset = _raster_fallback_asset(out_dir, Path(str(source_value)), assets_by_id)
        except OSError as exc:
            row.update({"slide": slide_index, "status": "file-error", "error": str(exc)})
            rows.append(row)
            continue
        if asset["id"] not in assets_by_id:
            assets.append(asset)
            assets_by_id[asset["id"]] = asset
        if replace_ids:
            replace_object_ids = {str(obj.get("id") or "") for obj in replace_targets}
            replace_track_ids = {str(obj.get("trackId") or "") for obj in replace_targets}
            slide["objects"] = [
                obj
                for obj in slide.get("objects", [])
                if str(obj.get("trackId") or "") not in replace_track_ids and str(obj.get("id") or "") not in replace_object_ids
            ]
        z = _raster_fallback_z(slide, override)
        object_id = str(
            _override_value(override, "object_id", "objectId", "id")
            or f"s{slide_index}-raster-fallback-{asset['id'].removeprefix('asset-raster-')[:8]}"
        )
        fallback_track_id = str(
            (replace_targets[0].get("trackId") if len(replace_targets) == 1 else None)
            or _override_value(override, "track_id", "trackId")
            or object_id
        )
        is_panel_border_fallback = _raster_fallback_requires_panel_target(override)
        scene_object = {
            "id": object_id,
            "trackId": fallback_track_id,
            "shapeId": None,
            "creationId": None,
            "name": str(_override_value(override, "name") or f"Raster fallback {slide_index}"),
            "groupPath": [],
            "kind": "image",
            "z": z,
            "geometry": _raster_fallback_geometry(override, slide_w, slide_h),
            "assetId": asset["id"],
            "posterAssetId": None,
            "text": "",
            "textStyle": {},
            "richText": [],
            "shape": "roundRect" if is_panel_border_fallback else "rect",
            "fill": None,
            "stroke": "scheme:bg1" if is_panel_border_fallback else None,
            "strokeWidth": None,
            "strokeWidthPct": (
                float(_override_value(override, "strokeWidthPct", "stroke_width_pct") or 0.011111111111111112)
                if is_panel_border_fallback
                else 0.0
            ),
            "opacity": float(_override_value(override, "opacity") or 1.0),
            "crop": None,
            "mediaTiming": {},
            "rasterFallback": {
                "source": str(_override_value(override, "source") or "config"),
                "settledOnly": bool(_override_value(override, "settled_only", "settledOnly") or False),
                "replaceTrackIds": sorted(replace_ids),
            },
            "provenance": {
                "source": "raster-fallback",
                "sourceFile": asset["file"],
                "configIndex": index,
            },
            "unsupported": [],
        }
        slide.setdefault("objects", []).append(scene_object)
        slide["objects"].sort(key=lambda obj: float(obj.get("z", 0) or 0))
        row.update(
            {
                "slide": slide_index,
                "status": "applied",
                "objectId": object_id,
                "trackId": scene_object["trackId"],
                "assetId": asset["id"],
                "file": asset["file"],
                "replaceCount": len(replace_targets) if replace_ids else 0,
                "settledOnly": scene_object["rasterFallback"]["settledOnly"],
            }
        )
        applied_count += 1
        rows.append(row)
    return {"appliedCount": applied_count, "rows": rows}


def _raster_fallback_is_full_slide_static(
    override: dict[str, Any],
    slide_w: float,
    slide_h: float,
) -> bool:
    source = str(_override_value(override, "source") or "").lower()
    settled_only = bool(_override_value(override, "settled_only", "settledOnly") or False)
    if source != "static-fallback" or not settled_only:
        return False
    geometry = _raster_fallback_geometry(override, slide_w, slide_h)
    return float(geometry.get("widthPct", 0.0) or 0.0) >= 0.9 and float(geometry.get("heightPct", 0.0) or 0.0) >= 0.9


def _raster_fallback_replace_targets(
    slide: dict[str, Any],
    replace_ids: set[str],
    override: dict[str, Any],
    slide_w: float,
    slide_h: float,
) -> list[dict[str, Any]]:
    if not replace_ids:
        return []
    candidates = [
        obj
        for obj in slide.get("objects", []) or []
        if str(obj.get("trackId") or "") in replace_ids or str(obj.get("id") or "") in replace_ids
    ]
    if not _raster_fallback_requires_panel_target(override):
        return candidates
    valid_targets = [
        obj
        for obj in candidates
        if _is_scene_panel_container(obj, slide_w, slide_h)
        and _raster_fallback_geometry_matches(obj.get("geometry") or {}, override, slide_w, slide_h)
    ]
    if valid_targets:
        return valid_targets
    geometry_targets = [
        obj
        for obj in slide.get("objects", []) or []
        if _is_scene_panel_container(obj, slide_w, slide_h)
        and _raster_fallback_geometry_matches(obj.get("geometry") or {}, override, slide_w, slide_h)
    ]
    return geometry_targets if len(geometry_targets) == 1 else []


def _raster_fallback_requires_panel_target(override: dict[str, Any]) -> bool:
    source = str(_override_value(override, "source") or "").lower()
    name = str(_override_value(override, "name") or "").lower()
    return source == "panel-border-fallback" or "panel border" in name


def _raster_fallback_geometry_matches(
    geometry: dict[str, Any],
    override: dict[str, Any],
    slide_w: float,
    slide_h: float,
) -> bool:
    expected = _override_value(override, "geometry")
    if not isinstance(expected, dict):
        return True
    width = max(float(expected.get("w", 0.0) or 0.0), float(geometry.get("w", 0.0) or 0.0), 1.0)
    height = max(float(expected.get("h", 0.0) or 0.0), float(geometry.get("h", 0.0) or 0.0), 1.0)
    x_tolerance = max(width * 0.03, slide_w * 0.01, 1.0)
    y_tolerance = max(height * 0.03, slide_h * 0.01, 1.0)
    size_tolerance = 0.04
    return (
        abs(float(geometry.get("x", 0.0) or 0.0) - float(expected.get("x", 0.0) or 0.0)) <= x_tolerance
        and abs(float(geometry.get("y", 0.0) or 0.0) - float(expected.get("y", 0.0) or 0.0)) <= y_tolerance
        and abs(float(geometry.get("w", 0.0) or 0.0) - float(expected.get("w", 0.0) or 0.0)) / width <= size_tolerance
        and abs(float(geometry.get("h", 0.0) or 0.0) - float(expected.get("h", 0.0) or 0.0)) / height <= size_tolerance
    )


def _raster_fallback_asset(
    out_dir: Path,
    source: Path,
    existing: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_path = source.expanduser()
    if not source_path.is_absolute():
        source_path = out_dir / source_path
    data = source_path.read_bytes()
    digest = sha256(data).hexdigest()
    asset_id = f"asset-raster-{digest[:16]}"
    if asset_id in existing:
        return existing[asset_id]
    extension = source_path.suffix.lower().lstrip(".") or "png"
    target_dir = out_dir / "assets" / "fallback"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{digest[:16]}.{extension}"
    if source_path.resolve() != target.resolve():
        shutil.copyfile(source_path, target)
    try:
        rel_file = target.relative_to(out_dir).as_posix()
    except ValueError:
        rel_file = target.as_posix()
    return {
        "id": asset_id,
        "sourcePath": str(source),
        "sourceFile": rel_file,
        "file": rel_file,
        "kind": "image",
        "extension": extension,
        "sizeBytes": len(data),
        "sha256": digest,
        "width": None,
        "height": None,
        "durationSec": None,
        "animated": False,
        "alpha": extension in {"png", "webp"},
        "warnings": ["raster-fallback"],
    }


def _raster_fallback_z(slide: dict[str, Any], override: dict[str, Any]) -> int:
    raw = _override_value(override, "z", "zIndex")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    values = [int(float(obj.get("z", 0) or 0)) for obj in slide.get("objects", [])]
    return (max(values) if values else 0) + 1


def _raster_fallback_geometry(
    override: dict[str, Any],
    slide_w: float,
    slide_h: float,
) -> dict[str, Any]:
    raw = _override_value(override, "geometry")
    if not isinstance(raw, dict):
        raw = {}
    left = _float_or(raw.get("leftPct"), 0.0)
    top = _float_or(raw.get("topPct"), 0.0)
    width = _float_or(raw.get("widthPct"), 1.0)
    height = _float_or(raw.get("heightPct"), 1.0)
    x = _float_or(raw.get("x"), left * slide_w)
    y = _float_or(raw.get("y"), top * slide_h)
    w = _float_or(raw.get("w"), width * slide_w)
    h = _float_or(raw.get("h"), height * slide_h)
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "leftPct": 0.0 if slide_w <= 0 else x / slide_w,
        "topPct": 0.0 if slide_h <= 0 else y / slide_h,
        "widthPct": 0.0 if slide_w <= 0 else w / slide_w,
        "heightPct": 0.0 if slide_h <= 0 else h / slide_h,
        "rotation": _float_or(raw.get("rotation"), 0.0),
        "flipH": bool(raw.get("flipH", False)),
        "flipV": bool(raw.get("flipV", False)),
    }


def _annotate_panel_relationships(
    slides: list[dict[str, Any]],
    slide_w: float,
    slide_h: float,
    *,
    infer_panels: bool = True,
    panel_border_on_top: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    applied_count = 0
    for slide in slides:
        slide_index = int(slide.get("index", 0) or 0)
        objects = [obj for obj in slide.get("objects", []) if isinstance(obj, dict)]
        panels = [obj for obj in objects if infer_panels and _is_scene_panel_container(obj, slide_w, slide_h)]
        for panel in panels:
            panel["panelRole"] = "container"
            panel["nodeRole"] = "panel"
            panel["childrenTrackIds"] = []
            panel["panelBorderOnTop"] = bool(panel_border_on_top)
            panel.setdefault("provenance", {})["panelRole"] = "container"
            rows.append(
                {
                    "slide": slide_index,
                    "status": "container",
                    "panelTrackId": panel.get("trackId"),
                    "panelObjectId": panel.get("id"),
                    "name": panel.get("name"),
                }
            )
        for obj in objects:
            if obj in panels:
                continue
            parent = _scene_panel_parent_for_object(obj, panels)
            if parent is None:
                continue
            panel_track_id = str(parent.get("trackId") or "")
            if not panel_track_id:
                continue
            obj["panelParentTrackId"] = panel_track_id
            obj["parentTrackId"] = panel_track_id
            obj["panelParentObjectId"] = parent.get("id")
            obj["localGeometry"] = _scene_local_geometry(
                obj.get("geometry") or {},
                parent.get("geometry") or {},
            )
            obj.setdefault("provenance", {})["panelParent"] = {
                "trackId": panel_track_id,
                "objectId": parent.get("id"),
                "source": "panel-containment",
            }
            applied_count += 1
            rows.append(
                {
                    "slide": slide_index,
                    "status": "child",
                    "childTrackId": obj.get("trackId"),
                    "childObjectId": obj.get("id"),
                    "panelTrackId": panel_track_id,
                    "panelObjectId": parent.get("id"),
                }
            )
        cluster_rows = _annotate_panel_clusters(slide_index, objects, panels)
        rows.extend(cluster_rows)
    return {"appliedCount": applied_count, "rows": rows}


def _annotate_panel_clusters(
    slide_index: int,
    objects: list[dict[str, Any]],
    panels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel in panels:
        panel_track_id = str(panel.get("trackId") or "")
        if not panel_track_id:
            continue
        children = [
            obj
            for obj in objects
            if str(obj.get("panelParentTrackId") or "") == panel_track_id
        ]
        if not children:
            continue
        cluster_max_z = max([_scene_z(panel), *[_scene_z(child) for child in children]])
        child_track_ids = [str(child.get("trackId") or "") for child in sorted(children, key=_scene_z)]
        panel["panelClusterId"] = panel_track_id
        panel["panelLayerRole"] = "frame"
        panel["panelClusterMaxZ"] = cluster_max_z
        panel["panelChildrenTrackIds"] = child_track_ids
        panel["childrenTrackIds"] = child_track_ids
        panel["renderZ"] = cluster_max_z
        panel.setdefault("provenance", {})["panelCluster"] = {
            "id": panel_track_id,
            "role": "frame",
            "children": child_track_ids,
        }
        for child in children:
            child["panelClusterId"] = panel_track_id
            child["panelLayerRole"] = "child"
            child["panelClusterMaxZ"] = cluster_max_z
            child["parentTrackId"] = panel_track_id
            child.setdefault("provenance", {})["panelCluster"] = {
                "id": panel_track_id,
                "role": "child",
                "frameTrackId": panel_track_id,
            }
        rows.append(
            {
                "slide": slide_index,
                "status": "cluster",
                "panelTrackId": panel_track_id,
                "panelObjectId": panel.get("id"),
                "childTrackIds": child_track_ids,
                "clusterMaxZ": cluster_max_z,
            }
        )
    return rows


def _scene_z(obj: dict[str, Any]) -> float:
    return _float_or(obj.get("z"), 0.0)


def _annotate_scene_graph_v2(
    slides: list[dict[str, Any]],
    slide_w: float,
    slide_h: float,
    *,
    explicit_groups: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    group_count = 0
    relationship_count = 0
    for slide in slides:
        slide_index = int(slide.get("index", 0) or 0)
        objects = sorted(
            [obj for obj in slide.get("objects", []) if isinstance(obj, dict)],
            key=_scene_z,
        )
        groups: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        seen_groups: set[str] = set()
        seen_relationships: set[tuple[str, str, str]] = set()
        group_nodes: list[dict[str, Any]] = []
        explicit_group_tracks: dict[tuple[str, ...], str] = {}
        explicit_group_geometries: dict[tuple[str, ...], dict[str, Any]] = {}

        if explicit_groups:
            explicit_group_members: dict[tuple[str, ...], list[dict[str, Any]]] = {}
            for obj in objects:
                group_path = obj.get("groupPath")
                if not isinstance(group_path, list) or not group_path:
                    continue
                normalized = tuple(str(item or "").strip() for item in group_path if str(item or "").strip())
                for depth in range(1, len(normalized) + 1):
                    explicit_group_members.setdefault(normalized[:depth], []).append(obj)
            explicit_group_geometries = {
                path: _scene_bounds_geometry(members, slide_w, slide_h)
                for path, members in explicit_group_members.items()
                if members
            }
            for obj in objects:
                group_path = obj.get("groupPath")
                if not isinstance(group_path, list) or not group_path:
                    continue
                parent_group_id: str | None = None
                parent_group_track: str | None = None
                normalized_path: list[str] = []
                for depth, raw_name in enumerate(group_path, start=1):
                    name = str(raw_name or f"Group {depth}")
                    normalized_path.append(name)
                    path_tuple = tuple(normalized_path)
                    group_id = _stable_group_id(slide_index, normalized_path)
                    group_track_id = _stable_group_track_id(normalized_path)
                    explicit_group_tracks[path_tuple] = group_track_id
                    if group_id not in seen_groups:
                        seen_groups.add(group_id)
                        geometry = explicit_group_geometries.get(path_tuple) or _empty_scene_geometry(slide_w, slide_h)
                        local_geometry = (
                            _scene_local_geometry(geometry, explicit_group_geometries[tuple(normalized_path[:-1])])
                            if len(normalized_path) > 1 and tuple(normalized_path[:-1]) in explicit_group_geometries
                            else None
                        )
                        group_node = {
                            "id": group_id,
                            "trackId": group_track_id,
                            "shapeId": None,
                            "creationId": None,
                            "name": name,
                            "kind": "group",
                            "nodeRole": "group",
                            "z": min((_scene_z(member) for member in explicit_group_members.get(path_tuple, [])), default=0.0) - 0.25,
                            "renderZ": min((_scene_z(member) for member in explicit_group_members.get(path_tuple, [])), default=0.0) - 0.25,
                            "geometry": geometry,
                            "assetId": None,
                            "posterAssetId": None,
                            "text": "",
                            "textStyle": {},
                            "richText": [],
                            "shape": None,
                            "fill": None,
                            "stroke": None,
                            "strokeWidth": None,
                            "strokeWidthPct": 0.0,
                            "opacity": 1.0,
                            "crop": None,
                            "mediaTiming": {},
                            "provenance": {
                                "source": "ppt-group-path",
                                "groupPath": list(normalized_path),
                            },
                            "unsupported": [],
                            "childrenTrackIds": [],
                        }
                        if parent_group_track and local_geometry:
                            group_node["parentTrackId"] = parent_group_track
                            group_node["parentId"] = parent_group_id
                            group_node["localGeometry"] = local_geometry
                        group_nodes.append(group_node)
                        groups.append(
                            {
                                "id": group_id,
                                "kind": "ppt-group-path",
                                "name": name,
                                "path": list(normalized_path),
                                "renderable": True,
                                "trackId": group_track_id,
                                "geometry": geometry,
                                "source": "ppt-group-path",
                            }
                        )
                        rows.append(
                            {
                                "slide": slide_index,
                                "status": "explicit-group-path",
                                "groupId": group_id,
                                "name": name,
                                "path": list(normalized_path),
                            }
                        )
                    if parent_group_id:
                        key = ("group-child", parent_group_id, group_id)
                        if key not in seen_relationships:
                            seen_relationships.add(key)
                            relationships.append(
                                {
                                    "type": "group-child",
                                    "parentId": parent_group_id,
                                    "parentTrackId": parent_group_track,
                                    "childId": group_id,
                                    "childTrackId": group_track_id,
                                    "source": "ppt-group-path",
                                }
                            )
                    parent_group_id = group_id
                    parent_group_track = group_track_id
                if parent_group_id:
                    obj["explicitGroupId"] = parent_group_id
                    obj["explicitGroupTrackId"] = parent_group_track
                    if not obj.get("parentTrackId") and parent_group_track:
                        obj["parentTrackId"] = parent_group_track
                        obj["parentId"] = parent_group_id
                        obj["localGeometry"] = _scene_local_geometry(
                            obj.get("geometry") or {},
                            explicit_group_geometries.get(tuple(normalized_path)) or _empty_scene_geometry(slide_w, slide_h),
                        )
                    key = ("group-member", parent_group_id, str(obj.get("trackId") or obj.get("id") or ""))
                    if key not in seen_relationships:
                        seen_relationships.add(key)
                        relationships.append(
                            {
                                "type": "group-member",
                                "parentId": parent_group_id,
                                "parentTrackId": parent_group_track,
                                "childTrackId": obj.get("trackId"),
                                "childObjectId": obj.get("id"),
                                "source": "ppt-group-path",
                            }
                        )
                    if parent_group_track:
                        for group_node in group_nodes:
                            if group_node.get("trackId") == parent_group_track:
                                child_track = str(obj.get("trackId") or "")
                                if child_track and child_track not in group_node["childrenTrackIds"]:
                                    group_node["childrenTrackIds"].append(child_track)
                                break

        for panel in objects:
            if panel.get("panelRole") != "container":
                continue
            panel_track_id = str(panel.get("trackId") or "")
            if not panel_track_id:
                continue
            children = [
                obj
                for obj in objects
                if str(obj.get("parentTrackId") or obj.get("panelParentTrackId") or "") == panel_track_id
            ]
            child_track_ids = [str(child.get("trackId") or "") for child in sorted(children, key=_scene_z)]
            group_id = f"panel-{panel_track_id}"
            if group_id not in seen_groups:
                seen_groups.add(group_id)
                groups.append(
                    {
                        "id": group_id,
                        "kind": "inferred-panel",
                        "renderable": True,
                        "trackId": panel_track_id,
                        "objectId": panel.get("id"),
                        "name": panel.get("name"),
                        "geometry": panel.get("geometry"),
                        "childrenTrackIds": child_track_ids,
                        "borderOnTop": bool(panel.get("panelBorderOnTop", True)),
                        "source": "panel-containment",
                    }
                )
                rows.append(
                    {
                        "slide": slide_index,
                        "status": "inferred-panel",
                        "groupId": group_id,
                        "panelTrackId": panel_track_id,
                        "childTrackIds": child_track_ids,
                    }
                )
            for child in children:
                child["parentId"] = group_id
                child["parentTrackId"] = panel_track_id
                if not isinstance(child.get("localGeometry"), dict):
                    child["localGeometry"] = _scene_local_geometry(
                        child.get("geometry") or {},
                        panel.get("geometry") or {},
                    )
                key = ("panel-contains", panel_track_id, str(child.get("trackId") or ""))
                if key not in seen_relationships:
                    seen_relationships.add(key)
                    relationships.append(
                        {
                            "type": "panel-contains",
                            "parentId": group_id,
                            "parentTrackId": panel_track_id,
                            "childTrackId": child.get("trackId"),
                            "childObjectId": child.get("id"),
                            "source": "panel-containment",
                        }
                    )
            if bool(panel.get("panelBorderOnTop", True)):
                relationships.append(
                    {
                        "type": "panel-border-overlay",
                        "parentId": group_id,
                        "trackId": panel_track_id,
                        "source": "panel-containment",
                    }
                )

        for group_node in group_nodes:
            parent_track = str(group_node.get("parentTrackId") or "")
            if not parent_track:
                continue
            for maybe_parent in group_nodes:
                if maybe_parent.get("trackId") == parent_track:
                    child_track = str(group_node.get("trackId") or "")
                    if child_track and child_track not in maybe_parent["childrenTrackIds"]:
                        maybe_parent["childrenTrackIds"].append(child_track)
                    break

        slide["nodes"] = sorted([*group_nodes, *objects], key=_scene_render_order)
        slide["groups"] = groups
        slide["relationships"] = relationships
        group_count += len(groups)
        relationship_count += len(relationships)

    return {
        "schema": "pptx-html-presenter.group-report.v1",
        "slideCount": len(slides),
        "groupCount": group_count,
        "relationshipCount": relationship_count,
        "rows": rows,
    }


def _stable_group_id(slide_index: int, group_path: list[str]) -> str:
    raw = f"{slide_index}|" + "|".join(group_path)
    digest = sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"s{slide_index}-group-{digest}"


def _stable_group_track_id(group_path: list[str]) -> str:
    raw = "|".join(group_path)
    digest = sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"track-group-{digest}"


def _scene_render_order(obj: dict[str, Any]) -> tuple[int, float, str]:
    role = str(obj.get("nodeRole") or "")
    rank = 0 if role == "group" else 1
    if role == "panel":
        rank = 1
    return (rank, _float_or(obj.get("renderZ", obj.get("z")), 0.0), str(obj.get("trackId") or obj.get("id") or ""))


def _scene_bounds_geometry(
    objects: list[dict[str, Any]],
    slide_w: float,
    slide_h: float,
) -> dict[str, Any]:
    geometries = [obj.get("geometry") or {} for obj in objects if isinstance(obj.get("geometry"), dict)]
    if not geometries:
        return _empty_scene_geometry(slide_w, slide_h)
    left = min(float(geometry.get("x", 0.0) or 0.0) for geometry in geometries)
    top = min(float(geometry.get("y", 0.0) or 0.0) for geometry in geometries)
    right = max(float(geometry.get("x", 0.0) or 0.0) + float(geometry.get("w", 0.0) or 0.0) for geometry in geometries)
    bottom = max(float(geometry.get("y", 0.0) or 0.0) + float(geometry.get("h", 0.0) or 0.0) for geometry in geometries)
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    return {
        "x": left,
        "y": top,
        "w": width,
        "h": height,
        "leftPct": 0.0 if slide_w <= 0 else left / slide_w,
        "topPct": 0.0 if slide_h <= 0 else top / slide_h,
        "widthPct": 0.0 if slide_w <= 0 else width / slide_w,
        "heightPct": 0.0 if slide_h <= 0 else height / slide_h,
        "rotation": 0.0,
        "flipH": False,
        "flipV": False,
    }


def _empty_scene_geometry(slide_w: float, slide_h: float) -> dict[str, Any]:
    return {
        "x": 0.0,
        "y": 0.0,
        "w": 1.0,
        "h": 1.0,
        "leftPct": 0.0,
        "topPct": 0.0,
        "widthPct": 0.0 if slide_w <= 0 else 1.0 / slide_w,
        "heightPct": 0.0 if slide_h <= 0 else 1.0 / slide_h,
        "rotation": 0.0,
        "flipH": False,
        "flipV": False,
    }


def _scene_local_geometry(
    geometry: dict[str, Any],
    parent_geometry: dict[str, Any],
) -> dict[str, Any]:
    parent_x = float(parent_geometry.get("x", 0.0) or 0.0)
    parent_y = float(parent_geometry.get("y", 0.0) or 0.0)
    parent_w = max(float(parent_geometry.get("w", 0.0) or 0.0), 1.0)
    parent_h = max(float(parent_geometry.get("h", 0.0) or 0.0), 1.0)
    x = float(geometry.get("x", 0.0) or 0.0) - parent_x
    y = float(geometry.get("y", 0.0) or 0.0) - parent_y
    w = float(geometry.get("w", 0.0) or 0.0)
    h = float(geometry.get("h", 0.0) or 0.0)
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "leftPct": x / parent_w,
        "topPct": y / parent_h,
        "widthPct": w / parent_w,
        "heightPct": h / parent_h,
        "rotation": _float_or(geometry.get("rotation"), 0.0),
        "flipH": bool(geometry.get("flipH", False)),
        "flipV": bool(geometry.get("flipV", False)),
    }


def _apply_panel_relationships_to_transitions(
    transitions: list[dict[str, Any]],
    slides: list[dict[str, Any]],
    slide_w: float,
    slide_h: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    applied_count = 0
    slides_by_index = {int(slide.get("index", 0) or 0): slide for slide in slides}
    for transition in transitions:
        from_slide = slides_by_index.get(int(transition.get("from", 0) or 0))
        to_slide = slides_by_index.get(int(transition.get("to", 0) or 0))
        if not from_slide or not to_slide:
            continue
        from_by_track = _scene_objects_by_track(from_slide)
        to_by_track = _scene_objects_by_track(to_slide)
        inferred = [row for row in transition.get("inferredMotions", []) or [] if isinstance(row, dict)]
        inferred_by_track = {str(row.get("trackId") or ""): row for row in inferred if row.get("trackId")}
        panel_deltas = _scene_panel_transition_deltas(from_by_track, to_by_track, inferred)
        if not panel_deltas:
            continue
        enter_tracks = {str(track) for track in transition.get("enterTrackIds", []) or []}
        exit_tracks = {str(track) for track in transition.get("exitTrackIds", []) or []}
        for track_id in sorted(enter_tracks | exit_tracks):
            obj = to_by_track.get(track_id) or from_by_track.get(track_id)
            if not obj:
                continue
            panel_track_id = str(obj.get("panelParentTrackId") or "")
            if not panel_track_id or panel_track_id not in panel_deltas:
                continue
            delta = panel_deltas[panel_track_id]
            existing = inferred_by_track.get(track_id)
            endpoint = to_by_track.get(track_id) if track_id in enter_tracks else from_by_track.get(track_id)
            if not endpoint:
                continue
            if existing is None:
                existing = {
                    "trackId": track_id,
                    "durationSec": transition.get("durationSec", 0.0),
                }
                inferred.append(existing)
                inferred_by_track[track_id] = existing
            if track_id in enter_tracks:
                existing["fromGeometry"] = _shifted_scene_geometry_dict(endpoint.get("geometry") or {}, -delta[0], -delta[1], slide_w, slide_h)
                existing["toGeometry"] = dict(endpoint.get("geometry") or {})
            else:
                existing["fromGeometry"] = dict(endpoint.get("geometry") or {})
                existing["toGeometry"] = _shifted_scene_geometry_dict(endpoint.get("geometry") or {}, delta[0], delta[1], slide_w, slide_h)
            existing["panelTrackId"] = panel_track_id
            existing["preserveOpacity"] = True
            existing["source"] = "inferred-panel-parent-motion"
            applied_count += 1
            rows.append(
                {
                    "from": transition.get("from"),
                    "to": transition.get("to"),
                    "trackId": track_id,
                    "panelTrackId": panel_track_id,
                    "status": "applied",
                }
            )
        if inferred:
            transition["inferredMotions"] = sorted(inferred, key=lambda row: str(row.get("trackId") or ""))
    return {"appliedCount": applied_count, "rows": rows}


def _scene_objects_by_track(slide: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(obj.get("trackId") or ""): obj
        for obj in slide.get("objects", []) or []
        if isinstance(obj, dict) and obj.get("trackId") and not (obj.get("rasterFallback") or {}).get("settledOnly")
    }


def _scene_panel_transition_deltas(
    from_by_track: dict[str, dict[str, Any]],
    to_by_track: dict[str, dict[str, Any]],
    inferred_rows: list[dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    deltas: dict[str, tuple[float, float]] = {}
    for track_id, left in from_by_track.items():
        right = to_by_track.get(track_id)
        if right and _scene_panelish(left) and _scene_panelish(right):
            deltas[track_id] = _scene_geometry_delta(left.get("geometry") or {}, right.get("geometry") or {})
    for row in inferred_rows:
        track_id = str(row.get("trackId") or "")
        if not track_id:
            continue
        source = from_by_track.get(track_id) or to_by_track.get(track_id)
        if not source or not _scene_panelish(source):
            continue
        from_geometry = row.get("fromGeometry") or {}
        to_geometry = row.get("toGeometry") or {}
        if from_geometry and to_geometry:
            deltas[track_id] = _scene_geometry_delta(from_geometry, to_geometry)
    return deltas


def _scene_geometry_delta(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, float]:
    return (
        float(right.get("x", 0.0) or 0.0) - float(left.get("x", 0.0) or 0.0),
        float(right.get("y", 0.0) or 0.0) - float(left.get("y", 0.0) or 0.0),
    )


def _scene_panelish(obj: dict[str, Any]) -> bool:
    return bool(obj.get("panelRole") == "container" or str(obj.get("name") or "").lower().startswith("powerpoint panel border"))


def _shifted_scene_geometry_dict(
    geometry: dict[str, Any],
    dx: float,
    dy: float,
    slide_w: float,
    slide_h: float,
) -> dict[str, Any]:
    out = dict(geometry)
    x = float(out.get("x", 0.0) or 0.0) + dx
    y = float(out.get("y", 0.0) or 0.0) + dy
    out["x"] = x
    out["y"] = y
    out["leftPct"] = 0.0 if slide_w <= 0 else x / slide_w
    out["topPct"] = 0.0 if slide_h <= 0 else y / slide_h
    return out


def _is_scene_panel_container(obj: dict[str, Any], slide_w: float, slide_h: float) -> bool:
    geometry = obj.get("geometry") or {}
    width = float(geometry.get("w", 0.0) or 0.0)
    height = float(geometry.get("h", 0.0) or 0.0)
    if width < slide_w * 0.35 or height < slide_h * 0.35:
        return False
    name = str(obj.get("name") or "").lower()
    if name.startswith("powerpoint panel border"):
        return True
    if str(obj.get("kind") or "") != "shape":
        return False
    shape = str(obj.get("shape") or "").lower()
    return "round" in shape and not obj.get("assetId") and not obj.get("text")


def _scene_panel_parent_for_object(
    obj: dict[str, Any],
    panels: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_area = math.inf
    obj_geometry = obj.get("geometry") or {}
    obj_width = float(obj_geometry.get("w", 0.0) or 0.0)
    obj_height = float(obj_geometry.get("h", 0.0) or 0.0)
    for panel in panels:
        if str(panel.get("trackId") or "") == str(obj.get("trackId") or ""):
            continue
        panel_geometry = panel.get("geometry") or {}
        panel_width = float(panel_geometry.get("w", 0.0) or 0.0)
        panel_height = float(panel_geometry.get("h", 0.0) or 0.0)
        if obj_width > panel_width * 1.12 or obj_height > panel_height * 1.12:
            continue
        if not _scene_contains_center(panel, obj):
            continue
        area = panel_width * panel_height
        if area < best_area:
            best = panel
            best_area = area
    return best


def _scene_contains_center(container: dict[str, Any], child: dict[str, Any]) -> bool:
    container_geometry = container.get("geometry") or {}
    child_geometry = child.get("geometry") or {}
    x = float(container_geometry.get("x", 0.0) or 0.0)
    y = float(container_geometry.get("y", 0.0) or 0.0)
    width = float(container_geometry.get("w", 0.0) or 0.0)
    height = float(container_geometry.get("h", 0.0) or 0.0)
    margin_x = max(width * 0.04, 1.0)
    margin_y = max(height * 0.04, 1.0)
    center_x = float(child_geometry.get("x", 0.0) or 0.0) + float(child_geometry.get("w", 0.0) or 0.0) / 2.0
    center_y = float(child_geometry.get("y", 0.0) or 0.0) + float(child_geometry.get("h", 0.0) or 0.0) / 2.0
    return x - margin_x <= center_x <= x + width + margin_x and y - margin_y <= center_y <= y + height + margin_y


def _float_or(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _override_value(override: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in override:
            return override[key]
    return None


def inspect_report(deck: PptxDeck) -> dict[str, Any]:
    media_by_ext: dict[str, dict[str, Any]] = {}
    media_usage: dict[str, set[int]] = {}
    unsupported: list[dict[str, Any]] = []
    for slide in deck.slides:
        for obj in slide.objects:
            if obj.asset_id:
                media_usage.setdefault(obj.asset_id, set()).add(slide.index)
            if obj.unsupported:
                unsupported.append(
                    {
                        "slide": slide.index,
                        "object": obj.id,
                        "name": obj.name,
                        "unsupported": obj.unsupported,
                    }
                )
    for asset in deck.assets.values():
        row = media_by_ext.setdefault(
            asset.extension,
            {"count": 0, "bytes": 0, "kind": asset.kind},
        )
        row["count"] += 1
        row["bytes"] += asset.size_bytes
    return {
        **deck.summary(),
        "mediaByExtension": media_by_ext,
        "topAssetReuse": [
            {"assetId": asset_id, "slideCount": len(slides), "slides": sorted(slides)}
            for asset_id, slides in sorted(
                media_usage.items(), key=lambda item: (-len(item[1]), item[0])
            )[:25]
        ],
        "unsupportedObjects": unsupported,
        "slides": [
            {
                "index": slide.index,
                "transition": slide.transition.to_scene(),
                "objectCount": len(slide.objects),
                "mediaObjectCount": len([o for o in slide.objects if o.asset_id]),
            }
            for slide in deck.slides
        ],
    }


def _assign_tracks(deck: PptxDeck, threshold: float) -> None:
    track_counter = 1
    if not deck.slides:
        return
    for obj in deck.slides[0].objects:
        obj.track_id = f"track-{track_counter:04d}"
        track_counter += 1
    for prev, current in zip(deck.slides, deck.slides[1:]):
        matches = _match_objects(prev.objects, current.objects, threshold)
        matched_to = {to_id for _, to_id, _ in matches}
        prev_by_id = {obj.id: obj for obj in prev.objects}
        for from_id, to_id, _score in matches:
            target = next(obj for obj in current.objects if obj.id == to_id)
            target.track_id = prev_by_id[from_id].track_id
        for obj in current.objects:
            if obj.id not in matched_to:
                obj.track_id = f"track-{track_counter:04d}"
                track_counter += 1


def _compile_transitions(deck: PptxDeck, config: PresenterConfig) -> list[dict[str, Any]]:
    out = []
    fade_overrides = _transition_unmatched_fade_overrides(
        config.morph_policy.transition_unmatched_fade_overrides
    )
    easing_overrides = _transition_easing_overrides(
        config.morph_policy.transition_easing_overrides
    )
    progress_overrides = _transition_progress_overrides(
        config.morph_policy.transition_progress_overrides
    )
    track_progress_overrides = _transition_track_progress_overrides(
        config.morph_policy.transition_track_progress_overrides
    )
    media_phase_overrides = _transition_media_phase_overrides(
        config.transition_media_phase_overrides
    )
    for prev, current in zip(deck.slides, deck.slides[1:]):
        duration = current.transition.duration_sec
        if current.transition.kind == "none" and duration <= 0:
            duration = 0.0
        elif current.transition.kind == "morph" and duration <= 0:
            duration = config.morph_policy.duration_default_sec
        elif duration <= 0:
            duration = config.morph_policy.duration_default_sec
        prev_by_track = by_track(prev)
        matches = [
            {
                "trackId": obj.track_id,
                "fromObjectId": prev_by_track.get(obj.track_id, {}).get("id"),
                "toObjectId": obj.id,
                "motion": _motion_metrics(prev_by_track[obj.track_id]["object"], obj, duration),
            }
            for obj in current.objects
            if obj.track_id in prev_by_track
        ]
        prev_tracks = {obj.track_id for obj in prev.objects}
        current_tracks = {obj.track_id for obj in current.objects}
        transition = {
            "from": prev.index,
            "to": current.index,
            "kind": current.transition.kind,
            "durationSec": duration,
            "matches": matches,
            "enterTrackIds": sorted(current_tracks - prev_tracks),
            "exitTrackIds": sorted(prev_tracks - current_tracks),
        }
        inferred_motions = _inferred_panel_motions(
            prev,
            current,
            prev_tracks,
            current_tracks,
            duration,
            deck.slide_width,
            deck.slide_height,
        )
        if inferred_motions:
            transition["inferredMotions"] = inferred_motions
        fade_override = fade_overrides.get((prev.index, current.index))
        if fade_override:
            transition["unmatchedFade"] = fade_override
        easing_override = easing_overrides.get((prev.index, current.index))
        if easing_override:
            transition["easing"] = easing_override
        progress_override = progress_overrides.get((prev.index, current.index))
        if progress_override:
            transition["progressMap"] = progress_override
        track_progress_override = track_progress_overrides.get((prev.index, current.index))
        if track_progress_override:
            transition["trackProgressOverrides"] = track_progress_override
        media_phase_override = media_phase_overrides.get((prev.index, current.index))
        if media_phase_override:
            transition["mediaPhaseOverrides"] = media_phase_override
        out.append(transition)
    return out


def _inferred_panel_motions(
    prev: Slide,
    current: Slide,
    prev_tracks: set[str | None],
    current_tracks: set[str | None],
    duration: float,
    slide_w: float,
    slide_h: float,
) -> list[dict[str, Any]]:
    prev_by_track = {obj.track_id: obj for obj in prev.objects if obj.track_id}
    current_by_track = {obj.track_id: obj for obj in current.objects if obj.track_id}
    deltas: list[tuple[float, float]] = []
    panel_deltas: dict[str, tuple[float, float]] = {}
    for track_id, left in prev_by_track.items():
        right = current_by_track.get(track_id)
        if not right:
            continue
        if _is_panel_container(left, slide_w, slide_h) and _is_panel_container(right, slide_w, slide_h):
            delta = (right.geometry.x - left.geometry.x, right.geometry.y - left.geometry.y)
            deltas.append(delta)
            panel_deltas[str(track_id)] = delta
    if not deltas:
        return []
    dx = _median_float([delta[0] for delta in deltas])
    dy = _median_float([delta[1] for delta in deltas])
    if abs(dx) < slide_w * 0.08 and abs(dy) < slide_h * 0.08:
        return []

    motions: list[dict[str, Any]] = []
    enter_tracks = current_tracks - prev_tracks
    exit_tracks = prev_tracks - current_tracks
    for obj in current.objects:
        if obj.track_id not in enter_tracks or not _is_panel_related(obj, current.objects, slide_w, slide_h):
            continue
        object_delta = _panel_delta_for_object(obj, current.objects, panel_deltas, slide_w, slide_h)
        object_dx, object_dy = object_delta[0] if object_delta else (dx, dy)
        from_geometry = _shifted_geometry_scene(obj, -object_dx, -object_dy, slide_w, slide_h)
        to_geometry = obj.geometry.to_scene(slide_w, slide_h)
        row: dict[str, Any] = {
            "trackId": obj.track_id,
            "fromGeometry": from_geometry,
            "toGeometry": to_geometry,
            "durationSec": duration,
            "preserveOpacity": True,
            "source": "inferred-panel-motion",
        }
        if object_delta and object_delta[1]:
            row["panelTrackId"] = object_delta[1]
        motions.append(row)
    for obj in prev.objects:
        if obj.track_id not in exit_tracks or not _is_panel_related(obj, prev.objects, slide_w, slide_h):
            continue
        from_geometry = obj.geometry.to_scene(slide_w, slide_h)
        object_delta = _panel_delta_for_object(obj, prev.objects, panel_deltas, slide_w, slide_h)
        object_dx, object_dy = object_delta[0] if object_delta else (dx, dy)
        to_geometry = _shifted_geometry_scene(obj, object_dx, object_dy, slide_w, slide_h)
        row = {
            "trackId": obj.track_id,
            "fromGeometry": from_geometry,
            "toGeometry": to_geometry,
            "durationSec": duration,
            "preserveOpacity": True,
            "source": "inferred-panel-motion",
        }
        if object_delta and object_delta[1]:
            row["panelTrackId"] = object_delta[1]
        motions.append(row)
    motions.extend(
        _inferred_carousel_foreground_motions(
            prev,
            current,
            enter_tracks,
            exit_tracks,
            duration,
            slide_w,
            slide_h,
            dx,
            dy,
            {str(row.get("trackId") or "") for row in motions},
        )
    )
    return sorted(motions, key=lambda row: str(row.get("trackId") or ""))


def _inferred_carousel_foreground_motions(
    prev: Slide,
    current: Slide,
    enter_tracks: set[str | None],
    exit_tracks: set[str | None],
    duration: float,
    slide_w: float,
    slide_h: float,
    dx: float,
    dy: float,
    existing_tracks: set[str],
) -> list[dict[str, Any]]:
    motions: list[dict[str, Any]] = []
    for obj in current.objects:
        if (
            obj.track_id not in enter_tracks
            or str(obj.track_id or "") in existing_tracks
            or _is_panel_related(obj, current.objects, slide_w, slide_h)
            or not _is_carousel_foreground_object(obj, slide_w, slide_h)
        ):
            continue
        motions.append(
            {
                "trackId": obj.track_id,
                "fromGeometry": _shifted_geometry_scene(obj, -dx, -dy, slide_w, slide_h),
                "toGeometry": obj.geometry.to_scene(slide_w, slide_h),
                "durationSec": duration,
                "preserveOpacity": True,
                "source": "inferred-carousel-foreground-motion",
            }
        )
    for obj in prev.objects:
        if (
            obj.track_id not in exit_tracks
            or str(obj.track_id or "") in existing_tracks
            or _is_panel_related(obj, prev.objects, slide_w, slide_h)
            or not _is_carousel_foreground_object(obj, slide_w, slide_h)
        ):
            continue
        motions.append(
            {
                "trackId": obj.track_id,
                "fromGeometry": obj.geometry.to_scene(slide_w, slide_h),
                "toGeometry": _shifted_geometry_scene(obj, dx, dy, slide_w, slide_h),
                "durationSec": duration,
                "preserveOpacity": True,
                "source": "inferred-carousel-foreground-motion",
            }
        )
    return motions


def _is_carousel_foreground_object(obj: SceneObject, slide_w: float, slide_h: float) -> bool:
    if not obj.track_id or not obj.asset_id:
        return False
    if obj.kind not in {"image", "video", "svg"}:
        return False
    if not _object_intersects_slide(obj, slide_w, slide_h):
        return False
    width = obj.geometry.cx
    height = obj.geometry.cy
    if width >= slide_w * 0.92 and height >= slide_h * 0.92:
        return False
    area = width * height
    slide_area = max(slide_w * slide_h, 1.0)
    return area >= slide_area * 0.025


def _object_intersects_slide(obj: SceneObject, slide_w: float, slide_h: float) -> bool:
    return (
        obj.geometry.x < slide_w
        and obj.geometry.y < slide_h
        and obj.geometry.x + obj.geometry.cx > 0
        and obj.geometry.y + obj.geometry.cy > 0
    )


def _panel_delta_for_object(
    obj: SceneObject,
    objects: list[SceneObject],
    panel_deltas: dict[str, tuple[float, float]],
    slide_w: float,
    slide_h: float,
) -> tuple[tuple[float, float], str | None] | None:
    best: tuple[SceneObject, str] | None = None
    for panel in objects:
        track_id = str(panel.track_id or "")
        if panel.id == obj.id or track_id not in panel_deltas or not _is_panel_container(panel, slide_w, slide_h):
            continue
        if obj.geometry.cx > panel.geometry.cx * 1.08 or obj.geometry.cy > panel.geometry.cy * 1.08:
            continue
        if not _contains_center(panel, obj):
            continue
        if best is None or panel.geometry.cx * panel.geometry.cy < best[0].geometry.cx * best[0].geometry.cy:
            best = (panel, track_id)
    if best is None:
        return None
    return panel_deltas[best[1]], best[1]


def _shifted_geometry_scene(
    obj: SceneObject,
    dx: float,
    dy: float,
    slide_w: float,
    slide_h: float,
) -> dict[str, Any]:
    return {
        "x": obj.geometry.x + dx,
        "y": obj.geometry.y + dy,
        "w": obj.geometry.cx,
        "h": obj.geometry.cy,
        "leftPct": 0.0 if slide_w <= 0 else (obj.geometry.x + dx) / slide_w,
        "topPct": 0.0 if slide_h <= 0 else (obj.geometry.y + dy) / slide_h,
        "widthPct": 0.0 if slide_w <= 0 else obj.geometry.cx / slide_w,
        "heightPct": 0.0 if slide_h <= 0 else obj.geometry.cy / slide_h,
        "rotation": obj.geometry.rotation,
        "flipH": obj.geometry.flip_h,
        "flipV": obj.geometry.flip_v,
    }


def _transition_unmatched_fade_overrides(
    overrides: tuple[dict[str, Any], ...],
) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in overrides:
        try:
            from_slide = int(_override_value(row, "from", "from_slide", "fromSlide"))
            to_slide = int(_override_value(row, "to", "to_slide", "toSlide"))
        except (TypeError, ValueError):
            continue
        fade: dict[str, Any] = {}
        field_map = {
            "enterStart": ("enter_start", "enterStart"),
            "enterEnd": ("enter_end", "enterEnd"),
            "exitStart": ("exit_start", "exitStart"),
            "exitEnd": ("exit_end", "exitEnd"),
        }
        for output_key, keys in field_map.items():
            value = _override_value(row, *keys)
            if value is None:
                continue
            try:
                fade[output_key] = round(_clamp01(float(value)), 3)
            except (TypeError, ValueError):
                continue
        if fade:
            if row.get("source"):
                fade["source"] = row.get("source")
            out[(from_slide, to_slide)] = fade
    return out


def _transition_easing_overrides(
    overrides: tuple[dict[str, Any], ...],
) -> dict[tuple[int, int], str | dict[str, Any]]:
    out: dict[tuple[int, int], str | dict[str, Any]] = {}
    for row in overrides:
        try:
            from_slide = int(_override_value(row, "from", "from_slide", "fromSlide"))
            to_slide = int(_override_value(row, "to", "to_slide", "toSlide"))
        except (TypeError, ValueError):
            continue
        easing = _override_value(row, "easing", "curve")
        if easing is None:
            continue
        if isinstance(easing, str):
            easing_value: str | dict[str, Any] = easing.strip()
            if not easing_value:
                continue
        elif isinstance(easing, dict):
            easing_value = dict(easing)
        else:
            continue
        out[(from_slide, to_slide)] = easing_value
    return out


def _transition_progress_overrides(
    overrides: tuple[dict[str, Any], ...],
) -> dict[tuple[int, int], list[dict[str, float]]]:
    out: dict[tuple[int, int], list[dict[str, float]]] = {}
    for row in overrides:
        try:
            from_slide = int(_override_value(row, "from", "from_slide", "fromSlide"))
            to_slide = int(_override_value(row, "to", "to_slide", "toSlide"))
        except (TypeError, ValueError):
            continue
        raw_points = _override_value(row, "points", "progress_map", "progressMap")
        if not isinstance(raw_points, list):
            continue
        points: list[dict[str, float]] = []
        for point in raw_points:
            if not isinstance(point, dict):
                continue
            try:
                progress = _clamp01(float(_override_value(point, "progress", "raw")))
                value = _clamp01(
                    float(
                        _override_value(
                            point,
                            "value",
                            "mapped_progress",
                            "mappedProgress",
                            "interpolation_progress",
                            "interpolationProgress",
                        )
                    )
                )
            except (TypeError, ValueError):
                continue
            points.append({"progress": round(progress, 4), "value": round(value, 4)})
        if len(points) >= 2:
            out[(from_slide, to_slide)] = sorted(points, key=lambda item: item["progress"])
    return out


def _transition_track_progress_overrides(
    overrides: tuple[dict[str, Any], ...],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    out: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in overrides:
        try:
            from_slide = int(_override_value(row, "from", "from_slide", "fromSlide"))
            to_slide = int(_override_value(row, "to", "to_slide", "toSlide"))
        except (TypeError, ValueError):
            continue
        track_id = _override_value(row, "track_id", "trackId")
        if not track_id:
            continue
        raw_points = _override_value(row, "points", "progress_map", "progressMap")
        if not isinstance(raw_points, list):
            continue
        points: list[dict[str, float]] = []
        for point in raw_points:
            if not isinstance(point, dict):
                continue
            try:
                progress = _clamp01(float(_override_value(point, "progress", "raw")))
                value = _clamp01(
                    float(
                        _override_value(
                            point,
                            "value",
                            "mapped_progress",
                            "mappedProgress",
                            "interpolation_progress",
                            "interpolationProgress",
                        )
                    )
                )
            except (TypeError, ValueError):
                continue
            points.append({"progress": round(progress, 4), "value": round(value, 4)})
        if len(points) < 2:
            continue
        normalized: dict[str, Any] = {
            "trackId": str(track_id),
            "points": sorted(points, key=lambda item: item["progress"]),
        }
        for output_key, keys in {
            "source": ("source",),
            "score": ("score",),
        }.items():
            value = _override_value(row, *keys)
            if value is not None:
                normalized[output_key] = value
        out.setdefault((from_slide, to_slide), []).append(normalized)
    for key in list(out):
        out[key] = sorted(out[key], key=lambda item: str(item.get("trackId", "")))
    return out


def _transition_media_phase_overrides(
    overrides: tuple[dict[str, Any], ...],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    out: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in overrides:
        try:
            from_slide = int(_override_value(row, "from", "from_slide", "fromSlide"))
            to_slide = int(_override_value(row, "to", "to_slide", "toSlide"))
            phase_sec = float(_override_value(row, "phase_sec", "phaseSec"))
        except (TypeError, ValueError):
            continue
        normalized: dict[str, Any] = {"phaseSec": round(phase_sec, 3)}
        field_map = {
            "trackId": ("track_id", "trackId"),
            "objectId": ("object_id", "objectId"),
            "assetId": ("asset_id", "assetId"),
            "name": ("name",),
            "source": ("source",),
            "score": ("score",),
        }
        for output_key, keys in field_map.items():
            value = _override_value(row, *keys)
            if value is not None:
                normalized[output_key] = value
        if len(normalized) <= 1:
            continue
        out.setdefault((from_slide, to_slide), []).append(normalized)
    for key in list(out):
        out[key] = sorted(
            out[key],
            key=lambda item: (
                str(item.get("trackId", "")),
                str(item.get("objectId", "")),
                str(item.get("assetId", "")),
                str(item.get("name", "")),
            ),
        )
    return out


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _motion_metrics(left: SceneObject, right: SceneObject, duration: float) -> dict[str, Any]:
    center_dx = (right.geometry.x + right.geometry.cx / 2.0) - (left.geometry.x + left.geometry.cx / 2.0)
    center_dy = (right.geometry.y + right.geometry.cy / 2.0) - (left.geometry.y + left.geometry.cy / 2.0)
    center_distance = math.sqrt(center_dx**2 + center_dy**2)
    return {
        "from": _geometry_snapshot(left),
        "to": _geometry_snapshot(right),
        "delta": {
            "x": round(right.geometry.x - left.geometry.x, 3),
            "y": round(right.geometry.y - left.geometry.y, 3),
            "w": round(right.geometry.cx - left.geometry.cx, 3),
            "h": round(right.geometry.cy - left.geometry.cy, 3),
            "rotation": round(right.geometry.rotation - left.geometry.rotation, 3),
            "centerX": round(center_dx, 3),
            "centerY": round(center_dy, 3),
        },
        "durationSec": duration,
        "centerDistance": round(center_distance, 3),
        "centerVelocityPerSec": round(center_distance / duration, 3) if duration > 0 else 0.0,
    }


def _geometry_snapshot(obj: SceneObject) -> dict[str, float]:
    return {
        "x": round(obj.geometry.x, 3),
        "y": round(obj.geometry.y, 3),
        "w": round(obj.geometry.cx, 3),
        "h": round(obj.geometry.cy, 3),
        "rotation": round(obj.geometry.rotation, 3),
    }


def by_track(slide: Slide) -> dict[str | None, dict[str, Any]]:
    return {obj.track_id: {"id": obj.id, "object": obj} for obj in slide.objects}


def _match_objects(
    prev_objects: list[SceneObject], current_objects: list[SceneObject], threshold: float
) -> list[tuple[str, str, float]]:
    candidates: list[tuple[float, SceneObject, SceneObject]] = []
    for left in prev_objects:
        for right in current_objects:
            if _is_container_shape(left) and _is_container_shape(right):
                continue
            score = _match_score(left, right)
            if score >= threshold:
                candidates.append((score, left, right))
    matches = _greedy_matches(candidates)
    matches = _extend_container_matches(prev_objects, current_objects, matches, threshold)
    return _extend_remaining_container_matches(prev_objects, current_objects, matches, threshold)


def _greedy_matches(
    candidates: list[tuple[float, SceneObject, SceneObject]],
    *,
    used_left: set[str] | None = None,
    used_right: set[str] | None = None,
) -> list[tuple[str, str, float]]:
    candidates.sort(key=lambda item: (-item[0], item[1].id, item[2].id))
    used_left = set(used_left or set())
    used_right = set(used_right or set())
    matches: list[tuple[str, str, float]] = []
    for score, left, right in candidates:
        if left.id in used_left or right.id in used_right:
            continue
        used_left.add(left.id)
        used_right.add(right.id)
        matches.append((left.id, right.id, round(score, 3)))
    return matches


def _extend_container_matches(
    prev_objects: list[SceneObject],
    current_objects: list[SceneObject],
    matches: list[tuple[str, str, float]],
    threshold: float,
) -> list[tuple[str, str, float]]:
    if not matches:
        return matches
    prev_by_id = {obj.id: obj for obj in prev_objects}
    current_by_id = {obj.id: obj for obj in current_objects}
    used_left = {left_id for left_id, _right_id, _score in matches}
    used_right = {right_id for _left_id, right_id, _score in matches}
    anchors = [
        (prev_by_id[left_id], current_by_id[right_id])
        for left_id, right_id, _score in matches
        if left_id in prev_by_id and right_id in current_by_id
    ]
    candidates: list[tuple[float, SceneObject, SceneObject]] = []
    for left in prev_objects:
        if left.id in used_left or not _is_container_shape(left):
            continue
        for right in current_objects:
            if right.id in used_right or not _is_container_shape(right):
                continue
            score = _container_match_score(left, right, anchors)
            if score >= threshold:
                candidates.append((score, left, right))
    if not candidates:
        return matches
    return [
        *matches,
        *_greedy_matches(candidates, used_left=used_left, used_right=used_right),
    ]


def _extend_remaining_container_matches(
    prev_objects: list[SceneObject],
    current_objects: list[SceneObject],
    matches: list[tuple[str, str, float]],
    threshold: float,
) -> list[tuple[str, str, float]]:
    used_left = {left_id for left_id, _right_id, _score in matches}
    used_right = {right_id for _left_id, right_id, _score in matches}
    candidates: list[tuple[float, SceneObject, SceneObject]] = []
    for left in prev_objects:
        if left.id in used_left or not _is_container_shape(left):
            continue
        for right in current_objects:
            if right.id in used_right or not _is_container_shape(right):
                continue
            score = _match_score(left, right)
            if score >= threshold:
                candidates.append((score, left, right))
    if not candidates:
        return matches
    return [
        *matches,
        *_greedy_matches(candidates, used_left=used_left, used_right=used_right),
    ]


def _match_score(left: SceneObject, right: SceneObject) -> float:
    left_name = _normalized_name(left.name)
    right_name = _normalized_name(right.name)
    names_match = bool(left_name and left_name == right_name)
    creation_matches = bool(left.creation_id and left.creation_id == right.creation_id)
    explicit_name_match = names_match and not _is_generic_powerpoint_name(left_name)
    if left.asset_id and right.asset_id and left.asset_id != right.asset_id and not (creation_matches or explicit_name_match):
        return 0.0

    score = 0.0
    if creation_matches:
        score += 140.0
    if left.asset_id and left.asset_id == right.asset_id:
        score += 75.0
    if left.shape_id and left.shape_id == right.shape_id:
        score += 18.0
    if names_match:
        score += 70.0 if explicit_name_match else 30.0
    if left.text and left.text == right.text:
        score += 60.0
    if _group_path(left) and _group_path(left) == _group_path(right):
        score += 22.0
    if left.kind == right.kind:
        score += 10.0
    if left.shape and left.shape == right.shape:
        score += 5.0
    score += _geometry_score(left, right)
    return score


def _container_match_score(
    left: SceneObject,
    right: SceneObject,
    anchors: list[tuple[SceneObject, SceneObject]],
) -> float:
    if left.shape and right.shape and left.shape != right.shape:
        return 0.0
    size_error = _size_error(left, right)
    if size_error > 0.08:
        return 0.0

    support = 0
    relative_error = 0.0
    for left_anchor, right_anchor in anchors:
        if left_anchor.id == left.id or right_anchor.id == right.id:
            continue
        if not _object_can_support_container(left_anchor):
            continue
        if (
            left_anchor.geometry.cx > left.geometry.cx * 1.08
            or left_anchor.geometry.cy > left.geometry.cy * 1.08
            or right_anchor.geometry.cx > right.geometry.cx * 1.08
            or right_anchor.geometry.cy > right.geometry.cy * 1.08
        ):
            continue
        if not _contains_center(left, left_anchor) or not _contains_center(right, right_anchor):
            continue
        error = _relative_child_error(left, right, left_anchor, right_anchor)
        if error > 0.35:
            continue
        support += 1
        relative_error += error

    if support <= 0:
        return 0.0

    score = 35.0
    score += min(70.0, support * 22.0)
    score += max(0.0, 18.0 * (1.0 - min(1.0, size_error / 0.08)))
    score += _geometry_score(left, right) * 0.5
    if left.shape and left.shape == right.shape:
        score += 8.0
    if left.fill and left.fill == right.fill:
        score += 6.0
    if left.stroke and left.stroke == right.stroke:
        score += 4.0
    score -= min(28.0, (relative_error / support) * 30.0)
    return score


def _group_path(obj: SceneObject) -> tuple[str, ...]:
    raw = obj.provenance.get("groupPath") if isinstance(obj.provenance, dict) else None
    if not isinstance(raw, list):
        return ()
    return tuple(str(item) for item in raw if str(item))


def _is_container_shape(obj: SceneObject) -> bool:
    return (
        obj.kind == "shape"
        and obj.asset_id is None
        and not obj.text
        and obj.geometry.cx > 0
        and obj.geometry.cy > 0
    )


def _is_panel_container(obj: SceneObject, slide_w: float, slide_h: float) -> bool:
    shape = str(obj.shape or "").lower()
    return (
        _is_container_shape(obj)
        and "round" in shape
        and obj.geometry.cx >= slide_w * 0.35
        and obj.geometry.cy >= slide_h * 0.35
    )


def _is_panel_related(
    obj: SceneObject,
    objects: list[SceneObject],
    slide_w: float,
    slide_h: float,
) -> bool:
    if _is_panel_container(obj, slide_w, slide_h):
        return True
    for panel in objects:
        if panel.id == obj.id or not _is_panel_container(panel, slide_w, slide_h):
            continue
        if obj.geometry.cx > panel.geometry.cx * 1.08 or obj.geometry.cy > panel.geometry.cy * 1.08:
            continue
        if _contains_center(panel, obj):
            return True
    return False


def _object_can_support_container(obj: SceneObject) -> bool:
    return bool(obj.asset_id or obj.text or obj.kind in {"video", "image", "svg", "text"})


def _contains_center(container: SceneObject, child: SceneObject) -> bool:
    margin_x = max(container.geometry.cx * 0.04, 1.0)
    margin_y = max(container.geometry.cy * 0.04, 1.0)
    cx = child.geometry.x + child.geometry.cx / 2.0
    cy = child.geometry.y + child.geometry.cy / 2.0
    return (
        container.geometry.x - margin_x <= cx <= container.geometry.x + container.geometry.cx + margin_x
        and container.geometry.y - margin_y <= cy <= container.geometry.y + container.geometry.cy + margin_y
    )


def _relative_child_error(
    left_container: SceneObject,
    right_container: SceneObject,
    left_child: SceneObject,
    right_child: SceneObject,
) -> float:
    left_rect = _relative_child_rect(left_container, left_child)
    right_rect = _relative_child_rect(right_container, right_child)
    return sum(abs(a - b) for a, b in zip(left_rect, right_rect)) / len(left_rect)


def _relative_child_rect(container: SceneObject, child: SceneObject) -> tuple[float, float, float, float]:
    width = max(container.geometry.cx, 1.0)
    height = max(container.geometry.cy, 1.0)
    return (
        ((child.geometry.x + child.geometry.cx / 2.0) - container.geometry.x) / width,
        ((child.geometry.y + child.geometry.cy / 2.0) - container.geometry.y) / height,
        child.geometry.cx / width,
        child.geometry.cy / height,
    )


def _size_error(left: SceneObject, right: SceneObject) -> float:
    width = max(left.geometry.cx, right.geometry.cx, 1.0)
    height = max(left.geometry.cy, right.geometry.cy, 1.0)
    return max(
        abs(left.geometry.cx - right.geometry.cx) / width,
        abs(left.geometry.cy - right.geometry.cy) / height,
    )


def _median_float(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _geometry_score(left: SceneObject, right: SceneObject) -> float:
    lx = left.geometry.x + left.geometry.cx / 2.0
    ly = left.geometry.y + left.geometry.cy / 2.0
    rx = right.geometry.x + right.geometry.cx / 2.0
    ry = right.geometry.y + right.geometry.cy / 2.0
    diagonal = math.sqrt((max(left.geometry.cx, right.geometry.cx, 1.0) ** 2) + (max(left.geometry.cy, right.geometry.cy, 1.0) ** 2))
    distance = math.sqrt(((lx - rx) ** 2) + ((ly - ry) ** 2))
    return max(0.0, 20.0 * (1.0 - min(1.0, distance / max(diagonal * 4.0, 1.0))))


def _normalized_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _is_generic_powerpoint_name(name: str) -> bool:
    return bool(
        re.match(
            r"^(picture|image|text ?box|rectangle(?:: .+)?|autoshape|shape|graphic|object|title|subtitle) \d+$",
            name,
        )
    )


def _provenance(scene: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for slide in scene["slides"]:
        for obj in slide["objects"]:
            rows.append(
                {
                    "slide": slide["index"],
                    "objectId": obj["id"],
                    "trackId": obj["trackId"],
                    "creationId": obj.get("creationId"),
                    "name": obj["name"],
                    "kind": obj["kind"],
                    "assetId": obj.get("assetId"),
                    "provenance": obj.get("provenance", {}),
                    "unsupported": obj.get("unsupported", []),
                }
            )
    return {"objects": rows}
