from __future__ import annotations

import html
import posixpath
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET

from .errors import PresenterError
from .models import AssetRef, Geometry, PptxDeck, SceneObject, Slide, Transition
from .utils import sha256_bytes, sha256_file

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
P14_NS = "http://schemas.microsoft.com/office/powerpoint/2010/main"
P159_NS = "http://schemas.microsoft.com/office/powerpoint/2015/09/main"
A14_NS = "http://schemas.microsoft.com/office/drawing/2010/main"

NS = {
    "p": P_NS,
    "a": A_NS,
    "r": R_NS,
    "rel": PKG_REL_NS,
    "p14": P14_NS,
    "p159": P159_NS,
    "a14": A14_NS,
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".wdp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".wmv", ".avi", ".mpeg", ".mpg"}


def parse_pptx(path: Path) -> PptxDeck:
    pptx_path = path.expanduser().resolve()
    if not pptx_path.exists():
        raise PresenterError(f"PPTX not found: {pptx_path}")
    if not zipfile.is_zipfile(pptx_path):
        raise PresenterError(f"Invalid PPTX zip: {pptx_path}")

    with zipfile.ZipFile(pptx_path) as zf:
        slide_paths = _ordered_slide_paths(zf)
        slide_width, slide_height = _slide_size(zf)
        title = _presentation_title(zf, pptx_path)
        assets = _collect_assets(zf)
        slides: list[Slide] = []
        for index, slide_path in enumerate(slide_paths, start=1):
            slides.append(_parse_slide(zf, slide_path, index, assets, slide_width, slide_height))

    return PptxDeck(
        source_path=str(pptx_path),
        source_sha256=sha256_file(pptx_path),
        title=title,
        slide_width=slide_width,
        slide_height=slide_height,
        slides=slides,
        assets=assets,
    )


def _presentation_title(zf: zipfile.ZipFile, pptx_path: Path) -> str:
    try:
        root = ET.parse(zf.open("docProps/core.xml")).getroot()
        for node in root.iter():
            if node.tag.endswith("}title") and node.text and node.text.strip():
                return node.text.strip()
    except Exception:
        pass
    return pptx_path.stem


def _ordered_slide_paths(zf: zipfile.ZipFile) -> list[str]:
    try:
        pres_root = ET.parse(zf.open("ppt/presentation.xml")).getroot()
        rel_root = ET.parse(zf.open("ppt/_rels/presentation.xml.rels")).getroot()
    except KeyError as exc:
        raise PresenterError(f"Missing presentation metadata: {exc}") from exc

    rel_map: dict[str, str] = {}
    for rel in rel_root.findall(".//rel:Relationship", NS):
        rid = rel.get("Id")
        target = rel.get("Target")
        if rid and target:
            rel_map[rid] = _normalize_ppt_path("ppt/presentation.xml", target)

    slide_paths: list[str] = []
    for sld in pres_root.findall(".//p:sldIdLst/p:sldId", NS):
        rid = sld.get(f"{{{R_NS}}}id")
        if rid and rid in rel_map:
            slide_paths.append(rel_map[rid])
    if slide_paths:
        return slide_paths

    return sorted(
        [n for n in zf.namelist() if re.match(r"^ppt/slides/slide\d+\.xml$", n)],
        key=lambda name: int(PurePosixPath(name).stem.replace("slide", "")),
    )


def _slide_size(zf: zipfile.ZipFile) -> tuple[int, int]:
    root = ET.parse(zf.open("ppt/presentation.xml")).getroot()
    size = root.find(".//p:sldSz", NS)
    if size is None:
        return (12192000, 6858000)
    return (int(size.get("cx", "12192000")), int(size.get("cy", "6858000")))


def _collect_assets(zf: zipfile.ZipFile) -> dict[str, AssetRef]:
    assets: dict[str, AssetRef] = {}
    for name in sorted(zf.namelist()):
        if not name.startswith("ppt/media/"):
            continue
        raw = zf.read(name)
        ext = PurePosixPath(name).suffix.lower()
        kind = "video" if ext in VIDEO_EXTS else "image"
        if ext == ".svg":
            kind = "svg"
        assets[name] = AssetRef(
            source_path=name,
            rel_id=None,
            kind=kind,
            extension=ext.lstrip("."),
            size_bytes=len(raw),
            sha256=sha256_bytes(raw),
            animated=ext == ".gif",
        )
    return assets


def _parse_slide(
    zf: zipfile.ZipFile,
    slide_path: str,
    index: int,
    assets: dict[str, AssetRef],
    slide_width: int,
    slide_height: int,
) -> Slide:
    root = ET.parse(zf.open(slide_path)).getroot()
    rels = _relationships(zf, slide_path)
    layout_path = _first_relationship_target(zf, slide_path, "slideLayout")
    layout_root = _xml_root(zf, layout_path) if layout_path else None
    layout_rels = _relationships(zf, layout_path) if layout_path else {}
    master_path = _first_relationship_target(zf, layout_path, "slideMaster") if layout_path else None
    master_root = _xml_root(zf, master_path) if master_path else None
    master_rels = _relationships(zf, master_path) if master_path else {}
    transition = _transition(root)
    media_timings = _media_timings(root)
    objects: list[SceneObject] = []
    notes: list[str] = []
    z = 0

    background = _effective_background_object(
        [
            (root, slide_path, rels, "slide-background", f"s{index}-slide"),
            (layout_root, layout_path, layout_rels, "layout-background", f"s{index}-layout"),
            (master_root, master_path, master_rels, "master-background", f"s{index}-master"),
        ],
        assets,
        slide_width,
        slide_height,
        z,
    )
    if background is not None:
        objects.append(background)
        z = background.z + 1

    show_master_shapes = _truthy_attr(root, "showMasterSp", True) and _truthy_attr(layout_root, "showMasterSp", True)
    if master_root is not None and master_path and show_master_shapes:
        inherited = _parse_object_tree(
            master_root,
            source_path=master_path,
            rels=master_rels,
            assets=assets,
            z_start=z,
            object_prefix=f"s{index}-master",
            layer="master",
            skip_placeholders=True,
            media_timings={},
        )
        objects.extend(inherited)
        z = _after_z(z, inherited)

    if layout_root is not None and layout_path:
        inherited = _parse_object_tree(
            layout_root,
            source_path=layout_path,
            rels=layout_rels,
            assets=assets,
            z_start=z,
            object_prefix=f"s{index}-layout",
            layer="layout",
            skip_placeholders=True,
            media_timings={},
        )
        objects.extend(inherited)
        z = _after_z(z, inherited)

    slide_objects = _parse_object_tree(
        root,
        source_path=slide_path,
        rels=rels,
        assets=assets,
        z_start=z,
        object_prefix=f"s{index}",
        layer="slide",
        skip_placeholders=False,
        media_timings=media_timings,
    )
    objects.extend(slide_objects)
    for obj in objects:
        if obj.geometry.cx <= 0 or obj.geometry.cy <= 0:
            obj.unsupported.append("zero-size-object")
        if obj.geometry.x + obj.geometry.cx <= 0 or obj.geometry.y + obj.geometry.cy <= 0:
            obj.unsupported.append("off-canvas")
        if obj.geometry.x >= slide_width or obj.geometry.y >= slide_height:
            obj.unsupported.append("off-canvas")
    return Slide(index=index, path=slide_path, transition=transition, objects=objects, notes=notes)


def _xml_root(zf: zipfile.ZipFile, path: str | None) -> ET.Element | None:
    if not path or path not in zf.namelist():
        return None
    return ET.parse(zf.open(path)).getroot()


def _first_relationship_target(zf: zipfile.ZipFile, base_path: str | None, type_suffix: str) -> str | None:
    if not base_path:
        return None
    for rel in _relationship_entries(zf, base_path):
        if rel["type"].endswith("/" + type_suffix) or rel["type"].endswith(type_suffix):
            return rel["target"]
    return None


def _relationship_entries(zf: zipfile.ZipFile, base_path: str) -> list[dict[str, str]]:
    rel_path = str(PurePosixPath(base_path).parent / "_rels" / f"{PurePosixPath(base_path).name}.rels")
    if rel_path not in zf.namelist():
        return []
    root = ET.parse(zf.open(rel_path)).getroot()
    out: list[dict[str, str]] = []
    for rel in root.findall(".//rel:Relationship", NS):
        rid = rel.get("Id")
        target = rel.get("Target")
        rel_type = rel.get("Type", "")
        if rid and target:
            out.append(
                {
                    "id": rid,
                    "type": rel_type,
                    "target": _normalize_ppt_path(base_path, target),
                }
            )
    return out


def _relationships(zf: zipfile.ZipFile, base_path: str) -> dict[str, str]:
    return {rel["id"]: rel["target"] for rel in _relationship_entries(zf, base_path)}


def _effective_background_object(
    candidates: list[tuple[ET.Element | None, str | None, dict[str, str], str, str]],
    assets: dict[str, AssetRef],
    slide_width: int,
    slide_height: int,
    z_start: int,
) -> SceneObject | None:
    for root, source_path, rels, layer, object_prefix in candidates:
        if root is None or source_path is None:
            continue
        obj = _background_object(
            root,
            source_path=source_path,
            rels=rels,
            assets=assets,
            slide_width=slide_width,
            slide_height=slide_height,
            z_start=z_start,
            object_prefix=object_prefix,
            layer=layer,
        )
        if obj is not None:
            return obj
    return None


def _background_object(
    root: ET.Element,
    *,
    source_path: str,
    rels: dict[str, str],
    assets: dict[str, AssetRef],
    slide_width: int,
    slide_height: int,
    z_start: int,
    object_prefix: str,
    layer: str,
) -> SceneObject | None:
    bg = root.find(".//p:cSld/p:bg", NS)
    if bg is None:
        return None
    rel_ids = _rel_ids(bg)
    media_targets = _media_targets(rel_ids, rels)
    asset_id: str | None = None
    kind = "shape"
    unsupported: list[str] = []
    if media_targets:
        target = _selected_media_target(media_targets, assets, prefer_video=True)
        asset = assets.get(target)
        if asset is not None:
            asset_id = asset.id
            kind = "video" if asset.kind == "video" else "image"
        else:
            unsupported.append(f"missing-media:{target}")
    fill = _solid_color(bg, ".//a:solidFill")
    if asset_id is None and fill is None:
        if bg.find(".//p:bgRef", NS) is not None:
            unsupported.append("theme-background-reference")
        if not unsupported:
            unsupported.append("unsupported-background-fill")
        return None
    return SceneObject(
        id=f"{object_prefix}-background",
        shape_id=None,
        creation_id=None,
        name=layer.replace("-", " ").title(),
        kind=kind,
        z=z_start,
        geometry=Geometry(x=0, y=0, cx=float(slide_width), cy=float(slide_height)),
        asset_id=asset_id,
        shape="rect",
        fill=fill,
        opacity=1.0,
        provenance={
            "sourcePath": source_path,
            "layer": layer,
            "relationshipIds": rel_ids,
            "mediaTargets": media_targets,
        },
        unsupported=unsupported,
    )


def _parse_object_tree(
    root: ET.Element,
    *,
    source_path: str,
    rels: dict[str, str],
    assets: dict[str, AssetRef],
    z_start: int,
    object_prefix: str,
    layer: str,
    skip_placeholders: bool,
    media_timings: dict[str, dict[str, Any]],
) -> list[SceneObject]:
    sp_tree = root.find(".//p:cSld/p:spTree", NS)
    objects: list[SceneObject] = []
    if sp_tree is None:
        return objects
    z = z_start
    for child in list(sp_tree):
        parsed = _parse_shape(
            child,
            source_path=source_path,
            rels=rels,
            assets=assets,
            z_start=z,
            parent_group=None,
            group_path=[],
            object_prefix=object_prefix,
            layer=layer,
            skip_placeholders=skip_placeholders,
            media_timings=media_timings,
        )
        objects.extend(parsed)
        z += max(1, len(parsed))
    return objects


def _after_z(current: int, objects: list[SceneObject]) -> int:
    if not objects:
        return current
    return max(obj.z for obj in objects) + 1


def _normalize_ppt_path(base_path: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    base = posixpath.dirname(base_path)
    normalized = posixpath.normpath(posixpath.join(base, target)).replace("\\", "/")
    return normalized.lstrip("./")


def _transition(root: ET.Element) -> Transition:
    nodes = [node for node in root.iter() if _local(node.tag) == "transition"]
    if not nodes:
        return Transition()
    node = nodes[0]
    raw = ET.tostring(node, encoding="unicode")
    kind = "custom"
    if "morph" in raw.lower():
        kind = "morph"
    else:
        for child in node.iter():
            local = _local(child.tag)
            if local not in {"transition", "ext", "extLst"}:
                kind = local
                break
    duration_ms = _first_int_attr(node, ["dur", f"{{{P14_NS}}}dur"])
    if duration_ms is None:
        duration_ms = _first_int_attr(root, ["dur", f"{{{P14_NS}}}dur"])
    return Transition(kind=kind, duration_sec=round((duration_ms or 0) / 1000.0, 3), raw=raw)


def _media_timings(root: ET.Element) -> dict[str, dict[str, Any]]:
    timings: dict[str, dict[str, Any]] = {}
    timing_root = root.find(".//p:timing", NS)
    if timing_root is None:
        return timings
    for cmd in timing_root.findall(".//p:cmd", NS):
        if cmd.get("type") != "call":
            continue
        command = cmd.get("cmd") or ""
        match = re.match(r"playFrom\(([0-9.]+)\)", command)
        target = cmd.find(".//p:tgtEl/p:spTgt", NS)
        if target is None or not target.get("spid"):
            continue
        spid = str(target.get("spid"))
        if command == "togglePause":
            timing = timings.setdefault(spid, {"kind": "media"})
            timing["paused"] = True
            continue
        if not match:
            continue
        behavior = cmd.find(".//p:cBhvr/p:cTn", NS)
        duration_ms = _int_attr(behavior, "dur") if behavior is not None else None
        prior = timings.get(spid, {})
        timings[spid] = {
            **prior,
            "kind": "playFrom",
            "startSec": float(match.group(1)),
            "durationSec": round(duration_ms / 1000.0, 3) if duration_ms is not None else None,
        }
    return timings


def _parse_shape(
    node: ET.Element,
    *,
    source_path: str,
    rels: dict[str, str],
    assets: dict[str, AssetRef],
    z_start: int,
    parent_group: dict[str, float] | None,
    group_path: list[str],
    object_prefix: str,
    layer: str,
    skip_placeholders: bool,
    media_timings: dict[str, dict[str, Any]],
) -> list[SceneObject]:
    local = _local(node.tag)
    if local == "grpSp":
        group_name, group_id, _group_creation_id = _shape_identity(node)
        group_xfrm = _group_transform(node, parent_group)
        objects: list[SceneObject] = []
        z = z_start
        for child in list(node):
            if _local(child.tag) in {"nvGrpSpPr", "grpSpPr"}:
                continue
            parsed = _parse_shape(
                child,
                source_path=source_path,
                rels=rels,
                assets=assets,
                z_start=z,
                parent_group=group_xfrm,
                group_path=[*group_path, group_name or f"group-{group_id or z_start}"],
                object_prefix=object_prefix,
                layer=layer,
                skip_placeholders=skip_placeholders,
                media_timings=media_timings,
            )
            objects.extend(parsed)
            z += max(1, len(parsed))
        return objects

    if local not in {"pic", "sp", "graphicFrame", "cxnSp"}:
        return []
    if skip_placeholders and _has_placeholder(node):
        return []

    name, shape_id, creation_id = _shape_identity(node)
    geom = _geometry(node, parent_group)
    rel_ids = _rel_ids(node)
    asset_id: str | None = None
    poster_asset_id: str | None = None
    kind = "shape"
    crop = _crop(node)
    unsupported: list[str] = []
    media_targets = _media_targets(rel_ids, rels)
    prefer_hdphoto = _has_hdphoto_image_layer(node)
    if media_targets:
        target = _selected_media_target(
            media_targets,
            assets,
            prefer_video=bool(media_timings.get(str(shape_id), {})),
            prefer_hdphoto=prefer_hdphoto,
        )
        asset = assets.get(target)
        if asset is not None:
            asset_id = asset.id
            kind = "video" if asset.kind == "video" else "image"
            if prefer_hdphoto and asset.extension.lower() == "wdp":
                unsupported.append("hdphoto-layer-selected-for-conversion")
            elif prefer_hdphoto:
                unsupported.append("hdphoto-layer-fallback-media-selected")
            if asset.kind == "video":
                poster = _selected_poster_target(media_targets, assets, selected_target=target)
                if poster is not None:
                    poster_asset_id = poster.id
                if media_targets[0] != target:
                    unsupported.append("video-media-preferred-over-poster")
        else:
            unsupported.append(f"missing-media:{target}")
    elif local == "graphicFrame":
        kind = "graphic"
        unsupported.append("graphic-frame-fallback")
    elif local == "cxnSp":
        kind = "shape"
        unsupported.append("connector")

    text = _shape_text(node)
    text_style = _text_style(node) if text else {}
    rich_text = _rich_text(node) if text else []
    if text and kind == "shape":
        kind = "text"
    shape = _preset_geometry(node)
    fill = _solid_color(node, ".//p:spPr/a:solidFill")
    stroke = _solid_color(node, ".//p:spPr/a:ln/a:solidFill")
    stroke_width = _line_width(node)
    opacity = _opacity(node, media=asset_id is not None)
    if asset_id is not None and opacity <= 0:
        opacity = 1.0
        unsupported.append("media-alpha-ignored")
    obj = SceneObject(
        id=f"{object_prefix}-o{shape_id or z_start}",
        shape_id=str(shape_id) if shape_id is not None else None,
        creation_id=creation_id,
        name=html.unescape(name or ""),
        kind=kind,
        z=z_start,
        geometry=geom,
        asset_id=asset_id,
        poster_asset_id=poster_asset_id,
        text=text,
        text_style=text_style,
        rich_text=rich_text,
        shape=shape,
        fill=fill,
        stroke=stroke,
        stroke_width=stroke_width,
        opacity=opacity,
        crop=crop,
        media_timing=media_timings.get(str(shape_id), {}),
        provenance={
            "sourcePath": source_path,
            "slidePath": source_path,
            "layer": layer,
            "groupPath": group_path,
            "relationshipIds": rel_ids,
            "mediaTargets": media_targets,
        },
        unsupported=unsupported,
    )
    return [obj]


def _shape_identity(node: ET.Element) -> tuple[str | None, str | None, str | None]:
    c_nv = node.find(".//p:cNvPr", NS)
    if c_nv is None:
        return (None, None, None)
    creation_id = None
    for child in c_nv.iter():
        if _local(child.tag) == "creationId" and child.get("id"):
            creation_id = child.get("id")
            break
    return (c_nv.get("name"), c_nv.get("id"), creation_id)


def _has_placeholder(node: ET.Element) -> bool:
    return node.find(".//p:ph", NS) is not None


def _geometry(node: ET.Element, parent_group: dict[str, float] | None) -> Geometry:
    xfrm = node.find(".//a:xfrm", NS)
    if xfrm is None:
        return Geometry()
    off = xfrm.find("a:off", NS)
    ext = xfrm.find("a:ext", NS)
    x = float(off.get("x", "0")) if off is not None else 0.0
    y = float(off.get("y", "0")) if off is not None else 0.0
    cx = float(ext.get("cx", "0")) if ext is not None else 0.0
    cy = float(ext.get("cy", "0")) if ext is not None else 0.0
    if parent_group:
        sx = parent_group["cx"] / parent_group["ch_cx"] if parent_group["ch_cx"] else 1.0
        sy = parent_group["cy"] / parent_group["ch_cy"] if parent_group["ch_cy"] else 1.0
        x = parent_group["x"] + ((x - parent_group["ch_x"]) * sx)
        y = parent_group["y"] + ((y - parent_group["ch_y"]) * sy)
        cx *= sx
        cy *= sy
    rotation = (float(xfrm.get("rot", "0") or 0) / 60000.0) % 360.0
    return Geometry(
        x=x,
        y=y,
        cx=cx,
        cy=cy,
        rotation=rotation,
        flip_h=xfrm.get("flipH") in {"1", "true"},
        flip_v=xfrm.get("flipV") in {"1", "true"},
    )


def _group_transform(node: ET.Element, parent_group: dict[str, float] | None) -> dict[str, float]:
    xfrm = node.find(".//p:grpSpPr/a:xfrm", NS)
    if xfrm is None:
        base = {"x": 0.0, "y": 0.0, "cx": 1.0, "cy": 1.0, "ch_x": 0.0, "ch_y": 0.0, "ch_cx": 1.0, "ch_cy": 1.0}
    else:
        off = xfrm.find("a:off", NS)
        ext = xfrm.find("a:ext", NS)
        ch_off = xfrm.find("a:chOff", NS)
        ch_ext = xfrm.find("a:chExt", NS)
        base = {
            "x": float(off.get("x", "0")) if off is not None else 0.0,
            "y": float(off.get("y", "0")) if off is not None else 0.0,
            "cx": float(ext.get("cx", "1")) if ext is not None else 1.0,
            "cy": float(ext.get("cy", "1")) if ext is not None else 1.0,
            "ch_x": float(ch_off.get("x", "0")) if ch_off is not None else 0.0,
            "ch_y": float(ch_off.get("y", "0")) if ch_off is not None else 0.0,
            "ch_cx": float(ch_ext.get("cx", "1")) if ch_ext is not None else 1.0,
            "ch_cy": float(ch_ext.get("cy", "1")) if ch_ext is not None else 1.0,
        }
    if parent_group:
        geom = _geometry_from_group(base, parent_group)
        base.update(geom)
    return base


def _geometry_from_group(group: dict[str, float], parent: dict[str, float]) -> dict[str, float]:
    sx = parent["cx"] / parent["ch_cx"] if parent["ch_cx"] else 1.0
    sy = parent["cy"] / parent["ch_cy"] if parent["ch_cy"] else 1.0
    return {
        "x": parent["x"] + ((group["x"] - parent["ch_x"]) * sx),
        "y": parent["y"] + ((group["y"] - parent["ch_y"]) * sy),
        "cx": group["cx"] * sx,
        "cy": group["cy"] * sy,
    }


def _rel_ids(node: ET.Element) -> list[str]:
    out: set[str] = set()
    for child in node.iter():
        for key, val in child.attrib.items():
            if key.endswith("}embed") or key.endswith("}link"):
                out.add(val)
    return sorted(out)


def _media_targets(rel_ids: list[str], rels: dict[str, str]) -> list[str]:
    return [rels[rid] for rid in rel_ids if rid in rels and rels[rid].startswith("ppt/media/")]


def _selected_media_target(
    media_targets: list[str],
    assets: dict[str, AssetRef],
    *,
    prefer_video: bool,
    prefer_hdphoto: bool = False,
) -> str:
    if prefer_video:
        for target in media_targets:
            asset = assets.get(target)
            if asset is not None and asset.kind == "video":
                return target
    if prefer_hdphoto:
        for target in media_targets:
            asset = assets.get(target)
            if asset is not None and asset.extension.lower() == "wdp":
                return target
    return media_targets[0]


def _has_hdphoto_image_layer(node: ET.Element) -> bool:
    for child in node.iter():
        if _local(child.tag) == "imgLayer":
            return True
    return False


def _selected_poster_target(
    media_targets: list[str],
    assets: dict[str, AssetRef],
    *,
    selected_target: str,
) -> AssetRef | None:
    for target in media_targets:
        if target == selected_target:
            continue
        asset = assets.get(target)
        if asset is not None and asset.kind != "video":
            return asset
    return None


def _shape_text(node: ET.Element) -> str:
    chunks: list[str] = []
    for paragraph in node.findall(".//a:p", NS):
        parts: list[str] = []
        for text in paragraph.findall(".//a:t", NS):
            if text.text:
                parts.append(text.text)
        if parts:
            chunks.append("".join(parts))
    return "\n".join(chunks).strip()


def _rich_text(node: ET.Element) -> list[dict[str, Any]]:
    paragraphs: list[dict[str, Any]] = []
    for paragraph in node.findall(".//a:p", NS):
        runs: list[dict[str, Any]] = []
        for child in list(paragraph):
            local = _local(child.tag)
            if local == "r":
                text = child.find("a:t", NS)
                if text is None or text.text is None:
                    continue
                style = _style_from_text_props(child.find("a:rPr", NS))
                runs.append({"text": _clean_text_run(text.text), "style": style})
            elif local == "br":
                runs.append({"text": "\n", "style": {}})
        if runs:
            ppr = paragraph.find("a:pPr", NS)
            row: dict[str, Any] = {"runs": runs}
            if ppr is not None and ppr.get("algn"):
                row["align"] = ppr.get("algn")
            paragraphs.append(row)
    return paragraphs


def _text_style(node: ET.Element) -> dict[str, Any]:
    style: dict[str, Any] = {}
    text_props = _first_text_props(node)
    if text_props is not None:
        style.update(_style_from_text_props(text_props))

    paragraph = node.find(".//a:pPr", NS)
    if paragraph is not None and paragraph.get("algn"):
        style["align"] = paragraph.get("algn")

    body = node.find(".//a:bodyPr", NS)
    if body is not None:
        if body.get("anchor"):
            style["anchor"] = body.get("anchor")
        if body.find("a:spAutoFit", NS) is not None:
            style["autoFit"] = "shape"
        insets = {}
        for key, default in (("lIns", 91440), ("rIns", 91440), ("tIns", 45720), ("bIns", 45720)):
            value = _int_attr(body, key)
            insets[key] = default if value is None else value
        style["insets"] = insets
    return style


def _style_from_text_props(text_props: ET.Element | None) -> dict[str, Any]:
    style: dict[str, Any] = {}
    if text_props is None:
        return style
    size = _int_attr(text_props, "sz")
    if size is not None:
        style["fontSizePt"] = round(size / 100.0, 3)
    if "b" in text_props.attrib:
        style["bold"] = _truthy_text_attr(text_props.get("b"))
    if "i" in text_props.attrib:
        style["italic"] = _truthy_text_attr(text_props.get("i"))
    color = _solid_color(text_props, ".//a:solidFill")
    if color:
        style["color"] = color
    latin = text_props.find(".//a:latin", NS)
    if latin is not None and latin.get("typeface"):
        style["typeface"] = latin.get("typeface")
    return style


def _clean_text_run(value: str) -> str:
    return value.replace("\u200b", "")


def _first_text_props(node: ET.Element) -> ET.Element | None:
    for local_name in ("rPr", "endParaRPr", "defRPr"):
        for child in node.iter():
            if _local(child.tag) == local_name and (
                child.get("sz") is not None
                or child.get("b") is not None
                or child.get("i") is not None
                or child.find(".//a:solidFill", NS) is not None
                or child.find(".//a:latin", NS) is not None
            ):
                return child
    return None


def _preset_geometry(node: ET.Element) -> str | None:
    geom = node.find(".//a:prstGeom", NS)
    return geom.get("prst") if geom is not None else None


def _solid_color(node: ET.Element, query: str) -> str | None:
    fill = node.find(query, NS)
    if fill is None:
        return None
    srgb = fill.find(".//a:srgbClr", NS)
    if srgb is not None and srgb.get("val"):
        return "#" + srgb.get("val")
    scheme = fill.find(".//a:schemeClr", NS)
    if scheme is not None and scheme.get("val"):
        return f"scheme:{scheme.get('val')}"
    return None


def _line_width(node: ET.Element) -> float | None:
    line = node.find(".//p:spPr/a:ln", NS)
    if line is None:
        return None
    value = line.get("w")
    if value is None:
        return 12700.0
    try:
        return float(value)
    except ValueError:
        return 12700.0


def _opacity(node: ET.Element, *, media: bool = False) -> float:
    if media:
        alpha = node.find(".//p:blipFill/a:blip/a:alpha", NS)
        if alpha is None:
            alpha = node.find(".//p:blipFill/a:blip/a:alphaModFix", NS)
    else:
        alpha = node.find(".//p:spPr/a:solidFill//a:alpha", NS)
    if alpha is None:
        return 1.0
    try:
        raw = alpha.get("val") or alpha.get("amt") or "100000"
        return max(0.0, min(1.0, int(raw) / 100000.0))
    except ValueError:
        return 1.0


def _crop(node: ET.Element) -> dict[str, float] | None:
    src_rect = node.find(".//a:srcRect", NS)
    if src_rect is None:
        return None
    out = {}
    for key in ("l", "t", "r", "b"):
        value = src_rect.get(key)
        if value is not None:
            out[key] = int(value) / 100000.0
    return out or None


def _truthy_attr(node: ET.Element | None, name: str, default: bool) -> bool:
    if node is None:
        return default
    value = node.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no"}


def _truthy_text_attr(value: str | None) -> bool:
    return (value or "").strip().lower() not in {"0", "false", "no"}


def _int_attr(node: ET.Element, name: str) -> int | None:
    value = node.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _first_int_attr(node: ET.Element, names: list[str]) -> int | None:
    for item in node.iter():
        for name in names:
            value = item.get(name)
            if value is not None:
                try:
                    return int(value)
                except ValueError:
                    continue
    return None


def _slide_number_from_path(path: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", path)
    return int(match.group(1)) if match else 0


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
