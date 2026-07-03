from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EMU_PER_DEGREE = 60000.0


@dataclass
class AssetRef:
    source_path: str
    rel_id: str | None
    kind: str
    extension: str
    size_bytes: int
    sha256: str
    output_file: str | None = None
    source_file: str | None = None
    width: int | None = None
    height: int | None = None
    duration_sec: float | None = None
    animated: bool = False
    alpha: bool | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"asset-{self.sha256[:16]}"

    def to_scene(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sourcePath": self.source_path,
            "sourceFile": self.source_file,
            "file": self.output_file,
            "kind": self.kind,
            "extension": self.extension,
            "sizeBytes": self.size_bytes,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "durationSec": self.duration_sec,
            "animated": self.animated,
            "alpha": self.alpha,
            "warnings": self.warnings,
        }


@dataclass
class Geometry:
    x: float = 0.0
    y: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    rotation: float = 0.0
    flip_h: bool = False
    flip_v: bool = False

    def to_scene(self, slide_w: float, slide_h: float) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.cx,
            "h": self.cy,
            "leftPct": 0.0 if slide_w <= 0 else self.x / slide_w,
            "topPct": 0.0 if slide_h <= 0 else self.y / slide_h,
            "widthPct": 0.0 if slide_w <= 0 else self.cx / slide_w,
            "heightPct": 0.0 if slide_h <= 0 else self.cy / slide_h,
            "rotation": self.rotation,
            "flipH": self.flip_h,
            "flipV": self.flip_v,
        }


@dataclass
class SceneObject:
    id: str
    shape_id: str | None
    creation_id: str | None
    name: str
    kind: str
    z: int
    geometry: Geometry
    asset_id: str | None = None
    poster_asset_id: str | None = None
    text: str = ""
    text_style: dict[str, Any] = field(default_factory=dict)
    rich_text: list[dict[str, Any]] = field(default_factory=list)
    shape: str | None = None
    fill: str | None = None
    stroke: str | None = None
    stroke_width: float | None = None
    opacity: float = 1.0
    crop: dict[str, float] | None = None
    media_timing: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported: list[str] = field(default_factory=list)
    track_id: str | None = None

    def to_scene(self, slide_w: float, slide_h: float) -> dict[str, Any]:
        group_path = self.provenance.get("groupPath") if isinstance(self.provenance, dict) else None
        return {
            "id": self.id,
            "trackId": self.track_id or self.id,
            "shapeId": self.shape_id,
            "creationId": self.creation_id,
            "name": self.name,
            "groupPath": group_path if isinstance(group_path, list) else [],
            "kind": self.kind,
            "z": self.z,
            "geometry": self.geometry.to_scene(slide_w, slide_h),
            "assetId": self.asset_id,
            "posterAssetId": self.poster_asset_id,
            "text": self.text,
            "textStyle": self.text_style,
            "richText": self.rich_text,
            "shape": self.shape,
            "fill": self.fill,
            "stroke": self.stroke,
            "strokeWidth": self.stroke_width,
            "strokeWidthPct": 0.0 if slide_h <= 0 or not self.stroke_width else self.stroke_width / slide_h,
            "opacity": self.opacity,
            "crop": self.crop,
            "mediaTiming": self.media_timing,
            "provenance": self.provenance,
            "unsupported": self.unsupported,
        }


@dataclass
class Transition:
    kind: str = "none"
    duration_sec: float = 0.0
    raw: str = ""

    def to_scene(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "durationSec": self.duration_sec,
        }


@dataclass
class Slide:
    index: int
    path: str
    transition: Transition
    objects: list[SceneObject]
    notes: list[str] = field(default_factory=list)

    def to_scene(self, slide_w: float, slide_h: float) -> dict[str, Any]:
        return {
            "index": self.index,
            "path": self.path,
            "transition": self.transition.to_scene(),
            "objects": [obj.to_scene(slide_w, slide_h) for obj in self.objects],
            "notes": self.notes,
        }


@dataclass
class PptxDeck:
    source_path: str
    source_sha256: str
    title: str
    slide_width: int
    slide_height: int
    slides: list[Slide]
    assets: dict[str, AssetRef]
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        media_bytes = sum(asset.size_bytes for asset in self.assets.values())
        transitions: dict[str, int] = {}
        for slide in self.slides:
            transitions[slide.transition.kind] = transitions.get(slide.transition.kind, 0) + 1
        return {
            "sourcePath": self.source_path,
            "sourceSha256": self.source_sha256,
            "title": self.title,
            "slideCount": len(self.slides),
            "slideSizeEmu": {"width": self.slide_width, "height": self.slide_height},
            "assetCount": len(self.assets),
            "assetBytes": media_bytes,
            "transitionCounts": transitions,
            "warnings": self.warnings,
        }
