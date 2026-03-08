from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SlideTimingSource:
    slide_number: int
    label: str
    transition_type: str
    transition_duration_s: float
    advance_after_s: float | None
    media_timing_s: float | None = None


@dataclass
class ResolvedSegment:
    slide_number: int
    label: str
    transition_type: str
    transition_duration_s: float
    slide_duration_s: float
    duration_source: str
    asset_kind: str = "video"
    static_image_file: str | None = None
    transition_play_mode: str = "manual"

    def to_legacy_json(self) -> dict[str, Any]:
        return {
            "slide_number": self.slide_number,
            "label": self.label,
            "slide_duration_s": round(self.slide_duration_s, 3),
            "transition_duration_s": round(self.transition_duration_s, 3),
            "transition_type": self.transition_type,
            "transition_play_mode": self.transition_play_mode,
            "asset_kind": self.asset_kind,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["slide_duration_s"] = round(self.slide_duration_s, 3)
        payload["transition_duration_s"] = round(self.transition_duration_s, 3)
        return payload


@dataclass
class MediaCandidate:
    shape_id: int | None
    rel_id: str
    target: str
    extension: str
    visible: bool
    off_canvas: bool
    supported: bool
    duration_s: float | None = None
    duration_probe_error: str | None = None


@dataclass
class SlideFeature:
    slide_number: int
    label: str
    transition_type: str
    transition_duration_s: float
    visible_media_count: int
    max_visible_media_duration_s: float | None
    unresolved_visible_media_count: int
    off_canvas_media_count: int
    unsupported_media_count: int
    media_candidates: list[MediaCandidate]
    static_classification: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["transition_duration_s"] = round(float(self.transition_duration_s), 3)
        if self.max_visible_media_duration_s is not None:
            payload["max_visible_media_duration_s"] = round(
                float(self.max_visible_media_duration_s), 3
            )
        return payload


@dataclass
class TimingDecision:
    slide_number: int
    label: str
    slide_duration_s: float
    transition_duration_s: float
    transition_type: str
    slide_reason: str
    transition_reason: str
    asset_kind: str = "video"
    static_image_file: str | None = None
    transition_play_mode: str = "manual"

    def to_segment(self) -> ResolvedSegment:
        return ResolvedSegment(
            slide_number=self.slide_number,
            label=self.label,
            transition_type=self.transition_type,
            transition_duration_s=round(float(self.transition_duration_s), 3),
            slide_duration_s=round(float(self.slide_duration_s), 3),
            duration_source=f"{self.slide_reason}/{self.transition_reason}",
            asset_kind=self.asset_kind,
            static_image_file=self.static_image_file,
            transition_play_mode=self.transition_play_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["slide_duration_s"] = round(float(self.slide_duration_s), 3)
        payload["transition_duration_s"] = round(float(self.transition_duration_s), 3)
        return payload


@dataclass
class Hiccup:
    code: str
    severity: str
    message: str
    slide_number: int | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["details"] = payload["details"] or {}
        return payload

