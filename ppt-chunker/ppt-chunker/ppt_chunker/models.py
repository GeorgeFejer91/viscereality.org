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


@dataclass
class ResolvedSegment:
    slide_number: int
    label: str
    transition_type: str
    transition_duration_s: float
    slide_duration_s: float
    duration_source: str

    def to_legacy_json(self) -> dict[str, Any]:
        return {
            "slide_number": self.slide_number,
            "label": self.label,
            "slide_duration_s": round(self.slide_duration_s, 3),
            "transition_duration_s": round(self.transition_duration_s, 3),
            "transition_type": self.transition_type,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["slide_duration_s"] = round(self.slide_duration_s, 3)
        payload["transition_duration_s"] = round(self.transition_duration_s, 3)
        return payload

