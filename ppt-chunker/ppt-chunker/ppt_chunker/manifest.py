from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import ResolvedSegment
from .utils import utc_now_iso


def build_manifest(
    *,
    presentation_id: str,
    source_ppt: Path | None,
    title: str,
    segments: list[ResolvedSegment],
    chunk_entries_legacy: list[dict[str, Any]],
    chunk_entries_extended: list[dict[str, Any]],
    encoding: dict[str, Any],
    master_file: str | None = None,
) -> dict[str, Any]:
    slides_block: list[dict[str, Any]] = []
    for seg in segments:
        slide_item: dict[str, Any] = {
            "index": seg.slide_number,
            "label": seg.label,
            "slide_segment_id": f"slide_{seg.slide_number:02d}",
        }
        if seg.transition_type != "none" and seg.transition_duration_s > 0:
            slide_item["transition_segment_id"] = f"trans_{seg.slide_number:02d}"
        slides_block.append(slide_item)

    return {
        # v1 compatibility
        "title": title,
        "total_slides": len(segments),
        "chunks": chunk_entries_legacy,
        # v2 extended contract
        "presentation_id": presentation_id,
        "source_ppt": str(source_ppt) if source_ppt else None,
        "generated_at_utc": utc_now_iso(),
        "encoding": encoding,
        "master_file": master_file,
        "segments": chunk_entries_extended,
        "slides": slides_block,
        "player_defaults": {
            "start_slide": 1,
            "prev_behavior": "jump",
            "next_behavior": "transition_then_loop",
        },
    }

