from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import read_json


@dataclass(frozen=True)
class Profile:
    name: str = "github-pages-1080"
    width: int = 1920
    height: int = 1080
    fps: int = 30


@dataclass(frozen=True)
class AssetPolicy:
    mode: str = "copy"
    soft_max_mb: float = 50.0
    hard_max_mb: float = 100.0
    transcode_gif: bool = True
    transcode_video: bool = True
    webp_quality: int = 88
    video_crf: int = 24
    allow_oversize_assets: bool = True
    prune_unreferenced_source_assets: bool = False
    transparent_animation: str = "preserve-alpha"


@dataclass(frozen=True)
class GroupPolicy:
    explicit_groups: bool = True
    infer_panels: bool = True
    panel_border_on_top: bool = True


@dataclass(frozen=True)
class OutlinePolicy:
    normalize_white_outlines: bool = True
    border_on_top: bool = True
    width_pct: float = 0.0055
    min_px: float = 3.0
    max_px: float = 7.0


@dataclass(frozen=True)
class LayerPolicy:
    panel_outline_on_top: bool = True
    decorative_tracks: tuple[str, ...] = field(default_factory=tuple)
    transition_layer_overrides: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FallbackPolicy:
    full_slide_static: bool = False


@dataclass(frozen=True)
class MorphPolicy:
    match_threshold: float = 55.0
    duration_default_sec: float = 2.0
    easing: str = "easeInOutQuad"
    fade_unmatched: bool = True
    unmatched_fade_start: float = 0.0
    unmatched_fade_end: float = 1.0
    transition_unmatched_fade_overrides: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    transition_easing_overrides: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    transition_progress_overrides: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    transition_track_progress_overrides: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    reverse: str = "mirror"


@dataclass(frozen=True)
class QaPolicy:
    slide_ssim: float = 0.985
    morph_ssim: float = 0.965
    transition_samples: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    slide_hold_sec: float = 1.0
    settled_offset_sec: float = 0.12
    transition_reference_lead_fraction: float = 0.0
    slide_timed_video_phase_sec: float = 0.0


@dataclass(frozen=True)
class VisualAuditPolicy:
    enabled: bool = False
    samples: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    reverse_midpoints: bool = True
    fail_on_timeout: bool = True


@dataclass(frozen=True)
class PresenterConfig:
    scene_schema_version: int = 2
    title: str | None = None
    slug: str | None = None
    output_path: str | None = None
    profile: Profile = field(default_factory=Profile)
    asset_policy: AssetPolicy = field(default_factory=AssetPolicy)
    group_policy: GroupPolicy = field(default_factory=GroupPolicy)
    outline_policy: OutlinePolicy = field(default_factory=OutlinePolicy)
    layer_policy: LayerPolicy = field(default_factory=LayerPolicy)
    fallback_policy: FallbackPolicy = field(default_factory=FallbackPolicy)
    morph_policy: MorphPolicy = field(default_factory=MorphPolicy)
    qa_policy: QaPolicy = field(default_factory=QaPolicy)
    visual_audit: VisualAuditPolicy = field(default_factory=VisualAuditPolicy)
    media_phase_overrides: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    transition_media_phase_overrides: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    transition_time_overrides: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    auto_advance: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    auto_segments: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    raster_fallback_overrides: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    publish_replace: bool = False


PROFILE_PRESETS: dict[str, Profile] = {
    "github-pages-1080": Profile("github-pages-1080", 1920, 1080, 30),
    "github-pages-720": Profile("github-pages-720", 1280, 720, 30),
    "source": Profile("source", 0, 0, 30),
}


def load_config(path: Path | None = None) -> PresenterConfig:
    if path is None:
        return PresenterConfig()
    raw = read_json(path)
    base_dir = path.parent
    profile_name = str(raw.get("profile", "github-pages-1080"))
    profile = PROFILE_PRESETS.get(profile_name, PROFILE_PRESETS["github-pages-1080"])
    asset_raw: dict[str, Any] = raw.get("asset_policy", {}) or {}
    group_raw: dict[str, Any] = raw.get("group_policy", {}) or {}
    outline_raw: dict[str, Any] = raw.get("outline_policy", {}) or {}
    layer_raw: dict[str, Any] = raw.get("layer_policy", {}) or {}
    fallback_raw: dict[str, Any] = raw.get("fallback_policy", {}) or {}
    morph_raw: dict[str, Any] = raw.get("morph_policy", {}) or {}
    qa_raw: dict[str, Any] = raw.get("qa_policy", {}) or {}
    visual_raw: dict[str, Any] = raw.get("visual_audit", {}) or {}
    runtime_raw: dict[str, Any] = raw.get("runtime", {}) or {}
    media_phase_overrides = _load_override_rows(
        raw,
        "media_phase_overrides",
        "media_phase_overrides_file",
        base_dir,
    )
    transition_time_overrides = _load_override_rows(
        raw,
        "transition_time_overrides",
        "transition_time_overrides_file",
        base_dir,
    )
    auto_advance = _load_override_rows(
        raw,
        "auto_advance",
        "auto_advance_file",
        base_dir,
    )
    auto_segments = _load_override_rows(
        raw,
        "auto_segments",
        "auto_segments_file",
        base_dir,
    )
    auto_segments.extend(dict(row) for row in (runtime_raw.get("auto_segments", []) or []))
    transition_media_phase_overrides = _load_override_rows(
        raw,
        "transition_media_phase_overrides",
        "transition_media_phase_overrides_file",
        base_dir,
    )
    raster_fallback_overrides = _load_override_rows(
        raw,
        "raster_fallback_overrides",
        "raster_fallback_overrides_file",
        base_dir,
    )
    transition_layer_overrides = _load_nested_override_rows(
        raw,
        layer_raw,
        "transition_layer_overrides",
        "transition_layer_overrides_file",
        base_dir,
    )
    transition_progress_overrides = _load_nested_override_rows(
        raw,
        morph_raw,
        "transition_progress_overrides",
        "transition_progress_overrides_file",
        base_dir,
    )
    transition_track_progress_overrides = _load_nested_override_rows(
        raw,
        morph_raw,
        "transition_track_progress_overrides",
        "transition_track_progress_overrides_file",
        base_dir,
    )
    return PresenterConfig(
        scene_schema_version=int(raw.get("scene_schema_version", 2)),
        title=raw.get("title"),
        slug=raw.get("slug"),
        output_path=raw.get("output_path"),
        profile=profile,
        asset_policy=AssetPolicy(
            mode=str(asset_raw.get("mode", "copy")),
            soft_max_mb=float(asset_raw.get("soft_max_mb", 50.0)),
            hard_max_mb=float(asset_raw.get("hard_max_mb", 100.0)),
            transcode_gif=bool(asset_raw.get("transcode_gif", True)),
            transcode_video=bool(asset_raw.get("transcode_video", True)),
            webp_quality=int(asset_raw.get("webp_quality", 88)),
            video_crf=int(asset_raw.get("video_crf", 24)),
            allow_oversize_assets=bool(asset_raw.get("allow_oversize_assets", True)),
            prune_unreferenced_source_assets=bool(
                asset_raw.get("prune_unreferenced_source_assets", False)
            ),
            transparent_animation=str(
                asset_raw.get("transparent_animation", "preserve-alpha")
            ),
        ),
        group_policy=GroupPolicy(
            explicit_groups=bool(group_raw.get("explicit_groups", True)),
            infer_panels=bool(group_raw.get("infer_panels", True)),
            panel_border_on_top=bool(group_raw.get("panel_border_on_top", True)),
        ),
        outline_policy=OutlinePolicy(
            normalize_white_outlines=bool(outline_raw.get("normalize_white_outlines", True)),
            border_on_top=bool(outline_raw.get("border_on_top", True)),
            width_pct=float(outline_raw.get("width_pct", 0.0055)),
            min_px=float(outline_raw.get("min_px", 3.0)),
            max_px=float(outline_raw.get("max_px", 7.0)),
        ),
        layer_policy=LayerPolicy(
            panel_outline_on_top=bool(layer_raw.get("panel_outline_on_top", True)),
            decorative_tracks=_string_tuple(layer_raw.get("decorative_tracks", [])),
            transition_layer_overrides=tuple(transition_layer_overrides),
        ),
        fallback_policy=FallbackPolicy(
            full_slide_static=bool(fallback_raw.get("full_slide_static", False)),
        ),
        morph_policy=MorphPolicy(
            match_threshold=float(morph_raw.get("match_threshold", 55.0)),
            duration_default_sec=float(morph_raw.get("duration_default_sec", 2.0)),
            easing=str(morph_raw.get("easing", "easeInOutQuad")),
            fade_unmatched=bool(morph_raw.get("fade_unmatched", True)),
            unmatched_fade_start=float(morph_raw.get("unmatched_fade_start", 0.0)),
            unmatched_fade_end=float(morph_raw.get("unmatched_fade_end", 1.0)),
            transition_unmatched_fade_overrides=tuple(
                dict(row)
                for row in (morph_raw.get("transition_unmatched_fade_overrides", []) or [])
            ),
            transition_easing_overrides=tuple(
                dict(row)
                for row in (morph_raw.get("transition_easing_overrides", []) or [])
            ),
            transition_progress_overrides=tuple(transition_progress_overrides),
            transition_track_progress_overrides=tuple(transition_track_progress_overrides),
            reverse=str(morph_raw.get("reverse", "mirror")),
        ),
        qa_policy=QaPolicy(
            slide_ssim=float(qa_raw.get("slide_ssim", 0.985)),
            morph_ssim=float(qa_raw.get("morph_ssim", 0.965)),
            transition_samples=tuple(
                float(v)
                for v in qa_raw.get(
                    "transition_samples", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
                )
            ),
            slide_hold_sec=float(qa_raw.get("slide_hold_sec", 1.0)),
            settled_offset_sec=float(qa_raw.get("settled_offset_sec", 0.12)),
            transition_reference_lead_fraction=float(
                qa_raw.get("transition_reference_lead_fraction", 0.0)
            ),
            slide_timed_video_phase_sec=float(qa_raw.get("slide_timed_video_phase_sec", 0.0)),
        ),
        visual_audit=VisualAuditPolicy(
            enabled=bool(visual_raw.get("enabled", False)),
            samples=tuple(
                float(v)
                for v in visual_raw.get(
                    "samples", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
                )
            ),
            reverse_midpoints=bool(visual_raw.get("reverse_midpoints", True)),
            fail_on_timeout=bool(visual_raw.get("fail_on_timeout", True)),
        ),
        media_phase_overrides=tuple(media_phase_overrides),
        transition_media_phase_overrides=tuple(transition_media_phase_overrides),
        transition_time_overrides=tuple(transition_time_overrides),
        auto_advance=tuple(auto_advance),
        auto_segments=tuple(auto_segments),
        raster_fallback_overrides=tuple(raster_fallback_overrides),
        publish_replace=bool(raw.get("publish_replace", False)),
    )


def _load_override_rows(
    raw: dict[str, Any],
    inline_key: str,
    file_key: str,
    base_dir: Path,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in (raw.get(inline_key, []) or [])]
    file_value = raw.get(file_key)
    if not file_value:
        return rows
    override_path = Path(str(file_value)).expanduser()
    if not override_path.is_absolute():
        override_path = base_dir / override_path
    payload = read_json(override_path)
    if isinstance(payload, list):
        rows.extend(dict(row) for row in payload)
    else:
        rows.extend(dict(row) for row in (payload.get(inline_key, []) or []))
    return rows


def _load_nested_override_rows(
    raw: dict[str, Any],
    nested: dict[str, Any],
    inline_key: str,
    file_key: str,
    base_dir: Path,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in (nested.get(inline_key, []) or [])]
    rows.extend(dict(row) for row in (raw.get(inline_key, []) or []))
    file_value = nested.get(file_key) or raw.get(file_key)
    if not file_value:
        return rows
    override_path = Path(str(file_value)).expanduser()
    if not override_path.is_absolute():
        override_path = base_dir / override_path
    payload = read_json(override_path)
    if isinstance(payload, list):
        rows.extend(dict(row) for row in payload)
    else:
        rows.extend(dict(row) for row in (payload.get(inline_key, []) or []))
    return rows


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())
