from __future__ import annotations

from pathlib import Path
from typing import Any

from .assets import _prune_unreferenced_asset_files, prepare_assets
from .config import PresenterConfig
from .player import write_player
from .pptx import parse_pptx
from .scene import compile_scene, inspect_report
from .utils import ensure_dir, slugify, utc_now_iso, write_json


def inspect_pptx(pptx: Path, output_dir: Path | None = None) -> dict[str, Any]:
    deck = parse_pptx(pptx)
    report = inspect_report(deck)
    if output_dir is not None:
        ensure_dir(output_dir)
        write_json(output_dir / "inspect-report.json", report)
    return report


def build_presentation(
    pptx: Path,
    out_dir: Path,
    config: PresenterConfig,
    *,
    title: str | None = None,
    slug: str | None = None,
    ffmpeg_bin: str | None = None,
    optimized_asset_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    deck = parse_pptx(pptx)
    effective_config = PresenterConfig(
        scene_schema_version=config.scene_schema_version,
        title=title or config.title or deck.title,
        slug=slug or config.slug or slugify(title or config.title or deck.title),
        output_path=str(out_dir),
        profile=config.profile,
        asset_policy=config.asset_policy,
        group_policy=config.group_policy,
        outline_policy=config.outline_policy,
        layer_policy=config.layer_policy,
        fallback_policy=config.fallback_policy,
        morph_policy=config.morph_policy,
        qa_policy=config.qa_policy,
        visual_audit=config.visual_audit,
        visual_effects=config.visual_effects,
        media_phase_overrides=config.media_phase_overrides,
        transition_media_phase_overrides=config.transition_media_phase_overrides,
        transition_time_overrides=config.transition_time_overrides,
        auto_advance=config.auto_advance,
        auto_segments=config.auto_segments,
        raster_fallback_overrides=config.raster_fallback_overrides,
        publish_replace=config.publish_replace,
    )
    inspect = inspect_report(deck)
    write_json(out_dir / "inspect-report.json", inspect)
    asset_report = prepare_assets(
        deck,
        out_dir,
        effective_config.asset_policy,
        ffmpeg_bin=ffmpeg_bin,
        optimized_asset_cache=optimized_asset_cache,
    )
    scene = compile_scene(deck, effective_config, out_dir)
    referenced_asset_files = {
        str(asset.get("file"))
        for asset in scene.get("assets", [])
        if asset.get("file")
    }
    pruned_fallback_assets = _prune_unreferenced_asset_files(
        out_dir / "assets" / "fallback",
        out_dir,
        referenced_asset_files,
    )
    write_player(out_dir)
    if effective_config.asset_policy.mode == "manifest-only":
        status = "manifest-only"
    elif asset_report.get("publishAssetSafe", asset_report.get("githubPagesSafe")):
        status = "ok"
    else:
        status = "blocked-by-asset-size"
    build_report = {
        "schema": "pptx-html-presenter.build.v1",
        "generatedAtUtc": utc_now_iso(),
        "source": {
            "path": deck.source_path,
            "sha256": deck.source_sha256,
        },
        "outputDir": str(out_dir.resolve()),
        "sceneFile": "deck.scene.json",
        "playerFile": "index.html",
        "githubPagesSafe": bool(asset_report.get("publishAssetSafe", asset_report.get("githubPagesSafe"))),
        "hardLimitSafe": bool(asset_report.get("hardLimitSafe", asset_report.get("githubPagesSafe"))),
        "preferredAssetSafe": bool(asset_report.get("preferredAssetSafe", asset_report.get("githubPagesSafe"))),
        "publishAssetSafe": bool(asset_report.get("publishAssetSafe", asset_report.get("githubPagesSafe"))),
        "assetMode": effective_config.asset_policy.mode,
        "slideCount": len(scene["slides"]),
        "assetCount": len(scene["assets"]),
        "prunedFallbackAssets": pruned_fallback_assets,
        "status": status,
    }
    write_json(out_dir / "build-report.json", build_report)
    return build_report
