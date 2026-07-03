from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .build import build_presentation
from .config import PresenterConfig, load_config
from .errors import PresenterError
from .publish import publish_build
from .qa import run_qa, run_visual_audit
from .pptx import parse_pptx
from .reference import export_reference_mp4
from .scene import inspect_report
from .utils import ensure_dir, find_binary, format_mb, read_json, repo_root_from, sha256_file, utc_now_iso, write_json

SHARED_SOURCE_SOFT_MAX_MB = 50.0
SHARED_SOURCE_HARD_MAX_MB = 100.0


@dataclass(frozen=True)
class FamilyDeck:
    id: str
    title: str
    source: Path
    staging: Path
    public_dir: str
    slug: str
    config: Path | None = None
    conference_url: str = ""
    conference_label: str = ""
    preview_image: str | None = None
    fallback_dir: str | None = None
    expected_slides: int | None = None


@dataclass(frozen=True)
class FamilyConfig:
    path: Path
    repo_root: Path
    family_id: str
    title: str
    shared_root: Path
    decks: tuple[FamilyDeck, ...]
    presenter_config: Path | None = None
    prune_local_assets: bool = True
    min_free_gb: float = 8.0


def load_family_config(path: Path) -> FamilyConfig:
    path = path.expanduser().resolve()
    raw = read_json(path)
    repo_root = _resolve_repo_root(raw, path)
    shared_raw = raw.get("shared_assets", {}) or {}
    shared_root = _resolve_repo_path(
        repo_root,
        str(shared_raw.get("root", "presentations/shared-assets/viscereality")),
    )
    decks: list[FamilyDeck] = []
    for row in raw.get("decks", []) or []:
        deck_id = str(row["id"])
        public_dir = str(row.get("public_dir", deck_id))
        config_path = row.get("config")
        decks.append(
            FamilyDeck(
                id=deck_id,
                title=str(row.get("title", deck_id)),
                source=_resolve_repo_path(repo_root, str(row["source"])),
                staging=_resolve_repo_path(repo_root, str(row["staging"])),
                public_dir=public_dir,
                slug=str(row.get("slug", public_dir)),
                config=_resolve_repo_path(repo_root, str(config_path)) if config_path else None,
                conference_url=str(row.get("conference_url", "")),
                conference_label=str(row.get("conference_label", "")),
                preview_image=str(row.get("preview_image")) if row.get("preview_image") else None,
                fallback_dir=str(row.get("fallback_dir", f"{public_dir}-chunked")),
                expected_slides=int(row["expected_slides"]) if row.get("expected_slides") else None,
            )
        )
    presenter_config = raw.get("presenter_config_file")
    return FamilyConfig(
        path=path,
        repo_root=repo_root,
        family_id=str(raw.get("family_id", path.stem)),
        title=str(raw.get("title", "Presentation family")),
        shared_root=shared_root,
        decks=tuple(decks),
        presenter_config=_resolve_repo_path(repo_root, str(presenter_config)) if presenter_config else None,
        prune_local_assets=bool(shared_raw.get("prune_local_deck_assets", True)),
        min_free_gb=float(raw.get("preflight", {}).get("min_free_gb", 8.0)),
    )


def inspect_family(config_path: Path) -> dict[str, Any]:
    family = load_family_config(config_path)
    preflight = _preflight(family, parse_assets=True)
    report = {
        "schema": "pptx-html-presenter.family.inspect.v1",
        "generatedAtUtc": utc_now_iso(),
        "familyId": family.family_id,
        "title": family.title,
        "repoRoot": str(family.repo_root),
        "sharedAssetRoot": _repo_rel(family.repo_root, family.shared_root),
        "status": "ok" if not preflight["failures"] else "blocked",
        "preflight": preflight,
    }
    ensure_dir(family.shared_root)
    write_json(family.shared_root / "family-inspect-report.json", report)
    return report


def build_family(
    config_path: Path,
    *,
    ffmpeg_bin: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    family = load_family_config(config_path)
    preflight = _preflight(family, parse_assets=True)
    if preflight["failures"] and not force:
        raise PresenterError(
            "Family preflight failed: "
            + "; ".join(str(item["message"]) for item in preflight["failures"])
        )
    ensure_dir(family.shared_root / "source")
    ensure_dir(family.shared_root / "optimized")
    optimized_asset_cache = _shared_optimized_asset_cache(family)
    share_reports: list[dict[str, Any]] = []
    for deck in family.decks:
        config = _deck_presenter_config(family, deck)
        report = build_presentation(
            deck.source,
            deck.staging,
            config,
            title=deck.title,
            slug=deck.slug,
            ffmpeg_bin=ffmpeg_bin,
            optimized_asset_cache=_deck_optimized_asset_cache(optimized_asset_cache, deck.id),
        )
        if deck.expected_slides and int(report.get("slideCount", 0)) != deck.expected_slides:
            report = {
                **report,
                "warnings": [
                    *report.get("warnings", []),
                    f"expected-slide-count-{deck.expected_slides}-got-{report.get('slideCount')}",
                ],
            }
            write_json(deck.staging / "build-report.json", report)
        share_reports.append(
            share_deck_assets(
                deck.staging,
                family.shared_root,
                deck_id=deck.id,
                repo_root=family.repo_root,
                prune_local_assets=family.prune_local_assets,
            )
        )
    pruned_shared_assets = prune_unreferenced_shared_assets(family)
    shared_asset_limits = _shared_asset_library_limit_report(family)
    build_reports = [_final_build_report(deck) for deck in family.decks]
    report = {
        "schema": "pptx-html-presenter.family.build.v1",
        "generatedAtUtc": utc_now_iso(),
        "familyId": family.family_id,
        "status": "ok"
        if all(row.get("status") in {"ok", "manifest-only"} for row in build_reports)
        and all(row.get("status") == "ok" for row in share_reports)
        and shared_asset_limits["githubPagesSafe"]
        and shared_asset_limits.get("preferredAssetSafe", True)
        else "needs-review",
        "preflight": preflight,
        "sharedAssetRoot": _repo_rel(family.repo_root, family.shared_root),
        "sharedAssetLimits": shared_asset_limits,
        "decks": build_reports,
        "sharedAssets": share_reports,
        "prunedSharedAssets": pruned_shared_assets,
    }
    write_json(family.shared_root / "family-build-report.json", report)
    return report


def visual_audit_family(
    config_path: Path,
    *,
    node_bin: str | None = None,
    playwright_dir: Path | None = None,
) -> dict[str, Any]:
    family = load_family_config(config_path)
    deck_reports: list[dict[str, Any]] = []
    for deck in family.decks:
        audit = run_visual_audit(
            deck.staging,
            node_bin=node_bin,
            playwright_dir=playwright_dir,
        )
        deck_reports.append(
            {
                "deckId": deck.id,
                "buildDir": str(deck.staging),
                "status": audit.get("status"),
                "summary": audit.get("summary", {}),
                "report": _repo_rel(family.repo_root, deck.staging / "qa" / "visual-audit" / "report.json"),
            }
        )
    status = "ok" if all(row.get("status") in {"ok", "passed"} for row in deck_reports) else "needs-review"
    report = {
        "schema": "pptx-html-presenter.family.visual-audit.v1",
        "generatedAtUtc": utc_now_iso(),
        "familyId": family.family_id,
        "status": status,
        "decks": deck_reports,
    }
    write_json(family.shared_root / "family-visual-audit-report.json", report)
    return report


def oracle_qa_family(
    config_path: Path,
    *,
    ffmpeg_bin: str | None = None,
    node_bin: str | None = None,
    playwright_dir: Path | None = None,
    target: str = "public",
    keep_reference: bool = False,
    force: bool = False,
    slides: set[int] | None = None,
    deck_ids: set[str] | None = None,
    min_free_gb: float | None = None,
    transition_reference_lead_fraction: float | None = None,
) -> dict[str, Any]:
    family = load_family_config(config_path)
    target = target.lower().strip()
    if target not in {"public", "staging"}:
        raise PresenterError("Family oracle QA target must be 'public' or 'staging'.")
    required_free_gb = family.min_free_gb if min_free_gb is None else float(min_free_gb)
    ffmpeg = find_binary("ffmpeg.exe", ffmpeg_bin) or find_binary("ffmpeg", ffmpeg_bin)
    free_gb = _free_gb(family.repo_root)
    blockers: list[str] = []
    if ffmpeg is None:
        blockers.append("ffmpeg-missing")
    if required_free_gb > 0 and free_gb < required_free_gb and not force:
        blockers.append(f"disk-free-below-minimum:{free_gb:.2f}GiB<{required_free_gb:.2f}GiB")

    deck_reports: list[dict[str, Any]] = []
    selected_decks = [deck for deck in family.decks if deck_ids is None or deck.id in deck_ids]
    if deck_ids is not None and not selected_decks:
        raise PresenterError(f"No family decks matched: {', '.join(sorted(deck_ids))}")
    for deck in selected_decks:
        build_dir = _deck_qa_target(family, deck, target)
        scene_path = build_dir / "deck.scene.json"
        deck_blockers = list(blockers)
        if not deck.source.exists():
            deck_blockers.append(f"source-pptx-missing:{_repo_rel(family.repo_root, deck.source)}")
        if not scene_path.exists():
            deck_blockers.append(f"scene-missing:{_repo_rel(family.repo_root, scene_path)}")
        if deck_blockers:
            deck_reports.append(
                {
                    "deckId": deck.id,
                    "status": "blocked",
                    "target": _repo_rel(family.repo_root, build_dir),
                    "blockers": deck_blockers,
                }
            )
            continue

        qa_oracle_dir = ensure_dir(build_dir / "qa" / "oracle")
        try:
            if keep_reference:
                reference_mp4 = qa_oracle_dir / f"{deck.id}-powerpoint-reference.mp4"
                reference_mp4.unlink(missing_ok=True)
                export_reference_mp4(
                    deck.source,
                    reference_mp4,
                    scene_path=build_dir,
                    ffmpeg_bin=str(ffmpeg) if ffmpeg else ffmpeg_bin,
                )
                qa_report = run_qa(
                    build_dir,
                    reference_mp4=reference_mp4,
                    ffmpeg_bin=str(ffmpeg) if ffmpeg else ffmpeg_bin,
                    node_bin=node_bin,
                    playwright_dir=playwright_dir,
                    reuse_html=False,
                    slides=slides,
                    visual_audit=False,
                    transition_reference_lead_fraction=transition_reference_lead_fraction,
                )
                deck_reports.append(_oracle_deck_report(family, deck, build_dir, qa_report, reference_mp4, True))
            else:
                with tempfile.TemporaryDirectory(prefix=f"{deck.id}-", dir=qa_oracle_dir) as temp_dir:
                    reference_mp4 = Path(temp_dir) / f"{deck.id}-powerpoint-reference.mp4"
                    export_reference_mp4(
                        deck.source,
                        reference_mp4,
                        scene_path=build_dir,
                        ffmpeg_bin=str(ffmpeg) if ffmpeg else ffmpeg_bin,
                    )
                    qa_report = run_qa(
                        build_dir,
                        reference_mp4=reference_mp4,
                        ffmpeg_bin=str(ffmpeg) if ffmpeg else ffmpeg_bin,
                        node_bin=node_bin,
                        playwright_dir=playwright_dir,
                        reuse_html=False,
                        slides=slides,
                        visual_audit=False,
                        transition_reference_lead_fraction=transition_reference_lead_fraction,
                    )
                    deck_reports.append(_oracle_deck_report(family, deck, build_dir, qa_report, reference_mp4, False))
        except Exception as exc:
            deck_reports.append(
                {
                    "deckId": deck.id,
                    "status": "blocked",
                    "target": _repo_rel(family.repo_root, build_dir),
                    "blockers": [f"{type(exc).__name__}: {exc}"],
                }
            )

    full_scope = slides is None
    if all(row.get("status") == "passed" for row in deck_reports) and full_scope:
        status = "ok"
    elif all(row.get("status") == "passed" for row in deck_reports):
        status = "partial"
    elif any(row.get("status") == "blocked" for row in deck_reports):
        status = "blocked"
    else:
        status = "failed"
    report = {
        "schema": "pptx-html-presenter.family.oracle-qa.v1",
        "generatedAtUtc": utc_now_iso(),
        "familyId": family.family_id,
        "status": status,
        "target": target,
        "deckFilter": sorted(deck_ids) if deck_ids else None,
        "fullScope": full_scope,
        "slides": sorted(slides) if slides else None,
        "disk": {
            "freeGbAtStart": round(free_gb, 3),
            "minimumFreeGb": required_free_gb,
            "force": force,
        },
        "ffmpeg": str(ffmpeg) if ffmpeg else None,
        "keepReference": keep_reference,
        "transitionReferenceLeadFractionOverride": transition_reference_lead_fraction,
        "decks": deck_reports,
        "notes": [
            "PowerPoint MP4 references are exported through PowerPoint COM using the deck scene timing.",
            "Strict pass/fail uses qa/report.json SSIM thresholds: slides >= 0.985 and transitions >= 0.965.",
            "Large raw frame folders are local QA artifacts and should stay out of Git unless explicitly requested.",
        ],
    }
    ensure_dir(family.shared_root)
    write_json(family.shared_root / "family-oracle-qa-report.json", report)
    return report


def publish_family(
    config_path: Path,
    *,
    force: bool = False,
    archive_chunked: bool = True,
) -> dict[str, Any]:
    family = load_family_config(config_path)
    presentations_dir = ensure_dir(family.repo_root / "presentations")
    publish_reports: list[dict[str, Any]] = []
    for deck in family.decks:
        build_report_path = deck.staging / "build-report.json"
        build_report = read_json(build_report_path) if build_report_path.exists() else {}
        if build_report.get("assetMode") == "manifest-only" and not force:
            raise PresenterError(f"{deck.id} is manifest-only and cannot be published without --force.")
        if not (deck.staging / "deck.scene.json").exists():
            raise PresenterError(f"Missing staging scene for {deck.id}: {deck.staging}")
        target = presentations_dir / deck.public_dir
        fallback_name = deck.fallback_dir or f"{deck.public_dir}-chunked"
        fallback = presentations_dir / fallback_name
        archived = False
        if archive_chunked and target.exists() and _looks_like_chunked_player(target):
            if fallback.exists() and not force:
                raise PresenterError(
                    f"Chunked fallback already exists for {deck.id}: {fallback}. "
                    "Use --force after reviewing it."
                )
            if fallback.exists() and force:
                shutil.rmtree(fallback)
            shutil.move(str(target), str(fallback))
            archived = True
        elif target.exists() and force:
            shutil.rmtree(target)
        report = publish_build(
            deck.staging,
            deck_id=deck.public_dir,
            repo_root=family.repo_root,
            force=force,
            update_shared_decks=False,
        )
        publish_reports.append({"deckId": deck.id, "archivedChunked": archived, **report})
    _write_shared_decks_js(family)
    _write_presentations_index(family)
    report = {
        "schema": "pptx-html-presenter.family.publish.v1",
        "generatedAtUtc": utc_now_iso(),
        "familyId": family.family_id,
        "status": "ok",
        "decks": publish_reports,
        "sharedDecks": _repo_rel(family.repo_root, family.repo_root / "presentations" / "shared" / "decks.js"),
        "index": _repo_rel(family.repo_root, family.repo_root / "presentations" / "index.html"),
    }
    write_json(family.shared_root / "family-publish-report.json", report)
    return report


def share_deck_assets(
    build_dir: Path,
    shared_root: Path,
    *,
    deck_id: str,
    repo_root: Path | None = None,
    prune_local_assets: bool = True,
) -> dict[str, Any]:
    build_dir = build_dir.expanduser().resolve()
    shared_root = shared_root.expanduser().resolve()
    repo_root = repo_root or repo_root_from(build_dir)
    scene_path = build_dir / "deck.scene.json"
    if not scene_path.exists():
        raise PresenterError(f"Missing deck.scene.json in {build_dir}")
    scene = read_json(scene_path)
    shared_root_rel = _repo_rel(repo_root, shared_root)
    index_path = shared_root / "asset-index.json"
    index = _load_asset_index(index_path)
    copied = 0
    reused = 0
    bytes_copied = 0
    rewritten = 0
    missing: list[dict[str, Any]] = []
    referenced_local_files: set[str] = set()
    for asset in scene.get("assets", []) or []:
        for field in ("file", "sourceFile"):
            value = asset.get(field)
            if not value or _is_shared_asset_ref(str(value)):
                continue
            source_abs = (build_dir / str(value)).resolve()
            if (
                field == "sourceFile"
                and asset.get("file")
                and _is_shared_asset_ref(str(asset.get("file")))
                and source_abs.exists()
                and format_mb(source_abs.stat().st_size) > SHARED_SOURCE_SOFT_MAX_MB
            ):
                asset[field] = asset["file"]
                asset.setdefault("warnings", []).append("source-file-over-soft-limit-not-published")
                rewritten += 1
                continue
            if not source_abs.exists():
                if field == "sourceFile" and asset.get("file") and _is_shared_asset_ref(str(asset.get("file"))):
                    asset[field] = asset["file"]
                    asset.setdefault("warnings", []).append("source-file-pruned-shared-runtime-file-used")
                    rewritten += 1
                    continue
                missing.append({"assetId": asset.get("id"), "field": field, "path": str(value)})
                continue
            bucket = "source" if str(value).startswith("assets/source/") else "optimized"
            digest = sha256_file(source_abs)
            ext = source_abs.suffix.lower()
            shared_rel = f"{shared_root_rel}/{bucket}/{digest}{ext}"
            shared_abs = repo_root / shared_rel
            if not shared_abs.exists():
                ensure_dir(shared_abs.parent)
                shutil.copy2(source_abs, shared_abs)
                copied += 1
                bytes_copied += shared_abs.stat().st_size
            else:
                reused += 1
            url_rel = "../" + shared_rel.removeprefix("presentations/").replace("\\", "/")
            asset[field] = url_rel
            referenced_local_files.add(str(value))
            rewritten += 1
            _upsert_asset_index_entry(
                index,
                digest=digest,
                bucket=bucket,
                ext=ext,
                shared_rel=shared_rel,
                source_abs=source_abs,
                deck_id=deck_id,
                asset=asset,
            )
    scene.setdefault("deck", {})["sharedAssetFamily"] = "viscereality"
    _rewrite_oversize_shared_sources(scene, repo_root)
    scene["sharedAssets"] = {
        "root": "../shared-assets/viscereality/",
        "index": "../shared-assets/viscereality/asset-index.json",
        "deckId": deck_id,
    }
    write_json(scene_path, scene)
    pruned = {"count": 0, "bytes": 0}
    if prune_local_assets:
        pruned = _prune_local_asset_tree(build_dir / "assets", build_dir, referenced_local_files)
    _write_asset_index(index_path, index)
    build_report_path = build_dir / "build-report.json"
    build_report = read_json(build_report_path) if build_report_path.exists() else {}
    build_report["sharedAssets"] = {
        "deckId": deck_id,
        "root": _repo_rel(repo_root, shared_root),
        "index": _repo_rel(repo_root, index_path),
        "rewrittenReferences": rewritten,
        "missingReferences": missing,
        "prunedLocalAssets": pruned,
    }
    shared_limit = _scene_shared_asset_limit_report(scene, repo_root)
    build_report["sharedAssets"]["githubPagesSafe"] = shared_limit["githubPagesSafe"]
    build_report["sharedAssets"]["preferredAssetSafe"] = shared_limit["preferredAssetSafe"]
    build_report["sharedAssets"]["maxAssetMb"] = shared_limit["maxAssetMb"]
    build_report["sharedAssets"]["oversizeAssets"] = shared_limit["oversizeAssets"]
    build_report["sharedAssets"]["softOversizeAssets"] = shared_limit["softOversizeAssets"]
    if build_report.get("assetMode") != "manifest-only" and shared_limit["githubPagesSafe"] and not missing:
        build_report["originalStatus"] = build_report.get("status")
        build_report["originalGithubPagesSafe"] = build_report.get("githubPagesSafe")
        build_report["githubPagesSafe"] = True
        build_report["status"] = "ok"
    write_json(build_report_path, build_report)
    return {
        "deckId": deck_id,
        "status": "ok" if not missing else "missing-assets",
        "buildDir": str(build_dir),
        "copied": copied,
        "reused": reused,
        "bytesCopied": bytes_copied,
        "mbCopied": format_mb(bytes_copied),
        "rewrittenReferences": rewritten,
        "missingReferences": missing,
        "prunedLocalAssets": pruned,
    }


def prune_unreferenced_shared_assets(family: FamilyConfig) -> dict[str, Any]:
    referenced = _family_shared_asset_refs(family)
    removed: list[str] = []
    bytes_removed = 0
    if family.shared_root.exists():
        for item in family.shared_root.rglob("*"):
            if not item.is_file() or item.name.endswith(".json"):
                continue
            if item.resolve() in referenced:
                continue
            size = item.stat().st_size
            item.unlink()
            removed.append(_repo_rel(family.repo_root, item))
            bytes_removed += size
    index_path = family.shared_root / "asset-index.json"
    if index_path.exists():
        index = read_json(index_path)
        assets = index.get("assets", {})
        for digest, entry in list(assets.items()):
            entry_path = family.repo_root / str(entry.get("path", ""))
            if not entry_path.exists():
                assets.pop(digest, None)
        _write_asset_index(index_path, index)
    return {
        "count": len(removed),
        "bytes": bytes_removed,
        "mb": format_mb(bytes_removed),
        "files": removed[:50],
        "truncated": len(removed) > 50,
    }


def _shared_optimized_asset_cache(family: FamilyConfig) -> dict[str, Any]:
    cache: dict[str, Any] = {"bySourceSha256": {}, "byDeckSourcePath": {}}
    index = _load_asset_index(family.shared_root / "asset-index.json")
    for entry in (index.get("assets", {}) or {}).values():
        if not isinstance(entry, dict) or entry.get("bucket") != "optimized":
            continue
        path = family.repo_root / str(entry.get("path", ""))
        if not path.exists() or format_mb(path.stat().st_size) > SHARED_SOURCE_HARD_MAX_MB:
            continue
        cache_entry = _optimized_cache_entry_from_index(entry, path)
        source_shas = entry.get("sourceSha256s") or []
        if entry.get("sourceSha256"):
            source_shas = [*source_shas, entry.get("sourceSha256")]
        for source_sha in sorted({str(item) for item in source_shas if item}):
            cache["bySourceSha256"].setdefault(source_sha, cache_entry)
        if not source_shas:
            for deck_id in entry.get("usedBy", []) or []:
                by_path = cache["byDeckSourcePath"].setdefault(str(deck_id), {})
                for source_path in entry.get("sourcePaths", []) or []:
                    by_path.setdefault(str(source_path), cache_entry)
    for deck in family.decks:
        public_scene = family.repo_root / "presentations" / deck.public_dir / "deck.scene.json"
        staging_scene = deck.staging / "deck.scene.json"
        for scene_path in (public_scene, staging_scene):
            _seed_optimized_cache_from_scene(cache, scene_path, deck.id)
    return cache


def _deck_optimized_asset_cache(cache: dict[str, Any], deck_id: str) -> dict[str, Any]:
    return {
        "bySourceSha256": dict(cache.get("bySourceSha256", {}) or {}),
        "bySourcePath": dict((cache.get("byDeckSourcePath", {}) or {}).get(deck_id, {}) or {}),
    }


def _seed_optimized_cache_from_scene(cache: dict[str, Any], scene_path: Path, deck_id: str) -> None:
    if not scene_path.exists():
        return
    try:
        scene = read_json(scene_path)
    except Exception:
        return
    for asset in scene.get("assets", []) or []:
        if not isinstance(asset, dict):
            continue
        value = str(asset.get("file") or "")
        if not value or "shared-assets/viscereality/optimized/" not in value.replace("\\", "/"):
            continue
        shared_path = _shared_ref_to_path(scene_path.parent, value)
        if not shared_path.exists() or format_mb(shared_path.stat().st_size) > SHARED_SOURCE_HARD_MAX_MB:
            continue
        entry = _optimized_cache_entry_from_scene_asset(asset, shared_path)
        source_sha = asset.get("sha256")
        if source_sha:
            cache["bySourceSha256"].setdefault(str(source_sha), entry)
        source_path = asset.get("sourcePath")
        if source_path and not source_sha:
            cache["byDeckSourcePath"].setdefault(deck_id, {}).setdefault(str(source_path), entry)


def _optimized_cache_entry_from_index(entry: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "extension": str(entry.get("extension") or path.suffix.lstrip(".")),
        "kind": entry.get("kind"),
        "animated": entry.get("animated"),
        "alpha": entry.get("alpha"),
        "bytes": path.stat().st_size,
    }


def _optimized_cache_entry_from_scene_asset(asset: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "extension": path.suffix.lower().lstrip(".") or str(asset.get("extension", "")),
        "kind": asset.get("kind"),
        "animated": asset.get("animated"),
        "alpha": asset.get("alpha"),
        "bytes": path.stat().st_size,
    }


def _final_build_report(deck: FamilyDeck) -> dict[str, Any]:
    path = deck.staging / "build-report.json"
    report = read_json(path) if path.exists() else {}
    return {"deckId": deck.id, **report}


def _deck_qa_target(family: FamilyConfig, deck: FamilyDeck, target: str) -> Path:
    if target == "public":
        return family.repo_root / "presentations" / deck.public_dir
    return deck.staging


def _oracle_deck_report(
    family: FamilyConfig,
    deck: FamilyDeck,
    build_dir: Path,
    qa_report: dict[str, Any],
    reference_mp4: Path,
    reference_kept: bool,
) -> dict[str, Any]:
    comparisons = qa_report.get("comparisons", []) or []
    failed = [row for row in comparisons if not row.get("passed", False)]
    reference_size = reference_mp4.stat().st_size if reference_mp4.exists() else 0
    return {
        "deckId": deck.id,
        "status": qa_report.get("status"),
        "target": _repo_rel(family.repo_root, build_dir),
        "report": _repo_rel(family.repo_root, build_dir / "qa" / "report.json"),
        "referenceKept": reference_kept,
        "referenceSizeMb": format_mb(reference_size),
        "sampleCount": len(qa_report.get("samples", []) or []),
        "comparisonCount": len(comparisons),
        "failedCount": len(failed),
        "blockers": qa_report.get("blockers", []),
        "minSsim": min((float(row.get("ssim", 1.0)) for row in comparisons), default=None),
    }


def _free_gb(path: Path) -> float:
    disk = shutil.disk_usage(path.anchor or path)
    return disk.free / (1024.0**3)


def _family_shared_asset_refs(family: FamilyConfig) -> set[Path]:
    refs: set[Path] = set()
    for deck in family.decks:
        scene_path = deck.staging / "deck.scene.json"
        if not scene_path.exists():
            continue
        scene = read_json(scene_path)
        for asset in scene.get("assets", []) or []:
            for field in ("file", "sourceFile"):
                value = asset.get(field)
                if not value or not _is_shared_asset_ref(str(value)):
                    continue
                refs.add(_shared_ref_to_path(deck.staging, str(value)))
    return refs


def _rewrite_oversize_shared_sources(scene: dict[str, Any], repo_root: Path) -> None:
    for asset in scene.get("assets", []) or []:
        source_file = asset.get("sourceFile")
        runtime_file = asset.get("file")
        if not source_file or not runtime_file:
            continue
        if not _is_shared_asset_ref(str(source_file)) or not _is_shared_asset_ref(str(runtime_file)):
            continue
        source_path = _shared_ref_to_path(repo_root / "presentations" / "_deck", str(source_file))
        if not source_path.exists() or format_mb(source_path.stat().st_size) <= SHARED_SOURCE_SOFT_MAX_MB:
            continue
        runtime_path = _shared_ref_to_path(repo_root / "presentations" / "_deck", str(runtime_file))
        if not runtime_path.exists() or format_mb(runtime_path.stat().st_size) > SHARED_SOURCE_HARD_MAX_MB:
            continue
        asset["sourceFile"] = runtime_file
        asset.setdefault("warnings", []).append("source-file-over-soft-limit-not-published")


def _scene_shared_asset_limit_report(scene: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    oversize: list[dict[str, Any]] = []
    soft_oversize: list[dict[str, Any]] = []
    max_mb = 0.0
    for asset in scene.get("assets", []) or []:
        for field in ("file", "sourceFile"):
            value = asset.get(field)
            if not value or not _is_shared_asset_ref(str(value)):
                continue
            path = _shared_ref_to_path(repo_root / "presentations" / "_deck", str(value))
            if not path.exists():
                continue
            mb = format_mb(path.stat().st_size)
            max_mb = max(max_mb, mb)
            if mb > SHARED_SOURCE_SOFT_MAX_MB:
                soft_oversize.append(
                    {
                        "assetId": asset.get("id"),
                        "field": field,
                        "path": _repo_rel(repo_root, path),
                        "mb": mb,
                    }
                )
            if mb > SHARED_SOURCE_HARD_MAX_MB:
                oversize.append(
                    {
                        "assetId": asset.get("id"),
                        "field": field,
                        "path": _repo_rel(repo_root, path),
                        "mb": mb,
                    }
                )
    return {
        "githubPagesSafe": not oversize,
        "preferredAssetSafe": not soft_oversize,
        "softMaxMb": SHARED_SOURCE_SOFT_MAX_MB,
        "hardMaxMb": SHARED_SOURCE_HARD_MAX_MB,
        "maxAssetMb": max_mb,
        "softOversizeAssets": soft_oversize,
        "oversizeAssets": oversize,
    }


def _shared_asset_library_limit_report(family: FamilyConfig) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    max_mb = 0.0
    oversize: list[dict[str, Any]] = []
    soft_oversize: list[dict[str, Any]] = []
    source_count = 0
    optimized_count = 0
    total_bytes = 0
    for bucket in ("source", "optimized"):
        bucket_dir = family.shared_root / bucket
        if not bucket_dir.exists():
            continue
        for item in sorted(bucket_dir.rglob("*")):
            if not item.is_file():
                continue
            size = item.stat().st_size
            mb = format_mb(size)
            total_bytes += size
            max_mb = max(max_mb, mb)
            if bucket == "source":
                source_count += 1
            else:
                optimized_count += 1
            row = {
                "path": _repo_rel(family.repo_root, item),
                "bucket": bucket,
                "mb": mb,
            }
            files.append(row)
            if mb > SHARED_SOURCE_SOFT_MAX_MB:
                soft_oversize.append(row)
            if mb > SHARED_SOURCE_HARD_MAX_MB:
                oversize.append(row)
    largest = sorted(files, key=lambda row: float(row["mb"]), reverse=True)[:10]
    return {
        "githubPagesSafe": not oversize,
        "preferredAssetSafe": not soft_oversize,
        "softMaxMb": SHARED_SOURCE_SOFT_MAX_MB,
        "hardMaxMb": SHARED_SOURCE_HARD_MAX_MB,
        "maxAssetMb": max_mb,
        "totalMb": format_mb(total_bytes),
        "sourceFileCount": source_count,
        "optimizedFileCount": optimized_count,
        "largestAssets": largest,
        "softOversizeAssets": soft_oversize,
        "oversizeAssets": oversize,
    }


def _shared_ref_to_path(base_dir: Path, value: str) -> Path:
    normalized = value.replace("\\", "/")
    if normalized.startswith("../"):
        return (base_dir / normalized).resolve()
    if normalized.startswith("/presentations/"):
        root = repo_root_from(base_dir)
        return (root / normalized.lstrip("/")).resolve()
    if normalized.startswith("presentations/"):
        root = repo_root_from(base_dir)
        return (root / normalized).resolve()
    return (base_dir / normalized).resolve()


def _preflight(family: FamilyConfig, *, parse_assets: bool) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    deck_reports: list[dict[str, Any]] = []
    disk = shutil.disk_usage(family.repo_root.anchor or family.repo_root)
    free_gb = disk.free / (1024.0**3)
    if free_gb < family.min_free_gb:
        failures.append(
            {
                "code": "disk-free-below-minimum",
                "message": f"Only {free_gb:.2f} GiB free; minimum is {family.min_free_gb:.2f} GiB.",
            }
        )
    chunked_total = 0
    shared_source_unique: dict[str, int] = {}
    for deck in family.decks:
        source_exists = deck.source.exists()
        if not source_exists:
            failures.append({"code": "missing-pptx", "deckId": deck.id, "message": str(deck.source)})
        else:
            try:
                with deck.source.open("rb"):
                    pass
            except OSError as exc:
                failures.append(
                    {"code": "pptx-unreadable", "deckId": deck.id, "message": f"{deck.source}: {exc}"}
                )
        target = family.repo_root / "presentations" / deck.public_dir
        chunked_size = _dir_size(target) if target.exists() else 0
        chunked_total += chunked_size
        report: dict[str, Any] = {
            "deckId": deck.id,
            "source": _repo_rel(family.repo_root, deck.source),
            "sourceExists": source_exists,
            "sourceSizeMb": format_mb(deck.source.stat().st_size) if source_exists else None,
            "staging": _repo_rel(family.repo_root, deck.staging),
            "publicDir": deck.public_dir,
            "currentPublicSizeMb": format_mb(chunked_size),
        }
        if parse_assets and source_exists:
            try:
                pptx_deck = parse_pptx(deck.source)
                inspect = inspect_report(pptx_deck)
                report.update(
                    {
                        "slideCount": inspect.get("slideCount"),
                        "assetCount": inspect.get("assetCount"),
                        "mediaSizeMb": format_mb(int(inspect.get("assetBytes") or 0)),
                        "transitionCounts": inspect.get("transitionCounts", {}),
                    }
                )
                for asset in pptx_deck.assets.values():
                    shared_source_unique.setdefault(asset.sha256, asset.size_bytes)
            except Exception as exc:
                warnings.append(
                    {
                        "code": "inspect-failed",
                        "deckId": deck.id,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
        deck_reports.append(report)
    unique_bytes = sum(shared_source_unique.values())
    total_media_bytes = sum(
        int(round(float(row.get("mediaSizeMb") or 0.0) * 1024 * 1024)) for row in deck_reports
    )
    return {
        "disk": {
            "path": str(family.repo_root),
            "freeGb": round(free_gb, 3),
            "minimumFreeGb": family.min_free_gb,
        },
        "decks": deck_reports,
        "currentChunkedTotalMb": format_mb(chunked_total),
        "estimatedSourceMediaTotalMb": format_mb(total_media_bytes),
        "estimatedUniqueSourceMediaMb": format_mb(unique_bytes),
        "estimatedSourceDedupSavingsMb": format_mb(max(0, total_media_bytes - unique_bytes)),
        "failures": failures,
        "warnings": warnings,
    }


def _deck_presenter_config(family: FamilyConfig, deck: FamilyDeck) -> PresenterConfig:
    if deck.config:
        return load_config(deck.config)
    if family.presenter_config:
        return load_config(family.presenter_config)
    return PresenterConfig()


def _resolve_repo_root(raw: dict[str, Any], config_path: Path) -> Path:
    if raw.get("repo_root"):
        root = Path(str(raw["repo_root"])).expanduser()
        if not root.is_absolute():
            root = config_path.parent / root
        return root.resolve()
    return repo_root_from(config_path)


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _repo_rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_asset_index(path: Path) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return {
        "schema": "pptx-html-presenter.shared-assets.v1",
        "familyId": "viscereality",
        "generatedAtUtc": utc_now_iso(),
        "assets": {},
    }


def _write_asset_index(path: Path, index: dict[str, Any]) -> None:
    index["generatedAtUtc"] = utc_now_iso()
    write_json(path, index)


def _upsert_asset_index_entry(
    index: dict[str, Any],
    *,
    digest: str,
    bucket: str,
    ext: str,
    shared_rel: str,
    source_abs: Path,
    deck_id: str,
    asset: dict[str, Any],
) -> None:
    assets = index.setdefault("assets", {})
    entry = assets.setdefault(
        digest,
        {
            "sha256": digest,
            "bucket": bucket,
            "extension": ext.lstrip("."),
            "path": shared_rel,
            "bytes": source_abs.stat().st_size,
            "mb": format_mb(source_abs.stat().st_size),
            "kind": asset.get("kind"),
            "animated": asset.get("animated"),
            "alpha": asset.get("alpha"),
            "usedBy": [],
            "sourcePaths": [],
        },
    )
    if deck_id not in entry["usedBy"]:
        entry["usedBy"].append(deck_id)
        entry["usedBy"].sort()
    source_path = asset.get("sourcePath")
    if source_path and source_path not in entry["sourcePaths"]:
        entry["sourcePaths"].append(source_path)
        entry["sourcePaths"].sort()
    source_sha = asset.get("sha256")
    if source_sha:
        source_shas = entry.setdefault("sourceSha256s", [])
        if source_sha not in source_shas:
            source_shas.append(source_sha)
            source_shas.sort()
        entry.setdefault("sourceSha256", source_sha)


def _is_shared_asset_ref(value: str) -> bool:
    return "shared-assets/viscereality/" in value.replace("\\", "/")


def _prune_local_asset_tree(asset_dir: Path, out_dir: Path, referenced_local_files: set[str]) -> dict[str, int]:
    if not asset_dir.exists():
        return {"count": 0, "bytes": 0}
    count = 0
    bytes_removed = 0
    for item in sorted(asset_dir.rglob("*"), key=lambda path: len(path.parts), reverse=True):
        if item.is_dir():
            try:
                item.rmdir()
            except OSError:
                pass
            continue
        rel = item.relative_to(out_dir).as_posix()
        if rel not in referenced_local_files and rel.startswith("assets/fallback/"):
            continue
        size = item.stat().st_size
        item.unlink()
        count += 1
        bytes_removed += size
    return {"count": count, "bytes": bytes_removed, "mb": format_mb(bytes_removed)}


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _looks_like_chunked_player(path: Path) -> bool:
    return (path / "manifest.json").exists() or any(path.glob("slide-*")) or (path / "segments").exists()


def _write_shared_decks_js(family: FamilyConfig) -> None:
    path = family.repo_root / "presentations" / "shared" / "decks.js"
    rows = []
    for deck in family.decks:
        preview = deck.preview_image or f"/presentations/{deck.public_dir}/preview.jpg"
        rows.append(
            f"""    {deck.id}: Object.freeze({{
      id: "{deck.id}",
      title: "{_js_string(deck.title)}",
      viewerPath: "/presentations/{deck.public_dir}/",
      manifestPath: "/presentations/{deck.public_dir}/deck.scene.json",
      previewImage: "{_js_string(preview)}",
      conferenceUrl: "{_js_string(deck.conference_url)}",
      conferenceLabel: "{_js_string(deck.conference_label)}",
    }}),"""
        )
    text = f"""(function () {{
  "use strict";

  // Replace with your deployed Worker endpoint, e.g.:
  // wss://viscereality-sync.your-subdomain.workers.dev/ws
  const relayWsBase = "wss://replace-with-your-relay-domain/ws";

  window.PRESENTATION_SYNC = Object.freeze({{
    relayWsBase,
    reconnectBaseMs: 1000,
    reconnectMaxMs: 12000,
  }});

  window.PRESENTATION_DECKS = Object.freeze({{
{chr(10).join(rows)}
  }});
}})();
"""
    path.write_text(text, encoding="utf-8")


def _write_presentations_index(family: FamilyConfig) -> None:
    path = family.repo_root / "presentations" / "index.html"
    cards = []
    for deck in family.decks:
        label = deck.conference_label or deck.title
        preview = deck.preview_image or f"./{deck.public_dir}/preview.jpg"
        external = (
            f'<a class="btn" href="{_html_attr(deck.conference_url)}" target="_blank" '
            f'rel="noopener noreferrer">{_html_text(label)}</a>'
            if deck.conference_url
            else ""
        )
        fallback = (
            f'<a class="btn ghost" href="./{_html_attr(deck.fallback_dir or deck.public_dir + "-chunked")}/">'
            "Chunked fallback</a>"
        )
        cards.append(
            f"""      <article class="card">
        <img class="thumb" src="{_html_attr(preview)}" alt="{_html_attr(deck.title)} preview frame">
        <div class="body">
          <div class="kicker">{_html_text(label)}</div>
          <div class="title">{_html_text(deck.title)}</div>
          <div class="meta">HTML scene recreation with shared media assets and reversible Morph-style navigation.</div>
          <div class="actions">
            <a class="btn" href="./{_html_attr(deck.public_dir)}/">Open Player</a>
            {external}
            {fallback}
          </div>
        </div>
      </article>"""
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Viscereality Presentations</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #050708;
      --panel: #11161d;
      --text: #e7edf7;
      --dim: #9aabbd;
      --accent: #66b2ff;
      --line: #213040;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      min-height: 100vh;
      background: #050708;
      color: var(--text);
      font-family: "Segoe UI", system-ui, sans-serif;
      padding: 48px 20px 56px;
    }}
    .wrap {{ max-width: 1080px; margin: 0 auto; }}
    h1 {{ font-size: clamp(1.8rem, 2.4vw, 2.6rem); margin-bottom: 8px; }}
    .subtitle {{ color: var(--dim); margin-bottom: 28px; line-height: 1.5; }}
    .grid {{ display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
    .card {{
      color: inherit;
      background: #10151d;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
      box-shadow: 0 14px 34px rgba(0, 0, 0, 0.34);
    }}
    .card:hover {{ transform: translateY(-3px); border-color: #39628a; box-shadow: 0 20px 42px rgba(0,0,0,0.44); }}
    .thumb {{ width: 100%; height: 190px; object-fit: cover; display: block; background: #0d1218; }}
    .body {{ padding: 15px 16px 18px; }}
    .kicker {{ color: var(--accent); font-size: 0.84rem; margin-bottom: 5px; letter-spacing: 0.06em; text-transform: uppercase; font-weight: 600; }}
    .title {{ font-size: 1.15rem; margin-bottom: 8px; font-weight: 650; }}
    .meta {{ color: var(--dim); line-height: 1.45; font-size: 0.94rem; }}
    .actions {{ margin-top: 12px; display: flex; gap: 10px; flex-wrap: wrap; }}
    .btn {{
      display: inline-block;
      text-decoration: none;
      color: #dce8f8;
      border: 1px solid #39628a;
      background: rgba(34, 56, 79, 0.45);
      border-radius: 8px;
      padding: 7px 11px;
      font-size: 0.86rem;
    }}
    .btn:hover {{ background: rgba(56, 91, 127, 0.6); }}
    .btn.ghost {{ border-color: #2b3b4d; color: #aebdcd; background: rgba(18, 24, 32, 0.65); }}
    .foot {{ margin-top: 30px; color: var(--dim); font-size: 0.9rem; }}
    .foot a {{ color: #b8d8ff; text-decoration: none; }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Presentations</h1>
    <p class="subtitle">Select a presentation. Each opens as its own player; shared assets stay invisible underneath the UI.</p>

    <section class="grid">
{chr(10).join(cards)}
    </section>

    <p class="foot">Return to <a href="/">viscereality.org</a>.</p>
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _js_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _html_attr(value: str) -> str:
    return _html_text(value).replace('"', "&quot;")


def _html_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
