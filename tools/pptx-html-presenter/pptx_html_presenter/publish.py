from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .errors import PresenterError
from .utils import ensure_dir, read_json, repo_root_from, utc_now_iso, write_json


def publish_build(
    build_dir: Path,
    *,
    deck_id: str,
    repo_root: Path | None = None,
    force: bool = False,
    update_shared_decks: bool = True,
) -> dict[str, Any]:
    build_dir = build_dir.expanduser().resolve()
    if not (build_dir / "deck.scene.json").exists():
        raise PresenterError(f"Missing deck.scene.json in {build_dir}")
    build_report_path = build_dir / "build-report.json"
    build_report = read_json(build_report_path) if build_report_path.exists() else {}
    if build_report.get("assetMode") == "manifest-only" and not force:
        raise PresenterError("Manifest-only builds do not include media assets and cannot be published.")
    if build_report.get("githubPagesSafe") is False and not force:
        raise PresenterError(
            "Build is blocked by GitHub Pages asset-size policy. Re-run publish with --force "
            "only for reviewed local/staging output."
        )
    status = str(build_report.get("status") or "ok")
    if status not in {"ok", "manifest-only"} and not force:
        raise PresenterError(
            f"Build status is {status}; re-run publish with --force only for reviewed output."
        )
    scene = read_json(build_dir / "deck.scene.json")
    referenced_asset_files = {
        str(asset.get("file"))
        for asset in scene.get("assets", [])
        if asset.get("file")
    }
    root = repo_root or repo_root_from(build_dir)
    presentations_dir = ensure_dir(root / "presentations")
    target = presentations_dir / deck_id
    ensure_dir(target)
    for item in build_dir.rglob("*"):
        if item.is_dir():
            continue
        if ".git" in item.parts:
            continue
        rel = item.relative_to(build_dir)
        rel_posix = rel.as_posix()
        if rel_posix.startswith(("qa/reference/", "qa/html/", "qa/diff/", "qa/visual-audit/html/")):
            continue
        if rel_posix.startswith("assets/source/") and rel_posix not in referenced_asset_files:
            continue
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dst)
    shared_updated = False
    if update_shared_decks:
        shared_updated = _upsert_shared_deck(root / "presentations" / "shared" / "decks.js", deck_id)
    report = {
        "schema": "pptx-html-presenter.publish.v1",
        "generatedAtUtc": utc_now_iso(),
        "sourceBuild": str(build_dir),
        "target": str(target),
        "deckId": deck_id,
        "sharedDecksUpdated": shared_updated,
    }
    write_json(target / "publish-report.json", report)
    return report


def _upsert_shared_deck(path: Path, deck_id: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    escaped = re.escape(deck_id)
    if re.search(rf"(?m)^\s*(?:{escaped}|[\"']{escaped}[\"'])\s*:\s*Object\.freeze", text):
        return False
    key = deck_id if re.fullmatch(r"[A-Za-z_$][\w$]*", deck_id) else f'"{deck_id}"'
    block = f"""    {key}: Object.freeze({{
      id: "{deck_id}",
      title: "{deck_id}",
      viewerPath: "/presentations/{deck_id}/",
      manifestPath: "/presentations/{deck_id}/deck.scene.json",
      previewImage: "/presentations/{deck_id}/preview.jpg",
      conferenceUrl: "",
      conferenceLabel: "",
    }}),
"""
    marker = "  });\n})();"
    if marker not in text:
        return False
    text = text.replace(marker, block + marker, 1)
    path.write_text(text, encoding="utf-8")
    return True
