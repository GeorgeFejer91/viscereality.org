from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pptx_html_presenter.assets import (
    _prune_unreferenced_asset_files,
    _prune_unreferenced_source_assets,
    _should_optimize_static_image,
    _should_convert_wdp,
    _should_transcode_gif,
    _try_convert_gif_with_ffmpeg,
    prepare_assets,
)
from pptx_html_presenter.build import build_presentation, inspect_pptx
from pptx_html_presenter.cli import _parse_float_list, _parse_slide_filter, _parse_track_filter
from pptx_html_presenter.config import AssetPolicy, FallbackPolicy, GroupPolicy, LayerPolicy, MorphPolicy, OutlinePolicy, PresenterConfig, VisualAuditPolicy, VisualEffectsPolicy, load_config
from pptx_html_presenter.family import load_family_config, oracle_qa_family, share_deck_assets
from pptx_html_presenter.models import AssetRef, Geometry, PptxDeck, SceneObject, Slide, Transition
from pptx_html_presenter.player import PLAYER_HTML
from pptx_html_presenter.pptx import _media_effects, _selected_media_target, _visual_effects, parse_pptx
from pptx_html_presenter.publish import _upsert_shared_deck
from pptx_html_presenter.qa import (
    _candidate_sweep_candidate_id,
    _candidate_sweep_dir_name,
    _candidate_sweep_samples,
    _filter_samples_for_slides,
    _global_ssim,
    _media_phase_config_overrides,
    _media_phase_target,
    _morph_progress_anchor_tracks,
    _morph_progress_candidate_samples,
    _morph_progress_config_overrides,
    _is_morph_progress_panel_anchor,
    _neutralized_progress_scene,
    _sample_plan,
    _track_progress_candidate_samples,
    _track_progress_config_overrides,
    _track_progress_monotonic_selection,
    _transition_effective_media_objects,
    _transition_media_phase_config_overrides,
    _transition_time_config_overrides,
    _visible_asset_patch,
    _visual_audit_sample_plan,
)
from pptx_html_presenter.reference import _effective_slide_hold, _effective_use_timings
import pptx_html_presenter.family as family_module
from pptx_html_presenter.scene import (
    _annotate_scene_graph_v2,
    _annotate_panel_relationships,
    _apply_panel_relationships_to_transitions,
    _apply_raster_fallback_overrides,
    _inferred_panel_motions,
    _match_objects,
)


class PresenterTests(unittest.TestCase):
    def test_parse_synthetic_pptx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pptx = Path(tmp) / "demo.pptx"
            _write_demo_pptx(pptx)
            deck = parse_pptx(pptx)
            self.assertEqual(len(deck.slides), 2)
            self.assertEqual(deck.slide_width, 12192000)
            self.assertEqual(deck.slides[1].transition.kind, "morph")
            self.assertEqual(len(deck.assets), 1)
            self.assertEqual(deck.slides[0].objects[0].kind, "image")
            self.assertEqual(deck.slides[0].objects[0].opacity, 1.0)
            self.assertEqual(deck.slides[1].objects[1].kind, "text")
            self.assertEqual(deck.slides[1].objects[1].text_style["fontSizePt"], 28.0)
            self.assertTrue(deck.slides[1].objects[1].text_style["bold"])
            self.assertEqual(deck.slides[1].objects[1].text_style["color"], "#FFFFFF")
            self.assertEqual(deck.slides[1].objects[1].text_style["autoFit"], "shape")
            self.assertEqual(deck.slides[1].objects[1].rich_text[0]["runs"][0]["text"], "Hello")
            self.assertEqual(deck.slides[1].objects[1].rich_text[0]["runs"][1]["text"], " Red")
            self.assertEqual(deck.slides[1].objects[1].rich_text[0]["runs"][1]["style"]["color"], "#FF0000")
            self.assertEqual(deck.slides[1].objects[1].stroke, "#FFFFFF")
            self.assertEqual(deck.slides[1].objects[0].media_timing["kind"], "playFrom")
            self.assertEqual(deck.slides[1].objects[0].media_timing["startSec"], 0.0)

    def test_parse_prefers_video_over_poster_image_for_timed_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pptx = Path(tmp) / "video-poster.pptx"
            _write_video_poster_pptx(pptx)
            deck = parse_pptx(pptx)
            obj = deck.slides[0].objects[0]
            asset = next(asset for asset in deck.assets.values() if asset.id == obj.asset_id)
            poster = next(asset for asset in deck.assets.values() if asset.id == obj.poster_asset_id)
            self.assertEqual(obj.kind, "video")
            self.assertEqual(asset.source_path, "ppt/media/media1.mp4")
            self.assertEqual(poster.source_path, "ppt/media/image1.png")
            self.assertEqual(obj.to_scene(deck.slide_width, deck.slide_height)["posterAssetId"], poster.id)
            self.assertIn("ppt/media/image1.png", obj.provenance["mediaTargets"])
            self.assertIn("ppt/media/media1.mp4", obj.provenance["mediaTargets"])
            self.assertIn("video-media-preferred-over-poster", obj.unsupported)
            self.assertTrue(obj.media_timing["paused"])

    def test_inherits_master_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pptx = Path(tmp) / "master-bg.pptx"
            _write_master_background_pptx(pptx)
            deck = parse_pptx(pptx)
            self.assertEqual(len(deck.slides), 1)
            self.assertEqual(len(deck.assets), 1)
            background = deck.slides[0].objects[0]
            self.assertEqual(background.kind, "image")
            self.assertEqual(background.provenance["sourcePath"], "ppt/slideMasters/slideMaster1.xml")
            self.assertEqual(background.provenance["layer"], "master-background")
            self.assertEqual(background.geometry.cx, deck.slide_width)
            self.assertEqual(background.geometry.cy, deck.slide_height)

    def test_inspect_and_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pptx = tmp_path / "demo.pptx"
            out = tmp_path / "site"
            _write_demo_pptx(pptx)
            report = inspect_pptx(pptx, tmp_path / "inspect")
            self.assertEqual(report["slideCount"], 2)
            build = build_presentation(
                pptx,
                out,
                PresenterConfig(visual_effects=VisualEffectsPolicy(glow_scale=0.5, glow_alpha_scale=0.8)),
                title="Demo",
                slug="demo",
            )
            self.assertEqual(build["status"], "ok")
            scene = json.loads((out / "deck.scene.json").read_text(encoding="utf-8"))
            self.assertEqual(scene["schema"], "pptx-html-presenter.scene.v2")
            self.assertEqual(scene["schemaVersion"], 2)
            self.assertEqual(scene["deck"]["id"], "demo")
            self.assertTrue((out / "index.html").exists())
            self.assertTrue((out / "asset-report.json").exists())
            self.assertTrue((out / "group-report.json").exists())
            self.assertEqual(scene["qa"]["slideHoldSec"], 1.0)
            self.assertEqual(scene["qa"]["settledOffsetSec"], 0.12)
            self.assertEqual(scene["qa"]["transitionReferenceLeadFraction"], 0.0)
            self.assertEqual(scene["qa"]["transitionTimeOverrides"], [])
            self.assertEqual(scene["qa"]["slideTimedVideoPhaseSec"], 0.0)
            self.assertEqual(scene["qa"]["mediaPhaseOverridesApplied"], 0)
            self.assertTrue(scene["runtime"]["fadeUnmatched"])
            self.assertEqual(scene["runtime"]["unmatchedFadeStart"], 0.0)
            self.assertEqual(scene["runtime"]["unmatchedFadeEnd"], 1.0)
            self.assertTrue(scene["runtime"]["groupRenderer"])
            self.assertEqual(scene["runtime"]["reverse"], "mirror")
            self.assertEqual(
                scene["runtime"]["visualEffects"],
                {
                    "glowScale": 0.5,
                    "glowAlphaScale": 0.8,
                },
            )
            self.assertEqual(
                scene["runtime"]["outlineStyle"],
                {
                    "normalizeWhiteOutlines": True,
                    "borderOnTop": True,
                    "widthPct": 0.0055,
                    "minPx": 3.0,
                    "maxPx": 7.0,
                },
            )
            self.assertEqual(
                scene["runtime"]["layerPolicy"],
                {
                    "decorativeTracks": [],
                    "panelOutlineOnTop": True,
                    "transitionLayerOverrides": [],
                },
            )
            self.assertEqual(scene["runtime"]["autoAdvance"], [])
            self.assertEqual(scene["runtime"]["autoSegments"], [])
            self.assertEqual(
                scene["qa"]["visualAudit"],
                {
                    "enabled": False,
                    "failOnTimeout": True,
                    "reverseMidpoints": True,
                    "samples": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
                },
            )
            self.assertEqual(scene["transitions"][0]["kind"], "morph")
            self.assertEqual(scene["transitions"][0]["durationSec"], 2.0)
            self.assertEqual(scene["transitions"][0]["matches"][0]["motion"]["delta"]["x"], 1000000.0)
            self.assertEqual(scene["transitions"][0]["matches"][0]["motion"]["durationSec"], 2.0)
            text_obj = scene["slides"][1]["objects"][1]
            self.assertEqual(text_obj["richText"][0]["runs"][1]["style"]["color"], "#FF0000")
            tracks = [obj["trackId"] for slide in scene["slides"] for obj in slide["objects"]]
            self.assertIn("track-0001", tracks)
            self.assertIn("nodes", scene["slides"][0])
            self.assertIn("relationships", scene["slides"][0])
            self.assertIn("groups", scene["slides"][0])

    def test_config_loads_unmatched_fade_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "morph_policy": {
                            "unmatched_fade_start": 0.5,
                            "unmatched_fade_end": 0.75,
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.morph_policy.unmatched_fade_start, 0.5)
            self.assertEqual(config.morph_policy.unmatched_fade_end, 0.75)

    def test_config_loads_transition_unmatched_fade_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "morph_policy": {
                            "transition_unmatched_fade_overrides": [
                                {
                                    "from": 2,
                                    "to": 3,
                                    "enter_start": 0.95,
                                    "enter_end": 1.0,
                                },
                                {
                                    "from": 2,
                                    "to": 3,
                                    "track_ids": ["track-title"],
                                    "exit_start": 0.0,
                                    "exit_end": 0.25,
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(
                config.morph_policy.transition_unmatched_fade_overrides[0]["enter_start"],
                0.95,
            )
            self.assertEqual(
                config.morph_policy.transition_unmatched_fade_overrides[1]["track_ids"],
                ["track-title"],
            )

    def test_config_loads_transition_easing_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "morph_policy": {
                            "easing": "linear",
                            "transition_easing_overrides": [
                                {
                                    "from": 17,
                                    "to": 18,
                                    "easing": "power:0.85",
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.morph_policy.easing, "linear")
            self.assertEqual(
                config.morph_policy.transition_easing_overrides[0]["easing"],
                "power:0.85",
            )

    def test_config_loads_transition_progress_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "morph_policy": {
                            "transition_progress_overrides": [
                                {
                                    "from": 17,
                                    "to": 18,
                                    "points": [
                                        {"progress": 0.0, "value": 0.0},
                                        {"progress": 0.5, "value": 0.46},
                                        {"progress": 1.0, "value": 1.0},
                                    ],
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(
                config.morph_policy.transition_progress_overrides[0]["points"][1]["value"],
                0.46,
            )

    def test_config_loads_transition_progress_overrides_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            overrides = root / "morph-progress.json"
            overrides.write_text(
                json.dumps(
                    {
                        "transition_progress_overrides": [
                            {
                                "from": 18,
                                "to": 19,
                                "points": [
                                    {"progress": 0.0, "value": 0.0},
                                    {"progress": 0.5, "value": 0.55},
                                    {"progress": 1.0, "value": 1.0},
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "morph_policy": {
                            "transition_progress_overrides_file": "morph-progress.json"
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(config_path)
            self.assertEqual(config.morph_policy.transition_progress_overrides[0]["from"], 18)
            self.assertEqual(
                config.morph_policy.transition_progress_overrides[0]["points"][1]["value"],
                0.55,
            )

    def test_config_loads_transition_track_progress_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "morph_policy": {
                            "transition_track_progress_overrides": [
                                {
                                    "from": 19,
                                    "to": 20,
                                    "track_id": "track-0087",
                                    "points": [
                                        {"progress": 0.0, "value": 0.0},
                                        {"progress": 0.5, "value": 0.58},
                                        {"progress": 1.0, "value": 1.0},
                                    ],
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            row = config.morph_policy.transition_track_progress_overrides[0]
            self.assertEqual(row["track_id"], "track-0087")
            self.assertEqual(row["points"][1]["value"], 0.58)

    def test_config_loads_public_asset_pruning_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "asset_policy": {
                            "prune_unreferenced_source_assets": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertTrue(config.asset_policy.prune_unreferenced_source_assets)

    def test_config_defaults_to_strict_public_asset_policy(self) -> None:
        config = load_config(None)
        self.assertTrue(config.asset_policy.optimize_static_images)
        self.assertFalse(config.asset_policy.allow_oversize_assets)

    def test_config_loads_visual_effects_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "visual_effects": {
                            "glow_scale": 0.5,
                            "glow_alpha_scale": 0.8,
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.visual_effects.glow_scale, 0.5)
            self.assertEqual(config.visual_effects.glow_alpha_scale, 0.8)

    def test_config_loads_text_rendering_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "text_rendering": {
                            "font_scale": 0.92,
                            "regular_weight": 350,
                            "bold_weight": 620,
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.text_rendering.font_scale, 0.92)
            self.assertEqual(config.text_rendering.regular_weight, 350)
            self.assertEqual(config.text_rendering.bold_weight, 620)

    def test_config_loads_auto_advance_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "auto_advance": [
                            {
                                "from": 3,
                                "to": 4,
                                "delay_sec": 0,
                                "source": "combine-slide-3-4",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.auto_advance[0]["from"], 3)
            self.assertEqual(config.auto_advance[0]["to"], 4)

    def test_config_loads_runtime_auto_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "runtime": {
                            "auto_segments": [
                                {
                                    "from": 3,
                                    "to": 4,
                                    "delay_sec": 0,
                                    "source": "combine-slide-3-4",
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.auto_segments[0]["from"], 3)
            self.assertEqual(config.auto_segments[0]["to"], 4)

    def test_config_loads_layer_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "layer_policy": {
                            "panel_outline_on_top": True,
                            "decorative_tracks": ["track-0076"],
                            "transition_layer_overrides": [
                                {
                                    "from": 19,
                                    "to": 20,
                                    "mode": "panels-above-decorative",
                                    "decorative_tracks": ["track-0076"],
                                    "z_boost": 1000,
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.layer_policy.decorative_tracks, ("track-0076",))
            self.assertEqual(config.layer_policy.transition_layer_overrides[0]["from"], 19)

    def test_config_loads_visual_audit_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "visual_audit": {
                            "enabled": True,
                            "samples": [0, 0.5, 1],
                            "reverse_midpoints": False,
                            "fail_on_timeout": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertTrue(config.visual_audit.enabled)
            self.assertEqual(config.visual_audit.samples, (0.0, 0.5, 1.0))
            self.assertFalse(config.visual_audit.reverse_midpoints)

    def test_config_loads_schema_group_and_fallback_policies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "scene_schema_version": 2,
                        "group_policy": {
                            "explicit_groups": True,
                            "infer_panels": True,
                            "panel_border_on_top": True,
                        },
                        "fallback_policy": {
                            "full_slide_static": False,
                        },
                        "outline_policy": {
                            "normalize_white_outlines": True,
                            "border_on_top": True,
                            "width_pct": 0.0055,
                            "min_px": 3,
                            "max_px": 7,
                        },
                        "morph_policy": {
                            "reverse": "mirror",
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.scene_schema_version, 2)
            self.assertEqual(config.group_policy, GroupPolicy())
            self.assertEqual(config.outline_policy, OutlinePolicy())
            self.assertEqual(config.fallback_policy, FallbackPolicy())
            self.assertEqual(config.morph_policy.reverse, "mirror")
            self.assertEqual(config.asset_policy.transparent_animation, "preserve-alpha")

    def test_config_loads_video_crf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"asset_policy": {"video_crf": 18}}),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.asset_policy.video_crf, 18)

    def test_config_loads_media_phase_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "media_phase_overrides": [
                            {
                                "slide": 2,
                                "name": "Picture 1",
                                "phase_sec": 2.5,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.media_phase_overrides[0]["slide"], 2)
            self.assertEqual(config.media_phase_overrides[0]["phase_sec"], 2.5)

    def test_config_loads_transition_media_phase_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "transition_media_phase_overrides": [
                            {
                                "from": 18,
                                "to": 19,
                                "track_id": "track-0086",
                                "phase_sec": -0.115,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.transition_media_phase_overrides[0]["track_id"], "track-0086")
            self.assertEqual(config.transition_media_phase_overrides[0]["phase_sec"], -0.115)

    def test_config_loads_raster_fallback_overrides_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            overrides = root / "fallbacks.json"
            overrides.write_text(
                json.dumps(
                    {
                        "raster_fallback_overrides": [
                            {
                                "slide": 1,
                                "file": "assets/fallback/static.png",
                                "settled_only": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            path = root / "config.json"
            path.write_text(
                json.dumps({"raster_fallback_overrides_file": "fallbacks.json"}),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.raster_fallback_overrides[0]["slide"], 1)
            self.assertTrue(config.raster_fallback_overrides[0]["settled_only"])

    def test_config_loads_transition_time_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "transition_time_overrides": [
                            {
                                "from": 1,
                                "to": 2,
                                "reference_offset_sec": -0.25,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.transition_time_overrides[0]["from"], 1)
            self.assertEqual(config.transition_time_overrides[0]["reference_offset_sec"], -0.25)

    def test_config_loads_transition_time_overrides_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            overrides = root / "transition-time.json"
            overrides.write_text(
                json.dumps(
                    {
                        "transition_time_overrides": [
                            {"from": 1, "to": 2, "reference_offset_sec": 0.125}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            path = root / "config.json"
            path.write_text(
                json.dumps({"transition_time_overrides_file": "transition-time.json"}),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.transition_time_overrides[0]["reference_offset_sec"], 0.125)

    def test_build_applies_media_phase_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pptx = tmp_path / "demo.pptx"
            out = tmp_path / "site"
            _write_demo_pptx(pptx)
            build_presentation(
                pptx,
                out,
                PresenterConfig(
                    media_phase_overrides=(
                        {
                            "slide": 2,
                            "name": "Picture 1",
                            "phase_sec": 2.5,
                            "source": "test",
                            "score": 0.99,
                        },
                    )
                ),
            )
            scene = json.loads((out / "deck.scene.json").read_text(encoding="utf-8"))
            slide_two_picture = next(
                obj
                for obj in scene["slides"][1]["objects"]
                if obj["name"] == "Picture 1"
            )
            self.assertEqual(slide_two_picture["mediaTiming"]["phaseSec"], 2.5)
            self.assertEqual(scene["qa"]["mediaPhaseOverridesApplied"], 1)
            self.assertEqual(
                slide_two_picture["provenance"]["mediaPhaseOverride"]["source"],
                "test",
            )

    def test_build_applies_transition_media_phase_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pptx = tmp_path / "demo.pptx"
            out = tmp_path / "site"
            _write_demo_pptx(pptx)
            build_presentation(
                pptx,
                out,
                PresenterConfig(
                    transition_media_phase_overrides=(
                        {
                            "from": 1,
                            "to": 2,
                            "track_id": "track-0001",
                            "phase_sec": -0.115,
                            "source": "test",
                        },
                    )
                ),
            )
            scene = json.loads((out / "deck.scene.json").read_text(encoding="utf-8"))
            self.assertEqual(scene["transitions"][0]["mediaPhaseOverrides"][0]["trackId"], "track-0001")
            self.assertEqual(scene["transitions"][0]["mediaPhaseOverrides"][0]["phaseSec"], -0.115)

    def test_build_applies_raster_fallback_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pptx = tmp_path / "demo.pptx"
            out = tmp_path / "site"
            fallback = tmp_path / "static.png"
            fallback.write_bytes(b"fake-png")
            _write_demo_pptx(pptx)
            build_presentation(
                pptx,
                out,
                PresenterConfig(
                    raster_fallback_overrides=(
                        {
                            "slide": 1,
                            "file": str(fallback),
                            "object_id": "static-slide-1",
                            "settled_only": True,
                            "source": "test",
                        },
                    )
                ),
            )
            scene = json.loads((out / "deck.scene.json").read_text(encoding="utf-8"))
            fallback_obj = next(
                obj
                for obj in scene["slides"][0]["objects"]
                if obj["id"] == "static-slide-1"
            )
            self.assertEqual(fallback_obj["kind"], "image")
            self.assertTrue(fallback_obj["rasterFallback"]["settledOnly"])
            self.assertTrue(fallback_obj["assetId"].startswith("asset-raster-"))
            self.assertEqual(scene["qa"]["rasterFallbacksApplied"], 1)
            asset = next(asset for asset in scene["assets"] if asset["id"] == fallback_obj["assetId"])
            self.assertTrue(asset["file"].startswith("assets/fallback/"))

    def test_panel_raster_fallback_skips_stale_video_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fallback = tmp_path / "panel.png"
            fallback.write_bytes(b"fake-png")
            slides = [
                {
                    "index": 1,
                    "objects": [
                        {
                            "id": "video",
                            "trackId": "track-panel-old",
                            "kind": "video",
                            "name": "Combined04",
                            "geometry": {"x": 0, "y": 0, "w": 1000, "h": 1000},
                            "z": 1,
                        }
                    ],
                }
            ]
            report = _apply_raster_fallback_overrides(
                slides,
                [],
                (
                    {
                        "slide": 1,
                        "file": str(fallback),
                        "track_id": "track-panel-old",
                        "replace_track_ids": ["track-panel-old"],
                        "name": "PowerPoint panel border track-panel-old",
                        "source": "panel-border-fallback",
                        "geometry": {"x": 0, "y": 0, "w": 1000, "h": 1000},
                    },
                ),
                tmp_path / "out",
                2000,
                1200,
            )
            self.assertEqual(report["appliedCount"], 0)
            self.assertEqual(report["rows"][0]["status"], "stale-replace-target")
            self.assertEqual([obj["id"] for obj in slides[0]["objects"]], ["video"])

    def test_panel_raster_fallback_skips_stale_panel_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fallback = tmp_path / "panel.png"
            fallback.write_bytes(b"fake-png")
            slides = [
                {
                    "index": 1,
                    "objects": [
                        {
                            "id": "panel",
                            "trackId": "track-panel",
                            "kind": "shape",
                            "shape": "roundRect",
                            "name": "Rectangle: Rounded Corners 1",
                            "geometry": {"x": 100, "y": 100, "w": 1000, "h": 800},
                            "z": 1,
                        }
                    ],
                }
            ]
            report = _apply_raster_fallback_overrides(
                slides,
                [],
                (
                    {
                        "slide": 1,
                        "file": str(fallback),
                        "track_id": "track-panel",
                        "replace_track_ids": ["track-panel"],
                        "name": "PowerPoint panel border track-panel",
                        "source": "panel-border-fallback",
                        "geometry": {"x": 3000, "y": 100, "w": 1000, "h": 800},
                    },
                ),
                tmp_path / "out",
                2000,
                1200,
            )
            self.assertEqual(report["appliedCount"], 0)
            self.assertEqual(report["rows"][0]["status"], "stale-replace-target")
            self.assertEqual([obj["id"] for obj in slides[0]["objects"]], ["panel"])

    def test_panel_raster_fallback_retargets_by_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fallback = tmp_path / "panel.png"
            fallback.write_bytes(b"fake-png")
            slides = [
                {
                    "index": 1,
                    "objects": [
                        {
                            "id": "panel-current",
                            "trackId": "track-panel-current",
                            "kind": "shape",
                            "shape": "roundRect",
                            "name": "Rectangle: Rounded Corners 1",
                            "geometry": {"x": 100, "y": 100, "w": 1000, "h": 800},
                            "z": 1,
                        }
                    ],
                }
            ]
            report = _apply_raster_fallback_overrides(
                slides,
                [],
                (
                    {
                        "slide": 1,
                        "file": str(fallback),
                        "track_id": "track-panel-old",
                        "replace_track_ids": ["track-panel-old"],
                        "name": "PowerPoint panel border track-panel-old",
                        "source": "panel-border-fallback",
                        "geometry": {"x": 100, "y": 100, "w": 1000, "h": 800},
                    },
                ),
                tmp_path / "out",
                2000,
                1200,
            )
            self.assertEqual(report["appliedCount"], 1)
            self.assertEqual(report["rows"][0]["status"], "applied")
            self.assertEqual(len(slides[0]["objects"]), 1)
            self.assertEqual(slides[0]["objects"][0]["kind"], "image")
            self.assertEqual(slides[0]["objects"][0]["trackId"], "track-panel-current")
            self.assertEqual(slides[0]["objects"][0]["shape"], "roundRect")
            self.assertEqual(slides[0]["objects"][0]["stroke"], "scheme:bg1")
            self.assertGreater(slides[0]["objects"][0]["strokeWidthPct"], 0)

    def test_build_applies_transition_unmatched_fade_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pptx = tmp_path / "demo.pptx"
            out = tmp_path / "site"
            _write_demo_pptx(pptx)
            build_presentation(
                pptx,
                out,
                PresenterConfig(
                    morph_policy=MorphPolicy(
                        transition_unmatched_fade_overrides=(
                            {
                                "from": 1,
                                "to": 2,
                                "enter_start": 0.95,
                                "enter_end": 1.0,
                                "source": "test",
                            },
                            {
                                "from": 1,
                                "to": 2,
                                "track_id": "track-0002",
                                "exit_start": 0.0,
                                "exit_end": 0.25,
                                "source": "track-test",
                            },
                        )
                    )
                ),
            )
            scene = json.loads((out / "deck.scene.json").read_text(encoding="utf-8"))
            self.assertEqual(
                scene["transitions"][0]["unmatchedFade"],
                {
                    "enterStart": 0.95,
                    "enterEnd": 1.0,
                    "source": "test",
                    "tracks": {
                        "track-0002": {
                            "exitStart": 0.0,
                            "exitEnd": 0.25,
                            "source": "track-test",
                        }
                    },
                },
            )

    def test_build_applies_transition_easing_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pptx = tmp_path / "demo.pptx"
            out = tmp_path / "site"
            _write_demo_pptx(pptx)
            build_presentation(
                pptx,
                out,
                PresenterConfig(
                    morph_policy=MorphPolicy(
                        transition_easing_overrides=(
                            {
                                "from": 1,
                                "to": 2,
                                "easing": "cubic-bezier(.2,0,.2,1)",
                            },
                        )
                    )
                ),
            )
            scene = json.loads((out / "deck.scene.json").read_text(encoding="utf-8"))
            self.assertEqual(scene["transitions"][0]["easing"], "cubic-bezier(.2,0,.2,1)")

    def test_build_applies_transition_progress_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pptx = tmp_path / "demo.pptx"
            out = tmp_path / "site"
            _write_demo_pptx(pptx)
            build_presentation(
                pptx,
                out,
                PresenterConfig(
                    morph_policy=MorphPolicy(
                        transition_progress_overrides=(
                            {
                                "from": 1,
                                "to": 2,
                                "points": [
                                    {"progress": 0.0, "value": 0.0},
                                    {"progress": 0.5, "value": 0.45},
                                    {"progress": 1.0, "value": 1.0},
                                ],
                            },
                        )
                    )
                ),
            )
            scene = json.loads((out / "deck.scene.json").read_text(encoding="utf-8"))
            self.assertEqual(
                scene["transitions"][0]["progressMap"],
                [
                    {"progress": 0.0, "value": 0.0},
                    {"progress": 0.5, "value": 0.45},
                    {"progress": 1.0, "value": 1.0},
                ],
            )

    def test_build_applies_transition_track_progress_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pptx = tmp_path / "demo.pptx"
            out = tmp_path / "site"
            _write_demo_pptx(pptx)
            build_presentation(
                pptx,
                out,
                PresenterConfig(
                    morph_policy=MorphPolicy(
                        transition_track_progress_overrides=(
                            {
                                "from": 1,
                                "to": 2,
                                "track_id": "track-0001",
                                "points": [
                                    {"progress": 0.0, "value": 0.0},
                                    {"progress": 0.5, "value": 0.6},
                                    {"progress": 1.0, "value": 1.0},
                                ],
                            },
                        )
                    )
                ),
            )
            scene = json.loads((out / "deck.scene.json").read_text(encoding="utf-8"))
            self.assertEqual(
                scene["transitions"][0]["trackProgressOverrides"],
                [
                    {
                        "trackId": "track-0001",
                        "points": [
                            {"progress": 0.0, "value": 0.0},
                            {"progress": 0.5, "value": 0.6},
                            {"progress": 1.0, "value": 1.0},
                        ],
                    }
                ],
            )

    def test_build_emits_auto_advance_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pptx = tmp_path / "demo.pptx"
            out = tmp_path / "site"
            _write_demo_pptx(pptx)
            build_presentation(
                pptx,
                out,
                PresenterConfig(
                    auto_advance=(
                        {
                            "from": 1,
                            "to": 2,
                            "delay_sec": 0.25,
                            "source": "test",
                        },
                    )
                ),
            )
            scene = json.loads((out / "deck.scene.json").read_text(encoding="utf-8"))
            self.assertEqual(
                scene["runtime"]["autoAdvance"],
                [{"from": 1, "to": 2, "delaySec": 0.25, "source": "test"}],
            )

    def test_player_phases_loop_media_without_slide_timed_global_offset(self) -> None:
        self.assertIn("node.dataset.mediaLoopSignature !== mediaSignature", PLAYER_HTML)
        self.assertIn("seekMedia(media, mediaStartTime(state));", PLAYER_HTML)
        self.assertIn("function mediaLoopTime(media, seconds)", PLAYER_HTML)
        self.assertIn("if (isSlideTimedMedia(state))", PLAYER_HTML)
        self.assertIn("slideTimedVideoPhaseSec", PLAYER_HTML)
        self.assertLess(
            PLAYER_HTML.index("let start = Number(timing.startSec || 0)"),
            PLAYER_HTML.index("slideTimedVideoPhaseSec"),
        )
        self.assertIn("((desired % duration) + duration) % duration", PLAYER_HTML)

    def test_player_supports_per_transition_unmatched_fade(self) -> None:
        self.assertIn("transition?.unmatchedFade", PLAYER_HTML)
        self.assertIn('direction === "exit" ? "exit" : "enter"', PLAYER_HTML)
        self.assertIn("fade.tracks[trackId]", PLAYER_HTML)
        self.assertIn("fade[`${prefix}Start`]", PLAYER_HTML)
        self.assertIn("function transitionWithCaptureOverrides", PLAYER_HTML)
        self.assertIn("captureOptions?.unmatchedFadeOverride", PLAYER_HTML)
        self.assertIn(
            "unmatchedFadeOverride: s.unmatchedFadeOverride || null",
            (ROOT / "pptx_html_presenter" / "browser_capture.mjs").read_text(encoding="utf-8"),
        )

    def test_player_does_not_double_apply_glow_to_text(self) -> None:
        self.assertIn('const isText = child.classList.contains("text");', PLAYER_HTML)
        self.assertIn("if (dropShadow && !isText) filters.push(dropShadow);", PLAYER_HTML)
        self.assertIn("child.style.textShadow = cssGlowTextShadow(glow, captureOptions);", PLAYER_HTML)
        self.assertIn("captureOptions?.visualEffectOverrides?.glowScale", PLAYER_HTML)
        self.assertIn("captureOptions?.visualEffectOverrides?.glowAlphaScale", PLAYER_HTML)
        self.assertIn(
            "visualEffectOverrides: s.visualEffectOverrides || null",
            (ROOT / "pptx_html_presenter" / "browser_capture.mjs").read_text(encoding="utf-8"),
        )

    def test_player_supports_configurable_transition_easing(self) -> None:
        self.assertIn("function easeForTransition", PLAYER_HTML)
        self.assertIn("transition?.easing ?? runtime.easing", PLAYER_HTML)
        self.assertIn("cubicBezierEase", PLAYER_HTML)
        self.assertIn("powerEase", PLAYER_HTML)

    def test_player_supports_transition_progress_map(self) -> None:
        self.assertIn("function interpolationProgress", PLAYER_HTML)
        self.assertIn("function progressMapValue", PLAYER_HTML)
        self.assertIn("transition?.progressMap", PLAYER_HTML)

    def test_player_supports_track_progress_overrides(self) -> None:
        self.assertIn("function trackInterpolationProgress", PLAYER_HTML)
        self.assertIn("transition?.trackProgressOverrides", PLAYER_HTML)
        self.assertIn("trackProgressOverrides: s.trackProgressOverrides || null", (ROOT / "pptx_html_presenter" / "browser_capture.mjs").read_text(encoding="utf-8"))

    def test_browser_capture_reports_stable_diagnostic_asset_paths(self) -> None:
        capture_js = (ROOT / "pptx_html_presenter" / "browser_capture.mjs").read_text(encoding="utf-8")
        self.assertIn("const stableSrc = (value) =>", capture_js)
        self.assertIn("return `${url.pathname}${url.search}${url.hash}`;", capture_js)
        self.assertIn("src: stableSrc(image.currentSrc || image.src)", capture_js)
        self.assertIn("src: stableSrc(video.currentSrc || video.src)", capture_js)

    def test_player_hides_settled_only_raster_fallbacks_during_transition(self) -> None:
        self.assertIn("function isSettledOnlyRasterFallback", PLAYER_HTML)
        self.assertIn("hideSettledOnlyFallbacks();", PLAYER_HTML)
        self.assertIn("!isSettledOnlyRasterFallback(obj)", PLAYER_HTML)
        self.assertIn('node.dataset.settledOnly === "1"', PLAYER_HTML)

    def test_player_keeps_visible_video_clock_during_transition_to_paused_copy(self) -> None:
        self.assertIn("function transitionMediaObject", PLAYER_HTML)
        self.assertIn("function transitionMediaPhaseOverride", PLAYER_HTML)
        self.assertIn("transition?.mediaPhaseOverrides", PLAYER_HTML)
        self.assertIn("function isAnimatedLoopAsset", PLAYER_HTML)
        self.assertIn("isAnimatedLoopAsset(fromAsset)", PLAYER_HTML)
        self.assertIn("function hasExplicitMediaPhase", PLAYER_HTML)
        self.assertIn("hasExplicitMediaPhase(to)", PLAYER_HTML)
        self.assertIn("isPausedMedia(from) && !isPausedMedia(to) && !isStateVisible(from)", PLAYER_HTML)
        self.assertIn("state.mediaTiming = structuredClone(mediaObj.mediaTiming || {})", PLAYER_HTML)

    def test_player_renders_paused_media_with_poster_asset(self) -> None:
        self.assertIn("function renderAssetForObject", PLAYER_HTML)
        self.assertIn("isPausedMedia(obj) && posterAsset?.file", PLAYER_HTML)
        self.assertIn("video.poster = posterAsset.file", PLAYER_HTML)

    def test_player_supports_inferred_panel_motion(self) -> None:
        self.assertIn("function inferredMotionMap", PLAYER_HTML)
        self.assertIn("function applyInferredMotion", PLAYER_HTML)
        self.assertIn("syntheticFrom.geometry = structuredClone(inferred.fromGeometry)", PLAYER_HTML)
        self.assertIn("inferred.preserveOpacity", PLAYER_HTML)
        self.assertIn("function slideObjects", PLAYER_HTML)
        self.assertIn("function attachNode", PLAYER_HTML)
        self.assertIn("function parentTrackForObject", PLAYER_HTML)
        self.assertIn("return fromParent === toParent ? fromParent : null", PLAYER_HTML)
        self.assertIn("state.localGeometry", PLAYER_HTML)
        self.assertIn("node.dataset.nodeRole === \"panel\"", PLAYER_HTML)
        self.assertIn("node.dataset.nodeRole === \"group\"", PLAYER_HTML)
        self.assertIn("obj-children", PLAYER_HTML)
        self.assertIn("panel-fill", PLAYER_HTML)
        self.assertIn("function createPanelOutlineChild", PLAYER_HTML)
        self.assertIn("function createOutlineChild", PLAYER_HTML)
        self.assertIn("function outlineModeForObject", PLAYER_HTML)
        self.assertIn("function applyOutlinedObjectState", PLAYER_HTML)
        self.assertIn("function normalizedStrokeWidthPct", PLAYER_HTML)
        self.assertIn("function isWhiteStroke", PLAYER_HTML)
        self.assertIn("node.dataset.outlineMode === \"top\"", PLAYER_HTML)
        self.assertIn("panel-outline-shape", PLAYER_HTML)

    def test_player_supports_reverse_morph_navigation(self) -> None:
        self.assertIn("runTransition(currentIndex, currentIndex - 1)", PLAYER_HTML)
        self.assertIn("reverse: true", PLAYER_HTML)
        self.assertIn("transition?.reverse", PLAYER_HTML)
        self.assertIn("mirroredProgressMapValue", PLAYER_HTML)
        self.assertIn("options?.direction === \"reverse\"", PLAYER_HTML)

    def test_player_supports_auto_advance_rules(self) -> None:
        self.assertIn("let autoAdvanceTimer = null", PLAYER_HTML)
        self.assertIn("function scheduleAutoAdvance", PLAYER_HTML)
        self.assertIn("function autoAdvanceRule", PLAYER_HTML)
        self.assertIn("scene?.runtime?.autoAdvance", PLAYER_HTML)
        self.assertIn("scene?.runtime?.autoSegments", PLAYER_HTML)
        self.assertIn("options.autoAdvance === false", PLAYER_HTML)
        self.assertIn("options.direction === \"reverse\"", PLAYER_HTML)

    def test_publish_upsert_detects_quoted_hyphenated_deck_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "decks.js"
            registry.write_text(
                """
(function () {
  window.PRESENTATION_DECKS = Object.freeze({
    "BBD26-scene": Object.freeze({
      id: "BBD26-scene",
      title: "Scene Player",
    }),
  });
})();
""",
                encoding="utf-8",
            )
            self.assertFalse(_upsert_shared_deck(registry, "BBD26-scene"))
            self.assertEqual(registry.read_text(encoding="utf-8").count('"BBD26-scene"'), 2)
            self.assertTrue(_upsert_shared_deck(registry, "New-Deck"))
            self.assertEqual(registry.read_text(encoding="utf-8").count("New-Deck"), 6)

    def test_qa_samples_include_media_clock(self) -> None:
        scene = {
            "assets": [{"id": "asset-video", "kind": "video"}],
            "slides": [
                {
                    "index": 1,
                    "objects": [
                        {"trackId": "track-1", "assetId": "asset-video", "kind": "image"},
                    ],
                },
                {
                    "index": 2,
                    "objects": [
                        {"trackId": "track-1", "assetId": "asset-video", "kind": "image"},
                        {"trackId": "track-2", "assetId": "asset-video", "kind": "image"},
                    ],
                },
            ],
            "transitions": [{"from": 1, "to": 2, "durationSec": 2.0}],
        }
        samples = _sample_plan(scene)
        self.assertEqual(samples[0]["id"], "slide-001-settled")
        self.assertEqual(samples[0]["mediaSec"], samples[0]["referenceSec"])
        self.assertEqual(samples[0]["mediaClocks"], {"track-1": 0.12})
        transition_sample = next(sample for sample in samples if sample["id"] == "trans-001-002-050")
        self.assertEqual(transition_sample["referenceSec"], 2.0)
        self.assertEqual(transition_sample["mediaSec"], transition_sample["referenceSec"])
        self.assertEqual(transition_sample["mediaClocks"], {"track-1": 2.0, "track-2": 1.0})
        slide_2 = next(sample for sample in samples if sample["id"] == "slide-002-settled")
        self.assertEqual(slide_2["mediaClocks"], {"track-1": 3.12, "track-2": 2.12})

    def test_visual_audit_sample_plan_includes_reverse_midpoints(self) -> None:
        scene = {
            "assets": [],
            "qa": {"slideHoldSec": 1.0, "settledOffsetSec": 0.12},
            "slides": [
                {"index": 1, "objects": []},
                {"index": 2, "objects": []},
            ],
            "transitions": [{"from": 1, "to": 2, "durationSec": 2.0}],
        }
        samples = _visual_audit_sample_plan(scene, samples=(0.0, 0.5, 1.0), reverse_midpoints=True)
        ids = {sample["id"] for sample in samples}
        self.assertIn("slide-001-settled", ids)
        self.assertIn("trans-001-002-050", ids)
        self.assertIn("reverse-002-001-050", ids)
        reverse = next(sample for sample in samples if sample["id"] == "reverse-002-001-050")
        self.assertEqual(reverse["direction"], "reverse")
        self.assertEqual(reverse["from"], 2)
        self.assertEqual(reverse["to"], 1)

    def test_qa_samples_start_slide_timed_video_even_when_offscreen(self) -> None:
        visible = {"leftPct": 0.1, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        offscreen = {"leftPct": 1.2, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        scene = {
            "qa": {"slideHoldSec": 3.0, "settledOffsetSec": 0.5, "transitionSamples": [0, 0.5, 1]},
            "assets": [{"id": "asset-video", "kind": "video"}],
            "slides": [
                {
                    "index": 1,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": offscreen,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0},
                        },
                    ],
                },
                {
                    "index": 2,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": visible,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0},
                        },
                    ],
                },
            ],
            "transitions": [{"from": 1, "to": 2, "durationSec": 2.0}],
        }
        samples = _sample_plan(scene)
        self.assertEqual(next(sample for sample in samples if sample["id"] == "slide-001-settled")["mediaClocks"], {"track-1": 0.5})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-050")["mediaClocks"], {"track-1": 4.0})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "slide-002-settled")["mediaClocks"], {"track-1": 0.5})

    def test_qa_samples_keep_hidden_loop_media_clock_running(self) -> None:
        offscreen = {"leftPct": 1.2, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        scene = {
            "qa": {"slideHoldSec": 3.0, "settledOffsetSec": 0.5, "transitionSamples": [0, 0.5, 1]},
            "assets": [{"id": "asset-video", "kind": "video"}],
            "slides": [
                {
                    "index": 1,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "image",
                            "geometry": offscreen,
                        },
                    ],
                },
                {
                    "index": 2,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "image",
                            "geometry": offscreen,
                        },
                    ],
                },
            ],
            "transitions": [{"from": 1, "to": 2, "durationSec": 2.0}],
        }
        samples = _sample_plan(scene)
        self.assertEqual(next(sample for sample in samples if sample["id"] == "slide-001-settled")["mediaClocks"], {"track-1": 0.5})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-050")["mediaClocks"], {"track-1": 4.0})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "slide-002-settled")["mediaClocks"], {"track-1": 5.5})

    def test_qa_samples_freeze_paused_media_clock(self) -> None:
        visible = {"leftPct": 0.1, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        scene = {
            "qa": {"slideHoldSec": 3.0, "settledOffsetSec": 0.5, "transitionSamples": [0, 0.5, 1]},
            "assets": [{"id": "asset-video", "kind": "video"}],
            "slides": [
                {
                    "index": 1,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": visible,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "paused": True},
                        },
                    ],
                },
                {
                    "index": 2,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": visible,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "paused": True},
                        },
                    ],
                },
            ],
            "transitions": [{"from": 1, "to": 2, "durationSec": 2.0}],
        }
        samples = _sample_plan(scene)
        self.assertEqual(next(sample for sample in samples if sample["id"] == "slide-001-settled")["mediaClocks"], {"track-1": 0.0})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-050")["mediaClocks"], {"track-1": 0.0})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "slide-002-settled")["mediaClocks"], {"track-1": 0.0})

    def test_qa_samples_treat_paused_poster_media_as_image(self) -> None:
        visible = {"leftPct": 0.1, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        scene = {
            "qa": {"slideHoldSec": 3.0, "settledOffsetSec": 0.5},
            "assets": [
                {"id": "asset-video", "kind": "video"},
                {"id": "asset-poster", "kind": "image", "file": "assets/source/poster.png"},
            ],
            "slides": [
                {
                    "index": 1,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "posterAssetId": "asset-poster",
                            "kind": "video",
                            "geometry": visible,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "paused": True},
                        },
                    ],
                }
            ],
            "transitions": [],
        }
        samples = _sample_plan(scene)
        self.assertEqual(samples[0]["mediaClocks"], {})

    def test_media_phase_targets_persistent_loop_origin(self) -> None:
        scene = {
            "slides": [
                {
                    "index": 1,
                    "objects": [
                        {
                            "id": "s1-o1",
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "name": "Loop",
                            "mediaTiming": {"phaseSec": 0.25},
                        }
                    ],
                },
                {
                    "index": 2,
                    "objects": [
                        {
                            "id": "s2-o1",
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "name": "Loop",
                            "mediaTiming": {},
                        }
                    ],
                },
            ]
        }
        observed = scene["slides"][1]["objects"][0]
        target = _media_phase_target(scene, 2, observed, "asset-video")
        self.assertEqual(target["slide"], 1)
        self.assertEqual(target["object"]["id"], "s1-o1")
        self.assertEqual(target["reason"], "persistent-loop-origin")
        overrides = _media_phase_config_overrides(
            [
                {
                    "slide": 2,
                    "objectId": "s2-o1",
                    "trackId": "track-1",
                    "assetId": "asset-video",
                    "name": "Loop",
                    "targetSlide": target["slide"],
                    "targetObjectId": target["object"]["id"],
                    "targetTrackId": target["object"]["trackId"],
                    "targetAssetId": target["object"]["assetId"],
                    "targetName": target["object"]["name"],
                    "targetReason": target["reason"],
                    "recommendedPhaseSec": -1.75,
                    "score": 0.9,
                }
            ],
            min_score=0.7,
        )
        self.assertEqual(overrides[0]["slide"], 1)
        self.assertEqual(overrides[0]["object_id"], "s1-o1")
        self.assertEqual(overrides[0]["observed_slide"], 2)
        self.assertEqual(overrides[0]["target_reason"], "persistent-loop-origin")

    def test_media_phase_overrides_group_duplicate_target_recommendations(self) -> None:
        rows = [
            {
                "slide": 1,
                "objectId": "s1-o1",
                "trackId": "track-1",
                "assetId": "asset-video",
                "name": "Loop",
                "targetSlide": 1,
                "targetObjectId": "s1-o1",
                "targetTrackId": "track-1",
                "targetAssetId": "asset-video",
                "targetName": "Loop",
                "recommendedPhaseSec": 1.2,
                "score": 0.9,
            },
            {
                "slide": 1,
                "objectId": "s1-o1",
                "trackId": "track-1",
                "assetId": "asset-video",
                "name": "Loop",
                "targetSlide": 1,
                "targetObjectId": "s1-o1",
                "targetTrackId": "track-1",
                "targetAssetId": "asset-video",
                "targetName": "Loop",
                "recommendedPhaseSec": -0.4,
                "score": 0.6,
            },
            {
                "slide": 1,
                "objectId": "s1-o1",
                "trackId": "track-1",
                "assetId": "asset-video",
                "name": "Loop",
                "targetSlide": 1,
                "targetObjectId": "s1-o1",
                "targetTrackId": "track-1",
                "targetAssetId": "asset-video",
                "targetName": "Loop",
                "recommendedPhaseSec": 1.2,
                "score": 0.88,
            },
        ]
        overrides = _media_phase_config_overrides(rows, min_score=0.55)
        self.assertEqual(len(overrides), 1)
        self.assertEqual(overrides[0]["phase_sec"], 1.2)
        self.assertEqual(overrides[0]["sample_count"], 3)

    def test_transition_media_phase_overrides_are_separate_from_settled_overrides(self) -> None:
        rows = [
            {
                "kind": "transition",
                "from": 18,
                "to": 19,
                "slide": 18,
                "objectId": "s19-o10",
                "trackId": "track-0086",
                "assetId": "asset-video",
                "name": "Panel video",
                "targetObjectId": "s19-o10",
                "targetTrackId": "track-0086",
                "targetAssetId": "asset-video",
                "targetName": "Panel video",
                "recommendedPhaseSec": 1.0,
                "recommendedTransitionPhaseSec": -0.115,
                "score": 0.9,
            },
            {
                "kind": "transition",
                "from": 18,
                "to": 19,
                "slide": 18,
                "objectId": "s19-o10",
                "trackId": "track-0086",
                "assetId": "asset-video",
                "name": "Panel video",
                "targetObjectId": "s19-o10",
                "targetTrackId": "track-0086",
                "targetAssetId": "asset-video",
                "targetName": "Panel video",
                "recommendedPhaseSec": 1.0,
                "recommendedTransitionPhaseSec": -0.2,
                "score": 0.8,
            },
        ]
        self.assertEqual(_media_phase_config_overrides(rows, min_score=0.55), [])
        transition_overrides = _transition_media_phase_config_overrides(rows, min_score=0.55)
        self.assertEqual(len(transition_overrides), 1)
        self.assertEqual(transition_overrides[0]["from"], 18)
        self.assertEqual(transition_overrides[0]["to"], 19)
        self.assertEqual(transition_overrides[0]["track_id"], "track-0086")
        self.assertEqual(transition_overrides[0]["phase_sec"], -0.158)
        self.assertEqual(transition_overrides[0]["sample_count"], 2)

    def test_transition_media_phase_overrides_skip_contradictory_phase_rows(self) -> None:
        rows = [
            {
                "kind": "transition",
                "from": 19,
                "to": 20,
                "trackId": "track-0086",
                "assetId": "asset-video",
                "recommendedTransitionPhaseSec": 0.885,
                "score": 0.91,
            },
            {
                "kind": "transition",
                "from": 19,
                "to": 20,
                "trackId": "track-0086",
                "assetId": "asset-video",
                "recommendedTransitionPhaseSec": -3.365,
                "score": 0.94,
            },
        ]
        self.assertEqual(_transition_media_phase_config_overrides(rows, min_score=0.7), [])

    def test_qa_samples_keep_visible_video_clock_during_transition_to_paused_copy(self) -> None:
        visible = {"leftPct": 0.1, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        scene = {
            "qa": {"slideHoldSec": 3.0, "settledOffsetSec": 0.5, "transitionSamples": [0, 0.5, 1]},
            "assets": [{"id": "asset-video", "kind": "video"}],
            "slides": [
                {
                    "index": 1,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": visible,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0},
                        },
                    ],
                },
                {
                    "index": 2,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": visible,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "paused": True},
                        },
                    ],
                },
            ],
            "transitions": [{"from": 1, "to": 2, "durationSec": 2.0}],
        }
        samples = _sample_plan(scene)
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-000")["mediaClocks"], {"track-1": 3.0})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-050")["mediaClocks"], {"track-1": 4.0})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "slide-002-settled")["mediaClocks"], {"track-1": 0.0})

    def test_qa_samples_keep_offscreen_animated_video_clock_during_transition_to_paused_copy(self) -> None:
        visible = {"leftPct": 0.1, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        offscreen = {"leftPct": 1.2, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        scene = {
            "qa": {"slideHoldSec": 3.0, "settledOffsetSec": 0.5, "transitionSamples": [0, 0.5, 1]},
            "assets": [{"id": "asset-video", "kind": "video", "animated": True, "sourceFile": "assets/source/loop.gif"}],
            "slides": [
                {
                    "index": 1,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": offscreen,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0},
                        },
                    ],
                },
                {
                    "index": 2,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": visible,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "paused": True},
                        },
                    ],
                },
            ],
            "transitions": [{"from": 1, "to": 2, "durationSec": 2.0}],
        }
        samples = _sample_plan(scene)
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-000")["mediaClocks"], {"track-1": 3.0})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-050")["mediaClocks"], {"track-1": 4.0})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "slide-002-settled")["mediaClocks"], {"track-1": 0.0})

    def test_qa_samples_use_poster_for_offscreen_plain_video_transition_to_paused_copy(self) -> None:
        visible = {"leftPct": 0.1, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        offscreen = {"leftPct": 1.2, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        scene = {
            "qa": {"slideHoldSec": 3.0, "settledOffsetSec": 0.5, "transitionSamples": [0, 0.5, 1]},
            "assets": [
                {"id": "asset-video", "kind": "video", "animated": False, "sourceFile": "assets/source/movie.mp4"},
                {"id": "asset-poster", "kind": "image", "file": "assets/source/movie-poster.png"},
            ],
            "slides": [
                {
                    "index": 1,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": offscreen,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0},
                        },
                    ],
                },
                {
                    "index": 2,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "posterAssetId": "asset-poster",
                            "kind": "video",
                            "geometry": visible,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "paused": True},
                        },
                    ],
                },
            ],
            "transitions": [{"from": 1, "to": 2, "durationSec": 2.0}],
        }
        samples = _sample_plan(scene)
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-000")["mediaClocks"], {})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-050")["mediaClocks"], {})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "slide-002-settled")["mediaClocks"], {})

    def test_qa_samples_use_live_incoming_video_when_previous_copy_is_paused_offscreen(self) -> None:
        visible = {"leftPct": 0.1, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        offscreen = {"leftPct": 1.2, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        scene = {
            "qa": {"slideHoldSec": 3.0, "settledOffsetSec": 0.5, "transitionSamples": [0, 0.5, 1]},
            "assets": [{"id": "asset-video", "kind": "video", "animated": False, "sourceFile": "assets/source/movie.mp4"}],
            "slides": [
                {
                    "index": 1,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": offscreen,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "paused": True},
                        },
                    ],
                },
                {
                    "index": 2,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": visible,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "phaseSec": 1.2},
                        },
                    ],
                },
            ],
            "transitions": [{"from": 1, "to": 2, "durationSec": 2.0}],
        }
        samples = _sample_plan(scene)
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-000")["mediaClocks"], {"track-1": 1.2})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-050")["mediaClocks"], {"track-1": 2.2})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "slide-002-settled")["mediaClocks"], {"track-1": 1.7})

    def test_qa_samples_apply_transition_scoped_media_phase_only_during_morph(self) -> None:
        visible = {"leftPct": 0.1, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        offscreen = {"leftPct": 1.2, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        scene = {
            "qa": {"slideHoldSec": 3.0, "settledOffsetSec": 0.5, "transitionSamples": [0, 0.5]},
            "assets": [{"id": "asset-video", "kind": "video", "animated": False, "sourceFile": "assets/source/movie.mp4"}],
            "slides": [
                {
                    "index": 1,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": offscreen,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "paused": True},
                        },
                    ],
                },
                {
                    "index": 2,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": visible,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "phaseSec": 1.0},
                        },
                    ],
                },
            ],
            "transitions": [
                {
                    "from": 1,
                    "to": 2,
                    "durationSec": 2.0,
                    "mediaPhaseOverrides": [{"trackId": "track-1", "phaseSec": -0.115}],
                }
            ],
        }
        samples = _sample_plan(scene)
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-000")["mediaClocks"], {"track-1": -0.115})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-050")["mediaClocks"], {"track-1": 0.885})
        self.assertEqual(next(sample for sample in samples if sample["id"] == "slide-002-settled")["mediaClocks"], {"track-1": 1.5})

    def test_transition_effective_media_objects_use_runtime_progress_and_source(self) -> None:
        from_slide = {
            "index": 1,
            "objects": [
                {
                    "id": "s1-o1",
                    "trackId": "track-1",
                    "assetId": "asset-video",
                    "kind": "video",
                    "geometry": {"leftPct": 1.2, "topPct": 0.0, "widthPct": 0.5, "heightPct": 0.5},
                    "mediaTiming": {"kind": "playFrom", "startSec": 0.0},
                }
            ],
        }
        to_slide = {
            "index": 2,
            "objects": [
                {
                    "id": "s2-o1",
                    "trackId": "track-1",
                    "assetId": "asset-video",
                    "kind": "video",
                    "geometry": {"leftPct": 0.2, "topPct": 0.0, "widthPct": 0.5, "heightPct": 0.5},
                    "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "paused": True},
                }
            ],
        }
        rows = _transition_effective_media_objects(
            from_slide,
            to_slide,
            {"from": 1, "to": 2, "progressMap": [{"progress": 0.0, "value": 0.0}, {"progress": 1.0, "value": 0.25}]},
            1.0,
            {"asset-video": {"id": "asset-video", "kind": "video", "animated": True, "sourceFile": "loop.gif"}},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sourceSlide"], 1)
        self.assertEqual(rows[0]["sourceObject"]["id"], "s1-o1")
        self.assertAlmostEqual(rows[0]["object"]["geometry"]["leftPct"], 0.95)

    def test_transition_effective_media_objects_use_incoming_live_copy_over_paused_offscreen_copy(self) -> None:
        from_slide = {
            "index": 1,
            "objects": [
                {
                    "id": "s1-o1",
                    "trackId": "track-1",
                    "assetId": "asset-video",
                    "kind": "video",
                    "geometry": {"leftPct": 1.2, "topPct": 0.0, "widthPct": 0.5, "heightPct": 0.5},
                    "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "paused": True},
                }
            ],
        }
        to_slide = {
            "index": 2,
            "objects": [
                {
                    "id": "s2-o1",
                    "trackId": "track-1",
                    "assetId": "asset-video",
                    "kind": "video",
                    "geometry": {"leftPct": 0.2, "topPct": 0.0, "widthPct": 0.5, "heightPct": 0.5},
                    "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "phaseSec": 1.0},
                }
            ],
        }
        rows = _transition_effective_media_objects(
            from_slide,
            to_slide,
            {
                "from": 1,
                "to": 2,
                "mediaPhaseOverrides": [{"trackId": "track-1", "phaseSec": -0.115}],
            },
            0.5,
            {"asset-video": {"id": "asset-video", "kind": "video", "animated": False, "sourceFile": "movie.mp4"}},
        )
        self.assertEqual(rows[0]["sourceSlide"], 2)
        self.assertEqual(rows[0]["sourceObject"]["id"], "s2-o1")
        self.assertEqual(rows[0]["object"]["mediaTiming"]["phaseSec"], -0.115)

    def test_transition_effective_media_objects_use_incoming_phased_video_when_previous_copy_offscreen(self) -> None:
        from_slide = {
            "index": 1,
            "objects": [
                {
                    "id": "s1-o1",
                    "trackId": "track-1",
                    "assetId": "asset-video",
                    "kind": "video",
                    "geometry": {"leftPct": 1.2, "topPct": 0.0, "widthPct": 0.5, "heightPct": 0.5},
                    "mediaTiming": {"kind": "playFrom", "startSec": 0.0},
                }
            ],
        }
        to_slide = {
            "index": 2,
            "objects": [
                {
                    "id": "s2-o1",
                    "trackId": "track-1",
                    "assetId": "asset-video",
                    "kind": "video",
                    "geometry": {"leftPct": 0.2, "topPct": 0.0, "widthPct": 0.5, "heightPct": 0.5},
                    "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "phaseSec": 2.0},
                }
            ],
        }
        rows = _transition_effective_media_objects(
            from_slide,
            to_slide,
            {"from": 1, "to": 2},
            0.5,
            {"asset-video": {"id": "asset-video", "kind": "video", "animated": False, "sourceFile": "movie.mp4"}},
        )
        self.assertEqual(rows[0]["sourceSlide"], 2)
        self.assertEqual(rows[0]["sourceObject"]["id"], "s2-o1")
        self.assertEqual(rows[0]["object"]["mediaTiming"]["phaseSec"], 2.0)

    def test_transition_effective_media_objects_keep_gif_loop_source_even_when_incoming_has_phase(self) -> None:
        from_slide = {
            "index": 1,
            "objects": [
                {
                    "id": "s1-o1",
                    "trackId": "track-1",
                    "assetId": "asset-video",
                    "kind": "video",
                    "geometry": {"leftPct": 1.2, "topPct": 0.0, "widthPct": 0.5, "heightPct": 0.5},
                    "mediaTiming": {"kind": "playFrom", "startSec": 0.0},
                }
            ],
        }
        to_slide = {
            "index": 2,
            "objects": [
                {
                    "id": "s2-o1",
                    "trackId": "track-1",
                    "assetId": "asset-video",
                    "kind": "video",
                    "geometry": {"leftPct": 0.2, "topPct": 0.0, "widthPct": 0.5, "heightPct": 0.5},
                    "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "phaseSec": 2.0},
                }
            ],
        }
        rows = _transition_effective_media_objects(
            from_slide,
            to_slide,
            {"from": 1, "to": 2},
            0.5,
            {"asset-video": {"id": "asset-video", "kind": "video", "animated": True, "sourceFile": "loop.gif"}},
        )
        self.assertEqual(rows[0]["sourceSlide"], 1)
        self.assertEqual(rows[0]["sourceObject"]["id"], "s1-o1")

    def test_visible_asset_patch_crops_same_offscreen_slice_as_viewport(self) -> None:
        from PIL import Image

        frame = Image.new("RGB", (100, 10), "red")
        for x in range(50, 100):
            for y in range(10):
                frame.putpixel((x, y), (0, 255, 0))
        obj = {
            "geometry": {
                "leftPct": -0.5,
                "topPct": 0.0,
                "widthPct": 1.0,
                "heightPct": 1.0,
            }
        }
        patch = _visible_asset_patch(frame, obj, (0, 0, 50, 10), (100, 10))
        self.assertEqual(patch.size, (50, 10))
        self.assertEqual(patch.getpixel((25, 5)), (0, 255, 0))

    def test_qa_samples_apply_per_object_phase_offset(self) -> None:
        visible = {"leftPct": 0.1, "topPct": 0.1, "widthPct": 0.5, "heightPct": 0.5}
        scene = {
            "assets": [{"id": "asset-video", "kind": "video"}],
            "slides": [
                {
                    "index": 1,
                    "objects": [
                        {
                            "trackId": "track-1",
                            "assetId": "asset-video",
                            "kind": "video",
                            "geometry": visible,
                            "mediaTiming": {"kind": "playFrom", "startSec": 0.0, "phaseSec": 1.25},
                        },
                    ],
                }
            ],
            "transitions": [],
        }
        samples = _sample_plan(scene)
        self.assertEqual(samples[0]["mediaClocks"], {"track-1": 1.37})

    def test_parse_slide_filter(self) -> None:
        self.assertEqual(_parse_slide_filter("1,3-5,8"), {1, 3, 4, 5, 8})
        self.assertIsNone(_parse_slide_filter(None))

    def test_filter_samples_for_slides_keeps_slide_and_outgoing_transition(self) -> None:
        samples = [
            {"id": "slide-001-settled", "kind": "slide", "slide": 1},
            {"id": "trans-001-002-050", "kind": "transition", "from": 1, "to": 2},
            {"id": "slide-002-settled", "kind": "slide", "slide": 2},
            {"id": "trans-002-003-050", "kind": "transition", "from": 2, "to": 3},
        ]
        filtered = _filter_samples_for_slides(samples, {2})
        self.assertEqual(
            [sample["id"] for sample in filtered],
            ["slide-002-settled", "trans-002-003-050"],
        )

    def test_qa_samples_read_scene_timing_profile(self) -> None:
        scene = {
            "qa": {"slideHoldSec": 3.0, "settledOffsetSec": 0.5, "transitionSamples": [0, 0.5, 1]},
            "slides": [{"index": 1}, {"index": 2}],
            "transitions": [{"from": 1, "to": 2, "durationSec": 2.0}],
        }
        samples = _sample_plan(scene)
        self.assertEqual(samples[0]["referenceSec"], 0.5)
        transition_sample = next(sample for sample in samples if sample["id"] == "trans-001-002-050")
        self.assertEqual(transition_sample["referenceSec"], 4.0)
        self.assertEqual(samples[-1]["referenceSec"], 5.5)

    def test_qa_samples_can_lead_powerpoint_reference_transition(self) -> None:
        scene = {
            "qa": {
                "slideHoldSec": 3.0,
                "settledOffsetSec": 0.5,
                "transitionSamples": [0, 0.5, 1],
                "transitionReferenceLeadFraction": 0.5,
            },
            "slides": [{"index": 1}, {"index": 2}],
            "transitions": [{"from": 1, "to": 2, "durationSec": 2.0}],
        }
        samples = _sample_plan(scene)
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-000")["referenceSec"], 2.0)
        transition_mid = next(sample for sample in samples if sample["id"] == "trans-001-002-050")
        self.assertEqual(transition_mid["referenceSec"], 3.0)
        self.assertEqual(transition_mid["mediaSec"], 3.0)
        self.assertEqual(next(sample for sample in samples if sample["id"] == "trans-001-002-100")["referenceSec"], 4.0)

    def test_qa_samples_apply_transition_time_override(self) -> None:
        scene = {
            "qa": {
                "slideHoldSec": 3.0,
                "settledOffsetSec": 0.5,
                "transitionSamples": [0, 0.5, 1],
                "transitionReferenceLeadFraction": 1.0,
                "transitionTimeOverrides": [
                    {
                        "from": 1,
                        "to": 2,
                        "reference_offset_sec": -0.25,
                        "progress_offsets": [
                            {"progress": 0.5, "reference_offset_sec": 0.4}
                        ],
                    }
                ],
            },
            "slides": [{"index": 1}, {"index": 2}],
            "transitions": [{"from": 1, "to": 2, "durationSec": 2.0}],
        }
        samples = _sample_plan(scene)
        transition_mid = next(sample for sample in samples if sample["id"] == "trans-001-002-050")
        self.assertEqual(transition_mid["referenceSec"], 2.4)
        self.assertEqual(transition_mid["mediaSec"], 2.4)
        transition_start = next(sample for sample in samples if sample["id"] == "trans-001-002-000")
        self.assertEqual(transition_start["referenceSec"], 0.75)

    def test_transition_time_config_overrides_use_median_delta(self) -> None:
        samples = [
            {"id": "trans-001-002-000", "kind": "transition", "from": 1, "to": 2},
            {"id": "trans-001-002-050", "kind": "transition", "from": 1, "to": 2},
            {"id": "trans-001-002-100", "kind": "transition", "from": 1, "to": 2},
        ]
        rows = [
            {"sampleId": "trans-001-002-000", "kind": "transition", "alignmentDeltaSec": -0.2, "alignedSsim": 0.8},
            {"sampleId": "trans-001-002-050", "kind": "transition", "alignmentDeltaSec": -0.4, "alignedSsim": 0.9},
            {"sampleId": "trans-001-002-100", "kind": "transition", "alignmentDeltaSec": 0.6, "alignedSsim": 0.2},
        ]
        overrides = _transition_time_config_overrides(rows, samples, {(1, 2): 0.1}, 0.55)
        self.assertEqual(overrides[0]["from"], 1)
        self.assertEqual(overrides[0]["to"], 2)
        self.assertEqual(overrides[0]["reference_offset_sec"], -0.2)
        self.assertEqual(overrides[0]["sample_count"], 2)
        self.assertEqual(overrides[0]["progress_offsets"][0]["progress"], 0.0)
        self.assertEqual(overrides[0]["progress_offsets"][0]["reference_offset_sec"], -0.1)

    def test_morph_progress_candidate_samples_preserve_media_clocks(self) -> None:
        samples = [
            {
                "id": "trans-001-002-050",
                "kind": "transition",
                "from": 1,
                "to": 2,
                "progress": 0.5,
                "mediaSec": 4.2,
                "mediaClocks": {"track-video": 1.3},
            }
        ]
        candidates = _morph_progress_candidate_samples(samples, [0.0, 0.5, 1.0])
        self.assertEqual(candidates[1]["id"], "trans-001-002-050-candidate-0500")
        self.assertEqual(candidates[1]["progress"], 0.5)
        self.assertEqual(candidates[1]["mediaClocks"], {"track-video": 1.3})
        self.assertEqual(candidates[1]["sourceSampleId"], "trans-001-002-050")

    def test_candidate_sweep_progress_samples_preserve_live_media_clocks(self) -> None:
        sample = {
            "id": "trans-001-002-050",
            "kind": "transition",
            "from": 1,
            "to": 2,
            "progress": 0.5,
            "mediaSec": 4.2,
            "mediaClocks": {"track-video": 1.3},
        }
        candidates = _candidate_sweep_samples(sample, "progress", [0.25, 0.5, 0.75])
        self.assertEqual(candidates[0]["progress"], 0.25)
        self.assertEqual(candidates[1]["progress"], 0.5)
        self.assertEqual(candidates[2]["progress"], 0.75)
        self.assertEqual(candidates[2]["mediaClocks"], {"track-video": 1.3})
        self.assertEqual(candidates[2]["sourceSampleId"], "trans-001-002-050")
        self.assertEqual(sample["progress"], 0.5)

    def test_candidate_sweep_phase_samples_only_change_target_track_clock(self) -> None:
        sample = {
            "id": "trans-001-002-050",
            "kind": "transition",
            "from": 1,
            "to": 2,
            "progress": 0.5,
            "mediaSec": 4.2,
            "mediaClocks": {"track-video": 1.3, "other-video": 9.0},
        }
        candidates = _candidate_sweep_samples(sample, "phase", [-0.115, 2.0], "track-video")
        self.assertEqual(candidates[0]["progress"], 0.5)
        self.assertEqual(candidates[0]["mediaClocks"], {"track-video": -0.115, "other-video": 9.0})
        self.assertEqual(candidates[1]["mediaClocks"], {"track-video": 2.0, "other-video": 9.0})
        self.assertEqual(candidates[1]["candidateSweep"]["trackId"], "track-video")
        self.assertEqual(sample["mediaClocks"], {"track-video": 1.3, "other-video": 9.0})

    def test_candidate_sweep_phase_offset_can_shift_track_groups(self) -> None:
        sample = {
            "id": "trans-001-002-050",
            "kind": "transition",
            "from": 1,
            "to": 2,
            "progress": 0.5,
            "mediaSec": 4.2,
            "mediaClocks": {"track-a": 1.3, "track-b": 9.0, "track-c": 12.0},
        }
        candidates = _candidate_sweep_samples(sample, "phase-offset", [-1.0], "track-a,track-b")
        self.assertEqual(candidates[0]["progress"], 0.5)
        self.assertEqual(candidates[0]["mediaClocks"], {"track-a": 0.3, "track-b": 8.0, "track-c": 12.0})
        self.assertEqual(candidates[0]["candidateSweep"]["trackIds"], ["track-a", "track-b"])
        self.assertEqual(sample["mediaClocks"], {"track-a": 1.3, "track-b": 9.0, "track-c": 12.0})

    def test_candidate_sweep_phase_offset_defaults_to_all_media_clocks(self) -> None:
        sample = {
            "id": "trans-001-002-050",
            "kind": "transition",
            "from": 1,
            "to": 2,
            "progress": 0.5,
            "mediaSec": 4.2,
            "mediaClocks": {"track-a": 1.3, "track-b": 9.0},
        }
        candidates = _candidate_sweep_samples(sample, "phase-offset", [0.5])
        self.assertEqual(candidates[0]["mediaClocks"], {"track-a": 1.8, "track-b": 9.5})
        self.assertEqual(candidates[0]["candidateSweep"]["trackIds"], ["track-a", "track-b"])

    def test_candidate_sweep_output_ids_compact_long_track_clusters(self) -> None:
        track_id = ",".join(f"track-{index:04d}" for index in range(40))
        dirname = _candidate_sweep_dir_name("trans-001-002-075", "track-progress", track_id)
        candidate_id = _candidate_sweep_candidate_id("trans-001-002-075", "track-progress", 0.5, track_id)

        self.assertLess(len(dirname), 80)
        self.assertLess(len(candidate_id), 90)
        self.assertIn("tracks-40-", dirname)
        self.assertIn("tracks-40-", candidate_id)
        self.assertNotIn("track-0039", dirname)

    def test_candidate_sweep_track_progress_samples_keep_global_progress(self) -> None:
        sample = {
            "id": "trans-001-002-050",
            "kind": "transition",
            "from": 1,
            "to": 2,
            "progress": 0.5,
            "mediaSec": 4.2,
            "mediaClocks": {"track-video": 1.3},
        }
        candidates = _candidate_sweep_samples(sample, "track-progress", [0.35, 0.65], "track-video")
        self.assertEqual(candidates[0]["progress"], 0.5)
        self.assertEqual(candidates[0]["trackProgressOverrides"], {"track-video": 0.35})
        self.assertEqual(candidates[1]["trackProgressOverrides"], {"track-video": 0.65})
        self.assertEqual(candidates[1]["mediaClocks"], {"track-video": 1.3})

    def test_candidate_sweep_track_progress_can_target_track_cluster(self) -> None:
        sample = {
            "id": "trans-001-002-050",
            "kind": "transition",
            "from": 1,
            "to": 2,
            "progress": 0.5,
            "mediaSec": 4.2,
            "mediaClocks": {"track-panel": 1.3, "track-video": 1.3},
        }
        candidates = _candidate_sweep_samples(
            sample,
            "track-progress",
            [0.35],
            "track-panel,track-video",
        )
        self.assertEqual(
            candidates[0]["trackProgressOverrides"],
            {"track-panel": 0.35, "track-video": 0.35},
        )
        self.assertEqual(candidates[0]["candidateSweep"]["trackIds"], ["track-panel", "track-video"])

    def test_candidate_sweep_phase_can_target_track_cluster(self) -> None:
        sample = {
            "id": "trans-001-002-050",
            "kind": "transition",
            "from": 1,
            "to": 2,
            "progress": 0.5,
            "mediaSec": 4.2,
            "mediaClocks": {"track-panel": 1.3, "track-video": 2.4},
        }
        candidates = _candidate_sweep_samples(
            sample,
            "phase",
            [3.5],
            "track-panel,track-video",
        )
        self.assertEqual(candidates[0]["mediaClocks"], {"track-panel": 3.5, "track-video": 3.5})
        self.assertEqual(candidates[0]["candidateSweep"]["trackIds"], ["track-panel", "track-video"])

    def test_candidate_sweep_unmatched_fade_samples_keep_global_progress(self) -> None:
        sample = {
            "id": "trans-001-002-050",
            "kind": "transition",
            "from": 1,
            "to": 2,
            "progress": 0.5,
            "mediaSec": 4.2,
            "mediaClocks": {"track-video": 1.3},
        }
        enter_candidates = _candidate_sweep_samples(sample, "fade-enter-end", [0.25, 0.75])
        exit_candidates = _candidate_sweep_samples(sample, "unmatched-exit-end", [0.2])
        self.assertEqual(enter_candidates[0]["progress"], 0.5)
        self.assertEqual(enter_candidates[0]["unmatchedFadeOverride"], {"enterStart": 0.0, "enterEnd": 0.25})
        self.assertEqual(enter_candidates[1]["unmatchedFadeOverride"], {"enterStart": 0.0, "enterEnd": 0.75})
        self.assertEqual(enter_candidates[1]["candidateSweep"]["vary"], "enter-fade-end")
        self.assertEqual(enter_candidates[1]["mediaClocks"], {"track-video": 1.3})
        self.assertEqual(exit_candidates[0]["unmatchedFadeOverride"], {"exitStart": 0.0, "exitEnd": 0.2})
        self.assertEqual(exit_candidates[0]["candidateSweep"]["vary"], "exit-fade-end")

    def test_candidate_sweep_unmatched_fade_can_target_track_cluster(self) -> None:
        sample = {
            "id": "trans-001-002-050",
            "kind": "transition",
            "from": 1,
            "to": 2,
            "progress": 0.5,
            "mediaSec": 4.2,
            "mediaClocks": {"track-bg": 1.3, "track-title": 1.3},
        }
        candidates = _candidate_sweep_samples(
            sample,
            "exit-fade-end",
            [0.2],
            "track-bg,track-title",
        )
        self.assertEqual(
            candidates[0]["unmatchedFadeOverride"],
            {
                "tracks": {
                    "track-bg": {"exitStart": 0.0, "exitEnd": 0.2},
                    "track-title": {"exitStart": 0.0, "exitEnd": 0.2},
                }
            },
        )
        self.assertEqual(candidates[0]["candidateSweep"]["trackIds"], ["track-bg", "track-title"])
        self.assertEqual(candidates[0]["progress"], 0.5)

    def test_candidate_sweep_glow_samples_can_target_settled_slides(self) -> None:
        sample = {
            "id": "slide-001-settled",
            "kind": "slide",
            "slide": 1,
            "progress": 0,
            "mediaSec": 0,
        }
        radius_candidates = _candidate_sweep_samples(sample, "glow-radius", [0.0, 0.5, 1.25])
        alpha_candidates = _candidate_sweep_samples(sample, "glow-alpha-scale", [0.2])
        self.assertEqual(radius_candidates[0]["visualEffectOverrides"], {"glowScale": 0.0})
        self.assertEqual(radius_candidates[1]["visualEffectOverrides"], {"glowScale": 0.5})
        self.assertEqual(radius_candidates[2]["visualEffectOverrides"], {"glowScale": 1.25})
        self.assertEqual(radius_candidates[2]["candidateSweep"]["vary"], "glow-scale")
        self.assertEqual(alpha_candidates[0]["visualEffectOverrides"], {"glowAlphaScale": 0.2})
        self.assertEqual(alpha_candidates[0]["candidateSweep"]["vary"], "glow-alpha-scale")

    def test_candidate_sweep_text_metric_samples_can_target_settled_slides(self) -> None:
        sample = {
            "id": "slide-001-settled",
            "kind": "slide",
            "slide": 1,
            "progress": 0,
            "mediaSec": 0,
        }
        scale_candidates = _candidate_sweep_samples(sample, "font-scale", [0.86, 1.0])
        weight_candidates = _candidate_sweep_samples(sample, "text-bold-weight", [560, 640])

        self.assertEqual(scale_candidates[0]["textRenderOverrides"], {"fontScale": 0.86})
        self.assertEqual(scale_candidates[0]["candidateSweep"]["vary"], "text-scale")
        self.assertEqual(weight_candidates[1]["textRenderOverrides"], {"boldWeight": 640.0})
        self.assertEqual(weight_candidates[1]["candidateSweep"]["vary"], "bold-weight")

    def test_track_progress_candidate_samples_include_scene_baseline(self) -> None:
        sample = {
            "id": "trans-001-002-050",
            "kind": "transition",
            "from": 1,
            "to": 2,
            "progress": 0.5,
            "mediaSec": 4.2,
            "mediaClocks": {"track-video": 1.3},
        }
        transitions = {
            (1, 2): {
                "from": 1,
                "to": 2,
                "progressMap": [
                    {"progress": 0.0, "value": 0.0},
                    {"progress": 0.5, "value": 0.45},
                    {"progress": 1.0, "value": 1.0},
                ],
            }
        }
        candidates = _track_progress_candidate_samples(
            [sample],
            {"trans-001-002-050": ["track-video"]},
            transitions,
            [0.0, 0.5, 1.0],
        )
        values = [row["candidateSweep"]["value"] for row in candidates]
        self.assertEqual(values, [0.0, 0.45, 0.5, 1.0])
        self.assertEqual(candidates[1]["trackProgressOverrides"], {"track-video": 0.45})

    def test_track_progress_monotonic_selection_rejects_late_then_early_combo(self) -> None:
        rows = [
            {
                "candidateRows": [
                    {"value": 0.65, "score": 0.484},
                    {"value": 1.0, "score": 0.489},
                ]
            },
            {
                "candidateRows": [
                    {"value": 0.75, "score": 0.658},
                    {"value": 1.0, "score": 0.309},
                ]
            },
        ]
        selected = _track_progress_monotonic_selection(rows)
        self.assertEqual([choice["value"] for _row, choice in selected], [0.65, 0.75])

    def test_track_progress_config_overrides_preserve_baseline_points(self) -> None:
        rows = [
            {
                "from": 19,
                "to": 20,
                "trackId": "track-0088",
                "progress": 0.5,
                "baselineProgressValue": 0.45,
                "baselineScore": 0.478,
                "baselinePoints": [
                    {"progress": 0.0, "value": 0.0},
                    {"progress": 0.25, "value": 0.125},
                    {"progress": 0.5, "value": 0.45},
                    {"progress": 0.75, "value": 0.875},
                    {"progress": 1.0, "value": 1.0},
                ],
                "candidateRows": [
                    {"value": 0.65, "score": 0.484},
                    {"value": 1.0, "score": 0.489},
                ],
            },
            {
                "from": 19,
                "to": 20,
                "trackId": "track-0088",
                "progress": 0.75,
                "baselineProgressValue": 0.875,
                "baselineScore": 0.505,
                "baselinePoints": [
                    {"progress": 0.0, "value": 0.0},
                    {"progress": 0.25, "value": 0.125},
                    {"progress": 0.5, "value": 0.45},
                    {"progress": 0.75, "value": 0.875},
                    {"progress": 1.0, "value": 1.0},
                ],
                "candidateRows": [
                    {"value": 0.75, "score": 0.658},
                    {"value": 1.0, "score": 0.309},
                ],
            },
        ]
        overrides = _track_progress_config_overrides(rows, min_score=0.0, min_improvement=0.002)
        self.assertEqual(overrides[0]["track_id"], "track-0088")
        self.assertEqual(
            overrides[0]["points"],
            [
                {"progress": 0.0, "value": 0.0},
                {"progress": 0.25, "value": 0.125},
                {"progress": 0.5, "value": 0.65},
                {"progress": 0.75, "value": 0.75},
                {"progress": 1.0, "value": 1.0},
            ],
        )

    def test_parse_float_list_accepts_lists_and_ranges(self) -> None:
        self.assertEqual(_parse_float_list("0, 0.5,1"), [0.0, 0.5, 1.0])
        self.assertEqual(_parse_float_list("0:0.5:0.25"), [0.0, 0.25, 0.5])
        self.assertEqual(_parse_float_list("1:0:-0.5"), [1.0, 0.5, 0.0])

    def test_parse_track_filter(self) -> None:
        self.assertEqual(_parse_track_filter("track-1, track-2"), {"track-1", "track-2"})
        self.assertIsNone(_parse_track_filter(None))

    def test_neutralized_progress_scene_sets_identity_map_for_target_pairs(self) -> None:
        scene = {
            "transitions": [
                {"from": 1, "to": 2, "easing": "easeInOutQuad"},
                {"from": 2, "to": 3, "progressMap": [{"progress": 0.0, "value": 0.2}]},
            ]
        }
        neutralized = _neutralized_progress_scene(scene, {(2, 3)})
        self.assertNotIn("progressMap", scene["transitions"][0])
        self.assertEqual(neutralized["transitions"][1]["easing"], "linear")
        self.assertEqual(
            neutralized["transitions"][1]["progressMap"],
            [{"progress": 0.0, "value": 0.0}, {"progress": 1.0, "value": 1.0}],
        )

    def test_morph_progress_config_overrides_enforce_monotonic_points(self) -> None:
        rows = [
            {"from": 1, "to": 2, "progress": 0.0, "bestProgressValue": 0.2, "score": 0.1},
            {"from": 1, "to": 2, "progress": 0.1, "bestProgressValue": 0.4, "score": 0.8},
            {"from": 1, "to": 2, "progress": 0.25, "bestProgressValue": 0.1, "score": 0.2},
            {"from": 1, "to": 2, "progress": 0.5, "bestProgressValue": 0.3, "score": 0.9},
            {"from": 1, "to": 2, "progress": 1.0, "bestProgressValue": 0.7, "score": 0.1},
        ]
        overrides = _morph_progress_config_overrides(rows, 0.55)
        self.assertEqual(overrides[0]["from"], 1)
        self.assertEqual(overrides[0]["to"], 2)
        self.assertEqual(
            overrides[0]["points"],
            [
                {"progress": 0.0, "value": 0.0},
                {"progress": 0.1, "value": 0.4},
                {"progress": 0.5, "value": 0.4},
                {"progress": 1.0, "value": 1.0},
            ],
        )

    def test_morph_progress_anchor_tracks_prefer_rounded_panels(self) -> None:
        self.assertFalse(
            _is_morph_progress_panel_anchor(
                {
                    "kind": "image",
                    "name": "Master Background",
                    "shape": "rect",
                    "geometry": {"widthPct": 1.0, "heightPct": 1.0},
                }
            )
        )
        panel_left = {
            "id": "panel-left",
            "trackId": "track-panel",
            "kind": "shape",
            "shape": "roundRect",
            "geometry": {"leftPct": 0.55, "topPct": 0.1, "widthPct": 0.4, "heightPct": 0.6},
        }
        panel_right = {
            "id": "panel-right",
            "trackId": "track-panel",
            "kind": "shape",
            "shape": "roundRect",
            "geometry": {"leftPct": 0.05, "topPct": 0.1, "widthPct": 0.4, "heightPct": 0.6},
        }
        child_left = {
            "id": "child-left",
            "trackId": "track-child",
            "kind": "video",
            "geometry": {"leftPct": 0.6, "topPct": 0.2, "widthPct": 0.1, "heightPct": 0.1},
        }
        child_right = {
            "id": "child-right",
            "trackId": "track-child",
            "kind": "video",
            "geometry": {"leftPct": 0.1, "topPct": 0.2, "widthPct": 0.1, "heightPct": 0.1},
        }
        tracks = _morph_progress_anchor_tracks(
            {"objects": [panel_left, child_left]},
            {"objects": [panel_right, child_right]},
            {
                "matches": [
                    {"trackId": "track-child", "fromObjectId": "child-left", "toObjectId": "child-right"},
                    {"trackId": "track-panel", "fromObjectId": "panel-left", "toObjectId": "panel-right"},
                ]
            },
        )
        self.assertEqual(tracks, ["track-panel"])

    def test_qa_ssim_uses_local_structure(self) -> None:
        import numpy as np

        left = np.zeros((64, 64, 3), dtype=np.float32)
        left[16:48, 16:48, :] = 180
        identical = left.copy()
        changed = left.copy()
        changed[28:36, 28:36, :] = 255

        self.assertAlmostEqual(_global_ssim(left, identical), 1.0, places=6)
        self.assertGreater(_global_ssim(left, changed), 0.85)
        self.assertLess(_global_ssim(left, changed), 1.0)

    def test_morph_matcher_rejects_different_media_with_generic_names(self) -> None:
        left = _scene_object("a", "Picture 16", "image", "asset-left", shape_id="17")
        right = _scene_object("b", "Picture 14", "image", "asset-right", shape_id="17")
        self.assertEqual(_match_objects([left], [right], 55), [])
        same_generic = _scene_object("c", "Picture 16", "image", "asset-right", shape_id="22")
        self.assertEqual(_match_objects([left], [same_generic], 55), [])

    def test_morph_matcher_allows_different_media_with_explicit_same_name(self) -> None:
        left = _scene_object("a", "!!portrait-anchor", "image", "asset-left", shape_id="17")
        right = _scene_object("b", "!!portrait-anchor", "image", "asset-right", shape_id="22")
        matches = _match_objects([left], [right], 55)
        self.assertEqual(matches[0][0:2], ("a", "b"))

    def test_morph_matcher_boosts_explicit_video_poster_names(self) -> None:
        left = _scene_object("a", "Combined01_spedup_30s_hq_playback", "video", "asset-video")
        right = _scene_object("b", "Combined01_spedup_30s_hq_playback", "image", "asset-poster")
        left.geometry = Geometry(x=1_000_000, y=500_000, cx=3_000_000, cy=3_000_000)
        right.geometry = Geometry(x=300_000, y=500_000, cx=3_000_000, cy=3_000_000)
        matches = _match_objects([left], [right], 55)
        self.assertEqual(matches[0][0:2], ("a", "b"))

    def test_scene_objects_expose_group_path(self) -> None:
        obj = _scene_object("a", "Panel child", "image", "asset-loop")
        obj.provenance["groupPath"] = ["Outer Group", "Inner Panel"]
        scene_obj = obj.to_scene(1000, 1000)
        self.assertEqual(scene_obj["groupPath"], ["Outer Group", "Inner Panel"])

    def test_morph_matcher_uses_group_path_as_panel_signal(self) -> None:
        left = _scene_object("a", "Shape 1", "shape", None, shape_id="7")
        right = _scene_object("b", "Shape 2", "shape", None, shape_id="7")
        left.provenance["groupPath"] = ["Panel Group"]
        right.provenance["groupPath"] = ["Panel Group"]
        matches = _match_objects([left], [right], 55)
        self.assertEqual(matches[0][0:2], ("a", "b"))

    def test_morph_matcher_pairs_container_shapes_by_matched_children(self) -> None:
        left_card = _scene_object("card-left", "Rectangle: Rounded Corners 23", "shape", None)
        right_card = _scene_object("card-right", "Rectangle: Rounded Corners 1", "shape", None)
        left_child = _scene_object("child-left", "Picture 25", "image", "asset-loop")
        right_child = _scene_object("child-right", "Picture 5", "image", "asset-loop")
        left_card.geometry = Geometry(x=1_000_000, y=400_000, cx=5_000_000, cy=4_000_000)
        right_card.geometry = Geometry(x=-6_000_000, y=400_000, cx=5_000_000, cy=4_000_000)
        left_child.geometry = Geometry(x=1_400_000, y=1_000_000, cx=2_000_000, cy=2_000_000)
        right_child.geometry = Geometry(x=-5_600_000, y=1_000_000, cx=2_000_000, cy=2_000_000)
        matches = _match_objects([left_card, left_child], [right_card, right_child], 55)
        self.assertIn(("child-left", "child-right"), [match[0:2] for match in matches])
        self.assertIn(("card-left", "card-right"), [match[0:2] for match in matches])

    def test_morph_matcher_anchors_panel_identity_to_swapped_children(self) -> None:
        prev_center = _scene_object("prev-center-panel", "Rectangle: Rounded Corners 1", "shape", None)
        prev_right = _scene_object("prev-right-panel", "Rectangle: Rounded Corners 2", "shape", None)
        prev_center_child = _scene_object("prev-center-child", "Combined04", "video", "asset-a")
        prev_right_child = _scene_object("prev-right-child", "Combined06", "video", "asset-b")
        next_center = _scene_object("next-center-panel", "Rectangle: Rounded Corners 1", "shape", None)
        next_left = _scene_object("next-left-panel", "Rectangle: Rounded Corners 2", "shape", None)
        next_center_child = _scene_object("next-center-child", "Combined06", "video", "asset-b")
        next_left_child = _scene_object("next-left-child", "Combined04", "video", "asset-a")
        prev_background = _scene_object("prev-background", "Picture 1", "image", "asset-bg")
        next_background = _scene_object("next-background", "Picture 1", "image", "asset-bg")

        for panel in (prev_center, prev_right, next_center, next_left):
            panel.shape = "roundRect"
            panel.geometry = Geometry(x=0, y=400_000, cx=5_000_000, cy=4_000_000)
        prev_center.geometry.x = 1_000_000
        prev_right.geometry.x = 7_000_000
        next_center.geometry.x = 1_000_000
        next_left.geometry.x = -5_000_000

        prev_center_child.geometry = Geometry(x=1_250_000, y=650_000, cx=4_500_000, cy=3_500_000)
        prev_right_child.geometry = Geometry(x=7_250_000, y=650_000, cx=4_500_000, cy=3_500_000)
        next_center_child.geometry = Geometry(x=1_250_000, y=650_000, cx=4_500_000, cy=3_500_000)
        next_left_child.geometry = Geometry(x=-4_750_000, y=650_000, cx=4_500_000, cy=3_500_000)
        prev_background.geometry = Geometry(x=0, y=0, cx=12_000_000, cy=6_750_000)
        next_background.geometry = Geometry(x=0, y=0, cx=12_000_000, cy=6_750_000)

        matches = _match_objects(
            [prev_background, prev_center, prev_center_child, prev_right, prev_right_child],
            [next_background, next_center, next_center_child, next_left, next_left_child],
            55,
        )
        pairs = [match[0:2] for match in matches]
        self.assertIn(("prev-center-child", "next-left-child"), pairs)
        self.assertIn(("prev-right-child", "next-center-child"), pairs)
        self.assertIn(("prev-center-panel", "next-left-panel"), pairs)
        self.assertIn(("prev-right-panel", "next-center-panel"), pairs)

    def test_inferred_panel_motions_move_unmatched_panel_children(self) -> None:
        slide_w = 12_000_000
        slide_h = 6_750_000
        left_panel = _scene_object("panel-left", "Rectangle: Rounded Corners 1", "shape", None)
        right_panel = _scene_object("panel-right", "Rectangle: Rounded Corners 1", "shape", None)
        entering_panel = _scene_object("panel-enter", "Rectangle: Rounded Corners 2", "shape", None)
        entering_child = _scene_object("child-enter", "Video 1", "video", "asset-video")
        left_panel.shape = right_panel.shape = entering_panel.shape = "roundRect"
        left_panel.track_id = right_panel.track_id = "track-panel"
        entering_panel.track_id = "track-enter-panel"
        entering_child.track_id = "track-enter-child"
        left_panel.geometry = Geometry(x=1_200_000, y=500_000, cx=5_000_000, cy=4_000_000)
        right_panel.geometry = Geometry(x=-800_000, y=500_000, cx=5_000_000, cy=4_000_000)
        entering_panel.geometry = Geometry(x=6_500_000, y=500_000, cx=5_000_000, cy=4_000_000)
        entering_child.geometry = Geometry(x=6_700_000, y=700_000, cx=4_600_000, cy=3_600_000)
        prev = Slide(1, "ppt/slides/slide1.xml", Transition("morph", 2.0), [left_panel])
        current = Slide(2, "ppt/slides/slide2.xml", Transition("morph", 2.0), [right_panel, entering_panel, entering_child])
        motions = _inferred_panel_motions(
            prev,
            current,
            {"track-panel"},
            {"track-panel", "track-enter-panel", "track-enter-child"},
            2.0,
            slide_w,
            slide_h,
        )
        by_track = {row["trackId"]: row for row in motions}
        self.assertIn("track-enter-child", by_track)
        self.assertAlmostEqual(
            by_track["track-enter-child"]["fromGeometry"]["x"],
            entering_child.geometry.x + 2_000_000,
        )
        self.assertEqual(by_track["track-enter-child"]["source"], "inferred-panel-motion")

    def test_inferred_panel_motions_use_containing_panel_delta(self) -> None:
        slide_w = 12_000_000
        slide_h = 6_750_000
        prev_panel_a = _scene_object("panel-a-prev", "Rectangle: Rounded Corners 1", "shape", None)
        next_panel_a = _scene_object("panel-a-next", "Rectangle: Rounded Corners 1", "shape", None)
        prev_panel_b = _scene_object("panel-b-prev", "Rectangle: Rounded Corners 2", "shape", None)
        next_panel_b = _scene_object("panel-b-next", "Rectangle: Rounded Corners 2", "shape", None)
        exit_child = _scene_object("child-exit", "Video 1", "video", "asset-video")
        for panel in (prev_panel_a, next_panel_a, prev_panel_b, next_panel_b):
            panel.shape = "roundRect"
            panel.geometry = Geometry(x=1_000_000, y=500_000, cx=5_000_000, cy=4_000_000)
        prev_panel_a.track_id = next_panel_a.track_id = "track-panel-a"
        prev_panel_b.track_id = next_panel_b.track_id = "track-panel-b"
        exit_child.track_id = "track-exit-child"
        next_panel_a.geometry.x = 0
        prev_panel_b.geometry.x = 7_000_000
        next_panel_b.geometry.x = 4_000_000
        exit_child.geometry = Geometry(x=1_400_000, y=900_000, cx=2_000_000, cy=2_000_000)
        prev = Slide(1, "ppt/slides/slide1.xml", Transition("morph", 2.0), [prev_panel_a, prev_panel_b, exit_child])
        current = Slide(2, "ppt/slides/slide2.xml", Transition("morph", 2.0), [next_panel_a, next_panel_b])
        motions = _inferred_panel_motions(
            prev,
            current,
            {"track-panel-a", "track-panel-b", "track-exit-child"},
            {"track-panel-a", "track-panel-b"},
            2.0,
            slide_w,
            slide_h,
        )
        by_track = {row["trackId"]: row for row in motions}
        self.assertEqual(by_track["track-exit-child"]["panelTrackId"], "track-panel-a")
        self.assertAlmostEqual(by_track["track-exit-child"]["toGeometry"]["x"], 400_000)

    def test_inferred_motions_slide_large_foreground_exits_with_carousel(self) -> None:
        slide_w = 10_000_000
        slide_h = 6_000_000
        prev_panel = _scene_object("panel-prev", "Rectangle: Rounded Corners 1", "shape", None)
        next_panel = _scene_object("panel-next", "Rectangle: Rounded Corners 1", "shape", None)
        foreground = _scene_object("foreground", "VR figure", "image", "asset-figure")
        prev_panel.shape = next_panel.shape = "roundRect"
        prev_panel.track_id = next_panel.track_id = "track-panel"
        foreground.track_id = "track-figure"
        prev_panel.geometry = Geometry(x=2_000_000, y=400_000, cx=6_000_000, cy=5_000_000)
        next_panel.geometry = Geometry(x=-4_000_000, y=400_000, cx=6_000_000, cy=5_000_000)
        foreground.geometry = Geometry(x=100_000, y=1_000_000, cx=2_000_000, cy=4_000_000)
        prev = Slide(16, "ppt/slides/slide16.xml", Transition("morph", 2.0), [prev_panel, foreground])
        current = Slide(17, "ppt/slides/slide17.xml", Transition("morph", 2.0), [next_panel])
        motions = _inferred_panel_motions(
            prev,
            current,
            {"track-panel", "track-figure"},
            {"track-panel"},
            2.0,
            slide_w,
            slide_h,
        )
        by_track = {row["trackId"]: row for row in motions}
        self.assertEqual(by_track["track-figure"]["source"], "inferred-carousel-foreground-motion")
        self.assertTrue(by_track["track-figure"]["preserveOpacity"])
        self.assertAlmostEqual(by_track["track-figure"]["toGeometry"]["x"], -5_900_000)

    def test_inferred_motions_do_not_slide_footer_sponsor_strips_with_carousel(self) -> None:
        slide_w = 10_000_000
        slide_h = 6_000_000
        prev_panel = _scene_object("panel-prev", "Rectangle: Rounded Corners 1", "shape", None)
        next_panel = _scene_object("panel-next", "Rectangle: Rounded Corners 1", "shape", None)
        sponsor_strip = _scene_object("sponsor-strip", "Sponsor logos", "image", "asset-sponsor")
        prev_panel.shape = next_panel.shape = "roundRect"
        prev_panel.track_id = next_panel.track_id = "track-panel"
        sponsor_strip.track_id = "track-sponsor"
        prev_panel.geometry = Geometry(x=2_000_000, y=400_000, cx=6_000_000, cy=5_000_000)
        next_panel.geometry = Geometry(x=-4_000_000, y=400_000, cx=6_000_000, cy=5_000_000)
        sponsor_strip.geometry = Geometry(x=5_500_000, y=5_250_000, cx=4_000_000, cy=600_000)
        prev = Slide(2, "ppt/slides/slide2.xml", Transition("morph", 2.0), [prev_panel, sponsor_strip])
        current = Slide(3, "ppt/slides/slide3.xml", Transition("morph", 2.0), [next_panel])
        motions = _inferred_panel_motions(
            prev,
            current,
            {"track-panel", "track-sponsor"},
            {"track-panel"},
            2.0,
            slide_w,
            slide_h,
        )

        by_track = {row["trackId"]: row for row in motions}
        self.assertNotIn("track-sponsor", by_track)

    def test_panel_relationship_annotation_links_panel_border_children(self) -> None:
        slides = [
            {
                "index": 1,
                "objects": [
                    {
                        "id": "bg",
                        "trackId": "track-bg",
                        "name": "Background",
                        "kind": "image",
                        "geometry": {"x": 0, "y": 0, "w": 12_000_000, "h": 6_750_000},
                    },
                    {
                        "id": "panel",
                        "trackId": "track-panel",
                        "name": "PowerPoint panel border track-panel",
                        "kind": "image",
                        "geometry": {"x": 1_000_000, "y": 500_000, "w": 5_000_000, "h": 4_000_000},
                    },
                    {
                        "id": "video",
                        "trackId": "track-video",
                        "name": "Panel video",
                        "kind": "video",
                        "geometry": {"x": 1_200_000, "y": 700_000, "w": 4_600_000, "h": 3_600_000},
                    },
                ],
            }
        ]
        report = _annotate_panel_relationships(slides, 12_000_000, 6_750_000)
        by_id = {obj["id"]: obj for obj in slides[0]["objects"]}
        self.assertEqual(by_id["panel"]["panelRole"], "container")
        self.assertEqual(by_id["panel"]["nodeRole"], "panel")
        self.assertTrue(by_id["panel"]["panelBorderOnTop"])
        self.assertEqual(by_id["video"]["panelParentTrackId"], "track-panel")
        self.assertEqual(by_id["video"]["parentTrackId"], "track-panel")
        self.assertAlmostEqual(by_id["video"]["localGeometry"]["leftPct"], 0.04)
        self.assertAlmostEqual(by_id["video"]["localGeometry"]["topPct"], 0.05)
        self.assertNotIn("panelParentTrackId", by_id["bg"])
        self.assertEqual(report["appliedCount"], 1)

    def test_scene_graph_v2_emits_panel_groups_and_relationships(self) -> None:
        slides = [
            {
                "index": 1,
                "objects": [
                    {
                        "id": "panel",
                        "trackId": "track-panel",
                        "name": "PowerPoint panel border track-panel",
                        "kind": "image",
                        "geometry": {"x": 1_000_000, "y": 500_000, "w": 5_000_000, "h": 4_000_000},
                        "groupPath": ["Carousel"],
                    },
                    {
                        "id": "video",
                        "trackId": "track-video",
                        "name": "Panel video",
                        "kind": "video",
                        "geometry": {"x": 1_200_000, "y": 700_000, "w": 4_600_000, "h": 3_600_000},
                    },
                ],
            }
        ]
        _annotate_panel_relationships(slides, 12_000_000, 6_750_000)
        report = _annotate_scene_graph_v2(slides, 12_000_000, 6_750_000)
        self.assertEqual(report["groupCount"], 2)
        self.assertEqual(len(slides[0]["nodes"]), 3)
        explicit_node = next(node for node in slides[0]["nodes"] if node.get("nodeRole") == "group")
        self.assertEqual(explicit_node["kind"], "group")
        self.assertEqual(explicit_node["name"], "Carousel")
        self.assertEqual(explicit_node["childrenTrackIds"], ["track-panel"])
        groups_by_kind = {group["kind"]: group for group in slides[0]["groups"]}
        self.assertTrue(groups_by_kind["ppt-group-path"]["renderable"])
        self.assertEqual(groups_by_kind["inferred-panel"]["trackId"], "track-panel")
        self.assertEqual(groups_by_kind["inferred-panel"]["childrenTrackIds"], ["track-video"])
        self.assertIn(
            {
                "type": "panel-contains",
                "parentId": "panel-track-panel",
                "parentTrackId": "track-panel",
                "childTrackId": "track-video",
                "childObjectId": "video",
                "source": "panel-containment",
            },
            slides[0]["relationships"],
        )
        self.assertEqual(slides[0]["objects"][1]["parentId"], "panel-track-panel")

    def test_panel_relationships_rewrite_child_transition_motion_from_parent_panel(self) -> None:
        slides = [
            {
                "index": 1,
                "objects": [
                    {
                        "id": "panel-prev",
                        "trackId": "track-panel",
                        "name": "PowerPoint panel border track-panel",
                        "kind": "image",
                        "panelRole": "container",
                        "geometry": {
                            "x": 1_000_000,
                            "y": 500_000,
                            "w": 5_000_000,
                            "h": 4_000_000,
                            "leftPct": 1_000_000 / 12_000_000,
                            "topPct": 500_000 / 6_750_000,
                            "widthPct": 5_000_000 / 12_000_000,
                            "heightPct": 4_000_000 / 6_750_000,
                        },
                    },
                    {
                        "id": "video-prev",
                        "trackId": "track-video",
                        "name": "Panel video",
                        "kind": "video",
                        "panelParentTrackId": "track-panel",
                        "geometry": {
                            "x": 1_200_000,
                            "y": 700_000,
                            "w": 4_600_000,
                            "h": 3_600_000,
                            "leftPct": 1_200_000 / 12_000_000,
                            "topPct": 700_000 / 6_750_000,
                            "widthPct": 4_600_000 / 12_000_000,
                            "heightPct": 3_600_000 / 6_750_000,
                        },
                    },
                ],
            },
            {"index": 2, "objects": []},
        ]
        transitions = [
            {
                "from": 1,
                "to": 2,
                "durationSec": 2.0,
                "exitTrackIds": ["track-panel", "track-video"],
                "enterTrackIds": [],
                "inferredMotions": [
                    {
                        "trackId": "track-panel",
                        "fromGeometry": slides[0]["objects"][0]["geometry"],
                        "toGeometry": {
                            **slides[0]["objects"][0]["geometry"],
                            "x": -2_000_000,
                            "leftPct": -2_000_000 / 12_000_000,
                        },
                        "source": "inferred-panel-motion",
                    },
                    {
                        "trackId": "track-video",
                        "fromGeometry": slides[0]["objects"][1]["geometry"],
                        "toGeometry": {
                            **slides[0]["objects"][1]["geometry"],
                            "x": 5_200_000,
                            "leftPct": 5_200_000 / 12_000_000,
                        },
                        "source": "inferred-panel-motion",
                    },
                ],
            }
        ]
        report = _apply_panel_relationships_to_transitions(transitions, slides, 12_000_000, 6_750_000)
        video_row = next(row for row in transitions[0]["inferredMotions"] if row["trackId"] == "track-video")
        self.assertEqual(video_row["panelTrackId"], "track-panel")
        self.assertEqual(video_row["source"], "inferred-panel-parent-motion")
        self.assertAlmostEqual(video_row["toGeometry"]["x"], -1_800_000)
        self.assertEqual(report["appliedCount"], 1)

    def test_reference_export_uses_scene_timings_by_default(self) -> None:
        self.assertTrue(_effective_use_timings(None, Path("deck.scene.json")))
        self.assertFalse(_effective_use_timings(None, None))
        self.assertFalse(_effective_use_timings(False, Path("deck.scene.json")))
        self.assertTrue(_effective_use_timings(True, None))

    def test_reference_export_reads_scene_slide_hold_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "deck.scene.json"
            scene.write_text('{"qa":{"slideHoldSec":3.0}}', encoding="utf-8")
            self.assertEqual(_effective_slide_hold(None, scene), 3.0)
            self.assertEqual(_effective_slide_hold(1.5, scene), 1.5)
            self.assertEqual(_effective_slide_hold(None, None), 1.0)

    def test_alpha_gif_is_not_flattened_to_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset = AssetRef(
                source_path="ppt/media/transparent.gif",
                rel_id=None,
                kind="image",
                extension="gif",
                size_bytes=1,
                sha256="a" * 64,
                animated=True,
                alpha=True,
            )
            converted = _try_convert_gif_with_ffmpeg(
                Path(tmp) / "transparent.gif",
                Path(tmp) / "out",
                asset,
                ffmpeg=Path(tmp) / "ffmpeg.exe",
                quality=88,
            )
            self.assertIsNone(converted)
            self.assertIn("gif-alpha-not-flattened-to-mp4", asset.warnings)

    def test_animated_gif_transcodes_even_under_size_limit_for_clock_control(self) -> None:
        asset = AssetRef(
            source_path="ppt/media/small-loop.gif",
            rel_id=None,
            kind="image",
            extension="gif",
            size_bytes=1024,
            sha256="b" * 64,
            animated=True,
            alpha=True,
        )
        self.assertTrue(_should_transcode_gif(asset, AssetPolicy(transcode_gif=True)))
        self.assertFalse(_should_transcode_gif(asset, AssetPolicy(transcode_gif=False)))

    def test_large_static_images_are_publish_optimization_candidates(self) -> None:
        asset = AssetRef(
            source_path="ppt/media/large.png",
            rel_id=None,
            kind="image",
            extension="png",
            size_bytes=60 * 1024 * 1024,
            sha256="c" * 64,
            animated=False,
            alpha=True,
        )
        self.assertTrue(_should_optimize_static_image(asset, AssetPolicy(soft_max_mb=50)))
        self.assertFalse(
            _should_optimize_static_image(
                asset,
                AssetPolicy(soft_max_mb=50, optimize_static_images=False),
            )
        )

    def test_soft_oversize_asset_blocks_publish_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = b"<svg viewBox='0 0 1 1'></svg>" * 256
            source_sha = hashlib.sha256(raw).hexdigest()
            pptx = root / "demo.pptx"
            with zipfile.ZipFile(pptx, "w") as zf:
                zf.writestr("ppt/media/large.svg", raw)
            deck = PptxDeck(
                source_path=str(pptx),
                source_sha256=hashlib.sha256(pptx.read_bytes()).hexdigest(),
                title="Demo",
                slide_width=16,
                slide_height=9,
                slides=[],
                assets={
                    "ppt/media/large.svg": AssetRef(
                        source_path="ppt/media/large.svg",
                        rel_id="rId1",
                        kind="svg",
                        extension="svg",
                        size_bytes=len(raw),
                        sha256=source_sha,
                    )
                },
            )

            report = prepare_assets(
                deck,
                root / "out",
                AssetPolicy(soft_max_mb=0.001, hard_max_mb=1, allow_oversize_assets=False),
            )

            row = report["assets"][0]
            self.assertFalse(report["githubPagesSafe"])
            self.assertTrue(report["hardLimitSafe"])
            self.assertFalse(report["preferredAssetSafe"])
            self.assertFalse(report["publishAssetSafe"])
            self.assertIn("github-soft-limit-warning", row["warnings"])
            self.assertIn("github-soft-limit-blocker", row["warnings"])
            self.assertNotIn("github-hard-limit-blocker", row["warnings"])

    def test_soft_oversize_asset_can_be_allowed_for_reviewed_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = b"<svg viewBox='0 0 1 1'></svg>" * 256
            source_sha = hashlib.sha256(raw).hexdigest()
            pptx = root / "demo.pptx"
            with zipfile.ZipFile(pptx, "w") as zf:
                zf.writestr("ppt/media/large.svg", raw)
            deck = PptxDeck(
                source_path=str(pptx),
                source_sha256=hashlib.sha256(pptx.read_bytes()).hexdigest(),
                title="Demo",
                slide_width=16,
                slide_height=9,
                slides=[],
                assets={
                    "ppt/media/large.svg": AssetRef(
                        source_path="ppt/media/large.svg",
                        rel_id="rId1",
                        kind="svg",
                        extension="svg",
                        size_bytes=len(raw),
                        sha256=source_sha,
                    )
                },
            )

            report = prepare_assets(
                deck,
                root / "out",
                AssetPolicy(soft_max_mb=0.001, hard_max_mb=1, allow_oversize_assets=True),
            )

            row = report["assets"][0]
            self.assertTrue(report["githubPagesSafe"])
            self.assertTrue(report["hardLimitSafe"])
            self.assertFalse(report["preferredAssetSafe"])
            self.assertTrue(report["publishAssetSafe"])
            self.assertIn("github-soft-limit-warning", row["warnings"])
            self.assertNotIn("github-soft-limit-blocker", row["warnings"])

    def test_wdp_assets_are_conversion_candidates(self) -> None:
        asset = AssetRef(
            source_path="ppt/media/hdphoto1.wdp",
            rel_id=None,
            kind="image",
            extension="wdp",
            size_bytes=1024,
            sha256="d" * 64,
        )
        self.assertTrue(_should_convert_wdp(asset, AssetPolicy()))
        self.assertFalse(_should_convert_wdp(asset, AssetPolicy(mode="source-only")))

    def test_prepare_assets_reuses_shared_optimized_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = b"fake animated gif payload"
            source_sha = hashlib.sha256(raw).hexdigest()
            pptx = root / "demo.pptx"
            with zipfile.ZipFile(pptx, "w") as zf:
                zf.writestr("ppt/media/loop.gif", raw)
            cached = root / "presentations" / "shared-assets" / "viscereality" / "optimized" / "cached.webm"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"cached-webm-runtime")
            deck = PptxDeck(
                source_path=str(pptx),
                source_sha256=hashlib.sha256(pptx.read_bytes()).hexdigest(),
                title="Demo",
                slide_width=16,
                slide_height=9,
                slides=[],
                assets={
                    "ppt/media/loop.gif": AssetRef(
                        source_path="ppt/media/loop.gif",
                        rel_id="rId1",
                        kind="image",
                        extension="gif",
                        size_bytes=len(raw),
                        sha256=source_sha,
                        animated=True,
                        alpha=True,
                    )
                },
            )

            prepare_assets(
                deck,
                root / "out",
                AssetPolicy(),
                optimized_asset_cache={
                    "bySourceSha256": {
                        source_sha: {
                            "path": str(cached),
                            "extension": "webm",
                            "kind": "video",
                            "animated": True,
                            "alpha": True,
                        }
                    }
                },
            )

            asset = deck.assets["ppt/media/loop.gif"]
            self.assertEqual(asset.kind, "video")
            self.assertEqual(asset.extension, "webm")
            self.assertEqual(asset.output_file, f"assets/optimized/{source_sha[:16]}-cached.webm")
            self.assertEqual((root / "out" / asset.output_file).read_bytes(), b"cached-webm-runtime")
            self.assertIn("optimized-asset-reused-from-shared-cache", asset.warnings)
            self.assertNotIn("gif-transcode-unavailable", asset.warnings)

    def test_prepare_assets_rejects_cached_optimized_asset_over_publish_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = b"fake animated gif payload" * 128
            source_sha = hashlib.sha256(raw).hexdigest()
            pptx = root / "demo.pptx"
            with zipfile.ZipFile(pptx, "w") as zf:
                zf.writestr("ppt/media/loop.gif", raw)
            cached = root / "presentations" / "shared-assets" / "viscereality" / "optimized" / "cached.webm"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"x" * 2048)
            deck = PptxDeck(
                source_path=str(pptx),
                source_sha256=hashlib.sha256(pptx.read_bytes()).hexdigest(),
                title="Demo",
                slide_width=16,
                slide_height=9,
                slides=[],
                assets={
                    "ppt/media/loop.gif": AssetRef(
                        source_path="ppt/media/loop.gif",
                        rel_id="rId1",
                        kind="image",
                        extension="gif",
                        size_bytes=len(raw),
                        sha256=source_sha,
                        animated=True,
                        alpha=True,
                    )
                },
            )

            report = prepare_assets(
                deck,
                root / "out",
                AssetPolicy(soft_max_mb=0.001, hard_max_mb=1.0, allow_oversize_assets=False),
                optimized_asset_cache={
                    "bySourceSha256": {
                        source_sha: {
                            "path": str(cached),
                            "extension": "webm",
                            "kind": "video",
                            "animated": True,
                            "alpha": True,
                        }
                    }
                },
            )

            asset = deck.assets["ppt/media/loop.gif"]
            self.assertNotEqual(asset.output_file, f"assets/optimized/{source_sha[:16]}-cached.webm")
            self.assertIn("cached-optimized-asset-over-publish-limit", asset.warnings)
            self.assertNotIn("optimized-asset-reused-from-shared-cache", asset.warnings)
            self.assertFalse(report["publishAssetSafe"])

    def test_hdphoto_media_target_is_preferred_when_powerpoint_marks_image_layer(self) -> None:
        assets = {
            "ppt/media/fallback.png": AssetRef(
                source_path="ppt/media/fallback.png",
                rel_id=None,
                kind="image",
                extension="png",
                size_bytes=1,
                sha256="e" * 64,
            ),
            "ppt/media/hdphoto1.wdp": AssetRef(
                source_path="ppt/media/hdphoto1.wdp",
                rel_id=None,
                kind="image",
                extension="wdp",
                size_bytes=1,
                sha256="f" * 64,
            ),
        }
        self.assertEqual(
            _selected_media_target(
                ["ppt/media/fallback.png", "ppt/media/hdphoto1.wdp"],
                assets,
                prefer_video=False,
                prefer_hdphoto=True,
            ),
            "ppt/media/hdphoto1.wdp",
        )

    def test_svg_media_target_is_preferred_over_bitmap_fallback(self) -> None:
        assets = {
            "ppt/media/fallback.png": AssetRef(
                source_path="ppt/media/fallback.png",
                rel_id=None,
                kind="image",
                extension="png",
                size_bytes=1,
                sha256="e" * 64,
            ),
            "ppt/media/vector.svg": AssetRef(
                source_path="ppt/media/vector.svg",
                rel_id=None,
                kind="svg",
                extension="svg",
                size_bytes=1,
                sha256="f" * 64,
            ),
        }
        self.assertEqual(
            _selected_media_target(
                ["ppt/media/fallback.png", "ppt/media/vector.svg"],
                assets,
                prefer_video=False,
            ),
            "ppt/media/vector.svg",
        )

    def test_hdphoto_image_layer_brightness_contrast_is_parsed(self) -> None:
        node = ET.fromstring(
            """
            <p:pic xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                   xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                   xmlns:a14="http://schemas.microsoft.com/office/drawing/2010/main"
                   xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
              <p:blipFill>
                <a:blip r:embed="rId1">
                  <a:extLst>
                    <a:ext uri="{BEBA8EAE-BF5A-486C-A8C5-ECC9F3942E4B}">
                      <a14:imgProps>
                        <a14:imgLayer r:embed="rId2">
                          <a14:imgEffect>
                            <a14:brightnessContrast bright="100000" contrast="-20000"/>
                          </a14:imgEffect>
                        </a14:imgLayer>
                      </a14:imgProps>
                    </a:ext>
                  </a:extLst>
                </a:blip>
              </p:blipFill>
            </p:pic>
            """
        )
        self.assertEqual(
            _media_effects(node),
            {"brightnessContrast": {"bright": 1.0, "contrast": -0.2}},
        )

    def test_powerpoint_glow_effect_is_parsed(self) -> None:
        node = ET.fromstring(
            """
            <p:pic xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                   xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
              <p:spPr>
                <a:effectLst>
                  <a:glow rad="381000">
                    <a:schemeClr val="bg1">
                      <a:alpha val="40000"/>
                    </a:schemeClr>
                  </a:glow>
                </a:effectLst>
              </p:spPr>
            </p:pic>
            """
        )
        self.assertEqual(
            _visual_effects(node),
            {"glow": {"radiusEmu": 381000, "color": "scheme:bg1", "alpha": 0.4}},
        )

    def test_prunes_unreferenced_source_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source = out / "assets" / "source"
            source.mkdir(parents=True)
            keep = source / "keep.png"
            remove = source / "remove.gif"
            keep.write_bytes(b"keep")
            remove.write_bytes(b"remove")
            report = _prune_unreferenced_source_assets(
                source,
                out,
                {"assets/source/keep.png"},
            )
            self.assertEqual(report["count"], 1)
            self.assertTrue(keep.exists())
            self.assertFalse(remove.exists())

    def test_prunes_unreferenced_optimized_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            optimized = out / "assets" / "optimized"
            optimized.mkdir(parents=True)
            keep = optimized / "keep-crf18.mp4"
            remove = optimized / "keep-crf24.mp4"
            keep.write_bytes(b"keep")
            remove.write_bytes(b"remove")
            report = _prune_unreferenced_asset_files(
                optimized,
                out,
                {"assets/optimized/keep-crf18.mp4"},
            )
            self.assertEqual(report["count"], 1)
            self.assertTrue(keep.exists())
            self.assertFalse(remove.exists())

    def test_family_config_resolves_paths_from_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            presentations = root / "presentations"
            presentations.mkdir()
            config = presentations / "family.json"
            config.write_text(
                json.dumps(
                    {
                        "repo_root": "..",
                        "family_id": "demo-family",
                        "presenter_config_file": "presentations/defaults.json",
                        "shared_assets": {
                            "root": "presentations/shared-assets/demo",
                        },
                        "decks": [
                            {
                                "id": "Demo",
                                "title": "Demo Deck",
                                "source": "presentations/demo.pptx",
                                "staging": "presentations/Demo-scene",
                                "public_dir": "Demo",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            family = load_family_config(config)
            self.assertEqual(family.repo_root, root.resolve())
            self.assertEqual(family.shared_root, (root / "presentations/shared-assets/demo").resolve())
            self.assertEqual(family.decks[0].source, (root / "presentations/demo.pptx").resolve())
            self.assertEqual(family.decks[0].staging, (root / "presentations/Demo-scene").resolve())

    def test_family_deck_config_merges_presenter_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            presentations = root / "presentations"
            presentations.mkdir()
            (presentations / "defaults.json").write_text(
                json.dumps(
                    {
                        "morph_policy": {
                            "unmatched_fade_start": 0.5,
                            "unmatched_fade_end": 0.75,
                        },
                        "visual_effects": {
                            "glow_scale": 0.5,
                            "glow_alpha_scale": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (presentations / "deck.json").write_text(
                json.dumps(
                    {
                        "morph_policy": {
                            "duration_default_sec": 1.25,
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = presentations / "family.json"
            config.write_text(
                json.dumps(
                    {
                        "repo_root": "..",
                        "family_id": "demo-family",
                        "presenter_config_file": "presentations/defaults.json",
                        "decks": [
                            {
                                "id": "Demo",
                                "source": "presentations/demo.pptx",
                                "staging": "presentations/Demo-scene",
                                "config": "presentations/deck.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            family = load_family_config(config)
            merged = family_module._deck_presenter_config(family, family.decks[0])
            self.assertEqual(merged.visual_effects.glow_scale, 0.5)
            self.assertEqual(merged.morph_policy.duration_default_sec, 1.25)
            self.assertEqual(merged.morph_policy.unmatched_fade_start, 0.5)

    def test_share_deck_assets_rewrites_scene_to_shared_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "presentations" / "Demo-scene"
            source_dir = build / "assets" / "source"
            optimized_dir = build / "assets" / "optimized"
            source_dir.mkdir(parents=True)
            optimized_dir.mkdir(parents=True)
            source_asset = source_dir / "source.png"
            optimized_asset = optimized_dir / "loop.mp4"
            source_asset.write_bytes(b"same-image")
            optimized_asset.write_bytes(b"optimized-video")
            scene = {
                "deck": {"id": "Demo"},
                "assets": [
                    {
                        "id": "asset-demo",
                        "sourcePath": "ppt/media/image1.png",
                        "sourceFile": "assets/source/source.png",
                        "file": "assets/optimized/loop.mp4",
                        "kind": "video",
                        "sha256": "sourcehashdemo",
                        "animated": True,
                        "alpha": False,
                    }
                ],
                "slides": [],
                "transitions": [],
            }
            (build / "deck.scene.json").write_text(json.dumps(scene), encoding="utf-8")
            (build / "build-report.json").write_text("{}", encoding="utf-8")
            shared = root / "presentations" / "shared-assets" / "viscereality"

            report = share_deck_assets(build, shared, deck_id="Demo", repo_root=root)

            self.assertEqual(report["status"], "ok")
            rewritten = json.loads((build / "deck.scene.json").read_text(encoding="utf-8"))
            asset = rewritten["assets"][0]
            self.assertTrue(asset["sourceFile"].startswith("../shared-assets/viscereality/source/"))
            self.assertTrue(asset["file"].startswith("../shared-assets/viscereality/optimized/"))
            self.assertFalse(source_asset.exists())
            self.assertFalse(optimized_asset.exists())
            index = json.loads((shared / "asset-index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(index["assets"]), 2)
            optimized_entries = [entry for entry in index["assets"].values() if entry["bucket"] == "optimized"]
            self.assertEqual(optimized_entries[0]["sourceSha256"], "sourcehashdemo")
            self.assertEqual(optimized_entries[0]["sourceSha256s"], ["sourcehashdemo"])

    def test_share_deck_assets_skips_large_source_when_runtime_is_optimized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "presentations" / "Demo-scene"
            source_dir = build / "assets" / "source"
            optimized_dir = build / "assets" / "optimized"
            source_dir.mkdir(parents=True)
            optimized_dir.mkdir(parents=True)
            source_asset = source_dir / "huge.gif"
            optimized_asset = optimized_dir / "loop.mp4"
            source_asset.write_bytes(b"large-source" * 256)
            optimized_asset.write_bytes(b"optimized-video")
            scene = {
                "deck": {"id": "Demo"},
                "assets": [
                    {
                        "id": "asset-demo",
                        "sourcePath": "ppt/media/huge.gif",
                        "sourceFile": "assets/source/huge.gif",
                        "file": "assets/optimized/loop.mp4",
                        "kind": "video",
                        "animated": True,
                        "alpha": False,
                    }
                ],
                "slides": [],
                "transitions": [],
            }
            (build / "deck.scene.json").write_text(json.dumps(scene), encoding="utf-8")
            (build / "build-report.json").write_text("{}", encoding="utf-8")
            shared = root / "presentations" / "shared-assets" / "viscereality"
            previous_soft_max = family_module.SHARED_SOURCE_SOFT_MAX_MB
            family_module.SHARED_SOURCE_SOFT_MAX_MB = 0.001
            try:
                report = share_deck_assets(build, shared, deck_id="Demo", repo_root=root)
            finally:
                family_module.SHARED_SOURCE_SOFT_MAX_MB = previous_soft_max

            self.assertEqual(report["status"], "ok")
            rewritten = json.loads((build / "deck.scene.json").read_text(encoding="utf-8"))
            asset = rewritten["assets"][0]
            self.assertEqual(asset["sourceFile"], asset["file"])
            self.assertIn("source-file-over-soft-limit-not-published", asset["warnings"])
            self.assertEqual(list((shared / "source").glob("*")), [])
            self.assertEqual(len(list((shared / "optimized").glob("*"))), 1)

    def test_family_oracle_qa_reports_missing_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            presentations = root / "presentations"
            public = presentations / "Demo"
            public.mkdir(parents=True)
            (presentations / "demo.pptx").write_bytes(b"not-a-real-pptx")
            (public / "deck.scene.json").write_text(
                json.dumps(
                    {
                        "deck": {"id": "Demo", "slideCount": 0},
                        "slides": [],
                        "transitions": [],
                        "assets": [],
                        "qa": {"transitionSamples": []},
                    }
                ),
                encoding="utf-8",
            )
            config = presentations / "family.json"
            config.write_text(
                json.dumps(
                    {
                        "repo_root": "..",
                        "family_id": "demo-family",
                        "shared_assets": {"root": "presentations/shared-assets/demo"},
                        "decks": [
                            {
                                "id": "Demo",
                                "title": "Demo Deck",
                                "source": "presentations/demo.pptx",
                                "staging": "presentations/Demo-scene",
                                "public_dir": "Demo",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = oracle_qa_family(
                config,
                ffmpeg_bin=str(root / "missing-ffmpeg.exe"),
                force=True,
                transition_reference_lead_fraction=0.25,
            )

            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["transitionReferenceLeadFractionOverride"], 0.25)
            self.assertEqual(report["decks"][0]["status"], "blocked")
            self.assertIn("ffmpeg-missing", report["decks"][0]["blockers"])
            self.assertTrue((presentations / "shared-assets" / "demo" / "family-oracle-qa-report.json").exists())


def _write_demo_pptx(path: Path, *, slide2_morph: bool = True, slide2_zero_transition: bool = False) -> None:
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
        b"\xff?\x00\x05\xfe\x02\xfeA\x8d\x9d\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
<Override PartName="/ppt/slides/slide2.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
</Types>""")
        zf.writestr("docProps/core.xml", """<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Demo deck</dc:title></cp:coreProperties>""")
        zf.writestr("ppt/presentation.xml", """<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldSz cx="12192000" cy="6858000"/><p:sldIdLst><p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId2"/></p:sldIdLst></p:presentation>""")
        zf.writestr("ppt/_rels/presentation.xml.rels", """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="slide" Target="slides/slide1.xml"/><Relationship Id="rId2" Type="slide" Target="slides/slide2.xml"/></Relationships>""")
        zf.writestr("ppt/media/image1.png", png)
        zf.writestr("ppt/slides/_rels/slide1.xml.rels", """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="image" Target="../media/image1.png"/></Relationships>""")
        zf.writestr("ppt/slides/_rels/slide2.xml.rels", """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="image" Target="../media/image1.png"/></Relationships>""")
        zf.writestr("ppt/slides/slide1.xml", _slide_xml("1", morph=False, text="", zero_transition=True))
        zf.writestr(
            "ppt/slides/slide2.xml",
            _slide_xml("2", morph=slide2_morph, text="Hello", zero_transition=slide2_zero_transition),
        )


def _write_master_background_pptx(path: Path) -> None:
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
        b"\xff?\x00\x05\xfe\x02\xfeA\x8d\x9d\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
</Types>""")
        zf.writestr("docProps/core.xml", """<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Master background</dc:title></cp:coreProperties>""")
        zf.writestr("ppt/presentation.xml", """<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldSz cx="12192000" cy="6858000"/><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>""")
        zf.writestr("ppt/_rels/presentation.xml.rels", """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>""")
        zf.writestr("ppt/slides/_rels/slide1.xml.rels", """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>""")
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>""")
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image1.png"/></Relationships>""")
        zf.writestr("ppt/media/image1.png", png)
        zf.writestr("ppt/slides/slide1.xml", """<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:cSld><p:spTree><p:nvGrpSpPr/><p:grpSpPr/></p:spTree></p:cSld></p:sld>""")
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", """<p:sldLayout xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:cSld><p:spTree><p:nvGrpSpPr/><p:grpSpPr/></p:spTree></p:cSld></p:sldLayout>""")
        zf.writestr("ppt/slideMasters/slideMaster1.xml", """<p:sldMaster xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:cSld><p:bg><p:bgPr><a:blipFill><a:blip r:embed="rId1"/><a:stretch><a:fillRect/></a:stretch></a:blipFill></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr/><p:grpSpPr/></p:spTree></p:cSld></p:sldMaster>""")


def _write_video_poster_pptx(path: Path) -> None:
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
        b"\xff?\x00\x05\xfe\x02\xfeA\x8d\x9d\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Default Extension="mp4" ContentType="video/mp4"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
</Types>""")
        zf.writestr("docProps/core.xml", """<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Video poster</dc:title></cp:coreProperties>""")
        zf.writestr("ppt/presentation.xml", """<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldSz cx="12192000" cy="6858000"/><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>""")
        zf.writestr("ppt/_rels/presentation.xml.rels", """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="slide" Target="slides/slide1.xml"/></Relationships>""")
        zf.writestr("ppt/media/image1.png", png)
        zf.writestr("ppt/media/media1.mp4", b"not-a-real-mp4-but-valid-for-parser")
        zf.writestr("ppt/slides/_rels/slide1.xml.rels", """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="image" Target="../media/image1.png"/><Relationship Id="rId2" Type="video" Target="../media/media1.mp4"/></Relationships>""")
        zf.writestr("ppt/slides/slide1.xml", """<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:cSld><p:spTree><p:nvGrpSpPr/><p:grpSpPr/><p:pic><p:nvPicPr><p:cNvPr id="2" name="Combined01_spedup_30s_hq_playback"/></p:nvPicPr><p:blipFill><a:blip r:embed="rId1"><a:extLst><a:ext><a:videoFile r:link="rId2"/></a:ext></a:extLst></a:blip></p:blipFill><p:spPr><a:xfrm><a:off x="1000000" y="1000000"/><a:ext cx="2000000" cy="1000000"/></a:xfrm><a:prstGeom prst="rect"/></p:spPr></p:pic></p:spTree></p:cSld><p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst><p:par><p:cTn id="3" fill="hold"><p:childTnLst><p:par><p:cTn id="4" presetID="1" presetClass="mediacall" fill="hold" nodeType="withEffect"><p:childTnLst><p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr><p:cTn id="5" dur="1000" fill="hold"/><p:tgtEl><p:spTgt spid="2"/></p:tgtEl></p:cBhvr></p:cmd></p:childTnLst></p:cTn></p:par><p:par><p:cTn id="6" presetID="2" presetClass="mediacall" fill="hold" nodeType="withEffect"><p:childTnLst><p:cmd type="call" cmd="togglePause"><p:cBhvr><p:cTn id="7" dur="1" fill="hold"/><p:tgtEl><p:spTgt spid="2"/></p:tgtEl></p:cBhvr></p:cmd></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:seq></p:childTnLst></p:cTn></p:par></p:tnLst></p:timing></p:sld>""")


def _slide_xml(num: str, *, morph: bool, text: str, zero_transition: bool = False) -> str:
    if morph:
        transition = (
        '<p:transition xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" p14:dur="2000">'
        '<p159:morph xmlns:p159="http://schemas.microsoft.com/office/powerpoint/2015/09/main" option="byObject"/>'
        "</p:transition>"
        )
    elif zero_transition:
        transition = '<p:transition xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" p14:dur="0"/>'
    else:
        transition = ""
    timing = (
        '<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">'
        '<p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq">'
        '<p:childTnLst><p:par><p:cTn id="3" fill="hold"><p:stCondLst><p:cond delay="indefinite"/>'
        '<p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond></p:stCondLst><p:childTnLst><p:par>'
        '<p:cTn id="4" presetID="1" presetClass="mediacall" presetSubtype="0" fill="hold" nodeType="withEffect">'
        '<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst><p:cmd type="call" cmd="playFrom(0.0)">'
        '<p:cBhvr><p:cTn id="5" dur="1000" fill="hold"/><p:tgtEl><p:spTgt spid="2"/></p:tgtEl></p:cBhvr>'
        '</p:cmd></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par></p:childTnLst>'
        '</p:cTn></p:seq></p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>'
        if text
        else ""
    )
    text_box = (
        f"""<p:sp><p:nvSpPr><p:cNvPr id="10" name="TextBox 1"/></p:nvSpPr><p:spPr><a:xfrm><a:off x="1000000" y="1000000"/><a:ext cx="3000000" cy="800000"/></a:xfrm><a:prstGeom prst="rect"/><a:ln><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></a:ln></p:spPr><p:txBody><a:bodyPr lIns="0" rIns="0" tIns="0" bIns="0"><a:spAutoFit/></a:bodyPr><a:p><a:pPr algn="ctr"/><a:r><a:rPr sz="2800" b="1"><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></a:rPr><a:t>{text}</a:t></a:r><a:r><a:rPr sz="2800" b="1"><a:solidFill><a:srgbClr val="FF0000"/></a:solidFill></a:rPr><a:t> Red</a:t></a:r></a:p></p:txBody></p:sp>"""
        if text
        else ""
    )
    x = "1000000" if num == "1" else "2000000"
    return f"""<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:cSld><p:spTree><p:nvGrpSpPr/><p:grpSpPr/><p:pic><p:nvPicPr><p:cNvPr id="2" name="Picture 1"/></p:nvPicPr><p:blipFill><a:blip r:embed="rId1"/></p:blipFill><p:spPr><a:xfrm><a:off x="{x}" y="1000000"/><a:ext cx="2000000" cy="1000000"/></a:xfrm><a:prstGeom prst="rect"/><a:effectLst><a:outerShdw><a:srgbClr val="000000"><a:alpha val="22000"/></a:srgbClr></a:outerShdw></a:effectLst></p:spPr></p:pic>{text_box}</p:spTree></p:cSld>{transition}{timing}</p:sld>"""


def _scene_object(
    object_id: str,
    name: str,
    kind: str,
    asset_id: str | None,
    *,
    shape_id: str | None = None,
) -> SceneObject:
    return SceneObject(
        id=object_id,
        shape_id=shape_id,
        creation_id=None,
        name=name,
        kind=kind,
        z=1,
        geometry=Geometry(x=0, y=0, cx=100, cy=100),
        asset_id=asset_id,
        shape="rect",
    )


if __name__ == "__main__":
    unittest.main()
