from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ppt_chunker.manifest import build_manifest
from ppt_chunker.models import Hiccup, MediaCandidate, ResolvedSegment, SlideFeature
from ppt_chunker.pipeline import (
    _has_error_hiccups,
    _resolve_timing_decisions,
    _slide_asset_stem,
    _transition_mode_token,
    _transition_asset_stem,
)
from ppt_chunker.timing import segments_from_config


class PipelineDecisionTests(unittest.TestCase):
    def test_media_slide_uses_max_media_duration(self) -> None:
        feature = SlideFeature(
            slide_number=1,
            label="Slide 1",
            transition_type="morph",
            transition_duration_s=2.0,
            visible_media_count=2,
            max_visible_media_duration_s=7.25,
            unresolved_visible_media_count=0,
            off_canvas_media_count=0,
            unsupported_media_count=0,
            media_candidates=[],
            static_classification="media",
        )
        decisions, hiccups = _resolve_timing_decisions(
            features=[feature],
            overrides={"slides": {}, "defaults": {}},
            default_static_sec=4.0,
            default_transition_sec=0.5,
        )
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].slide_reason, "media_max")
        self.assertAlmostEqual(decisions[0].slide_duration_s, 7.25, places=3)
        self.assertFalse(_has_error_hiccups(hiccups))

    def test_tiny_transition_is_snapped_to_none(self) -> None:
        feature = SlideFeature(
            slide_number=4,
            label="Slide 4",
            transition_type="morph",
            transition_duration_s=0.01,
            visible_media_count=0,
            max_visible_media_duration_s=None,
            unresolved_visible_media_count=0,
            off_canvas_media_count=0,
            unsupported_media_count=0,
            media_candidates=[],
            static_classification="static",
        )
        decisions, _ = _resolve_timing_decisions(
            features=[feature],
            overrides={"slides": {}, "defaults": {}},
            default_static_sec=4.0,
            default_transition_sec=0.5,
        )
        self.assertEqual(decisions[0].transition_type, "none")
        self.assertEqual(decisions[0].transition_duration_s, 0.0)
        self.assertEqual(decisions[0].transition_reason, "tiny_as_none")

    def test_unresolved_media_raises_error_hiccup(self) -> None:
        feature = SlideFeature(
            slide_number=2,
            label="Slide 2",
            transition_type="none",
            transition_duration_s=0.0,
            visible_media_count=1,
            max_visible_media_duration_s=None,
            unresolved_visible_media_count=1,
            off_canvas_media_count=0,
            unsupported_media_count=0,
            media_candidates=[
                MediaCandidate(
                    shape_id=1,
                    rel_id="rId1",
                    target="ppt/media/video1.mp4",
                    extension=".mp4",
                    visible=True,
                    off_canvas=False,
                    supported=True,
                    duration_s=None,
                    duration_probe_error="probe failed",
                )
            ],
            static_classification="media",
        )
        _, hiccups = _resolve_timing_decisions(
            features=[feature],
            overrides={"slides": {}, "defaults": {}},
            default_static_sec=4.0,
            default_transition_sec=0.5,
        )
        self.assertTrue(_has_error_hiccups(hiccups))

    def test_immediate_override_is_preserved_even_without_authored_duration(self) -> None:
        feature = SlideFeature(
            slide_number=2,
            label="Slide 2",
            transition_type="none",
            transition_duration_s=0.0,
            visible_media_count=0,
            max_visible_media_duration_s=None,
            unresolved_visible_media_count=0,
            off_canvas_media_count=0,
            unsupported_media_count=0,
            media_candidates=[],
            static_classification="static",
        )
        decisions, _ = _resolve_timing_decisions(
            features=[feature],
            overrides={
                "slides": {
                    "2": {
                        "transition_play_mode": "immediate",
                    }
                },
                "defaults": {},
            },
            default_static_sec=4.0,
            default_transition_sec=0.5,
        )
        self.assertEqual(decisions[0].transition_play_mode, "immediate")
        self.assertEqual(decisions[0].transition_duration_s, 0.0)


class ManifestTests(unittest.TestCase):
    def test_sequence_filename_helpers(self) -> None:
        self.assertEqual(
            _slide_asset_stem(slide_number=1, duration_token="4", width=2, extension=".mp4"),
            "01_Slide_01_dur4.mp4",
        )
        self.assertEqual(
            _transition_asset_stem(
                slide_to=2,
                duration_token="1p5",
                transition_mode_token="manual",
                width=2,
            ),
            "02_Transition_01_to_02_dur1p5_navmanual.mp4",
        )
        self.assertEqual(_transition_mode_token("immediate"), "immediate")

    def test_segments_from_config_preserves_immediate_transition_mode(self) -> None:
        segments = segments_from_config(
            {
                "segments": [
                    {
                        "slide_number": 2,
                        "label": "Slide 2",
                        "transition_type": "none",
                        "transition_duration_s": 0.0,
                        "transition_play_mode": "immediate",
                        "slide_duration_s": 4.0,
                    }
                ]
            }
        )
        self.assertEqual(segments[0].transition_play_mode, "immediate")

    def test_manifest_mixed_asset_kind(self) -> None:
        segments = [
            ResolvedSegment(
                slide_number=1,
                label="Slide 1",
                transition_type="none",
                transition_duration_s=0.0,
                slide_duration_s=4.0,
                duration_source="static",
                asset_kind="image",
                static_image_file="assets/slide_01.abc123.png",
            ),
            ResolvedSegment(
                slide_number=2,
                label="Slide 2",
                transition_type="none",
                transition_duration_s=0.0,
                slide_duration_s=8.0,
                duration_source="media",
                asset_kind="video",
                transition_play_mode="immediate",
            ),
        ]
        manifest = build_manifest(
            presentation_id="test_deck",
            source_ppt=None,
            title="Test",
            segments=segments,
            chunk_entries_legacy=[
                {
                    "type": "slide",
                    "file": "assets/01_Slide_01_dur4.abc123.png",
                    "duration": 4.0,
                    "slide_number": 1,
                    "label": "Slide 1",
                    "loop": True,
                    "asset_kind": "image",
                },
                {
                    "type": "transition",
                    "file": "chunks/02_Transition_01_to_02_dur0p033_navimmediate.abc123.mp4",
                    "duration": 0.033,
                    "slide_from": 1,
                    "slide_to": 2,
                    "asset_kind": "video",
                    "transition_play_mode": "immediate",
                },
                {
                    "type": "slide",
                    "file": "chunks/03_Slide_02_dur8.abc123.mp4",
                    "duration": 8.0,
                    "slide_number": 2,
                    "label": "Slide 2",
                    "loop": True,
                    "asset_kind": "video",
                },
            ],
            chunk_entries_extended=[],
            encoding={},
        )
        self.assertEqual(manifest["slides"][0]["asset_kind"], "image")
        self.assertIn("static_image_file", manifest["slides"][0])
        self.assertEqual(manifest["slides"][0]["transition_segment_id"], "trans_02")
        self.assertEqual(manifest["slides"][1]["asset_kind"], "video")
        self.assertNotIn("transition_segment_id", manifest["slides"][1])


if __name__ == "__main__":
    unittest.main()
