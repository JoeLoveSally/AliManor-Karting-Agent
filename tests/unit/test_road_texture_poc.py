"""Deterministic CPU-only contracts for texture/geometry visual quantization."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from karting_agent.train import road_texture_poc as subject  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "_audit_road_texture_poc_test", ROOT / "scripts/audit_road_texture_poc.py"
)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def synthetic_track(*, road_low=(76, 95, 135), road_high=(105, 120, 155)):
    """Checker tiles on a smooth contrasting background, no kart sprite."""
    frame = np.zeros((800, 360, 3), np.uint8)
    frame[:] = (186, 148, 91)
    for y in range(300, 510, 24):
        for x in range(95, 270, 24):
            frame[y:y + 24, x:x + 24] = (
                road_high if ((x // 24) + (y // 24)) % 2 else road_low
            )
    return frame


def test_texture_responds_to_repeated_road_paving_not_flat_background():
    image = synthetic_track()
    texture = subject.texture_evidence(image)
    road_energy = float(texture[325:485, 115:250].mean())
    background_energy = float(texture[50:150, 10:100].mean())
    assert texture.shape == image.shape[:2]
    assert texture.dtype == np.float32
    assert 0 <= texture.min() <= texture.max() <= 1
    assert road_energy > background_energy + 0.12


def test_theme_color_does_not_remove_grayscale_checker_response():
    blue = synthetic_track(road_low=(65, 89, 130), road_high=(100, 123, 161))
    teal = synthetic_track(road_low=(95, 125, 40), road_high=(132, 165, 70))
    for image in (blue, teal):
        score = subject.texture_evidence(image)
        assert score[335:475, 125:245].mean() > 0.12


def test_empty_flat_frame_refuses_to_fabricate_road():
    image = np.full((800, 360, 3), (100, 145, 190), np.uint8)
    maps, info = subject.quantify_texture_frame(image, (180.0, 400.0))
    assert info["candidate_valid_unreviewed"] is False
    assert info["reason"] == "no_texture_seed"
    assert np.count_nonzero(maps["candidate"]) == 0
    assert np.all(maps["unknown"] == 255)


def test_missing_kart_invalid_even_if_checker_texture_visible():
    image = synthetic_track()
    maps, info = subject.quantify_texture_frame(image, None)
    assert info["candidate_valid_unreviewed"] is False
    assert info["reason"] == "kart_missing"
    assert np.count_nonzero(maps["candidate"]) == 0


def test_local_texture_seed_is_present_for_checker_track():
    image = synthetic_track()
    maps, info = subject.quantify_texture_frame(image, (180.0, 400.0))
    assert info["candidate_valid_unreviewed"] is True
    assert info["seed_support_pixels"] >= 12
    assert maps["candidate"].shape == (800, 360)
    assert maps["candidate"][360, 260] > 0.0
    assert info["line_count_near_kart"] >= 1


def test_geometry_does_not_require_two_line_families():
    image = np.zeros((400, 240, 3), np.uint8)
    cv2.line(image, (30, 140), (215, 140), (255, 255, 255), 3)
    geometry, meta = subject.line_geometry_evidence(image, (100., 140.))
    assert geometry.shape == image.shape[:2]
    assert 0 <= geometry.min() <= geometry.max() <= 1
    assert meta["line_count_near_kart"] > 0
    assert len(meta["dominant_axes_deg"]) >= 1


def test_unknown_near_kart_and_action_roi_masks_feature_planes():
    image = synthetic_track()
    redaction = np.zeros((800, 360), np.uint8)
    redaction[660:780, 285:355] = 255
    maps, info = subject.quantify_texture_frame(
        image, (180., 400.), redaction_mask=redaction
    )
    assert maps["unknown"][400, 180] == 255
    assert maps["candidate"][400, 180] == 0
    assert maps["unknown"][720, 318] == 255
    assert maps["texture"][720, 318] == 0
    assert maps["geometry"][720, 318] == 0
    assert info["unknown_fraction"] > 0


def test_eight_channels_preserve_unknown_and_local_wide_difference():
    maps, _ = subject.quantify_texture_frame(
        synthetic_track(), (180., 400.)
    )
    grids = audit.prepare_feature_grids(maps, (180., 400.))
    assert grids.shape == (8, 32, 32)
    assert grids.dtype == np.float32
    assert np.all((0 <= grids) & (grids <= 1))
    assert audit.CHANNELS[0] == "local_candidate"
    assert audit.CHANNELS[4] == "wide_candidate"
    assert grids[3, 16, 16] < .5  # physically occluded kart location
    assert grids[7, 16, 16] < .5


def test_preview_is_actual_frame_with_evidence_not_generic_illustration():
    image = synthetic_track()
    maps, _ = subject.quantify_texture_frame(image, (180., 400.))
    grids = audit.prepare_feature_grids(maps, (180., 400.))
    hsv = np.zeros(image.shape[:2], dtype=np.uint8)
    panel = audit.preview_panel(image, hsv, maps, grids, (180., 400.))
    assert panel.shape == (192, 8 * 192, 3)
    assert np.count_nonzero(panel) > 0


def test_invalid_shapes_raise_clear_errors():
    with pytest.raises(ValueError, match="expected BGR"):
        subject.texture_evidence(np.zeros((50, 50), np.uint8))
    with pytest.raises(ValueError, match="texture/geometry"):
        subject.anchored_soft_candidate(
            np.zeros((8, 8), np.float32),
            np.zeros((7, 7), np.float32),
            (4., 4.),
        )
    with pytest.raises(ValueError, match="redaction mask"):
        subject.quantify_texture_frame(
            np.zeros((100, 100, 3), np.uint8), (50., 50.),
            redaction_mask=np.ones((20, 20), np.uint8),
        )
