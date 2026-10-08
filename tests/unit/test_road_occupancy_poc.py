"""Synthetic, CPU-only tests for kart-anchored occupancy representation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/audit_road_occupancy_poc.py"
spec = importlib.util.spec_from_file_location("_road_occupancy_poc_test", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_local_component_preferred_over_bigger_distant_component():
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[2:48, 2:48] = 255
    mask[65:90, 65:90] = 255
    selected, meta = module.select_kart_local_component(
        mask, (76.0, 76.0), radius_px=15,
    )
    assert meta["seed_found"] is True
    assert selected[76, 76] == 255
    assert selected[10, 10] == 0


def test_no_road_near_kart_is_marked_uncertain_not_remote_association():
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[0:20, 0:20] = 255
    selected, meta = module.select_kart_local_component(
        mask, (100.0, 100.0), radius_px=12,
    )
    assert not meta["seed_found"]
    assert np.count_nonzero(selected) == 0
    assert meta["reason"] == "no_local_road_support"


def test_missing_kart_never_selects_far_component():
    mask = np.ones((80, 80), dtype=np.uint8) * 255
    selected, meta = module.select_kart_local_component(mask, None)
    assert not meta["seed_found"]
    assert meta["reason"] == "kart_not_detected"
    assert np.count_nonzero(selected) == 0


def test_out_of_bounds_is_different_from_off_road():
    mask = np.full((64, 64), 255, dtype=np.uint8)
    crop, road, known = module.crop_and_downsample(
        mask, (0, 0), patch_width=64, patch_height=64, out_size=32
    )
    assert crop.shape == (64, 64)
    assert road.shape == (32, 32)
    assert known.shape == (32, 32)
    assert known.mean() < 0.4
    assert np.all(road <= known + 1e-6)


def test_quantizer_shapes_and_local_wide_view_differ():
    frame = np.zeros((480, 360, 3), dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[170:320, 90:280] = 255
    planes, audit, images = module.quantize_frame(
        frame, mask, (180.0, 240.0),
        nominal_center=(180.0, 240.0),
    )
    assert planes.shape == (4, 32, 32)
    assert np.all((planes >= 0) & (planes <= 1))
    assert audit["quantizer_valid_unreviewed"] is True
    assert planes[0].mean() > planes[2].mean()
    assert images["selected_mask"].shape == mask.shape


def test_quantizer_missing_pose_marks_invalid_even_with_road_color():
    frame = np.zeros((480, 360, 3), dtype=np.uint8)
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    planes, audit, _ = module.quantize_frame(
        frame, mask, None, nominal_center=(180.0, 240.0)
    )
    assert audit["quantizer_valid_unreviewed"] is False
    assert audit["reason"] == "kart_not_detected"
    assert np.count_nonzero(planes[0]) == 0
    assert planes[1].mean() > 0


def test_component_mask_requires_uint8():
    mask = np.zeros((30, 30), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        module.select_kart_local_component(mask, (15, 15))


def test_model_input_planes_do_not_include_action_labels():
    frame = np.zeros((480, 360, 3), dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[170:320, 90:280] = 255
    _, audit, _ = module.quantize_frame(
        frame, mask, (180.0, 240.0), nominal_center=(180, 240)
    )
    assert not any(key in audit for key in ("pressed", "action", "switch"))


def test_enclosed_hole_is_unknown_not_confirmed_obstacle_or_road():
    selected = np.zeros((96, 96), dtype=np.uint8)
    selected[8:88, 8:88] = 255
    selected[40:48, 44:52] = 0
    unknown, meta = module.enclosed_mask_unknown(selected)
    assert meta["enclosed_unknown_regions"] == 1
    assert unknown[44, 48] == 255
    assert unknown[0, 0] == 0

    observed = np.where(unknown > 0, 0, 255).astype(np.uint8)
    _, road, known = module.crop_and_downsample(
        selected, (48, 48), patch_width=96, patch_height=96,
        observed_mask=observed, out_size=32,
    )
    assert road[15, 16] == 0.0
    assert known[15, 16] == 0.0
    assert known[0, 0] == 1.0


def test_background_connected_gap_remains_known_offroad():
    selected = np.zeros((64, 64), dtype=np.uint8)
    selected[16:48, 16:48] = 255
    selected[16:35, 31:34] = 0  # reaches the exterior at road top edge
    unknown, meta = module.enclosed_mask_unknown(selected)
    assert np.count_nonzero(unknown) == 0
    assert meta["enclosed_unknown_regions"] == 0


def test_large_internal_gap_not_silently_hallucinated_as_occlusion():
    selected = np.full((100, 100), 255, dtype=np.uint8)
    selected[25:75, 25:75] = 0
    unknown, meta = module.enclosed_mask_unknown(
        selected, max_hole_area_fraction=0.01,
    )
    assert meta["enclosed_unknown_regions"] == 0
    assert unknown[50, 50] == 0


def test_quantize_exposes_occlusion_in_existing_known_planes():
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    road_mask = np.zeros((240, 240), dtype=np.uint8)
    road_mask[15:225, 15:225] = 255
    road_mask[115:125, 115:125] = 0
    planes, meta, images = module.quantize_frame(
        frame, road_mask, (120, 120), nominal_center=(120, 120),
    )
    assert planes.shape == (4, 32, 32)
    assert meta["enclosed_unknown_regions"] == 1
    assert images["unknown_mask"][120, 120] == 255
    assert planes[1, 16, 16] < 1.0
    assert planes[3, 16, 16] < 1.0


def test_touch_marker_roi_is_unknown_and_excluded_from_road_features():
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    road_mask = np.ones((240, 240), dtype=np.uint8) * 255
    # Deliberately cover a road portion with the action-label exclusion ROI.
    redaction = module.roi_redaction_mask(
        road_mask.shape, (0.70, 0.70, 0.90, 0.90)
    )
    planes, meta, images = module.quantize_frame(
        frame, road_mask, (120, 120), nominal_center=(120, 120),
        local_size=(240, 240), wide_size=(240, 240),
        redaction_mask=redaction,
    )
    assert meta["label_ui_redaction_enabled"] is True
    assert images["selected_mask"][190, 190] == 0
    assert images["unknown_mask"][190, 190] == 255
    # Redacted input must be zero occupancy AND zero known, not a confident
    # negative feature from the original touch-marker action label.
    assert planes[0, 25, 25] == 0
    assert planes[1, 25, 25] == 0
    assert planes[2, 25, 25] == 0
    assert planes[3, 25, 25] == 0
    assert planes[0, 12, 12] > .99
    assert planes[1, 12, 12] > .99


def test_normalized_touch_roi_validation():
    import pytest
    with pytest.raises(ValueError, match="normalized touch ROI"):
        module.roi_redaction_mask((100, 100), (0.9, 0.8, 0.4, 0.95))
    result = module.roi_redaction_mask(
        (100, 100), (0.78, 0.82, 0.98, 0.98)
    )
    assert result.shape == (100, 100)
    assert result.dtype == np.uint8
    assert result[90, 85] == 255
    assert result[30, 30] == 0


def test_existing_adb_mode_remains_unredacted_by_default():
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    road_mask = np.ones((240, 240), dtype=np.uint8) * 255
    _, meta, images = module.quantize_frame(
        frame, road_mask, (120, 120), nominal_center=(120, 120)
    )
    assert meta["label_ui_redaction_enabled"] is False
    assert meta["label_ui_redaction_fraction"] == 0.0
    assert images["unknown_mask"].shape == (240, 240)
