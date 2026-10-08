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
