"""Pure-Python tests for read-only kart-state quantization semantics."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/audit_visual_state_observability.py"
spec = importlib.util.spec_from_file_location("_visual_observability_test", SCRIPT)
assert spec is not None and spec.loader is not None
subject = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subject
spec.loader.exec_module(subject)


def pose(angle=10.0):
    return {
        "center_x_norm": 0.45, "center_y_norm": 0.50,
        "heading_angle_deg": angle, "heading_quality": 0.80,
    }


def relation(lateral=0.2, heading=10.0, axis=0.0, confidence=0.8):
    return {
        "lateral_offset_norm": lateral,
        "heading_error_deg": heading,
        "road_angle_deg": axis,
        "road_support_fraction": 0.5,
        "inside_road": True,
        "confidence": confidence,
    }


def make(t, previous=None, kart=None, road=None, geometry=None, tracker="stable"):
    return subject.quantize_measurement(
        frame_index=int(t), timestamp_ms=float(t),
        timestamp_source="adb_json_host_decoded_timestamp",
        pose=pose() if kart is None else kart,
        relation=relation() if road is None else road,
        geometry={} if geometry is None else geometry,
        tracker_mode=tracker,
        previous=previous,
    )


def test_valid_measurement_exports_explicitly_unknown_next_corner():
    result = make(100.0, geometry={
        "geometry_class": "corner_visible",
        "corner_visible": True, "corner_distance_norm": 0.17,
        "corner_score": 0.6,
    })
    assert result["state_valid"]
    assert result["visible_axis_intersection_distance_norm"] == pytest.approx(0.17)
    assert result["next_corner_direction"] is None
    assert result["next_corner_distance"] is None
    assert result["next_corner_valid"] is False
    assert result["lateral_rate_per_s"] is None


def test_missing_pose_masks_derived_state():
    result = make(100.0, kart={}, road=None)
    # A malformed pose cannot silently be treated as valid. Use the physically
    # missing case to test teacher coverage instead.
    result = subject.quantize_measurement(
        frame_index=1, timestamp_ms=100.0,
        timestamp_source="synthetic",
        pose=None, relation=relation(), geometry={},
        tracker_mode="missing",
    )
    assert result["state_valid"] is False
    assert result["lateral_offset_norm"] is None
    assert result["heading_error_axial_deg"] is None


def test_low_confidence_masks_state_but_preserves_raw_diagnostic():
    result = make(100.0, road=relation(confidence=0.01))
    assert result["state_valid"] is False
    assert result["lateral_offset_norm"] is None
    assert result["raw_relation"]["lateral_offset_norm"] == 0.2


def test_derivatives_are_temporal_differences_not_world_velocity():
    first = make(100.0, road=relation(lateral=0.2, heading=10.0))
    second = make(200.0, previous=first,
                  road=relation(lateral=0.3, heading=15.0))
    assert second["lateral_rate_per_s"] == pytest.approx(1.0)
    assert second["heading_rate_deg_per_s"] == pytest.approx(50.0)
    assert second["temporal_pair_dt_ms"] == pytest.approx(100.0)


def test_derivative_rejects_large_gap_and_axis_reassignment():
    first = make(100.0)
    gap = make(600.0, previous=first)
    assert gap["lateral_rate_per_s"] is None
    jumped = make(200.0, previous=first,
                  road=relation(axis=65.0))
    assert jumped["heading_rate_deg_per_s"] is None


def test_axial_wrap_has_small_signed_difference():
    assert subject.axial_difference_deg(2.0, 178.0) == pytest.approx(4.0)
    assert subject.axial_difference_deg(178.0, 2.0) == pytest.approx(-4.0)


def test_summary_is_coverage_not_accuracy():
    first = make(100.0)
    second = make(200.0, previous=first,
                  road=relation(lateral=0.3))
    unavailable = make(300.0, road=relation(confidence=0.0))
    summary = subject.summarize_rows([first, second, unavailable])
    assert summary["sample_count"] == 3
    assert summary["valid_relation_rate"] == pytest.approx(2/3)
    assert summary["temporal_rate_coverage"] == pytest.approx(1/3)
    assert summary["candidate_next_corner_coverage"] == 0.0
    assert "NOT accuracy" in summary["note"]
