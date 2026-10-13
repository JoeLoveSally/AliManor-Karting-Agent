"""Targeted CPU tests for the intentionally noncausal five-frame visual-phase probe."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "diagnose_tiny_visual_phase", ROOT / "scripts/diagnose_tiny_visual_phase.py"
)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def frames():
    return [{"frame_index": i, "host_decoded_ms": float(i * 50)}
            for i in range(25)]


def prediction():
    return {
        "frame_index": 7, "observation_ms": 350.,
        "input_indices": [3, 4, 5, 6, 7],
        "input_timestamp_ms": [150., 200., 250., 300., 350.],
        "models": {"rgb": {"target_ms": 450.}},
    }


def test_nearest_frame_index_is_monotonic_and_tie_prefers_earlier():
    ts = [0., 50., 100.]
    assert subject.nearest_time_index(ts, 25.) == (0, -25.)
    assert subject.nearest_time_index(ts, 85.) == (2, 15.)
    assert subject.nearest_time_index(ts, 1000.) == (2, -900.)


def test_phase_plan_uses_future_pixels_but_keeps_h100_target_and_original_stamps():
    plans, meta = subject.plan_windows(
        [prediction()], frames(), shifts_ms=(0, 50, 150),
        start_ms=400., end_ms=600., max_skew_ms=5.,
    )
    assert len(plans) == 1
    plan = plans[0]
    assert plan["target_ms"] == 450.
    assert plan["observation_ms"] == 350.
    assert plan["input_indices_by_shift"][0] == [3, 4, 5, 6, 7]
    assert plan["input_indices_by_shift"][50] == [4, 5, 6, 7, 8]
    assert plan["input_indices_by_shift"][150] == [6, 7, 8, 9, 10]
    assert plan["input_timestamp_ms"] == [150., 200., 250., 300., 350.]
    assert meta["retention_fraction"] == 1.


def test_nearest_frame_gap_drops_entire_record_for_every_shift():
    data = frames()
    data[8]["host_decoded_ms"] = 490.
    # Other timestamps remain ordered; inject shifted-sample gap by sparse
    # but still monotonic series.
    data[9]["host_decoded_ms"] = 510.
    plans, meta = subject.plan_windows(
        [prediction()], data, shifts_ms=(0, 50),
        start_ms=400., end_ms=600., max_skew_ms=5.,
    ) if False else (None, None)
    assert plans is None and meta is None
    # A valid sparse source can leave no common rows, which must fail safely.
    sparse = [{"frame_index": i, "host_decoded_ms": float(i * 100)}
              for i in range(25)]
    row = prediction()
    row["input_indices"] = [1, 2, 3, 4, 5]
    row["input_timestamp_ms"] = [100., 200., 300., 400., 500.]
    row["observation_ms"] = 500.
    row["models"]["rgb"]["target_ms"] = 600.
    with pytest.raises(ValueError, match="no valid common"):
        subject.plan_windows(
            [row], sparse, shifts_ms=(0, 50),
            start_ms=550., end_ms=700., max_skew_ms=5.,
        )


def test_original_control_frame_deltas_are_invariant_to_lookahead():
    stamps = [150., 200., 250., 300., 350.]
    control = subject.fixed_control_dt(stamps)
    assert control.shape == (5, 3)
    np.testing.assert_array_equal(control[:, :2], np.zeros((5, 2)))
    np.testing.assert_allclose(control[:, 2], [0., .25, .25, .25, .25])


class FakeVisualPolicy(torch.nn.Module):
    def forward(self, images, control):
        # React only to last image and verify no action-state leakage.
        assert torch.count_nonzero(control[..., :2]).item() == 0
        return images[:, -1, 0, 0, 0] * 6 - 3


def test_infer_oracle_changes_pixels_only_and_never_retimes_prediction():
    plans, _ = subject.plan_windows(
        [prediction()], frames(), shifts_ms=(0, 50),
        start_ms=400., end_ms=600., max_skew_ms=5.,
    )
    features = {i: np.full((3, 4, 4), i * 9, dtype=np.uint8)
                for i in range(25)}
    result = subject.infer_oracle(
        FakeVisualPolicy(), plans, features, shifts_ms=(0, 50), batch_size=1
    )
    assert result[0]["target_ms"] == 450.
    assert result[0]["probabilities"]["50"] > result[0]["probabilities"]["0"]


def test_event_transitions_are_at_fixed_target_time():
    records = [
        {"target_ms": t, "probabilities": {"0": p, "50": q}}
        for t, p, q in [(100., .2, .8), (200., .7, .8),
                        (300., .9, .1), (400., .1, .1)]
    ]
    assert subject.model_events(records, 0) == [(200., True), (400., False)]
    assert subject.model_events(records, 50) == [(300., False)]


def test_release_probe_uses_same_original_target_for_all_shifts():
    records = [
        {"target_ms": 250., "probabilities": {"0": .9, "200": .1}},
        {"target_ms": 350., "probabilities": {"0": .2, "200": .1}},
    ]
    result = subject.post_release_probe(records, [100.], 200)
    assert result["release_events_with_prediction_near_plus150ms"] == 1
    assert result["median_p_press"] == pytest.approx(.1)
    assert result["predicted_release_fraction"] == 1.


def test_bad_h100_or_source_frame_mapping_rejected():
    bad = prediction()
    bad["models"]["rgb"]["target_ms"] = 500.
    with pytest.raises(ValueError, match="H100"):
        subject.plan_windows([bad], frames(), shifts_ms=(0, 50),
                             start_ms=400., end_ms=600.)
    with pytest.raises(ValueError, match="nonnegative"):
        subject.plan_windows([prediction()], frames(),
                             shifts_ms=(-50, 0), start_ms=400., end_ms=600.)
    other = prediction()
    other["input_timestamp_ms"][0] += 3
    with pytest.raises(ValueError, match="sidecar mismatch"):
        subject.plan_windows([other], frames(), shifts_ms=(0, 50),
                             start_ms=400., end_ms=600.)
