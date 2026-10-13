"""CPU-only tests for read-only Tiny policy history sensitivity diagnosis."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "diagnose_tiny_action_history_sensitivity",
    ROOT / "scripts/diagnose_tiny_action_history_sensitivity.py",
)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def stamps():
    return [0., 50., 100., 150., 200.]


def controls(pressed=False, age_ms=250.):
    return subject.coherent_held_controls(
        stamps(), pressed=pressed, age_at_observation_ms=age_ms,
    )


def test_held_controls_follow_causal_ages_and_frame_delta():
    state = controls(pressed=True, age_ms=350.)
    np.testing.assert_array_equal(state[:, 0], np.ones(5))
    np.testing.assert_allclose(state[:, 1], [.3, .4, .5, .6, .7])
    np.testing.assert_allclose(state[:, 2], [0., .25, .25, .25, .25])
    with pytest.raises(ValueError, match="no later"):
        subject.coherent_held_controls(stamps(), pressed=False,
                                       age_at_observation_ms=150.)


def test_factors_replace_exactly_one_channel_without_inplace_mutation():
    teacher = controls(pressed=False, age_ms=350.)
    virtual = controls(pressed=True, age_ms=500.)
    baseline = teacher.copy()
    scenarios = subject.make_scenarios(stamps(), teacher, virtual)
    assert set(scenarios) == set(subject.SCENARIOS)
    np.testing.assert_array_equal(scenarios["teacher"], teacher)
    np.testing.assert_array_equal(scenarios["virtual"], virtual)
    np.testing.assert_array_equal(scenarios["virtual_state_teacher_age"][:, 0], virtual[:, 0])
    np.testing.assert_array_equal(scenarios["virtual_state_teacher_age"][:, 1], teacher[:, 1])
    np.testing.assert_array_equal(scenarios["teacher_state_virtual_age"][:, 0], teacher[:, 0])
    np.testing.assert_array_equal(scenarios["teacher_state_virtual_age"][:, 1], virtual[:, 1])
    np.testing.assert_array_equal(teacher, baseline)
    for scenario in scenarios.values():
        np.testing.assert_array_equal(scenario[:, 2], teacher[:, 2])


def test_release_selection_uses_absolute_target_time_and_no_duplicate_rows():
    records = [{"frame_index": i, "models": {"rgb": {"target_ms": float(i * 50)}}}
               for i in range(1, 15)]
    selected = subject.select_release_rows(records, [300., 350.], lower=-100., upper=150.)
    assert len(selected) == len({r["frame_index"] for r, _ in selected})
    assert [r["frame_index"] for r, _ in selected] == list(range(4, 11))
    assert subject.release_events([(0., False), (100., True), (300., False)],
                                  0., 1000.) == [300.]
    # An asymmetric window must choose an eligible earlier event even if a
    # slightly closer future event falls outside the allowed negative window.
    row = {"models": {"rgb": {"target_ms": 1200.}}, "frame_index": 77}
    assert subject.select_release_rows([row], [950., 1410.],
                                       lower=-200., upper=300.) == [(row, 250.)]


class ModelUsesControls(torch.nn.Module):
    def forward(self, images, control):
        return 4 * (control[:, -1, 0] - .5) + 2 * (control[:, -1, 1] - .5)


def test_paired_inference_uses_identical_pixels_and_original_virtual():
    # Inject real project's small control-feature contract without importing
    # repo-wide experiment dependencies into this pure unit test.
    import sys
    import types
    from bisect import bisect_right

    fake = types.ModuleType("evaluate_tiny_live_recording")

    def control_features(ts, events):
        out = np.zeros((len(ts), 3), dtype=np.float32)
        for i, t in enumerate(ts):
            idx = bisect_right([e[0] for e in events], t) - 1
            at, pressed = events[idx]
            out[i, 0] = pressed
            out[i, 1] = min((t - at) / 500, 1)
            out[i, 2] = (t - ts[i - 1]) / 200 if i else 0
        return out

    fake.control_features = control_features
    existing = sys.modules.get("evaluate_tiny_live_recording")
    sys.modules["evaluate_tiny_live_recording"] = fake
    try:
        selected = [({"frame_index": 4, "observation_ms": 200.,
                      "input_timestamp_ms": stamps(), "input_indices": list(range(5)),
                      "models": {"rgb": {"target_ms": 300., "probability": .5}}},
                     0.)]
        features = {i: np.zeros((3, 96, 96), dtype=np.uint8) for i in range(5)}
        rows = subject.predict_scenarios(
            ModelUsesControls(), selected, features,
            [(0., False)], [(0., True)], batch_size=1,
        )
        assert len(rows) == 1
        probs = rows[0]["probabilities"]
        assert probs["virtual"] > probs["teacher"]
        assert probs["virtual_state_teacher_age"] > probs["teacher"]
        assert probs["held_release_500"] < probs["held_press_500"]
        assert "images" not in rows[0]
    finally:
        if existing is None:
            del sys.modules["evaluate_tiny_live_recording"]
        else:
            sys.modules["evaluate_tiny_live_recording"] = existing


def test_metrics_separate_state_age_and_interaction():
    item = {k: .5 for k in subject.SCENARIOS}
    item.update({"teacher": .2, "virtual": .8,
                 "virtual_state_teacher_age": .6, "teacher_state_virtual_age": .3})
    metrics, effects = subject.sensitivity_metrics([{"probabilities": item}])
    assert metrics["virtual"]["threshold_disagreements_with_teacher"] == 1
    assert effects["state_replacement"]["median_delta_p_press"] == pytest.approx(.4)
    assert effects["age_replacement"]["median_delta_p_press"] == pytest.approx(.1)
    assert effects["interaction"]["median_delta_p_press"] == pytest.approx(.1)
    bins = subject.release_bin_summary([{"relative_to_release_ms": 151.,
                                         "probabilities": item}], half_width=20)
    assert bins["150"]["n"] == 1
    assert bins["150"]["teacher_release_virtual_hold_count"] == 1
