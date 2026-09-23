"""Synthetic, CPU-only checks for paired H200 threshold selection."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/sweep_v4c2_history_threshold.py"
SPEC = importlib.util.spec_from_file_location("v4c2_threshold_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def test_base_threshold_stays_fixed_when_candidate_changes():
    labels = np.array([0, 0, 1, 1])
    pb = np.array([.7, .2, .7, .3])
    pn = np.array([.8, .4, .8, .4])
    low = sweep.score(labels, pb, pn, .60, .60)
    high = sweep.score(labels, pb, pn, .60, .90)
    assert (low["base_fp"], high["base_fp"], low["base_tp"], high["base_tp"]) == (1, 1, 1, 1)
    assert (low["new_fp"], high["new_fp"]) == (1, 0)


def test_pair_count_invariant():
    result = sweep.score(np.array([0, 0, 1, 1]), np.array([.8, .2, .2, .8]),
                         np.array([.1, .8, .8, .1]), .60, .60)
    assert result["fixed_fp"] == result["added_fp"] == 1
    assert result["fixed_fn"] == result["added_fn"] == 1
    assert result["base_fp"] - result["fixed_fp"] + result["added_fp"] == result["new_fp"]
    assert result["base_fn"] - result["fixed_fn"] + result["added_fn"] == result["new_fn"]


def test_scan_uses_all_and_early_window_gates():
    labels = np.array([0, 1, 0, 1])
    pb = np.array([.7, .8, .3, .4])
    pn = np.array([.65, .9, .61, .85])
    rows = {"all": (labels, pb, pn),
            "post_switch_0_100ms": (labels[:2], pb[:2], pn[:2])}
    result, selected = sweep.scan(rows, baseline_threshold=.6, thresholds=[.6, .7])
    assert not result[0]["eligible"]
    assert result[1]["eligible"]
    assert selected["threshold"] == .7
    assert selected["groups"]["all"]["new_tp"] == 2


def test_no_eligible_candidate_does_not_select():
    labels = np.array([0, 1])
    pb = np.array([.1, .9])
    pn = np.array([.8, .2])
    rows = {"all": (labels, pb, pn), "post_switch_0_100ms": (labels, pb, pn)}
    results, selected = sweep.scan(rows, baseline_threshold=.6, thresholds=[.6, .7])
    assert selected is None and all(not row["eligible"] for row in results)


def test_grid_and_invalid_intervals():
    assert sweep.threshold_grid(.55, .57, .01) == [.55, .56, .57]
    with pytest.raises(ValueError):
        sweep.threshold_grid(.55, .575, .01)
    with pytest.raises(ValueError):
        sweep.threshold_grid(0, .6, .01)


def test_reject_missing_early_positive():
    rows = {"all": (np.array([0, 1]), np.array([.2, .8]), np.array([.2, .8])),
            "post_switch_0_100ms": (np.array([0]), np.array([.2]), np.array([.2]))}
    with pytest.raises(ValueError):
        sweep.scan(rows, baseline_threshold=.6, thresholds=[.6])


def test_reject_nonfinite_probability():
    with pytest.raises(ValueError):
        sweep.score(np.array([1]), np.array([float("nan")]), np.array([.8]), .6, .6)
