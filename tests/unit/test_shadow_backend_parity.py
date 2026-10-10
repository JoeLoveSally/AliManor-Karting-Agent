"""CPU-only regression coverage for frozen teacher CPU/CUDA parity diagnosis."""
import math

import pytest

from karting_agent.train.shadow_backend_parity import compare_backend_probabilities


def values():
    return {
        ("validation_A.mp4", 300.): .49,
        ("validation_A.mp4", 333.333): .50000006,
        ("validation_A.mp4", 366.667): .93,
        ("validation_B.mp4", 300.): .2,
    }


def test_reports_exact_threshold_crossing_and_target_timestamp():
    cpu = values()
    cuda = dict(cpu)
    cuda[("validation_A.mp4",333.333)] = .49999994
    result = compare_backend_probabilities(cpu, cuda)
    assert result["samples"] == 4
    assert result["disagreeing_binary_samples"] == 1
    assert result["threshold_crossings"][0]["video"] == "validation_A.mp4"
    assert result["threshold_crossings"][0]["target_ms"] == 333.333
    assert result["near_threshold_within_0p01"] >= 1
    assert result["max_absolute_probability_difference"] == pytest.approx(1.2e-7)


def test_equal_predictions_have_no_artificial_crossings():
    cpu = values()
    result = compare_backend_probabilities(cpu,dict(cpu))
    assert result["max_absolute_probability_difference"] == 0
    assert result["disagreeing_binary_samples"] == 0
    assert not result["threshold_crossings"]


def test_all_models_return_same_keys_not_approximate_timestamps():
    cpu = values()
    cuda = dict(cpu)
    del cuda[("validation_B.mp4",300.)]
    with pytest.raises(ValueError,match="sets differ"):
        compare_backend_probabilities(cpu,cuda)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -.2, 1.1])
def test_probability_validation(invalid):
    cpu = values()
    cuda = dict(cpu)
    cuda[("validation_A.mp4",300.)] = invalid
    with pytest.raises(ValueError,match="invalid"):
        compare_backend_probabilities(cpu,cuda)


def test_crossing_limit_does_not_hide_count():
    cpu = values()
    cuda = {key: 1-value for key,value in cpu.items()}
    result = compare_backend_probabilities(cpu,cuda,max_rows=1)
    assert result["disagreeing_binary_samples"] == 4
    assert len(result["threshold_crossings"]) == 1
    assert result["threshold_crossings_truncated"]


@pytest.mark.parametrize("kwargs", [{"threshold":0}, {"threshold":1},
                                    {"max_rows":0}])
def test_invalid_diagnostic_arguments(kwargs):
    with pytest.raises(ValueError):
        compare_backend_probabilities(values(), values(), **kwargs)


def test_reference_timestamp_differences_are_not_normalized_away():
    cpu = {("validation_A", 300.0): .5}
    cuda = {("validation_A", 300.00001): .5}
    with pytest.raises(ValueError,match="sets differ"):
        compare_backend_probabilities(cpu,cuda)
