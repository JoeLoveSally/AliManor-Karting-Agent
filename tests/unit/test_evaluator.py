import numpy as np

from karting_agent.train.evaluator import BinaryMetricAccumulator


def test_binary_metric_accumulator() -> None:
    metrics = BinaryMetricAccumulator(threshold=0.5)
    metrics.update(
        np.asarray([0.9, 0.8, 0.2, 0.1]),
        np.asarray([1.0, 0.0, 1.0, 0.0]),
    )

    result = metrics.result()

    assert result.samples == 4
    assert result.accuracy == 0.5
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.f1 == 0.5


def test_empty_metric_accumulator_returns_zero_metrics() -> None:
    result = BinaryMetricAccumulator().result()

    assert result.samples == 0
    assert result.accuracy == 0.0
    assert result.f1 == 0.0
