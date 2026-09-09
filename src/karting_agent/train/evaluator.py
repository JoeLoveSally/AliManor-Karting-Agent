"""Binary classification metrics used during training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class BinaryMetricAccumulator:
    def __init__(self, threshold: float = 0.5) -> None:
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must be in (0, 1)")
        self.threshold = threshold
        self.samples = 0
        self.correct = 0
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def update(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1)
        if probabilities.shape != targets.shape:
            raise ValueError("probabilities and targets must have the same shape")

        predicted = probabilities >= self.threshold
        expected = targets >= 0.5

        self.samples += int(expected.size)
        self.correct += int(np.equal(predicted, expected).sum())
        self.true_positive += int(np.logical_and(predicted, expected).sum())
        self.false_positive += int(np.logical_and(predicted, ~expected).sum())
        self.false_negative += int(np.logical_and(~predicted, expected).sum())

    def result(self) -> BinaryMetrics:
        def ratio(numerator: int, denominator: int) -> float:
            return numerator / denominator if denominator else 0.0

        precision = ratio(
            self.true_positive,
            self.true_positive + self.false_positive,
        )
        recall = ratio(
            self.true_positive,
            self.true_positive + self.false_negative,
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return BinaryMetrics(
            samples=self.samples,
            accuracy=ratio(self.correct, self.samples),
            precision=precision,
            recall=recall,
            f1=f1,
        )
