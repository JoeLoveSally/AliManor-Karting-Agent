"""Analysis helpers for comparing dense V5 switch horizons on one live run."""

from __future__ import annotations

import math
from collections.abc import Sequence


def first_threshold_crossing(
    timestamps_ms: Sequence[float],
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> tuple[int, float] | None:
    """Return the first index/timestamp whose probability reaches the threshold."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if len(timestamps_ms) != len(probabilities):
        raise ValueError("timestamps and probabilities must have equal length")

    last_timestamp: float | None = None
    for index, (raw_timestamp, raw_probability) in enumerate(
        zip(timestamps_ms, probabilities, strict=True)
    ):
        timestamp_ms = float(raw_timestamp)
        probability = float(raw_probability)
        if not math.isfinite(timestamp_ms):
            raise ValueError("timestamps must be finite")
        if last_timestamp is not None and timestamp_ms < last_timestamp - 1e-6:
            raise ValueError("timestamps must be nondecreasing")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probabilities must be finite and in [0, 1]")
        if probability >= threshold:
            return index, timestamp_ms
        last_timestamp = timestamp_ms
    return None


def crossing_lead_ms(
    reference_timestamp_ms: float,
    candidate_timestamp_ms: float,
) -> float:
    """Positive values mean the candidate crossed earlier than the reference."""

    reference = float(reference_timestamp_ms)
    candidate = float(candidate_timestamp_ms)
    if not math.isfinite(reference) or not math.isfinite(candidate):
        raise ValueError("crossing timestamps must be finite")
    return reference - candidate
