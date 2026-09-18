from __future__ import annotations

import pytest

from karting_agent.train.live_horizon_analysis import (
    crossing_lead_ms,
    first_threshold_crossing,
)


def test_first_threshold_crossing_returns_first_hit() -> None:
    crossing = first_threshold_crossing(
        [0.0, 33.0, 66.0, 99.0],
        [0.1, 0.4, 0.7, 0.8],
        threshold=0.6,
    )

    assert crossing == pytest.approx((2, 66.0))


def test_first_threshold_crossing_returns_none_without_hit() -> None:
    assert (
        first_threshold_crossing(
            [0.0, 33.0],
            [0.1, 0.5],
            threshold=0.6,
        )
        is None
    )


def test_crossing_lead_is_positive_for_earlier_candidate() -> None:
    assert crossing_lead_ms(200.0, 125.0) == pytest.approx(75.0)


def test_first_threshold_crossing_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        first_threshold_crossing(
            [0.0],
            [1.1],
            threshold=0.6,
        )
