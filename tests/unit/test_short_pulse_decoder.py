from __future__ import annotations

import pytest

from karting_agent.runtime.short_pulse_decoder import decode_short_pulse


HORIZONS = (100.0, 200.0, 300.0)


def decode(probabilities: tuple[float, float, float], *, mode: str):
    return decode_short_pulse(
        horizons_ms=HORIZONS,
        probabilities=probabilities,
        short_horizon_ms=100.0,
        control_horizon_ms=200.0,
        anticipation_horizon_ms=300.0,
        threshold=0.6,
        min_pulse_width_ms=100.0,
        mode=mode,
    )


def test_strict_min_hold_stretches_predicted_sub_100ms_pulse() -> None:
    candidate = decode(
        (0.9712004661560059, 0.22295613586902618, 0.07440562546253204),
        mode="strict_min_hold",
    )

    assert candidate is not None
    assert candidate.start_horizon_ms == pytest.approx(61.7792, abs=1e-4)
    assert candidate.end_horizon_ms == pytest.approx(149.6095, abs=1e-4)
    assert candidate.raw_width_ms == pytest.approx(87.8303, abs=1e-4)
    assert candidate.effective_width_ms == pytest.approx(100.0)
    assert candidate.effective_end_horizon_ms == pytest.approx(161.7792, abs=1e-4)


def test_reject_too_short_drops_predicted_sub_100ms_pulse() -> None:
    candidate = decode(
        (0.9712004661560059, 0.22295613586902618, 0.07440562546253204),
        mode="reject_too_short",
    )

    assert candidate is None


def test_reject_too_short_keeps_wide_pulse() -> None:
    candidate = decode((0.70, 0.59, 0.10), mode="reject_too_short")

    assert candidate is not None
    assert candidate.start_horizon_ms == pytest.approx(85.7142857)
    assert candidate.end_horizon_ms == pytest.approx(190.9090909)
    assert candidate.raw_width_ms == pytest.approx(105.1948052)
    assert candidate.effective_width_ms == pytest.approx(candidate.raw_width_ms)


@pytest.mark.parametrize(
    "probabilities",
    [
        (0.59, 0.20, 0.10),
        (0.80, 0.60, 0.10),
        (0.80, 0.20, 0.60),
    ],
)
def test_non_pulse_shapes_are_rejected(
    probabilities: tuple[float, float, float],
) -> None:
    assert decode(probabilities, mode="strict_min_hold") is None


def test_decoder_validates_probability_width() -> None:
    with pytest.raises(ValueError, match="probability width"):
        decode_short_pulse(
            horizons_ms=HORIZONS,
            probabilities=(0.8, 0.2),
            short_horizon_ms=100.0,
            control_horizon_ms=200.0,
            anticipation_horizon_ms=300.0,
        )
