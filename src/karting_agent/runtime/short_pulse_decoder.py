"""Decode short two-transition pulse patterns from learned horizon probabilities.

This module is intentionally model-only and side-effect free. It does not alter
runtime control; it exists so pulse hypotheses can be validated offline before
any closed-loop integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence


PulseMode = Literal["strict_min_hold", "reject_too_short"]


@dataclass(frozen=True)
class ShortPulseCandidate:
    """A decoded away-and-back pulse relative to the supplied current state."""

    start_horizon_ms: float
    end_horizon_ms: float
    raw_width_ms: float
    effective_end_horizon_ms: float
    effective_width_ms: float
    short_probability: float
    control_probability: float
    anticipation_probability: float


def decode_short_pulse(
    *,
    horizons_ms: Sequence[float],
    probabilities: Sequence[float],
    short_horizon_ms: float,
    control_horizon_ms: float,
    anticipation_horizon_ms: float,
    threshold: float = 0.6,
    min_pulse_width_ms: float = 100.0,
    mode: PulseMode = "strict_min_hold",
) -> ShortPulseCandidate | None:
    """Decode ``high -> low -> low`` as a future short pulse.

    The switch heads predict whether ``action(t + h)`` differs from the supplied
    current action. Therefore ``Hshort >= T, Hcontrol < T, Hanticipation < T``
    can represent an away-and-back pulse rather than an overdue single switch.

    The rising crossing is linearly interpolated from an implicit probability 0
    at horizon 0 to ``Hshort``. The falling crossing is interpolated between the
    short and control horizons. The implicit zero is a decoder hypothesis, not a
    learned H0 output, which is why this decoder is kept offline for validation.
    """

    horizons = tuple(float(value) for value in horizons_ms)
    probs = tuple(float(value) for value in probabilities)
    if len(horizons) != len(probs):
        raise ValueError("probability width does not match prediction horizons")
    if not horizons:
        raise ValueError("prediction horizons must not be empty")
    if tuple(sorted(horizons)) != horizons or len(set(horizons)) != len(horizons):
        raise ValueError("prediction horizons must be sorted and unique")
    if any(not math.isfinite(value) or value < 0.0 for value in horizons):
        raise ValueError("prediction horizons must be finite and >= 0")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probs):
        raise ValueError("switch probabilities must be finite and in [0, 1]")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if not math.isfinite(min_pulse_width_ms) or min_pulse_width_ms < 0.0:
        raise ValueError("min_pulse_width_ms must be finite and >= 0")
    if mode not in ("strict_min_hold", "reject_too_short"):
        raise ValueError(f"unsupported pulse mode: {mode}")

    try:
        short_index = horizons.index(float(short_horizon_ms))
        control_index = horizons.index(float(control_horizon_ms))
        anticipation_index = horizons.index(float(anticipation_horizon_ms))
    except ValueError as exc:
        raise ValueError("decoder horizons must be present in prediction horizons") from exc

    short_horizon = horizons[short_index]
    control_horizon = horizons[control_index]
    anticipation_horizon = horizons[anticipation_index]
    if not 0.0 < short_horizon < control_horizon < anticipation_horizon:
        raise ValueError(
            "decoder horizons must satisfy 0 < short < control < anticipation"
        )

    short_probability = probs[short_index]
    control_probability = probs[control_index]
    anticipation_probability = probs[anticipation_index]
    if not (
        short_probability >= threshold
        and control_probability < threshold
        and anticipation_probability < threshold
    ):
        return None

    start_horizon = short_horizon * threshold / short_probability
    falling_denominator = short_probability - control_probability
    if falling_denominator <= 0.0:
        return None
    falling_fraction = (short_probability - threshold) / falling_denominator
    falling_fraction = min(1.0, max(0.0, falling_fraction))
    end_horizon = short_horizon + falling_fraction * (
        control_horizon - short_horizon
    )
    raw_width = end_horizon - start_horizon
    if raw_width < 0.0:
        return None

    if mode == "reject_too_short" and raw_width + 1e-6 < min_pulse_width_ms:
        return None

    effective_end = end_horizon
    if mode == "strict_min_hold":
        effective_end = max(effective_end, start_horizon + min_pulse_width_ms)
    effective_width = effective_end - start_horizon

    return ShortPulseCandidate(
        start_horizon_ms=start_horizon,
        end_horizon_ms=end_horizon,
        raw_width_ms=raw_width,
        effective_end_horizon_ms=effective_end,
        effective_width_ms=effective_width,
        short_probability=short_probability,
        control_probability=control_probability,
        anticipation_probability=anticipation_probability,
    )
