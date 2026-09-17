"""Project absolute future-action probabilities into KEEP/SWITCH probabilities."""

from __future__ import annotations

from collections.abc import Sequence


def project_switch_probabilities(
    future_action_probabilities: Sequence[float],
    *,
    current_pressed: bool,
) -> tuple[float, ...]:
    """Return P(future action != current physical state) for each horizon.

    The v4-C2 ``future_action_head`` predicts absolute PRESS probability from
    visual features only. For a currently RELEASED controller state, switch
    probability is therefore exactly P(PRESS). For a currently PRESSED state,
    switch probability is P(RELEASE) = 1 - P(PRESS).

    This projection enforces exact RELEASE/PRESS complement symmetry and avoids
    asking a separately learned state-conditioned switch head to relearn the same
    Boolean relationship.
    """

    values = tuple(float(value) for value in future_action_probabilities)
    if not values:
        raise ValueError("future_action_probabilities must not be empty")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("future-action probabilities must be in [0, 1]")
    if current_pressed:
        return tuple(1.0 - value for value in values)
    return values
