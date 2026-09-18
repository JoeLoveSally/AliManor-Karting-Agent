from __future__ import annotations

import pytest

from karting_agent.runtime.consensus_transition_lifecycle import (
    ConsensusTransitionLifecycle,
)


def test_lifecycle_waits_for_previous_state_h0_confirmation() -> None:
    guard = ConsensusTransitionLifecycle(threshold=0.6, min_state_hold_ms=100.0)
    pending = guard.begin(
        executed_at_ms=1000.0,
        previous_pressed=False,
        current_pressed=True,
        evidence_lead_ms=180.0,
    )

    assert pending.confirmation_deadline_ms == pytest.approx(1180.0)
    assert (
        guard.observe(
            timestamp_ms=1100.0,
            previous_state_h0_probability=0.4,
        )
        == "waiting"
    )
    assert guard.active
    assert (
        guard.observe(
            timestamp_ms=1140.0,
            previous_state_h0_probability=0.7,
        )
        == "confirmed"
    )
    assert not guard.active


def test_lifecycle_uses_min_hold_as_confirmation_floor() -> None:
    guard = ConsensusTransitionLifecycle(threshold=0.6, min_state_hold_ms=100.0)
    pending = guard.begin(
        executed_at_ms=1000.0,
        previous_pressed=True,
        current_pressed=False,
        evidence_lead_ms=20.0,
    )

    assert pending.confirmation_deadline_ms == pytest.approx(1100.0)


def test_lifecycle_expires_without_confirmation() -> None:
    guard = ConsensusTransitionLifecycle(threshold=0.6, min_state_hold_ms=100.0)
    guard.begin(
        executed_at_ms=1000.0,
        previous_pressed=False,
        current_pressed=True,
        evidence_lead_ms=150.0,
    )

    assert (
        guard.observe(
            timestamp_ms=1150.0,
            previous_state_h0_probability=0.2,
        )
        == "expired"
    )
    assert not guard.active


def test_lifecycle_rejects_non_transition() -> None:
    guard = ConsensusTransitionLifecycle(threshold=0.6, min_state_hold_ms=100.0)
    with pytest.raises(ValueError, match="must change state"):
        guard.begin(
            executed_at_ms=1000.0,
            previous_pressed=True,
            current_pressed=True,
            evidence_lead_ms=150.0,
        )
