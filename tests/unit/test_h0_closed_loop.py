from __future__ import annotations

import pytest

from karting_agent.train.h0_closed_loop import simulate_h0_closed_loop


def test_h0_replay_uses_probability_for_simulated_state() -> None:
    decisions = simulate_h0_closed_loop(
        [0.0, 33.0, 66.0],
        switch_if_release=[0.9, 0.1, 0.9],
        switch_if_press=[0.2, 0.8, 0.1],
        initial_pressed=False,
        threshold=0.6,
    )

    assert [decision.state_before for decision in decisions] == [False, True, False]
    assert [decision.probability for decision in decisions] == pytest.approx([0.9, 0.8, 0.9])
    assert [decision.switched for decision in decisions] == [True, True, True]
    assert [decision.state_after for decision in decisions] == [True, False, True]


def test_h0_replay_min_hold_blocks_early_reversal() -> None:
    decisions = simulate_h0_closed_loop(
        [0.0, 50.0, 100.0, 150.0],
        switch_if_release=[0.9, 0.1, 0.1, 0.9],
        switch_if_press=[0.1, 0.9, 0.9, 0.9],
        initial_pressed=False,
        threshold=0.6,
        min_state_hold_ms=100.0,
    )

    assert decisions[0].switched
    assert decisions[1].blocked_by_min_hold
    assert not decisions[1].switched
    assert decisions[2].switched
    assert not decisions[2].blocked_by_min_hold
    assert decisions[3].blocked_by_min_hold


def test_h0_replay_rejects_decreasing_timestamps() -> None:
    with pytest.raises(ValueError, match="nondecreasing"):
        simulate_h0_closed_loop(
            [10.0, 9.0],
            [0.1, 0.1],
            [0.1, 0.1],
            initial_pressed=False,
        )


def test_h0_replay_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        simulate_h0_closed_loop(
            [0.0],
            [1.1],
            [0.1],
            initial_pressed=False,
        )
