from __future__ import annotations

import numpy as np
import pytest

from karting_agent.runtime.h0_coherence import (
    desired_press_from_switch_pair,
    project_desired_press_to_switch,
)


def test_switch_pair_projection_is_exactly_coherent() -> None:
    desired = desired_press_from_switch_pair(
        np.asarray([0.8, 0.2]),
        np.asarray([0.3, 0.9]),
    )
    release_switch, press_switch = project_desired_press_to_switch(desired)

    assert desired.tolist() == pytest.approx([0.75, 0.15])
    assert release_switch.tolist() == pytest.approx([0.75, 0.15])
    assert press_switch.tolist() == pytest.approx([0.25, 0.85])
    assert (release_switch + press_switch).tolist() == pytest.approx([1.0, 1.0])


def test_switch_pair_projection_averages_contradictory_evidence() -> None:
    desired = desired_press_from_switch_pair(
        np.asarray([0.87]),
        np.asarray([0.63]),
    )

    assert desired[0] == pytest.approx(0.62)


def test_switch_pair_projection_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        desired_press_from_switch_pair(
            np.asarray([0.2]),
            np.asarray([0.2, 0.3]),
        )


def test_desired_press_projection_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        project_desired_press_to_switch(np.asarray([1.1]))
