import pytest

from karting_agent.runtime.future_action_projection import (
    project_switch_probabilities,
)


def test_release_projection_matches_press_probability() -> None:
    assert project_switch_probabilities(
        (0.1, 0.6, 0.9), current_pressed=False
    ) == pytest.approx((0.1, 0.6, 0.9))


def test_press_projection_is_complement() -> None:
    assert project_switch_probabilities(
        (0.1, 0.6, 0.9), current_pressed=True
    ) == pytest.approx((0.9, 0.4, 0.1))


def test_release_and_press_projections_are_exact_complements() -> None:
    release = project_switch_probabilities((0.2, 0.5, 0.8), current_pressed=False)
    press = project_switch_probabilities((0.2, 0.5, 0.8), current_pressed=True)
    assert tuple(a + b for a, b in zip(release, press, strict=True)) == pytest.approx(
        (1.0, 1.0, 1.0)
    )


@pytest.mark.parametrize("values", [(), (-0.1,), (1.1,)])
def test_projection_rejects_invalid_probabilities(values) -> None:
    with pytest.raises(ValueError):
        project_switch_probabilities(values, current_pressed=False)
