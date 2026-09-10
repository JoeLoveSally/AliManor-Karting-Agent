import pytest

from karting_agent.control.controller import (
    ControlAction,
    HysteresisConfig,
    HysteresisController,
    hysteresis_states,
)


def test_hysteresis_controller_holds_inside_deadband() -> None:
    controller = HysteresisController(
        HysteresisConfig(press_threshold=0.7, release_threshold=0.3)
    )

    assert controller.update(0.6) is ControlAction.HOLD
    assert controller.pressed is False

    assert controller.update(0.8) is ControlAction.PRESS
    assert controller.pressed is True

    assert controller.update(0.5) is ControlAction.HOLD
    assert controller.pressed is True

    assert controller.update(0.2) is ControlAction.RELEASE
    assert controller.pressed is False


def test_hysteresis_states_infers_initial_state_without_startup_transition() -> None:
    states = hysteresis_states(
        [0.9, 0.6, 0.4, 0.2, 0.5, 0.8],
        HysteresisConfig(press_threshold=0.7, release_threshold=0.3),
    )
    assert states == (True, True, True, False, False, True)


def test_hysteresis_config_rejects_overlapping_thresholds() -> None:
    with pytest.raises(ValueError):
        HysteresisConfig(press_threshold=0.3, release_threshold=0.3).validate()
