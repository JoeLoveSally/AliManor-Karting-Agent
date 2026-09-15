import numpy as np

from karting_agent.control.controller import ControlAction
from karting_agent.data_flow.execute.mock import MockExecutor
from karting_agent.data_flow.input.common import Frame
from karting_agent.runtime.state_conditioned_engine import (
    StateConditionedRuntimeConfig,
    StateConditionedRuntimeEngine,
)
from karting_agent.vision.preprocess import PreprocessConfig


class FakeSwitchModel:
    def __init__(self) -> None:
        self.states: list[bool] = []

    def predict_switch(self, inputs: np.ndarray, current_pressed: bool) -> float:
        self.states.append(current_pressed)
        # Flip RELEASE -> PRESS on the first decision, then keep PRESS.
        return 0.9 if not current_pressed else 0.1


def make_frame(index: int, timestamp_ms: float) -> Frame:
    return Frame(
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        frame_index=index,
        timestamp_ms=timestamp_ms,
    )


def test_v3_runtime_feeds_back_its_own_physical_state() -> None:
    model = FakeSwitchModel()
    executor = MockExecutor()
    engine = StateConditionedRuntimeEngine(
        model=model,
        preprocess_config=PreprocessConfig(input_width=8, input_height=8),
        executor=executor,
        config=StateConditionedRuntimeConfig(
            target_fps=10,
            frame_offsets_ms=(0.0,),
            prediction_horizon_ms=100,
            switch_threshold=0.6,
        ),
    )

    first = engine.ingest(make_frame(0, 0.0))
    second = engine.ingest(make_frame(1, 100.0))

    assert first is not None
    assert first.action is ControlAction.PRESS
    assert first.pressed is True
    assert second is not None
    assert second.action is ControlAction.HOLD
    assert second.pressed is True
    assert model.states == [False, True]
    assert executor.calls == [True]

    assert engine.shutdown() is True
    assert executor.calls == [True, False]
