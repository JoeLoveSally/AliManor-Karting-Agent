import numpy as np

from karting_agent.data_flow.execute.mock import MockExecutor
from karting_agent.data_flow.input.common import Frame
from karting_agent.runtime.sequential_state_conditioned_engine import (
    SequentialStateConditionedRuntimeConfig,
    SequentialStateConditionedRuntimeEngine,
)
from karting_agent.vision.preprocess import PreprocessConfig


class FakeSequentialModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = iter(probabilities)
        self.frame_indices: list[tuple[int, ...]] = []
        self.states: list[bool] = []

    def predict_switch(
        self,
        inputs: np.ndarray,
        current_pressed: bool,
        *,
        frame_indices=None,
    ) -> float:
        assert inputs.dtype == np.float32
        assert frame_indices is not None
        self.frame_indices.append(tuple(int(value) for value in frame_indices))
        self.states.append(bool(current_pressed))
        return next(self.probabilities)


def make_frame(index: int, timestamp_ms: float) -> Frame:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    return Frame(image=image, frame_index=index, timestamp_ms=timestamp_ms)


def test_v4a_runtime_passes_selected_frame_ids_and_feedback_state() -> None:
    model = FakeSequentialModel([0.9, 0.1])
    executor = MockExecutor()
    engine = SequentialStateConditionedRuntimeEngine(
        model=model,
        preprocess_config=PreprocessConfig(input_width=8, input_height=8),
        executor=executor,
        config=SequentialStateConditionedRuntimeConfig(
            target_fps=10.0,
            frame_offsets_ms=(0.0,),
            prediction_horizon_ms=100.0,
            switch_threshold=0.6,
        ),
    )

    first = engine.ingest(make_frame(3, 0.0))
    second = engine.ingest(make_frame(4, 100.0))

    assert first is not None and first.pressed is True
    assert second is not None and second.pressed is True
    assert model.frame_indices == [(3,), (4,)]
    assert model.states == [False, True]
    assert executor.calls == [True]
