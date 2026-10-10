"""CPU-only live Tiny Shadow tests using synthetic decoder frames; no ADB."""
import numpy as np
import pytest
import torch

from karting_agent.data_flow.input.common import Frame
from karting_agent.runtime.tiny_live_shadow import TinyLiveShadow, LiveShadowConfig
from karting_agent.train.shadow_control import ShadowController


def frame(i, t):
    return Frame(np.full((80, 36, 3), i % 255, dtype=np.uint8), i, float(t))


class Spy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.controls = []
        self.imgs = []

    def forward(self, x, c):
        self.imgs.append(x.clone())
        self.controls.append(c.clone())
        return (1 - 2 * c[:, -1, 0]) * 8


def builder(image):
    return np.full((4, 96, 96), image[0, 0, 0], dtype=np.uint8)


def shadow(clock=None):
    models = {mode: Spy() for mode in ("rgb", "rgb_hsv")}
    return TinyLiveShadow(models, builder, clock_ms=clock), models


def test_five_timestamp_history_and_same_rgb_pixels_in_both_modes():
    runtime, models = shadow()
    result = None
    for i in range(40):
        result = runtime.ingest(frame(i, i * 20)) or result
        if result is not None and result["kind"] == "prediction":
            break
    assert result["observation_ms"] == 200
    assert result["input_timestamp_ms"] == pytest.approx([0, 40, 100, 140, 200])
    assert models["rgb"].imgs[0].shape == (1, 5, 3, 96, 96)
    assert models["rgb_hsv"].imgs[0].shape == (1, 5, 4, 96, 96)
    assert torch.equal(models["rgb"].imgs[0], models["rgb_hsv"].imgs[0][:, :, :3])
    assert models["rgb"].controls[0].shape == (1, 5, 3)
    assert result["models"]["rgb"]["target_ms"] == pytest.approx(300)


def test_virtual_pending_action_not_visible_before_target_time():
    runtime, _ = shadow()
    rows = []
    for i in range(35):
        record = runtime.ingest(frame(i, i * 20))
        if record and record["kind"] == "prediction":
            rows.append(record)
    assert len(rows) >= 6
    assert rows[0]["models"]["rgb"]["proposed_pressed"]
    assert not rows[1]["models"]["rgb"]["virtual_pressed_at_observation"]
    assert not rows[2]["models"]["rgb"]["virtual_pressed_at_observation"]
    assert any(row["models"]["rgb"]["virtual_pressed_at_observation"]
               for row in rows[3:])
    assert runtime.controllers["rgb"].events[1].time_ms == pytest.approx(300)


def test_actual_inference_ready_time_drives_virtual_schedule():
    clock = [0.]
    runtime, _ = shadow(clock=lambda: clock[0])
    for i, t in enumerate(range(0, 210, 20)):
        clock[0] = float(t)
        runtime.ingest(frame(i, t))
    clock[0] = 220.
    assert runtime.ingest(frame(11, 220)) is None
    clock[0] = 350.
    record = runtime.ingest(frame(12, 240))
    assert record["kind"] == "prediction"
    assert record["models"]["rgb"]["late"]
    assert record["models"]["rgb"]["scheduled_ms"] == pytest.approx(350)
    assert runtime.decisions == 2


def test_stale_video_is_discarded_without_inference():
    clock = [0.]
    runtime, _ = shadow(clock=lambda: clock[0])
    for i, t in enumerate(range(0, 200, 20)):
        clock[0] = float(t)
        assert runtime.ingest(frame(i, t)) is None
    clock[0] = 400.
    record = runtime.ingest(frame(10, 200))
    assert record["kind"] == "stale"
    assert runtime.skipped_stale == 1


def test_missing_historical_frame_returns_diagnostic():
    runtime, _ = shadow()
    runtime.ingest(frame(0, 0))
    record = runtime.ingest(frame(1, 400))
    assert record["kind"] == "missing_history"
    assert runtime.skipped_history == 1


def test_monotonicity_and_frozen_config_validation():
    runtime, _ = shadow()
    runtime.ingest(frame(0, 0))
    with pytest.raises(ValueError, match="nonmonotonic"):
        runtime.ingest(frame(0, 20))
    with pytest.raises(ValueError, match="nonmonotonic"):
        runtime.ingest(frame(1, 0))
    with pytest.raises(ValueError):
        TinyLiveShadow({"rgb": Spy()}, builder)
    with pytest.raises(ValueError):
        LiveShadowConfig(threshold=.51).validate()


def test_invalid_training_feature_shape_fails_before_any_proposal():
    models = {mode: Spy() for mode in ("rgb", "rgb_hsv")}
    runtime = TinyLiveShadow(models, lambda _: np.zeros((3, 96, 96), np.uint8))
    for i in range(10):
        runtime.ingest(frame(i, i * 20))
    with pytest.raises(ValueError, match="training feature"):
        runtime.ingest(frame(10, 200))
    assert all(len(x.events) == 1 for x in runtime.controllers.values())


def test_explicit_ready_timestamp_retains_offline_default_semantics():
    controller = ShadowController(initial_pressed=False)
    controller.advance_to(200)
    old = controller.propose(observe_ms=200, target_ms=300, probability=.8)
    assert old.ready_ms == 200 and old.execute_ms == 300
    controller.advance_to(250)
    new = controller.propose(observe_ms=250, target_ms=350,
                             probability=.1, ready_ms=380)
    assert new.execute_ms == 380
    with pytest.raises(ValueError):
        controller.propose(observe_ms=250, target_ms=350,
                           probability=.5, ready_ms=249)
