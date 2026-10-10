"""Read-only Tiny CNN+GRU inference on already-decoded Android frames.

The PRESS state is virtual: there is NO Android executor or human action GT.
Video is real, but virtual decisions do not alter future pixels.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from karting_agent.data_flow.input.common import Frame
from karting_agent.train.shadow_control import ShadowController

OFFSETS_MS = (-200.0, -150.0, -100.0, -50.0, 0.0)
MODES = ("rgb", "rgb_hsv")


@dataclass(frozen=True)
class LiveShadowConfig:
    target_fps: float = 30.0
    horizon_ms: float = 100.0
    threshold: float = 0.5
    max_history_lag_ms: float = 80.0
    max_decode_lag_ms: float = 150.0

    def validate(self) -> None:
        if (not 0 < self.target_fps <= 120 or
                self.horizon_ms != 100.0 or
                self.threshold != 0.5 or
                not math.isfinite(self.max_history_lag_ms) or
                self.max_history_lag_ms <= 0 or
                not math.isfinite(self.max_decode_lag_ms) or
                self.max_decode_lag_ms <= 0):
            raise ValueError("unsupported live shadow timing/threshold configuration")


class TinyLiveShadow:
    """Process latest frames at 30Hz, never requesting Android controls.

    builder(BGR): uint8 [4,96,96] from frozen training preprocessing.
    clock_ms(): host monotonic elapsed milliseconds since AdbVideoInput origin.
    With no clock (tests/file replay), observation timestamps drive scheduling.
    """

    def __init__(
        self, models: Mapping[str, torch.nn.Module],
        builder: Callable[[np.ndarray], np.ndarray], *,
        config: LiveShadowConfig = LiveShadowConfig(),
        clock_ms: Callable[[], float] | None = None,
        initial_pressed: bool = False,
    ) -> None:
        config.validate()
        if set(models) != set(MODES):
            raise ValueError("exactly rgb and rgb_hsv frozen models required")
        self.models = dict(models)
        for model in self.models.values():
            model.eval()
        self.builder = builder
        self.config = config
        self.clock_ms = clock_ms
        self.controllers = {
            mode: ShadowController(initial_pressed=initial_pressed,
                                   horizon_ms=config.horizon_ms,
                                   threshold=config.threshold)
            for mode in MODES
        }
        self.frames: deque[Frame] = deque()
        self.features: dict[int, np.ndarray] = {}
        self.next_due_ms: float | None = None
        self.last_time_ms = -1.0
        self.last_frame_index = -1
        self.processed_frames = 0
        self.skipped_stale = 0
        self.skipped_history = 0
        self.decisions = 0

    def _selected(self, t: float) -> tuple[Frame, ...] | None:
        chosen: list[Frame] = []
        frames = tuple(self.frames)
        for offset in OFFSETS_MS:
            target = t + offset
            selected = next((f for f in reversed(frames)
                             if f.timestamp_ms <= target + 1e-6), None)
            if selected is None or target - selected.timestamp_ms > self.config.max_history_lag_ms:
                return None
            chosen.append(selected)
        return tuple(chosen)

    def ingest(self, frame: Frame) -> dict | None:
        t = float(frame.timestamp_ms)
        if t <= self.last_time_ms or frame.frame_index <= self.last_frame_index:
            raise ValueError("nonmonotonic live frame index or decode timestamp")
        self.last_time_ms = t
        self.last_frame_index = frame.frame_index
        self.processed_frames += 1
        self.frames.append(frame)
        cutoff = t - 200.0 - self.config.max_history_lag_ms - 50.0
        while len(self.frames) > 1 and self.frames[1].timestamp_ms < cutoff:
            old = self.frames.popleft()
            self.features.pop(old.frame_index, None)
        if self.next_due_ms is None:
            self.next_due_ms = t + 200.0
        if t + 1e-6 < self.next_due_ms:
            return None
        # No burst catch-up, even if the previous inference was slow.
        while self.next_due_ms <= t + 1e-6:
            self.next_due_ms += 1000.0 / self.config.target_fps
        now = float(self.clock_ms()) if self.clock_ms is not None else t
        if not math.isfinite(now) or now + 1e-6 < t:
            raise ValueError("clock must use capture monotonic timestamp base")
        decode_lag_ms = max(0.0, now - t)
        if decode_lag_ms > self.config.max_decode_lag_ms:
            self.skipped_stale += 1
            return {"kind": "stale", "frame_index": frame.frame_index,
                    "observation_ms": t, "decode_to_read_ms": decode_lag_ms}
        selected = self._selected(t)
        if selected is None:
            self.skipped_history += 1
            return {"kind": "missing_history", "frame_index": frame.frame_index,
                    "observation_ms": t}
        processing_started = time.perf_counter()
        for chosen in selected:
            if chosen.frame_index not in self.features:
                img = self.builder(chosen.image)
                if img.dtype != np.uint8 or img.shape != (4, 96, 96):
                    raise ValueError("training feature builder must return uint8 [4,96,96]")
                self.features[chosen.frame_index] = img
        preprocessing_ms = (time.perf_counter() - processing_started) * 1000.0
        timestamps = tuple(x.timestamp_ms for x in selected)
        images = np.stack([self.features[x.frame_index] for x in selected])
        results = {}
        with torch.inference_mode():
            for mode in MODES:
                controller = self.controllers[mode]
                controller.advance_to(t)
                feedback = controller.control_features(timestamps)
                channels = 3 if mode == "rgb" else 4
                imgs = torch.from_numpy(
                    np.ascontiguousarray(images[:, :channels], dtype=np.float32) / 255.0
                ).unsqueeze(0)
                ctl = torch.from_numpy(feedback).unsqueeze(0)
                started = time.perf_counter()
                logit = self.models[mode](imgs, ctl)
                probs = torch.sigmoid(logit).detach().cpu().reshape(-1)
                if probs.numel() != 1:
                    raise ValueError("model must return one PRESS logit")
                probability = float(probs[0])
                inference_ms = (time.perf_counter() - started) * 1000.0
                if not math.isfinite(probability):
                    raise ValueError("invalid frozen model output")
                ready_ms = (float(self.clock_ms()) if self.clock_ms is not None
                            else t + inference_ms)
                ready_ms = max(ready_ms, t)
                proposal = controller.propose(
                    observe_ms=t, target_ms=t + 100.0,
                    probability=probability, ready_ms=ready_ms)
                results[mode] = {
                    "probability": probability,
                    "virtual_pressed_at_observation": bool(feedback[-1, 0]),
                    "proposed_pressed": proposal.pressed,
                    "target_ms": proposal.target_ms,
                    "ready_ms": proposal.ready_ms,
                    "scheduled_ms": proposal.execute_ms,
                    "late": proposal.execute_ms > proposal.target_ms + 1e-6,
                    "inference_ms": inference_ms,
                }
        total_processing_ms = (time.perf_counter() - processing_started) * 1000.0
        complete_decode_lag_ms = (max(0.0, float(self.clock_ms()) - t)
                                  if self.clock_ms is not None else None)
        self.decisions += 1
        return {
            "kind": "prediction", "frame_index": frame.frame_index,
            "observation_ms": t, "decoded_to_read_ms": decode_lag_ms,
            "decoded_to_completion_ms": complete_decode_lag_ms,
            "feature_preprocess_ms": preprocessing_ms,
            "total_dual_model_processing_ms": total_processing_ms,
            "input_indices": [x.frame_index for x in selected],
            "input_timestamp_ms": list(timestamps),
            "models": results,
        }

    def summary(self) -> dict:
        return {
            "frames_received": self.processed_frames,
            "inference_ticks": self.decisions,
            "skipped_stale": self.skipped_stale,
            "skipped_missing_history": self.skipped_history,
            "virtual_events": {
                mode: [{"ms": e.time_ms, "pressed": e.pressed}
                       for e in ctrl.events]
                for mode, ctrl in self.controllers.items()
            },
            "virtual_action_note": "No Android input; no human action ground truth",
        }
