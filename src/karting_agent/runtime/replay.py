"""Fast-as-possible video replay through the runtime engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from karting_agent.control.controller import ControlAction
from karting_agent.data_flow.input.video import VideoInput
from karting_agent.runtime.engine import RuntimeEngine, RuntimeStep


@dataclass(frozen=True)
class ReplayResult:
    video: str
    source_fps: float
    source_frames_read: int
    target_fps: float
    elapsed_seconds: float
    safety_release: bool
    steps: tuple[RuntimeStep, ...]

    def summary(self) -> dict[str, object]:
        latencies = np.asarray(
            [step.inference_ms for step in self.steps], dtype=np.float64
        )
        press_count = sum(
            step.action is ControlAction.PRESS for step in self.steps
        )
        release_count = sum(
            step.action is ControlAction.RELEASE for step in self.steps
        )
        return {
            "steps": len(self.steps),
            "press_actions": press_count,
            "release_actions": release_count,
            "state_changes": press_count + release_count,
            "mean_inference_ms": (
                float(latencies.mean()) if latencies.size else 0.0
            ),
            "p50_inference_ms": (
                float(np.percentile(latencies, 50)) if latencies.size else 0.0
            ),
            "p95_inference_ms": (
                float(np.percentile(latencies, 95)) if latencies.size else 0.0
            ),
            "max_inference_ms": (
                float(latencies.max()) if latencies.size else 0.0
            ),
            "processing_fps": (
                len(self.steps) / self.elapsed_seconds
                if self.elapsed_seconds > 0
                else 0.0
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "video": self.video,
            "source_fps": self.source_fps,
            "source_frames_read": self.source_frames_read,
            "target_fps": self.target_fps,
            "elapsed_seconds": self.elapsed_seconds,
            "safety_release": self.safety_release,
            "summary": self.summary(),
            "steps": [
                {
                    **asdict(step),
                    "action": step.action.value,
                }
                for step in self.steps
            ],
        }


class ReplayRunner:
    """Feed sequential video frames into the same causal runtime engine."""

    def __init__(self, engine: RuntimeEngine) -> None:
        self.engine = engine

    def run(
        self,
        video_input: VideoInput,
        *,
        max_seconds: float | None = None,
    ) -> ReplayResult:
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")

        max_timestamp_ms = (
            None if max_seconds is None else max_seconds * 1000.0
        )
        steps: list[RuntimeStep] = []
        frames_read = 0
        started = time.perf_counter()
        safety_release = False
        try:
            while True:
                frame = video_input.read()
                if frame is None:
                    break
                if (
                    max_timestamp_ms is not None
                    and frame.timestamp_ms > max_timestamp_ms
                ):
                    break
                frames_read += 1
                step = self.engine.ingest(frame)
                if step is not None:
                    steps.append(step)
        finally:
            safety_release = self.engine.shutdown()

        elapsed_seconds = time.perf_counter() - started
        return ReplayResult(
            video=str(video_input.path),
            source_fps=video_input.fps,
            source_frames_read=frames_read,
            target_fps=self.engine.config.target_fps,
            elapsed_seconds=elapsed_seconds,
            safety_release=safety_release,
            steps=tuple(steps),
        )
