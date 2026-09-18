"""Datasets for V4-C4 current-action plus first-transition-time training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from karting_agent.train.dataset import DatasetSample
from karting_agent.train.kart_relative_labels import (
    KartRelativePseudoLabel,
    KartRelativeSupervisedVideoDataset,
)
from karting_agent.vision.preprocess import PreprocessConfig


def load_transition_times_by_video(
    labels_dir: Path,
    videos: Sequence[str],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for video in sorted(set(videos)):
        path = Path(labels_dir) / f"{Path(video).stem}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        events = payload.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError(f"label file has no events: {path}")
        result[video] = np.asarray(
            [float(event["timestamp_ms"]) for event in events[1:]],
            dtype=np.float64,
        )
    return result


def event_time_class(
    transition_times_ms: np.ndarray,
    observation_ms: float,
    *,
    max_horizon_ms: float,
    bin_ms: float,
) -> tuple[int, float | None]:
    """Return 0-based event bin, with the last class reserved for no event."""

    if max_horizon_ms <= 0 or bin_ms <= 0:
        raise ValueError("event-time horizons must be positive")
    bin_count_float = max_horizon_ms / bin_ms
    bin_count = int(round(bin_count_float))
    if abs(bin_count_float - bin_count) > 1e-6:
        raise ValueError("max_horizon_ms must be divisible by bin_ms")

    index = int(np.searchsorted(transition_times_ms, observation_ms, side="right"))
    no_event_class = bin_count
    if index >= transition_times_ms.size:
        return no_event_class, None

    delay_ms = float(transition_times_ms[index] - observation_ms)
    if delay_ms <= 1e-6:
        delay_ms = 1e-6
    if delay_ms > max_horizon_ms:
        return no_event_class, delay_ms

    class_index = min(
        bin_count - 1,
        max(0, int(np.ceil(delay_ms / bin_ms)) - 1),
    )
    return class_index, delay_ms


class EventTimeKartRelativeVideoDataset:
    """Attach current expert action and next-transition-time targets."""

    def __init__(
        self,
        samples: Sequence[DatasetSample],
        *,
        relation_labels: dict[tuple[str, int], KartRelativePseudoLabel],
        labels_dir: Path,
        project_root: Path,
        max_horizon_ms: float = 300.0,
        bin_ms: float = 50.0,
        preprocess_config: PreprocessConfig = PreprocessConfig(),
        cache_root: Path | None = None,
        require_cache: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.max_horizon_ms = float(max_horizon_ms)
        self.bin_ms = float(bin_ms)
        self.bin_count = int(round(self.max_horizon_ms / self.bin_ms))
        if self.bin_count < 1:
            raise ValueError("event-time bin count must be >= 1")
        if abs(self.bin_count * self.bin_ms - self.max_horizon_ms) > 1e-6:
            raise ValueError("max_horizon_ms must be divisible by bin_ms")
        if any(sample.current_pressed is None for sample in self.samples):
            raise ValueError("event-time samples must contain current_pressed")

        self.transition_times_by_video = load_transition_times_by_video(
            Path(labels_dir),
            [sample.video for sample in self.samples],
        )
        self.base = KartRelativeSupervisedVideoDataset(
            self.samples,
            relation_labels=relation_labels,
            project_root=project_root,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=require_cache,
            counterfactual_states=False,
        )

    @property
    def event_time_classes(self) -> int:
        return self.bin_count + 1

    @property
    def no_event_class(self) -> int:
        return self.bin_count

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = dict(self.base[index])
        sample = self.samples[index]
        recorded = sample.current_pressed
        assert recorded is not None
        observation_ms = float(sample.input_timestamps_ms[-1])
        target_class, delay_ms = event_time_class(
            self.transition_times_by_video[sample.video],
            observation_ms,
            max_horizon_ms=self.max_horizon_ms,
            bin_ms=self.bin_ms,
        )
        item["current_action_target"] = np.float32(1.0 if recorded else 0.0)
        item["event_time_target"] = np.int64(target_class)
        item["event_time_delay_ms"] = np.float32(
            -1.0 if delay_ms is None else delay_ms
        )
        return item

    def close(self) -> None:
        self.base.close()

    def __getstate__(self) -> dict[str, object]:
        return self.__dict__.copy()

    def __del__(self) -> None:
        self.close()
