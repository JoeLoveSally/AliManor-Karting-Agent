"""Factual (non-counterfactual) action history from existing video label events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from karting_agent.model.action_history import encode_action_history
from karting_agent.train.dataset import DatasetSample
from karting_agent.train.state_conditioned_dataset import StateConditionedVideoDataset
from karting_agent.vision.preprocess import PreprocessConfig


class ActionHistoryVideoDataset:
    """Augment original 5-frame RGB samples with actual touch-marker history.

    The video-level label JSON comes from the SAME cleaned touch-marker timeline
    used to generate samples.jsonl; no inferred agent outcomes are used as labels.
    """

    def __init__(
        self,
        samples: Sequence[DatasetSample],
        *,
        project_root: Path,
        labels_dir: Path,
        preprocess_config: PreprocessConfig = PreprocessConfig(),
        cache_root: Path | None = None,
        require_cache: bool = False,
        max_age_ms: float = 500.0,
    ) -> None:
        self.samples = list(samples)
        self.project_root = Path(project_root).resolve()
        self.max_age_ms = float(max_age_ms)
        self.events: dict[str, tuple[bool, tuple[tuple[float, bool], ...]]] = {}
        for sample in self.samples:
            if sample.current_pressed is None:
                raise ValueError("every factual sample needs current_pressed")
            if sample.video in self.events:
                continue
            label_path = Path(labels_dir) / (Path(sample.video).stem + ".json")
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            labeled_video = Path(str(payload["video"]))
            requested_video = Path(sample.video)
            if not labeled_video.is_absolute():
                labeled_video = self.project_root / labeled_video
            if not requested_video.is_absolute():
                requested_video = self.project_root / requested_video
            if labeled_video.resolve() != requested_video.resolve():
                raise ValueError(f"label video mismatch: {label_path}")
            raw_events = payload.get("events", [])
            if not raw_events:
                raise ValueError(f"no action events in {label_path}")
            events = [
                (float(event["timestamp_ms"]), bool(event["pressed"]))
                for event in raw_events
            ]
            if abs(events[0][0]) > 1e-6:
                raise ValueError(f"first label event must be at t=0: {label_path}")
            self.events[sample.video] = (events[0][1], tuple(events[1:]))

        # Deliberately no counterfactual_states: fabricated current states contradict
        # the recorded frame-by-frame action history.
        self.base = StateConditionedVideoDataset(
            self.samples,
            project_root=self.project_root,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=require_cache,
            counterfactual_states=False,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        initial, transitions = self.events[sample.video]
        history = encode_action_history(
            sample.input_timestamps_ms,
            sample.input_timestamps_ms[-1],
            initial_pressed=initial,
            transitions=transitions,
            current_pressed=bool(sample.current_pressed),
            max_age_ms=self.max_age_ms,
        )
        item = dict(self.base[index])
        item["action_history"] = history
        return item

    def close(self) -> None:
        self.base.close()

    def __getstate__(self) -> dict[str, object]:
        return self.__dict__.copy()

    def __del__(self) -> None:
        if hasattr(self, "base"):
            self.close()
