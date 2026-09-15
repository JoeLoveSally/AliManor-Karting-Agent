"""Pseudo-label loading and dataset adapter for v4-C1 road-axis supervision."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from karting_agent.train.dataset import DatasetSample
from karting_agent.train.state_conditioned_dataset import StateConditionedVideoDataset
from karting_agent.vision.preprocess import PreprocessConfig


@dataclass(frozen=True)
class AxisPseudoLabel:
    video: str
    frame_index: int
    angle_deg: float | None
    target_x: float
    target_y: float
    weight: float
    straight_confidence: float
    corner_score: float
    road_area_fraction: float

    @property
    def key(self) -> tuple[str, int]:
        return self.video, self.frame_index


def load_axis_pseudo_labels(path: Path) -> dict[tuple[str, int], AxisPseudoLabel]:
    labels: dict[tuple[str, int], AxisPseudoLabel] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("axis label must be a mapping")
                angle = raw.get("angle_deg")
                label = AxisPseudoLabel(
                    video=str(raw["video"]),
                    frame_index=int(raw["frame_index"]),
                    angle_deg=None if angle is None else float(angle),
                    target_x=float(raw.get("target_x", 0.0)),
                    target_y=float(raw.get("target_y", 0.0)),
                    weight=float(raw.get("weight", 0.0)),
                    straight_confidence=float(raw.get("straight_confidence", 0.0)),
                    corner_score=float(raw.get("corner_score", 0.0)),
                    road_area_fraction=float(raw.get("road_area_fraction", 0.0)),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid axis label at {path}:{line_number}") from exc
            if not 0.0 <= label.weight <= 1.0:
                raise ValueError(f"axis label weight must be in [0,1] at {path}:{line_number}")
            if label.key in labels:
                raise ValueError(f"duplicate axis label key: {label.key}")
            labels[label.key] = label
    return labels


class AxisSupervisedVideoDataset:
    """Add current-frame road-axis targets to the unchanged v3 control dataset.

    The road-axis label is attached to the observation frame at ``t`` (the last
    frame in the v3 history). Counterfactual PRESS/RELEASE duplication therefore
    reuses the same visual axis target, as intended.
    """

    def __init__(
        self,
        samples: Sequence[DatasetSample],
        *,
        axis_labels: dict[tuple[str, int], AxisPseudoLabel],
        project_root: Path,
        preprocess_config: PreprocessConfig = PreprocessConfig(),
        cache_root: Path | None = None,
        require_cache: bool = False,
        counterfactual_states: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.axis_labels = axis_labels
        self.counterfactual_states = counterfactual_states
        self.base = StateConditionedVideoDataset(
            self.samples,
            project_root=project_root,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=require_cache,
            counterfactual_states=counterfactual_states,
        )

        missing = []
        for sample in self.samples:
            key = (sample.video, int(sample.input_frame_indices[-1]))
            if key not in self.axis_labels:
                missing.append(key)
                if len(missing) >= 5:
                    break
        if missing:
            raise ValueError(f"axis labels missing sample observation frames, examples={missing}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = dict(self.base[index])
        sample_index = index // 2 if self.counterfactual_states else index
        sample = self.samples[sample_index]
        label = self.axis_labels[(sample.video, int(sample.input_frame_indices[-1]))]
        item["axis_target"] = np.asarray(
            [label.target_x, label.target_y], dtype=np.float32
        )
        item["axis_weight"] = np.float32(label.weight)
        return item

    def close(self) -> None:
        self.base.close()

    def __getstate__(self) -> dict[str, object]:
        return self.__dict__.copy()

    def __del__(self) -> None:
        self.close()
