"""Pseudo-label loading and dataset adapter for v4-C2 kart-relative supervision."""

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
class KartRelativePseudoLabel:
    video: str
    frame_index: int
    lateral_target: float
    heading_target_x: float
    heading_target_y: float
    edge_risk_target: float
    weight: float
    raw_lateral_offset_norm: float | None
    heading_error_deg: float | None
    inside_road: bool | None
    confidence: float
    heading_quality: float
    road_support_fraction: float
    valid_cross_sections: int
    cross_section_count: int

    @property
    def key(self) -> tuple[str, int]:
        return self.video, self.frame_index


def load_kart_relative_pseudo_labels(
    path: Path,
) -> dict[tuple[str, int], KartRelativePseudoLabel]:
    labels: dict[tuple[str, int], KartRelativePseudoLabel] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("kart-relative label must be a mapping")
                raw_lateral = raw.get("raw_lateral_offset_norm")
                heading_error = raw.get("heading_error_deg")
                inside_road = raw.get("inside_road")
                label = KartRelativePseudoLabel(
                    video=str(raw["video"]),
                    frame_index=int(raw["frame_index"]),
                    lateral_target=float(raw.get("lateral_target", 0.0)),
                    heading_target_x=float(raw.get("heading_target_x", 0.0)),
                    heading_target_y=float(raw.get("heading_target_y", 0.0)),
                    edge_risk_target=float(raw.get("edge_risk_target", 0.0)),
                    weight=float(raw.get("weight", 0.0)),
                    raw_lateral_offset_norm=(
                        None if raw_lateral is None else float(raw_lateral)
                    ),
                    heading_error_deg=(
                        None if heading_error is None else float(heading_error)
                    ),
                    inside_road=(None if inside_road is None else bool(inside_road)),
                    confidence=float(raw.get("confidence", 0.0)),
                    heading_quality=float(raw.get("heading_quality", 0.0)),
                    road_support_fraction=float(raw.get("road_support_fraction", 0.0)),
                    valid_cross_sections=int(raw.get("valid_cross_sections", 0)),
                    cross_section_count=int(raw.get("cross_section_count", 0)),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid kart-relative label at {path}:{line_number}"
                ) from exc
            if not 0.0 <= label.weight <= 1.0:
                raise ValueError(
                    f"kart-relative label weight must be in [0,1] at {path}:{line_number}"
                )
            if not 0.0 <= label.edge_risk_target <= 1.0:
                raise ValueError(
                    f"edge-risk target must be in [0,1] at {path}:{line_number}"
                )
            if label.key in labels:
                raise ValueError(f"duplicate kart-relative label key: {label.key}")
            labels[label.key] = label
    return labels


class KartRelativeSupervisedVideoDataset:
    """Attach current-frame kart-relative targets to the unchanged v3 dataset."""

    def __init__(
        self,
        samples: Sequence[DatasetSample],
        *,
        relation_labels: dict[tuple[str, int], KartRelativePseudoLabel],
        project_root: Path,
        preprocess_config: PreprocessConfig = PreprocessConfig(),
        cache_root: Path | None = None,
        require_cache: bool = False,
        counterfactual_states: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.relation_labels = relation_labels
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
            if key not in self.relation_labels:
                missing.append(key)
                if len(missing) >= 5:
                    break
        if missing:
            raise ValueError(
                "kart-relative labels missing sample observation frames, "
                f"examples={missing}"
            )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = dict(self.base[index])
        sample_index = index // 2 if self.counterfactual_states else index
        sample = self.samples[sample_index]
        label = self.relation_labels[
            (sample.video, int(sample.input_frame_indices[-1]))
        ]
        item["relation_weight"] = np.float32(label.weight)
        item["lateral_target"] = np.float32(label.lateral_target)
        item["heading_error_target"] = np.asarray(
            [label.heading_target_x, label.heading_target_y], dtype=np.float32
        )
        item["edge_risk_target"] = np.float32(label.edge_risk_target)
        return item

    def close(self) -> None:
        self.base.close()

    def __getstate__(self) -> dict[str, object]:
        return self.__dict__.copy()

    def __del__(self) -> None:
        self.close()
