"""CPU/GPU inference adapter for the V4-C4 absolute current-action head.

Inputs are the normalized RGB history produced by TemporalVideoDataset.
The model's temporal-delta representation is applied exactly once here.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from karting_agent.model.temporal_delta import (
    transform_temporal_input_numpy,
    validate_temporal_input_representation,
)


class EventTimeActionRunner:
    def __init__(
        self,
        model_path: Path,
        *,
        metadata_path: Path | None = None,
        device: str = "cpu",
        torch_num_threads: int | None = None,
    ) -> None:
        import torch
        from karting_agent.model.event_time import build_event_time_model

        if torch_num_threads is not None:
            if torch_num_threads < 1:
                raise ValueError("torch_num_threads must be >= 1")
            torch.set_num_threads(torch_num_threads)
        self.model_path = Path(model_path).resolve()
        self.metadata_path = (
            Path(metadata_path).resolve()
            if metadata_path is not None
            else self.model_path.with_name("metadata.json")
        )
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_family") != "event_time_v4c4":
            raise ValueError("checkpoint is not an event_time_v4c4 artifact")
        self.metadata = metadata
        self.frame_stack = int(metadata["frame_stack"])
        if self.frame_stack != 5:
            raise ValueError("this V4-C4 runtime requires a five-frame history")
        self.input_representation = validate_temporal_input_representation(
            str(metadata["input_representation"])
        )
        if self.input_representation != "current_rgb_plus_adjacent_deltas":
            raise ValueError("this V4-C4 runtime requires adjacent temporal deltas")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self._torch = torch
        self.model = build_event_time_model(
            architecture=str(metadata["architecture"]),
            frame_stack=self.frame_stack,
            pretrained=False,
            event_time_classes=int(metadata["event_time_classes"]),
            visual_feature_dim=int(metadata["visual_feature_dim"]),
            hidden_dim=int(metadata["hidden_dim"]),
        ).to(self.device)
        weights = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()

    def predict_action(self, normalized_rgb_stack: np.ndarray) -> float:
        """Return PRESS probability from oldest-to-newest normalized RGB frames."""
        expected = (3 * self.frame_stack, 224, 224)
        if normalized_rgb_stack.shape != expected or normalized_rgb_stack.dtype != np.float32:
            raise ValueError(
                f"expected float32 normalized RGB stack {expected}, got "
                f"{normalized_rgb_stack.shape} {normalized_rgb_stack.dtype}"
            )
        transformed = transform_temporal_input_numpy(
            normalized_rgb_stack,
            frame_stack=self.frame_stack,
            representation=self.input_representation,
        )
        tensor = self._torch.from_numpy(np.ascontiguousarray(transformed)).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self._torch.float32)
        with self._torch.inference_mode():
            current_action_logit, *_ = self.model(tensor)
            return float(self._torch.sigmoid(current_action_logit)[0].item())

    def warmup(self, iterations: int = 3) -> tuple[float, ...]:
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        data = np.zeros((3 * self.frame_stack, 224, 224), dtype=np.float32)
        result: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.predict_action(data)
            result.append((time.perf_counter() - start) * 1000.0)
        return tuple(result)
