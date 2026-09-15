"""Runtime loader for v4-A with cached per-frame CNN features."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from karting_agent.model.runner import _runtime_spec, _select_device
from karting_agent.model.sequential_state_conditioned import (
    build_sequential_state_conditioned_model,
)


class SequentialStateConditionedModelRunner:
    """Predict v4-A KEEP/SWITCH while reusing CNN features across windows."""

    def __init__(
        self,
        model_path: Path,
        *,
        metadata_path: Path | None = None,
        device: str | None = None,
        feature_cache_size: int = 64,
        torch_num_threads: int | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                'PyTorch is required; install with: python -m pip install -e ".[train]"'
            ) from exc
        if feature_cache_size < 1:
            raise ValueError("feature_cache_size must be >= 1")
        if torch_num_threads is not None and torch_num_threads < 1:
            raise ValueError("torch_num_threads must be >= 1")

        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model artifact not found: {self.model_path}")
        self.metadata_path = (
            Path(metadata_path).resolve()
            if metadata_path is not None
            else self.model_path.with_name("metadata.json")
        )
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"model metadata not found: {self.metadata_path}")

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("model metadata root must be a mapping")
        if metadata.get("model_family") != "sequential_state_conditioned_v4a":
            raise ValueError("metadata is not a v4-A sequential artifact")

        self.metadata = metadata
        self.spec = _runtime_spec(metadata)
        self._torch = torch
        self.device = _select_device(torch, device)
        if self.device.type == "cpu" and torch_num_threads is not None:
            torch.set_num_threads(torch_num_threads)
        self.torch_num_threads = torch.get_num_threads()

        self.model = build_sequential_state_conditioned_model(
            self.spec.architecture,
            pretrained=False,
            frame_stack=self.spec.frame_stack,
            horizon_count=len(self.spec.prediction_horizons_ms),
            visual_feature_dim=int(metadata["visual_feature_dim"]),
            gru_hidden_dim=int(metadata["gru_hidden_dim"]),
            gru_layers=int(metadata["gru_layers"]),
            state_embedding_dim=int(metadata["state_embedding_dim"]),
            policy_hidden_dim=int(metadata["policy_hidden_dim"]),
        ).to(self.device)
        try:
            state_dict = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.feature_cache_size = feature_cache_size
        self._feature_cache: OrderedDict[int, object] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (
            3 * self.spec.frame_stack,
            self.spec.preprocess_config.input_height,
            self.spec.preprocess_config.input_width,
        )

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return (
            3,
            self.spec.preprocess_config.input_height,
            self.spec.preprocess_config.input_width,
        )

    def clear_feature_cache(self) -> None:
        self._feature_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def _split_input(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.shape != self.input_shape or inputs.dtype != np.float32:
            raise ValueError(
                f"model input must be float32 with shape {self.input_shape}, got "
                f"{inputs.dtype} {inputs.shape}"
            )
        return np.ascontiguousarray(
            inputs.reshape(self.spec.frame_stack, *self.frame_shape)
        )

    def _encode_frame(self, frame: np.ndarray):
        tensor = self._torch.from_numpy(np.ascontiguousarray(frame)).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self._torch.float32)
        with self._torch.inference_mode():
            return self.model.visual_encoder(tensor)[0].detach()

    def _cached_features(
        self,
        inputs: np.ndarray,
        frame_indices: Sequence[int],
    ):
        frames = self._split_input(inputs)
        indices = tuple(int(value) for value in frame_indices)
        if len(indices) != self.spec.frame_stack:
            raise ValueError(
                f"expected {self.spec.frame_stack} frame indices, got {len(indices)}"
            )

        features = []
        for frame, frame_index in zip(frames, indices, strict=True):
            feature = self._feature_cache.get(frame_index)
            if feature is None:
                feature = self._encode_frame(frame)
                self._feature_cache[frame_index] = feature
                self.cache_misses += 1
                if len(self._feature_cache) > self.feature_cache_size:
                    self._feature_cache.popitem(last=False)
            else:
                self._feature_cache.move_to_end(frame_index)
                self.cache_hits += 1
            features.append(feature)
        return self._torch.stack(features, dim=0).unsqueeze(0)

    def _uncached_features(self, inputs: np.ndarray):
        frames = self._split_input(inputs)
        tensor = self._torch.from_numpy(frames).to(
            device=self.device, dtype=self._torch.float32
        )
        with self._torch.inference_mode():
            encoded = self.model.visual_encoder(tensor)
        return encoded.unsqueeze(0)

    def predict_switch_all(
        self,
        inputs: np.ndarray,
        current_pressed: bool,
        *,
        frame_indices: Sequence[int] | None = None,
    ) -> tuple[float, ...]:
        sequence = (
            self._uncached_features(inputs)
            if frame_indices is None
            else self._cached_features(inputs, frame_indices)
        )
        state = self._torch.tensor(
            [1 if current_pressed else 0],
            device=self.device,
            dtype=self._torch.long,
        )
        with self._torch.inference_mode():
            switch_logits, _ = self.model.forward_from_features(sequence, state)
            probabilities = self._torch.sigmoid(switch_logits)[0]
        return tuple(float(value) for value in probabilities.detach().cpu().tolist())

    def predict_switch(
        self,
        inputs: np.ndarray,
        current_pressed: bool,
        *,
        frame_indices: Sequence[int] | None = None,
    ) -> float:
        probabilities = self.predict_switch_all(
            inputs,
            current_pressed,
            frame_indices=frame_indices,
        )
        return probabilities[self.spec.control_output_index]

    def warmup(self, iterations: int = 3) -> tuple[float, ...]:
        if iterations < 1:
            raise ValueError("warmup iterations must be >= 1")
        dummy = np.zeros(self.input_shape, dtype=np.float32)
        latencies: list[float] = []
        for index in range(iterations):
            if self.device.type == "cuda":
                self._torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            self.predict_switch(dummy, current_pressed=bool(index % 2))
            if self.device.type == "cuda":
                self._torch.cuda.synchronize(self.device)
            latencies.append((time.perf_counter() - started) * 1000.0)
        self.clear_feature_cache()
        return tuple(latencies)
