"""Runtime model artifact loader and single-sample inference."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from karting_agent.vision.preprocess import PreprocessConfig, preprocess_config_from_mapping


@dataclass(frozen=True)
class ModelRuntimeSpec:
    architecture: str
    frame_stack: int
    frame_offsets_ms: tuple[float, ...]
    prediction_horizon_ms: float
    preprocess_config: PreprocessConfig

    @property
    def history_ms(self) -> float:
        return max(0.0, -min(self.frame_offsets_ms))


def _runtime_spec(metadata: dict[str, object]) -> ModelRuntimeSpec:
    raw_config = metadata.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("model metadata must contain the training config")
    model_config = raw_config.get("model", {})
    dataset_config = raw_config.get("dataset", {})
    if not isinstance(model_config, dict) or not isinstance(dataset_config, dict):
        raise ValueError("metadata model/dataset config must be mappings")

    architecture = str(
        metadata.get("architecture", model_config.get("architecture", ""))
    )
    frame_stack = int(metadata.get("frame_stack", model_config.get("frame_stack", 0)))
    frame_interval_ms = float(dataset_config.get("frame_interval_ms", 0.0))
    prediction_horizon_ms = float(dataset_config.get("prediction_horizon_ms", 0.0))
    history_ms = float(dataset_config.get("history_ms", 0.0))

    if not architecture:
        raise ValueError("model architecture is missing from metadata")
    if frame_stack < 1:
        raise ValueError("frame_stack must be >= 1")
    if frame_interval_ms < 0 or prediction_horizon_ms < 0:
        raise ValueError("runtime temporal intervals must be >= 0")
    expected_history_ms = (frame_stack - 1) * frame_interval_ms
    if not math.isclose(history_ms, expected_history_ms, abs_tol=1e-6):
        raise ValueError(
            "metadata history_ms must equal (frame_stack - 1) * frame_interval_ms"
        )

    frame_offsets_ms = tuple(
        -(frame_stack - 1 - index) * frame_interval_ms
        for index in range(frame_stack)
    )
    return ModelRuntimeSpec(
        architecture=architecture,
        frame_stack=frame_stack,
        frame_offsets_ms=frame_offsets_ms,
        prediction_horizon_ms=prediction_horizon_ms,
        preprocess_config=preprocess_config_from_mapping(raw_config),
    )


def _select_device(torch, requested: str | None):
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelRunner:
    """Load one trained artifact and predict PRESS probability for one stack."""

    def __init__(
        self,
        model_path: Path,
        *,
        metadata_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                'PyTorch is required; install with: python -m pip install -e ".[train]"'
            ) from exc

        from karting_agent.model.base import build_model

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
        self.metadata = metadata
        self.spec = _runtime_spec(metadata)
        self._torch = torch
        self.device = _select_device(torch, device)

        self.model = build_model(
            self.spec.architecture,
            frame_stack=self.spec.frame_stack,
            pretrained=False,
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

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (
            3 * self.spec.frame_stack,
            self.spec.preprocess_config.input_height,
            self.spec.preprocess_config.input_width,
        )

    def predict(self, inputs: np.ndarray) -> float:
        expected_shape = self.input_shape
        if inputs.shape != expected_shape or inputs.dtype != np.float32:
            raise ValueError(
                f"model input must be float32 with shape {expected_shape}, got "
                f"{inputs.dtype} {inputs.shape}"
            )

        tensor = self._torch.from_numpy(np.ascontiguousarray(inputs)).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self._torch.float32)
        with self._torch.inference_mode():
            probability = self._torch.sigmoid(self.model(tensor))[0].item()
        return float(probability)

    def warmup(self, iterations: int = 3) -> tuple[float, ...]:
        """Run untimed-control dummy predictions before entering RUNNING state."""
        if iterations < 1:
            raise ValueError("warmup iterations must be >= 1")

        dummy = np.zeros(self.input_shape, dtype=np.float32)
        latencies: list[float] = []
        for _ in range(iterations):
            if self.device.type == "cuda":
                self._torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            self.predict(dummy)
            if self.device.type == "cuda":
                self._torch.cuda.synchronize(self.device)
            latencies.append((time.perf_counter() - started) * 1000.0)
        return tuple(latencies)
