"""Runtime loader for state-conditioned transition policies."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from karting_agent.model.runner import _runtime_spec, _select_device
from karting_agent.model.state_conditioned import build_state_conditioned_model
from karting_agent.runtime.future_action_projection import project_switch_probabilities


_SUPPORTED_MODEL_FAMILIES = {
    "state_conditioned_transition_v3",
    "state_conditioned_axis_v4c1",
    "state_conditioned_kart_relative_v4c2",
    "state_conditioned_kart_relative_v5_h0",
}

_SUPPORTED_SWITCH_PROBABILITY_SOURCES = {
    "native_switch",
    "future_action_projection",
}

_AUXILIARY_HEAD_PREFIXES = {
    "state_conditioned_axis_v4c1": ("axis_head.",),
    "state_conditioned_kart_relative_v4c2": (
        "lateral_head.",
        "heading_error_head.",
        "edge_risk_head.",
    ),
    "state_conditioned_kart_relative_v5_h0": (
        "lateral_head.",
        "heading_error_head.",
        "edge_risk_head.",
    ),
}


def _control_state_dict(state_dict: dict[str, object], model_family: str) -> dict[str, object]:
    """Return only parameters required by the deployed v3 control path."""

    prefixes = _AUXILIARY_HEAD_PREFIXES.get(model_family, ())
    if not prefixes:
        return state_dict
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(prefixes)
    }


def _switch_probability_source(metadata: dict[str, object]) -> str:
    """Resolve the runtime switch-probability source from artifact metadata."""

    source = str(metadata.get("switch_probability_source", "native_switch"))
    if source not in _SUPPORTED_SWITCH_PROBABILITY_SOURCES:
        expected = ", ".join(sorted(_SUPPORTED_SWITCH_PROBABILITY_SOURCES))
        raise ValueError(
            f"unsupported switch_probability_source {source!r}; expected one of: {expected}"
        )
    return source


class StateConditionedModelRunner:
    """Load a state-conditioned artifact and predict KEEP/SWITCH probability."""

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
        model_family = str(metadata.get("model_family", ""))
        if model_family not in _SUPPORTED_MODEL_FAMILIES:
            raise ValueError(
                "metadata is not a supported state-conditioned artifact: "
                f"{model_family or '<missing>'}"
            )

        self.metadata = metadata
        self.model_family = model_family
        self.switch_probability_source = _switch_probability_source(metadata)
        self.spec = _runtime_spec(metadata)
        self._torch = torch
        self.device = _select_device(torch, device)
        self.model = build_state_conditioned_model(
            self.spec.architecture,
            frame_stack=self.spec.frame_stack,
            pretrained=False,
            horizon_count=len(self.spec.prediction_horizons_ms),
            visual_feature_dim=int(metadata["visual_feature_dim"]),
            state_embedding_dim=int(metadata["state_embedding_dim"]),
            hidden_dim=int(metadata["hidden_dim"]),
        ).to(self.device)

        try:
            state_dict = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(self.model_path, map_location=self.device)
        if not isinstance(state_dict, dict):
            raise ValueError("model checkpoint must contain a state dict mapping")
        self.model.load_state_dict(_control_state_dict(state_dict, model_family))
        self.model.eval()

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (
            3 * self.spec.frame_stack,
            self.spec.preprocess_config.input_height,
            self.spec.preprocess_config.input_width,
        )

    def _tensor(self, inputs: np.ndarray):
        if inputs.shape != self.input_shape or inputs.dtype != np.float32:
            raise ValueError(
                f"model input must be float32 with shape {self.input_shape}, got "
                f"{inputs.dtype} {inputs.shape}"
            )
        tensor = self._torch.from_numpy(np.ascontiguousarray(inputs)).unsqueeze(0)
        return tensor.to(device=self.device, dtype=self._torch.float32)

    def predict_switch_all(
        self,
        inputs: np.ndarray,
        current_pressed: bool,
    ) -> tuple[float, ...]:
        tensor = self._tensor(inputs)
        state = self._torch.tensor(
            [1 if current_pressed else 0],
            device=self.device,
            dtype=self._torch.long,
        )
        with self._torch.inference_mode():
            switch_logits, future_logits = self.model(tensor, state)
            if self.switch_probability_source == "native_switch":
                probabilities = self._torch.sigmoid(switch_logits)[0]
                return tuple(
                    float(value) for value in probabilities.detach().cpu().tolist()
                )
            future_probabilities = self._torch.sigmoid(future_logits)[0]
            future_values = tuple(
                float(value)
                for value in future_probabilities.detach().cpu().tolist()
            )
        return project_switch_probabilities(
            future_values,
            current_pressed=current_pressed,
        )

    def predict_native_switch_and_future_all(
        self,
        inputs: np.ndarray,
        current_pressed: bool,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return native switch and absolute future-action probabilities in one pass."""

        tensor = self._tensor(inputs)
        state = self._torch.tensor(
            [1 if current_pressed else 0],
            device=self.device,
            dtype=self._torch.long,
        )
        with self._torch.inference_mode():
            switch_logits, future_logits = self.model(tensor, state)
            switch_probabilities = self._torch.sigmoid(switch_logits)[0]
            future_probabilities = self._torch.sigmoid(future_logits)[0]
            switch_values = tuple(
                float(value)
                for value in switch_probabilities.detach().cpu().tolist()
            )
            future_values = tuple(
                float(value)
                for value in future_probabilities.detach().cpu().tolist()
            )
        return switch_values, future_values

    def predict_switch(self, inputs: np.ndarray, current_pressed: bool) -> float:
        probabilities = self.predict_switch_all(inputs, current_pressed)
        return probabilities[self.spec.control_output_index]

    def predict_future_all(self, inputs: np.ndarray) -> tuple[float, ...]:
        tensor = self._tensor(inputs)
        state = self._torch.zeros(1, device=self.device, dtype=self._torch.long)
        with self._torch.inference_mode():
            _, future_logits = self.model(tensor, state)
            probabilities = self._torch.sigmoid(future_logits)[0]
        return tuple(float(value) for value in probabilities.detach().cpu().tolist())

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
        return tuple(latencies)
