"""Explicit-history inference for a trained OFFLINE V4-C2 residual candidate.

Not compatible with the original ADB runner's two-argument model interface.
There is deliberately no implicit fallback to guessing action history from pixels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from karting_agent.model.action_history import encode_action_history, history_feature_width
from karting_agent.model.action_history_residual import ActionHistoryResidualPolicy
from karting_agent.model.state_conditioned_runner import StateConditionedModelRunner


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


class ActionHistoryAdapterRunner:
    def __init__(
        self,
        *,
        base_model_path: Path,
        base_metadata_path: Path,
        adapter_path: Path,
        adapter_metadata_path: Path,
        device: str | None = None,
    ) -> None:
        import torch

        manifest = json.loads(Path(adapter_metadata_path).read_text(encoding="utf-8"))
        if manifest.get("model_family") != "experimental_v4c2_action_history_residual_v1":
            raise ValueError("adapter metadata has unsupported model family")
        if bool(manifest.get("deployment_ready", True)):
            raise ValueError("experimental adapter may not be marked deployment-ready")
        base_model_path = Path(base_model_path).resolve()
        base_metadata_path = Path(base_metadata_path).resolve()
        if _sha256(base_model_path) != manifest["base_model_sha256"]:
            raise ValueError("base checkpoint SHA-256 differs from the training checkpoint")
        if _sha256(base_metadata_path) != manifest["base_metadata_sha256"]:
            raise ValueError("base metadata SHA-256 differs from the training metadata")
        self.baseline = StateConditionedModelRunner(
            base_model_path, metadata_path=base_metadata_path, device=device
        )
        self._torch = torch
        self.frame_stack = self.baseline.spec.frame_stack
        if self.frame_stack != int(manifest["frame_stack"]):
            raise ValueError("adapter frame stack mismatch")
        if list(self.baseline.spec.prediction_horizons_ms) != manifest["prediction_horizons_ms"]:
            raise ValueError("adapter horizon mismatch")
        width = history_feature_width(self.frame_stack)
        if width != int(manifest["history_feature_width"]):
            raise ValueError("adapter history-feature width mismatch")
        self.max_age_ms = float(manifest["max_age_ms"])
        self.model = ActionHistoryResidualPolicy(
            self.baseline.model, history_width=width
        ).to(self.baseline.device)
        adapter_state = torch.load(
            Path(adapter_path), map_location=self.baseline.device, weights_only=True
        )
        self.model.residual.load_state_dict(adapter_state, strict=True)
        self.model.eval()

    def predict_switch_all(
        self,
        images: np.ndarray,
        *,
        frame_timestamps_ms: Sequence[float],
        observation_timestamp_ms: float,
        initial_pressed: bool,
        transitions: Sequence[tuple[float, bool]],
        current_pressed: bool,
    ) -> tuple[float, ...]:
        if len(frame_timestamps_ms) != self.frame_stack:
            raise ValueError("frame timestamp count differs from model frame stack")
        action_history = encode_action_history(
            frame_timestamps_ms,
            observation_timestamp_ms,
            initial_pressed=initial_pressed,
            transitions=transitions,
            current_pressed=current_pressed,
            max_age_ms=self.max_age_ms,
        )
        tensor = self.baseline._tensor(images)
        states = self._torch.tensor([int(current_pressed)], device=self.baseline.device)
        history_tensor = self._torch.from_numpy(action_history).unsqueeze(0).to(
            self.baseline.device
        )
        with self._torch.inference_mode():
            logits, _ = self.model(tensor, states, history_tensor)
            probabilities = self._torch.sigmoid(logits)[0]
        return tuple(float(value) for value in probabilities.detach().cpu().tolist())
