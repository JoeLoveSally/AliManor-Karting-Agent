"""Shared image preprocessing for training and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


Roi = tuple[float, float, float, float]


def _validate_roi(roi: Roi, name: str) -> None:
    x0, y0, x1, y1 = roi
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"{name} must contain normalized coordinates in [0, 1]")


@dataclass(frozen=True)
class PreprocessConfig:
    input_width: int = 224
    input_height: int = 224
    mask_touch_area: bool = True
    touch_roi: Roi = (0.78, 0.82, 0.98, 0.98)
    mask_rois: tuple[Roi, ...] = ()
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def validate(self) -> None:
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("input size must be positive")
        _validate_roi(self.touch_roi, "touch_roi")
        for index, roi in enumerate(self.mask_rois):
            _validate_roi(roi, f"mask_rois[{index}]")
        if any(value <= 0 for value in self.std):
            raise ValueError("std values must be positive")


def preprocess_config_from_mapping(raw: dict[str, object]) -> PreprocessConfig:
    model = raw.get("model", {})
    preprocess = raw.get("preprocess", {})
    if not isinstance(model, dict) or not isinstance(preprocess, dict):
        raise ValueError("model and preprocess config must be mappings")

    size = int(model.get("input_size", 224))
    roi = tuple(
        float(value)
        for value in preprocess.get("touch_roi", (0.78, 0.82, 0.98, 0.98))
    )
    if len(roi) != 4:
        raise ValueError("preprocess.touch_roi must contain four values")

    raw_mask_rois = preprocess.get("mask_rois", ())
    if not isinstance(raw_mask_rois, (list, tuple)):
        raise ValueError("preprocess.mask_rois must be a sequence")
    mask_rois: list[Roi] = []
    for index, raw_roi in enumerate(raw_mask_rois):
        if not isinstance(raw_roi, (list, tuple)) or len(raw_roi) != 4:
            raise ValueError(f"preprocess.mask_rois[{index}] must contain four values")
        mask_rois.append(tuple(float(value) for value in raw_roi))

    config = PreprocessConfig(
        input_width=size,
        input_height=size,
        mask_touch_area=bool(preprocess.get("mask_touch_area", True)),
        touch_roi=roi,  # type: ignore[arg-type]
        mask_rois=tuple(mask_rois),
    )
    config.validate()
    return config


def mask_area(frame: np.ndarray, roi: Roi) -> np.ndarray:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image with shape HxWx3")

    height, width = frame.shape[:2]
    x0, y0, x1, y1 = roi
    left = max(0, min(width, round(x0 * width)))
    top = max(0, min(height, round(y0 * height)))
    right = max(left, min(width, round(x1 * width)))
    bottom = max(top, min(height, round(y1 * height)))

    masked = frame.copy()
    masked[top:bottom, left:right] = 0
    return masked


def mask_touch_area(frame: np.ndarray, roi: Roi) -> np.ndarray:
    return mask_area(frame, roi)


def prepare_frame(
    frame: np.ndarray,
    config: PreprocessConfig = PreprocessConfig(),
) -> np.ndarray:
    """Apply deterministic spatial preprocessing and return RGB uint8 HWC."""
    config.validate()
    image = frame
    if config.mask_touch_area:
        image = mask_area(image, config.touch_roi)
    for roi in config.mask_rois:
        image = mask_area(image, roi)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (config.input_width, config.input_height),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(image, dtype=np.uint8)


def normalize_prepared_frame(
    image: np.ndarray,
    config: PreprocessConfig = PreprocessConfig(),
) -> np.ndarray:
    """Normalize one prepared RGB uint8 HWC frame into CHW float32."""
    config.validate()
    if (
        image is None
        or image.ndim != 3
        or image.shape != (config.input_height, config.input_width, 3)
        or image.dtype != np.uint8
    ):
        raise ValueError(
            "prepared frame must be RGB uint8 with configured HxWx3 shape"
        )

    normalized = image.astype(np.float32) / 255.0
    mean = np.asarray(config.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(config.std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (normalized - mean) / std
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)


def preprocess_frame(
    frame: np.ndarray,
    config: PreprocessConfig = PreprocessConfig(),
) -> np.ndarray:
    return normalize_prepared_frame(prepare_frame(frame, config), config)


def stack_frames(
    frames: Sequence[np.ndarray],
    config: PreprocessConfig = PreprocessConfig(),
) -> np.ndarray:
    if not frames:
        raise ValueError("frames must not be empty")
    processed = [preprocess_frame(frame, config) for frame in frames]
    return np.concatenate(processed, axis=0)


def stack_prepared_frames(
    frames: Sequence[np.ndarray],
    config: PreprocessConfig = PreprocessConfig(),
) -> np.ndarray:
    if not frames:
        raise ValueError("frames must not be empty")
    processed = [normalize_prepared_frame(frame, config) for frame in frames]
    return np.concatenate(processed, axis=0)
