"""Mode-specific expert video tensors for matched visual feature ablations."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def make_mode_images(
    frame_indices: Sequence[int],
    frames_by_index: Mapping[int, np.ndarray] | None,
    *,
    mode: str,
    frame_size: int = 96,
) -> np.ndarray:
    """Return [T,C,H,W] floats in [0,1] with an image-free control.

    action_only intentionally presents a constant zero image to the SAME
    CNN/GRU topology as RGB. It must not even fetch/cache source frames.
    """
    if mode not in ("rgb", "rgb_hsv", "action_only"):
        raise ValueError("mode must be rgb, rgb_hsv or action_only")
    if frame_size <= 0 or not frame_indices:
        raise ValueError("invalid frame size or empty history")
    if mode == "action_only":
        return np.zeros((len(frame_indices), 3, frame_size, frame_size), np.float32)
    if frames_by_index is None:
        raise ValueError("RGB/HSV mode requires the decoded frame cache")
    channels = 4 if mode == "rgb_hsv" else 3
    images = []
    for frame_index in frame_indices:
        frame = frames_by_index[int(frame_index)]
        if (frame.shape != (4, frame_size, frame_size)
                or frame.dtype != np.uint8):
            raise ValueError("cached frames must be uint8 [4,H,W]")
        images.append(frame[:channels])
    return np.stack(images).astype(np.float32) / 255.0
