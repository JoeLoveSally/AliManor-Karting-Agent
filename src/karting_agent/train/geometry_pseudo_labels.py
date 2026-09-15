"""Offline pseudo-label helpers for structured visual supervision."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RoadMaskConfig:
    """Theme-specific HSV road-mask configuration for the initial v4 POC."""

    hsv_lower: tuple[int, int, int] = (103, 90, 50)
    hsv_upper: tuple[int, int, int] = (130, 255, 220)
    close_kernel: int = 9
    open_kernel: int = 3
    min_component_area_fraction: float = 0.02

    def validate(self) -> None:
        if any(value < 0 for value in self.hsv_lower):
            raise ValueError("hsv_lower values must be >= 0")
        if self.hsv_lower[0] > 179 or self.hsv_upper[0] > 179:
            raise ValueError("OpenCV HSV hue must be <= 179")
        if any(value > 255 for value in self.hsv_lower[1:] + self.hsv_upper[1:]):
            raise ValueError("HSV saturation/value must be <= 255")
        if any(low > high for low, high in zip(self.hsv_lower, self.hsv_upper)):
            raise ValueError("hsv_lower must not exceed hsv_upper")
        if self.close_kernel < 1 or self.open_kernel < 1:
            raise ValueError("morphology kernels must be >= 1")
        if not 0.0 <= self.min_component_area_fraction <= 1.0:
            raise ValueError("min_component_area_fraction must be in [0, 1]")


def _odd_kernel(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def extract_road_mask(frame_bgr: np.ndarray, config: RoadMaskConfig) -> np.ndarray:
    """Return a binary uint8 road mask using theme-specific HSV supervision."""

    config.validate()
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must have shape HxWx3")

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.asarray(config.hsv_lower, dtype=np.uint8),
        np.asarray(config.hsv_upper, dtype=np.uint8),
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_kernel(config.close_kernel), _odd_kernel(config.close_kernel)),
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_kernel(config.open_kernel), _odd_kernel(config.open_kernel)),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if component_count <= 1:
        return np.zeros(mask.shape, dtype=np.uint8)

    image_area = mask.shape[0] * mask.shape[1]
    min_area = image_area * config.min_component_area_fraction
    candidate_labels = [
        label
        for label in range(1, component_count)
        if stats[label, cv2.CC_STAT_AREA] >= min_area
    ]
    if not candidate_labels:
        return np.zeros(mask.shape, dtype=np.uint8)

    largest = max(candidate_labels, key=lambda label: stats[label, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def row_centerline(
    mask: np.ndarray,
    *,
    row_step: int = 16,
    min_run_width: int = 12,
) -> list[tuple[int, float]]:
    """Return diagnostic center points from the widest road run on sampled rows."""

    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    if row_step < 1 or min_run_width < 1:
        raise ValueError("row_step and min_run_width must be >= 1")

    points: list[tuple[int, float]] = []
    width = mask.shape[1]
    for y in range(0, mask.shape[0], row_step):
        xs = np.flatnonzero(mask[y] > 0)
        if xs.size == 0:
            continue

        breaks = np.where(np.diff(xs) > 1)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [xs.size - 1]))
        runs = [(int(xs[start]), int(xs[end])) for start, end in zip(starts, ends)]
        left, right = max(runs, key=lambda run: run[1] - run[0] + 1)
        if right - left + 1 < min_run_width:
            continue
        center_x = (left + right) * 0.5
        points.append((y, center_x / max(1, width - 1)))
    return points


def road_mask_metrics(mask: np.ndarray, centerline: list[tuple[int, float]]) -> dict[str, float]:
    """Return cheap diagnostics for pseudo-label inspection."""

    area_fraction = float(np.mean(mask > 0))
    possible_rows = max(1, int(np.ceil(mask.shape[0] / 16.0)))
    return {
        "road_area_fraction": area_fraction,
        "centerline_valid_fraction": min(1.0, len(centerline) / possible_rows),
    }


def overlay_road_geometry(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    centerline: list[tuple[int, float]],
) -> np.ndarray:
    """Render road mask and centerline for human pseudo-label inspection."""

    overlay = frame_bgr.copy()
    tint = np.zeros_like(frame_bgr)
    tint[..., 1] = 255
    selected = mask > 0
    overlay[selected] = (
        0.65 * overlay[selected].astype(np.float32)
        + 0.35 * tint[selected].astype(np.float32)
    ).astype(np.uint8)

    width = frame_bgr.shape[1]
    for y, x_norm in centerline:
        x = int(round(x_norm * max(1, width - 1)))
        cv2.circle(overlay, (x, int(y)), 3, (0, 255, 255), -1)
    return overlay
