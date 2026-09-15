from __future__ import annotations

import cv2
import numpy as np

from karting_agent.train.geometry_pseudo_labels import (
    RoadMaskConfig,
    extract_road_mask,
    row_centerline,
)


def test_extract_road_mask_selects_blue_track_over_cyan_background() -> None:
    hsv = np.zeros((160, 120, 3), dtype=np.uint8)
    hsv[:, :] = (94, 130, 220)
    hsv[:, 40:90] = (112, 165, 150)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    mask = extract_road_mask(frame, RoadMaskConfig())

    assert mask[:, 50:80].mean() > 240
    assert mask[:, :20].mean() < 5
    assert mask[:, 100:].mean() < 5


def test_row_centerline_uses_widest_contiguous_run() -> None:
    mask = np.zeros((64, 100), dtype=np.uint8)
    mask[:, 10:20] = 255
    mask[:, 40:80] = 255

    points = row_centerline(mask, row_step=16, min_run_width=5)

    assert len(points) == 4
    for _, x_norm in points:
        assert abs(x_norm - 59.5 / 99.0) < 1e-6
