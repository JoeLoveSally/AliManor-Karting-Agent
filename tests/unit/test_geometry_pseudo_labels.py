from __future__ import annotations

import cv2
import numpy as np

from karting_agent.train.geometry_pseudo_labels import (
    RectilinearGeometryConfig,
    RoadMaskConfig,
    estimate_rectilinear_geometry,
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


def test_extract_road_mask_combines_multiple_visual_themes() -> None:
    hsv = np.zeros((180, 180, 3), dtype=np.uint8)
    hsv[:, :] = (94, 120, 230)  # bright cyan background, intentionally excluded
    hsv[20:80, 20:100] = (116, 180, 150)  # blue road
    hsv[100:160, 70:160] = (60, 170, 150)  # green road
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    config = RoadMaskConfig(
        hsv_ranges=(
            ((103, 70, 35), (135, 255, 210)),
            ((35, 55, 30), (85, 255, 205)),
        ),
        min_component_area_fraction=0.01,
        max_components=4,
    )
    mask = extract_road_mask(frame, config)

    assert mask[30:70, 30:90].mean() > 240
    assert mask[110:150, 80:150].mean() > 240
    assert mask[:10, :10].mean() < 5


def test_row_centerline_uses_widest_contiguous_run() -> None:
    mask = np.zeros((64, 100), dtype=np.uint8)
    mask[:, 10:20] = 255
    mask[:, 40:80] = 255

    points = row_centerline(mask, row_step=16, min_run_width=5)

    assert len(points) == 4
    for _, x_norm in points:
        assert abs(x_norm - 59.5 / 99.0) < 1e-6


def test_rectilinear_geometry_classifies_single_straight_axis() -> None:
    mask = np.zeros((240, 240), dtype=np.uint8)
    mask[90:140, 20:220] = 255

    geometry = estimate_rectilinear_geometry(
        mask,
        RectilinearGeometryConfig(
            hough_threshold=20,
            # Long-axis filtering suppresses the short road-width end caps.
            min_line_length_fraction=0.16,
            max_line_gap_fraction=0.03,
        ),
    )

    assert geometry.geometry_class == "straight"
    assert geometry.primary_angle_deg is not None
    assert min(abs(geometry.primary_angle_deg), abs(180 - geometry.primary_angle_deg)) < 5
    assert geometry.secondary_angle_deg is None
    assert not geometry.corner_visible


def test_rectilinear_geometry_detects_visible_l_corner() -> None:
    mask = np.zeros((260, 260), dtype=np.uint8)
    # Thick L-shaped drivable area. Both road legs are much longer than road width.
    mask[70:120, 30:220] = 255
    mask[70:230, 170:220] = 255

    geometry = estimate_rectilinear_geometry(
        mask,
        RectilinearGeometryConfig(
            hough_threshold=18,
            min_line_length_fraction=0.16,
            max_line_gap_fraction=0.04,
            axis_tolerance_deg=8.0,
            min_axis_separation_deg=40.0,
            corner_extension_fraction=0.12,
            anchor_x_norm=0.5,
            anchor_y_norm=0.5,
        ),
    )

    assert geometry.primary_angle_deg is not None
    assert geometry.secondary_angle_deg is not None
    separation = abs(geometry.primary_angle_deg - geometry.secondary_angle_deg) % 180
    separation = min(separation, 180 - separation)
    assert separation > 70
    assert geometry.corner_score > 0
    assert geometry.corner_visible
    assert geometry.corner_x_norm is not None
    assert geometry.corner_y_norm is not None
    assert geometry.corner_distance_norm is not None
