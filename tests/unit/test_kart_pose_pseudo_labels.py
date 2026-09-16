from __future__ import annotations

import cv2
import numpy as np
import pytest

from karting_agent.train.geometry_pseudo_labels import RectilinearGeometry
from karting_agent.train.kart_pose_pseudo_labels import (
    KartPose,
    KartPoseConfig,
    KartRoadRelationConfig,
    estimate_kart_pose,
    estimate_kart_road_relation,
    signed_axial_difference_deg,
)


def _geometry(angle_deg: float) -> RectilinearGeometry:
    return RectilinearGeometry(
        geometry_class="edge_only",
        primary_angle_deg=angle_deg,
        secondary_angle_deg=None,
        primary_support=100.0,
        secondary_support=0.0,
        straight_confidence=0.9,
        corner_score=0.05,
        primary_center_axis=None,
        secondary_center_axis=None,
        corner_visible=False,
        corner_x_norm=None,
        corner_y_norm=None,
        corner_distance_norm=None,
        segments=(),
    )


def test_kart_pose_detects_warm_elongated_chassis() -> None:
    hsv = np.zeros((200, 240, 3), dtype=np.uint8)
    hsv[:, :] = (95, 120, 230)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    rectangle = ((120.0, 100.0), (40.0, 10.0), 30.0)
    box = np.int32(np.rint(cv2.boxPoints(rectangle)))
    chassis_hsv = np.full((200, 240, 3), (5, 220, 220), dtype=np.uint8)
    chassis_bgr = cv2.cvtColor(chassis_hsv, cv2.COLOR_HSV2BGR)
    mask = np.zeros((200, 240), dtype=np.uint8)
    cv2.fillConvexPoly(mask, box, 255)
    frame[mask > 0] = chassis_bgr[mask > 0]

    pose, _ = estimate_kart_pose(frame, KartPoseConfig())

    assert pose is not None
    assert pose.center_x == pytest.approx(120.0, abs=2.0)
    assert pose.center_y == pytest.approx(100.0, abs=2.0)
    assert pose.heading_angle_deg is not None
    error = abs(signed_axial_difference_deg(pose.heading_angle_deg, 30.0))
    assert error < 5.0
    assert pose.heading_quality > 0.5
    assert pose.red_fraction > 0.8


def test_kart_pose_rejects_orange_only_ui_candidate() -> None:
    hsv = np.zeros((200, 240, 3), dtype=np.uint8)
    hsv[:, :] = (95, 120, 230)
    hsv[90:110, 105:135] = (25, 220, 220)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    pose, _ = estimate_kart_pose(frame, KartPoseConfig())

    assert pose is None


def test_kart_road_relation_recovers_local_offset_and_heading_error() -> None:
    road_mask = np.zeros((200, 240), dtype=np.uint8)
    road_mask[80:120, 20:220] = 255
    pose = KartPose(
        center_x=120.0,
        center_y=110.0,
        center_x_norm=120.0 / 239.0,
        center_y_norm=110.0 / 199.0,
        heading_angle_deg=10.0,
        heading_quality=0.8,
        area_fraction=0.002,
        red_fraction=0.6,
        anchor_distance_norm=0.05,
        bbox_x=100,
        bbox_y=100,
        bbox_width=40,
        bbox_height=20,
    )

    relation = estimate_kart_road_relation(
        road_mask,
        _geometry(0.0),
        pose,
        KartRoadRelationConfig(
            tangent_offsets_fraction=(-0.10, 0.10),
            max_normal_distance_fraction=0.4,
            min_road_width_fraction=0.02,
            max_road_width_fraction=0.30,
        ),
    )

    assert relation is not None
    assert relation.heading_error_deg == pytest.approx(10.0)
    assert relation.abs_heading_error_deg == pytest.approx(10.0)
    assert relation.local_road_usable
    assert relation.local_road_center_y == pytest.approx(99.5, abs=1.0)
    assert relation.local_road_width_px == pytest.approx(40.0, abs=2.0)
    assert relation.lateral_offset_norm == pytest.approx(0.525, abs=0.08)
    assert relation.inside_road is True
    assert relation.valid_cross_sections == 2


def test_signed_axial_difference_wraps_half_turn() -> None:
    assert signed_axial_difference_deg(10.0, 170.0) == pytest.approx(20.0)
    assert signed_axial_difference_deg(170.0, 10.0) == pytest.approx(-20.0)
