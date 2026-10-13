"""Diagnostic kart-anchored texture/edge road evidence (no learned labels).

This is a *soft candidate representation*, not a semantic road mask or an
action policy. It is theme-agnostic by construction but not proven to work on
every theme. Use the offline audit before attempting training.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _clip01(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _disk(shape: tuple[int, int], xy: tuple[float, float], radius: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return (xx - xy[0]) ** 2 + (yy - xy[1]) ** 2 <= radius ** 2


def _axial_difference_degrees(left: np.ndarray | float, right: float) -> np.ndarray:
    return np.abs((np.asarray(left) - right + 90.0) % 180.0 - 90.0)


def texture_evidence(frame_bgr: np.ndarray) -> np.ndarray:
    """Local multiscale contrast: responds to tiles, grids and repeated edges.

    A local contrast score is *not* necessarily a road probability.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("expected BGR HxWx3")
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lightness = cv2.GaussianBlur(lab[:, :, 0].astype(np.float32), (0, 0), 1.0)
    mid = cv2.GaussianBlur(lightness, (0, 0), 4.0)
    broad = cv2.GaussianBlur(lightness, (0, 0), 12.0)
    band_small = cv2.boxFilter(
        np.abs(lightness - mid), cv2.CV_32F, (25, 25), normalize=True
    )
    band_large = cv2.boxFilter(
        np.abs(mid - broad), cv2.CV_32F, (37, 37), normalize=True
    )
    return _clip01(0.65 * band_small / 4.0 + 0.35 * band_large / 6.0)


def line_geometry_evidence(
    frame_bgr: np.ndarray,
    xy: tuple[float, float],
    *,
    vicinity_px: float = 190.0,
) -> tuple[np.ndarray, dict]:
    """Orientation agreement with long line evidence near the kart.

    Do not force two directions: some themes use triangles or non-square
    pavers. The output is a diagnostic edge channel, not a quadrilateral fit.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 35, 105)
    found = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=18,
        minLineLength=16, maxLineGap=5,
    )
    families: list[tuple[float, float]] = []
    if found is not None:
        for x1, y1, x2, y2 in found.reshape(-1, 4):
            midpoint = ((int(x1) + int(x2)) * .5, (int(y1) + int(y2)) * .5)
            d = math.hypot(midpoint[0] - xy[0], midpoint[1] - xy[1])
            # The kart itself has strong straight edges. Do not learn line
            # directions from the chassis/wheels immediately at the anchor.
            if d > vicinity_px or d < 55.0:
                continue
            length = math.hypot(int(x2) - int(x1), int(y2) - int(y1))
            if length < 16:
                continue
            direction = math.degrees(
                math.atan2(int(y2) - int(y1), int(x2) - int(x1))
            ) % 180.0
            families.append((direction, length / (1.0 + d / vicinity_px)))
    if not families:
        return np.zeros((h, w), np.float32), {
            "line_count_near_kart": 0, "dominant_axes_deg": [],
        }
    histogram = np.zeros(36, np.float64)
    for direction, weight in families:
        histogram[int(direction // 5) % 36] += weight
    peaks = []
    pool = histogram.copy()
    for _ in range(2):
        index = int(np.argmax(pool))
        if pool[index] <= 0.0:
            break
        angle = (index + 0.5) * 5.0
        peaks.append(angle)
        for other in range(36):
            if _axial_difference_degrees((other + .5) * 5.0, angle) < 25.0:
                pool[other] = 0
    gray_float = gray.astype(np.float32)
    gx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    tangents = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0
    alignment = np.zeros_like(gradient)
    for direction in peaks:
        difference = _axial_difference_degrees(tangents, direction)
        alignment = np.maximum(
            alignment, np.exp(-.5 * (difference / 18.0) ** 2)
        )
    supported = _clip01(gradient / 75.0) * alignment
    # Spread aligned edge evidence across tiles, but avoid synthesizing a line
    # or claiming explicit connectivity across a corner.
    score = cv2.GaussianBlur(supported, (0, 0), 11.0)
    score = _clip01(score / 0.12)
    return score, {
        "line_count_near_kart": len(families),
        "dominant_axes_deg": [round(x, 1) for x in peaks],
    }


def anchored_soft_candidate(
    texture: np.ndarray,
    geometry: np.ndarray,
    xy: tuple[float, float] | None,
    *,
    minimum_support_pixels: int = 12,
) -> tuple[np.ndarray, dict]:
    """Pick nearby evidence, without automatically connecting remote islands.

    A weak or missing seed produces an invalid sample and all-zero candidate.
    The gate does NOT claim that the component is the road.
    """
    if texture.ndim != 2 or texture.shape != geometry.shape:
        raise ValueError("texture/geometry shape mismatch")
    score = _clip01(cv2.GaussianBlur(
        0.75 * texture + 0.25 * geometry, (0, 0), 5.0
    ))
    invalid = np.zeros_like(score)
    if xy is None or not all(map(math.isfinite, xy)):
        return invalid, {"candidate_valid_unreviewed": False,
                         "seed_support_pixels": 0, "reason": "kart_missing"}
    evidence = np.where(score >= 0.19, 255, 0).astype(np.uint8)
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    evidence = cv2.morphologyEx(evidence, cv2.MORPH_CLOSE, close)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(evidence, 8)
    # Exclude the sprite and its immediate texture halo when seeding road.
    # A large gap is safer than choosing a bright kart as a "road" component.
    ring = _disk(score.shape, xy, 135.0) & ~_disk(score.shape, xy, 65.0)
    counts = np.bincount(labels[ring & (labels > 0)], minlength=count)
    if counts.size <= 1:
        return invalid, {"candidate_valid_unreviewed": False,
                         "seed_support_pixels": 0, "reason": "no_texture_seed"}
    counts[0] = 0
    winner = int(np.argmax(counts))
    support = int(counts[winner])
    if support < minimum_support_pixels:
        return invalid, {"candidate_valid_unreviewed": False,
                         "seed_support_pixels": support, "reason": "weak_texture_seed"}
    chosen = np.where(labels == winner, 255, 0).astype(np.uint8)
    dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    selected = cv2.dilate(chosen, dilation) > 0
    candidate = np.where(selected, score, 0.0).astype(np.float32)
    return candidate, {
        "candidate_valid_unreviewed": True,
        "seed_support_pixels": support, "reason": "near_kart_texture_component",
        "selected_evidence_area_fraction": float(
            stats[winner, cv2.CC_STAT_AREA] / score.size
        ),
    }


def quantify_texture_frame(
    frame_bgr: np.ndarray,
    kart_xy: tuple[float, float] | None,
    *,
    redaction_mask: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Compute diagnostic planes and explicit unknown mask.

    Input is the normalized 360x800 BGR frame. A redacted region is masked
    with a margin covering the maximum spatial filter footprint.
    """
    h, w = frame_bgr.shape[:2]
    if redaction_mask is not None and (
        redaction_mask.shape != (h, w) or redaction_mask.dtype != np.uint8
    ):
        raise ValueError("redaction mask must be uint8 HxW")
    texture = texture_evidence(frame_bgr)
    anchor = kart_xy if kart_xy is not None else (.55 * w, .5 * h)
    geometry, geometry_info = line_geometry_evidence(frame_bgr, anchor)
    candidate, seed_info = anchored_soft_candidate(texture, geometry, kart_xy)

    unknown = np.zeros((h, w), dtype=np.uint8)
    if kart_xy is not None:
        unknown[_disk((h, w), kart_xy, 55.0)] = 255
    if redaction_mask is not None:
        # Bleed from filters around action UI is conservatively excluded.
        guard = cv2.dilate(
            redaction_mask, cv2.getStructuringElement(
                cv2.MORPH_RECT, (61, 61)
            )
        )
        unknown = np.maximum(unknown, guard)
    for plane in (texture, geometry, candidate):
        plane[unknown > 0] = 0.0
    # Missing kart/texture evidence means "unverified", not a negative road
    # observation. Preserve diagnostic maps but invalidate all policy pixels.
    if not seed_info["candidate_valid_unreviewed"]:
        unknown[:] = 255
        candidate[:] = 0
    maps = {
        "texture": texture, "geometry": geometry,
        "candidate": candidate, "unknown": unknown,
    }
    return maps, {
        **seed_info, **geometry_info,
        "unknown_fraction": float(np.count_nonzero(unknown) / unknown.size),
        "texture_fraction_gt_0_2": float(np.mean(texture > 0.2)),
        "candidate_fraction_gt_0_2": float(np.mean(candidate > 0.2)),
        "advisory": (
            "Soft feature strength only; no independent ROAD/BACKGROUND truth."
        ),
    }
