"""Coherent projections for V5 H0 state-conditioned switch outputs."""

from __future__ import annotations

import numpy as np


def desired_press_from_switch_pair(
    switch_if_release: np.ndarray,
    switch_if_press: np.ndarray,
) -> np.ndarray:
    """Fuse two conditioned H0 outputs into one latent desired-PRESS score."""

    release = np.asarray(switch_if_release, dtype=np.float64)
    press = np.asarray(switch_if_press, dtype=np.float64)
    if release.shape != press.shape:
        raise ValueError("switch pair shapes must match")
    if np.any(~np.isfinite(release)) or np.any(~np.isfinite(press)):
        raise ValueError("switch probabilities must be finite")
    if np.any((release < 0.0) | (release > 1.0)):
        raise ValueError("switch_if_release must be in [0, 1]")
    if np.any((press < 0.0) | (press > 1.0)):
        raise ValueError("switch_if_press must be in [0, 1]")
    return 0.5 * (release + (1.0 - press))


def project_desired_press_to_switch(
    desired_press: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project an absolute desired-PRESS score to coherent switch probabilities."""

    desired = np.asarray(desired_press, dtype=np.float64)
    if np.any(~np.isfinite(desired)):
        raise ValueError("desired_press must be finite")
    if np.any((desired < 0.0) | (desired > 1.0)):
        raise ValueError("desired_press must be in [0, 1]")
    return desired, 1.0 - desired
