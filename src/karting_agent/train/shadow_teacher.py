"""Reproduce the original Validation DataLoader inference batching exactly.

The frozen Tiny policy was selected from its full-video Validation DataLoader,
not from sequential paired expert/self-fed samples. The canonical teacher
reference therefore uses the SAME dataset order and batch size.
"""
from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader


def canonical_teacher_probabilities(model, dataset, *, batch_size: int) -> dict[tuple[str, float], float]:
    """Return p(PRESS at t+100ms) keyed by video and future target time."""
    if batch_size < 1 or not getattr(dataset, "samples", None):
        raise ValueError("invalid teacher dataset or reference batch size")
    model.eval()
    predictions: list[float] = []
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    with torch.inference_mode():
        for images, controls, _target in loader:
            probs = torch.sigmoid(model(images, controls)).cpu().tolist()
            if not isinstance(probs, list):
                raise ValueError("teacher model must produce one logit per sample")
            predictions.extend(float(p) for p in probs)
    if len(predictions) != len(dataset.samples):
        raise ValueError("reference inference/sample count mismatch")
    result: dict[tuple[str, float], float] = {}
    for sample, prob in zip(dataset.samples, predictions):
        key = (sample.video, float(sample.target_timestamp_ms))
        if key in result:
            raise ValueError("duplicate teacher target timestamp")
        if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
            raise ValueError("invalid teacher prediction")
        result[key] = prob
    return result
