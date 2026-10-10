"""Pure, read-only comparison of CPU and CUDA frozen teacher predictions."""
from __future__ import annotations

import math


def compare_backend_probabilities(
    cpu: dict[tuple[str, float], float],
    cuda: dict[tuple[str, float], float],
    *,
    threshold: float = 0.5,
    max_rows: int = 20,
) -> dict:
    """Identify genuinely backend-sensitive threshold crossings.

    No threshold selection: 0.5 is the frozen policy's fixed decision threshold.
    A changed crossing is diagnostic evidence, not grounds to retune Test.
    """
    if not 0.0 < threshold < 1.0 or max_rows < 1:
        raise ValueError("invalid frozen decision threshold or max_rows")
    if not cpu or set(cpu) != set(cuda):
        raise ValueError("CPU and CUDA prediction sets differ")
    rows = []
    total_near = 0
    max_difference = 0.0
    for (video, timestamp), a in cpu.items():
        b = cuda[(video, timestamp)]
        if (not math.isfinite(a) or not math.isfinite(b) or
                not 0.0 <= a <= 1.0 or not 0.0 <= b <= 1.0):
            raise ValueError("invalid predicted probability")
        difference = abs(a-b)
        max_difference = max(max_difference, difference)
        if min(abs(a-threshold), abs(b-threshold)) < 0.01:
            total_near += 1
        if (a >= threshold) != (b >= threshold):
            rows.append({
                "video": video,
                "target_ms": timestamp,
                "cpu_probability": a,
                "cuda_probability": b,
                "absolute_difference": difference,
            })
    rows.sort(key=lambda row: (row["video"], row["target_ms"]))
    return {
        "samples": len(cpu),
        "max_absolute_probability_difference": max_difference,
        "near_threshold_within_0p01": total_near,
        "disagreeing_binary_samples": len(rows),
        "threshold_crossings": rows[:max_rows],
        "threshold_crossings_truncated": len(rows) > max_rows,
    }
