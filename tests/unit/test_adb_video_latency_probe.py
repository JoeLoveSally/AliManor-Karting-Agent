from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from karting_agent.runtime.video_latency_probe import (
    ScreenChangeGate,
    changed_pixel_fraction,
)


def image(value: int = 0) -> np.ndarray:
    return np.full((100, 100, 3), value, dtype=np.uint8)


def test_ignores_status_bar_and_detects_large_central_change() -> None:
    before = image()
    after = before.copy()
    after[:9, :, :] = 255
    after[-9:, :, :] = 255
    assert changed_pixel_fraction(before, after) == 0.0
    after[15:85, 15:85] = 255
    assert changed_pixel_fraction(before, after) > 0.7


def test_gate_requires_two_frames_and_returns_first_observation() -> None:
    detector = ScreenChangeGate(image(), consecutive_frames=2)
    changed = image(255)
    assert detector.observe(image=changed, observed_monotonic_ms=1100.0, frame_index=5) is None
    onset = detector.observe(image=changed, observed_monotonic_ms=1120.0, frame_index=6)
    assert onset is not None
    assert onset[0] == pytest.approx(1100.0)
    assert onset[1] == 5


def test_gate_resets_after_single_changed_frame() -> None:
    detector = ScreenChangeGate(image(), consecutive_frames=2)
    changed = image(255)
    assert detector.observe(image=changed, observed_monotonic_ms=10.0, frame_index=1) is None
    assert detector.observe(image=image(), observed_monotonic_ms=20.0, frame_index=2) is None
    assert detector.observe(image=changed, observed_monotonic_ms=30.0, frame_index=3) is None
    result = detector.observe(image=changed, observed_monotonic_ms=40.0, frame_index=4)
    assert result is not None and result[0] == pytest.approx(30.0)


def test_rejects_invalid_images_and_thresholds() -> None:
    with pytest.raises(ValueError):
        changed_pixel_fraction(image(), image()[:, :, 0])
    with pytest.raises(ValueError):
        changed_pixel_fraction(image().astype(np.float32), image())
    with pytest.raises(ValueError):
        ScreenChangeGate(image(), changed_fraction_threshold=0)
    with pytest.raises(ValueError):
        ScreenChangeGate(image(), consecutive_frames=0)


def test_latency_script_has_all_legacy_video_configuration_fields() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/diagnose_adb_video_latency.py"
    spec = importlib.util.spec_from_file_location("diagnose_adb_video_latency", script)
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(script.parent))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.parse_args([])
        for name in ("adb", "serial", "ffmpeg", "decode_width", "video_bit_rate", "video_warmup_seconds"):
            assert hasattr(args, name)
        assert args.trials == 6
        assert args.decode_width == 360
    finally:
        sys.path.remove(str(script.parent))
