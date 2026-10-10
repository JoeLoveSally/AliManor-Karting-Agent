"""Pure-CPU and no-video-data tests for matched visual ablation."""
from collections.abc import Mapping

import numpy as np
import pytest
from karting_agent.train.visual_ablation import make_mode_images


class ForbiddenCache(Mapping):
    def __getitem__(self, key):
        raise AssertionError("image cache accessed by action-only ablation")
    def __iter__(self):
        raise AssertionError("image cache iterated")
    def __len__(self):
        raise AssertionError("image cache length queried")


def test_action_only_is_zero_and_does_not_access_images():
    images = make_mode_images(range(5), ForbiddenCache(), mode="action_only")
    assert images.shape == (5, 3, 96, 96)
    assert images.dtype == np.float32
    assert not np.any(images)


def test_action_only_requires_no_video_cache():
    assert make_mode_images(
        [12, 22], None, mode="action_only", frame_size=16
    ).shape == (2, 3, 16, 16)


def test_rgb_rgb_hsv_identical_rgb_and_keep_hsv_distinct():
    frames = {
        i: np.random.default_rng(i).integers(
            0, 256, (4, 16, 16), dtype=np.uint8
        ) for i in range(5)
    }
    rgb = make_mode_images(range(5), frames, mode="rgb", frame_size=16)
    hsv = make_mode_images(range(5), frames, mode="rgb_hsv", frame_size=16)
    assert rgb.shape == (5, 3, 16, 16)
    assert hsv.shape == (5, 4, 16, 16)
    assert np.array_equal(rgb, hsv[:, :3])
    assert hsv.dtype == np.float32


def test_source_frame_cache_unchanged():
    data = np.arange(4 * 8 * 8, dtype=np.uint8).reshape(4, 8, 8)
    original = data.copy()
    _ = make_mode_images([12], {12: data}, mode="rgb_hsv", frame_size=8)
    assert np.array_equal(data, original)


@pytest.mark.parametrize("mode", ["action_only", "rgb", "rgb_hsv"])
def test_time_axis_and_channel_count(mode):
    frames = {i: np.ones((4, 16, 16), dtype=np.uint8) for i in range(5)}
    x = make_mode_images([0, 1, 2, 3, 4], frames, mode=mode, frame_size=16)
    assert x.shape == (5, 4 if mode == "rgb_hsv" else 3, 16, 16)


@pytest.mark.parametrize("kwargs", [
    {"mode": "invalid"},
    {"mode": "rgb", "frame_size": 0},
    {"mode": "action_only", "frame_indices": []},
])
def test_invalid_mode_or_size(kwargs):
    args = {
        "frame_indices": [0, 1],
        "frames_by_index": None,
        "mode": "action_only",
    }
    args.update(kwargs)
    with pytest.raises(ValueError):
        make_mode_images(**args)


def test_rgb_mode_does_not_silently_accept_missing_or_wrong_cache():
    with pytest.raises(ValueError, match="requires"):
        make_mode_images([0], None, mode="rgb")
    with pytest.raises(ValueError, match="uint8"):
        make_mode_images([0], {0: np.zeros((3, 96, 96), np.uint8)}, mode="rgb")
