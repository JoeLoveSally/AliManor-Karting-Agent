import numpy as np

from karting_agent.vision.preprocess import (
    PreprocessConfig,
    mask_touch_area,
    normalize_prepared_frame,
    prepare_frame,
    preprocess_frame,
    stack_frames,
    stack_prepared_frames,
)


def test_touch_mask_is_unconditional_and_fixed() -> None:
    frame = np.full((100, 200, 3), 255, dtype=np.uint8)
    masked = mask_touch_area(frame, (0.75, 0.8, 1.0, 1.0))

    assert np.all(masked[80:100, 150:200] == 0)
    assert np.all(masked[:80, :150] == 255)


def test_three_frames_become_nine_channels() -> None:
    frames = [
        np.full((20, 30, 3), value, dtype=np.uint8)
        for value in (32, 64, 96)
    ]
    config = PreprocessConfig(
        input_width=16,
        input_height=12,
        mask_touch_area=False,
    )

    stacked = stack_frames(frames, config)

    assert stacked.shape == (9, 12, 16)
    assert stacked.dtype == np.float32


def test_prepared_cache_path_matches_direct_preprocess() -> None:
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    frame[..., 0] = 10
    frame[..., 1] = 20
    frame[..., 2] = 30
    config = PreprocessConfig(
        input_width=6,
        input_height=4,
        mask_touch_area=False,
    )

    prepared = prepare_frame(frame, config)
    direct = preprocess_frame(frame, config)
    from_cache = normalize_prepared_frame(prepared, config)

    assert prepared.shape == (4, 6, 3)
    assert prepared.dtype == np.uint8
    assert np.allclose(from_cache, direct)

    stacked = stack_prepared_frames([prepared, prepared, prepared], config)
    assert stacked.shape == (9, 4, 6)
