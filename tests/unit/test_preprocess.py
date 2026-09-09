import numpy as np

from karting_agent.vision.preprocess import (
    PreprocessConfig,
    mask_touch_area,
    stack_frames,
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
