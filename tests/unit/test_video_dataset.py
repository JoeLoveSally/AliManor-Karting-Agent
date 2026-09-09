import numpy as np

from karting_agent.train.dataset import DatasetSample
from karting_agent.train.video_dataset import TemporalVideoDataset
from karting_agent.vision.preprocess import PreprocessConfig


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.position = 0
        self.opened = True

    def isOpened(self) -> bool:
        return self.opened

    def set(self, _property: int, value: float) -> bool:
        self.position = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()

    def release(self) -> None:
        self.opened = False


def test_temporal_video_dataset_reads_requested_stack(monkeypatch, tmp_path) -> None:
    frames = [
        np.full((4, 4, 3), fill_value=index * 10, dtype=np.uint8)
        for index in range(6)
    ]
    capture = FakeCapture(frames)

    monkeypatch.setattr(
        "karting_agent.train.video_dataset.cv2.VideoCapture",
        lambda _path: capture,
    )

    sample = DatasetSample(
        video="data/raw/test.mp4",
        input_frame_indices=(1, 3, 5),
        input_timestamps_ms=(0.0, 50.0, 100.0),
        target_frame_index=5,
        target_timestamp_ms=200.0,
        target_pressed=True,
        transition_distance_ms=0.0,
        near_transition=True,
        near_short_correction=False,
    )
    preprocess = PreprocessConfig(
        input_width=4,
        input_height=4,
        mask_touch_area=False,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    dataset = TemporalVideoDataset(
        [sample],
        project_root=tmp_path,
        preprocess_config=preprocess,
    )

    item = dataset[0]

    assert item["input"].shape == (9, 4, 4)
    assert item["input"].dtype == np.float32
    assert np.allclose(item["input"][0:3], 10 / 255.0)
    assert np.allclose(item["input"][3:6], 30 / 255.0)
    assert np.allclose(item["input"][6:9], 50 / 255.0)
    assert item["target"] == np.float32(1.0)
    assert item["near_transition"] is True

    dataset.close()
    assert capture.opened is False
