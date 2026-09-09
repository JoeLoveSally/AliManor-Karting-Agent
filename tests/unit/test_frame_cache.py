import numpy as np

from karting_agent.train.dataset import DatasetSample
from karting_agent.train.frame_cache import (
    build_frame_cache,
    cache_paths,
    read_cache_metadata,
)
from karting_agent.train.video_dataset import TemporalVideoDataset
from karting_agent.vision.preprocess import PreprocessConfig


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.position = 0
        self.opened = True

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()

    def release(self) -> None:
        self.opened = False


def make_sample() -> DatasetSample:
    return DatasetSample(
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


def test_frame_cache_build_and_dataset_read(monkeypatch, tmp_path) -> None:
    frames = [
        np.full((4, 4, 3), fill_value=index * 10, dtype=np.uint8)
        for index in range(6)
    ]
    capture = FakeCapture(frames)
    monkeypatch.setattr(
        "karting_agent.train.frame_cache.cv2.VideoCapture",
        lambda _path: capture,
    )

    sample = make_sample()
    config = PreprocessConfig(
        input_width=4,
        input_height=4,
        mask_touch_area=False,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    cache_root = tmp_path / "cache"

    summary = build_frame_cache(
        [sample],
        project_root=tmp_path,
        cache_root=cache_root,
        preprocess_config=config,
    )

    assert summary["videos"] == 1
    assert summary["frames"] == 3
    assert summary["built"] == 1
    assert capture.opened is False

    frames_path, index_path = cache_paths(cache_root, sample.video)
    cached = np.load(frames_path, mmap_mode="r")
    metadata = read_cache_metadata(index_path)

    assert cached.shape == (3, 4, 4, 3)
    assert cached.dtype == np.uint8
    assert metadata.frame_indices == (1, 3, 5)

    monkeypatch.setattr(
        "karting_agent.train.video_dataset.cv2.VideoCapture",
        lambda _path: (_ for _ in ()).throw(AssertionError("MP4 fallback used")),
    )
    dataset = TemporalVideoDataset(
        [sample],
        project_root=tmp_path,
        preprocess_config=config,
        cache_root=cache_root,
        require_cache=True,
    )

    item = dataset[0]

    assert item["input"].shape == (9, 4, 4)
    assert np.allclose(item["input"][0:3], 10 / 255.0)
    assert np.allclose(item["input"][3:6], 30 / 255.0)
    assert np.allclose(item["input"][6:9], 50 / 255.0)
    dataset.close()


def test_frame_cache_second_build_is_skipped(monkeypatch, tmp_path) -> None:
    frames = [
        np.full((4, 4, 3), fill_value=index * 10, dtype=np.uint8)
        for index in range(6)
    ]
    monkeypatch.setattr(
        "karting_agent.train.frame_cache.cv2.VideoCapture",
        lambda _path: FakeCapture(frames),
    )
    sample = make_sample()
    config = PreprocessConfig(
        input_width=4,
        input_height=4,
        mask_touch_area=False,
    )
    cache_root = tmp_path / "cache"

    build_frame_cache(
        [sample],
        project_root=tmp_path,
        cache_root=cache_root,
        preprocess_config=config,
    )

    monkeypatch.setattr(
        "karting_agent.train.frame_cache.cv2.VideoCapture",
        lambda _path: (_ for _ in ()).throw(AssertionError("cache rebuilt")),
    )
    summary = build_frame_cache(
        [sample],
        project_root=tmp_path,
        cache_root=cache_root,
        preprocess_config=config,
    )

    assert summary["built"] == 0
    assert summary["cached"] == 1
