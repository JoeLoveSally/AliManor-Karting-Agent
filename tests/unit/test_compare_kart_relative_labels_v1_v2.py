from __future__ import annotations

from pathlib import Path

from scripts.compare_kart_relative_labels_v1_v2 import _split_map, _video_id


def test_video_id_accepts_paths_and_numeric_values() -> None:
    assert _video_id("data/raw/video_20260130_172959.mp4") == 172959
    assert _video_id(173545) == 173545
    assert _video_id("173835") == 173835


def test_split_map_accepts_path_entries(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text(
        """
split:
  train:
    - data/raw/video_20260130_172959.mp4
  validation:
    - data/raw/video_20260130_173545.mp4
  test:
    - data/raw/video_20260130_173835.mp4
""".lstrip(),
        encoding="utf-8",
    )

    assert _split_map(config) == {
        172959: "train",
        173545: "validation",
        173835: "test",
    }
