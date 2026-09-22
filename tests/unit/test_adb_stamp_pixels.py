from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    from diagnose_adb_stamp_pixels import inspect_pixels  # noqa: E402
    from diagnose_adb_video_frame_age import SYNC  # noqa: E402
finally:
    sys.path.remove(str(SCRIPTS))


def bars(sequence: int, *, width: int = 360, height: int = 800) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    value = (SYNC << 16) | sequence
    for bit in range(24):
        if value & (1 << (23-bit)):
            x0 = round(bit * width / 24)
            x1 = round((bit + 1) * width / 24)
            image[:, x0:x1] = 255
    return image


def test_numeric_inspector_decodes_fullscreen_stamp() -> None:
    report = inspect_pixels(bars(2048))
    assert report['existing_decoder_sequence'] == 2048
    assert report['width'] == 360
    assert all(row['sync_matches'] for row in report['rows'])
    assert all(len(row['brightness_24']) == 24 for row in report['rows'])


def test_numeric_inspector_can_locate_letterboxed_browser_content() -> None:
    image = bars(202)
    image[:320] = 125
    report = inspect_pixels(image)
    assert report['existing_decoder_sequence'] == 202
    assert report['rows'][0]['sync_matches'] is False
    assert report['rows'][-1]['sync_matches'] is True


def test_numeric_inspector_reports_nonprobe_screen_without_image_data() -> None:
    image = np.full((800, 360, 3), 128, dtype=np.uint8)
    report = inspect_pixels(image)
    assert report['existing_decoder_sequence'] is None
    assert not any(row['sync_matches'] for row in report['rows'])
    assert all(row['neutral_samples'] == 24 for row in report['rows'])
    assert 'image' not in report
