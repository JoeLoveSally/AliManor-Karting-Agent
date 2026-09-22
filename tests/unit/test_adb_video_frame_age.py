from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    from diagnose_adb_video_frame_age import (  # noqa: E402
        SYNC,
        StampBook,
        decode_stamp,
        summarize_ages,
    )
finally:
    sys.path.remove(str(SCRIPTS))


def stamped_image(sequence: int, *, height: int = 800, width: int = 360) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    bits = (SYNC << 16) | sequence
    for index in range(24):
        if bits & (1 << (23 - index)):
            left = round(index * width / 24)
            right = round((index + 1) * width / 24)
            frame[:, left:right] = 255
    return frame


@pytest.mark.parametrize('sequence', [0, 1, 255, 256, 32768, 65535])
def test_decode_stamp_from_360px_frame(sequence: int) -> None:
    assert decode_stamp(stamped_image(sequence)) == sequence


def test_decode_stamp_rejects_unrelated_page_and_unreadable_bars() -> None:
    assert decode_stamp(np.zeros((800, 360, 3), dtype=np.uint8)) is None
    assert decode_stamp(np.full((800, 360, 3), 128, dtype=np.uint8)) is None
    assert decode_stamp(stamped_image(123, width=200)) is None


def test_stamp_book_records_host_timestamps() -> None:
    book = StampBook()
    sequence = book.stamp()
    assert sequence == 1
    assert book.get(sequence) is not None
    assert book.get(999) is None


def test_summary_requires_samples_and_reports_distribution() -> None:
    with pytest.raises(ValueError, match='no unique timestamps'):
        summarize_ages([])
    summary = summarize_ages([10, 20, 30, 40])
    assert summary['unique_stamps'] == 4
    assert summary['median_ms'] == 25.0
    assert summary['max_ms'] == 40.0
