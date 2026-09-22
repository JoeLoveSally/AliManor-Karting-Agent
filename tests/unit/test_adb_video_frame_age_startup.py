from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    from diagnose_adb_video_frame_age import (  # noqa: E402
        StampBook,
        video_health,
        wait_for_browser_stamps,
    )
finally:
    sys.path.remove(str(SCRIPTS))


def test_browser_preflight_accepts_multiple_stamp_requests() -> None:
    book = StampBook()
    for _ in range(3):
        book.stamp()
    assert wait_for_browser_stamps(book, timeout_seconds=0.01) == 3


def test_browser_preflight_rejects_missing_requests() -> None:
    with pytest.raises(RuntimeError, match='only 0 time stamps'):
        wait_for_browser_stamps(StampBook(), timeout_seconds=0.01)


def test_capture_health_reports_started_processes_without_reading_live_stderr() -> None:
    fake = SimpleNamespace(
        started=True,
        decoded_frames=12,
        dropped_frames=1,
        _error=None,
        _recorder=SimpleNamespace(poll=lambda: None),
        _ffmpeg=SimpleNamespace(poll=lambda: 1),
    )
    assert video_health(fake) == {
        'video_started': True,
        'decoded_frames': 12,
        'dropped_frames': 1,
        'reader_error': None,
        'recorder_returncode': None,
        'ffmpeg_returncode': 1,
    }
