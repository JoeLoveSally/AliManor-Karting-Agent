from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from diagnose_event_time_action_only import _audit_video  # noqa: E402
from karting_agent.runtime.event_time_policy import EventTimePolicyConfig  # noqa: E402
from karting_agent.train.sequence_evaluator import (  # noqa: E402
    ReleaseSegment,
    Transition,
)


def test_trace_matches_reference_and_identifies_pending_origin() -> None:
    video = "synthetic.mp4"
    timestamps = (1000.0, 1040.0, 1070.0, 1110.0, 1140.0)
    # Initial RELEASE -> PRESS at 1000, action-hold RELEASE warning at 1040
    # and 1070, then scheduled RELEASE at the 1100ms hold expiry.
    probabilities = np.asarray((0.9, 0.1, 0.1, 0.1, 0.1))
    samples = [
        SimpleNamespace(
            video=video,
            input_timestamps_ms=(timestamp,),
            current_pressed=False if i == 0 else True,
        )
        for i, timestamp in enumerate(timestamps)
    ]
    config = EventTimePolicyConfig(
        bin_ms=50.0,
        event_bins=6,
        no_event_class=6,
        action_threshold=0.5,
        min_state_hold_ms=100.0,
    )
    expected = [
        Transition(video=video, timestamp_ms=1000.0, pressed=True),
        Transition(video=video, timestamp_ms=1100.0, pressed=False),
    ]
    releases = [
        ReleaseSegment(video=video, start_ms=1100.0, end_ms=1300.0),
    ]

    predicted, missed, summary, reasons = _audit_video(
        video=video,
        samples=samples,
        indices=list(range(len(samples))),
        probabilities=probabilities,
        config=config,
        label_data=(expected, releases),
        tolerance_ms=100.0,
        window_ms=110.0,
    )

    assert len(predicted) == 2
    assert not missed
    assert summary["matched"] == 2
    assert summary["predicted"] == 2
    assert reasons["pending_execute"] == 1
    assert predicted[0]["reason"] == "action_mismatch"
    assert predicted[1]["reason"] == "pending_execute"
    assert predicted[1]["timestamp_ms"] == pytest.approx(1100.0)
    assert predicted[1]["pending_origin"] == {
        "timestamp_ms": 1070.0,
        "action_probability": 0.1,
        "due_at_ms": 1100.0,
        "desired_pressed": False,
    }
