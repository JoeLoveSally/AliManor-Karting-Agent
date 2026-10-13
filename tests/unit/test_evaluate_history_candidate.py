"""Offline candidate rollout tests with a deterministic fake policy."""
from pathlib import Path
import importlib.util
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "evaluate_history_candidate_test",
    ROOT / "scripts/evaluate_history_candidate.py",
)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class ReflectFeedback(torch.nn.Module):
    def forward(self, images, control):
        # Predict PRESS if the last executed state is RELEASE.
        return 4.0 * (1. - 2. * control[:, -1, 0])


def test_candidate_rollout_uses_own_executed_history_not_baseline():
    fake = ReflectFeedback()
    features = {i: np.zeros((3, 96, 96), dtype=np.uint8) for i in range(5)}
    records = [
        {"frame_index": j + 4, "observation_ms": float(200 + j * 50),
         "input_indices": [0, 1, 2, 3, 4],
         "input_timestamp_ms": [float(j * 50), float(50 + j * 50),
                                float(100 + j * 50), float(150 + j * 50),
                                float(200 + j * 50)],
         "models": {"rgb": {"target_ms": float(300 + j * 50)}}}
        for j in range(6)
    ]
    actual = [(0.0, False), (250.0, True), (400.0, False)]
    rows, events = subject.replay_candidate(fake, features, records, actual)
    assert len(rows) == 6
    assert rows[0]["rollout_prob"] > .5
    assert rows[1]["rollout_prob"] > .5
    assert rows[2]["rollout_prob"] < .5  # proposal at 300ms is now executed
    assert rows[2]["teacher_prob"] < .5
    assert events[1] == (300.0, True)
    assert events[2] == (400.0, False)


def test_compare_candidate_keeps_absolute_target_time():
    truth = [(0., False), (100., True), (500., False)]
    rows = [
        {"target_ms": 150., "teacher_prob": .9, "rollout_prob": .9},
        {"target_ms": 550., "teacher_prob": .1, "rollout_prob": .1},
    ]
    scores = subject.compare_run(
        rows, truth, [(0., False), (150., True), (550., False)],
        [(100., 500., 400.)], 0, 1000,
    )
    assert scores["rollout"]["state_at_target"]["samples"] == 2
    assert scores["rollout"]["release_events"]["matched"] == 1
