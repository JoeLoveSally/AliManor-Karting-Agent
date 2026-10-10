"""Regression: canonical teacher batch-16 predictions are not recomputed in batch-2."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from karting_agent.train.shadow_replay import replay_fixed_video
from karting_agent.train.shadow_teacher import canonical_teacher_probabilities


class BatchSensitiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, images, controls):
        self.batches.append(int(images.shape[0]))
        value = 1.0 if images.shape[0] == 16 else -1.0
        return torch.ones(images.shape[0]) * value


class Samples(Dataset):
    def __init__(self):
        self.samples = [
            SimpleNamespace(
                video="validation_A.mp4" if i < 19 else "validation_B.mp4",
                input_timestamps_ms=tuple(200.0 + i * 50 + j * 50 for j in range(5)),
                target_timestamp_ms=500.0 + i * 50,
                current_pressed=False,
                target_pressed=False,
            ) for i in range(37)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return (
            torch.ones(5, 3, 16, 16, dtype=torch.float32),
            torch.zeros(5, 3, dtype=torch.float32),
            torch.tensor(0., dtype=torch.float32),
        )


def expert_features(stamps, events):
    return np.zeros((5, 3), dtype=np.float32)


def images(sample):
    return np.ones((5, 3, 16, 16), dtype=np.float32)


def test_shadow_uses_frozen_teacher_batch16_and_sequential_self_feedback():
    model = BatchSensitiveModel()
    data = Samples()
    refs = canonical_teacher_probabilities(model, data, batch_size=16)
    assert model.batches == [16, 16, 5]
    video_rows = data.samples[:5]
    replay = replay_fixed_video(
        model, video_rows, frames_for_sample=images,
        expert_events=(SimpleNamespace(timestamp_ms=0., pressed=False),),
        expert_features=expert_features, teacher_probabilities=refs,
    )
    assert model.batches == [16, 16, 5, 1, 1, 1, 1, 1]
    assert all(row["teacher_probability"] > .5 for row in replay["rows"])
    assert all(row["shadow_probability"] < .5 for row in replay["rows"])
    assert all(not row["simulated_current_pressed"] for row in replay["rows"])


def test_shadow_canonical_missing_prediction_is_rejected():
    with pytest.raises(ValueError, match="missing canonical teacher"):
        replay_fixed_video(
            BatchSensitiveModel(), Samples().samples[:1],
            frames_for_sample=images,
            expert_events=(SimpleNamespace(timestamp_ms=0., pressed=False),),
            expert_features=expert_features,
            teacher_probabilities={},
        )


def test_shadow_canonical_invalid_probability_is_rejected():
    sample = Samples().samples[0]
    with pytest.raises(ValueError, match="invalid canonical teacher probability"):
        replay_fixed_video(
            BatchSensitiveModel(), [sample],
            frames_for_sample=images,
            expert_events=(SimpleNamespace(timestamp_ms=0., pressed=False),),
            expert_features=expert_features,
            teacher_probabilities={(sample.video, sample.target_timestamp_ms): float("nan")},
        )
