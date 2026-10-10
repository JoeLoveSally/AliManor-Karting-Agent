"""Guard the original batch-16 teacher reference against input-order drift."""
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from karting_agent.train.shadow_teacher import canonical_teacher_probabilities


class TinyDataset(Dataset):
    def __init__(self, n=37):
        self.samples = [
            SimpleNamespace(
                video="valA" if i < 19 else "valB",
                target_timestamp_ms=i * 33.333,
            )
            for i in range(n)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return (
            torch.full((5, 3, 8, 8), float(i)),
            torch.full((5, 3), float(i)),
            torch.tensor(i % 2, dtype=torch.float32),
        )


class BatchSpy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, images, controls):
        self.batches.append(len(images))
        return controls[:, -1, 0] * 0 + (1 if len(images) == 16 else -1)


def test_teacher_uses_full_dataset_batch16_across_video_boundary():
    m = BatchSpy()
    probabilities = canonical_teacher_probabilities(
        m, TinyDataset(), batch_size=16
    )
    assert m.batches == [16, 16, 5]
    assert len(probabilities) == 37
    assert probabilities[("valA", 18 * 33.333)] == pytest.approx(
        torch.sigmoid(torch.tensor(1.)).item()
    )
    assert probabilities[("valB", 36 * 33.333)] == pytest.approx(
        torch.sigmoid(torch.tensor(-1.)).item()
    )


def test_teacher_rejects_duplicate_targets_and_invalid_batch():
    ds = TinyDataset(2)
    ds.samples[1].target_timestamp_ms = ds.samples[0].target_timestamp_ms
    with pytest.raises(ValueError, match="duplicate"):
        canonical_teacher_probabilities(BatchSpy(), ds, batch_size=16)
    with pytest.raises(ValueError, match="batch"):
        canonical_teacher_probabilities(BatchSpy(), TinyDataset(), batch_size=0)


def test_teacher_model_not_modified_and_order_is_stable():
    model = BatchSpy()
    data = TinyDataset()
    a = canonical_teacher_probabilities(model, data, batch_size=16)
    b = canonical_teacher_probabilities(model, data, batch_size=16)
    assert a == b
    assert not model.training
