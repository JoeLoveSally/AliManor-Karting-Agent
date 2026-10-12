"""Train-only, reproducible action-history corruption guards."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from karting_agent.train.history_robustness import (
    HistoryCorruptionConfig, HistoryCorruptor,
)


def history():
    x = torch.zeros(8, 5, 3)
    x[:, :, 0] = torch.tensor([0, 0, 1, 1, 1])
    x[:, :, 1] = torch.tensor([.3, .4, 0, .1, .2])
    x[:, 1:, 2] = .25
    return x


def test_zero_probability_is_exact_noop():
    original = history()
    result = HistoryCorruptor(HistoryCorruptionConfig(0.0))(original)
    assert result is original
    assert torch.equal(result, original)


def test_full_probability_yields_coherent_causal_hold_without_changing_dt():
    original = history()
    out = HistoryCorruptor(HistoryCorruptionConfig(1.0), seed=2)(original)
    assert torch.equal(original[:, :, 2], out[:, :, 2])
    assert torch.equal(original[:, :, 0], history()[:, :, 0])
    for row in out:
        assert torch.all(row[1:, 0] == row[:-1, 0])
        assert torch.all(row[1:, 1] >= row[:-1, 1])
        assert torch.all((row[:, 1] >= 0) & (row[:, 1] <= 1))
        assert torch.all((row[:, 0] == 0) | (row[:, 0] == 1))


def test_seed_controls_reproducibility_and_input_is_unmodified():
    original = history()
    a = HistoryCorruptor(HistoryCorruptionConfig(.5), seed=19)(original)
    b = HistoryCorruptor(HistoryCorruptionConfig(.5), seed=19)(original)
    assert torch.equal(a, b)
    assert torch.equal(original, history())


def test_invalid_corruption_rejected():
    for prob in (-.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            HistoryCorruptionConfig(prob).validate()
    with pytest.raises(ValueError, match="expected"):
        HistoryCorruptor(HistoryCorruptionConfig(.5))(torch.zeros(5, 3))


def test_training_epoch_only_augmentation_smoke():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "tiny_temporal_experiment_robust_test",
        ROOT / "scripts/experiment_tiny_temporal_policy.py",
    )
    subject = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subject)
    from karting_agent.model.tiny_temporal_policy import TinyTemporalPolicy
    model = TinyTemporalPolicy(image_channels=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    images = torch.zeros(2, 5, 3, 32, 32)
    controls = history()[:2]
    target = torch.tensor([0., 1.])
    original = controls.clone()
    corrupter = HistoryCorruptor(HistoryCorruptionConfig(1.0), seed=1)
    result = subject.run_epoch(
        model, [(images, controls, target)], torch.device("cpu"), optimizer,
        history_corruptor=corrupter,
    )
    assert result["samples"] == 2
    assert torch.equal(controls, original)
    with pytest.raises(ValueError, match="validation"):
        subject.run_epoch(
            model, [(images, controls, target)], torch.device("cpu"),
            history_corruptor=corrupter,
        )
