from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from karting_agent.train import action_history_dataset as subject


class StubRgbDataset:
    def __init__(self, samples, **kwargs):
        assert kwargs["counterfactual_states"] is False
        self.samples = samples

    def __getitem__(self, index):
        return {"input": "rgb", "switch_target": [0., 1., 0.]}

    def close(self):
        pass


def make_sample(tmp_path, *, current=True):
    labels = tmp_path / "data/processed/v3/labels"
    labels.mkdir(parents=True)
    (labels / "one.json").write_text(json.dumps({
        "video": "data/raw/one.mp4",
        "events": [
            {"timestamp_ms": 0., "pressed": False},
            {"timestamp_ms": 120., "pressed": True},
        ],
    }))
    sample = SimpleNamespace(
        video="data/raw/one.mp4",
        input_timestamps_ms=(0., 50., 100., 150., 200.),
        current_pressed=current,
    )
    return labels, sample


def test_factual_dataset_never_duplicates_historical_states(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "StateConditionedVideoDataset", StubRgbDataset)
    labels, sample = make_sample(tmp_path)
    data = subject.ActionHistoryVideoDataset([sample], project_root=tmp_path, labels_dir=labels)
    assert len(data) == 1
    item = data[0]
    assert item["input"] == "rgb"
    assert item["action_history"].tolist()[::3][:5] == [0., 0., 0., 1., 1.]


def test_factual_dataset_rejects_bad_observed_state(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "StateConditionedVideoDataset", StubRgbDataset)
    labels, sample = make_sample(tmp_path, current=False)
    data = subject.ActionHistoryVideoDataset([sample], project_root=tmp_path, labels_dir=labels)
    with pytest.raises(ValueError, match="disagrees"):
        _ = data[0]
