"""CPU-only guards for expert temporal policy training and timing."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from karting_agent.model.tiny_temporal_policy import TinyTemporalPolicy  # noqa: E402
from karting_agent.train.labels.touch_marker import ActionEvent  # noqa: E402
from karting_agent.vision.preprocess import PreprocessConfig  # noqa: E402
from inspect_geometry_pseudo_labels import load_config as load_road  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "_temporal_policy_experiment_test",
    ROOT / "scripts/experiment_tiny_temporal_policy.py",
)
assert spec and spec.loader
subject = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subject
spec.loader.exec_module(subject)


def actions():
    return (
        ActionEvent(0, 0.0, False),
        ActionEvent(10, 120.0, True),
        ActionEvent(20, 230.0, False),
    )


def sample():
    return SimpleNamespace(
        video="data/raw/synthetic.mp4",
        input_frame_indices=(0, 1, 2, 3, 4),
        input_timestamps_ms=(0., 50., 100., 150., 200.),
        current_pressed=True,
        target_timestamp_ms=300.0,
        target_timestamps_ms=(300., 400., 500.),
        target_pressed=False,
        target_pressed_by_horizon=(False, False, False),
    )


def test_model_rgb_and_rgb_hsv_have_same_output_semantics():
    for channels in (3, 4):
        model = TinyTemporalPolicy(image_channels=channels)
        images = torch.randn((2, 5, channels, 96, 96))
        control = torch.zeros((2, 5, 3))
        logits = model(images, control)
        assert logits.shape == (2,)
        logits.sum().backward()
        assert model.gru.weight_ih_l0.grad is not None


def test_model_rejects_inconsistent_visual_and_control_axes():
    model = TinyTemporalPolicy(image_channels=3)
    with pytest.raises(ValueError, match="control_features"):
        model(torch.zeros((1, 5, 3, 96, 96)),
              torch.zeros((1, 4, 3)))


def test_control_history_is_strictly_causal_and_age_resets_on_event():
    state = subject.observed_state_features(
        (0., 50., 100., 150., 200.), actions()
    )
    assert state.shape == (5, 3)
    assert state[:, 0].tolist() == [0, 0, 0, 1, 1]
    assert state[3, 1] == pytest.approx(30/500)
    assert state[4, 1] == pytest.approx(80/500)
    assert state[1, 2] == pytest.approx(.25)


def test_sample_must_match_expert_action_and_100ms_future_label():
    row = sample()
    assert subject.validate_sample(row, actions()) is False
    bad = SimpleNamespace(**vars(row))
    bad.current_pressed = False
    with pytest.raises(ValueError, match="expert action state"):
        subject.validate_sample(bad, actions())
    bad = SimpleNamespace(**vars(row))
    bad.target_timestamp_ms = 350.0
    with pytest.raises(ValueError, match="inconsistent observation"):
        subject.validate_sample(bad, actions())
    bad = SimpleNamespace(**vars(row))
    bad.target_pressed = True
    with pytest.raises(ValueError, match="future target disagreement"):
        subject.validate_sample(bad, actions())


def test_sequence_points_use_target_time_not_observation_time():
    row = sample()
    point = subject.points_from_predictions([row], [.77])[0]
    baseline = subject.points_from_predictions(
        [row], [.77], persistence=True,
    )[0]
    assert point.timestamp_ms == 300.
    assert point.probability == pytest.approx(.77)
    assert baseline.timestamp_ms == 300.
    assert baseline.probability == 1.


def test_predictions_reject_mixed_videos_and_duplicate_targets():
    one = sample()
    two = SimpleNamespace(**vars(one))
    two.video = "data/raw/another.mp4"
    with pytest.raises(ValueError, match="mixed video"):
        subject.points_from_predictions([one, two], [.5, .6])
    with pytest.raises(ValueError, match="duplicate"):
        subject.points_from_predictions([one, one], [.5, .6])


def test_rgb_hsv_share_identical_rgb_pixels_and_mask_expert_touch():
    cfg = PreprocessConfig(input_width=96, input_height=96)
    road, _, _ = load_road(ROOT/"configs/geometry_pseudo_labels.yaml")
    image = np.full((800, 360, 3), (180, 140, 50), dtype=np.uint8)
    changed = image.copy()
    changed[660:780, 280:356, :] = (2, 240, 250)
    features_a = subject.prepare_video_feature(
        image, preprocess=cfg, road_config=road,
    )
    features_b = subject.prepare_video_feature(
        changed, preprocess=cfg, road_config=road,
    )
    assert features_a.shape == (4, 96, 96)
    assert features_a.dtype == np.uint8
    assert np.array_equal(features_a, features_b)
    assert np.array_equal(features_a[:3], features_a[:3])


def test_dataset_training_target_is_expert_future_absolute_state():
    features = {"data/raw/synthetic.mp4": {
        i: np.full((4, 96, 96), i*10, np.uint8) for i in range(5)
    }}
    dataset = subject.TemporalExpertDataset(
        [sample()], features, {"data/raw/synthetic.mp4": actions()},
        mode="rgb_hsv",
    )
    images, state, target = dataset[0]
    assert images.shape == (5, 4, 96, 96)
    assert state.shape == (5, 3)
    assert target.item() == 0.0
    assert images[4, 0, 2, 2].item() == pytest.approx(40/255)
    rgb_only = subject.TemporalExpertDataset(
        [sample()], features, {"data/raw/synthetic.mp4": actions()}, mode="rgb",
    )
    assert rgb_only[0][0].shape == (5, 3, 96, 96)
    assert torch.equal(images[:, :3], rgb_only[0][0])


def test_loss_smoke_cpu_trains_without_any_controller():
    model = TinyTemporalPolicy(image_channels=3)
    frames = torch.zeros((2,5,3,32,32))
    states = torch.zeros((2,5,3))
    labels = torch.tensor([0.,1.])
    before = {key: value.clone() for key,value in model.state_dict().items()}
    loader = [(frames,states,labels)]
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
    result = subject.run_epoch(model,loader,torch.device("cpu"),optimizer)
    assert result["samples"] == 2
    assert np.isfinite(result["bce"])
    assert any(not torch.equal(before[key],value)
               for key,value in model.state_dict().items())


def test_source_dataset_reserves_test_video_split():
    from karting_agent.train.trainer import load_video_split
    split = load_video_split(ROOT/"configs/train_v4c2_temporal_v2.yaml")
    assert len(split.train) == 11
    assert len(split.validation) == 2
    assert len(split.test) == 2
    assert set(split.validation).isdisjoint(split.train)
