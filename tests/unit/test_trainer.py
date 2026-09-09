from karting_agent.train.dataset import DatasetSample
from karting_agent.train.trainer import (
    SamplingConfig,
    build_sample_weights,
    sampling_summary,
    select_samples_by_videos,
)


def sample(
    video: str,
    *,
    near_transition: bool = False,
    near_short: bool = False,
) -> DatasetSample:
    return DatasetSample(
        video=video,
        input_frame_indices=(0, 1, 2),
        input_timestamps_ms=(0.0, 50.0, 100.0),
        target_frame_index=3,
        target_timestamp_ms=200.0,
        target_pressed=False,
        transition_distance_ms=0.0 if near_transition else None,
        near_transition=near_transition,
        near_short_correction=near_short,
    )


def test_sample_weights_prioritize_short_corrections() -> None:
    samples = [
        sample("a.mp4"),
        sample("a.mp4", near_transition=True),
        sample("a.mp4", near_transition=True, near_short=True),
    ]
    config = SamplingConfig(
        stable_weight=1.0,
        transition_weight=2.0,
        short_correction_weight=3.0,
    )

    assert build_sample_weights(samples, config) == [1.0, 2.0, 3.0]


def test_sampling_summary_reports_weighted_mix() -> None:
    samples = (
        [sample("a.mp4") for _ in range(7)]
        + [sample("a.mp4", near_transition=True) for _ in range(2)]
        + [sample("a.mp4", near_transition=True, near_short=True)]
    )
    summary = sampling_summary(samples, SamplingConfig())

    assert summary["samples"] == 10
    assert summary["stable_samples"] == 7
    assert summary["transition_samples"] == 2
    assert summary["short_correction_samples"] == 1
    assert abs(summary["stable_weighted_share"] - 0.5) < 1e-9
    assert abs(summary["transition_weighted_share"] - 2 / 7) < 1e-9
    assert abs(summary["short_correction_weighted_share"] - 3 / 14) < 1e-9


def test_select_samples_by_videos_uses_exact_video_ids() -> None:
    samples = [sample("a.mp4"), sample("b.mp4"), sample("c.mp4")]
    selected = select_samples_by_videos(samples, ["a.mp4", "c.mp4"])

    assert [item.video for item in selected] == ["a.mp4", "c.mp4"]
