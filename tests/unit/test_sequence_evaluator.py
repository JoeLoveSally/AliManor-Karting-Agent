from karting_agent.train.sequence_evaluator import (
    ReleaseSegment,
    SequencePoint,
    Transition,
    combine_evaluations,
    evaluate_sequence,
)


def test_sequence_transition_matching_and_timing() -> None:
    points = [
        SequencePoint("video.mp4", 0.0, 0.9),
        SequencePoint("video.mp4", 50.0, 0.9),
        SequencePoint("video.mp4", 100.0, 0.1),
        SequencePoint("video.mp4", 150.0, 0.1),
        SequencePoint("video.mp4", 200.0, 0.9),
    ]
    expected = [
        Transition("video.mp4", 90.0, False),
        Transition("video.mp4", 210.0, True),
    ]

    result = evaluate_sequence(
        points,
        expected,
        [],
        threshold=0.5,
        tolerance_ms=20.0,
    ).summary()

    assert result["transition"]["all"]["f1"] == 1.0
    assert result["onset_timing_ms"]["release"]["mean_error_ms"] == 10.0
    assert result["onset_timing_ms"]["press"]["mean_error_ms"] == -10.0


def test_extra_chatter_is_counted_as_false_positive() -> None:
    points = [
        SequencePoint("video.mp4", 0.0, 0.9),
        SequencePoint("video.mp4", 50.0, 0.1),
        SequencePoint("video.mp4", 100.0, 0.9),
        SequencePoint("video.mp4", 150.0, 0.1),
        SequencePoint("video.mp4", 200.0, 0.1),
    ]
    expected = [Transition("video.mp4", 50.0, False)]

    result = evaluate_sequence(
        points,
        expected,
        [],
        tolerance_ms=20.0,
    ).summary()

    metrics = result["transition"]["all"]
    assert metrics["matched"] == 1
    assert metrics["predicted"] == 3
    assert metrics["false_positive"] == 2
    assert metrics["precision"] == 1 / 3


def test_short_correction_requires_both_boundaries() -> None:
    expected = [
        Transition("video.mp4", 100.0, False),
        Transition("video.mp4", 300.0, True),
    ]
    releases = [ReleaseSegment("video.mp4", 100.0, 300.0)]

    detected_points = [
        SequencePoint("video.mp4", 0.0, 0.9),
        SequencePoint("video.mp4", 100.0, 0.1),
        SequencePoint("video.mp4", 200.0, 0.1),
        SequencePoint("video.mp4", 300.0, 0.9),
        SequencePoint("video.mp4", 400.0, 0.9),
    ]
    missed_end_points = [
        SequencePoint("video.mp4", 0.0, 0.9),
        SequencePoint("video.mp4", 100.0, 0.1),
        SequencePoint("video.mp4", 200.0, 0.1),
        SequencePoint("video.mp4", 300.0, 0.1),
        SequencePoint("video.mp4", 400.0, 0.1),
    ]

    detected = evaluate_sequence(
        detected_points,
        expected,
        releases,
        tolerance_ms=20.0,
    ).summary()
    missed = evaluate_sequence(
        missed_end_points,
        expected,
        releases,
        tolerance_ms=20.0,
    ).summary()

    assert detected["release_segment_recall"]["short_100_300ms"]["recall"] == 1.0
    assert missed["release_segment_recall"]["short_100_300ms"]["recall"] == 0.0


def test_combined_evaluation_aggregates_videos_without_cross_matching() -> None:
    first = evaluate_sequence(
        [
            SequencePoint("a.mp4", 0.0, 0.9),
            SequencePoint("a.mp4", 100.0, 0.1),
        ],
        [Transition("a.mp4", 100.0, False)],
        [],
        tolerance_ms=10.0,
    )
    second = evaluate_sequence(
        [
            SequencePoint("b.mp4", 0.0, 0.1),
            SequencePoint("b.mp4", 100.0, 0.9),
        ],
        [Transition("b.mp4", 100.0, True)],
        [],
        tolerance_ms=10.0,
    )

    result = combine_evaluations([first, second]).summary()

    assert result["transition"]["all"]["ground_truth"] == 2
    assert result["transition"]["all"]["matched"] == 2
    assert result["transition"]["all"]["f1"] == 1.0
