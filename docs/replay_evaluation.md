# Replay Runtime Evaluation

## Purpose

Replay Runtime exists to verify the production-style causal data path independently of the training Dataset and Frame Cache:

```text
Recorded MP4
    ↓ sequential decode
VideoInput
    ↓
TemporalFrameBuffer
    ↓ causal t-100 / t-50 / t
Shared Vision Preprocess
    ↓
ModelRunner
    ↓
HysteresisController
    ↓
MockExecutor
    ↓
Replay JSON
```

Replay itself does not read `samples.jsonl`, Frame Cache, or Touch Marker labels.

## Frozen v1 controller

Validation threshold sweep selected:

```text
press_threshold   = 0.55
release_threshold = 0.45
```

This reduced extra transitions relative to raw `0.5` classification while preserving the strict `±50ms` Transition Recall and Short Correction Recall on Validation.

## Model warmup

CUDA cold start can make the first real inference much slower than steady-state inference. Runtime therefore performs dummy model predictions before entering the replay/control loop. Warmup latency is reported separately and is excluded from replay inference latency statistics.

Warmup is a startup operation only; it does not update model weights or controller state.

## Two replay timelines

Each Runtime step records:

```text
observation_timestamp_ms = t
prediction_target_timestamp_ms = t + prediction_horizon_ms
```

For the current model, `prediction_horizon_ms=100`.

Replay evaluation therefore distinguishes two different timelines.

### Target timeline

```text
controller state @ prediction_target_timestamp_ms
```

This evaluates the semantic question the model was trained to answer: what action should be active at `t+100ms`? Target-timeline metrics should be compared with the offline Sequence Evaluator.

### Observation timeline

```text
controller state @ observation_timestamp_ms
```

This represents when the current software pipeline emits the decision and `MockExecutor` receives it. It is not the final physical touch time, because camera capture, inference, serial transport, bleOTG, Bluetooth HID, and Android input latency are not yet included.

If a 100ms-ahead prediction is executed immediately with near-zero downstream latency, observation-timeline transitions are expected to appear roughly 100ms earlier than the demonstrated action. This does not by itself indicate a model error; it shows why `prediction_horizon_ms` must ultimately be aligned with measured end-to-end actuation latency.

## Performance metrics

Replay reports:

- source frames read
- Runtime inference steps
- controller state changes
- mean / p50 / p95 / max model inference latency
- fast-as-possible processing throughput
- whether a safety RELEASE was required at shutdown

`processing_fps` is offline throughput, not the control frequency. Runtime inference slots are still capped at `target_fps=30` according to source-video timestamps.

The main real-time compute requirement is that steady-state inference latency remains comfortably below the 30Hz frame budget:

```text
1000 / 30 = 33.3ms
```

## Replay-to-label evaluation

`python scripts/evaluate_replay.py ...` is a separate analysis step that may read labels after Replay has finished. This preserves the separation between Runtime and evaluation while allowing us to verify that the causal Runtime path reproduces the behavior seen in the offline Sequence Evaluator.

Evaluate both `±100ms` and strict `±50ms` tolerance before moving to live ADB control.
