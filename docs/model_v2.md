# Model v2: Multi-Horizon Temporal Control

## 1. Why v1 is being replaced

The v1 model uses three RGB frames spanning 100 ms and predicts one binary target:

```text
frames(t-100, t-50, t) -> action(t+100)
```

Closed-loop ADB runs and occlusion sensitivity exposed a shortcut problem. A single small region around the kart can move the predicted PRESS probability from one extreme to the other, while HUD regions also contribute to some decisions. The training objective does not require the network to model track geometry or an upcoming transition; because action persistence is high, recovering the current action and copying it forward by 100 ms is already a strong shortcut.

v2 changes the supervision before collecting more gameplay data.

## 2. v2 input

```text
frame(t-200ms)
frame(t-150ms)
frame(t-100ms)
frame(t-50ms)
frame(t)
      ↓
15-channel MobileNetV3-Small
```

Configuration:

```text
frame_stack       = 5
history_ms        = 200
frame_interval_ms = 50
input_size        = 224x224
```

The longer history is intended to expose lateral motion and heading trend rather than only the latest visual state.

## 3. v2 output

The classifier has three binary heads sharing one backbone:

```text
                 ┌─ action(t+100ms)
visual history ──┼─ action(t+200ms)
                 └─ action(t+300ms)
```

All three logits are trained jointly with BCE. This creates supervision such as:

```text
+100ms: PRESS
+200ms: RELEASE
+300ms: RELEASE
```

near an upcoming exit. A representation that only copies the current action is therefore penalized by the farther-horizon heads.

The initial runtime control head is `+200ms`. This is a configuration choice, not a permanent constant; the three heads must be compared offline and in closed loop before freezing it.

## 4. Fixed UI masks

Touch-marker masking remains enabled. v2 additionally masks two fixed HUD regions identified by the v1 focus diagnostic:

```text
[0.18, 0.03, 0.62, 0.14]
[0.72, 0.00, 1.00, 0.13]
```

Coordinates are normalized `(x0, y0, x1, y1)`.

The purpose is not to mask the kart or road. It removes timer/buttons that can correlate with race progress but are not causal steering evidence.

## 5. Data and cache isolation

v1 artifacts remain intact. v2 uses separate derived data:

```text
data/processed/v2/
  samples.jsonl
  manifest.json
  labels/

data/processed/frame_cache_v2/

artifacts/models/mobilenet_v3_small_v2/
```

Frame-cache metadata version 2 includes all fixed masks, so a preprocessing change invalidates stale cache automatically.

## 6. Build and smoke-test

```bash
python scripts/build_dataset.py \
  --config configs/train_v2.yaml \
  --output data/processed/v2

python scripts/build_frame_cache.py \
  --config configs/train_v2.yaml \
  --samples data/processed/v2/samples.jsonl \
  --force

python scripts/train_model_v2.py \
  --config configs/train_v2.yaml \
  --samples data/processed/v2/samples.jsonl \
  --require-cache \
  --smoke
```

Run the full unit suite before training:

```bash
python -m pytest
```

## 7. Full training

After the local smoke test passes, run full training on Spark using the same config and v2 cache. The artifact metadata records:

- `prediction_horizons_ms`
- `control_horizon_ms`
- `frame_stack`
- full preprocessing config including HUD masks

`ModelRunner` is backward-compatible with v1 and automatically loads multi-horizon v2 artifacts. `predict()` returns the configured control head; `predict_all()` exposes all three probabilities for diagnostics.

## 8. v2 acceptance criteria

v2 is not accepted solely because sample accuracy or F1 increases. Evaluation must include:

1. per-horizon validation/test F1;
2. transition timing on held-out videos;
3. persistence-baseline comparison at each horizon;
4. focus diagnostic showing reduced HUD dependence and meaningful sensitivity to track/kart relationship;
5. ADB replay/closed-loop completion quality.

The main question is whether v2 learns upcoming control transitions from visual geometry and motion, not whether it can reproduce the majority persistent state.

## 9. Post-training multi-horizon evaluation

Use the dedicated v2 evaluator after training. It evaluates the already-saved best checkpoint, so no retraining is required:

```bash
python scripts/evaluate_model_v2.py \
  --config configs/train_v2.yaml \
  --samples data/processed/v2/samples.jsonl \
  --labels-dir data/processed/v2/labels \
  --split test \
  --require-cache \
  --num-workers 4
```

For each `+100ms`, `+200ms`, and `+300ms` head it reports:

- ordinary sample F1;
- F1 on that horizon's own `near_transition` subset;
- F1 on that horizon's own `near_short_correction` subset;
- sequence-level transition F1 and onset timing;
- short-release-segment recall;
- the same classification/sequence metrics for the label-only persistence baseline `action(t) -> action(t+horizon)`.

This evaluator intentionally reads `near_transition_by_horizon` and `near_short_correction_by_horizon` from the v2 manifest instead of using the aggregate `any(horizon)` flags. Older training logs that print `transition_f1` and `short_f1` for the selected control head therefore should not be used as the final v2 subset metrics.

### Current held-out test result

The first v2 run selected epoch 2 by minimum validation BCE. On the held-out two-video test split:

| Horizon | Model F1 | Persistence F1 | Near-transition F1 | Persistence near-transition F1 | Sequence transition F1 | Short release recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| +100 ms | 0.970 | 0.917 | 0.909 | 0.746 | 0.920 | 0.889 |
| +200 ms | 0.961 | 0.843 | 0.885 | 0.517 | 0.889 | 0.667 |
| +300 ms | 0.950 | 0.788 | 0.857 | 0.551 | 0.718 | 0.389 |

The persistence gap grows with horizon, which is evidence that v2 is learning future information beyond simply copying the current action. However, `+300ms` loses too much sequence timing and short-correction quality. `+100ms` currently has the strongest sequence metrics, while `+200ms` has a much larger margin over persistence and remains a viable candidate.

## 10. Runtime replay comparison

Before real ADB control, compare `+100ms` and `+200ms` using the same held-out test videos and the real runtime hysteresis path. `replay_video.py` accepts `--control-horizon-ms` without modifying the artifact metadata.

Example for one test video:

```bash
python scripts/replay_video.py data/raw/video_20260130_173426.mp4 \
  --model artifacts/models/mobilenet_v3_small_v2/model.pt \
  --control-horizon-ms 100

python scripts/replay_video.py data/raw/video_20260130_173426.mp4 \
  --model artifacts/models/mobilenet_v3_small_v2/model.pt \
  --control-horizon-ms 200
```

Evaluate each replay against the v2 labels:

```bash
python scripts/evaluate_replay.py \
  artifacts/replays/mobilenet_v3_small_v2/video_20260130_173426_h100.json \
  --labels-dir data/processed/v2/labels

python scripts/evaluate_replay.py \
  artifacts/replays/mobilenet_v3_small_v2/video_20260130_173426_h200.json \
  --labels-dir data/processed/v2/labels
```

Repeat for `video_20260130_173835.mp4`. Compare both the target timeline and observation timeline. The target timeline measures whether the future-action semantics line up with labels; the observation timeline shows when the runtime actually emits the decision before actuator latency.

### Replay result and control-head decision

Across the two held-out videos, the RuntimeEngine + hysteresis replay produced:

```text
                           matched / GT    predicted    transition F1
+100ms target                 81 / 88          82           0.953
+200ms target                 72 / 88          74           0.889

+100ms observation            57 / 88          82           0.671
+200ms observation             2 / 88          74           0.025
```

On the target timeline, `+100ms` also detected 16/18 short 100-300 ms RELEASE segments, versus 12/18 for `+200ms`. On the observation timeline the corresponding counts were 10/18 and 0/18.

The near-zero `+200ms` observation score is partly mechanical: Runtime executes the selected future state immediately, so a correct `t+200ms` prediction is emitted about 200 ms before the recorded human action and therefore falls outside the 100 ms matching tolerance. Even after accounting for that, `+100ms` is stronger on the target timeline too: it has higher transition recall and short-correction recall on both held-out videos.

Decision: use `+100ms` as the next runtime control head. Keep `+200ms` and `+300ms` as auxiliary training heads; their value is to regularize the shared representation away from current-action persistence shortcuts, not necessarily to execute those heads directly.

Before armed ADB testing, run the focus diagnostic on the `+100ms` replay head and verify that dominant evidence is no longer HUD/action-state shortcut evidence.
