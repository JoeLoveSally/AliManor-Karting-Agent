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
