# Model v4: Structured Model-Driven Policy

## 1. Goal

v4 keeps deployment fully model-driven while separating two questions:

1. temporal representation;
2. visual supervision that encourages causal road/kart geometry instead of shortcut features.

Analytic CV is offline teacher/debug tooling only. It never sends runtime actions.

## 2. v3 reference

v3 remains the best control baseline:

```text
5 RGB frames
→ concatenate as 15 channels
→ MobileNetV3-Small
→ visual feature
+ current PRESS / RELEASE embedding
→ KEEP / SWITCH
```

The first 15-channel convolution is only temporal-average-equivalent at initialization; it remains trainable. Temporal counterfactual tests confirmed learned frame-order use:

```text
repeat_latest flip rate = 0.351
reverse       flip rate = 0.622
```

Relevant v3 stateful reference at threshold `0.60`:

```text
target transition F1 = 0.918
matched              = 78 / 88
predicted            = 82
short recall         = 0.778
chatter <100ms       = 1
chatter <200ms       = 9
observation F1       = 0.753
```

## 3. Temporal ablation conclusion

The controlled v3 → v4-A → v4-A2 experiment kept the same data, video split, 200ms/5-frame history, targets, state conditioning, sampling and training semantics.

```text
v3 15-channel temporal CNN         → best overall baseline
v4-A per-frame CNN + GRU           → smoother but worse short corrections
v4-A2 per-frame CNN + temporal MLP → close to v3, still no net win
```

Representative best stateful results:

```text
                 v3@0.60   v4-A@0.50   v4-A2@0.70
transition F1      0.918       0.892        0.907
short recall       0.778       0.722        0.722
observation F1     0.753       0.554        0.744
```

Therefore temporal architecture search is paused. v4-C returns to the v3 control formulation and changes visual supervision only.

## 4. Track geometry model

The tracks are better modeled as piecewise-straight corridors with discrete corners than as continuously curved racing roads:

```text
straight segment
→ discrete corner
→ straight segment
→ discrete corner
```

The curved trajectories visible during drifting mostly describe vehicle motion rather than a continuously curved road centerline.

Geometry priorities are therefore:

```text
1. local straight-road orientation
2. later: visible corner state
3. later: kart center / heading
4. later: lateral offset and heading error
5. later: travel-relative next-corner distance/direction
```

Continuous curvature is not a primary target.

## 5. Geometry teacher findings

### 5.1 Multi-theme road mask

The first blue-only HSV teacher failed on several themes. A multi-theme candidate mask now covers blue, cyan/teal, green and dark-neutral road appearances.

The second 15-video audit showed that long road-edge evidence is available almost everywhere.

### 5.2 Hough lines are road edges

`Canny(mask) → HoughLinesP` naturally produces road-boundary segments. That is expected:

```text
edge A  ─────────────────

        center direction

edge B  ─────────────────
```

For orientation only, a straight road edge and the corresponding road center line share the same tangent orientation. For road *position* or corner *location*, raw edges are not sufficient.

### 5.3 Center-axis pairing audit

A stricter teacher paired roughly parallel overlapping road edges and inferred their midpoint center axis. On 180 audited frames (12 × 15 videos), observed coverage was approximately:

```text
edge-axis detected     = 98.9%
primary center axis    = 22.2%
secondary center axis  = 6.1%
center-axis corner     = 0.0%
```

The low center-axis coverage is intentional precision-first behavior, but it is too sparse for the first auxiliary task. The corner teacher is not ready at all.

Therefore:

- do **not** train center-axis position;
- do **not** train corner location yet;
- do **not** loosen pairing thresholds just to inflate label coverage;
- retain center-axis inference only as diagnostic work for later stages.

## 6. v4-C1: straight-road axial orientation supervision

The first structured supervision experiment uses only the reliable part of the teacher: dominant straight-road orientation.

The control path is unchanged from v3:

```text
15-channel v3 visual history
        ↓
MobileNetV3-Small
        ↓
shared visual feature
   ┌──────────────┬─────────────────┬──────────────────┐
   ↓              ↓                 ↓
KEEP/SWITCH   future-action     road-axis auxiliary
   head           head               head
   ↑
current PRESS / RELEASE
```

The road-axis head is training supervision only.

### 6.1 Why edge orientation is acceptable here

v4-C1 predicts orientation, not road location. On a locally straight corridor, both side boundaries have the same axial orientation as the center line. Therefore high-confidence dominant edge orientation is a valid weak label for this specific task.

Frames with ambiguous geometry are down-weighted or ignored.

### 6.2 Label gate

The full-label builder uses the current observation frame `t` and accepts a teacher target only when:

```text
primary axis exists
straight_confidence >= configured threshold
corner_score        <= configured threshold
road area            within plausible range
```

Accepted labels receive a continuous confidence weight:

```text
weight = straight_confidence × (1 - corner_score)
```

The full v3 manifest contains `19,844` observation frames. The full-label audit produced:

```text
overall detected rate = 0.967
overall accepted rate = 0.343
```

Per-video accepted rates are approximately `0.274–0.461`, with no held-out visual-theme collapse:

```text
validation:
video_20260130_173545  accepted = 0.349
video_20260130_174012  accepted = 0.461

test:
video_20260130_173426  accepted = 0.333
video_20260130_173835  accepted = 0.304
```

Per-video mean accepted weight is approximately `0.864–0.941`. This label gate passes.

### 6.3 Axial target representation

A road orientation is axial: `θ` and `θ + 180°` are equivalent. v4-C1 therefore predicts:

```text
(cos 2θ, sin 2θ)
```

rather than direct degrees.

## 7. v4-C1 loss

```text
L = L_switch
  + 0.5 * L_future_action
  + λ_axis * L_axis
```

`L_axis` is confidence-weighted cosine loss on `(cos 2θ, sin 2θ)`.

Checkpoint selection remains **lowest validation switch loss**. Geometry quality is diagnostic; it does not replace the control objective.

## 8. First v4-C1 result: λ_axis = 0.10

The first full Spark run completed all 30 epochs. Best validation switch loss occurred at epoch 1:

```text
best epoch                  = 1
best validation switch loss = 0.19249
```

Best-checkpoint test metrics:

```text
switch@100 F1      = 0.811
transition subset F1 = 0.818
short subset F1      = 0.818
future-action F1:
  h100 = 0.968
  h200 = 0.964
  h300 = 0.949
axis mean error      = 26.21°
```

Stateful test result at threshold `0.60`:

```text
v3 baseline                    v4-C1 λ=0.10
transition F1  0.918           0.906
matched        78 / 88         77 / 88
predicted      82              82
short recall   0.778           0.778
chatter <100   1               0
chatter <200   9               10
observation F1 0.753           0.612
```

Threshold `0.70` also produced target transition F1 `0.906` and short recall `0.778`. Raising the threshold to `0.80/0.90` reduced short recall to `0.667`, so threshold tuning does not recover the v3 control gate.

Conclusion:

```text
road-axis task is learnable
+ static switch classification does not collapse
- stateful control does not beat v3
```

Do not implement an ADB runtime for this checkpoint.

### 8.1 Why λ=0.10 may be too strong

At the selected test checkpoint:

```text
switch loss = 0.1341
future loss = 0.1353
axis loss   = 0.6330
```

Weighted auxiliary contributions are therefore approximately:

```text
0.5 × future loss ≈ 0.0676
0.1 × axis loss   ≈ 0.0633
```

Although `0.1` looks numerically small, the axis objective contributes nearly as much as the entire future-action auxiliary objective. The next controlled experiment therefore reduces only `λ_axis`.

## 9. v4-C1 sweep and training-efficiency changes

The next primary point is:

```text
λ_axis = 0.03
```

If it still degrades control, test `0.01`. A `0.0` run is available as an objective-level ablation, but it is not bit-for-bit identical to v3 because the v4-C1 model still instantiates the unused axis head.

The training script now supports:

```text
--axis-weight
--artifact-name
--epochs
--early-stopping-patience
--early-stopping-min-delta
```

When `--axis-weight` is supplied without an explicit artifact name, the run is automatically separated from the original artifact. Example:

```text
--axis-weight 0.03
→ artifacts/models/mobilenet_v3_small_v4c1_aw0p03/
```

The default config now uses validation-switch-loss early stopping:

```text
patience = 4 epochs
min_delta = 0
maximum epochs = 30
```

With the previous best epoch at 1, this prevents repeatedly running the remaining 20+ overfit epochs unless validation improves again.

Training resilience is also improved. After every completed epoch the script atomically updates:

```text
history.json
training_state.json
```

The best `model.pt` is still saved immediately on validation improvement. Therefore an SSH/session interruption no longer hides all completed epoch history.

## 10. Next λ=0.03 experiment

After syncing the updated code to Spark, run:

```bash
python scripts/train_model_v4c1.py \
  --config configs/train_v4c1.yaml \
  --samples data/processed/v3/samples.jsonl \
  --axis-labels data/processed/v4c1/axis_labels.jsonl \
  --axis-weight 0.03 \
  --require-cache \
  --num-workers 4
```

The automatic artifact directory is:

```text
artifacts/models/mobilenet_v3_small_v4c1_aw0p03/
```

Evaluate it with:

```bash
python scripts/evaluate_model_v4c1.py \
  --config configs/train_v4c1.yaml \
  --samples data/processed/v3/samples.jsonl \
  --labels-dir data/processed/v3/labels \
  --axis-labels data/processed/v4c1/axis_labels.jsonl \
  --artifact-name mobilenet_v3_small_v4c1_aw0p03 \
  --split test \
  --require-cache \
  --num-workers 4
```

The control gate remains:

```text
transition F1 >= 0.918
short recall  >= 0.778
```

Axis error should remain meaningfully below a random axial predictor, but a lower axis error alone does not justify keeping the auxiliary objective.

## 11. Later geometry stages

### v4-C2: corner state

Only after a reliable center-axis or travel-aware teacher exists:

```text
corner visible
corner proximity/location
```

Do not call a visible intersection `next_corner` without travel direction.

### v4-C3: kart-relative geometry

The second-bend failure is more directly described by:

```text
kart center
kart heading
road orientation
lateral offset
heading error
```

If `λ_axis = 0.01/0.03/0.10` all fail to improve v3 control, stop tuning road-only supervision and move to kart-relative state rather than adding more road-only heads.

## 12. Covariate shift remains independent

Structured perception does not solve expert-only behavior-cloning distribution shift. Closed-loop deviations still require valid pre-failure correction/recovery data through iterative behavior cloning or a DAgger-like process.

Do not fabricate recovery supervision after the kart is already irreversibly off track.

## 13. Shadow teacher

Analytic geometry remains useful after model training for failure diagnosis:

```text
recorded MP4
   ├── learned road-axis prediction
   ├── analytic teacher orientation
   └── policy KEEP/SWITCH
```

This separates representation errors from control errors and off-distribution failures.

## 14. Current implementation order

```text
1. pytest + Ruff after sweep/early-stopping changes
2. rsync updated code to Spark
3. run λ_axis=0.03 with early stopping
4. stateful evaluator vs v3
5. if control gate passes, consider ADB closed-loop A/B
6. if 0.03 still misses, test λ_axis=0.01
7. if road-axis sweep has no net control gain, move to kart-relative supervision
```

## 15. What v4 is not

v4 is not a hand-written geometry controller.

v4 is not based on the claim that v3 lacked temporal information.

v4-C1 is not training the model to reproduce road-edge pixel locations. It uses road-edge evidence only to derive a high-confidence weak label for straight-road **orientation**.

Deployment remains neural perception plus learned control.
