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

Current feasibility defaults are stored in `configs/geometry_pseudo_labels.yaml` and remain tunable teacher parameters, not model hyperparameters.

### 6.3 Axial target representation

A road orientation is axial: `θ` and `θ + 180°` are equivalent. v4-C1 therefore uses:

```text
(cos 2θ, sin 2θ)
```

rather than direct angle regression.

This removes the `0° / 180°` discontinuity.

## 7. v4-C1 loss

The first experiment changes only one training objective relative to v3:

```text
L = L_switch
  + 0.5 * L_future_action
  + λ_axis * L_axis
```

`L_axis` is confidence-weighted cosine loss on `(cos 2θ, sin 2θ)`.

Initial `λ_axis = 0.1` is a starting experiment, not a final constant. Checkpoint selection remains **lowest validation switch loss**, matching v3 semantics, so the geometry objective cannot silently replace the control objective.

The held-out evaluator reports both:

```text
control metrics
+ weighted road-axis angular error
```

A lower axis error is not sufficient for model acceptance; control must remain competitive with v3.

## 8. v4-C1 implementation

Implemented files:

```text
configs/train_v4c1.yaml
scripts/build_axis_pseudo_labels.py
scripts/train_model_v4c1.py
scripts/evaluate_model_v4c1.py
src/karting_agent/model/axis_supervised.py
src/karting_agent/train/axis_labels.py
tests/unit/test_v4c1_axis_supervision.py
```

The v3 sample manifest and `frame_cache_v2` are reused. No new temporal dataset is created.

Pseudo-label generation reads raw current frames because the HSV teacher should operate on the original visual theme rather than normalized model tensors.

## 9. Required gate before Spark training

First build the full axis-label file on the PC:

```bash
python scripts/build_axis_pseudo_labels.py \
  --samples data/processed/v3/samples.jsonl
```

Outputs:

```text
data/processed/v4c1/axis_labels.jsonl
data/processed/v4c1/axis_manifest.json
```

Before training, inspect the console/manifest for:

```text
detected_rate
accepted_rate
mean_accepted_weight
per-video accepted_rate
```

The goal is not maximum coverage. A moderate high-precision straight-frame subset is preferable to broad ambiguous labels.

If one held-out visual theme has near-zero accepted labels while training themes are high, stop and revisit the teacher before interpreting model results.

## 10. Training/evaluation gate

If the label distribution is reasonable:

```text
PC smoke
→ Spark smoke
→ full v4-C1 training
→ stateful held-out evaluation
```

The main control reference remains:

```text
v3 @ 0.60
transition F1 = 0.918
short recall  = 0.778
observation F1= 0.753
```

Interpretation:

```text
axis error improves + control improves/holds
→ structured orientation supervision is promising

axis error improves + control degrades
→ auxiliary objective is learned but conflicts with control;
  tune/remove λ_axis rather than adding more geometry heads

axis error remains poor
→ teacher/representation is not useful enough
```

Do not add corner/kart heads in the same experiment.

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

These become later targets only when their teacher labels are demonstrably reliable.

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
1. build full v4-C1 axis pseudo-labels on PC
2. inspect total + per-video accepted-rate distribution
3. pytest + Ruff
4. v4-C1 PC smoke
5. copy the small axis-label files to Spark
6. Spark smoke
7. full v4-C1 training
8. stateful evaluator vs v3
9. only if offline control gate passes, implement/run ADB closed-loop A/B
10. keep center-axis/corner and kart-pose work as later independent experiments
```

## 15. What v4 is not

v4 is not a hand-written geometry controller.

v4 is not based on the claim that v3 lacked temporal information.

v4-C1 is not training the model to reproduce road-edge pixel locations. It uses road-edge evidence only to derive a high-confidence weak label for straight-road **orientation**.

Deployment remains neural perception plus learned control.
