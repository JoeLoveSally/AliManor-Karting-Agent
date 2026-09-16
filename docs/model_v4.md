# Model v4: Structured Model-Driven Policy

## 1. Goal

v4 keeps deployment fully model-driven while separating two questions:

1. temporal representation;
2. visual supervision that encourages causal road/kart state instead of shortcut features.

Analytic CV is an offline teacher/debugger only. It never sends runtime actions.

## 2. v3 reference

The reference control formulation remains:

```text
5 RGB frames at t-200,-150,-100,-50,t
→ concatenate as 15 channels
→ MobileNetV3-Small
→ visual feature
+ current PRESS / RELEASE embedding
→ KEEP / SWITCH at +100/+200/+300ms
```

Future action is retained as a visual-only auxiliary task. Counterfactual training pairs each expert visual observation with both current controller states.

Original-v3 stateful test reference at threshold `0.60`:

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

The controlled v3 → v4-A → v4-A2 experiments kept data, split, 200ms/5-frame history, targets, state conditioning, sampling and training semantics fixed.

```text
v3 15-channel temporal CNN         → best reference formulation
v4-A per-frame CNN + GRU           → smoother, worse short corrections
v4-A2 per-frame CNN + temporal MLP → close, no net win
```

Representative stateful results:

```text
                 v3@0.60   v4-A@0.50   v4-A2@0.70
transition F1      0.918       0.892        0.907
short recall       0.778       0.722        0.722
observation F1     0.753       0.554        0.744
```

Temporal architecture search is therefore paused. v4-C keeps the v3 control architecture and changes supervision.

## 4. Track geometry model

Tracks are better treated as piecewise-straight corridors with discrete corners than as continuously curved centerlines. The curved trajectory during drift describes vehicle motion more than road curvature.

Priority is therefore:

```text
1. local road orientation
2. kart center / heading
3. kart-relative lateral offset / heading error
4. later: visible corner state
5. later: travel-relative next-corner semantics
```

## 5. Road teacher findings

A multi-theme HSV mask covers the blue, teal, green and dark-neutral road themes. `Canny(mask) → HoughLinesP` naturally detects road boundaries. For orientation, a straight boundary has the same axial tangent as the road center line.

A stricter parallel-edge pairing experiment attempted to infer center-axis position. Across 180 audit frames:

```text
edge-axis detected     ≈ 98.9%
primary center axis    ≈ 22.2%
secondary center axis  ≈ 6.1%
center-axis corner     = 0%
```

Therefore center-axis position and corner location are not approved training targets.

## 6. v4-C1: road-axis auxiliary

v4-C1 preserved the v3 control path and added a visual-only axial road-orientation head encoded as:

```text
(cos 2θ, sin 2θ)
```

The full 19,844-frame label gate produced:

```text
axis detected = 0.967
accepted      = 0.343
```

with no held-out visual-theme collapse.

The loss was:

```text
L = L_switch + 0.5 L_future + λ_axis L_axis
```

### λ_axis = 0.10

```text
axis error             = 26.21°
stateful target F1@.60 = 0.906
short recall           = 0.778
observation F1         = 0.612
```

### λ_axis = 0.03

Early stopping selected epoch 1. At threshold `0.60`:

```text
axis error             = 27.49°
stateful target F1     = 0.849
matched                = 79 / 88
predicted              = 98
short recall           = 0.833
chatter <100/<200ms    = 8 / 21
observation F1         = 0.667
```

### λ_axis = 0 control ablation

Validation selected threshold `0.70`:

```text
validation target F1 = 0.933
matched              = 70 / 72
predicted            = 78
short recall         = 0.917
chatter <100/<200ms  = 0 / 3
```

On test at that validation-selected threshold:

```text
target F1            = 0.913
matched              = 84 / 88
predicted            = 96
short recall         = 0.944
chatter <100/<200ms  = 8 / 12
observation F1       = 0.696
```

This checkpoint is more aggressive than original v3, not a clean offline replacement.

## 7. Closed-loop result of λ_axis = 0

The checkpoint was loaded through the ordinary v3 runtime, dropping the unused axis-head parameters. Runtime health was good:

```text
input fps       ≈ 55.6
control fps     ≈ 29.2
inference mean  ≈ 6.7 ms
inference p95   ≈ 9.2 ms
dropped frames  = 1
```

State changes:

```text
PRESS    source frame 399
RELEASE  source frame 516
PRESS    source frame 595
```

The important failure is the long KEEP-RELEASE interval. In approximately frames `532–548`, the kart is already deviating while correction can still matter, but switch probability remains very low. By roughly frame 560 the kart is already outside the useful correction regime.

This is not primarily a runtime-throughput or threshold problem.

## 8. Road-only conclusion

The road-axis experiment now answers the intended question:

```text
road orientation is learnable
but road orientation alone does not solve the second-bend failure
```

No further `λ_axis` sweep is planned. The next experiment targets kart-relative state.

## 9. Kart-relative teacher

The offline teacher is implemented in:

```text
src/karting_agent/train/kart_pose_pseudo_labels.py
scripts/inspect_kart_relative_geometry.py
configs/geometry_pseudo_labels.yaml
```

Pipeline:

```text
red/orange chassis → kart center → axial PCA heading
road mask + local Hough evidence → locally relevant road corridor
kart + corridor → lateral offset + heading error
```

The road axis is selected locally around the kart. The teacher does not blindly use the globally dominant road orientation at intersections.

Dense audit of the known failed run over frames `500–570` produced:

```text
previews           = 18
kart detected      = 1.000
heading usable     = 1.000
relation usable    = 0.833
offroad            = 0.133
mean |heading err| = 39.2°
mean |lateral|     = 0.49 road-half-widths
```

This passes the critical-run POC, but the teacher still requires a sparse 15-video visual audit and full-label distribution gate before training.

## 10. v4-C2: kart-relative auxiliary supervision

v4-C2 is implemented but **not yet approved for full training**.

Implementation:

```text
scripts/build_kart_relative_pseudo_labels.py
src/karting_agent/train/kart_relative_labels.py
src/karting_agent/model/kart_relative_supervised.py
scripts/train_model_v4c2.py
scripts/evaluate_model_v4c2.py
configs/train_v4c2.yaml
```

The v3 control path is unchanged. From the shared visual feature, v4-C2 additionally predicts:

```text
lateral offset
axial heading-error vector
edge-risk logit
```

### 10.1 Lateral target

```text
lateral_target = clip(offset / (road_width/2), -1.25, +1.25)
```

Loss: confidence-weighted Smooth-L1.

### 10.2 Heading-error target

Because kart and road heading are axial, heading error is encoded as:

```text
(cos 2e, sin 2e)
```

Loss: confidence-weighted cosine loss.

### 10.3 Edge-risk target

Expert demonstrations should contain few true off-road frames, so a pure `offroad` classifier would be almost all-negative. The first auxiliary therefore predicts a pre-failure edge-risk condition:

```text
edge_risk = |lateral_offset_norm| >= 0.70
```

This is training supervision only, not a hard-coded controller.

### 10.4 Initial loss weights

```text
L = L_switch
  + 0.50 L_future_action
  + 0.03 L_lateral
  + 0.03 L_heading
  + 0.01 L_edge_risk
```

These weights are intentionally conservative after v4-C1 showed that an auxiliary can be learnable while still harming control. Checkpoint selection and early stopping monitor validation **switch loss**.

## 11. v4-C2 gate

Before Spark training, build pseudo-labels for all v3 observation frames and inspect:

```text
pose rate
heading rate
relation rate
accepted rate
edge-risk positive rate
true off-road rate
mean teacher weight
mean |lateral offset|
mean |heading error|
```

Held-out videos must not collapse:

```text
validation: 173545, 174012
test:       173426, 173835
```

Then perform a sparse overlay audit across all 15 expert recordings. If a visual theme or local-road selector is systematically wrong, fix the teacher before training.

After the gate, v4-C2 evaluation reports both the existing static/stateful control metrics and:

```text
weighted lateral MAE
weighted heading-error degrees
edge-risk F1
```

Control threshold selection remains validation-only; test is final reporting.

## 12. Covariate shift remains independent

Structured state supervision does not solve expert-only behavior-cloning distribution shift. If v4-C2 learns kart-relative state but closed-loop failure remains, the next data step is iterative collection of valid pre-failure correction states. Do not fabricate recovery supervision after the kart is already irreversibly off track.

## 13. Current implementation order

```text
1. pytest + Ruff for v4-C2 code
2. build full kart-relative pseudo-label manifest on PC
3. check per-video distribution, especially held-out themes
4. sparse overlay audit across all 15 expert videos
5. if the teacher gate passes, run PC smoke
6. rsync code + small v4-C2 label files to Spark
7. Spark smoke → full train with early stopping
8. select threshold on validation, then report test
9. only if offline gate is credible, run closed-loop A/B
```

## 14. What v4 is not

v4 is not a hand-written geometry controller. It is not based on the claim that v3 lacked all temporal information. The analytic teachers generate structured supervision and diagnostics only.

Deployment remains neural perception plus learned control.
