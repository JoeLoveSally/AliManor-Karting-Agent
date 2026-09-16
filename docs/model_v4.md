# Model v4: Structured Model-Driven Policy

## 1. Goal

v4 keeps deployment fully model-driven while separating two questions:

1. temporal representation;
2. visual supervision that encourages causal road/kart geometry instead of shortcut features.

Analytic CV is offline teacher/debug tooling only. It never sends runtime actions.

## 2. v3 reference

v3 remains the reference control formulation:

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

Relevant original-v3 stateful reference at threshold `0.60`:

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
2. kart center / heading
3. lateral offset and heading error
4. later: visible corner state
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

- do **not** train global center-axis position;
- do **not** train corner location yet;
- do **not** loosen pairing thresholds just to inflate label coverage;
- retain center-axis inference only as diagnostic work.

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

## 8. λ_axis = 0.10 result

The first full Spark run completed all 30 epochs. Best validation switch loss occurred at epoch 1:

```text
best epoch                   = 1
best validation switch loss = 0.19249
```

Best-checkpoint test metrics:

```text
switch@100 F1         = 0.811
transition subset F1  = 0.818
short subset F1       = 0.818
future-action F1 h100 = 0.968
future-action F1 h200 = 0.964
future-action F1 h300 = 0.949
axis mean error       = 26.21°
```

Stateful test result at threshold `0.60`:

```text
original v3                 v4-C1 λ=0.10
transition F1  0.918        0.906
matched        78 / 88      77 / 88
predicted      82           82
short recall   0.778        0.778
chatter <100   1            0
chatter <200   9            10
observation F1 0.753        0.612
```

The axis task is learned, but control does not improve.

## 9. λ_axis = 0.03 result

Early stopping reduced the run to five epochs because validation switch loss stopped improving after epoch 1:

```text
best epoch                   = 1
best validation switch loss = 0.201525
```

Selected-checkpoint test metrics:

```text
axis error = 27.49°
```

At threshold `0.60`:

```text
stateful target F1 = 0.849
matched            = 79 / 88
predicted          = 98
short recall       = 0.833
chatter <100ms     = 8
chatter <200ms     = 21
observation F1     = 0.667
```

Reducing the road-axis weight did not recover control. The checkpoint became substantially more trigger-happy and chatters more often.

## 10. λ_axis = 0 control ablation

The v4-C1 harness was also trained with the axis objective disabled. The axis head still exists structurally but receives no loss.

```text
best epoch                   = 3
best validation switch loss = 0.227882
axis error                   = 37.36°  # untrained head; diagnostic only
```

Threshold selection was performed on validation, not test. Validation `0.60` and `0.70` tied on target metrics:

```text
threshold 0.60 / 0.70
stateful target F1 = 0.933
matched            = 70 / 72
predicted          = 78
short recall       = 0.917
chatter <100ms     = 0
chatter <200ms     = 3
```

`0.70` was selected because it preserved the same target/short/chatter result with slightly better observation alignment.

On test at the validation-selected `0.70`:

```text
stateful target F1 = 0.913
matched            = 84 / 88
predicted          = 96
short recall       = 0.944
chatter <100ms     = 8
chatter <200ms     = 12
observation F1     = 0.696
```

This is not a clean improvement over the original v3 checkpoint. It is a more aggressive controller: it catches more short corrections but also emits more extra transitions.

## 11. Closed-loop result of λ_axis = 0

The `λ_axis=0` checkpoint was loaded through the ordinary v3 runtime after dropping the unused `axis_head.*` parameters. It was run on the real device at threshold `0.70`.

Runtime health was good:

```text
input fps       ≈ 55.6
control fps     ≈ 29.2
inference mean  ≈ 6.7 ms
inference p95   ≈ 9.2 ms
dropped frames  = 1
```

Observed state changes:

```text
PRESS    source frame 399
RELEASE  source frame 516
PRESS    source frame 595
```

The critical failure is the long KEEP-RELEASE interval after frame 516. Dense video/JSON inspection shows the useful pre-failure region is approximately:

```text
frame 532 → frame 548
```

During this window the kart is still in a state where an earlier correction could matter, but `p_switch` remains very low. By roughly frame 560 the kart is already outside the useful correction regime. The later PRESS near frame 595 happens after the failure state is visually established.

Therefore this failure is not mainly a runtime-throughput problem and is not plausibly fixed by another threshold sweep.

## 12. Road-only supervision conclusion

The road-axis experiments now answer the main question:

```text
road orientation is learnable
but
road orientation alone does not solve the second-bend control failure
```

Do not continue `λ_axis` sweeping. Do not add more road-only heads before testing kart-relative state.

The missing state is better described as:

```text
kart center
kart heading
road orientation
→ lateral offset
→ heading error
```

## 13. Kart-relative teacher POC

The next offline teacher is implemented separately and documented in:

```text
docs/kart_relative_teacher.md
src/karting_agent/train/kart_pose_pseudo_labels.py
scripts/inspect_kart_relative_geometry.py
```

The first POC uses:

```text
red/orange chassis evidence
→ kart center
→ PCA axial kart heading

road mask + dominant road orientation
→ local cross-sections around kart
→ local road center/width
→ lateral offset
→ road-relative heading error
```

The local road relation intentionally does **not** depend on the low-coverage global center-axis pairing teacher.

No kart-relative model head is approved yet. The teacher must first pass a visual audit on the known closed-loop failure and then on sparse samples across all expert videos.

## 14. Covariate shift remains independent

Structured perception does not solve expert-only behavior-cloning distribution shift. Closed-loop deviations still require valid pre-failure correction data through iterative behavior cloning or a DAgger-like process.

The frame `532–548` region in the latest run is useful because the kart is deviating but not yet irreversibly lost. Do not fabricate recovery supervision from later fully off-road/failure frames.

## 15. Shadow teacher

Analytic geometry is now intended to be a shadow debugger:

```text
recorded MP4
   ├── road orientation
   ├── kart center / axial heading
   ├── local lateral offset / heading error
   └── policy p_switch / KEEP/SWITCH
```

This can distinguish:

```text
perception/state error
vs
control mapping error
vs
closed-loop distribution shift
```

## 16. Current implementation order

```text
1. pytest + Ruff for kart-relative teacher POC
2. dense audit of adb_20260916T003536Z frames 500–570
3. inspect whether center/heading/offset become abnormal before frame 548
4. sparse kart-relative audit across all 15 expert videos
5. only if teacher precision is acceptable, design a kart-relative auxiliary target
6. keep deployment fully neural; analytic geometry remains offline only
```

## 17. What v4 is not

v4 is not a hand-written geometry controller.

v4 is not based on the claim that v3 lacked temporal information.

v4-C1 does not train the model to reproduce road-edge pixel locations. It uses road-edge evidence only to derive a weak straight-road orientation target.

The kart-relative teacher is not part of runtime deployment.

Deployment remains neural perception plus learned control.
