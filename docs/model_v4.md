# Model v4: Structured Model-Driven Policy

## 1. Goal

v4 keeps the deployed controller fully model-driven while separating two questions that v1-v3 mixed together:

1. how temporal information should be represented;
2. whether visual supervision should force the model to understand road geometry instead of relying on shortcuts.

Traditional computer vision is allowed only as an offline pseudo-label teacher and shadow debugger. It never sends runtime control actions.

## 2. v3 reference

v3 remains the best control baseline at the end of the temporal ablation.

Architecture:

```text
5 RGB frames
→ concatenate as 15 channels
→ MobileNetV3-Small
→ visual feature
+ current PRESS / RELEASE embedding
→ KEEP / SWITCH
```

The expanded first convolution is initialized by repeating pretrained RGB weights and dividing by `frame_stack`, but it is trainable. Therefore v3 is only equivalent to temporal averaging at initialization, not after training.

Previous temporal counterfactual analysis confirmed that the trained model uses frame order:

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

## 3. Controlled temporal ablation

The v3 → v4-A → v4-A2 comparison kept these fixed:

- same 15 source videos and video-level split;
- same v3 sample manifest;
- same five timestamps `[-200,-150,-100,-50,0] ms`;
- same `224×224` preprocessing and HUD/touch masks;
- same current-state conditioning and counterfactual train states;
- same KEEP/SWITCH targets;
- same future-action auxiliary targets and weight `0.5`;
- same `100/200/300ms` horizons and `100ms` control horizon;
- same sampling weights, optimizer, batch size, epoch count, and seed;
- same checkpoint rule: lowest validation `switch_loss`;
- no geometry loss.

The explicit-time models reused the same per-frame cache in `data/processed/frame_cache_v2`.

## 4. v4-A: shared per-frame CNN + GRU

```text
5 RGB frames
→ shared MobileNetV3-Small per frame
→ 5 × 128-d feature sequence
→ GRU(128)
→ temporal feature
+ current action
→ KEEP / SWITCH
```

Best checkpoint was epoch 1 because validation loss worsened rapidly after that.

Held-out result:

```text
static switch@100 F1 = 0.776
aux h100/h200/h300   = 0.960 / 0.959 / 0.962
```

Best useful stateful point was approximately threshold `0.50`:

```text
target transition F1 = 0.892
matched              = 74 / 88
predicted            = 78
short recall         = 0.722
chatter <100ms       = 0
chatter <200ms       = 1
observation F1       = 0.554
```

Interpretation: GRU produced a smoother controller, but it missed useful short corrections and did not beat v3.

## 5. v4-A2: shared per-frame CNN + ordered temporal MLP

v4-A2 removed recurrence while preserving explicit frame order:

```text
frame -200 → CNN → z1
frame -150 → CNN → z2
frame -100 → CNN → z3
frame  -50 → CNN → z4
frame    0 → CNN → z5

[z1,z2,z3,z4,z5]
→ ordered concatenation (640-d)
→ Linear(640,128) + Hardswish + Dropout
→ temporal feature
+ current action
→ KEEP / SWITCH
```

The model again overfit quickly. Lowest validation `switch_loss` occurred at epoch 1.

Held-out static result:

```text
switch@100 precision = 0.793
switch@100 recall    = 0.769
switch@100 F1        = 0.781
aux h100/h200/h300   = 0.965 / 0.962 / 0.959
```

Stateful evaluator:

```text
threshold 0.50:
  target F1      = 0.897
  matched        = 78 / 88
  predicted      = 86
  short recall   = 0.722
  chatter <100   = 2
  chatter <200   = 8
  observation F1 = 0.701

threshold 0.60:
  target F1      = 0.897
  matched        = 78 / 88
  predicted      = 86
  short recall   = 0.722
  chatter <100   = 2
  chatter <200   = 8
  observation F1 = 0.736

threshold 0.70:
  target F1      = 0.907
  matched        = 78 / 88
  predicted      = 84
  short recall   = 0.722
  chatter <100   = 1
  chatter <200   = 4
  observation F1 = 0.744

threshold 0.90:
  target F1      = 0.894
  matched        = 76 / 88
  predicted      = 82
  short recall   = 0.667
  chatter <100   = 0
  chatter <200   = 3
  observation F1 = 0.812
```

The best target-timeline point is threshold `0.70`. It is close to v3 on transition F1 and has less chatter, but still loses on the control behavior we care about most:

```text
                 v3@0.60   v4-A2@0.70
transition F1      0.918       0.907
matched            78/88       78/88
short recall       0.778       0.722
observation F1     0.753       0.744
chatter <200ms         9           4
```

The higher v4-A2 observation F1 at threshold `0.90` is not selected because it is achieved by delaying decisions and reducing short-correction recall to `0.667`.

## 6. Temporal conclusion

The controlled temporal experiment is now closed.

Evidence:

```text
v3 15-channel temporal CNN        → best overall control baseline
v4-A per-frame CNN + GRU          → worse, overly smooth
v4-A2 per-frame CNN + temporal MLP→ closer, but still no meaningful win
```

Therefore:

- do not implement v4-A/v4-A2 ADB runtime;
- do not spend the next iteration on GRU tuning;
- do not proceed to `400/800ms` temporal history yet;
- do not add TCN/attention simply to continue architecture search.

This result does not prove explicit time is inherently inferior. It shows that with the current 15 expert videos, explicit per-frame encoders add capacity and overfit quickly without producing better held-out control behavior.

## 7. Next phase: v4-C structured visual supervision

The next experiment keeps the best v3 control formulation and asks a different question:

> Can auxiliary road supervision force the visual backbone to encode causal track geometry and reduce shortcut reliance?

Initial target:

```text
road segmentation
```

Later targets such as centerline, curvature, bend distance, kart center, or heading are added only if their labels are reliable.

The intended experimental principle is again to change one thing at a time. The first v4-C policy should stay as close to v3 as possible while adding an auxiliary geometry objective.

Conceptually:

```text
15-channel v3 visual history
        ↓
MobileNetV3 shared backbone
        ├── policy feature → state-conditioned KEEP/SWITCH
        ├── future-action auxiliary head
        └── road-geometry auxiliary supervision
```

The exact geometry head should only be finalized after teacher-label quality is audited.

## 8. Geometry pseudo-label teacher

Current teacher:

```text
expert video frame
→ HSV threshold for the current blue-track theme
→ morphology
→ largest valid road component
→ road mask
→ optional row-centerline diagnostic
```

The teacher is supervision tooling only. It is not a runtime controller.

Before pseudo-labels enter training loss, audit all 15 videos:

1. sample representative frames from every video;
2. inspect road-mask coverage through straights and bends;
3. check whether kart occlusion breaks the road region;
4. check whether UI/background regions leak into the mask;
5. identify visual themes or intervals where the teacher is unreliable;
6. exclude or down-weight unreliable pseudo-labels rather than silently training on them.

The inspection script writes per-video sheets plus one global overview sheet:

```text
artifacts/geometry_pseudo_labels/
├── overview_contact_sheet.jpg
├── index.json
└── <video>/
    ├── contact_sheet.jpg
    ├── frame_XXXXXX.jpg
    ├── frame_XXXXXX_mask.png
    └── summary.json
```

## 9. Geometry loss

When the teacher passes audit, v4-C may use:

```text
L = L_switch
  + λ_action * L_future_action
  + λ_road * L_road
```

`λ_road` is an explicit hyperparameter, not a default constant. Too little supervision may leave the shortcut unchanged; too much may over-optimize segmentation at the expense of control.

Geometry quality and policy quality must be evaluated separately.

## 10. Covariate shift remains independent

v4-C addresses visual representation, not expert-only behavior-cloning covariate shift.

Even with better perception, model-driven closed loop can still create observations outside the expert distribution after an earlier timing error. Later iterations therefore still need valid pre-failure deviation/recovery data through iterative behavior cloning or a DAgger-like process.

Do not create labels around impossible recovery after the kart is already irreversibly off track.

## 11. Shadow teacher

Keep the analytic teacher after training for offline post-run diagnosis:

```text
recorded MP4
   ├── learned geometry prediction
   ├── analytic teacher geometry
   └── policy KEEP/SWITCH output
```

Failure classification:

```text
learned geometry wrong
→ perception/representation problem

geometry reasonable, policy wrong
→ control problem

both reasonable, state outside expert distribution
→ data/covariate-shift problem
```

## 12. Current implementation order

```text
1. run road pseudo-label inspection on all 15 expert videos
2. manually audit overview + suspicious per-video contact sheets
3. define trusted/untrusted teacher coverage
4. only after audit, implement full road-label generation
5. implement v4-C with v3 control path + road auxiliary supervision
6. tune λ_road using held-out geometry and control metrics
7. stateful evaluation against the same v3 reference
8. only if offline gate passes, run ADB closed-loop A/B
9. preserve failure/deviation runs for later DAgger-like data iteration
```

## 13. What v4 is not

v4 is not a hand-written geometric controller.

v4 is not based on the claim that v3 lacked temporal information.

v4 is not committed to GRU or explicit per-frame encoding when the evidence does not support them.

The project remains model-driven: analytic CV is used to create supervision and diagnostics, while deployment remains neural inference plus learned control.
