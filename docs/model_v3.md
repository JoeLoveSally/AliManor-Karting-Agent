# Model v3: State-Conditioned Transition Policy

## 1. Motivation

v2 improved the original current-action shortcut problem: multi-horizon supervision, HUD masking, held-out replay, spatial focus, and temporal counterfactuals all show that the model uses future visual information rather than only copying the current action.

The first armed v2 `+100ms` closed-loop run still exposed a more specific failure mode. The model entered the first bend correctly, but kept `PRESS` with near-saturated confidence for too long after the main turn had already developed. By the time it emitted `RELEASE`, the trajectory had moved outside the expert-like state distribution and the kart could not recover.

This suggests that the control question should be represented explicitly as:

> Given the visual history and the action that is physically active now, should that action be kept or switched to match the desired near-future control state?

v3 therefore makes the current physical action an explicit model input and makes `KEEP / SWITCH` the primary prediction target.

## 2. Input

The visual input remains unchanged from v2:

```text
frame(t-200ms)
frame(t-150ms)
frame(t-100ms)
frame(t-50ms)
frame(t)
      ↓
15-channel visual encoder
```

The controller state is supplied separately:

```text
current_action ∈ {RELEASE, PRESS}
```

The kart is **not** permanently masked. Its heading, lateral position, and relation to track boundaries are potentially causal state information. The fixed touch/HUD masks from v2 remain enabled. Kart-region masking or random erasing is reserved for a later ablation if v3 still shows action-effect shortcuts.

## 3. Primary target: state-conditioned KEEP / SWITCH

For each prediction horizon `h`, let:

```text
future_action_h = action(t + h)
conditioned_state = the physical action supplied to the model
```

The primary target is:

```text
switch_h = conditioned_state XOR future_action_h
```

Therefore:

```text
switch_h = 0  -> KEEP the supplied current state
switch_h = 1  -> SWITCH the supplied current state
```

Initial horizons remain:

```text
+100ms
+200ms
+300ms
```

The initial runtime candidate is `switch@+100ms` because v2 held-out replay showed the strongest transition timing at that horizon.

This target is deliberately defined as "does the desired state at this horizon differ from the supplied current state?" rather than "did any transition occur inside the interval?". Two transitions inside one horizon can return to the original state, in which case the correct target is KEEP.

## 4. Counterfactual state conditioning

A naive state-conditioned model trained only with the expert's actual `action(t)` would still have a closed-loop mismatch. If runtime switches slightly early, the model's physical state can differ from the expert state for the same visual scene.

To make that case part of training, every training visual sample is paired with **both** possible conditioned states:

```text
same visual history + RELEASE -> switch targets relative to RELEASE
same visual history + PRESS   -> switch targets relative to PRESS
```

For each horizon, these two targets are exact complements. This has two useful properties:

1. the primary switch target is balanced by construction during training;
2. the model is explicitly trained to answer from either runtime state, including the counterfactual state that may occur after an early model-driven switch.

Validation and test do **not** duplicate states. They use the recorded expert `action(t)` so transition metrics remain interpretable on the held-out expert trajectory.

## 5. Model structure

```text
5-frame visual history
        │
        ▼
 MobileNetV3-Small
        │
  visual feature ────────────────┐
        │                        │
        │                        ▼
        │                 future-action heads
        │                  +100/+200/+300
        │
        ├──────────────┐
        │              │
        │       current-action embedding
        │              │
        └──── concat ──┘
               │
               ▼
        nonlinear switch head
               │
        +100/+200/+300
```

The switch head must be nonlinear because KEEP/SWITCH is an XOR-style relation between the supplied state and the desired future state.

## 6. Auxiliary future-action supervision

The v2 future-action objective remains as an auxiliary task:

```text
action(t+100)
action(t+200)
action(t+300)
```

The auxiliary heads consume only the visual feature; they do **not** receive the supplied current state. This preserves the v2 pressure for the visual backbone to learn future track/vehicle information rather than using the explicit action state as a shortcut.

Training loss:

```text
loss = BCE(switch_logits, switch_targets)
     + auxiliary_action_weight * BCE(action_logits, future_action_targets)
```

Initial `auxiliary_action_weight = 0.5`.

The primary checkpoint criterion is validation switch loss, not auxiliary-action loss.

## 7. Data and artifact isolation

v3 uses a separate manifest and model artifact:

```text
data/processed/v3/
  samples.jsonl
  manifest.json
  labels/

artifacts/models/mobilenet_v3_small_v3/
```

The visual preprocessing is identical to v2, so the existing v2 frame cache can be reused:

```text
data/processed/frame_cache_v2/
```

No 5.6 GB cache rebuild is required unless visual preprocessing changes.

## 8. Initial training configuration

```text
frame_stack            = 5
history_ms             = 200
frame_interval_ms      = 50
prediction_horizons_ms = [100, 200, 300]
control_horizon_ms     = 100
input_size             = 224
visual_feature_dim     = 128
state_embedding_dim    = 8
auxiliary_action_weight = 0.5
```

Sampling keeps the existing stable / transition / short-correction priority. Training samples are duplicated only through counterfactual state conditioning; source videos and train/validation/test video splits remain unchanged.

## 9. Acceptance gates before runtime integration

v3 runtime integration is intentionally deferred until the offline model passes these gates:

1. the counterfactual training switch target is approximately balanced for every horizon;
2. held-out validation/test `switch@+100ms` precision, recall, and F1 are materially better than a trivial KEEP baseline;
3. false switch predictions on stable sections remain low enough that a 30 Hz controller will not chatter;
4. `+100/+200/+300` auxiliary future-action heads remain competitive with v2, showing that the visual representation did not collapse;
5. transition and short-correction subsets are inspected separately.

If these gates pass, the next stage adds a v3 runner and a state-conditioned RuntimeEngine path where the current executor/controller state is fed into the model on every inference step. A high `switch@+100ms` probability flips the physical state; otherwise Runtime holds it.

## 10. Build and smoke-test plan

```bash
python scripts/build_dataset.py \
  --config configs/train_v3.yaml \
  --output data/processed/v3

python scripts/train_model_v3.py \
  --config configs/train_v3.yaml \
  --samples data/processed/v3/samples.jsonl \
  --require-cache \
  --smoke
```

After the local smoke test passes, run the same command without `--smoke` on Spark. Runtime/ADB changes are made only after the trained v3 artifact passes the offline acceptance gates above.
