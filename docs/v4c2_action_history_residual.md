# Experimental V4-C2 action-history residual (offline only)

This branch extends the read-only action-history audit with a **separate** factual-data training path. It does not modify the deployed V4-C2 `model.pt`, `metadata_h200.json`, or ADB control code. It does not claim to fix the 7-second failures. Keep V4-C4 PR Draft.

## Hypothesis and limits

The deployed switch classifier sees 5 RGB frames spanning 200 ms but conditions its switch head only on the current PRESS/RELEASE state. Predictions immediately after a state change may reflect images produced under an earlier input; this is plausible, **not an established cause of failure**. Existing video touch-marker labels are observed marker states, not measured Android command timing or game reaction timing, so training and live deployment have a command-to-visual domain gap. This is why there is no Armed integration in this experiment.

## Feature contract

`encode_action_history` is shared by training and the explicit-history inference runner. Given chronologically ordered input-frame timestamps, observation timestamp, initial pressed state, and *already known* transitions, it emits 17 floats for five frames: `(state, age_since_change / 500ms clipped, has_prior_change)` for every frame plus observation age and known flag. Before any change: `age=1`, `known=0`; an initial state at video time zero is **not** a transition. The current state must agree with the history or the sample is rejected. The inference caller must build the transition list **before** appending the action chosen at that observation. These are commanded/marker-based states, not proof of actual vehicle response.

Training uses each existing human-video sample exactly once, with its **factual** current state and action events from the existing `data/processed/v3/labels/<video-stem>.json` of the same video. Do not duplicate states counterfactually: the opposite current-state label is not a physically consistent history of these pixels. The original video-level train/validation/test split and RGB preprocessing are retained. Failed autonomous runs are only diagnostics, never demonstration labels.

The base visual encoder, existing state embedding and switch head are loaded from the V4-C2 checkpoint via the existing `StateConditionedModelRunner`, frozen and held in eval mode. A zero-initialized MLP predicts additive switch **logit** corrections from the frozen visual features, existing state embedding, and action-history vector. At initialization it is exactly the baseline; after training it is a **new model**, not a faithful replay of original weights.

## Training and evaluation

Run from a checkout containing the original `data/raw`, `data/processed/v3/samples.jsonl`, `data/processed/v3/labels`, frame cache (optional), and the original V4-C2 model directory:

```bash
python -m pytest -q tests/unit/test_v4c2_action_history_model.py
python scripts/train_v4c2_action_history.py --smoke --device cpu
python scripts/train_v4c2_action_history.py --device cpu
```

If using a new Git worktree, link `data` and `artifacts` from the original checkout first, after checking both names do not already exist. Select `--device cuda` only if PyTorch can see the desired GPU. Default output is `artifacts/models/mobilenet_v3_small_v4c2_history_residual_v1/`, containing **`history_adapter.pt`**, **`history_adapter_metadata.json`** (with base checkpoint and metadata SHA-256), and *no* `model.pt` that the old ADB runner could accidentally load. Existing output adapters are not overwritten. Smoke loads one train and one validation batch but saves nothing.

The script reports original and adjusted H200 binary cross entropy, plus false positives among the **recorded human** samples occurring within 100 ms of the last recorded switch. These are offline imitation metrics; they do not certify that any short reversal is useful, nor do they measure closed-loop progress. It reports the held-out test split only after selecting the best validation epoch. No new physical recording is required to run the smoke or training if original labeled videos are available.

`ActionHistoryAdapterRunner.predict_switch_all(...)` is an explicit-history **offline** inference interface. It checks that the base model and metadata SHA-256 match training, requires the exact five input-frame timestamps and already known transitions, and refuses to infer missing action history from imagery. It is intentionally *not* connected to `scripts/run_adb_closed_loop_v3.py`; do not pass the adapter file to that entry point.

## Next gating work

1. Run smoke against the *actual local* samples and cached/video data (not available to this development environment); resolve any data/metadata mismatch explicitly rather than relaxing validation.
2. Train the adapter and compare held-out H200 loss and post-switch FPR with the frozen baseline; inspect early-transient cases in both historically long and failing runs offline, always separating original-policy states from hypothetical changed-policy states.
3. Only if those checks are favorable, implement an opt-in runtime adapter with a command-time timeline and timed-action hook. Measure command/pixel latency, evaluate first divergence offline, and then perform separately named Armed tests. Keep original V4-C2 baseline unchanged.
