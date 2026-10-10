# Freeze-parity diagnosis: CPU vs CUDA original Validation reference

## Problem

The first shadow replay stopped on `rgb: teacher ... all.predicted`.
The original tiny RGB model was trained and validated on **CUDA:0** in
`batch_size=16`. Rebuilding the original full-Validation batch order on CPU
still yields `all.predicted=76` versus frozen CUDA metadata `78`.
The device/backend hypothesis is **not yet proven**.

Do not remove the strict reference check, retune threshold=0.5, run Test or
change model weights to make these values agree.

## Read-only diagnostic

`scripts/diagnose_tiny_policy_backend_parity.py` runs the ORIGINAL
`v4c2_temporal_v2` two-video Validation dataset, same 96x96 masked
frames, same 5-frame controls, batch size read from frozen training metadata,
on BOTH CPU and a *limited* CUDA allocation. It reports:

- Original frozen Validation GT/predicted/matched/FP/FN event counts
- CPU canonical event counts and CUDA canonical event counts
- Whether each device matches the original counts exactly
- Max probability discrepancy, number of samples crossing fixed p=0.5
  between CPU/CUDA, and their actual target timestamps/probabilities

The same `sequence_evaluator` and 100ms matching tolerance are used.
The script performs **NO training, simulator scheduling, Test evaluation,
file writes, or Armed/ADB action**. It refuses to proceed if CUDA cannot be
acquired, rather than silently calling CPU a CUDA comparison. The PyTorch
CUDA fraction is only a caching-allocator ceiling, not GPU memory isolation;
the co-resident vLLM can still cause preflight OOM.

## Execute

**WSL:**

```bash
cd /home/jianqiao/workspace/AliManor-Karting-Agent-v4c2-history
git pull --ff-only origin exp/v4c2-action-history-residual
rspark \
  ./src/karting_agent/train/shadow_teacher.py \
  ./src/karting_agent/train/shadow_backend_parity.py \
  ./scripts/diagnose_tiny_policy_backend_parity.py \
  ./tests/unit/test_shadow_backend_parity.py
```

**Spark:**

```bash
cd /home/grg/Workspace/AliManor-Karting-Agent
python -m pytest -q \
  tests/unit/test_shadow_backend_parity.py \
  tests/unit/test_shadow_teacher.py \
  tests/unit/test_shadow_teacher_replay.py

set -o pipefail
python scripts/diagnose_tiny_policy_backend_parity.py \
  --checkpoint-dir /tmp/tiny_policy_export \
  --mode rgb \
  --cuda-memory-fraction 0.03 \
  --min-cuda-free-mib 2048 \
  --cpu-threads 4 \
  2>&1 | tee /tmp/tiny_policy_backend_parity.log
```

Inspect `expected`, `cpu`, `cuda`, `cpu_cuda_comparison`.

**Decision tree:**

- If `cuda_matches_frozen_counts=true` and CPU differs: GPU-vs-CPU
  numerical differences are supported. We can then run both canonical
  teacher and sequential self-fed inference on the same restricted CUDA
  backend and keep strict parity.
- If CUDA and CPU both yield 76: this is not explained simply by
  CPU-vs-CUDA backend. Investigate source-frame hashes, preprocessing,
  package/library versions, original training metadata and model tensors.
- If CUDA returns other counts: inspect differences and avoid deploying
  further shadow results until the mismatch is resolved.

**Do not rerun the failed shadow report yet**; no valid results exist
from its aborted execution. Do not run HSV until RGB cause is understood.

## Local verification

In this environment, pure CPU parity helpers passed 12/12 tests with
synthetic predictions, including 0.5-threshold crossing detection.
True CUDA integration and original expert-video replay cannot be tested
here because this environment has no CUDA GPU or source Validation MP4s.
