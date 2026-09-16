#!/usr/bin/env python3
"""Inspect which image regions drive transition-policy predictions.

This is an architecture-agnostic occlusion-sensitivity diagnostic. It reads the
same temporal source-frame indices saved by ``run_adb_closed_loop.py`` or
``replay_video.py``, rebuilds the model input, masks one spatial grid cell across
all temporal frames, and measures the resulting probability change.

For state-conditioned policies the probability is KEEP/SWITCH probability, so
the current control state is reconstructed from each RuntimeStep before
recomputing a saved prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.model.runner import ModelRunner  # noqa: E402
from karting_agent.model.state_conditioned_runner import (  # noqa: E402
    StateConditionedModelRunner,
    _SUPPORTED_MODEL_FAMILIES,
)
from karting_agent.vision.preprocess import (  # noqa: E402
    PreprocessConfig,
    prepare_frame,
    stack_prepared_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate occlusion-sensitivity maps for an ADB run or replay artifact."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--source-frame",
        type=int,
        action="append",
        default=None,
        help=(
            "Analyze a specific latest temporal source-frame index. May be "
            "repeated. By default all state-changing steps are analyzed."
        ),
    )
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_run(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid run JSON: {path}")
    if isinstance(raw.get("steps"), list):
        return raw
    replay = raw.get("replay")
    if isinstance(replay, dict) and isinstance(replay.get("steps"), list):
        return replay
    raise ValueError(f"run JSON contains no steps: {path}")


def selected_steps(
    run: dict[str, object],
    requested_frames: list[int] | None,
) -> list[dict[str, object]]:
    steps = [step for step in run["steps"] if isinstance(step, dict)]
    if requested_frames is None:
        selected = [step for step in steps if step.get("action") != "HOLD"]
        if not selected:
            raise ValueError("run contains no state-changing steps")
        return selected

    by_latest_frame: dict[int, dict[str, object]] = {}
    for step in steps:
        indices = step.get("input_frame_indices")
        if isinstance(indices, list) and indices:
            by_latest_frame[int(indices[-1])] = step

    selected: list[dict[str, object]] = []
    for frame_index in requested_frames:
        step = by_latest_frame.get(frame_index)
        if step is None:
            raise ValueError(
                f"no RuntimeStep whose latest input frame is {frame_index}"
            )
        selected.append(step)
    return selected


def read_video_frames(path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    """Decode requested MP4 frames sequentially using decoder-order indices."""
    requested = sorted(set(int(index) for index in frame_indices))
    if not requested:
        return {}
    if requested[0] < 0:
        raise ValueError("frame indices must be >= 0")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open debug video: {path}")

    wanted = set(requested)
    last = requested[-1]
    frames: dict[int, np.ndarray] = {}
    try:
        frame_index = 0
        while frame_index <= last:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index in wanted:
                frames[frame_index] = frame.copy()
                if len(frames) == len(wanted):
                    break
            frame_index += 1
    finally:
        capture.release()

    missing = [index for index in requested if index not in frames]
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise RuntimeError(
            f"debug video ended before source frames were decoded: {preview}{suffix}"
        )
    return frames


def load_diagnostic_runner(
    model_path: Path,
    *,
    metadata_path: Path | None,
    device: str | None,
):
    """Load either a legacy temporal classifier or a state-conditioned policy."""
    resolved_model = Path(model_path).resolve()
    resolved_metadata = (
        Path(metadata_path).resolve()
        if metadata_path is not None
        else resolved_model.with_name("metadata.json")
    )
    metadata = json.loads(resolved_metadata.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("model metadata root must be a mapping")

    model_family = str(metadata.get("model_family", ""))
    if model_family in _SUPPORTED_MODEL_FAMILIES:
        return StateConditionedModelRunner(
            resolved_model,
            metadata_path=resolved_metadata,
            device=device,
        )
    return ModelRunner(
        resolved_model,
        metadata_path=resolved_metadata,
        device=device,
    )


def pre_action_pressed(step: dict[str, object]) -> bool:
    """Recover the control state supplied to the model before this decision."""
    action = str(step.get("action", "HOLD")).upper()
    post_action_pressed = bool(step.get("pressed", False))
    if action == "PRESS":
        return False
    if action == "RELEASE":
        return True
    return post_action_pressed


def predict_probability(
    runner,
    inputs: np.ndarray,
    *,
    current_pressed: bool,
) -> float:
    if isinstance(runner, StateConditionedModelRunner):
        return runner.predict_switch(inputs, current_pressed=current_pressed)
    return runner.predict(inputs)


def occlusion_valid_mask(
    config: PreprocessConfig,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Return prepared-image pixels that may be perturbed by occlusion."""
    valid = np.ones((height, width), dtype=bool)
    rois = list(config.mask_rois)
    if config.mask_touch_area:
        rois.append(config.touch_roi)

    for x0, y0, x1, y1 in rois:
        left = max(0, min(width, round(x0 * width)))
        top = max(0, min(height, round(y0 * height)))
        right = max(left, min(width, round(x1 * width)))
        bottom = max(top, min(height, round(y1 * height)))
        valid[top:bottom, left:right] = False
    return valid


def occlusion_sensitivity(
    runner,
    prepared_frames: list[np.ndarray],
    *,
    current_pressed: bool,
    grid: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    if grid < 2:
        raise ValueError("--grid must be >= 2")

    config = runner.spec.preprocess_config
    baseline_input = stack_prepared_frames(prepared_frames, config)
    baseline = predict_probability(
        runner,
        baseline_input,
        current_pressed=current_pressed,
    )
    height, width = prepared_frames[-1].shape[:2]
    sensitivity = np.zeros((grid, grid), dtype=np.float32)
    unmasked_fraction = np.zeros((grid, grid), dtype=np.float32)
    fill_rgb = np.rint(np.asarray(config.mean) * 255.0).astype(np.uint8)
    valid_mask = occlusion_valid_mask(config, height=height, width=width)

    y_edges = np.rint(np.linspace(0, height, grid + 1)).astype(int)
    x_edges = np.rint(np.linspace(0, width, grid + 1)).astype(int)

    for row in range(grid):
        y0, y1 = y_edges[row], y_edges[row + 1]
        for col in range(grid):
            x0, x1 = x_edges[col], x_edges[col + 1]
            cell_valid = valid_mask[y0:y1, x0:x1]
            unmasked_fraction[row, col] = float(cell_valid.mean())
            if not cell_valid.any():
                continue

            occluded = [frame.copy() for frame in prepared_frames]
            for frame in occluded:
                patch = frame[y0:y1, x0:x1]
                patch[cell_valid] = fill_rgb
            probability = predict_probability(
                runner,
                stack_prepared_frames(occluded, config),
                current_pressed=current_pressed,
            )
            sensitivity[row, col] = baseline - probability

    return baseline, sensitivity, unmasked_fraction


def render_heatmap(image_rgb: np.ndarray, sensitivity: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    max_abs = float(np.max(np.abs(sensitivity)))
    if max_abs <= 1e-12:
        normalized = np.zeros_like(sensitivity, dtype=np.float32)
    else:
        normalized = sensitivity / max_abs

    expanded = cv2.resize(
        normalized,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    overlay = image_rgb.astype(np.float32) * 0.55

    # RGB: red = SWITCH evidence, blue = KEEP evidence.
    positive = np.clip(expanded, 0.0, 1.0)[..., None]
    negative = np.clip(-expanded, 0.0, 1.0)[..., None]
    overlay += positive * np.array([115.0, 0.0, 0.0], dtype=np.float32)
    overlay += negative * np.array([0.0, 0.0, 115.0], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def top_cells(
    sensitivity: np.ndarray,
    unmasked_fraction: np.ndarray,
    *,
    current_pressed: bool,
    count: int = 5,
) -> dict[str, list[dict[str, object]]]:
    grid = sensitivity.shape[0]
    cells: list[dict[str, object]] = []
    for row in range(grid):
        for col in range(grid):
            if unmasked_fraction[row, col] <= 0.0:
                continue
            cells.append(
                {
                    "row": row,
                    "col": col,
                    "x0": col / grid,
                    "y0": row / grid,
                    "x1": (col + 1) / grid,
                    "y1": (row + 1) / grid,
                    "unmasked_fraction": float(unmasked_fraction[row, col]),
                    "delta_switch_probability": float(sensitivity[row, col]),
                }
            )

    switch_evidence = sorted(
        cells,
        key=lambda item: float(item["delta_switch_probability"]),
        reverse=True,
    )[:count]
    keep_evidence = sorted(
        cells,
        key=lambda item: float(item["delta_switch_probability"]),
    )[:count]
    if current_pressed:
        press_evidence = keep_evidence
        release_evidence = switch_evidence
    else:
        press_evidence = switch_evidence
        release_evidence = keep_evidence
    return {
        "switch_evidence": switch_evidence,
        "keep_evidence": keep_evidence,
        "press_evidence": press_evidence,
        "release_evidence": release_evidence,
    }


def labeled_pair(
    image_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    *,
    title: str,
) -> np.ndarray:
    pair = np.concatenate((image_rgb, heatmap_rgb), axis=1)
    label_height = 34
    canvas = np.zeros(
        (pair.shape[0] + label_height, pair.shape[1], 3),
        dtype=np.uint8,
    )
    canvas[label_height:] = pair
    cv2.putText(
        canvas,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> int:
    args = parse_args()
    if args.grid < 2:
        raise ValueError("--grid must be >= 2")

    run_path = args.run_json.resolve()
    video_path = args.video.resolve()
    runner = load_diagnostic_runner(
        args.model,
        metadata_path=args.metadata,
        device=args.device,
    )
    run = load_run(run_path)
    steps = selected_steps(run, args.source_frame)

    all_indices: list[int] = []
    for step in steps:
        indices = step.get("input_frame_indices")
        if not isinstance(indices, list) or len(indices) != runner.spec.frame_stack:
            raise ValueError("RuntimeStep frame indices do not match model frame_stack")
        all_indices.extend(int(index) for index in indices)
    source_frames = read_video_frames(video_path, all_indices)

    output_path = (
        args.output.resolve()
        if args.output is not None
        else ROOT / "artifacts" / "analysis" / f"focus_{run_path.stem}.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.with_suffix(".json")

    rows: list[np.ndarray] = []
    report: list[dict[str, object]] = []
    for step in steps:
        indices = [int(index) for index in step["input_frame_indices"]]
        prepared = [
            prepare_frame(source_frames[index], runner.spec.preprocess_config)
            for index in indices
        ]
        current_pressed = pre_action_pressed(step)
        recomputed, sensitivity, unmasked_fraction = occlusion_sensitivity(
            runner,
            prepared,
            current_pressed=current_pressed,
            grid=args.grid,
        )
        saved_probability = float(step["probability"])
        mismatch = abs(recomputed - saved_probability)
        latest = indices[-1]
        heatmap = render_heatmap(prepared[-1], sensitivity)
        action = str(step.get("action", "?"))
        state_label = "P" if current_pressed else "R"
        title = (
            f"src={latest} state={state_label} action={action} "
            f"saved={saved_probability:.3f} recomputed={recomputed:.3f} "
            f"diff={mismatch:.3f}"
        )
        rows.append(labeled_pair(prepared[-1], heatmap, title=title))
        report.append(
            {
                "source_frame": latest,
                "input_frame_indices": indices,
                "observation_timestamp_ms": float(step["observation_timestamp_ms"]),
                "action": action,
                "current_pressed": current_pressed,
                "pressed_after_action": bool(step["pressed"]),
                "saved_probability": saved_probability,
                "recomputed_probability": recomputed,
                "absolute_probability_mismatch": mismatch,
                "grid": args.grid,
                **top_cells(
                    sensitivity,
                    unmasked_fraction,
                    current_pressed=current_pressed,
                ),
            }
        )
        print(title)

    sheet_rgb = np.concatenate(rows, axis=0)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(sheet_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write diagnostic image: {output_path}")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Focus image: {output_path}")
    print(f"Focus data:  {report_path}")
    print("Legend: red supports SWITCH; blue supports KEEP.")
    print("For state=P: SWITCH=RELEASE. For state=R: SWITCH=PRESS.")
    print("Fixed HUD/touch-mask pixels are preserved during occlusion.")
    print(
        "Important: if saved/recomputed probability differs materially, "
        "do not interpret the heatmap until recording/frame alignment is fixed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
