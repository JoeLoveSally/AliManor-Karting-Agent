#!/usr/bin/env python3
"""Replay ALL V4-C4 heads on existing Armed MP4 frames without ADB/touches.

Diagnostic only: auxiliary heads have not been validated on the live visual domain.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_adb_closed_loop as legacy  # noqa: E402
from diagnose_v4c4_recorded_input import (  # noqa: E402
    replay_recorded_steps,
    validate_run,
)
from karting_agent.model.event_time_runner import EventTimeActionRunner  # noqa: E402
from karting_agent.model.temporal_delta import transform_temporal_input_numpy  # noqa: E402
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402

SAMPLE_FRAMES = (320, 350, 370, 376, 390, 400, 410, 420, 430, 440, 444, 450)


def infer_all_heads(runner: EventTimeActionRunner, stack: np.ndarray) -> dict[str, Any]:
    """Match runner.predict_action input conversion, but retain every model output."""
    expected = (3 * runner.frame_stack, 224, 224)
    if stack.shape != expected or stack.dtype != np.float32:
        raise ValueError(f'expected float32 normalized RGB stack {expected}; got {stack.shape} {stack.dtype}')
    transformed = transform_temporal_input_numpy(
        stack, frame_stack=runner.frame_stack,
        representation=runner.input_representation,
    )
    torch = runner._torch
    tensor = torch.from_numpy(np.ascontiguousarray(transformed)).unsqueeze(0)
    tensor = tensor.to(device=runner.device, dtype=torch.float32)
    with torch.inference_mode():
        action_logit, event_logits, lateral, heading, edge_logit = runner.model(tensor)
        action_probability = float(torch.sigmoid(action_logit)[0].item())
        event_probabilities = torch.softmax(event_logits, dim=-1)[0].tolist()
        lateral_raw = float(lateral[0].item())
        heading_raw = [float(x) for x in heading[0].tolist()]
        edge_risk_probability = float(torch.sigmoid(edge_logit)[0].item())
    cos2, sin2 = heading_raw
    heading_norm = math.hypot(cos2, sin2)
    heading_half_angle_deg = math.degrees(math.atan2(sin2, cos2) / 2)
    return {
        'action_probability': action_probability,
        'event_time_probabilities': [float(x) for x in event_probabilities],
        'lateral_raw': lateral_raw,
        'heading_cos2_raw': cos2,
        'heading_sin2_raw': sin2,
        'heading_vector_norm': heading_norm,
        'heading_half_angle_deg_mod_180': heading_half_angle_deg,
        'edge_risk_probability': edge_risk_probability,
    }


def inspection_windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Index-based inspection windows; NOT ground-truth state labels."""
    boundaries = (
        ('frames_320_375', 320, 375),
        ('frames_376_399', 376, 399),
        ('frames_400_430', 400, 430),
        ('frames_431_444', 431, 444),
        ('frames_445_onward', 445, None),
    )
    windows: list[dict[str, Any]] = []
    for label, lo, hi in boundaries:
        subset = [row for row in rows if lo <= row['source_frame_index'] and (hi is None or row['source_frame_index'] <= hi)]
        if not subset:
            continue
        window: dict[str, Any] = {
            'name': label,
            'steps': len(subset),
            'first_frame_index': subset[0]['source_frame_index'],
            'last_frame_index': subset[-1]['source_frame_index'],
        }
        for key in ('action_probability', 'lateral_raw', 'heading_half_angle_deg_mod_180', 'heading_vector_norm', 'edge_risk_probability'):
            values = [row[key] for row in subset]
            window[key] = {
                'first': values[0], 'last': values[-1],
                'minimum': min(values), 'maximum': max(values),
                'median': float(np.median(values)),
            }
        windows.append(window)
    return windows


def select_inspection_frames(rows: list[dict[str, Any]], targets: tuple[int, ...] = SAMPLE_FRAMES) -> list[dict[str, Any]]:
    """Closest observed model step to each target, deduplicated."""
    if not rows:
        return []
    selected = {}
    for target in targets:
        row = min(rows, key=lambda entry: (abs(entry['source_frame_index'] - target), entry['source_frame_index']))
        selected[row['source_frame_index']] = row
    return [selected[index] for index in sorted(selected)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='No-ADB multi-head V4-C4 replay of an existing Armed run')
    parser.add_argument('--run-json', type=Path, required=True)
    parser.add_argument('--video', type=Path, default=None)
    parser.add_argument('--model', type=Path, default=ROOT / 'artifacts/models/mobilenet_v3_small_v4c4_event_time/model.pt')
    parser.add_argument('--metadata', type=Path, default=None)
    parser.add_argument('--train-config', type=Path, default=ROOT / 'configs/train_v4c4_event_time.yaml')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--torch-num-threads', type=int, default=4)
    parser.add_argument('--decision-threshold', type=float, default=0.5)
    parser.add_argument('--max-action-replay-error', type=float, default=0.05)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.decision_threshold < 1 or args.max_action_replay_error < 0:
        raise ValueError('invalid action replay comparison threshold')
    run_path = args.run_json.resolve()
    run = json.loads(run_path.read_text(encoding='utf-8'))
    steps = validate_run(run)
    video_path = (args.video or run_path.with_suffix('.mp4')).resolve()
    preprocess = preprocess_config_from_mapping(legacy.load_mapping(args.train_config))
    runner = EventTimeActionRunner(
        args.model, metadata_path=args.metadata,
        device=args.device, torch_num_threads=args.torch_num_threads,
    )
    heads: list[dict[str, Any]] = []

    def collect_heads(stack: np.ndarray) -> float:
        result = infer_all_heads(runner, stack)
        heads.append(result)
        return result['action_probability']

    replay = replay_recorded_steps(
        video_path, steps, predict_action=collect_heads, preprocess_config=preprocess,
    )
    if len(replay) != len(steps) or len(heads) != len(steps):
        raise RuntimeError('replay/head count does not match recorded step count')
    rows = []
    for step, action_replay, output_heads in zip(steps, replay, heads):
        rows.append({
            'source_frame_index': int(step['source_frame_index']),
            'observation_timestamp_ms': float(step['observation_timestamp_ms']),
            'history_frame_indices': [int(i) for i in step['history_frame_indices']],
            'logged_action_probability': float(step['action_probability']),
            'action_replay_absolute_error': float(action_replay['absolute_error']),
            'logged_pressed_state': bool(step['pressed']),
            **output_heads,
        })
    max_error = max(row['action_replay_absolute_error'] for row in rows)
    threshold_disagreements = [row['source_frame_index'] for row in rows if
        (row['logged_action_probability'] >= args.decision_threshold)
        != (row['action_probability'] >= args.decision_threshold)]
    passed = max_error <= args.max_action_replay_error and not threshold_disagreements
    result = {
        'status': 'completed' if passed else 'action_replay_inconsistent',
        'steps': len(rows),
        'action_replay_max_absolute_error': max_error,
        'action_replay_threshold_disagreement_frames': threshold_disagreements,
        'action_replay_passed': passed,
        'run_json': str(run_path),
        'recording': str(video_path),
        'model': str(Path(args.model).resolve()),
        'metadata': str(runner.metadata_path),
        'head_semantics': {
            'lateral_raw': 'regression output; labels use offset divided by road half width; live sign/accuracy unverified',
            'heading': 'unconstrained raw cos(2*heading_error), sin(2*heading_error) regression; half-angle is modulo 180 degrees; output vector norm shows inconsistency with a unit-vector target',
            'edge_risk_probability': 'sigmoid of edge-risk logit; trained target is abs(lateral)>=0.70; not calibrated or validated on this live domain',
            'event_time_probabilities': 'softmax next-expert-transition bins, not used in existing runtime; not a safe automatic intervention signal',
        },
        'note': 'Index windows are for inspection, NOT independent labels of on/off-road state. MP4 replay checks model outputs on recorded pixels, not capture freshness, geometry ground truth or recovery ability. Do not auto-enable auxiliary heads in control.',
        'inspection_frames': select_inspection_frames(rows),
        'inspection_windows': inspection_windows(rows),
        'rows': rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"status={result['status']} steps={len(rows)} max_action_replay_error={max_error:.9g} threshold_disagreements={len(threshold_disagreements)}")
    for row in result['inspection_frames']:
        print('frame={source_frame_index} action={action_probability:.5f} lateral={lateral_raw:+.4f} heading_deg={heading_half_angle_deg_mod_180:+.1f} heading_norm={heading_vector_norm:.3f} edge_risk={edge_risk_probability:.4f}'.format(**row))
    print(f'output={output}')
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
