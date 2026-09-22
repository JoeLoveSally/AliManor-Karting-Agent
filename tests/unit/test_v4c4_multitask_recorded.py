from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    from diagnose_v4c4_multitask_recorded import (  # noqa: E402
        infer_all_heads,
        inspection_windows,
        select_inspection_frames,
    )
finally:
    sys.path.remove(str(SCRIPTS))


class FixedHeads(torch.nn.Module):
    def forward(self, tensor):
        assert tuple(tensor.shape) == (1, 15, 224, 224)
        return (
            torch.tensor([0.0]),
            torch.tensor([[0.0, 1.0, 2.0]]),
            torch.tensor([0.8]),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([2.0]),
        )


def test_infer_all_heads_decodes_logits_and_preserves_input() -> None:
    stack = np.zeros((15, 224, 224), dtype=np.float32)
    unchanged = stack.copy()
    runner = SimpleNamespace(
        frame_stack=5,
        input_representation='current_rgb_plus_adjacent_deltas',
        device=torch.device('cpu'),
        _torch=torch,
        model=FixedHeads(),
    )
    result = infer_all_heads(runner, stack)
    assert result['action_probability'] == pytest.approx(0.5)
    assert result['lateral_raw'] == pytest.approx(0.8)
    assert result['heading_cos2_raw'] == pytest.approx(0.0)
    assert result['heading_sin2_raw'] == pytest.approx(1.0)
    assert result['heading_vector_norm'] == pytest.approx(1.0)
    assert result['heading_half_angle_deg_mod_180'] == pytest.approx(45.0)
    assert result['edge_risk_probability'] == pytest.approx(0.880797, abs=1e-6)
    assert sum(result['event_time_probabilities']) == pytest.approx(1.0)
    np.testing.assert_array_equal(stack, unchanged)


def test_infer_all_heads_rejects_unexpected_shape_and_dtype() -> None:
    runner = SimpleNamespace(frame_stack=5)
    with pytest.raises(ValueError, match='normalized RGB stack'):
        infer_all_heads(runner, np.zeros((12, 224, 224), dtype=np.float32))
    with pytest.raises(ValueError, match='normalized RGB stack'):
        infer_all_heads(runner, np.zeros((15, 224, 224), dtype=np.float64))


def test_inspection_windows_keep_index_boundaries_and_no_empty_windows() -> None:
    rows = [dict(source_frame_index=index, action_probability=0.7, lateral_raw=-0.3,
                 heading_half_angle_deg_mod_180=12.0, heading_vector_norm=0.6,
                 edge_risk_probability=0.2)
            for index in (320, 376, 400, 430, 431, 444, 445)]
    summary = inspection_windows(rows)
    assert [item['name'] for item in summary] == [
        'frames_320_375', 'frames_376_399', 'frames_400_430',
        'frames_431_444', 'frames_445_onward',
    ]
    assert summary[2]['steps'] == 2
    assert summary[3]['steps'] == 2
    assert summary[4]['lateral_raw']['median'] == pytest.approx(-0.3)
    assert inspection_windows([]) == []


def test_inspection_frames_choose_closest_unique_observation() -> None:
    rows = [{'source_frame_index': 320}, {'source_frame_index': 376},
            {'source_frame_index': 390}, {'source_frame_index': 444}]
    assert [row['source_frame_index'] for row in select_inspection_frames(rows, (321, 322, 381, 441))] == [320, 376, 444]
    assert select_inspection_frames([]) == []
