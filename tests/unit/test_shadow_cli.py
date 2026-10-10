"""Protocol tests for Validation-only CLI (no access to Spark source videos)."""
from __future__ import annotations

import ast
import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE=Path(__file__).resolve().parents[2] / 'scripts/audit_tiny_policy_shadow.py'
TREE=ast.parse(SOURCE.read_text(encoding='utf-8'))
TARGETS={'sha256','planned_output_paths','compare_frozen_validation','write_csv'}
SELECTED=[node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name in TARGETS]
NAMESPACE={'Path':Path,'hashlib':hashlib,'csv':csv,'RESULT':'tiny_policy_shadow_validation.json','MODES':('rgb','rgb_hsv')}
exec(compile(ast.Module(body=SELECTED,type_ignores=[]),str(SOURCE),'exec'),NAMESPACE)


def test_output_locations_are_flat_and_validation_only(tmp_path):
    split=SimpleNamespace(validation=('data/raw/val1.mp4','data/raw/val2.mp4'),
                          test=('data/raw/test1.mp4',))
    result=NAMESPACE['planned_output_paths'](tmp_path,split)
    assert len(result)==5
    assert len(set(result))==5
    assert all(path.parent==tmp_path for path in result)
    assert all('test1' not in str(path) for path in result)
    assert any(path.name=='tiny_shadow_rgb_val1.csv' for path in result)
    assert any(path.name=='tiny_shadow_rgb_hsv_val2.csv' for path in result)


def test_cannot_overwrite_existing_csv(tmp_path):
    row={
        'observation_ms':200.,'target_ms':300.,'expert_current_pressed':False,
        'simulated_current_pressed':False,'expert_future_pressed':True,
        'teacher_probability':.2,'shadow_probability':.8,
        'shadow_desired_pressed':True,'inference_ready_ms':200.,
        'simulated_execution_ms':300.,'simulated_executed_at_target':True,
        'executed_matches_expert_at_target':True,
        'shadow_current_action_age_ms':200.,'per_sample_forward_ms':3.,
    }
    out=tmp_path/'trace.csv'
    NAMESPACE['write_csv'](out,[row])
    with out.open(newline='') as f:
        records=list(csv.DictReader(f))
    assert records[0]['target_ms']=='300.0'
    assert records[0]['simulated_execution_ms']=='300.0'
    with pytest.raises(FileExistsError):
        NAMESPACE['write_csv'](out,[row])


def test_validation_metrics_parity_rejects_changed_match_counts():
    def seq(matched):
        return {'transition':{
            k: {'ground_truth':10,'predicted':12,'matched':matched,
                'false_positive':12-matched,'false_negative':10-matched}
            for k in ('all','press','release')},
            'release_segment_recall':{
                k:{'segments':3,'detected':2,'recall':2/3}
                for k in ('short_100_300ms','100_200ms','200_300ms')},
        }
    original={'validation_sequence':{'candidate':seq(8)}}
    NAMESPACE['compare_frozen_validation'](original,SimpleNamespace(summary=lambda:seq(8)),mode='rgb')
    with pytest.raises(ValueError,match='diverged'):
        NAMESPACE['compare_frozen_validation'](
            original,SimpleNamespace(summary=lambda:seq(9)),mode='rgb')
    changed=seq(8)
    changed['release_segment_recall']['short_100_300ms']['detected']=1
    with pytest.raises(ValueError,match='short-release parity'):
        NAMESPACE['compare_frozen_validation'](
            original,SimpleNamespace(summary=lambda:changed),mode='rgb')


def test_cli_has_no_test_dataset_call_or_optimizer_or_armed_side_effects():
    source=SOURCE.read_text(encoding='utf-8')
    assert "['validation']" in source
    assert "['test']" not in source
    assert 'optimizer' not in source
    assert 'adb' not in source.lower().split('def main():',1)[1]
    assert 'compare_frozen_validation' in source
    assert "model.load_state_dict(state,strict=True)" in source
    assert "torch.load(item['weight_path'],map_location='cpu',weights_only=True)" in source
