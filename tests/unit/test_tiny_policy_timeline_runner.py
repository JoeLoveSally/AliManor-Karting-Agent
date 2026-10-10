"""Test standalone CLI's preflight and metric parity with an AST-loaded module.

AST extraction permits CPU-only testing without importing Spark-only modules.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

SRC = Path(__file__).resolve().parents[2] / 'scripts/audit_tiny_policy_timeline.py'
AST = ast.parse(SRC.read_text(encoding='utf-8'))
TARGETS = {'sha256','check_artifacts','compare_reproduced_metric','output_paths'}
FUNCTIONS = [x for x in AST.body if isinstance(x, ast.FunctionDef) and x.name in TARGETS]


def lib():
    ns = {'Path': Path, 'hashlib': hashlib, 'json': json}
    code = compile(ast.Module(body=FUNCTIONS, type_ignores=[]), str(SRC), 'exec')
    exec(code, ns)
    return SimpleNamespace(**ns)


def split():
    return SimpleNamespace(name='frozen', test=('test_a.mp4','test_b.mp4'),
                           train=('train.mp4',), validation=('val.mp4',))


def run_data(path):
    checkpoint = path/'tiny_policy_rgb.pt'
    checkpoint.write_bytes(b'example frozen weights')
    metadata = path/'tiny_policy_rgb.json'
    meta = {'mode':'rgb'}
    metadata.write_text(json.dumps(meta))
    l=lib()
    report={'kind':'frozen_tiny_policy_heldout_test',
            'test_videos':list(split().test),'split':split().name,
            'threshold':.5,'transition_tolerance_ms':100.,
            'prediction_horizon_ms':100.,'expert_teacher_forced':True,
            'results':{'rgb':{'checkpoint_sha256':l.sha256(checkpoint),
                              'training_report_sha256':l.sha256(metadata)}}}
    return report


def test_artifact_preflight_rejects_unverified_checkpoint(tmp_path):
    # Patch the on-disk report/protocol check without touching a real model.
    namespace={'Path':Path,'hashlib':hashlib,'json':json,
               'require_frozen_checkpoint':lambda meta,mode,split:None,
               'MODES':('rgb',)}
    exec(compile(ast.Module(body=FUNCTIONS,type_ignores=[]),str(SRC),'exec'),namespace)
    chk=namespace['check_artifacts']
    heldout=run_data(tmp_path)
    chk(tmp_path,heldout,split())
    (tmp_path/'tiny_policy_rgb.pt').write_bytes(b'mutated')
    with pytest.raises(ValueError,match='differs'):
        chk(tmp_path,heldout,split())


def test_protocol_rejects_nonfrozen_or_changed_window(tmp_path):
    namespace={'Path':Path,'hashlib':hashlib,'json':json,
               'require_frozen_checkpoint':lambda meta,mode,split:None,
               'MODES':('rgb',)}
    exec(compile(ast.Module(body=FUNCTIONS,type_ignores=[]),str(SRC),'exec'),namespace)
    info=run_data(tmp_path)
    info['transition_tolerance_ms']=50.
    with pytest.raises(ValueError,match='protocol'):
        namespace['check_artifacts'](tmp_path,info,split())


def test_output_paths_are_flat_and_do_not_include_models(tmp_path):
    files=lib().output_paths(tmp_path,['data/raw/a.mp4','data/raw/b.mp4'])
    assert len(files)==7
    assert all(f.parent==tmp_path for f in files)
    assert all(f.suffix in ('.json','.png','.jpg','.csv') for f in files)


def test_compare_replayed_counts_guards_anonymous_metric_change():
    sample={'transition':{'all':{'ground_truth':2,'predicted':2,'matched':1,'false_positive':1,'false_negative':1},
                          'press':{'ground_truth':1,'predicted':1,'matched':1,'false_positive':0,'false_negative':0},
                          'release':{'ground_truth':1,'predicted':1,'matched':0,'false_positive':1,'false_negative':1}},
            'release_segment_recall':{'short_100_300ms':{'segments':1,'detected':0,'recall':0.}}}
    reference={'results':{'rgb':{'per_video':{'A':{'sequence':{'candidate':sample}}}}}}
    ev=SimpleNamespace(summary=lambda:sample)
    lib().compare_reproduced_metric(reference,'rgb','A',ev)
    changed=json.loads(json.dumps(sample))
    changed['transition']['all']['matched']=0
    with pytest.raises(ValueError,match='diverges'):
        lib().compare_reproduced_metric(reference,'rgb','A',SimpleNamespace(summary=lambda:changed))
