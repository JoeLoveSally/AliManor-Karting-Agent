#!/usr/bin/env python3
"""Audit frozen RGB vs RGB+HSV held-out timelines; never modify a checkpoint.

One-shot event diagnostics. No training, threshold selection, Armed or ADB.
Test has already been evaluated once; this only explains those frozen results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'src', ROOT / 'scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiment_tiny_temporal_policy import (  # noqa: E402
    TemporalExpertDataset, cache_required_frames, label_events,
    points_from_predictions, validate_sample,
)
from inspect_geometry_pseudo_labels import load_config as load_road_config  # noqa: E402
from karting_agent.model.tiny_temporal_policy import TinyTemporalPolicy  # noqa: E402
from karting_agent.train.heldout_guard import require_frozen_checkpoint  # noqa: E402
from karting_agent.train.sequence_evaluator import (  # noqa: E402
    ReleaseSegment, Transition, evaluate_sequence,
)
from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402
from karting_agent.train.tiny_policy_timeline import (  # noqa: E402
    MODES, build_focus_cases, event_diagnostics, probabilities_at_points,
    render_focus_contact, render_timeline,
)
from karting_agent.train.trainer import load_video_split, partition_samples  # noqa: E402
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_artifacts(checkpoint_dir, heldout, split):
    if heldout.get('kind') != 'frozen_tiny_policy_heldout_test':
        raise ValueError('not a frozen held-out test report')
    if (heldout.get('test_videos') != list(split.test) or
            heldout.get('split') != split.name or
            heldout.get('threshold') != .5 or
            heldout.get('transition_tolerance_ms') != 100.0 or
            heldout.get('prediction_horizon_ms') != 100.0 or
            heldout.get('expert_teacher_forced') is not True):
        raise ValueError('held-out report protocol mismatch')
    reference = heldout.get('results', {})
    for mode in MODES:
        ck = checkpoint_dir / f'tiny_policy_{mode}.pt'
        meta_path = checkpoint_dir / f'tiny_policy_{mode}.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        require_frozen_checkpoint(meta, mode=mode, split=split)
        if (sha256(ck) != reference[mode]['checkpoint_sha256'] or
                sha256(meta_path) != reference[mode]['training_report_sha256']):
            raise ValueError(f'{mode} checkpoint/report differs from frozen Test')


def compare_reproduced_metric(heldout, mode, video, evaluation):
    reference = heldout['results'][mode]['per_video'][video]['sequence']['candidate']
    current = evaluation.summary()
    for direction in ('all','press','release'):
        for field in ('ground_truth','predicted','matched','false_positive','false_negative'):
            if current['transition'][direction][field] != reference['transition'][direction][field]:
                raise ValueError(f'{mode}/{video}: replay diverges on {direction}.{field}')
    if (current['release_segment_recall']['short_100_300ms'] !=
            reference['release_segment_recall']['short_100_300ms']):
        raise ValueError(f'{mode}/{video}: replay diverges on short-release counts')


def output_paths(output_dir, videos):
    files = [output_dir / 'tiny_policy_timeline_audit.json']
    for video in videos:
        stem = Path(video).stem
        files.extend((
            output_dir / f'tiny_timeline_{stem}.png',
            output_dir / f'tiny_focus_{stem}.jpg',
            output_dir / f'tiny_trace_{stem}.csv',
        ))
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path,
                    default=ROOT/'configs/train_v4c2_temporal_v2.yaml')
    ap.add_argument('--samples', type=Path,
                    default=ROOT/'data/processed/v3/samples.jsonl')
    ap.add_argument('--labels-dir', type=Path,
                    default=ROOT/'data/processed/v3/labels')
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--heldout-report', type=Path, default=None)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--cpu-threads', type=int, default=4)
    args = ap.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        ap.error('batch-size and CPU threads must be positive')
    if not args.checkpoint_dir.is_dir() or not args.output_dir.is_dir():
        ap.error('checkpoint-dir and output-dir must already exist')
    reference_path = (args.heldout_report or
                      args.checkpoint_dir/'tiny_policy_heldout_test.json')
    heldout = json.loads(reference_path.read_text(encoding='utf-8'))
    split = load_video_split(args.config)
    check_artifacts(args.checkpoint_dir, heldout, split)
    paths = output_paths(args.output_dir, split.test)
    collisions = [str(p) for p in paths if p.exists()]
    if collisions:
        ap.error(f'refusing to overwrite diagnostic outputs: {collisions}')
    samples = partition_samples(load_v3_samples(args.samples), split)['test']
    if len(samples) != heldout['test_samples']:
        raise ValueError('test sample count differs from frozen report')
    events = {v: label_events(args.labels_dir,v) for v in split.test}
    for sample in samples:
        validate_sample(sample, events[sample.video])
    torch.set_num_threads(args.cpu_threads)
    print('Read-only expert Test replay; frozen weights, CPU, threshold=0.5, '
          'tolerance=100ms', flush=True)
    raw = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    prep = preprocess_config_from_mapping(raw)
    if not prep.mask_touch_area:
        raise ValueError('touch-marker area must be masked')
    road_cfg,_,_ = load_road_config(ROOT/'configs/geometry_pseudo_labels.yaml')
    frames = cache_required_frames(samples,preprocess=prep,road_config=road_cfg)
    by_video: dict[str,list] = defaultdict(list)
    for sample in samples:
        by_video[sample.video].append(sample)
    points_by_video: dict[str,dict] = {v:{} for v in split.test}
    evaluations: dict[str,dict] = {v:{} for v in split.test}
    for mode in MODES:
        ck = args.checkpoint_dir/f'tiny_policy_{mode}.pt'
        model = TinyTemporalPolicy(image_channels=4 if mode=='rgb_hsv' else 3)
        model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),
                              strict=True)
        model.eval()
        dataset = TemporalExpertDataset(samples,frames,events,mode=mode)
        loader = DataLoader(dataset,batch_size=args.batch_size,
                            shuffle=False,num_workers=0)
        probabilities=[]
        with torch.inference_mode():
            for img,control,_ in loader:
                probabilities.extend(torch.sigmoid(model(img,control)).tolist())
        if len(probabilities)!=len(samples):
            raise ValueError('model output/sample count mismatch')
        prob_by_video = defaultdict(list)
        for sample,prob in zip(samples,probabilities):
            prob_by_video[sample.video].append((sample,prob))
        for video in split.test:
            ordered = sorted(prob_by_video[video],key=lambda x:x[0].target_timestamp_ms)
            points = points_from_predictions(
                [sample for sample,_ in ordered],
                [prob for _,prob in ordered],
            )
            gt = events[video]
            expected=[Transition(video,e.timestamp_ms,e.pressed) for e in gt[1:]]
            releases=[ReleaseSegment(video,a.timestamp_ms,b.timestamp_ms)
                      for a,b in zip(gt,gt[1:]) if not a.pressed and b.pressed]
            evaluation = evaluate_sequence(points,expected,releases,
                                           threshold=.5,tolerance_ms=100.0)
            compare_reproduced_metric(heldout,mode,video,evaluation)
            points_by_video[video][mode]=points
            evaluations[video][mode]=evaluation
        print(f'Validated {mode} inference against frozen Test report',flush=True)
        del model
    result={
        'kind':'frozen_tiny_policy_timeline_audit',
        'diagnostic_only_no_tuning':True,
        'source_heldout_report_sha256':sha256(reference_path),
        'model_sha256':{mode:heldout['results'][mode]['checkpoint_sha256']
                        for mode in MODES},
        'threshold':0.5,'tolerance_ms':100.0,'test_videos':list(split.test),
        'videos':{},
    }
    # Compute every diagnostic in memory before writing any artifact.
    artifacts=[]
    for video in split.test:
        ps=points_by_video[video]
        times,values=probabilities_at_points(ps)
        model_events={mode:event_diagnostics(evaluations[video][mode]) for mode in MODES}
        cases=build_focus_cases(model_events,maximum=16)
        full=render_timeline(times,values,events[video],model_events)
        zoom=render_focus_contact(times,values,events[video],cases)
        stem=Path(video).stem
        artifacts.append((video,stem,times,values,full,zoom,cases,model_events))
        result['videos'][video]={
            'samples':len(ps['rgb']),
            'events_by_mode':model_events,
            'focus_cases_displayed':cases,
            'focus_display_count':len(cases),
            'focus_selection':'missed short releases first, then extra switches, max 16',
        }
        print(f'{stem}: missing_short_rgb='
              f'{sum(not s["detected"] for s in model_events["rgb"]["short_releases"])} '
              f'missing_short_hsv='
              f'{sum(not s["detected"] for s in model_events["rgb_hsv"]["short_releases"])} '
              f'zoomed_cases={len(cases)}',flush=True)
    for video,stem,times,values,full,zoom,cases,_ in artifacts:
        trace=args.output_dir/f'tiny_trace_{stem}.csv'
        rows=sorted(by_video[video],key=lambda s:s.target_timestamp_ms)
        with trace.open('x',encoding='utf-8',newline='') as output:
            writer=csv.writer(output)
            writer.writerow(('target_ms','gt_future_pressed','actual_pressed_at_observation',
                             'rgb_probability','rgb_hsv_probability',
                             'rgb_predicted_pressed','rgb_hsv_predicted_pressed'))
            for i,sample in enumerate(rows):
                writer.writerow((f'{times[i]:.6f}',int(sample.target_pressed),
                                 int(sample.current_pressed),
                                 f'{values["rgb"][i]:.8f}',f'{values["rgb_hsv"][i]:.8f}',
                                 int(values['rgb'][i]>=.5),int(values['rgb_hsv'][i]>=.5)))
        if not cv2.imwrite(str(args.output_dir/f'tiny_timeline_{stem}.png'),full):
            raise RuntimeError(f'could not write {stem} timeline')
        if not cv2.imwrite(str(args.output_dir/f'tiny_focus_{stem}.jpg'),zoom):
            raise RuntimeError(f'could not write {stem} focus')
    out=args.output_dir/'tiny_policy_timeline_audit.json'
    with out.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False)
        stream.write('\n')
    print(f'Wrote read-only audit: {out}',flush=True)


if __name__=='__main__':
    main()
