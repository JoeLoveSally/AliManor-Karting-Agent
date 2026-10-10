#!/usr/bin/env python3
"""Non-Armed fixed-video shadow replay of frozen tiny policies on Validation.

This is NOT closed-loop driving. Visual frames always come from the expert
video, even after the simulated action history disagrees with the expert.
No training, ADB, hardware interfaces, runtime integration or Test exposure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / 'src', ROOT / 'scripts'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from experiment_tiny_temporal_policy import (  # noqa: E402
    cache_required_frames, label_events, observed_state_features,
    validate_sample,
)
from inspect_geometry_pseudo_labels import load_config as load_road_config  # noqa: E402
from karting_agent.model.tiny_temporal_policy import TinyTemporalPolicy  # noqa: E402
from karting_agent.train.heldout_guard import require_frozen_checkpoint  # noqa: E402
from karting_agent.train.sequence_evaluator import (  # noqa: E402
    SequencePoint, Transition, ReleaseSegment, evaluate_sequence, combine_evaluations,
)
from karting_agent.train.shadow_replay import replay_fixed_video  # noqa: E402
from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402
from karting_agent.train.trainer import load_video_split, partition_samples  # noqa: E402
from karting_agent.train.visual_ablation import make_mode_images  # noqa: E402
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402

MODES = ('rgb', 'rgb_hsv')
RESULT = 'tiny_policy_shadow_validation.json'


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def planned_output_paths(output_dir, split) -> list[Path]:
    return ([output_dir / RESULT] + [
        output_dir / f'tiny_shadow_{mode}_{Path(video).stem}.csv'
        for mode in MODES for video in split.validation
    ])


def sequence_reports(rows, events, video):
    ground_truth = [Transition(video, ev.timestamp_ms, ev.pressed)
                    for ev in events[1:]]
    releases = [ReleaseSegment(video, a.timestamp_ms, b.timestamp_ms)
                for a,b in zip(events,events[1:]) if not a.pressed and b.pressed]
    def score(field):
        points=[SequencePoint(video,row['target_ms'],float(row[field])) for row in rows]
        return evaluate_sequence(points,ground_truth,releases,
                                 threshold=.5,tolerance_ms=100.)
    return {
        'teacher_forced':score('teacher_probability'),
        'self_fed_desired':score('shadow_probability'),
        'simulated_executed':score('simulated_executed_at_target'),
    }


def compare_frozen_validation(reference, combined_teacher, *, mode):
    """Fail if paired inference diverged from original frozen Validation."""
    expected=reference['validation_sequence']['candidate']
    got=combined_teacher.summary()
    for direction in ('all','press','release'):
        for key in ('ground_truth','predicted','matched','false_positive','false_negative'):
            if got['transition'][direction][key]!=expected['transition'][direction][key]:
                raise ValueError(f'{mode}: teacher replay diverged from frozen validation {direction}.{key}')
    for group in ('short_100_300ms','100_200ms','200_300ms'):
        for key in ('segments','detected'):
            if (got['release_segment_recall'][group][key] !=
                    expected['release_segment_recall'][group][key]):
                raise ValueError(f'{mode}: teacher replay short-release parity failed')


def write_csv(file, rows):
    if not rows:
        raise ValueError('empty replay output')
    columns=(
        'observation_ms','target_ms','expert_current_pressed',
        'simulated_current_pressed','expert_future_pressed',
        'teacher_probability','shadow_probability','shadow_desired_pressed',
        'inference_ready_ms','simulated_execution_ms',
        'simulated_executed_at_target','executed_matches_expert_at_target',
        'shadow_current_action_age_ms','per_sample_forward_ms',
    )
    with file.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:row[key] for key in columns})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/train_v4c2_temporal_v2.yaml')
    parser.add_argument('--samples',type=Path,default=ROOT/'data/processed/v3/samples.jsonl')
    parser.add_argument('--labels-dir',type=Path,default=ROOT/'data/processed/v3/labels')
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--simulated-latency-ms',type=float,default=0.0)
    parser.add_argument('--cpu-threads',type=int,default=4)
    args=parser.parse_args()
    if args.cpu_threads < 1 or args.simulated_latency_ms < 0:
        parser.error('CPU threads must be positive; latency must be nonnegative')
    if not args.output_dir.is_dir() or not args.checkpoint_dir.is_dir():
        parser.error('checkpoint-dir and output-dir must already exist')

    # Do not accept Test, arbitrary split overrides or mismatched train reports.
    split=load_video_split(args.config)
    if (not split.validation or
        set(split.test)&(set(split.validation)|set(split.train)) or
        set(split.validation)&set(split.train)):
        raise ValueError('not a disjoint Validation-only experiment')
    config=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if (int(config['model']['frame_stack']) != 5 or
        tuple(config['dataset']['prediction_horizons_ms'])!=(100,200,300) or
        not config['preprocess']['mask_touch_area']):
        raise ValueError('unexpected frozen V4-C2 data and masking protocol')
    artifacts={}
    for mode in MODES:
        checkpoint=args.checkpoint_dir/f'tiny_policy_{mode}.pt'
        meta_path=args.checkpoint_dir/f'tiny_policy_{mode}.json'
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        require_frozen_checkpoint(meta,mode=mode,split=split)
        artifacts[mode]={
            'weight_path':checkpoint,
            'checkpoint_sha256':sha256(checkpoint),
            'metadata_sha256':sha256(meta_path),
            'training_metadata':meta,
        }
    paths=planned_output_paths(args.output_dir,split)
    if any(p.exists() for p in paths):
        parser.error('refusing to overwrite previous shadow reports/CSVs')
    dataset=partition_samples(load_v3_samples(args.samples),split)['validation']
    if {s.video for s in dataset}!=set(split.validation):
        raise ValueError('missing validation source videos')
    for mode in MODES:
        expected=artifacts[mode]['training_metadata']['validation_sequence']['candidate']['samples']
        if len(dataset)!=expected:
            raise ValueError('validation sample count differs from frozen training')
    native={video:label_events(args.labels_dir,video) for video in split.validation}
    for sample in dataset:
        validate_sample(sample,native[sample.video])
    print(f'Validation-only fixed-video replay: {len(dataset)} samples, '
          f'{len(split.validation)} source videos; no Test/Armed',flush=True)
    prep=preprocess_config_from_mapping(config)
    road_cfg,_,_=load_road_config(ROOT/'configs/geometry_pseudo_labels.yaml')
    torch.set_num_threads(args.cpu_threads)
    frames=cache_required_frames(dataset,preprocess=prep,road_config=road_cfg)
    samples_by_video={video:sorted(
        (s for s in dataset if s.video==video),
        key=lambda s:s.target_timestamp_ms,
    ) for video in split.validation}
    final={
        'kind':'tiny_policy_validation_fixed_video_shadow',
        'simulated_only_no_game_control':True,
        'not_closed_loop':True,
        'expert_video_pixels_do_not_respond_to_simulated_actions':True,
        'model_weight_selection_or_threshold_changed':False,
        'split':split.name,
        'train_videos_decoded':[],
        'test_videos_decoded':[],
        'validation_videos':list(split.validation),
        'sample_count':len(dataset),
        'forecast_horizon_ms':100.,
        'threshold':.5,
        'event_matching_tolerance_ms':100.,
        'simulated_compute_latency_ms':args.simulated_latency_ms,
        'execution_timing':'max(observation+100ms, observation+simulated_latency)',
        'initial_control':'expert state at t=0 only; all future actions simulated',
        'models':{},
    }
    output_rows={}
    for mode in MODES:
        item=artifacts[mode]
        model=TinyTemporalPolicy(image_channels=4 if mode=='rgb_hsv' else 3)
        state=torch.load(item['weight_path'],map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
        model.eval()
        video_data={}
        aggregates={'teacher_forced':[],'self_fed_desired':[],'simulated_executed':[]}
        for video in split.validation:
            rows=samples_by_video[video]
            def get_images(sample):
                return make_mode_images(sample.input_frame_indices,
                                        frames[sample.video],mode=mode)
            replay=replay_fixed_video(
                model,rows,frames_for_sample=get_images,
                expert_events=native[video],expert_features=observed_state_features,
                latency_ms=args.simulated_latency_ms,
            )
            evaluations=sequence_reports(replay['rows'],native[video],video)
            for label,score in evaluations.items():
                aggregates[label].append(score)
            output_rows[(mode,video)]=replay['rows']
            video_data[video]={
                'replay':{key:value for key,value in replay.items() if key!='rows'},
                'events':{label:score.summary() for label,score in evaluations.items()},
            }
        combined={label:combine_evaluations(values) for label,values in aggregates.items()}
        compare_frozen_validation(item['training_metadata'],combined['teacher_forced'],mode=mode)
        final['models'][mode]={
            'checkpoint_sha256':item['checkpoint_sha256'],
            'training_metadata_sha256':item['metadata_sha256'],
            'teacher_validation_parity_verified':True,
            'aggregate_events':{label:ev.summary() for label,ev in combined.items()},
            'per_video':video_data,
        }
        summary=final['models'][mode]['aggregate_events']
        print(f"{mode}: teacher F1={summary['teacher_forced']['transition']['all']['f1']:.4f}, "
              f"self-fed(desired) F1={summary['self_fed_desired']['transition']['all']['f1']:.4f}, "
              f"simulated-executed F1={summary['simulated_executed']['transition']['all']['f1']:.4f}",
              flush=True)
        del model
    # All parity checks and all inference finish BEFORE writing any file.
    for mode in MODES:
        for video in split.validation:
            stem=Path(video).stem
            write_csv(args.output_dir/f'tiny_shadow_{mode}_{stem}.csv',
                      output_rows[(mode,video)])
    output=args.output_dir/RESULT
    with output.open('x',encoding='utf-8') as fp:
        json.dump(final,fp,ensure_ascii=False,indent=2)
        fp.write('\n')
    print(f'Saved fixed-video validation-only shadow replay: {output}',flush=True)


if __name__=='__main__':
    main()
