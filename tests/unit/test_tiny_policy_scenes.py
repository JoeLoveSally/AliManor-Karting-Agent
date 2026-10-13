"""CPU-only synthetic-video tests of frozen Test error scene extraction."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from audit_tiny_policy_scenes import (
    candidate_cases,get_frames_for_cases,make_contact_sheet,validate_trace,
)


def events():
    return {
        "rgb":{
            "short_releases":[{"start_ms":1000.,"end_ms":1170.,"detected":False}],
            "predicted":[{"status":"false_positive","time_ms":1510.,"pressed":False},
                         {"status":"false_positive","time_ms":1560.,"pressed":True}],
            "expected":[{"time_ms":1000.,"pressed":False},
                        {"time_ms":1170.,"pressed":True}],
        },
        "rgb_hsv":{
            "short_releases":[{"start_ms":1000.,"end_ms":1170.,"detected":False}],
            "predicted":[{"status":"false_positive","time_ms":1520.,"pressed":False}],
            "expected":[{"time_ms":1000.,"pressed":False},
                        {"time_ms":1170.,"pressed":True}],
        },
    }


def test_short_release_priority_and_duplicate_switch_cluster():
    cases=candidate_cases(events())
    assert len(cases)==2
    assert cases[0]["kind"]=="missed_short_release"
    assert cases[0]["models"]==["rgb","rgb_hsv"]
    assert cases[1]["kind"]=="extra_switch"


def test_model_protocol_rejects_invalid_selection():
    with pytest.raises(ValueError):
        candidate_cases(events(),max_cases=0)
    with pytest.raises(ValueError):
        candidate_cases({"rgb":events()["rgb"]})


def make_video(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),30,(90,160))
    assert writer.isOpened()
    for i in range(90):
        frame=np.full((160,90,3),i%250,np.uint8)
        cv2.putText(frame,str(i),(5,80),cv2.FONT_HERSHEY_SIMPLEX,.65,(250,0,0),2)
        writer.write(frame)
    writer.release()


def test_synthetic_video_scene_samples_and_sheet(tmp_path):
    video=tmp_path/"source.mp4"
    make_video(video)
    cases=candidate_cases(events())
    fps,count,requests,decoded=get_frames_for_cases(video,cases)
    assert fps==pytest.approx(30,abs=.5)
    assert count==90
    assert requests[(0,1)][0]==round(cases[0]["time_ms"]/1000*fps)
    result=make_contact_sheet(cases,requests,decoded,view_width=120)
    assert result.dtype==np.uint8
    assert result.shape[1]==330+3*130+10
    assert result.shape[0]==2*(round(120*160/90)+82)
    assert cv2.imwrite(str(tmp_path/"scene.jpg"),result)


def test_missing_source_video_failfast(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_frames_for_cases(tmp_path/"missing.mp4",candidate_cases(events()))


def write_trace(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("target_ms","gt_future_pressed",
                         "rgb_probability","rgb_hsv_probability"))
        writer.writerows(rows)


def test_trace_count_and_timestamps_are_checked(tmp_path):
    trace=tmp_path/"trace.csv"
    write_trace(trace,[(200,0,.1,.2),(233.333,1,.9,.7)])
    validate_trace(trace,2)
    with pytest.raises(ValueError,match="trace CSV"):
        validate_trace(trace,1)
    write_trace(trace,[(200,0,.1,.2),(200,1,.9,.7)])
    with pytest.raises(ValueError,match="trace CSV"):
        validate_trace(trace,2)


def test_cli_rejects_existing_output_before_video_read(tmp_path):
    video_root=tmp_path/"root"
    (video_root/"data/raw").mkdir(parents=True)
    make_video(video_root/"data/raw/example.mp4")
    audit={
        "kind":"frozen_tiny_policy_timeline_audit",
        "diagnostic_only_no_tuning":True,
        "threshold":.5,"tolerance_ms":100.,
        "test_videos":["data/raw/example.mp4"],
        "videos":{"data/raw/example.mp4":{
            "samples":2,"events_by_mode":events()}},
    }
    path=tmp_path/"audit.json"
    path.write_text(json.dumps(audit),encoding="utf-8")
    write_trace(tmp_path/"tiny_trace_example.csv",
                [(200,0,.1,.2),(233.333,1,.9,.7)])
    cmd=[
        sys.executable,str(ROOT/"scripts/audit_tiny_policy_scenes.py"),
        "--audit",str(path),
        "--trace-dir",str(tmp_path),
        "--video-root",str(video_root),
        "--output-dir",str(tmp_path),
    ]
    first=subprocess.run(cmd,capture_output=True,text=True,check=False)
    assert first.returncode==0,first.stderr
    assert (tmp_path/"tiny_scenes_example.jpg").is_file()
    repeated=subprocess.run(cmd,capture_output=True,text=True,check=False)
    assert repeated.returncode!=0
    assert "refusing to overwrite" in repeated.stderr
