from types import SimpleNamespace
import numpy as np
import pytest
import torch
from karting_agent.train.shadow_replay import replay_fixed_video


def samples(num=5):
    return [SimpleNamespace(video='validation.mp4',
        input_timestamps_ms=(t-200,t-150,t-100,t-50,t),
        target_timestamp_ms=t+100,
        current_pressed=False,target_pressed=True,
    ) for t in (200+50*i for i in range(num))]


def expert_features(stamps,events):
    return np.zeros((5,3),dtype=np.float32)


class ObserveControlModel(torch.nn.Module):
    def forward(self,image,controls):
        # Sign changes if simulated state eventually becomes PRESSED.
        # Expert remains RELEASED in synthetic features.
        return (1. - 2.*controls[:,-1,0])*8.


def frames(row):
    return np.ones((5,3,16,16),np.float32)


def test_shadow_feedback_diverges_after_first_scheduled_decision():
    result=replay_fixed_video(ObserveControlModel(),samples(),
        frames_for_sample=frames,
        expert_events=(SimpleNamespace(timestamp_ms=0.,pressed=False),),
        expert_features=expert_features)
    rows=result['rows']
    # t=200 predicts PRESS for t=300, t=250 still cannot see pending press
    assert rows[0]['shadow_desired_pressed']
    assert not rows[1]['simulated_current_pressed']
    assert rows[2]['simulated_current_pressed']
    assert rows[2]['shadow_probability'] < .5
    assert all(row['teacher_probability']>.5 for row in rows)
    assert rows[0]['simulated_execution_ms']==300
    assert result['first_simulated_divergence_observation_ms']==300
    assert result['simulated_events'][1]['time_ms']==300


def test_late_scheduling_keeps_old_controls_until_ready():
    out=replay_fixed_video(ObserveControlModel(),samples(),
        frames_for_sample=frames,
        expert_events=(SimpleNamespace(timestamp_ms=0.,pressed=False),),
        expert_features=expert_features,latency_ms=130)
    assert out['late_deadlines']==5
    assert out['rows'][0]['simulated_execution_ms']==330
    assert not out['rows'][2]['simulated_current_pressed']
    assert out['rows'][3]['simulated_current_pressed']
    assert out['rows'][0]['simulated_executed_at_target'] is False


def test_no_future_expert_label_or_state_flows_into_visual_or_controls():
    seen=[]
    class Spy(torch.nn.Module):
        def forward(self,img,ctl):
            seen.append((img.detach().clone(),ctl.detach().clone()))
            return torch.tensor([0.,0.])
    data=samples(3)
    def must_not_see_future(row):
        assert row.target_timestamp_ms-row.input_timestamps_ms[-1]==100
        return frames(row)
    result=replay_fixed_video(Spy(),data,
        frames_for_sample=must_not_see_future,
        expert_events=(SimpleNamespace(timestamp_ms=0.,pressed=False),),
        expert_features=expert_features)
    assert len(seen)==3
    assert seen[0][1].shape==(2,5,3)
    assert torch.equal(seen[0][0][0],seen[0][0][1])
    assert result['simulated_events'][0]['time_ms']==0


def test_samples_from_multiple_videos_rejected():
    data=samples(2);data[1].video='test.mp4'
    with pytest.raises(ValueError,match='one video'):
        replay_fixed_video(ObserveControlModel(),data,frames_for_sample=frames,
            expert_events=(SimpleNamespace(timestamp_ms=0.,pressed=False),),
            expert_features=expert_features)


def test_wrong_horizon_rejected():
    data=samples(1);data[0].target_timestamp_ms=210
    with pytest.raises(ValueError,match='horizon'):
        replay_fixed_video(ObserveControlModel(),data,frames_for_sample=frames,
            expert_events=(SimpleNamespace(timestamp_ms=0.,pressed=False),),
            expert_features=expert_features)
