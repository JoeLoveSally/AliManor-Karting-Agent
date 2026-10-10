import numpy as np
import pytest
from karting_agent.train.shadow_control import ShadowController


def test_pending_press_not_visible_before_due():
    ctrl = ShadowController(initial_pressed=False)
    ctrl.advance_to(200)
    q = ctrl.propose(observe_ms=200, target_ms=300, probability=.9)
    assert q.execute_ms == 300
    ctrl.advance_to(299)
    assert not ctrl.pressed
    assert not ctrl.control_features([200,250,299])[-1, 0]
    ctrl.advance_to(300)
    assert ctrl.pressed
    assert ctrl.events[-1].time_ms == 300


def test_feedback_age_matches_training_semantics():
    ctrl = ShadowController(initial_pressed=True)
    ctrl.advance_to(200)
    ctrl.propose(observe_ms=200, target_ms=300, probability=0.0)
    ctrl.advance_to(350)
    x = ctrl.control_features((150,200,250,300,350))
    assert x[:,0].tolist() == [1,1,1,0,0]
    assert x[-1,1] == pytest.approx(50/500)
    assert x[-1,2] == pytest.approx(50/200)
    assert x[3,1] == pytest.approx(0)


def test_late_inference_executes_no_earlier_than_ready():
    ctrl = ShadowController(initial_pressed=False, latency_ms=140)
    ctrl.advance_to(200)
    q = ctrl.propose(observe_ms=200,target_ms=300,probability=.99)
    assert q.ready_ms == 340 and q.execute_ms == 340
    ctrl.advance_to(325)
    assert ctrl.state_at(325) is False
    ctrl.advance_to(340)
    assert ctrl.pressed


def test_redundant_proposals_do_not_generate_fake_switches():
    ctrl = ShadowController(initial_pressed=True)
    for t in (200, 233.333, 266.667):
        ctrl.advance_to(t)
        ctrl.propose(observe_ms=t,target_ms=t+100,probability=.9)
    ctrl.flush()
    assert len(ctrl.events)==1
    assert ctrl.redundant==3


def test_rapid_switches_applied_at_real_due_times():
    ctrl = ShadowController(initial_pressed=False)
    for t,p in [(200,.9),(233.333,.1),(266.667,.9)]:
        ctrl.advance_to(t)
        ctrl.propose(observe_ms=t,target_ms=t+100,probability=p)
    ctrl.flush()
    assert [e.pressed for e in ctrl.events] == [False,True,False,True]
    assert [e.time_ms for e in ctrl.events[1:]] == pytest.approx([300,333.333,366.667])


def test_previous_inference_appears_at_correct_history_timestamp():
    ctrl=ShadowController(initial_pressed=False)
    ctrl.advance_to(200)
    ctrl.propose(observe_ms=200,target_ms=300,probability=.9)
    ctrl.advance_to(333.333)
    x=ctrl.control_features([133.333,183.333,233.333,283.333,333.333])
    assert x[:,0].tolist()==[0,0,0,0,1]
    assert x[-1,1]==pytest.approx(33.333/500)


def test_bad_clock_proposal_and_future_query_rejected():
    ctrl=ShadowController(initial_pressed=True)
    with pytest.raises(ValueError):
        ctrl.propose(observe_ms=100,target_ms=200,probability=.9)
    with pytest.raises(ValueError):
        ctrl.control_features([1])
    with pytest.raises(ValueError):
        ctrl.advance_to(-1)
    ctrl.advance_to(100)
    with pytest.raises(ValueError):
        ctrl.advance_to(90)
    with pytest.raises(ValueError):
        ctrl.propose(observe_ms=100,target_ms=300,probability=.9)
    with pytest.raises(ValueError):
        ctrl.propose(observe_ms=100,target_ms=200,probability=float('nan'))
    with pytest.raises(ValueError):
        ctrl.state_at(101)


@pytest.mark.parametrize('kwargs',[{'latency_ms':-10},{'horizon_ms':0},{'threshold':1.0},{'latency_ms':float('nan')}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ShadowController(initial_pressed=False,**kwargs)
