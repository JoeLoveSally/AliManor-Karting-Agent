from dataclasses import dataclass
import numpy as np
import pytest

from karting_agent.train.tiny_policy_timeline import (
    build_focus_cases, draw_probability_chart, event_diagnostics,
    probabilities_at_points, render_timeline, render_focus_contact,
)

@dataclass(frozen=True)
class Event:
    video: str
    timestamp_ms: float
    pressed: bool

@dataclass(frozen=True)
class Match:
    expected: Event
    predicted: Event
    @property
    def error_ms(self):
        return self.predicted.timestamp_ms-self.expected.timestamp_ms

@dataclass(frozen=True)
class Segment:
    video: str
    start_ms: float
    end_ms: float
    @property
    def duration_ms(self):
        return self.end_ms-self.start_ms
    @property
    def start_transition(self):
        return Event(self.video,self.start_ms,False)
    @property
    def end_transition(self):
        return Event(self.video,self.end_ms,True)

@dataclass(frozen=True)
class Point:
    video: str
    timestamp_ms: float
    probability: float

class Eval:
    pass


def evaluation():
    obj=Eval()
    obj.expected_transitions=(Event('A',110,False),Event('A',250,True),Event('A',800,False))
    obj.predicted_transitions=(Event('A',130,False),Event('A',500,True),Event('A',820,False))
    obj.matches=(Match(obj.expected_transitions[0],obj.predicted_transitions[0]),
                 Match(obj.expected_transitions[2],obj.predicted_transitions[2]))
    obj.release_segments=(Segment('A',110,250),)
    obj.release_detected=(False,)
    return obj


def points():
    return {'rgb':[Point('A',float(i*50),.6 if i%3==0 else .2) for i in range(25)],
            'rgb_hsv':[Point('A',float(i*50),.5 if i%5==0 else .1) for i in range(25)]}


def test_exact_matching_for_missed_short_release():
    x=event_diagnostics(evaluation())
    assert len(x['expected'])==3
    assert [v['status'] for v in x['expected']]==['matched','missed','matched']
    assert x['expected'][0]['error_ms']==20
    assert x['predicted'][1]['status']=='false_positive'
    assert x['short_releases']==[{'start_ms':110.,'end_ms':250.,'duration_ms':140.,
                                  'detected':False,'release_onset_matched':True,
                                  'press_onset_matched':False}]


def test_focus_prioritizes_shared_misses_and_then_fp():
    a=event_diagnostics(evaluation())
    b=event_diagnostics(evaluation())
    cases=build_focus_cases({'rgb':a,'rgb_hsv':b},maximum=2)
    assert cases[0]['kind']=='short_release'
    assert cases[0]['missing_models']==['rgb','rgb_hsv']
    assert cases[1]['kind']=='extra_switch'


def test_zero_length_focus_rejected():
    with pytest.raises(ValueError,match='maximum'):
        build_focus_cases({'rgb':{},'rgb_hsv':{}},maximum=0)


def test_probabilities_require_same_timestamp_grid():
    t,values=probabilities_at_points(points())
    assert t.shape==(25,)
    assert values['rgb'][0]==.6
    bad=points();bad['rgb_hsv'][0]=Point('A',3.,.5)
    with pytest.raises(ValueError,match='mismatched'):
        probabilities_at_points(bad)


def test_probability_grid_rejects_mixed_video_and_nan():
    bad=points();bad['rgb_hsv'][0]=Point('B',0.,.5)
    with pytest.raises(ValueError,match='mixed video'):
        probabilities_at_points(bad)
    bad=points();bad['rgb_hsv'][0]=Point('A',0.,float('nan'))
    with pytest.raises(ValueError,match='probability'):
        probabilities_at_points(bad)


def test_render_png_images_and_zoom_panels():
    x=event_diagnostics(evaluation()); t,values=probabilities_at_points(points())
    full=render_timeline(t,values,evaluation().expected_transitions,
                         {'rgb':x,'rgb_hsv':x},width=1200)
    assert full.shape==(430,1200,3)
    assert full.dtype==np.uint8
    chart=draw_probability_chart(t,values,evaluation().expected_transitions,
                                 start_ms=0,end_ms=1200,width=650)
    assert chart.shape==(290,650,3)
    focus=render_focus_contact(t,values,evaluation().expected_transitions,
            build_focus_cases({'rgb':x,'rgb_hsv':x},maximum=2))
    assert focus.shape==(550,1200,3)


def test_detection_inconsistency_rejected():
    ev=evaluation();ev.release_detected=(True,)
    with pytest.raises(ValueError,match='inconsistent'):
        event_diagnostics(ev)


def test_render_no_failures_produces_empty_placeholder():
    t,values=probabilities_at_points(points())
    image=render_focus_contact(t,values,evaluation().expected_transitions,[])
    assert image.shape==(220,1200,3)
