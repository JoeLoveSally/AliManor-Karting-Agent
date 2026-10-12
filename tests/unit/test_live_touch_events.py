"""Pure parser/device-discovery tests; no Android connection."""

import pytest

from karting_agent.train.labels.live_touch_events import (
    TouchTransitionParser, choose_touch_device, parse_kernel_event, decoded_frame_rows,
)


def event(timestamp, kind, code, value):
    result = parse_kernel_event(
        f"[ {timestamp:.6f}] /dev/input/event4: {kind} {code} {value}"
    )
    assert result is not None
    return result


def test_touch_device_discovery_requires_unambiguous_multitouch():
    listing = """add device 1: /dev/input/event4
  name: "touchscreen"
    ABS_MT_POSITION_X
    ABS_MT_POSITION_Y
    ABS_MT_TRACKING_ID
add device 2: /dev/input/event2
  name: "keyboard"
    EV_KEY
"""
    assert choose_touch_device(listing) == "/dev/input/event4"
    with pytest.raises(ValueError, match="expected one"):
        choose_touch_device(listing + """add device 3: /dev/input/event6
    ABS_MT_POSITION_X ABS_MT_POSITION_Y ABS_MT_TRACKING_ID
""")
    with pytest.raises(ValueError, match="expected one"):
        choose_touch_device("no touch devices")


def test_sustained_press_no_duplicate_and_real_100ms_release():
    parser = TouchTransitionParser()
    result = []
    inputs = [
        event(10.0, "EV_ABS", "ABS_MT_TRACKING_ID", "00000015"),
        event(10.0, "EV_SYN", "SYN_REPORT", "00000000"),
        event(10.5, "EV_ABS", "ABS_MT_POSITION_X", "000000c8"),
        event(10.5, "EV_SYN", "SYN_REPORT", "00000000"),
        event(12.0, "EV_ABS", "ABS_MT_TRACKING_ID", "ffffffff"),
        event(12.0, "EV_SYN", "SYN_REPORT", "00000000"),
        event(12.1, "EV_ABS", "ABS_MT_TRACKING_ID", "00000016"),
        event(12.1, "EV_SYN", "SYN_REPORT", "00000000"),
    ]
    for item in inputs:
        value = parser.ingest(item)
        if value is not None:
            result.append((item.device_timestamp_ms, value))
    assert result == [(10000.0, True), (12000.0, False), (12100.0, True)]


def test_multitouch_does_not_release_while_other_slot_active():
    parser = TouchTransitionParser()
    events = [
        ("EV_ABS", "ABS_MT_TRACKING_ID", "00000001"),
        ("EV_SYN", "SYN_REPORT", "00000000"),
        ("EV_ABS", "ABS_MT_SLOT", "00000001"),
        ("EV_ABS", "ABS_MT_TRACKING_ID", "00000002"),
        ("EV_SYN", "SYN_REPORT", "00000000"),
        ("EV_ABS", "ABS_MT_SLOT", "00000000"),
        ("EV_ABS", "ABS_MT_TRACKING_ID", "ffffffff"),
        ("EV_SYN", "SYN_REPORT", "00000000"),
        ("EV_ABS", "ABS_MT_SLOT", "00000001"),
        ("EV_ABS", "ABS_MT_TRACKING_ID", "ffffffff"),
        ("EV_SYN", "SYN_REPORT", "00000000"),
    ]
    changes = [parser.ingest(event(5+i*.01, *e)) for i,e in enumerate(events)]
    assert [c for c in changes if c is not None] == [True, False]


def test_btn_touch_fallback_and_invalid_lines():
    parser = TouchTransitionParser()
    assert parse_kernel_event("not an event") is None
    assert parser.ingest(event(1, "EV_KEY", "BTN_TOUCH", "DOWN")) is None
    assert parser.ingest(event(1, "EV_SYN", "SYN_REPORT", "00000000")) is True
    assert parser.ingest(event(2, "EV_KEY", "BTN_TOUCH", "UP")) is None
    assert parser.ingest(event(2, "EV_SYN", "SYN_REPORT", "00000000")) is False


def test_frame_index_mapping_uses_real_host_timestamps():
    rows = list(decoded_frame_rows([1800.0, 1816.1, 1836.7], 100000.0))
    assert [r["frame_index"] for r in rows] == [0, 1, 2]
    assert [r["host_decoded_perf_counter_ms"] for r in rows] == [101800.0, 101816.1, 101836.7]
    with pytest.raises(ValueError, match="strictly increase"):
        list(decoded_frame_rows([0.0, 0.0], 1))


def test_getevent_single_device_output_requires_explicit_device_context():
    # Actual vivo_ts /dev/input/event6 output, without a device prefix.
    line = "[  268465.539282] EV_ABS       ABS_MT_TRACKING_ID   0000a559"
    assert parse_kernel_event(line) is None
    result = parse_kernel_event(line, default_device="/dev/input/event6")
    assert result is not None
    assert result.device == "/dev/input/event6"
    assert result.code == "ABS_MT_TRACKING_ID"
    assert result.device_timestamp_ms == pytest.approx(268465539.282)

    prefixed = "[  268465.539282] /dev/input/event9: EV_ABS ABS_MT_TRACKING_ID 0000a559"
    result = parse_kernel_event(prefixed, default_device="/dev/input/event6")
    assert result is not None and result.device == "/dev/input/event9"


def test_real_vivo_short_and_long_touch_trace_not_split():
    # Abridged genuine vivo_ts -lt output: four 58-67ms taps plus 1.38s hold.
    base = 268465.0
    spans = [
        (268465.539282, 268465.596803),
        (268465.655552, 268465.722747),
        (268466.343360, 268466.401708),
        (268466.468821, 268466.527600),
        (268467.232311, 268468.608807),
    ]
    parser = TouchTransitionParser()
    changes = []
    for start, end in spans:
        for stamp, value in ((start, "0000a559"), (end, "ffffffff")):
            for code, field in (
                ("EV_ABS", f"ABS_MT_TRACKING_ID   {value}"),
                ("EV_SYN", "SYN_REPORT           00000000"),
            ):
                line = f"[  {stamp:.6f}] {code} {field}"
                parsed = parse_kernel_event(
                    line, default_device="/dev/input/event6"
                )
                assert parsed is not None
                state = parser.ingest(parsed)
                if state is not None:
                    changes.append((parsed.device_timestamp_ms, state))
    assert len(changes) == 10
    assert [state for _, state in changes] == [True, False] * 5
    actual_ms = [
        changes[i+1][0] - changes[i][0] for i in range(0, 10, 2)
    ]
    assert actual_ms == pytest.approx(
        [57.521, 67.195, 58.348, 58.779, 1376.496], abs=0.002,
    )
