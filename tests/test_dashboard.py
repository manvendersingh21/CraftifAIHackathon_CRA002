"""Fallback event schema -> dashboard parser -> unique violators / cooldown."""
import json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fallback")); sys.path.insert(0, str(ROOT / "dashboard"))
from events import EventSink
from parser import parse


def test_sink_output_parses_as_violation():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "ev.jsonl"
        s = EventSink(str(f), sensor="camX")
        s.emit({"track_id": 7, "label": "no_helmet", "confidence": 0.81, "bbox": [1, 2, 3, 4], "frame": 42})
        s.close()
        m = parse(f.read_text().strip())
    v = m["violations"]
    assert len(v) == 1 and v[0]["id"] == "7" and v[0]["source"] == "camX" and v[0]["frame"] == 42


def test_app_counts_unique_violators():
    import app as dash
    dash.COOLDOWN = 10
    for tid in (1, 1, 2):
        dash.ingest(json.dumps({"sensorId": "c", "object": {"label": "no_helmet", "tracker_id": tid, "confidence": .9}}), "t", "test")
    s = dash.snapshot_stats()
    assert s["unique_violators"] == 2 and s["incidents"] == 2   # repeat of track 1 inside cooldown is suppressed


def test_compliance_calc_and_zero_guard():
    import app as dash
    assert dash.compliance_pct(0, 0) is None          # no persons -> no divide by zero
    assert dash.compliance_pct(4, 1) == 75.0
    assert dash.compliance_pct(2, 5) == 0.0            # clamped
    assert dash.compliance_pct(3, 0) == 100.0


def test_reset_endpoint_clears_state():
    import app as dash
    dash.COOLDOWN = 10
    dash.ingest(json.dumps({"sensorId": "c", "object": {"label": "Person", "tracker_id": 9, "confidence": .9}}), "t", "test")
    dash.ingest(json.dumps({"sensorId": "c", "object": {"label": "no_helmet", "tracker_id": 9, "confidence": .9}}), "t", "test")
    s = dash.snapshot_stats()
    assert s["unique_persons"] >= 1 and s["incidents"] >= 1 and sum(s["timeline"]) >= 1
    r = dash.app.test_client().post("/api/reset")
    assert r.status_code == 200
    s = dash.snapshot_stats()
    assert s["incidents"] == s["unique_violators"] == s["unique_persons"] == s["messages"] == 0
    assert s["compliance"] is None and sum(s["timeline"]) == 0
    assert dash.app.test_client().get("/api/state").get_json()["incidents"] == []


def test_parser_keeps_bbox_and_frame_size():
    m = parse(json.dumps({"frame_size": [960, 540], "sensorId": "c", "frame_num": 3,
                          "object": {"label": "no_helmet", "tracker_id": 9, "confidence": .8, "bbox": [1, 2, 30, 40]}}))
    v = m["violations"][0]
    assert v["bbox"] == [1, 2, 30, 40] and v["size"] == [960, 540]
