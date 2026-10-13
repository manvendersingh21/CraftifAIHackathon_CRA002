"""Live-camera path end to end: JPEG frames in -> /api/analyze -> detections, tracked people, dashboard state."""
import sys
from pathlib import Path
import cv2, pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
MODEL, CLIP = ROOT / "models" / "ppe_yolov8n.onnx", ROOT / "samples" / "sample.mp4"
pytestmark = pytest.mark.skipif(not (MODEL.exists() and CLIP.exists()), reason="needs trained model + sample clip")


@pytest.fixture(scope="module")
def client():
    import app as dash, live
    dash.app._got_first_request = False             # other test modules may have served requests already
    live.init(dash, str(MODEL))
    dash.app.config["TESTING"] = True
    return dash, dash.app.test_client()


def frames(start, count, step=3):
    cap = cv2.VideoCapture(str(CLIP)); cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(count):
        ok, f = cap.read()
        if not ok:
            return
        for _ in range(step - 1):
            cap.read()
        yield cv2.imencode(".jpg", cv2.resize(f, (640, 360)))[1].tobytes()


def test_status_ready(client):
    _, c = client
    d = c.get("/api/live/status").get_json()
    assert d["ready"] and d["backend"] and d["model"] == "ppe_yolov8n.onnx"


def test_bad_input_rejected(client):
    _, c = client
    assert c.post("/api/analyze", data=b"not a jpeg").status_code == 400


def test_helmeted_scene_has_people_and_no_violation(client):
    dash, c = client
    dash.api_reset()
    last = None
    for jpg in frames(150, 24):                         # clip 2 (frames 150-224): three helmeted workers
        last = c.post("/api/analyze", data=jpg, headers={"X-Session": "t-good"}).get_json()
    assert last["ok"] and last["w"] == 640 and last["h"] == 360 and last["ms"] > 0
    assert last["people"], "expected tracked people"
    assert all(not p["violation"] for p in last["people"])
    assert all(set(d) == {"b", "n", "s"} and d["n"] != "Person" for d in last["dets"])


def test_no_helmet_scene_raises_incident_with_evidence(client):
    dash, c = client
    dash.api_reset()
    banners = []
    for jpg in frames(75, 25):                          # clip 1: person without helmet
        banners.append(c.post("/api/analyze", data=jpg, headers={"X-Session": "t-bad"}).get_json()["banner"])
    assert any(banners), "expected a NO HELMET banner"
    st = c.get("/api/state").get_json()
    assert st["stats"]["unique_violators"] >= 1 and st["incidents"]
    assert st["stats"]["video"]["live"] and st["stats"]["video"]["source"] == "camera (browser)"
    ev = st["incidents"][0]["evidence"]
    assert ev and c.get(ev).status_code == 200


def test_reset_clears_sessions(client):
    dash, c = client
    c.post("/api/analyze", data=next(frames(0, 1)), headers={"X-Session": "t-reset"})
    import live
    assert live.SESSIONS
    dash.api_reset()
    assert not live.SESSIONS
