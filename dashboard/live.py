"""Live camera analysis service: browsers (or any client) POST JPEG frames, get detections + tracked people back.

  POST /api/analyze   raw JPEG, header X-Session  -> detections + tracked people (+ updates dashboard state/events)
  POST /api/detect    raw JPEG                    -> stateless detections only (lets a laptop borrow a GPU box)
  GET  /api/live/status

Detection runs locally (ONNX Runtime, CUDA if present) or remotely (--detector-url http://gpu-box:5000, which calls
that box's /api/detect); tracking, the PPE rules, events and evidence always run here, per browser session.
"""
import json, sys, threading, time
from pathlib import Path

import cv2, numpy as np
from flask import Blueprint, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fallback"))
from detector import Detector                       # noqa: E402
from events import EventSink                        # noqa: E402
from rules import NO_HELMET, PPEMonitor             # noqa: E402

bp = Blueprint("live", __name__)
CFG = {"detector": None, "hooks": None, "remote": None, "monitor_kw": {}, "error": None}
SESSIONS, SLOCK, DLOCK = {}, threading.Lock(), threading.Lock()
SESSION_TTL = 120.0


class RemoteDetector:
    """Detection on another machine's /api/detect; same interface as Detector."""

    def __init__(self, url, timeout=3.0):
        import http.client, urllib.parse
        u = urllib.parse.urlparse(url)
        self._mk = lambda: http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
        self._conn = self._mk()
        self.provider, self.model_name, self.url = "remote", "remote", url
        st = self._get("/api/live/status")
        self.provider = f"remote:{st.get('backend', '?')}"
        self.model_name = st.get("model", "?")

    def _get(self, path):
        c = self._mk(); c.request("GET", path); r = c.getresponse(); d = json.loads(r.read()); c.close(); return d

    def detect(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        try:
            self._conn.request("POST", "/api/detect", buf.tobytes(), {"Content-Type": "image/jpeg"})
            d = json.loads(self._conn.getresponse().read())
        except Exception:
            self._conn.close(); self._conn = self._mk(); raise
        return [(*b, n, s) for b, n, s in ((x["b"], x["n"], x["s"]) for x in d["dets"])]


def init(app_module, model, labels=None, conf=0.35, detector_url=None, monitor_kw=None):
    """Called by dashboard/app.py once: hooks = ingest/evidence/video/broadcast functions of the dashboard."""
    CFG["hooks"], CFG["monitor_kw"] = app_module, monitor_kw or {}
    try:
        CFG["detector"] = RemoteDetector(detector_url) if detector_url else Detector(model, labels, conf)
        CFG["remote"] = bool(detector_url)
        print(f"[live] detector ready: {CFG['detector'].provider} {CFG['detector'].model_name}")
    except Exception as e:                             # missing model etc.: dashboard still works, live mode 503s
        CFG["error"] = str(e)
        print(f"[live] detector unavailable: {e}")
    if bp.name not in app_module.app.blueprints:
        app_module.app.register_blueprint(bp)


def reset_sessions():
    with SLOCK:
        SESSIONS.clear()


def _session(sid, w, h):
    now = time.time()
    with SLOCK:
        for k in [k for k, v in SESSIONS.items() if now - v["seen"] > SESSION_TTL]:
            del SESSIONS[k]
        s = SESSIONS.get(sid)
        if s is None or s["size"] != (w, h):
            hooks = CFG["hooks"]
            s = {"mon": PPEMonitor(**CFG["monitor_kw"]), "n": 0, "size": (w, h), "fps": 0.0, "last": now,
                 "sink": EventSink(None, None, "live", f"cam-{sid[:6]}", (w, h),
                                   on_message=lambda line: hooks.ingest(line, "live", "browser"))}
            SESSIONS[sid] = s
        s["seen"] = now
        return s


def _decode():
    arr = np.frombuffer(request.get_data(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None


@bp.route("/api/live/status")
def status():
    d = CFG["detector"]
    return jsonify({"ready": d is not None, "backend": d.provider if d else None,
                    "model": d.model_name if d else None, "remote": bool(CFG["remote"]), "error": CFG["error"]})


@bp.route("/api/detect", methods=["POST"])
def detect():
    d, frame = CFG["detector"], _decode()
    if d is None:
        return jsonify({"ok": False, "error": CFG["error"] or "no detector"}), 503
    if frame is None:
        return jsonify({"ok": False, "error": "bad JPEG"}), 400
    t = time.perf_counter()
    with DLOCK:
        dets = d.detect(frame)
    return jsonify({"ok": True, "w": frame.shape[1], "h": frame.shape[0], "ms": round((time.perf_counter() - t) * 1000, 1),
                    "dets": [{"b": [int(a) for a in x[:4]], "n": x[4], "s": round(float(x[5]), 3)} for x in dets]})


@bp.route("/api/analyze", methods=["POST"])
def analyze():
    d, hooks = CFG["detector"], CFG["hooks"]
    if d is None:
        return jsonify({"ok": False, "error": CFG["error"] or "no detector"}), 503
    frame = _decode()
    if frame is None:
        return jsonify({"ok": False, "error": "bad JPEG"}), 400
    H, W = frame.shape[:2]
    s = _session(request.headers.get("X-Session", "anon"), W, H)
    t0 = time.perf_counter()
    try:
        with DLOCK:
            named = d.detect(frame)
    except Exception as e:                             # remote GPU hiccup: tell the client, keep the session
        return jsonify({"ok": False, "error": f"detector: {e}"}), 502
    ms = (time.perf_counter() - t0) * 1000
    mon, n, sink = s["mon"], s["n"], s["sink"]
    s["n"] += 1
    events = mon.update(named, n)
    for tid, conf in mon.new_persons:
        bx = mon.box_of(tid)
        sink.emit({"track_id": tid, "label": "Person", "confidence": round(conf, 3),
                   "bbox": [int(v) for v in bx] if bx else [], "frame": n})
    for ev in events:
        hooks.put_evidence(frame, ev["bbox"], n, ev["track_id"])   # before emit so the card's thumbnail exists
        sink.emit(ev)
    people = []
    for box, tid, labs in mon.annotations(n):
        people.append({"id": str(tid), "b": [int(v) for v in box], "violation": bool(labs), "labels": sorted(labs)})
    hooks.broadcast({"type": "track", "data": [{"id": p["id"], "label": "Person", "bbox": p["b"], "size": [W, H],
                                                 "violation": p["violation"]} for p in people]})
    now = time.time()
    s["fps"] = 0.9 * s["fps"] + 0.1 / max(now - s["last"], 1e-3) if s["n"] > 1 else 0.0
    s["last"] = now
    src = request.headers.get("X-Source", "")[:120]     # 'video:<file>' when the browser plays real recorded footage
    hooks.video_tick({"backend": f"ONNX Runtime {d.provider}" if not CFG["remote"] else f"ONNX Runtime ({d.provider})",
                      "source": src[6:] if src.startswith("video:") else "camera (browser)", "infer_fps": round(s["fps"], 1), "width": W, "height": H,
                      "model": d.model_name})
    bad = sum(p["violation"] for p in people)
    return jsonify({"ok": True, "w": W, "h": H, "ms": round(ms, 1),
                    "dets": [{"b": [int(a) for a in x[:4]], "n": "no_ppe" if x[4] == "none" else x[4],
                              "s": round(float(x[5]), 3)} for x in named if x[4] != "Person"],
                    "people": people, "banner": f"NO HELMET x{bad}" if bad else None})
