#!/usr/bin/env python3
"""Workplace Safety Sentinel live dashboard: MQTT and/or JSON-log tail -> SSE -> browser.

  python dashboard/app.py                                   # MQTT localhost:1883, topic '#'
  python dashboard/app.py --broker 10.0.0.5 --topic ppe/events
  python dashboard/app.py --no-mqtt --file ds_events.jsonl  # tail a JSON log (one JSON per line,
                                                            # or concatenated pretty-printed objects)
Env vars: MQTT_HOST, MQTT_PORT, MQTT_TOPIC, JSON_FILE, PORT.
"""
import argparse, json, os, queue, sys, threading, time
from pathlib import Path
from collections import OrderedDict, deque
from flask import Flask, Response, jsonify, render_template, request

from parser import parse

app = Flask(__name__)
LOCK = threading.Lock()
STATE = {"messages": 0, "incidents": 0, "violators": set(), "persons": set(), "label_counts": {},
         "last_seen": {}, "status": {}, "started": time.time()}
RECENT = deque(maxlen=200)      # recent incident cards
RAW = deque(maxlen=50)          # recent raw messages
VIDEO = {"jpeg": None, "seq": 0, "meta": {}, "times": deque(maxlen=60)}   # latest annotated frame + arrival times
VCOND = threading.Condition()
EVIDENCE = OrderedDict()        # (frame, id) -> JPEG crop of the violator, newest last
INC_TIMES = deque(maxlen=5000)  # receive time of each incident, for the per-minute timeline
TIMELINE_MINUTES = 15
SUBS = []                       # SSE subscriber queues
COOLDOWN = 10.0                 # same (track id, label) re-alerts only after this many seconds unseen


def broadcast(ev):
    for q in list(SUBS):
        try:
            q.put_nowait(ev)
        except queue.Full:
            pass


def ingest(raw, topic, origin):
    msg = parse(raw, topic)
    new = []
    now = time.time()
    with LOCK:
        STATE["messages"] += 1
        RAW.append({"topic": topic, "origin": origin, "raw": msg["raw"][:1500], "t": now})
        for o in msg["objects"]:
            STATE["label_counts"][o["label"]] = STATE["label_counts"].get(o["label"], 0) + 1
            if o["person"] and o["id"]:
                STATE["persons"].add((o["source"], o["id"]))
        for v in msg["violations"]:
            key = (str(v["source"]), v["id"] or f"anon-{STATE['incidents']}", v["label"].lower())
            last = STATE["last_seen"].get(key)
            STATE["last_seen"][key] = now
            if last is not None and now - last < COOLDOWN:
                continue
            STATE["incidents"] += 1
            if v["id"]:
                STATE["violators"].add((str(v["source"]), v["id"]))
            card = {"n": STATE["incidents"], "label": v["label"], "id": v["id"], "conf": v["conf"],
                    "ts": v["ts"], "frame": v["frame"], "source": v["source"], "topic": topic,
                    "origin": origin, "received": now, "raw": msg["raw"][:1500],
                    "bbox": v.get("bbox"), "size": v.get("size"),
                    "evidence": f"/evidence/{v['frame']}/{v['id']}.jpg" if v["frame"] is not None and v["id"] else None}
            RECENT.appendleft(card)
            INC_TIMES.append(now)
            new.append(card)
        stats = snapshot_stats()
    tracks = [{"id": o["id"], "label": o["label"], "bbox": o["bbox"], "size": o["size"], "conf": o["conf"],
               "violation": o["violation"], "person": o["person"], "frame": o["frame"], "source": o["source"]}
              for o in msg["objects"] if o.get("bbox")]
    if tracks:
        broadcast({"type": "track", "data": tracks})
    broadcast({"type": "raw", "data": {"topic": topic, "origin": origin, "raw": msg["raw"][:1500],
                                       "parsed": len(msg["objects"]), "json_ok": msg["json_ok"]}})
    for c in new:
        broadcast({"type": "incident", "data": c})
    broadcast({"type": "stats", "data": stats})


def compliance_pct(persons, violators):
    """Compliance % = (1 - violators/persons) * 100, clamped to 0..100; None when no persons seen."""
    if not persons:
        return None
    return round(max(0.0, min(1.0, 1 - violators / persons)) * 100, 1)


def timeline(now=None, minutes=TIMELINE_MINUTES):
    """Incident counts per minute for the last `minutes` minutes, oldest first."""
    now = now or time.time()
    cur = int(now // 60)
    buckets = [0] * minutes
    for t in INC_TIMES:
        i = minutes - 1 - (cur - int(t // 60))
        if 0 <= i < minutes:
            buckets[i] += 1
    return buckets


def video_stats():
    t = list(VIDEO["times"])
    fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 and t[-1] > t[0] else 0.0
    live = bool(t) and time.time() - t[-1] < 3
    m = VIDEO["meta"]
    return {"live": live, "fps": round(fps, 1) if live else 0.0, "frames": VIDEO["seq"],
            "backend": m.get("backend"), "source": m.get("source"), "infer_fps": m.get("infer_fps"),
            "width": m.get("width"), "height": m.get("height"), "model": m.get("model")}


def snapshot_stats():
    return {"video": video_stats(), "compliance": compliance_pct(len(STATE["persons"]), len(STATE["violators"])),
            "timeline": timeline(), "messages": STATE["messages"], "incidents": STATE["incidents"],
            "unique_violators": len(STATE["violators"]), "unique_persons": len(STATE["persons"]),
            "label_counts": STATE["label_counts"], "status": STATE["status"]}


def set_status(k, v):
    with LOCK:
        STATE["status"][k] = v
        s = snapshot_stats()
    broadcast({"type": "stats", "data": s})


# ---------------- sources ----------------
def mqtt_worker(host, port, topic):
    import paho.mqtt.client as mqtt
    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:  # paho < 2
        c = mqtt.Client()

    def on_connect(cl, ud, flags, rc, *a):
        set_status("mqtt", f"connected {host}:{port} '{topic}'")
        cl.subscribe(topic)

    def on_disconnect(cl, *a):
        set_status("mqtt", f"disconnected from {host}:{port}, retrying")

    c.on_connect = on_connect
    c.on_disconnect = on_disconnect
    c.on_message = lambda cl, ud, m: ingest(m.payload, m.topic, "mqtt")
    c.reconnect_delay_set(1, 10)
    while True:
        try:
            c.connect(host, port, 30)
            c.loop_forever(retry_first_connection=True)
        except Exception as e:
            set_status("mqtt", f"error {host}:{port}: {e}; retrying")
            time.sleep(3)


def iter_json_chunks(buf):
    """Yield complete JSON values from a buffer (JSONL or concatenated objects). Returns rest."""
    dec = json.JSONDecoder()
    out, i = [], 0
    while True:
        while i < len(buf) and buf[i] in " \t\r\n,":
            i += 1
        if i >= len(buf):
            return out, ""
        try:
            _, end = dec.raw_decode(buf, i)
            out.append(buf[i:end]); i = end
        except json.JSONDecodeError:
            nl = buf.find("\n", i)
            # a non-JSON complete line (e.g. log prefix): pass it through as raw text
            if nl != -1 and buf[i] not in "{[":
                out.append(buf[i:nl]); i = nl + 1
                continue
            return out, buf[i:]


def file_worker(path, from_start):
    pos, buf, ino = None, "", None
    while True:
        try:
            st = os.stat(path)
        except FileNotFoundError:
            set_status("file", f"waiting for {path}")
            time.sleep(1); continue
        if ino != st.st_ino or (pos is not None and st.st_size < pos):  # new / rotated / truncated
            ino, buf = st.st_ino, ""
            pos = 0 if (from_start or pos is not None) else st.st_size
            set_status("file", f"tailing {path}")
        if st.st_size > pos:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(pos); data = f.read(); pos = f.tell()
            chunks, buf = iter_json_chunks(buf + data)
            for ch in chunks:
                ingest(ch, os.path.basename(path), "file")
        time.sleep(0.25)


# ---------------- web ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    with LOCK:
        return jsonify({"stats": snapshot_stats(), "incidents": list(RECENT)[:50], "raw": list(RAW)[-20:]})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Clear all counters/incidents so the demo can restart clean (sources keep running)."""
    with LOCK:
        STATE.update({"messages": 0, "incidents": 0, "violators": set(), "persons": set(),
                      "label_counts": {}, "last_seen": {}, "started": time.time()})
        RECENT.clear(); RAW.clear(); INC_TIMES.clear(); EVIDENCE.clear()
        stats = snapshot_stats()
    try:
        import live
        live.reset_sessions()
    except ImportError:
        pass
    broadcast({"type": "reset", "data": stats})
    return {"ok": True}


def video_tick(meta):
    """Called per analyzed browser frame so stats.video reports the true processed fps / backend."""
    with VCOND:
        VIDEO["meta"] = meta
        VIDEO["times"].append(time.time())
        VIDEO["seq"] += 1
        VCOND.notify_all()


def put_evidence(frame_bgr, box, frame_no, tid, pad=0.25, width=320):
    """Crop the violator from a numpy BGR frame and store it for /evidence/<frame>/<id>.jpg."""
    import cv2
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    crop = frame_bgr[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
    if crop.size == 0:
        return
    if crop.shape[1] > width:
        crop = cv2.resize(crop, (width, int(crop.shape[0] * width / crop.shape[1])))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if ok:
        with LOCK:
            EVIDENCE[(str(frame_no), str(tid))] = buf.tobytes()
            while len(EVIDENCE) > 300:
                EVIDENCE.popitem(last=False)


@app.route("/api/frame", methods=["POST"])
def api_frame():
    """Inference process pushes the latest annotated JPEG (+ optional X-Meta JSON header) here."""
    jpg = request.get_data()
    if not jpg.startswith(b"\xff\xd8"):
        return {"ok": False, "error": "not a JPEG"}, 400
    try:
        meta = json.loads(request.headers.get("X-Meta", "{}"))
    except ValueError:
        meta = {}
    with VCOND:
        VIDEO["jpeg"], VIDEO["meta"] = jpg, meta or VIDEO["meta"]
        VIDEO["seq"] += 1
        VIDEO["times"].append(time.time())
        VCOND.notify_all()
    return {"ok": True}


@app.route("/video")
def video():
    """MJPEG stream of the annotated frames (use as <img src="/video">)."""
    def gen():
        seq = -1
        while True:
            with VCOND:
                VCOND.wait_for(lambda: VIDEO["seq"] != seq, timeout=5)
                jpg, seq = VIDEO["jpeg"], VIDEO["seq"]
            if jpg is None:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


@app.route("/snapshot.jpg")
def snapshot():
    if VIDEO["jpeg"] is None:
        return Response(status=404)
    return Response(VIDEO["jpeg"], mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/api/evidence", methods=["POST"])
def api_evidence():
    """POST a JPEG crop of a violator: /api/evidence?frame=N&id=TRACK. Served at /evidence/N/TRACK.jpg."""
    jpg = request.get_data()
    key = (request.args.get("frame", ""), request.args.get("id", ""))
    if not jpg.startswith(b"\xff\xd8") or not all(key):
        return {"ok": False}, 400
    with LOCK:
        EVIDENCE[key] = jpg
        while len(EVIDENCE) > 300:
            EVIDENCE.popitem(last=False)
    return {"ok": True}


@app.route("/evidence/<frame>/<tid>.jpg")
def evidence(frame, tid):
    jpg = EVIDENCE.get((frame, tid))
    return Response(jpg, mimetype="image/jpeg") if jpg else Response(status=404)


@app.route("/api/tracks", methods=["POST"])
def api_tracks():
    """Per-frame positions of every tracked person (no counters touched): [{id,label,bbox,size,violation}]."""
    try:
        tracks = request.get_json(force=True)
    except Exception:
        return {"ok": False}, 400
    if isinstance(tracks, list):
        broadcast({"type": "track", "data": tracks[:100]})
    return {"ok": True}


VIDEOS = Path(__file__).resolve().parents[1] / "data" / "videos"


@app.route("/api/videos")
def api_videos():
    """Real recorded footage available for live testing (data/videos/manifest.json: source URL + license per clip)."""
    try:
        clips = json.loads((VIDEOS / "manifest.json").read_text())
    except (OSError, ValueError):
        clips = []
    return jsonify([c for c in clips if (VIDEOS / c.get("file", "")).is_file()])


@app.route("/videos/<path:name>")
def videos(name):
    from flask import send_from_directory
    return send_from_directory(VIDEOS, name, conditional=True)    # Range requests -> seekable <video>


@app.route("/api/ingest", methods=["POST"])
def api_ingest():  # handy for HTTP-posting probes / curl tests
    ingest(request.get_data(), "http", "http")
    return {"ok": True}


@app.route("/stream")
def stream():
    q = queue.Queue(maxsize=1000)
    SUBS.append(q)

    def gen():
        try:
            with LOCK:
                yield f"data: {json.dumps({'type': 'stats', 'data': snapshot_stats()})}\n\n"
            while True:
                try:
                    ev = q.get(timeout=15)
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            SUBS.remove(q)
    return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache",
                                                                  "X-Accel-Buffering": "no"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default=os.getenv("MQTT_HOST", "localhost"))
    ap.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", 1883)))
    ap.add_argument("--topic", default=os.getenv("MQTT_TOPIC", "#"))
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--file", default=os.getenv("JSON_FILE"), help="JSON/JSONL log to tail")
    ap.add_argument("--from-start", action="store_true", help="replay existing file content")
    ap.add_argument("--cooldown", type=float, default=10.0)
    ap.add_argument("--model", default=os.getenv("PPE_MODEL", str(Path(__file__).resolve().parents[1] / "models" / "ppe_yolov8n.onnx")),
                    help="ONNX model for live camera analysis")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--detector-url", default=os.getenv("DETECTOR_URL"),
                    help="run detection on another box, e.g. http://gpu-box:5000 (its /api/detect)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", 5000)))
    a = ap.parse_args()
    global COOLDOWN
    COOLDOWN = a.cooldown
    import live
    live.init(sys.modules[__name__], a.model, a.labels, a.conf, a.detector_url)
    if not a.no_mqtt:
        threading.Thread(target=mqtt_worker, args=(a.broker, a.mqtt_port, a.topic), daemon=True).start()
    if a.file:
        threading.Thread(target=file_worker, args=(a.file, a.from_start), daemon=True).start()
    app.run(host=a.host, port=a.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
