"""Bridge: the PipeGen-built DeepStream pipeline -> our PPE rules -> the live dashboard.

PipeGen compiles and runs the edge pipeline (TensorRT engine + DeepStream app). Its outputs are consumed here:
  * detections: PipeGen's MQTT JSON output (--mqtt host:port --topic T) or its JSON metadata file (--json PATH)
  * video:      PipeGen's RTSP output (--rtsp rtsp://host:8554/...) -> shown live in the dashboard (optional)
Each frame's detections go through fallback/rules.PPEMonitor (tracking, helmet-absence rule, cooldown), and the
resulting events/evidence/tracks are posted to dashboard/app.py exactly like the browser-camera path.

  python integrations/pipegen_bridge.py --mqtt localhost:1883 --topic '#' --rtsp rtsp://localhost:8554/pipegen \
      --dashboard http://localhost:5000

Frame schema handled (PipeGen v0.5.1 lab, JSON output): {"frame": n, "width": W, "height": H,
  "objects": [{"label": "...", "confidence": c, "bbox": {"left","top","width","height"}}], ...}; also a whole file
{"frames": [...]} and lists of such frames. Unknown layouts are logged, never guessed.
"""
import argparse, json, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fallback"))
from events import DashboardPublisher, EventSink   # noqa: E402
from rules import PPEMonitor                       # noqa: E402

LABEL_ALIASES = {"person": "Person", "people": "Person", "hardhat": "helmet", "hard_hat": "helmet",
                 "no-helmet": "no_helmet", "nohelmet": "no_helmet", "no_hardhat": "no_helmet"}


def frames_of(doc):
    """Yield frame dicts from a PipeGen JSON message/file (one frame, {'frames': [...]}, or a list)."""
    if isinstance(doc, list):
        for d in doc:
            yield from frames_of(d)
    elif isinstance(doc, dict):
        if isinstance(doc.get("frames"), list):
            yield from frames_of(doc["frames"])
        elif isinstance(doc.get("objects"), list):
            yield doc


def dets_of(frame):
    """PipeGen objects -> [(x1, y1, x2, y2, label, score)] in pixels."""
    out = []
    for o in frame.get("objects", []):
        lab = str(o.get("label", o.get("class", "")))
        lab = LABEL_ALIASES.get(lab.lower(), lab)
        b = o.get("bbox") or {}
        if isinstance(b, dict) and {"left", "top", "width", "height"} <= set(b):
            x1, y1 = float(b["left"]), float(b["top"])
            x2, y2 = x1 + float(b["width"]), y1 + float(b["height"])
        elif isinstance(b, (list, tuple)) and len(b) == 4:        # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(float, b)
        else:
            continue
        out.append((int(x1), int(y1), int(x2), int(y2), lab, float(o.get("confidence", o.get("score", 0.0)))))
    return out


class Bridge:
    def __init__(self, dashboard, sensor="pipegen", monitor_kw=None):
        self.dash = dashboard.rstrip("/")
        self.mon = PPEMonitor(**(monitor_kw or {}))
        self.pub = DashboardPublisher(self.dash)
        self.sink = EventSink(None, None, sensor=sensor, on_message=self._post_event)
        self.latest = None            # latest RTSP frame (BGR) for evidence crops
        self.n = 0
        self.unknown = 0

    def _post_event(self, line):
        req = urllib.request.Request(self.dash + "/api/ingest", line.encode(), {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=2).read()
        except OSError as e:
            print("[bridge] dashboard unreachable:", e)

    def on_frame(self, frame):
        dets = dets_of(frame)
        W, H = frame.get("width"), frame.get("height")
        if W and H:
            self.sink.frame_size = (int(W), int(H))
        n = int(frame.get("frame", self.n))
        self.n = n + 1
        events = self.mon.update(dets, n)
        for tid, conf in self.mon.new_persons:
            bx = self.mon.box_of(tid)
            self.sink.emit({"track_id": tid, "label": "Person", "confidence": round(conf, 3),
                            "bbox": [int(v) for v in bx] if bx else [], "frame": n})
        for ev in events:
            if self.latest is not None:
                self.pub.evidence(self.latest, ev["bbox"], n, ev["track_id"])
            self.sink.emit(ev)
        tracks = [{"id": str(t), "label": "Person", "bbox": [int(v) for v in b], "size": [W, H], "violation": bool(l)}
                  for b, t, l in self.mon.annotations(n)]
        if tracks:
            body = json.dumps(tracks).encode()
            try:
                urllib.request.urlopen(urllib.request.Request(self.dash + "/api/tracks", body,
                                                              {"Content-Type": "application/json"}), timeout=2).read()
            except OSError:
                pass

    def on_message(self, raw):
        try:
            doc = json.loads(raw)
        except ValueError:
            self.unknown += 1
            return
        got = False
        for fr in frames_of(doc):
            got = True
            self.on_frame(fr)
        if not got:
            self.unknown += 1
            if self.unknown <= 3:
                print("[bridge] unrecognised PipeGen message (first 300 chars):", str(raw)[:300])


def rtsp_worker(bridge, url):
    """Show PipeGen's own annotated RTSP output as the dashboard hero video."""
    import cv2
    while True:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print("[bridge] waiting for RTSP", url); time.sleep(2); continue
        while True:
            ok, f = cap.read()
            if not ok:
                break
            bridge.latest = f
            bridge.pub.push(f, {"backend": "PipeGen DeepStream (TensorRT)", "source": url.rsplit("/", 1)[-1],
                                "width": f.shape[1], "height": f.shape[0], "model": "PipeGen pipeline"})
        cap.release(); time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mqtt", help="host[:port] of the broker PipeGen's MQTT output publishes to")
    ap.add_argument("--topic", default="#")
    ap.add_argument("--json", help="PipeGen JSON metadata file (read once after the run, or --follow)")
    ap.add_argument("--follow", action="store_true", help="re-read --json as it grows")
    ap.add_argument("--rtsp", help="PipeGen RTSP output URL to show in the dashboard")
    ap.add_argument("--dashboard", default="http://localhost:5000")
    ap.add_argument("--absence", type=int, default=15)
    a = ap.parse_args()
    br = Bridge(a.dashboard, monitor_kw={"absence": a.absence})
    if a.rtsp:
        threading.Thread(target=rtsp_worker, args=(br, a.rtsp), daemon=True).start()
    if a.json:
        seen = 0
        while True:
            try:
                doc = json.loads(Path(a.json).read_text())
                frames = list(frames_of(doc))
                for fr in frames[seen:]:
                    br.on_frame(fr)
                seen = len(frames)
            except (OSError, ValueError):
                pass                                    # file still being written / not there yet
            if not a.follow:
                break
            time.sleep(1)
        print(f"[bridge] processed {seen} frames, persons={len(br.mon.persons)} violators={len(br.mon.violators)}")
    if a.mqtt:
        import paho.mqtt.client as mqtt
        host, _, port = a.mqtt.partition(":")
        try:
            c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            c = mqtt.Client()
        c.on_connect = lambda cl, *x: (cl.subscribe(a.topic), print("[bridge] subscribed", a.topic))
        c.on_message = lambda cl, ud, m: br.on_message(m.payload)
        c.connect(host, int(port or 1883), 30)
        c.loop_forever()
    elif a.rtsp:
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
