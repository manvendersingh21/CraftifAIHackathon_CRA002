"""Violation event sink: JSONL file and/or MQTT, in the schema dashboard/parser.py already understands."""
import json, time


class EventSink:
    def __init__(self, jsonl=None, broker=None, topic="ppe/events", sensor="cam0", frame_size=None, on_message=None):
        self.sensor, self.topic, self.n, self.frame_size = sensor, topic, 0, frame_size
        self.on_message = on_message          # callable(json_line): in-process delivery (dashboard/live.py)
        self.f = open(jsonl, "a", buffering=1) if jsonl else None
        self.mqtt = None
        if broker:
            import paho.mqtt.client as mqtt
            host, _, port = broker.partition(":")
            try:
                self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            except AttributeError:  # paho < 2
                self.mqtt = mqtt.Client()
            self.mqtt.connect(host, int(port or 1883), 30)
            self.mqtt.loop_start()

    def emit(self, ev):
        msg = {"@timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z",
               "sensorId": self.sensor, "frame_num": ev["frame"],
               "object": {"label": ev["label"], "tracker_id": ev["track_id"], "confidence": ev["confidence"],
                          "bbox": ev["bbox"]}}
        if self.frame_size:
            msg["frame_size"] = list(self.frame_size)
        line = json.dumps(msg)
        if self.f:
            self.f.write(line + "\n")
        if self.mqtt:
            self.mqtt.publish(self.topic, line)
        if self.on_message:
            self.on_message(line)
        self.n += 1

    def close(self):
        if self.f:
            self.f.close()
        if self.mqtt:
            time.sleep(0.3)                    # let queued QoS0 publishes drain before tearing the loop down
            self.mqtt.disconnect(); self.mqtt.loop_stop()


class DashboardPublisher:
    """Pushes annotated frames, track positions and violator evidence crops to dashboard/app.py over HTTP.
    A single background thread always sends the *latest* frame (drops older ones), so inference never blocks."""

    def __init__(self, url, max_width=960, quality=72, max_fps=30.0):
        import http.client, threading, urllib.parse
        u = urllib.parse.urlparse(url)
        self._mk = lambda: http.client.HTTPConnection(u.hostname, u.port or 80, timeout=3)
        self.max_width, self.quality, self.min_dt = max_width, quality, 1.0 / max_fps
        self._lock, self._cv = threading.Lock(), threading.Condition()
        self._frame = self._tracks = None
        self._evidence = []
        self._last, self._stop = 0.0, False
        self.errors = 0
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _enc(self, img, width=None):
        import cv2
        h, w = img.shape[:2]
        if width and w > width:
            img = cv2.resize(img, (width, int(h * width / w)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ok else None

    def push(self, frame_bgr, meta, tracks=None):
        now = time.time()
        if now - self._last < self.min_dt:
            return
        self._last = now
        jpg = self._enc(frame_bgr, self.max_width)
        with self._cv:
            self._frame, self._tracks = (jpg, meta), tracks
            self._cv.notify()

    def evidence(self, frame_bgr, box, frame_no, tid, pad=0.25):
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        crop = frame_bgr[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
        if crop.size == 0:
            return
        jpg = self._enc(crop, 320)
        with self._cv:
            self._evidence.append((frame_no, tid, jpg))
            self._cv.notify()

    def _post(self, conn, path, body, ctype, headers=None):
        h = {"Content-Type": ctype, **(headers or {})}
        conn.request("POST", path, body, h)
        conn.getresponse().read()

    def _run(self):
        conn = self._mk()
        while not self._stop:
            with self._cv:
                self._cv.wait_for(lambda: self._frame or self._evidence or self._stop, timeout=1)
                frame, tracks, ev = self._frame, self._tracks, self._evidence
                self._frame, self._tracks, self._evidence = None, None, []
            try:
                for fno, tid, jpg in ev:
                    self._post(conn, f"/api/evidence?frame={fno}&id={tid}", jpg, "image/jpeg")
                if tracks:
                    self._post(conn, "/api/tracks", json.dumps(tracks).encode(), "application/json")
                if frame:
                    self._post(conn, "/api/frame", frame[0], "image/jpeg", {"X-Meta": json.dumps(frame[1])})
            except Exception:
                self.errors += 1
                conn.close(); conn = self._mk(); time.sleep(0.5)

    def close(self):
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        self._t.join(timeout=3)
