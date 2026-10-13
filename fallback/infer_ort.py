"""ONNX Runtime + OpenCV PPE monitor (fallback when DeepStream/PipeGen is unavailable).
Usage: python fallback/infer_ort.py --source video.mp4|0 --out out.mp4 [--show]
           [--events ds_events.jsonl] [--mqtt localhost:1883 --topic ppe/events]   # feed dashboard/
"""
import argparse, sys, time
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from detector import ALERT, GOOD, Detector, draw_box
from rules import NO_HELMET, PPEMonitor
from events import DashboardPublisher, EventSink

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ppe_yolov8n.onnx")
    ap.add_argument("--labels", default="models/labels.txt")
    ap.add_argument("--source", default="0")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--events", help="append violation events to this JSONL file (dashboard --file)")
    ap.add_argument("--mqtt", help="host[:port] MQTT broker to publish violation events to")
    ap.add_argument("--topic", default="ppe/events")
    ap.add_argument("--dashboard", help="dashboard base URL, e.g. http://localhost:5000: live video + evidence + tracks")
    ap.add_argument("--loop", action="store_true", help="loop a video file forever (demo mode)")
    ap.add_argument("--sensor", default="cam0")
    ap.add_argument("--require-vest", action="store_true", help="also flag persons without a vest (no_vest)")
    ap.add_argument("--helmet-conf", type=float, default=0.2,
                    help="lower confidence floor for the helmet class (a sighting only resets the absence rule)")
    ap.add_argument("--absence", type=int, default=15,
                    help="helmet-absence rule: frames without a helmet on a tracked person before alerting (0=off)")
    ap.add_argument("--min-hits", type=int, default=3, help="frames a person must be tracked before alerting")
    ap.add_argument("--cooldown", type=int, default=150, help="frames before the same track re-alerts")
    a = ap.parse_args()
    det = Detector(a.model, a.labels, a.conf, a.iou, a.helmet_conf)
    print("providers:", det.sess.get_providers())
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened(): raise SystemExit(f"cannot open {a.source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))
    mon = PPEMonitor(require_vest=a.require_vest, min_hits=a.min_hits, cooldown=a.cooldown, absence=a.absence)
    sink = EventSink(a.events, a.mqtt, a.topic, a.sensor, (W, H)) if (a.events or a.mqtt) else None
    pub = DashboardPublisher(a.dashboard) if a.dashboard else None
    backend = "ONNX Runtime " + det.provider
    src_name = "camera" if a.source.isdigit() else Path(a.source).name
    total_viol, n, t0, fps = 0, 0, time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok and a.loop and not a.source.isdigit():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, frame = cap.read()
        if not ok: break
        t = time.time()
        named = det.detect(frame)
        events = mon.update(named, n)
        for tid, conf in mon.new_persons:               # one 'Person' event per new track -> dashboard headcount
            bx = mon.box_of(tid)
            if sink:
                sink.emit({"track_id": tid, "label": "Person", "confidence": round(conf, 3),
                           "bbox": [int(v) for v in bx] if bx else [], "frame": n})
        for ev in events:
            if sink:
                sink.emit(ev)
            if pub:
                pub.evidence(frame, ev["bbox"], n, ev["track_id"])
        total_viol += len(events)
        for x1, y1, x2, y2, name, s in named:
            if name == "Person":
                continue                                # people are drawn from tracks below
            col = (60, 60, 255) if name in ALERT else (110, 220, 90) if name in GOOD else (230, 190, 60)
            draw_box(frame, x1, y1, x2, y2, col, f"{name} {s:.0%}", thin=True)
        now_bad, tracks = 0, []
        for (bx1, by1, bx2, by2), tid, labs in mon.annotations(n):
            bad = bool(labs)
            now_bad += bad
            col = (60, 60, 255) if bad else (110, 220, 90)
            draw_box(frame, int(bx1), int(by1), int(bx2), int(by2), col,
                     f"#{tid} " + ("NO HELMET" if NO_HELMET in labs else "NO VEST" if labs else "PPE OK"), thin=False)
            tracks.append({"id": str(tid), "label": "Person", "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
                           "size": [W, H], "violation": bad})
        fps = 0.9 * fps + 0.1 * (1 / max(time.time() - t, 1e-6)) if n else 1 / max(time.time() - t, 1e-6)
        if now_bad:
            cv2.rectangle(frame, (0, 0), (W, 50), (0, 0, 255), -1)
            cv2.putText(frame, f"NO HELMET x{now_bad}", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        cv2.putText(frame, f"FPS {fps:.1f}  persons {len(mon.persons)}  violators {len(mon.violators)}  alerts {total_viol}",
                    (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if pub:
            pub.push(frame, {"backend": backend, "source": src_name, "infer_fps": round(fps, 1), "width": W,
                             "height": H, "model": det.model_name, "frame": n}, tracks)
        writer.write(frame); n += 1
        if a.show:
            cv2.imshow("PPE", frame)
            if cv2.waitKey(1) == 27: break
        if a.max_frames and n >= a.max_frames: break
    cap.release(); writer.release()
    if sink:
        sink.close()
    if pub:
        pub.close()
    print(f"frames={n} avg_fps={n / (time.time() - t0):.2f} unique_persons={len(mon.persons)} "
          f"unique_violators={len(mon.violators)} alerts={total_viol} -> {a.out}")

if __name__ == "__main__":
    main()
