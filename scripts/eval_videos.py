"""Run the live pipeline (Detector + PPEMonitor, same code as dashboard/live.py) over every real video in
data/videos/manifest.json and record MEASURED per-clip results + annotated contact sheets for human review.

  python scripts/eval_videos.py [--manifest data/videos/manifest.json] [--stride 2] [--max-seconds 90]

Outputs: results/real_videos/eval.json, results/real_videos/<clip>_sheet.jpg (12 evenly spaced annotated frames).
Nothing here is ground truth: the sheets exist so a person can check the alerts by eye.
"""
import argparse, json, sys, time
from pathlib import Path

import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fallback"))
from detector import Detector, draw_box          # noqa: E402
from rules import NO_HELMET, PPEMonitor          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=str(ROOT / "data" / "videos" / "manifest.json"))
ap.add_argument("--model", default=str(ROOT / "models" / "ppe_yolov8n.onnx"))
ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (tracker sees the same stride)")
ap.add_argument("--max-seconds", type=float, default=90)
ap.add_argument("--out", default=str(ROOT / "results" / "real_videos"))
a = ap.parse_args()

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
clips = json.loads(Path(a.manifest).read_text())
det = Detector(a.model)
print(f"detector: {det.provider} {det.model_name}; {len(clips)} clips")
report = []
for c in clips:
    path = Path(a.manifest).parent / c["file"]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        report.append({"file": c["file"], "error": "cannot open"}); continue
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0, a.max_seconds * fps_in))
    mon = PPEMonitor(absence=max(3, int(15 / a.stride)), min_hits=max(2, int(3 / a.stride)),
                     cooldown=max(10, int(150 / a.stride)))
    picks = set(np.linspace(0, max(total - 1, 0), 12).astype(int).tolist())
    sheet, n, idx, ms, viol_frames, people_frames, alerts = [], 0, 0, [], 0, 0, []
    while idx < total:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % a.stride:
            idx += 1; continue
        t = time.perf_counter(); dets = det.detect(frame); ms.append((time.perf_counter() - t) * 1000)
        evs = mon.update(dets, n)
        alerts += [{"t": round(idx / fps_in, 2), "track": e["track_id"], "label": e["label"]} for e in evs]
        ann = mon.annotations(n)
        people_frames += bool(ann); viol_frames += any(l for _, _, l in ann)
        hit = next((p for p in picks if abs(idx - p) < a.stride), None)
        if hit is not None:
            picks.discard(hit)
            f = frame.copy()
            for x1, y1, x2, y2, name, s in dets:
                if name != "Person":
                    draw_box(f, x1, y1, x2, y2, (60, 60, 255) if name in ("no_helmet", "head") else (110, 220, 90), f"{name} {s:.0%}", thin=True)
            for (bx1, by1, bx2, by2), tid, labs in ann:
                draw_box(f, int(bx1), int(by1), int(bx2), int(by2), (60, 60, 255) if labs else (110, 220, 90),
                         f"#{tid} " + ("NO HELMET" if NO_HELMET in labs else "OK"))
            cv2.putText(f, f"t={idx / fps_in:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            sheet.append(cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1]))))
        n += 1; idx += 1
    cap.release()
    if sheet:
        h = max(s.shape[0] for s in sheet)
        sheet = [cv2.copyMakeBorder(s, 0, h - s.shape[0], 0, 0, cv2.BORDER_CONSTANT) for s in sheet]
        while len(sheet) % 4:
            sheet.append(np.zeros_like(sheet[0]))
        cv2.imwrite(str(out / f"{Path(c['file']).stem}_sheet.jpg"),
                    np.vstack([np.hstack(sheet[i:i + 4]) for i in range(0, len(sheet), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    r = {"file": c["file"], "provider": c.get("provider"), "license": c.get("license"), "scene": c.get("scene"),
         "helmets_expected": c.get("helmets"), "frames_processed": n, "stride": a.stride,
         "detect_ms_median": round(float(np.median(ms)), 1) if ms else None,
         "unique_persons": len(mon.persons), "unique_violators": len(mon.violators), "alerts": len(alerts),
         "frames_with_people_pct": round(100 * people_frames / max(n, 1), 1),
         "frames_with_violation_pct": round(100 * viol_frames / max(n, 1), 1), "first_alerts": alerts[:10]}
    report.append(r)
    print(f"{c['file']:40s} expected={str(c.get('helmets')):12s} persons={r['unique_persons']:3d} "
          f"violators={r['unique_violators']:3d} viol_frames={r['frames_with_violation_pct']:5.1f}% det={r['detect_ms_median']}ms")
(out / "eval.json").write_text(json.dumps({"detector": f"{det.provider} {det.model_name}", "stride": a.stride,
                                            "clips": report}, indent=2))
print("wrote", out / "eval.json")
