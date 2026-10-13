"""Measure PyTorch (ultralytics) vs ONNX Runtime FPS on the same frames. Prints measured numbers only."""
import argparse, json, os, time, cv2, numpy as np, onnxruntime as ort, torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--pt", default="models/ppe_yolov8n.pt")
ap.add_argument("--onnx", default="models/ppe_yolov8n.onnx")
ap.add_argument("--source", default=None, help="video; random frames if omitted")
ap.add_argument("--n", type=int, default=100); ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--json", default="results/bench.json")
ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

frames = []
if a.source:
    cap = cv2.VideoCapture(a.source)
    while len(frames) < a.n:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.resize(f, (640, 640)))
if not frames:
    frames = [np.random.randint(0, 255, (640, 640, 3), np.uint8) for _ in range(a.n)]

def bench(fn):
    for f in frames[:a.warmup]: fn(f)
    t = time.perf_counter()
    for f in frames: fn(f)
    return len(frames) / (time.perf_counter() - t)

m = YOLO(a.pt)
pt_fps = bench(lambda f: m.predict(f, imgsz=640, device=a.device, verbose=False))
provs = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
s = ort.InferenceSession(a.onnx, providers=provs); name = s.get_inputs()[0].name
def ort_fn(f):
    x = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
    s.run(None, {name: x})
ort_fps = bench(ort_fn)
print(f"device={a.device} ort_providers={s.get_providers()} frames={len(frames)}")
print(f"PyTorch (ultralytics predict, incl. pre/post): {pt_fps:.2f} FPS")
print(f"ONNX Runtime (raw forward + preproc):          {ort_fps:.2f} FPS")
os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
json.dump({"device": a.device, "ort_providers": s.get_providers(), "frames": len(frames),
           "pytorch_fps": pt_fps, "pytorch_latency_ms": 1000 / pt_fps, "pt_size_mb": os.path.getsize(a.pt) / 1e6,
           "ort_fps": ort_fps, "ort_latency_ms": 1000 / ort_fps, "onnx_size_mb": os.path.getsize(a.onnx) / 1e6,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, open(a.json, "w"), indent=2)
