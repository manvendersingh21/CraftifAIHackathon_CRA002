"""Fine-tune YOLOv8n on PPE data and export ONNX (opset 12, static 640)."""
import argparse, json, shutil
from pathlib import Path
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/ppe.yaml")
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--device", default=None, help="0 for GPU, cpu for CPU")
p.add_argument("--weights", default="yolov8n.pt", help="e.g. yolov8s.pt for a bigger model")
p.add_argument("--name", default="ppe", help="run name; non-default names save to models/<name>/ and results/<name>_metrics.json")
p.add_argument("--fraction", type=float, default=1.0, help="fraction of train set")
a = p.parse_args()

model = YOLO(a.weights)
r = model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                device=a.device, fraction=a.fraction, project="runs", name=a.name, exist_ok=True)
best = Path(r.save_dir) / "weights" / "best.pt"
out = Path("models") if a.name == "ppe" else Path("models") / a.name
out.mkdir(parents=True, exist_ok=True)
shutil.copy(best, out / "ppe_yolov8n.pt")
onnx = YOLO(str(best)).export(format="onnx", opset=12, dynamic=False, imgsz=a.imgsz, simplify=True)
shutil.copy(onnx, out / "ppe_yolov8n.onnx")
with open(out / "labels.txt", "w") as f:
    f.write("\n".join(model.names[i] for i in sorted(model.names)) + "\n")
print(f"Saved {out}/ppe_yolov8n.pt, {out}/ppe_yolov8n.onnx, {out}/labels.txt")

# measured validation metrics -> results/metrics.json (consumed by scripts/fill_submission.py)
m = YOLO(str(best)).val(data=a.data, imgsz=a.imgsz, device=a.device, split="val", verbose=False)
names = {i: model.names[i] for i in model.names}
per = {names[int(c)]: {"precision": float(m.box.p[i]), "recall": float(m.box.r[i]), "map50": float(m.box.ap50[i]),
                       "map50_95": float(m.box.ap[i])} for i, c in enumerate(m.box.ap_class_index)}
Path("results").mkdir(exist_ok=True)
json.dump({"epochs": a.epochs, "imgsz": a.imgsz, "data": a.data, "map50": float(m.box.map50),
           "map50_95": float(m.box.map), "precision": float(m.box.mp), "recall": float(m.box.mr), "per_class": per},
          open("results/metrics.json" if a.name == "ppe" else f"results/{a.name}_metrics.json", "w"), indent=2)
print("Saved results/metrics.json  mAP50=%.3f mAP50-95=%.3f" % (m.box.map50, m.box.map))
