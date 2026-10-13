"""Shared detector: ONNX Runtime YOLOv8 inference (letterbox, decode, NMS) + box drawing. Used by the CLI
(fallback/infer_ort.py), the browser-camera service (dashboard/live.py) and anything else needing detections."""
import time
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort

DEFAULT_NAMES = ["helmet","gloves","vest","boots","goggles","none","Person",
                 "no_helmet","no_goggle","no_gloves","no_boots"]
ALERT = {"no_helmet"}
GOOD = {"helmet", "vest"}


def draw_box(img, x1, y1, x2, y2, col, label, thin=False):
    """Corner-bracket box with a filled label chip (cleaner than a full rectangle)."""
    t = 1 if thin else 3
    L = max(10, int(min(x2 - x1, y2 - y1) * 0.18))
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * L, py), col, t, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * L), col, t, cv2.LINE_AA)
    if not thin:
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
    fs = 0.45 if thin else 0.6
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    top = max(0, y1 - th - 10)
    cv2.rectangle(img, (x1, top), (x1 + tw + 10, top + th + 8), col, -1)
    cv2.putText(img, label, (x1 + 5, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (15, 15, 15), 1, cv2.LINE_AA)


def letterbox(img, size):
    h, w = img.shape[:2]; r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top+nh, left:left+nw] = cv2.resize(img, (nw, nh))
    return out, r, left, top

def postprocess(pred, conf, iou, r, dx, dy, W, H, class_conf=None):
    p = pred[0].T                      # (8400, 4+nc)
    scores = p[:, 4:]; cls = scores.argmax(1); sc = scores.max(1)
    thr = np.full(len(sc), conf, np.float32)
    for c, v in (class_conf or {}).items():   # per-class floor, e.g. a lower threshold for helmet
        thr[cls == c] = v
    m = sc > thr; p, cls, sc = p[m], cls[m], sc[m]
    if not len(p): return []
    xy, wh = p[:, :2], p[:, 2:4]
    boxes = np.concatenate([xy - wh / 2, wh], 1)
    keep = cv2.dnn.NMSBoxesBatched(boxes.tolist(), sc.tolist(), cls.tolist(), min(conf, *(class_conf or {0: conf}).values()), iou)
    out = []
    for i in np.array(keep).flatten():
        x, y, w, h = boxes[i]
        x1, y1 = (x - dx) / r, (y - dy) / r
        x2, y2 = x1 + w / r, y1 + h / r
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            continue                                   # entirely in the letterbox padding -> spurious
        out.append((int(cx1), int(cy1), int(cx2), int(cy2), int(cls[i]), float(sc[i])))
    return out


class Detector:
    """Loads the ONNX model once; `detect(bgr)` returns [(x1,y1,x2,y2,name,score)] in frame pixel coordinates."""

    def __init__(self, model, labels=None, conf=0.35, iou=0.45, helmet_conf=0.2, providers=None):
        labels = Path(labels) if labels else Path(model).with_name("labels.txt")
        self.names = labels.read_text().split() if labels.exists() else DEFAULT_NAMES
        try:                                           # find CUDA/cuDNN shipped as pip wheels (e.g. via torch)
            ort.preload_dlls()
        except Exception:
            pass
        avail = ort.get_available_providers()
        # CUDA on NVIDIA boxes, CoreML on Apple silicon (identical detections, ~6x faster than CPU), else CPU.
        prov = providers or [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")
                             if p in avail]
        self.sess = ort.InferenceSession(str(model), providers=prov)
        inp = self.sess.get_inputs()[0]
        self.inp_name, self.size = inp.name, int(inp.shape[2])
        self.conf, self.iou = conf, iou
        self.class_conf = {self.names.index("helmet"): helmet_conf} if "helmet" in self.names else None
        self.provider = self.sess.get_providers()[0].replace("ExecutionProvider", "")
        self.model_name = Path(model).name

    def detect(self, frame):
        H, W = frame.shape[:2]
        img, r, dx, dy = letterbox(frame, self.size)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
        raw = postprocess(self.sess.run(None, {self.inp_name: x})[0], self.conf, self.iou, r, dx, dy, W, H,
                          self.class_conf)
        return [(x1, y1, x2, y2, self.names[c] if c < len(self.names) else str(c), sc)
                for x1, y1, x2, y2, c, sc in raw]
