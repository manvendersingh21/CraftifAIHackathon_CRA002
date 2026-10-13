"""Fill SUBMISSION.md placeholders from MEASURED result files only (results/*.json). Anything without data keeps
its [PLACEHOLDER], so nothing is ever invented. Usage: python scripts/fill_submission.py [--out SUBMISSION.filled.md]
"""
import argparse, json, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="SUBMISSION.md")
ap.add_argument("--out", default="SUBMISSION.filled.md")
a = ap.parse_args()
load = lambda f: json.loads(Path(f).read_text()) if Path(f).exists() else {}
met, ben, trt = load("results/metrics.json"), load("results/bench.json"), load("results/trt.json")
n = lambda v, d=1: f"{v:.{d}f}"
rep = {}
if met:
    nh = met.get("per_class", {}).get("no_helmet", {})
    rep.update(MAP50=n(met["map50"], 3), MAP50_95=n(met["map50_95"], 3), MAP_PT=n(met["map50"], 3))
    if nh:
        rep.update(P_NO_HELMET=n(nh["precision"], 3), R_NO_HELMET=n(nh["recall"], 3))
if ben:
    rep.update(FPS_PT=n(ben["pytorch_fps"]), LAT_PT=n(ben["pytorch_latency_ms"]), SIZE_PT=n(ben["pt_size_mb"]),
               FPS_ORT=n(ben["ort_fps"]), LAT_ORT=n(ben["ort_latency_ms"]), SIZE_ONNX=n(ben["onnx_size_mb"]))
    hw = ben.get("gpu")
    if hw:
        rep["HARDWARE"] = hw
for tag in ("fp16", "int8"):
    t = trt.get(tag, {})
    if "fps" in t:
        T = tag.upper()
        rep.update({f"FPS_{T}": n(t["fps"]), f"LAT_{T}": n(t["latency_ms"], 2) if t.get("latency_ms") else f"[LAT_{T}]",
                    f"SIZE_{T}": n(t["engine_size_mb"])})
if "fps" in trt.get("fp16", {}) and "fps" in trt.get("int8", {}):
    rep["SPEEDUP_INT8"] = n(trt["int8"]["fps"] / trt["fp16"]["fps"], 2)
try:
    ds = Path("data/construction-ppe/images/train")
    rep["N_TRAIN"] = str(len(list(ds.iterdir())))
except OSError:
    pass
txt = Path(a.src).read_text()
for k, v in rep.items():
    txt = txt.replace(f"[{k}]", v)
if trt.get("int8", {}).get("fps") and not trt["int8"].get("calibrated"):
    txt += "\n> INT8 speed measured with an UNCALIBRATED engine (trtexec default scales); INT8 accuracy not reported.\n"
Path(a.out).write_text(txt)
left = sorted({m for m in __import__("re").findall(r"\[([A-Z0-9_]+)\]", txt)})
print(f"filled {len(rep)} values -> {a.out}\nstill unmeasured: {', '.join(left) or 'none'}")
