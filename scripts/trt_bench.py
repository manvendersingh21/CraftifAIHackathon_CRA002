"""Build TensorRT engines with trtexec (FP16, INT8) and record measured throughput/latency/size -> results/trt.json.

NOTE: trtexec --int8 without a calibration cache uses placeholder scales: the speed number is valid, accuracy is
NOT. Pass --calib-cache (e.g. produced by DeepStream-Yolo INT8 calibration) for a meaningful INT8 engine.
Usage: python scripts/trt_bench.py [--onnx models/ppe_yolov8n.onnx] [--calib-cache calib.table]
"""
import argparse, json, os, re, subprocess, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default="models/ppe_yolov8n.onnx")
ap.add_argument("--trtexec", default=shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec")
ap.add_argument("--calib-cache")
ap.add_argument("--out", default="results/trt.json")
a = ap.parse_args()
os.makedirs("results", exist_ok=True)

def run(tag, flags):
    eng = f"models/ppe_yolov8n_{tag}.engine"
    cmd = [a.trtexec, f"--onnx={a.onnx}", f"--saveEngine={eng}", "--warmUp=500", "--iterations=500",
           "--avgRuns=100", *flags]
    log = subprocess.run(cmd, capture_output=True, text=True)
    open(f"results/trtexec_{tag}.log", "w").write(log.stdout + log.stderr)
    txt = log.stdout
    thr = re.findall(r"Throughput: ([\d.]+) qps", txt)
    lat = re.findall(r"Latency: .*?mean = ([\d.]+) ms", txt)
    if log.returncode or not thr:
        return {"error": f"trtexec failed (see results/trtexec_{tag}.log)"}
    return {"fps": float(thr[-1]), "latency_ms": float(lat[-1]) if lat else None,
            "engine_size_mb": os.path.getsize(eng) / 1e6, "flags": flags}

res = {"fp16": run("fp16", ["--fp16"])}
i8 = ["--int8"] + ([f"--calib={a.calib_cache}"] if a.calib_cache else [])
res["int8"] = run("int8", i8)
res["int8"]["calibrated"] = bool(a.calib_cache)
json.dump(res, open(a.out, "w"), indent=2)
print(json.dumps(res, indent=2))
