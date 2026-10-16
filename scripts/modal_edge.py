"""Run scripts/edge_bench.py on a Modal GPU (real repo files, real clip). Writes results/edge.json locally.

  modal run scripts/modal_edge.py            (profile manvendersingh21, GPU from MODAL_GPU, default A10G)
"""
import os, pathlib, modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
GPU = os.environ.get("MODAL_GPU", "A10G")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install("ultralytics==8.3.*", "onnx", "onnxslim", "onnxruntime-gpu", "tensorrt-cu12==10.3.0",
                 "opencv-python-headless", "pyyaml", "numpy", "pytest")
    .add_local_dir(ROOT / "scripts", "/app/scripts")
    .add_local_dir(ROOT / "models", "/app/models")
    .add_local_dir(ROOT / "data" / "videos", "/app/data/videos")
    .add_local_file(ROOT / "data" / "ppe.yaml", "/app/data/ppe.yaml")
)
app = modal.App("ppe-edge-bench", image=image)


@app.function(gpu=GPU, timeout=3600)
def bench() -> dict:
    import subprocess, json, pathlib as p
    os.chdir("/app")
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], check=False)
    cmd = ["python", "scripts/edge_bench.py", "--device", "0", "--source", "data/videos/wm_concrete_pour_far.mp4",
           "--n", "500", "--calib-fraction", "1.0", "--out", "results/edge.json"]
    log = subprocess.run(cmd, capture_output=True, text=True)
    return {"rc": log.returncode, "stdout": log.stdout[-4000:], "stderr": log.stderr[-4000:],
            "edge": json.loads(p.Path("results/edge.json").read_text()) if p.Path("results/edge.json").exists() else None}


@app.local_entrypoint()
def main():
    out = bench.remote()
    print("rc", out["rc"])
    print(out["stderr"][-1500:])
    if out["edge"] is None:
        raise SystemExit("no results/edge.json produced")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "edge.json").write_text(__import__("json").dumps(out["edge"], indent=2))
    print("wrote results/edge.json")
