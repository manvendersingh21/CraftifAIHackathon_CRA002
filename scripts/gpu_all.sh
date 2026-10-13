#!/usr/bin/env bash
# One-command GPU-day runner. Every stage can be skipped with SKIP_<STAGE>=1.
# Stages: ENV INSTALL DATA TRAIN BENCH TRT ORT FILL.  Optional stages never abort the run; TRAIN failure does
# skip stages that need the model (BENCH/TRT/ORT) because they would be meaningless without it.
# Env knobs: EPOCHS (50) DEVICE (0) SOURCE (samples/sample.mp4) PY (python3) BENCH_N (200)
# NOT TESTED ON A GPU HOST - read results/gpu_all.log afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}; EPOCHS=${EPOCHS:-50}; DEVICE=${DEVICE:-0}; SOURCE=${SOURCE:-samples/sample.mp4}; BENCH_N=${BENCH_N:-200}
mkdir -p results models
exec > >(tee -a results/gpu_all.log) 2>&1
echo "== gpu_all.sh $(date -u +%FT%TZ) =="
declare -a SUMMARY=()
stage() { # name, required(0/1), cmd...
  local name=$1 req=$2; shift 2
  local flag="SKIP_$name"
  if [ "${!flag:-0}" = 1 ]; then echo "[skip] $name"; SUMMARY+=("$name: skipped"); return 0; fi
  echo; echo "=== [$name] $* ==="
  local rc=0; "$@" || rc=$?
  if [ "$rc" = 0 ]; then SUMMARY+=("$name: ok"); return 0; fi
  echo "!!! stage $name FAILED (rc=$rc)"; SUMMARY+=("$name: FAILED rc=$rc")
  [ "$req" = 1 ] && return "$rc"; return 0
}
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing prerequisite: $1"; return 1; }; }

env_log() {
  {
    echo "date: $(date -u +%FT%TZ)"; echo "host: $(uname -a)"
    if command -v nvidia-smi >/dev/null; then nvidia-smi; else echo "nvidia-smi NOT FOUND (no NVIDIA driver?)"; fi
    command -v nvcc >/dev/null && nvcc --version || echo "nvcc: not found"
    command -v trtexec >/dev/null && echo "trtexec: $(command -v trtexec)" || echo "trtexec: not on PATH (/usr/src/tensorrt/bin/trtexec?)"
    command -v deepstream-app >/dev/null && deepstream-app --version-all || echo "deepstream-app: not found"
    "$PY" --version
    "$PY" - <<'PYEOF'
import importlib
for m in ("torch","ultralytics","onnxruntime","cv2","onnx"):
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "NOT IMPORTABLE:", e)
try:
    import torch; print("torch.cuda.is_available:", torch.cuda.is_available())
except Exception: pass
try:
    import onnxruntime as o; print("ORT providers:", o.get_available_providers())
except Exception: pass
PYEOF
  } > results/env.txt 2>&1 || true
  cat results/env.txt
}
install_reqs() {
  need "$PY"
  if "$PY" -c "import ultralytics, onnxruntime, cv2, onnx" 2>/dev/null; then echo "requirements already importable"; return 0; fi
  "$PY" -m pip install -r requirements.txt
}
train() {
  need "$PY"; [ -f data/ppe.yaml ] || { echo "data/ppe.yaml missing (DATA stage failed?)"; return 1; }
  "$PY" train.py --epochs "$EPOCHS" --device "$DEVICE"
}
model_ok() { [ -f models/ppe_yolov8n.onnx ] || { echo "models/ppe_yolov8n.onnx missing; skipping"; return 1; }; }
bench() { model_ok && [ -f "$SOURCE" ] && "$PY" benchmark.py --source "$SOURCE" --n "$BENCH_N"; }
trt() {
  model_ok
  if command -v trtexec >/dev/null || [ -x /usr/src/tensorrt/bin/trtexec ]; then "$PY" scripts/trt_bench.py
  else echo "trtexec not found - skipping TensorRT bench"; return 1; fi
}
ort() {
  model_ok; [ -f "$SOURCE" ] || { echo "$SOURCE missing"; return 1; }
  "$PY" fallback/infer_ort.py --source "$SOURCE" --out results/ort_output.mp4 --events results/events.jsonl
}
fill() { "$PY" scripts/fill_submission.py; }

stage ENV     0 env_log
stage INSTALL 1 install_reqs
stage DATA    1 bash data/get_dataset.sh
stage TRAIN   1 train || { echo "Training failed/skipped required stage; continuing only with stages that can run"; }
stage BENCH   0 bench
stage TRT     0 trt
stage ORT     0 ort
stage FILL    0 fill

echo; echo "=== SUMMARY ==="; printf '%s\n' "${SUMMARY[@]}"
echo "Outputs: results/env.txt results/*.json results/ort_output.mp4 results/events.jsonl SUBMISSION.filled.md"
echo "Next (DeepStream): bash scripts/export_deepstream.sh ; bash scripts/calib_images.sh"
