#!/usr/bin/env bash
# Prepare DeepStream-Yolo: clone, export our .pt with its exporter, build the custom parser.
# Usage: bash scripts/export_deepstream.sh [DS_YOLO_DIR]   (default /opt/DeepStream-Yolo; needs write access)
# Env: CUDA_VER (auto-detected from nvcc), WEIGHTS (models/ppe_yolov8n.pt), PY (python3)
# NOT TESTED ON A GPU HOST.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
DIR=${1:-/opt/DeepStream-Yolo}; PY=${PY:-python3}; WEIGHTS=${WEIGHTS:-models/ppe_yolov8n.pt}
[ -f "$WEIGHTS" ] || { echo "ERROR: $WEIGHTS not found - run train.py first" >&2; exit 1; }
command -v git >/dev/null  || { echo "ERROR: git missing" >&2; exit 1; }
command -v make >/dev/null || { echo "ERROR: make missing (apt-get install build-essential)" >&2; exit 1; }
if [ -z "${CUDA_VER:-}" ]; then
  command -v nvcc >/dev/null || { echo "ERROR: nvcc not found; set CUDA_VER=12.x (e.g. 12.6 for DS 7.1) or add /usr/local/cuda/bin to PATH" >&2; exit 1; }
  CUDA_VER=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -n1)
fi
[ -n "$CUDA_VER" ] || { echo "ERROR: could not determine CUDA_VER" >&2; exit 1; }
echo "[export] CUDA_VER=$CUDA_VER  DeepStream-Yolo dir=$DIR"
if [ ! -d "$DIR/.git" ]; then
  mkdir -p "$(dirname "$DIR")" || { echo "ERROR: cannot create $(dirname "$DIR") (try sudo or pass another dir)" >&2; exit 1; }
  git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo "$DIR"
fi
[ -f "$DIR/utils/export_yoloV8.py" ] || { echo "ERROR: $DIR/utils/export_yoloV8.py missing (repo layout changed?)" >&2; exit 1; }
# The exporter needs the weights next to where it runs; it writes <name>.onnx and labels.txt in the cwd.
OUT=$REPO/models/deepstream_export; mkdir -p "$OUT"
cp "$WEIGHTS" "$OUT/"; W=$(basename "$WEIGHTS")
( cd "$OUT" && "$PY" "$DIR/utils/export_yoloV8.py" -w "$W" --opset 12 -s 640 )
ls "$OUT"
ONNX=$(ls "$OUT"/*.onnx 2>/dev/null | head -n1 || true)
[ -n "$ONNX" ] && [ -f "$OUT/labels.txt" ] || { echo "ERROR: exporter did not produce onnx + labels.txt in $OUT (check exporter output above; if it printed a different location, copy manually)" >&2; exit 1; }
CUDA_VER=$CUDA_VER make -C "$DIR/nvdsinfer_custom_impl_Yolo" clean || true
CUDA_VER=$CUDA_VER make -C "$DIR/nvdsinfer_custom_impl_Yolo"
LIB=$DIR/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
[ -f "$LIB" ] || { echo "ERROR: build did not produce $LIB" >&2; exit 1; }
cat <<MSG

[export] OK. Edit deepstream/config_infer_primary_yolov8_ppe.txt:
  onnx-file=../models/deepstream_export/$(basename "$ONNX")
  model-engine-file=../models/deepstream_export/$(basename "$ONNX")_b1_gpu0_fp16.engine   # delete old engine if present
  labelfile-path=../models/deepstream_export/labels.txt
  custom-lib-path=$LIB
  parse-bbox-func-name=NvDsInferParseYolo
  num-detected-classes=11   # verify against labels.txt line count: $(wc -l < "$OUT/labels.txt")
  cluster-mode=2            # DeepStream-Yolo's own configs use 2 (NMS in parser); keep nms-iou-threshold
Notes:
 - The DeepStream-Yolo exporter's ONNX has its own output layout (boxes+scores, NMS-free); do NOT mix it with
   models/ppe_yolov8n.onnx, which the ORT fallback and trtexec bench keep using.
 - Its sample configs also set  maintain-aspect-ratio=1 symmetric-padding=1 ; already in our config.
 - Check the labels.txt order matches data/ppe.yaml (helmet,gloves,vest,... no_helmet=7); ppe_deepstream.py reads
   models/labels.txt for class ids, so the two files must agree.
MSG
