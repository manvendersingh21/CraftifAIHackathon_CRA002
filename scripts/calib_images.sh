#!/usr/bin/env bash
# Pick N train images for INT8 calibration. Usage: bash scripts/calib_images.sh [N=500] [OUT_DIR=calib_images]
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-500}; OUT=${2:-calib_images}; SRC=${SRC:-data/construction-ppe/images/train}
case "$N" in ''|*[!0-9]*) echo "ERROR: N must be an integer" >&2; exit 1;; esac
[ -d "$SRC" ] || { echo "ERROR: $SRC missing - run bash data/get_dataset.sh" >&2; exit 1; }
mkdir -p "$OUT"; rm -f "$OUT"/*
COUNT=0
while IFS= read -r f; do cp "$f" "$OUT/"; COUNT=$((COUNT+1)); done < <(find "$SRC" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort | head -n "$N")
[ "$COUNT" -gt 0 ] || { echo "ERROR: no images found in $SRC" >&2; exit 1; }
[ "$COUNT" -lt "$N" ] && echo "WARNING: only $COUNT images available (< $N requested)"
ABS=$(cd "$OUT" && pwd); BATCH=${BATCH:-8}
cat <<MSG
Copied $COUNT images to $ABS

export INT8_CALIB_IMG_PATH=$ABS
export INT8_CALIB_BATCH_SIZE=$BATCH      # DeepStream-Yolo reads these env vars; the nvinfer engine build does calibration

Config edits in deepstream/config_infer_primary_yolov8_ppe.txt:
  network-mode=1
  int8-calib-file=../models/calib.table
  batch-size=$BATCH     # keep equal to INT8_CALIB_BATCH_SIZE (verify against DeepStream-Yolo docs/INT8Calibration.md)
  model-engine-file=../models/deepstream_export/<onnx>_b${BATCH}_gpu0_int8.engine   # remove any stale engine first
Then run once with the env vars exported; calib.table is written on first engine build, later runs reuse it.
Optional: python scripts/trt_bench.py --calib-cache models/calib.table
MSG
