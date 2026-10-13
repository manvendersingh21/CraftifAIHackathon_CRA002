#!/usr/bin/env bash
# 60-second demo. Usage: ./demo.sh [source]   SHOW=1 for a window; DS_MODE=py to use the pyds app instead of deepstream-app
# DeepStream if available (needs models/ + DeepStream-Yolo parser, see README), else ORT fallback.
set -euo pipefail
cd "$(dirname "$0")"
SRC=${1:-samples/sample.mp4}
[ -f "$SRC" ] || { echo "[demo] source not found: $SRC" >&2; exit 1; }
if command -v deepstream-app >/dev/null && [ -f /opt/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so ]; then
  if [ "${DS_MODE:-app}" = py ]; then
    echo "[demo] DeepStream python pad-probe app"
    python3 deepstream/ppe_deepstream.py --source samples/sample_h264.mp4 --out ds_output.mp4 --events results/events.jsonl ${SHOW:+--display} \
      || { echo "[demo] DeepStream failed, falling back to ORT" >&2; DS_FAIL=1; }
  else
    echo "[demo] deepstream-app (config uses /app paths; run in the container or edit uri/output-file)"
    (cd deepstream && deepstream-app -c deepstream_app_config.txt) || { echo "[demo] DeepStream failed, falling back to ORT" >&2; DS_FAIL=1; }
  fi
else
  DS_FAIL=1
fi
if [ "${DS_FAIL:-0}" = 1 ]; then
  echo "[demo] ORT fallback"; mkdir -p results
  python3 fallback/infer_ort.py --source "$SRC" --out demo_output.mp4 --events results/events.jsonl ${SHOW:+--show}
fi
python3 benchmark.py --source "$SRC" --n 100 || echo "[demo] benchmark failed (non-fatal)"
