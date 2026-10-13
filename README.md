# PPE PipeGen — hard-hat / vest safety monitor (CraftifAI EMBEDDED SF, PipeGen track)

YOLOv8n fine-tuned on Ultralytics **construction-ppe** (11 classes: helmet, gloves, vest, boots, goggles, none,
Person, no_helmet, no_goggle, no_gloves, no_boots) → ONNX (opset 12, static 1x3x640x640, output 1x15x8400)
→ NVIDIA DeepStream (PipeGen, or the manual configs in `deepstream/`) with an ONNX Runtime fallback.

## Live demo (this is the demo)
```bash
pip install -r requirements.txt flask paho-mqtt     # onnxruntime-gpu on NVIDIA, plain onnxruntime on a Mac (CoreML is used automatically)
python dashboard/app.py --no-mqtt                   # open http://localhost:5000, click "Go live" (or press C), allow the camera
```
The browser camera streams JPEG frames to `/api/analyze`; the server runs the YOLOv8n ONNX model, tracks people, applies the
helmet rule and pushes everything back (boxes over your own video, KPI tiles, 3D twin, incident cards with evidence photos).
Everything shown (FPS, ms, backend, model) is measured live. Use `localhost` or https (browsers block cameras on plain http
to other hosts). Measured on an Apple-silicon laptop: ~9 ms detection per frame (CoreML) vs ~37 ms (CPU).
`--detector-url http://gpu-box:5000` makes this dashboard use another machine's GPU for detection, but network latency
decides whether that is usable live (from our venue-less test: 17 ms on a T4 but ~500 ms round trip over Tailscale).
`P` = presentation mode, `F` = fullscreen, `C` = camera on/off.

## Layout
```
data/get_dataset.sh        download dataset (YOLO format) -> data/ppe.yaml ; SUBSET=N for a tiny smoke set
train.py                   fine-tune yolov8n + export models/ppe_yolov8n.{pt,onnx}, models/labels.txt
fallback/infer_ort.py      ORT+OpenCV: boxes, red NO HELMET banner, violation counter, FPS, mp4 out
benchmark.py               PyTorch vs ORT FPS (measured, nothing hard-coded)
deepstream/                nvinfer config, deepstream-app config, PARSER_NOTES.md (custom YOLOv8 parser)
scripts/                   gpu_all.sh, export_deepstream.sh, calib_images.sh, trt_bench.py, fill_submission.py
Dockerfile, requirements.txt, demo.sh
```

## Run
```bash
pip install -r requirements.txt            # onnxruntime instead of -gpu on CPU
bash data/get_dataset.sh                   # full set (1132 train / 143 val / 141 test)
python train.py --epochs 50 --device 0     # GPU; smoke test: SUBSET=40 bash data/get_dataset.sh && python train.py --data data/ppe-subset.yaml --epochs 1 --device cpu
python fallback/infer_ort.py --source samples/sample.mp4 --out output.mp4   # --source 0 for webcam, --show for window
python benchmark.py --source samples/sample.mp4 --n 200
```

### Fallback flags (`fallback/infer_ort.py`, tracked violations)
| Flag | Meaning |
|---|---|
| `--events FILE` | append violation events as JSONL (feed to `dashboard/app.py --no-mqtt --file FILE`) |
| `--mqtt host[:port]` / `--topic` | publish events to an MQTT broker (default topic `ppe/events`) |
| `--require-vest` | also flag persons without a vest |
| `--min-hits N` | frames a person must be tracked before alerting (default 3) |
| `--cooldown N` | frames before the same track may re-alert (default 150) |
Also `--sensor`, `--conf`, `--iou`, `--max-frames`.

### Scripts
- `scripts/gpu_all.sh` - whole GPU day in one command (env log, install, data, train, benchmark, trtexec, ORT run, fill submission). Skip stages with `SKIP_ENV|INSTALL|DATA|TRAIN|BENCH|TRT|ORT|FILL=1`; knobs `EPOCHS DEVICE SOURCE PY`. Log: `results/gpu_all.log`.
- `scripts/export_deepstream.sh [dir]` - clone DeepStream-Yolo (default `/opt/DeepStream-Yolo`), export with its exporter into `models/deepstream_export/`, build the parser, print the config edits.
- `scripts/calib_images.sh [N=500] [dir]` - pick calibration images and print INT8 env vars/config.
- `scripts/trt_bench.py`, `scripts/fill_submission.py` - measured TensorRT numbers and SUBMISSION fill (nothing invented).

### Samples
`samples/sample.mp4` (any codec, for ORT/benchmark) and `samples/sample_h264.mp4` (H.264, required by the DeepStream `qtdemux ! h264parse ! nvv4l2decoder` path). Regenerate with `samples/make_sample.py`.

### DeepStream (Docker, NVIDIA GPU + nvidia-container-toolkit)
```bash
docker build -t ppe-ds .                           # --build-arg DS_TAG=... to change base image
# Way 1: deepstream-app (config paths assume /app)
docker run --gpus all --rm -v $PWD:/app ppe-ds     # writes /app/ds_output.mp4
# Way 2: python pad-probe app (tracker-ID rule, NO HELMET banner, events)
docker run --gpus all --rm -v $PWD:/app ppe-ds python3 deepstream/ppe_deepstream.py \
    --source samples/sample_h264.mp4 --out ds_output.mp4 --events ds_events.jsonl
```
First run builds the TensorRT FP16 engine (takes minutes). The DeepStream-Yolo exporter writes its own ONNX +
`labels.txt`; run `scripts/export_deepstream.sh` and apply the printed config edits (see `deepstream/PARSER_NOTES.md`).
Both DeepStream paths are unverified on a GPU until you run them. `pyds` is installed from a release wheel in the Dockerfile; confirm with `python3 -c "import pyds"`.

### GPU day checklist
1. `nvidia-smi` works; `docker run --gpus all` works if using Docker; `git pull` / copy repo + `samples/`.
2. `bash scripts/gpu_all.sh` (check `results/env.txt`, `results/gpu_all.log` summary; failures are listed, not hidden).
3. Review `results/metrics.json`, `bench.json`, `trt.json`; `SUBMISSION.filled.md` has only measured values, rest stay `[PLACEHOLDER]`.
4. `bash scripts/export_deepstream.sh`, edit `deepstream/config_infer_primary_yolov8_ppe.txt` as printed, delete stale `.engine` files.
5. `python3 deepstream/ppe_deepstream.py ...` or `deepstream-app`; confirm class ids/labels match `models/labels.txt` and boxes look right.
6. `bash scripts/calib_images.sh 500`, export the printed env vars, switch `network-mode=1`, rebuild engine, compare FP16 vs INT8.
7. Start dashboard (`dashboard/app.py`) with `--events` file or MQTT; record the demo; keep `./demo.sh` as ORT fallback if DeepStream misbehaves.

## Demo script (≈2 min)
1. Problem: construction sites, missing hard hats = #1 head-injury cause; manual watching doesn't scale.
2. Show `./demo.sh samples/sample.mp4` output: green helmet/vest, red box + **NO HELMET** banner, counter, FPS.
3. Show PipeGen-generated pipeline (or `deepstream/` configs): source → nvstreammux → nvinfer → nvtracker → nvdsosd → sink.
4. Show `benchmark.py` numbers measured on the demo hardware (PyTorch vs ORT vs DeepStream perf log).
5. Next: alerts via MQTT/Slack, per-tracked-person counting, Jetson deployment.

## Notes
- The model from a 1-epoch / 40-image smoke run is NOT usable (lots of false positives) – train ≥50 epochs on the full set.
- Violation counter = no_helmet detections summed per frame (not unique people); use tracker IDs for unique counts.

## Live dashboard (`dashboard/`)
```bash
pip install -r dashboard/requirements.txt
mosquitto -p 1883 &                                    # or any broker
python dashboard/app.py --broker localhost --topic '#' # http://localhost:5000, events from DeepStream (nvmsgbroker) or infer_ort --mqtt
# no broker? tail the JSONL the pipeline writes instead:
python dashboard/app.py --no-mqtt --file ds_events.jsonl
```
No simulated data anywhere: every event comes from the model running on a real camera or real recorded footage.
Click "enable sound" once in the browser (autoplay policy) for the beep. See `SUBMISSION.md` for the write-up draft.
