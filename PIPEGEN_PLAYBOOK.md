# PipeGen Playbook: PPE / hard-hat pipeline on the T4 VM

Target: `ssh pipegen@100.95.90.74` (GCP, Ubuntu 24.04, Tesla T4, VNC desktop on `:1`).
Flow in PipeGen: **Requirement cards -> Pipeline IR graph -> Model inspect/compile/benchmark -> Compile Pipeline -> Run (video console + latency trace) -> Export Project**.

Only a human does the steps marked **[HUMAN]**: sign-up, login, portal downloads, and anything in the PipeGen UI that needs an account.

---

## 0. What is already prepared on the VM

| Item | Where / value | Status |
|---|---|---|
| socat | `/usr/bin/socat` | installed |
| Docker + NVIDIA Container Toolkit, user in `docker` group | n/a | already present; `docker run --gpus all ... nvidia-smi` shows Tesla T4, driver 595.99 |
| PPE model (raw Ultralytics ONNX) | `/home/pipegen/pipegen_inputs/ppe_yolov8n.onnx` | opset 12, input `images` [1,3,640,640], output `output0` [1,15,8400] |
| Labels (11 lines, order matters) | `/home/pipegen/pipegen_inputs/labels.txt` | helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots |
| Fallback DeepStream-Yolo export (only if needed, see 6.3) | `/home/pipegen/pipegen_inputs/ds/ppe_yolov8n_dsyolo.onnx` | output `output` [1,8400,6] (x1,y1,x2,y2,score,class) |
| Real site footage (Wikimedia Commons; licences in `data/videos/manifest.json`) | `/home/pipegen/pipegen_inputs/videos/wm_*.mp4` + `site_loop.mp4` (4 clips joined, 1280x720, 25 fps, H.264 Main, no B-frames, GOP 25, 112 s) | present |
| **RTSP camera** | **`rtsp://localhost:8554/site`** (from a bridge-network container: `rtsp://172.17.0.1:8554/site`; over Tailscale: `rtsp://100.95.90.74:8554/site`) | live, looping forever; verified with ffprobe and with `nvv4l2decoder` inside the DeepStream 7.1 container (25 fps, 0 dropped) |
| MQTT broker (mosquitto) | `127.0.0.1:1883` and `172.17.0.1:1883`, anonymous | running |
| Dashboard (ours) | `http://<vm>:5000`, `~/ppe-pipegen/dashboard/app.py` | running with `--no-mqtt`; restart it with MQTT in step 8 |

Services (systemd, start on boot): `pipegen-mediamtx`, `pipegen-rtsp-feed`, `mosquitto`.
```
systemctl status pipegen-mediamtx pipegen-rtsp-feed mosquitto
sudo systemctl restart pipegen-rtsp-feed          # restart the loop
ffprobe -v error -show_entries stream=codec_name,width,height -of csv=p=0 rtsp://localhost:8554/site
```
mediamtx config: `~/pipegen_inputs/mediamtx/mediamtx.yml` (only RTSP is enabled; RTMP, HLS, WebRTC and SRT are off).

Disk: 84 GB free before PipeGen. PipeGen's image needs about 35 GB, which leaves about 49 GB, so the existing 31.7 GB `nvcr.io/nvidia/deepstream:7.1-triton-multiarch` image is **kept**. Our repo `Dockerfile` builds `FROM nvcr.io/nvidia/deepstream:${DS_TAG}`, and PipeGen also targets DeepStream 7.1, so it may share layers. If space runs short later, delete it with `docker rmi nvcr.io/nvidia/deepstream:7.1-triton-multiarch` (frees about 32 GB).

---

## 1. What we know about PipeGen (sources)

- **Models come in as ONNX**, either trained in Edge MLOps or your own ("Models come in as ONNX, whether you trained them in Edge MLOps or bring your own"). The deployment package includes a "Custom output parser written for your model's format, plus its build file" and an "Application configuration and class label file". Source: https://craftifai.com/products/pipegen
- The **raw YOLOv8 tensor layout is the case PipeGen is built for**. CraftifAI's parser article walks through YOLOv8 `[1, 84, N]` (features x proposals, `output[feature * num_proposals + i]`) and an `NvDsInferParseYoloV8` parser. PipeGen's output for that article is `deepstream_app_config.txt, config_infer_primary.txt, nvdsparser_yolov8m.cpp, Makefile, ds_source.cfg, ds_preprocess.cfg, ds_output.cfg, labels.txt, run.sh`, on DeepStream 7.1 for Jetson Orin and x86 NVIDIA GPUs. Our `[1,15,8400]` (4 box + 11 classes) is the same layout, so **no DeepStream-Yolo re-export is needed**. Source: https://craftifai.com/deepstream-custom-parser
- The reference nvinfer config in the config article uses `net-scale-factor=0.00392156862745098`, `cluster-mode=2`, `maintain-aspect-ratio=0`, `parse-bbox-func-name=NvDsInferParseYoloV8`, `output-blob-names=output0`, `labelfile-path=...labels.txt`, `num-detected-classes=80` (ours: **11**). Source: https://craftifai.com/deepstream-pipeline-config
- **Inputs:** USB/CSI camera, video file, RTSP stream, Intel RealSense, and an image folder (v0.5). **Outputs:** on-screen display, recorded video file, RTSP publish, **MQTT metadata**, **JSON metadata**. The FAQ says "It does not generate new protocols", so there is no Kafka. Sources: products/pipegen FAQ; https://craftifai.com/product-updates
- **MQTT output** (v0.4): "Publish detections as JSON with configurable broker, topic, QoS, per-frame or per-object messages, publish interval, empty-frame skipping, confidence floor and credentials. Per-object topics let subscribers filter by class with a wildcard." **JSON file export:** path, source label, confidence floor; "Track IDs in the file match the ones drawn on the recording." The VMS log shows time, camera, label, ID and confidence. "Rules, alerts and event recording are coming next", so **violation logic lives in our dashboard**. Source: product-updates, products/pipegen
- **System requirements:** NVIDIA GPU and driver, Docker with the NVIDIA Container Toolkit, socat, user in the docker group, and **an X11 session ("A Wayland session is refused at startup")**. PipeGen checks all of these at launch. Model inspection sorts ops into "runs as is / reworked automatically / needs custom code". INT8 calibration uses sample images ("ideally your own footage"). Source: product-updates
- The exact JSON schema of the MQTT and JSON payloads is **not published**. `dashboard/parser.py` is schema-tolerant: it walks recursively for label, class, track_id, confidence and bbox keys. Check it against the first real message (step 8).

---

## 2. [HUMAN] Download and install

1. Sign up or log in at the CraftifAI developer portal (https://developer.craftifai.com/register). You type the email, password and emailed code yourself.
2. Download `pipegen_prerequisites_ubuntu_x86.sh` and `PipeGen_<ver>_amd64.deb` (latest, v0.5.x or newer).
3. Copy them to the VM and install:
   ```bash
   scp pipegen_prerequisites_ubuntu_x86.sh PipeGen_*_amd64.deb pipegen@100.95.90.74:~/Downloads/
   ssh pipegen@100.95.90.74
   cd ~/Downloads
   bash pipegen_prerequisites_ubuntu_x86.sh        # idempotent: docker/nvidia-ctk/socat already present
   sudo apt install ./PipeGen_*_amd64.deb
   ```
4. Open the VNC desktop (display `:1`; password in `~/VNC_PASSWORD.txt`). This is an X11 session, which PipeGen requires. Launch PipeGen from the app menu, or from a terminal *inside the VNC desktop*:
   ```bash
   DISPLAY=:1 pipegen
   ```
5. The first launch pulls about 12 GB (about 35 GB on disk). Leave it alone until the pull finishes. Watch progress with `docker images` and `df -h /`.
6. [HUMAN] Log in inside the app.

Check: the launch check passes (GPU, docker, nvidia runtime, socat, X11). If it complains about Wayland, you are not in the VNC/X11 session.

---

## 3. Requirement cards (type this)

New project name: `ppe-site-safety`. Target device: **x86 + NVIDIA GPU (Tesla T4), DeepStream 7.1**.

Requirement sentence (paste as-is):

> Detect construction workers and their PPE (hard hats, vests, gloves, boots, goggles) from an RTSP camera at rtsp://localhost:8554/site using my own YOLOv8 ONNX model, track each worker with a multi-object tracker, flag anyone without a hard hat (class "no_helmet"), overlay boxes, labels and track IDs on the video, and publish every detection as JSON over MQTT to broker 127.0.0.1:1883 on topic ppe/events, and also save detections to a JSON file.

If it asks follow-ups, use these model facts:

> Model: /home/pipegen/pipegen_inputs/ppe_yolov8n.onnx, Ultralytics YOLOv8n detect, opset 12, single input "images" 1x3x640x640 RGB float, scale 1/255, no mean subtraction. Single output "output0" shape [1,15,8400], channels-first: rows 0-3 are cx, cy, w, h in 640x640 input pixels, rows 4-14 are 11 class scores (already sigmoid). No objectness score, no NMS in the graph, so apply NMS (IoU 0.45) after a confidence threshold of 0.30. Labels: /home/pipegen/pipegen_inputs/labels.txt (11 classes, in order).

Check: the cards show **Input = RTSP**, **Model = custom ONNX (bring your own)**, **Tracker = yes**, **Outputs = OSD/display + MQTT + JSON**. Freeze or approve them.

## 4. Pipeline IR graph

Expect: `RTSP source -> (decode) -> nvstreammux -> nvinfer (ppe_yolov8n) -> nvtracker -> nvdsosd -> display/console`, plus a metadata branch to MQTT and JSON.
Check: one camera node with the RTSP URL, `num-detected-classes=11`, labels path set, and a tracker present. If the camera node defaults to USB or a file, type: *"change the input to the RTSP stream rtsp://localhost:8554/site"*.

## 5. Model: inspect -> compile -> benchmark

- Choose **bring your own ONNX** and pick `/home/pipegen/pipegen_inputs/ppe_yolov8n.onnx`, with labels `/home/pipegen/pipegen_inputs/labels.txt`. Do **not** use the `ds/` file for the first attempt.
- **Inspect:** expect every op (Conv, Sigmoid/SiLU, Mul, Add, Concat, Split, Reshape, Transpose, Softmax, MaxPool, Resize, Slice, Sub, Div) to land in "runs as it is". Input 1x3x640x640 is static (batch 1).
- **Compile: FP16** (the T4 has tensor cores; FP16 is the safe default). INT8 is optional: calibrate on our footage, `/home/pipegen/pipegen_inputs/videos/`. Only keep INT8 if its box-IoU accuracy vs FP32 is close.
- **Benchmark:** for YOLOv8n FP16 on a T4, expect roughly 1.5-3 ms mean model latency and several hundred FPS. Record the mean, p99, engine size and accuracy-vs-original numbers for the submission (screenshot them).

## 6. Compile Pipeline

PipeGen generates the config, a YOLOv8 parser `.cpp` + Makefile, labels and `run.sh`, then builds and self-corrects.
Check in the generated `config_infer_primary*.txt`:
- `onnx-file` points at our ONNX; `labelfile-path` at our labels; `num-detected-classes=11`
- `net-scale-factor=0.0039215...`, `model-color-format=0` (RGB), `output-blob-names=output0`
- `cluster-mode=2` (NMS), a `pre-cluster-threshold` of about 0.25-0.35
- Better accuracy: `maintain-aspect-ratio=1` + `symmetric-padding=1` (Ultralytics trains with letterbox). Ask for it if it is 0.
- The parser loops over **11** classes and indexes `output[c * 8400 + i]`. If it hard-codes 80 classes or reads `[8400, 15]`, the boxes will be garbage.

### 6.1 Pitfalls
| Symptom | Cause | Fix (type into PipeGen) |
|---|---|---|
| Pipeline plays, **zero boxes** | Parser layout or class count is wrong, or the threshold is too high | "The output is [1,15,8400] channels-first: 4 box rows then 11 class score rows; set num-detected-classes=11 and threshold 0.3" |
| Boxes everywhere / huge boxes | Transposed read, or boxes treated as normalized 0-1 | "Box values are in 640-pixel input coordinates (cx,cy,w,h), not normalized" |
| Wrong label names / off-by-one | Labels file order or count mismatch | Re-select `labels.txt` (exactly 11 lines; `Person` is capitalized; `none` is a real class) |
| Boxes shifted vertically | Letterbox mismatch | `maintain-aspect-ratio=1`, `symmetric-padding=1` |
| RTSP source fails to start | Container not on the host network | Use `rtsp://172.17.0.1:8554/site` (verified working from a bridge container) |
| "RTSP publish" output fails | Port 8554 is taken by our mediamtx | Set the RTSP-out port to **8555** (or skip RTSP publish) |
| MQTT connect refused | Broker address is wrong for the container network | Use `172.17.0.1` instead of `127.0.0.1` (both listen on 1883) |
| Video-file input "not found" | File not visible inside the container | Prefer RTSP; or ask PipeGen which host paths it mounts |
| Launch refused | Wayland, or not on X11 | Run inside VNC `:1` |

### 6.2 Violation logic
The model has an explicit `no_helmet` class (also `no_goggle`, `no_gloves`, `no_boots`), so "a worker without a hard hat" = a tracked `no_helmet` detection. PipeGen does not do rules or alerts yet, so the **dashboard** turns `no_helmet` messages into violation cards (`dashboard/parser.py` `VIOLATION_RE`) and counts unique tracker IDs.

### 6.3 Fallback only (if PipeGen's generated parser cannot decode the raw output after two correction attempts)
`/home/pipegen/pipegen_inputs/ds/ppe_yolov8n_dsyolo.onnx` (+ `ds/labels.txt`) is a DeepStream-Yolo export: input `input` [1,3,640,640], output `output` [1,8400,6] = x1,y1,x2,y2,score,class_id (decoded, still needs NMS). Tell PipeGen: *"output is [1,8400,6] rows of x1,y1,x2,y2,score,class_id in 640 input pixels; apply NMS 0.45"*. Alternatively use the DeepStream-Yolo parser `NvDsInferParseYolo` from `ds/src/nvdsinfer_custom_impl_Yolo` (build in the container with `CUDA_VER=12.6 make -C nvdsinfer_custom_impl_Yolo`). The manual reference configs are in the repo's `deepstream/`.

## 7. Run (video console + latency tracing)

- Re-run the compiled pipeline into the built-in VMS console. Expect live site video with boxes, labels and track IDs, and a detection log with time, camera, label, ID and confidence.
- Run the **latency trace** (about a 5 s instrumented run). It shows decode / convert / infer / track / draw time. Check that the end-to-end time supports 25 fps (the source is 25 fps).
- Screenshot: the console with `no_helmet` boxes, the latency breakdown, and the benchmark table.

## 8. Connect PipeGen output to our dashboard

On the VM, check that messages arrive:
```bash
mosquitto_sub -h 127.0.0.1 -t 'ppe/#' -v | head -5
```
Restart our dashboard with MQTT. It currently runs with `--no-mqtt` on :5000:
```bash
cd ~/ppe-pipegen
pkill -f 'dashboard/app.py' ; sleep 1
setsid nohup .venv/bin/python dashboard/app.py --broker 127.0.0.1 --topic 'ppe/#' --port 5000 \
  > results/dashboard_mqtt.log 2>&1 < /dev/null &
```
Or tail PipeGen's JSON file instead: `.venv/bin/python dashboard/app.py --no-mqtt --file /path/from/pipegen/detections.json --port 5000`.

Use **per-object** messages if offered (the topic then ends in the class, e.g. `ppe/events/no_helmet`, and `ppe/#` still matches). Open `http://100.95.90.74:5000` (Tailscale). The raw panel shows each payload and how many objects it parsed (`parsed`, `json_ok`). If `parsed` is 0 for real detections, copy one payload and add its label/ID key names to `LABEL_KEYS` / `ID_KEYS` in `dashboard/parser.py`.

## 9. Export Project

Use **Export Project** (v0.5) and save the export (it contains `run.sh`, configs, the parser, labels and binaries) into the repo, e.g. `~/ppe-pipegen/pipegen_export/`. Then copy it back to the laptop for the submission:
```bash
rsync -az pipegen@100.95.90.74:ppe-pipegen/pipegen_export/ /Users/manubaba/Downloads/ppe-pipegen/pipegen_export/
```
Check: `bash run.sh` from the export runs standalone (no PipeGen runtime dependency). Keep the benchmark report and the screenshots with it.

---

## Teardown / housekeeping
```bash
sudo systemctl disable --now pipegen-rtsp-feed pipegen-mediamtx     # stop the fake camera
sudo systemctl disable --now mosquitto                              # stop the broker
# remove the configs: /etc/systemd/system/pipegen-*.service, /etc/mosquitto/conf.d/pipegen.conf,
#                     /etc/systemd/system/mosquitto.service.d/after-docker.conf
```
