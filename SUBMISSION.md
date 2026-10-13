# Workplace Safety Sentinel — CraftifAI EMBEDDED SF · Track A: PipeGen

> DRAFT. Every bracketed value (e.g. `[FPS_FP16]`) is a placeholder to be filled with numbers **measured on the
> demo hardware** (`benchmark.py`, DeepStream perf logs, `trtexec`). Nothing here is a measured result yet.

## Problem
On construction sites and in warehouses, missing hard hats and hi-vis vests are a leading cause of serious injury.
Safety officers can't watch every camera all the time, and cloud video analytics brings latency, bandwidth cost and
privacy problems. Workplace Safety Sentinel runs PPE compliance detection **on the edge**, in real time, on existing
cameras, and sends only small metadata events (no video) to a live dashboard.

## Pipeline (generated with PipeGen; manual equivalents in `deepstream/`)
| Stage | Choice |
|---|---|
| **Inputs** | USB camera (`/dev/video0`, v4l2) or video file / RTSP (`samples/sample.mp4`) |
| **Tasks** | 1) object detection (people + PPE items) · 2) multi-object tracking (nvtracker, persistent tracker IDs) · 3) PPE rule: a tracked person with `no_helmet` / `no_vest` (or missing helmet/vest overlap) → violation event, de-duplicated per tracker ID |
| **Model** | YOLOv8n fine-tuned on Ultralytics *construction-ppe* (11 classes: helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots) → ONNX (opset 12, 1x3x640x640) |
| **Inference** | TensorRT via DeepStream `nvinfer`: **FP16** engine and **INT8** engine (calibrated on [N_CALIB_IMAGES] dataset images) |
| **Outputs** | MQTT (`nvmsgconv` + `nvmsgbroker` → topic `[MQTT_TOPIC]`) · JSON log file · on-screen display (nvdsosd: boxes, red NO HELMET banner, counters) · annotated MP4 recording |

```
v4l2src / filesrc → nvstreammux → nvinfer (YOLOv8n PPE, TRT FP16|INT8) → nvtracker → PPE rule probe
   → nvdsosd → display / nvv4l2h264enc → mp4
   → nvmsgconv → nvmsgbroker (MQTT) → dashboard/ (Flask + SSE live incident cards)
```

## Dashboard (`dashboard/`)
Flask + paho-mqtt + Server-Sent Events. Subscribes to a configurable MQTT topic (default `#`, `localhost:1883`)
and/or tails a JSON log file as a fallback. Parses DeepStream / PipeGen metadata generically (nvmsgconv full and
minimal schemas, custom probe JSON, unknown layouts shown raw). Shows live incident cards, total incidents, unique
violators by tracker ID, and beeps + flashes the screen on each new violation.

## Results (fill in from measurements)
| Metric | Value |
|---|---|
| mAP50 (val) / mAP50-95 | [MAP50] / [MAP50_95] |
| no_helmet precision / recall | [P_NO_HELMET] / [R_NO_HELMET] |
| Hardware | [HARDWARE e.g. Jetson model / GPU] , JetPack/DeepStream [DS_VERSION] |

## Edge Optimization (bonus)
| Variant | FPS | Latency (ms) | Engine size (MB) | mAP50 | Power (W) |
|---|---|---|---|---|---|
| PyTorch FP32 (baseline) | [FPS_PT] | [LAT_PT] | [SIZE_PT] | [MAP_PT] | [PWR_PT] |
| ONNX Runtime | [FPS_ORT] | [LAT_ORT] | [SIZE_ONNX] | [MAP_ORT] | [PWR_ORT] |
| TensorRT FP16 | [FPS_FP16] | [LAT_FP16] | [SIZE_FP16] | [MAP_FP16] | [PWR_FP16] |
| TensorRT INT8 | [FPS_INT8] | [LAT_INT8] | [SIZE_INT8] | [MAP_INT8] | [PWR_INT8] |

- FP16 → INT8 speedup: [SPEEDUP_INT8]x for an accuracy change of [MAP_DELTA] mAP50.
- Streams sustained in real time at 30 FPS: [N_STREAMS].
- Other optimisations: nano model (3M params), static 640 input, tracker reuses detections so the rule runs per
  track (not per frame), and only metadata leaves the device (≈[BYTES_PER_EVENT] B per event instead of video).

## Demo (≈2 min)
1. Problem framing (15 s).  2. Live camera / sample video with OSD: green PPE, red NO HELMET.  3. Dashboard: incident
cards appear with beep/flash, unique-violator counter increments by tracker ID.  4. PipeGen pipeline graph.
5. FP16 vs INT8 table.  6. Next steps.

## Limitations & next steps
Small dataset ([N_TRAIN] train images); `no_vest` is not a class in construction-ppe (vest rule = Person without
overlapping vest box); YOLOv8 in DeepStream needs the DeepStream-Yolo custom parser (`deepstream/PARSER_NOTES.md`).
Next: zone rules (hard-hat areas only), Slack/SMS alerts, face-blurred evidence clips, multi-camera Jetson deployment.

## Team / links
[TEAM_NAMES] · repo: [REPO_URL] · video: [DEMO_VIDEO_URL]
