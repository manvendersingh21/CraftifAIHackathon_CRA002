# Workplace Safety Sentinel — CraftifAI EMBEDDED SF · Track A: PipeGen

> DRAFT. Every bracketed value (e.g. `320.3`) is a placeholder to be filled with numbers **measured on the
> demo hardware** (`benchmark.py`, DeepStream perf logs, `trtexec`). Nothing here is a measured result yet.

## Problem
On construction sites and in warehouses, missing hard hats and hi-vis vests are a leading cause of serious injury.
Safety officers can't watch every camera all the time, and cloud video analytics brings latency, bandwidth cost and
privacy problems. Workplace Safety Sentinel runs PPE compliance detection **on the edge**, in real time, on existing
cameras, and sends only small metadata events (no video) to a live dashboard.

## Pipeline (DeepStream configs in `deepstream/`; PipeGen generation was attempted but not run, see note)
| Stage | Choice |
|---|---|
| **Inputs** | USB camera (`/dev/video0`, v4l2) or video file / RTSP (`samples/sample.mp4`) |
| **Tasks** | 1) object detection (people + PPE items) · 2) multi-object tracking (nvtracker, persistent tracker IDs) · 3) PPE rule: a tracked person with `no_helmet` / `no_vest` (or missing helmet/vest overlap) → violation event, de-duplicated per tracker ID |
| **Model** | YOLOv8n fine-tuned on Ultralytics *construction-ppe* (11 classes: helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots) → ONNX (opset 12, 1x3x640x640) |
| **Inference** | TensorRT via DeepStream `nvinfer`: **FP16** engine and **INT8** engine (calibrated on 1132 dataset images) |
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
| mAP50 (val) / mAP50-95 | 0.571 / 0.278 |
| no_helmet precision / recall | 0.426 / 0.222 |
| Hardware | NVIDIA RTX A6000, driver 550.90.07, TensorRT 10.3.0 (edge benchmark); Apple M4 Mac mini (CoreML, live demo) · DeepStream [DS_VERSION] |

## Edge Optimization (bonus)
| Variant | FPS | Latency (ms) | Engine size (MB) | mAP50 | Power (W) |
|---|---|---|---|---|---|
| PyTorch FP32 (baseline) | 162.8 | 6.14 | 6.5 | 0.571 | 125.2 |
| ONNX Runtime | 319.3 | 3.13 | 12.8 | 0.580 | 145.8 |
| TensorRT FP16 | 320.3 | 3.12 | 9.2 | 0.579 | 99.8 |
| TensorRT INT8 | 335.2 | 2.98 | 6.2 | 0.584 | 96.5 |

- FP16 → INT8 speedup: 1.05x for an accuracy change of +0.006 mAP50.
- Streams sustained in real time at 30 FPS: [N_STREAMS].
- Other optimisations: nano model (3M params), static 640 input, tracker reuses detections so the rule runs per
  track (not per frame), and only metadata leaves the device (≈[BYTES_PER_EVENT] B per event instead of video).

## Demo (≈2 min)
1. Problem framing (15 s).  2. Live camera / sample video with OSD: green PPE, red NO HELMET.  3. Dashboard: incident
cards appear with beep/flash, unique-violator counter increments by tracker ID.  4. DeepStream pipeline graph from `deepstream/` (unverified on a GPU, see limitations).
5. FP16 vs INT8 table.  6. Next steps.

## Limitations & next steps
PipeGen: the PipeGen-generated pipeline was **not** run. Its VM host was unreachable during the build, so the DeepStream path is the manual config in `deepstream/`, which has not been executed on a GPU yet.
Small dataset ([N_TRAIN] train images); `no_vest` is not a class in construction-ppe (vest rule = Person without
overlapping vest box); YOLOv8 in DeepStream needs the DeepStream-Yolo custom parser (`deepstream/PARSER_NOTES.md`).
Next: zone rules (hard-hat areas only), Slack/SMS alerts, face-blurred evidence clips, multi-camera Jetson deployment.

## Team / links
[TEAM_NAMES] · repo: [REPO_URL] · video: [DEMO_VIDEO_URL]
