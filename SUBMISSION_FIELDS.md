# Workplace Safety Sentinel: hackathon submission fields

**In one line:** Watches construction-site camera footage on a laptop and flags workers who are not wearing a hard hat, with a live incident feed and evidence photos.

**Who it is for:** A site safety officer who cannot watch every camera all day. Sentinel runs on the edge, on existing cameras, and sends only small event messages (no video) to a live dashboard. Each violation appears with a timestamp and photo. It is a prompt for a human check, not automatic enforcement.

**Area:** Factories and worker safety

**Outputs:** live browser dashboard, annotated MP4, evidence photos per incident, JSON event log, dashboard incident feed.
Not built: tiled display, RTSP stream. MQTT code path exists but is not verified end to end.

**Output destination:** The dashboard, watched by the site safety officer. No automatic alerts to supervisors or phones.

**Inputs tested:** 32 licensed construction-site clips (Wikimedia Commons, Mixkit). Browser camera and RTSP were not tested.

**Pipeline, in order:**
1. Read frames from the clip or browser camera.
2. Detect people and PPE with a YOLOv8n model fine-tuned on construction PPE (11 classes, ONNX).
3. Track each person across frames so one person keeps one ID.
4. Hard-hat rule: a tracked person with a `no_helmet` detection overlapping the head, held for a few frames, raises an alert. A cooldown prevents repeats.
5. Draw boxes and a NO HELMET banner, save an evidence photo, write a JSON event, push it to the dashboard.

PipeGen was not run. This is our own pipeline, not a PipeGen-generated one.

**Speed (real footage):**
- RTX A6000, 500 frames: PyTorch FP32 163 FPS (6.1 ms); ONNX Runtime CUDA 319 FPS (3.1 ms); TensorRT FP16 320 FPS (3.1 ms); TensorRT INT8 335 FPS (3.0 ms).
- Apple M4 Mac mini (CoreML), end-to-end with tracking and rules: about 36 to 45 FPS.
- Multi-camera throughput: not measured.

**Limits:**
- `no_helmet` precision 0.43, recall 0.22 on the validation split. Misses most bare heads.
- False alarms in the per-clip review (for example an excavator bucket tracked as a person).
- Tracker IDs can split one person, so unique-violator counts can overcount.
- No zone rules and no RTSP output.
- DeepStream path: pipeline runs on an A10, but detections are not yet verified.
- PipeGen was not run.
