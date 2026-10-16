# DeepStream run status (Modal A10, real clip wm_excavator_operator.mp4)

Measured:
- Container: nvcr.io/nvidia/deepstream:7.1-triton-multiarch on Modal A10G
- pyds 1.2.0 installed from the NVIDIA release wheel; GStreamer pipeline starts
- DeepStream-Yolo parser built from source with CUDA 12.6
- TensorRT engine built from models/ppe_yolov8n.onnx (about 4 minutes)
- Output: 378 frames decoded, 1280x720 H.264, saved as ds_output.mp4 (6.7 MB)
- Output frame 100 contains about 1,448 saturated pixels in nvdsosd box colours, so boxes are drawn

Not verified:
- The probe reports persons=0, violators=0, alerts=0 and an empty ds_events.jsonl.
  Likely causes, not yet tested: class ids from the parser do not match models/labels.txt,
  or the object tracker gives UNTRACKED ids so the probe skips every person.
- No NO HELMET results from DeepStream. The ONNX Runtime path produced the reported results.

PipeGen: NOT run. Its launch check fails on CIS-A6000 (no docker group, no X session,
driver 550.90.07 is older than 560.35.03, no nvidia-container-toolkit, no socat).
