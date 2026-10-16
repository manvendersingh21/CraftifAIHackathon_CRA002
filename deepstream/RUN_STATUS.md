# DeepStream run status (Modal A10, real clip wm_excavator_operator.mp4)

Measured:
- Container: nvcr.io/nvidia/deepstream:7.1-triton-multiarch on Modal A10G
- pyds 1.2.0 installed from the NVIDIA release wheel; GStreamer pipeline starts
- DeepStream-Yolo parser built from source with CUDA 12.6
- TensorRT engine built from models/ppe_yolov8n.onnx (about 4 minutes)
- Output: 378 frames decoded, 1280x720 H.264, saved as ds_output.mp4 (6.7 MB)
- Output frame 100 contains about 1,448 saturated pixels in nvdsosd box colours, so boxes are drawn

Not verified:
- Diagnostic (frames 0-2): the probe receives 0 objects. The probe is not the cause.
- The nvinfer engine loads ('Load new model ... success') with no parser error, but NvDsInferParseYolo
  emits no objects. Most likely cause: the raw Ultralytics ONNX output [1,15,8400] does not match the
  layout DeepStream-Yolo's NvDsInferParseYolo expects. The fix is to re-export with DeepStream-Yolo's
  exporter (deepstream/PARSER_NOTES.md), rebuild the engine, and rerun. NOT yet done.
- Earlier claim of drawn boxes was wrong: colour-matched pixels in the frame were not verified as nvdsosd boxes.
- No NO HELMET results from DeepStream. The ONNX Runtime path produced the reported results.

PipeGen: NOT run. Its launch check fails on CIS-A6000 (no docker group, no X session,
driver 550.90.07 is older than 560.35.03, no nvidia-container-toolkit, no socat).
