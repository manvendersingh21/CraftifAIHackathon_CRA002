# Custom YOLOv8 parser for nvinfer

Ultralytics YOLOv8 ONNX output is `[1, 4+nc, 8400]` (cx,cy,w,h + per-class scores, no objectness, no NMS).
DeepStream's built-in parsers don't understand this, so use one of:

1. **DeepStream-Yolo (recommended)** – https://github.com/marcoslucianops/DeepStream-Yolo
   ```
   git clone https://github.com/marcoslucianops/DeepStream-Yolo /opt/DeepStream-Yolo
   cd /opt/DeepStream-Yolo
   CUDA_VER=12.x make -C nvdsinfer_custom_impl_Yolo   # set to the container's CUDA (nvcc --version)
   ```
   Best results: re-export with their `utils/export_yoloV8.py` (adds a transposed/boxes+scores output that
   `NvDsInferParseYolo` expects). Their exporter: `python3 export_yoloV8.py -w models/ppe_yolov8n.pt --opset 12 -s 640`.
   The plain ultralytics ONNX from train.py may need that re-export for the parser to decode correctly – verify on GPU.
2. **Write your own** `NvDsInferParseCustomYoloV8`: transpose output to 8400x15, take argmax score per anchor,
   threshold, convert cx,cy,w,h → left,top,w,h in network coords, fill `NvDsInferObjectDetectionInfo`;
   let nvinfer cluster (`cluster-mode=2` NMS). Export with `extern "C"` and `CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE`.

## "NO HELMET" alert in DeepStream
Class 7 (`no_helmet`) – add a pad probe on the OSD sink pad (Python bindings `pyds`) that iterates
`NvDsObjectMeta`, sets `rect_params.border_color` red for class_id 7, and adds an `NvDsDisplayMeta`
text "NO HELMET" + running counter (count unique tracker `object_id`s to avoid per-frame double counting).
For PipeGen: describe the same pipeline: `filesrc/v4l2src → nvstreammux → nvinfer(this config) → nvtracker → nvdsosd → encoder/sink`.
