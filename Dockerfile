# DeepStream x86 dGPU image. Tag 7.1-triton-multiarch is listed in NVIDIA NGC (checked via web search, not pulled).
# Override: docker build --build-arg DS_TAG=7.1-samples-multiarch ...   (Jetson: use nvcr.io/nvidia/deepstream-l4t:<tag>)
# Host needs an NVIDIA driver new enough for the tag (see DeepStream 7.1 release notes) + nvidia-container-toolkit.
ARG DS_TAG=7.1-triton-multiarch
FROM nvcr.io/nvidia/deepstream:${DS_TAG}
ENV DEBIAN_FRONTEND=noninteractive
# build-essential for the custom parser; gi/gstreamer python for ppe_deepstream.py; ffmpeg for sample handling
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl unzip ffmpeg build-essential python3-pip python3-gi python3-gst-1.0 gir1.2-gst-rtsp-server-1.0 \
      libgirepository1.0-dev libcairo2-dev && rm -rf /var/lib/apt/lists/*
# DeepStream-Yolo custom parser; CUDA_VER detected from nvcc (falls back to /usr/local/cuda version file).
RUN git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo /opt/DeepStream-Yolo && \
    CUDA_VER=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -n1) && \
    [ -n "$CUDA_VER" ] && echo "CUDA_VER=$CUDA_VER" && \
    CUDA_VER=$CUDA_VER make -C /opt/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
# pyds (DeepStream Python bindings) are NOT in pip. Install the wheel from NVIDIA-AI-IOT/deepstream_python_apps
# matching DS version + python (7.1 -> v1.2.0, cp310 x86_64). Build does not fail if it 404s; check at runtime.
ARG PYDS_WHL=https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_x86_64.whl
RUN (python3 -c "import pyds" 2>/dev/null && echo "pyds already present") || \
    (pip3 install --no-cache-dir "$PYDS_WHL" || echo "WARNING: pyds install failed; deepstream-app mode still works, ppe_deepstream.py will not")
WORKDIR /app
COPY requirements.txt .
# onnxruntime-gpu/ultralytics are only needed for the fallback/training; the DeepStream path needs none of it.
RUN pip3 install --no-cache-dir -r requirements.txt || echo "WARNING: requirements install failed (fallback tools unavailable)"
COPY . .
# Way 1 (default): deepstream-app with deepstream/deepstream_app_config.txt  -> /app/ds_output.mp4
#   docker run --gpus all --rm -v $PWD:/app ppe-ds
# Way 2: Python pad-probe app (tracker-ID rule, red NO HELMET banner, JSONL/MQTT events):
#   docker run --gpus all --rm -v $PWD:/app ppe-ds python3 deepstream/ppe_deepstream.py \
#       --source samples/sample_h264.mp4 --out ds_output.mp4 --events ds_events.jsonl
# Mount the repo (-v) so models/*.onnx, engines and outputs persist on the host.
CMD ["bash", "-c", "cd deepstream && deepstream-app -c deepstream_app_config.txt"]
