# RunPod DeepSeek OCR Server

This package contains a lightweight FastAPI service that runs the DeepSeek OCR
model on a GPU-enabled RunPod instance. The main application can POST images to
this server and receive recognised text together with per-line geometry used for
highlight rendering.

## Features

- `/health` – quick readiness probe
- `/ocr_base64` – accepts base64 encoded images and returns text + line metadata
- Lazy initialisation of `DeepSeekOCR` so the model loads only once per process
- Optional `DEEPSEEK_DEVICE` environment variable to override the GPU device

## Getting started on RunPod

```bash
# 1) Start from the repository root on the RunPod instance
cd /workspace/Intelligent-Reference/document_rag_back

# 2) Install the runtime dependencies for the OCR server
pip install -r runpod_server/requirements.txt
#    (install the correct torch build separately, e.g.
#     pip install torch==2.2.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121)

# 3) Launch the API (adjust --host/--port as required by your pod)
uvicorn runpod_server.app:app --host 0.0.0.0 --port 8000
```

The main backend should be configured with `USE_REMOTE_GPU=True` and
`RUNPOD_ENDPOINT` pointing to the public URL of the pod (e.g. the HTTPS proxy
address). Requests issued by the backend already include `include_geometry=True`
so the OCR response contains all the metadata needed for downstream processing.

## Configuration options

- `DEEPSEEK_DEVICE`: Defaults to `cuda`. Set to `cuda:1`, `cpu`, etc. if the pod
  exposes alternative devices.
- Request payload supports `prompt_type` (defaults to `plain`) and
  `include_geometry` (defaults to `true`). The backend currently always requests
  geometry.

## Response schema

```json
{
  "success": true,
  "text": "...",
  "lines": [
    {
      "line_id": "ln_ab12cd34",
      "text": "Line contents",
      "conf": 0.98,
      "poly": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "bbox_px": [x, y, width, height]
    }
  ],
  "processing_time": 1.23,
  "prompt_used": "markdown",
  "image_width": 2480,
  "image_height": 3508,
  "backend": "local"
}
```

The FastAPI service emits JSON that mirrors the structure expected by the
backend's `GPUBackendManager`, so no further post-processing is required.
