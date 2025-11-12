"""FastAPI server exposing DeepSeek OCR over HTTP for RunPod pods."""
from __future__ import annotations

import base64
import io
import os
import time
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from PIL import Image

from infrastructure.gpu_backends import LocalGPUBackend, OCRResult

app = FastAPI(title="DeepSeek OCR RunPod Server")


class OCRLine(BaseModel):
    """Response model describing a recognised line."""

    line_id: str = Field(..., description="Stable identifier for the line")
    text: str = Field(..., description="Text recognised for this line")
    conf: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    poly: List[List[int]] = Field(
        ..., description="Quadrilateral bounding the line in clockwise order"
    )
    bbox_px: List[int] = Field(
        ..., description="Bounding box in pixels [x, y, width, height]"
    )


class OCRRequest(BaseModel):
    """Payload for the /ocr_base64 endpoint."""

    image: str = Field(..., description="Base64 encoded PNG/JPEG image")
    prompt_type: str = Field(
        "plain", description="Prompt style to pass to DeepSeek OCR"
    )
    include_geometry: bool = Field(
        True, description="Return line polygons alongside the plain text"
    )

    @field_validator("image")
    @classmethod
    def _validate_base64(cls, value: str) -> str:
        if not value:
            raise ValueError("image must be a non-empty base64 string")
        try:
            base64.b64decode(value, validate=True)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ValueError("image must be valid base64 data") from exc
        return value


class OCRResponse(BaseModel):
    """Response payload returned after DeepSeek OCR is executed."""

    success: bool
    text: str
    lines: Optional[List[OCRLine]] = None
    processing_time: float
    prompt_used: str
    image_width: int
    image_height: int
    backend: str = Field("local", description="Execution backend identifier")


@lru_cache(maxsize=1)
def get_backend() -> LocalGPUBackend:
    """Initialise the DeepSeek OCR backend once per process."""

    device = os.getenv("DEEPSEEK_DEVICE", "cuda")
    return LocalGPUBackend(device=device)


def _decode_image(image_b64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_b64)
        buffer = io.BytesIO(image_bytes)
        image = Image.open(buffer).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}")
    return image


def _normalise_bbox(bbox: Optional[List[int]], poly: Optional[List[List[int]]]) -> List[int]:
    if bbox and len(bbox) >= 4:
        return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

    if not poly:
        return [0, 0, 0, 0]

    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def _serialise_lines(result: OCRResult) -> Optional[List[OCRLine]]:
    if not result.lines:
        return None

    serialised: List[OCRLine] = []
    for index, entry in enumerate(result.lines):
        try:
            poly_raw = entry.get("poly") or entry.get("polygon")
            poly = [
                [int(point[0]), int(point[1])] for point in (poly_raw or []) if len(point) >= 2
            ]
            if len(poly) < 4:
                poly = []

            bbox_raw = entry.get("bbox_px") or entry.get("bbox")
            bbox_px = _normalise_bbox(bbox_raw, poly if poly else None)

            if not poly:
                x, y, w, h = bbox_px
                poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

            try:
                conf = float(entry.get("conf", entry.get("confidence", 0.0)))
            except (TypeError, ValueError):
                conf = 0.0

            serialised.append(
                OCRLine(
                    line_id=str(entry.get("line_id") or f"ln_{index:04d}"),
                    text=str(entry.get("text", "")),
                    conf=conf,
                    poly=poly,
                    bbox_px=bbox_px,
                )
            )
        except Exception:  # pragma: no cover - defensive
            continue
    return serialised or None


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Simple health-check endpoint used by the orchestrator."""

    backend = get_backend()
    return {"status": "ok", "device": backend.device}


@app.post("/ocr_base64", response_model=OCRResponse)
async def ocr_base64(payload: OCRRequest) -> OCRResponse:
    """Run DeepSeek OCR on a base64 encoded image."""

    image = _decode_image(payload.image)

    backend = get_backend()
    start = time.time()

    result = await run_in_threadpool(
        backend.run_deepseek_ocr,
        image,
        prompt_type=payload.prompt_type,
    )

    processing_time = result.processing_time or (time.time() - start)

    lines = _serialise_lines(result) if payload.include_geometry else None
    width, height = image.size

    return OCRResponse(
        success=True,
        text=result.text or "",
        lines=lines,
        processing_time=processing_time,
        prompt_used=payload.prompt_type,
        image_width=width,
        image_height=height,
        backend=result.backend or "local",
    )


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
