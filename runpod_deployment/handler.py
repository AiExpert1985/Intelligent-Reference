"""
RunPod serverless handler for PaddleOCR.
This handler receives base64-encoded images and returns OCR results.
"""
import runpod
import base64
import io
import time
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional


def load_paddleocr_model():
    """Load PaddleOCR model on pod startup."""
    try:
        from paddleocr import PaddleOCR

        # Initialize PaddleOCR
        # use_angle_cls=True enables text orientation detection
        # lang='ar' for Arabic (also supports English)
        # GPU will be auto-detected if available
        model = PaddleOCR(
            use_angle_cls=True,
            lang='ar'
        )
        print("✅ PaddleOCR loaded successfully on GPU")
        return model
    except Exception as e:
        print(f"❌ Failed to load PaddleOCR: {e}")
        raise


# Load model once at startup (not per request)
ocr_model = load_paddleocr_model()


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function.

    Expected input format:
    {
        "input": {
            "operation": "ocr",
            "processor": "paddleocr",
            "image": "<base64-encoded-image>"
        }
    }

    Returns:
    {
        "text": "extracted text",
        "lines": [...],  # Line geometry data
        "confidence": 0.95,
        "processing_time": 1.23
    }
    """
    try:
        start_time = time.time()

        # Extract input data
        job_input = event.get("input", {})
        operation = job_input.get("operation", "ocr")
        processor = job_input.get("processor", "paddleocr")
        image_b64 = job_input.get("image")

        if not image_b64:
            return {"error": "No image provided"}

        if operation != "ocr":
            return {"error": f"Unsupported operation: {operation}"}

        # Accept both paddleocr and deepseek for compatibility
        if processor not in ["paddleocr", "deepseek"]:
            return {"error": f"Unsupported processor: {processor}"}

        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to decode image: {str(e)}"}

        # Run OCR
        try:
            # Convert PIL image to numpy array (PaddleOCR format)
            img_array = np.array(image)

            # Run PaddleOCR
            result = ocr_model.ocr(img_array, cls=True)

            # Parse result
            text, lines = parse_paddleocr_result(result, image.size)

            processing_time = time.time() - start_time

            return {
                "text": text,
                "lines": lines,
                "confidence": calculate_confidence(lines),
                "processing_time": processing_time
            }

        except Exception as e:
            return {"error": f"OCR processing failed: {str(e)}"}

    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}


def parse_paddleocr_result(raw_result: Any, image_size: tuple) -> tuple:
    """
    Parse PaddleOCR output into text and line geometry.

    PaddleOCR returns: [[page_results]] where each page contains:
    [
        [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)],
        ...
    ]
    """
    width, height = image_size

    # Handle empty results
    if not raw_result or not raw_result[0]:
        return "", []

    # Extract first page results
    page_result = raw_result[0]

    if not page_result:
        return "", []

    # Parse lines
    parsed_lines = []
    text_parts = []

    for line_data in page_result:
        if not line_data or len(line_data) < 2:
            continue

        bbox_points = line_data[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text_info = line_data[1]    # (text, confidence)

        # Extract text and confidence
        if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
            text = str(text_info[0]).strip()
            confidence = float(text_info[1])
        else:
            text = str(text_info).strip()
            confidence = 0.0

        if not text:
            continue

        # Normalize polygon coordinates
        polygon = normalize_polygon(bbox_points, width, height)
        bbox_px = poly_to_bbox(polygon)

        line = {
            "line_id": f"ln_{time.time_ns():x}",
            "poly": polygon,
            "bbox_px": bbox_px,
            "text": text,
            "conf": confidence,
        }

        parsed_lines.append(line)
        text_parts.append(text)

    # Join text with newlines
    full_text = "\n".join(text_parts)

    return full_text, parsed_lines


def normalize_polygon(polygon: List, width: int, height: int) -> List[List[int]]:
    """Normalize polygon coordinates to be within image bounds."""
    if not polygon or len(polygon) < 4:
        return [[0, 0], [width, 0], [width, height], [0, height]]

    normalized = []
    for point in polygon[:4]:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            x = int(max(0, min(width, float(point[0]))))
            y = int(max(0, min(height, float(point[1]))))
            normalized.append([x, y])
        else:
            normalized.append([0, 0])

    # Ensure we have exactly 4 points
    while len(normalized) < 4:
        normalized.append([0, 0])

    return normalized[:4]


def poly_to_bbox(polygon: List[List[int]]) -> tuple:
    """Convert polygon to bounding box (x, y, width, height)."""
    if not polygon or len(polygon) < 4:
        return (0, 0, 0, 0)

    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    return (x1, y1, x2 - x1, y2 - y1)


def calculate_confidence(lines: List[Dict]) -> float:
    """Calculate average confidence from lines."""
    if not lines:
        return 0.0

    confidences = [line.get("conf", 0.0) for line in lines if line.get("conf") is not None]
    return sum(confidences) / len(confidences) if confidences else 0.0


if __name__ == "__main__":
    # Start the RunPod serverless worker
    print("🚀 Starting RunPod serverless worker with PaddleOCR...")
    runpod.serverless.start({"handler": handler})
