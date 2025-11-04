"""
RunPod serverless handler for DeepSeek OCR.
This handler receives base64-encoded images and returns OCR results.
"""
import runpod
import base64
import io
import time
from PIL import Image
from typing import Any, Dict


def load_deepseek_model():
    """Load DeepSeek OCR model on pod startup."""
    try:
        from deepseek_ocr import DeepSeekOCR

        # Initialize on GPU if available
        model = DeepSeekOCR(device="cuda")
        print("✅ DeepSeek OCR loaded successfully on GPU")
        return model
    except Exception as e:
        print(f"❌ Failed to load DeepSeek OCR: {e}")
        raise


# Load model once at startup (not per request)
deepseek_model = load_deepseek_model()


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function.

    Expected input format:
    {
        "input": {
            "operation": "ocr",
            "processor": "deepseek",
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
        processor = job_input.get("processor", "deepseek")
        image_b64 = job_input.get("image")

        if not image_b64:
            return {"error": "No image provided"}

        if operation != "ocr":
            return {"error": f"Unsupported operation: {operation}"}

        if processor != "deepseek":
            return {"error": f"Unsupported processor: {processor}"}

        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to decode image: {str(e)}"}

        # Run OCR
        try:
            # Try recognize method first, fallback to process_image
            if hasattr(deepseek_model, "recognize"):
                result = deepseek_model.recognize(image)
            else:
                result = deepseek_model.process_image(image)

            # Parse result
            text, lines = parse_deepseek_result(result, image.size)

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


def parse_deepseek_result(raw_result: Any, image_size: tuple) -> tuple:
    """Parse DeepSeek OCR output into text and line geometry."""
    width, height = image_size

    # Extract items from result
    items = extract_items(raw_result)

    if not items:
        # Fallback: try to extract text directly
        text = extract_text_fallback(raw_result)
        return text, []

    # Parse lines
    parsed_lines = []
    text_parts = []

    for entry in items:
        line = parse_line(entry, width, height)
        if line:
            parsed_lines.append(line)
            text_parts.append(line["text"])

    return "\n".join(text_parts), parsed_lines


def extract_items(raw_result: Any) -> list:
    """Extract line-like items from DeepSeek result."""
    if isinstance(raw_result, dict):
        for key in ("lines", "items", "data", "result", "predictions"):
            candidate = raw_result.get(key)
            if isinstance(candidate, list):
                return candidate

    if isinstance(raw_result, list):
        return raw_result

    if hasattr(raw_result, "lines") and isinstance(raw_result.lines, list):
        return raw_result.lines

    return []


def extract_text_fallback(raw_result: Any) -> str:
    """Extract text when no structured lines are present."""
    if isinstance(raw_result, dict):
        return str(raw_result.get("text", "")).strip()

    if hasattr(raw_result, "text"):
        return str(getattr(raw_result, "text", "")).strip()

    return ""


def parse_line(entry: Any, width: int, height: int) -> dict:
    """Parse an individual line entry."""
    # Extract text
    text = extract_line_text(entry)
    if not text:
        return None

    # Extract polygon
    polygon = extract_polygon(entry, width, height)

    # Extract confidence
    confidence = extract_confidence(entry)

    return {
        "line_id": f"ln_{time.time_ns():x}",
        "poly": polygon,
        "bbox_px": poly_to_bbox(polygon),
        "text": text,
        "conf": confidence,
    }


def extract_line_text(entry: Any) -> str:
    """Extract text from a line entry."""
    if isinstance(entry, dict):
        for key in ("text", "sentence", "value", "content"):
            value = entry.get(key)
            if value:
                return str(value).strip()

    if isinstance(entry, (list, tuple)) and entry:
        return str(entry[0]).strip()

    return ""


def extract_polygon(entry: Any, width: int, height: int) -> list:
    """Extract polygon from entry."""
    polygon = None

    if isinstance(entry, dict):
        polygon = entry.get("polygon") or entry.get("poly")
        if polygon is None:
            bbox = entry.get("bbox") or entry.get("box")
            polygon = bbox_to_poly(bbox, width, height)
    elif isinstance(entry, (list, tuple)) and len(entry) > 1:
        polygon = bbox_to_poly(entry[1], width, height)

    return normalise_polygon(polygon, width, height)


def extract_confidence(entry: Any) -> float:
    """Extract confidence score from entry."""
    if isinstance(entry, dict):
        for key in ("confidence", "score", "probability", "conf"):
            value = entry.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0

    if isinstance(entry, (list, tuple)) and len(entry) > 2:
        try:
            return float(entry[2])
        except (TypeError, ValueError):
            return 0.0

    return 0.0


def bbox_to_poly(bbox: Any, width: int, height: int) -> list:
    """Convert bounding box to polygon."""
    if not bbox:
        return None

    if isinstance(bbox, dict):
        x = bbox.get("x") or bbox.get("left") or 0
        y = bbox.get("y") or bbox.get("top") or 0
        w = bbox.get("w") or bbox.get("width") or width
        h = bbox.get("h") or bbox.get("height") or height
        x, y, w, h = float(x), float(y), float(w), float(h)
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x, y, w, h = map(float, bbox[:4])
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    return None


def normalise_polygon(polygon: list, width: int, height: int) -> list:
    """Normalise polygon coordinates."""
    if not polygon or len(polygon) < 4:
        return [[0, 0], [width, 0], [width, height], [0, height]]

    normalised = []
    for x, y in polygon[:4]:
        norm_x = int(max(0, min(width, x)))
        norm_y = int(max(0, min(height, y)))
        normalised.append([norm_x, norm_y])

    return normalised


def poly_to_bbox(polygon: list) -> tuple:
    """Convert polygon to bounding box."""
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return (x1, y1, x2 - x1, y2 - y1)


def calculate_confidence(lines: list) -> float:
    """Calculate average confidence from lines."""
    if not lines:
        return 0.0

    confidences = [line.get("conf", 0.0) for line in lines]
    return sum(confidences) / len(confidences) if confidences else 0.0


if __name__ == "__main__":
    # Start the RunPod serverless worker
    runpod.serverless.start({"handler": handler})
