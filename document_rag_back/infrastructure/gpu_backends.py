"""GPU backend abstraction for local and remote execution."""
from __future__ import annotations

import base64
import io
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from PIL import Image

from config import settings

logger = logging.getLogger(settings.LOGGER_NAME)


class OCRProcessor(str, Enum):
    """Supported OCR processors for the GPU backend layer."""

    DEEPSEEK = "deepseek"
    PADDLEOCR = "paddleocr"
    TESSERACT = "tesseract"


@dataclass
class GPUBackendConfig:
    """Configuration for selecting a GPU execution backend."""

    use_remote: bool = False
    runpod_api_key: Optional[str] = None
    runpod_endpoint: Optional[str] = None
    runpod_timeout: int = 300
    local_device: str = "cuda"

    @classmethod
    def from_settings(cls, settings: Any) -> "GPUBackendConfig":
        """Build configuration from the application settings object."""

        return cls(
            use_remote=bool(getattr(settings, "USE_REMOTE_GPU", False)),
            runpod_api_key=getattr(settings, "RUNPOD_API_KEY", None),
            runpod_endpoint=getattr(settings, "RUNPOD_ENDPOINT", None),
            runpod_timeout=int(getattr(settings, "RUNPOD_TIMEOUT", 300)),
            local_device=getattr(settings, "LOCAL_GPU_DEVICE", "cuda"),
        )


@dataclass
class OCRResult:
    """Result returned by GPU backends."""

    text: str
    lines: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    processing_time: float = 0.0
    source_file: Optional[str] = None
    processor: Optional[str] = None
    backend: Optional[str] = None


class GPUBackend(ABC):
    """Abstract base class for GPU execution backends."""

    @abstractmethod
    def run_deepseek_ocr(
        self,
        image: Union[str, Path, Image.Image],
        **kwargs: Any,
    ) -> OCRResult:
        """Run DeepSeek OCR inference."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when the backend can accept work."""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return backend metadata for debugging and telemetry."""


class LocalGPUBackend(GPUBackend):
    """Run workloads directly on the local machine."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._deepseek_engine: Optional[Any] = None

    def _ensure_image(
        self, image: Union[str, Path, Image.Image]
    ) -> tuple[Image.Image, Optional[str]]:
        """Coerce supported inputs to a PIL image."""

        if isinstance(image, Image.Image):
            return image.convert("RGB"), None
        source = str(image)
        pil_image = Image.open(image).convert("RGB")
        return pil_image, source

    def _get_deepseek_engine(self) -> Any:
        """Lazy-load and cache the DeepSeek OCR engine."""

        if self._deepseek_engine is None:
            try:
                from deepseek_ocr import DeepSeekOCR  # type: ignore

                self._deepseek_engine = DeepSeekOCR(device=self.device)
                logger.info("DeepSeek OCR loaded on %s", self.device)
            except ImportError as exc:  # pragma: no cover - import guard
                raise RuntimeError(
                    "DeepSeek OCR not available. Install with: pip install deepseek-ocr"
                ) from exc
            except Exception as exc:  # pragma: no cover - defensive guard
                raise RuntimeError(f"Failed to initialise DeepSeek OCR: {exc}") from exc
        return self._deepseek_engine

    def run_deepseek_ocr(
        self,
        image: Union[str, Path, Image.Image],
        **kwargs: Any,
    ) -> OCRResult:
        """Run DeepSeek OCR locally."""

        pil_image, source = self._ensure_image(image)
        start = time.time()

        engine = self._get_deepseek_engine()
        try:
            if hasattr(engine, "recognize"):
                raw_result = engine.recognize(pil_image, **kwargs)
            else:
                raw_result = engine.process_image(pil_image, **kwargs)
            text, lines = self._parse_deepseek_result(raw_result, pil_image.size)
        except Exception as exc:
            logger.error("DeepSeek OCR failed: %s", exc)
            raise

        return OCRResult(
            text=text,
            lines=lines,
            processing_time=time.time() - start,
            source_file=source,
            processor=OCRProcessor.DEEPSEEK.value,
            backend="local",
        )

    def _parse_deepseek_result(
        self, raw_result: Any, image_size: tuple[int, int]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Parse DeepSeek OCR output into text and line geometry."""

        width, height = image_size
        items = self._extract_items(raw_result)
        if not items:
            text = self._extract_text_fallback(raw_result)
            return text, []

        parsed_lines: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        for entry in items:
            line = self._parse_line(entry, width, height)
            if line is None:
                continue
            parsed_lines.append(line)
            text_parts.append(line["text"])

        return "\n".join(text_parts), parsed_lines

    def _extract_items(self, raw_result: Any) -> List[Any]:
        """Extract line-like items from the DeepSeek result."""

        if isinstance(raw_result, dict):
            for key in ("lines", "items", "data", "result", "predictions"):
                candidate = raw_result.get(key)
                if isinstance(candidate, list):
                    return candidate
        if isinstance(raw_result, list):
            return raw_result
        lines = getattr(raw_result, "lines", None)
        if isinstance(lines, list):
            return lines
        return []

    def _extract_text_fallback(self, raw_result: Any) -> str:
        """Extract text when no structured lines are present."""

        if isinstance(raw_result, dict):
            return str(raw_result.get("text", "")).strip()
        if hasattr(raw_result, "text"):
            return str(getattr(raw_result, "text", "")).strip()
        return ""

    def _parse_line(
        self, entry: Any, width: int, height: int
    ) -> Optional[Dict[str, Any]]:
        """Parse an individual line entry."""

        text = self._extract_line_text(entry)
        if not text:
            return None

        polygon = self._extract_polygon(entry, width, height)
        confidence = self._extract_confidence(entry)

        return {
            "line_id": f"ln_{time.time_ns():x}",
            "poly": polygon,
            "bbox_px": self._poly_to_bbox(polygon),
            "text": text,
            "conf": confidence,
        }

    def _extract_line_text(self, entry: Any) -> str:
        if isinstance(entry, dict):
            for key in ("text", "sentence", "value", "content"):
                value = entry.get(key)
                if value:
                    return str(value).strip()
        if isinstance(entry, (list, tuple)) and entry:
            return str(entry[0]).strip()
        return ""

    def _extract_polygon(
        self, entry: Any, width: int, height: int
    ) -> List[List[int]]:
        polygon: Optional[List[List[float]]] = None

        if isinstance(entry, dict):
            polygon = entry.get("polygon") or entry.get("poly")
            if polygon is None:
                bbox = entry.get("bbox") or entry.get("box")
                polygon = self._bbox_to_poly(bbox)
        elif isinstance(entry, (list, tuple)) and len(entry) > 1:
            polygon = self._bbox_to_poly(entry[1])

        return self._normalise_polygon(polygon, width, height)

    def _extract_confidence(self, entry: Any) -> float:
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

    def _bbox_to_poly(self, bbox: Any) -> Optional[List[List[float]]]:
        if not bbox:
            return None

        if isinstance(bbox, dict):
            x = bbox.get("x") or bbox.get("left") or 0
            y = bbox.get("y") or bbox.get("top") or 0
            w = bbox.get("w") or bbox.get("width") or 0
            h = bbox.get("h") or bbox.get("height") or 0
            x, y, w, h = float(x), float(y), float(w), float(h)
            return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x, y, w, h = map(float, bbox[:4])
            return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

        return None

    def _normalise_polygon(
        self, polygon: Optional[List[List[float]]], width: int, height: int
    ) -> List[List[int]]:
        if not polygon or len(polygon) < 4:
            return [[0, 0], [width, 0], [width, height], [0, height]]

        normalised: List[List[int]] = []
        for x, y in polygon[:4]:
            norm_x = int(max(0, min(width, x)))
            norm_y = int(max(0, min(height, y)))
            normalised.append([norm_x, norm_y])
        return normalised

    def _poly_to_bbox(self, polygon: List[List[int]]) -> tuple[int, int, int, int]:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return (x1, y1, x2 - x1, y2 - y1)

    def is_available(self) -> bool:
        if self.device == "cpu":
            return True
        try:
            import torch  # type: ignore

            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"backend": "local", "device": self.device}
        if self.device == "cuda":
            try:
                import torch  # type: ignore

                info["cuda_available"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    info["gpu_name"] = torch.cuda.get_device_name(0)
            except ImportError:
                info["cuda_available"] = False
        return info


class RemoteGPUBackend(GPUBackend):
    """Run workloads on a remote GPU endpoint (provider-agnostic)."""

    def __init__(self, api_key: str, endpoint: str, timeout: int = 300) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers.update(headers)

    def run_deepseek_ocr(
        self,
        image: Union[str, Path, Image.Image],
        **kwargs: Any,
    ) -> OCRResult:
        """Run DeepSeek OCR on the remote endpoint."""

        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
            source = None
        else:
            source = str(image)
            pil_image = Image.open(image).convert("RGB")

        # Log image details
        image_size = pil_image.size
        logger.info(f"[RUNPOD] Preparing to send image to RunPod: size={image_size}, source={source}")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        image_kb = len(image_b64) / 1024
        logger.info(f"[RUNPOD] Image encoded: {image_kb:.1f} KB base64")

        # Payload format for our FastAPI server
        prompt_type = str(kwargs.get("prompt_type") or "markdown")
        payload = {
            "image": image_b64,
            "prompt_type": prompt_type,
            "include_geometry": True,
        }

        start = time.time()
        logger.info(f"[RUNPOD] Sending OCR request to RunPod: {self.endpoint}/ocr_base64")

        try:
            response = self._session.post(
                f"{self.endpoint}/ocr_base64",
                json=payload,
                timeout=self.timeout,
            )

            logger.info(f"[RUNPOD] RunPod response status: {response.status_code}")

            response.raise_for_status()
            result = response.json()

            elapsed = time.time() - start
            logger.info(f"[RUNPOD] RunPod OCR completed in {elapsed:.2f}s")
            logger.info(f"[RUNPOD] Response keys: {list(result.keys())}")
            logger.info(f"[RUNPOD] Response success: {result.get('success')}")

        except requests.Timeout as exc:
            logger.error(f"[RUNPOD] RunPod request TIMEOUT after {self.timeout}s: {exc}")
            raise RuntimeError(f"RunPod request timeout after {self.timeout}s") from exc
        except requests.HTTPError as exc:
            logger.error(f"[RUNPOD] RunPod HTTP error {response.status_code}: {exc}")
            logger.error(f"Response body: {response.text[:500]}")
            raise RuntimeError(f"RunPod HTTP error {response.status_code}: {exc}") from exc
        except requests.RequestException as exc:
            logger.error(f"[RUNPOD] RunPod request failed: {exc}")
            raise RuntimeError(f"Remote GPU request failed: {exc}") from exc
        except (ValueError, RuntimeError) as exc:
            logger.error("Remote OCR returned error payload: %s", exc)
            raise RuntimeError(f"Remote GPU returned error: {exc}") from exc

        return self._build_result(result, source, start)

    def _build_result(
        self, payload: Dict[str, Any], source: Optional[str], start_time: float
    ) -> OCRResult:
        """Normalise the JSON payload returned by the RunPod server."""

        text = str(payload.get("text", ""))
        text_length = len(text)

        logger.info(f"[RUNPOD] Extracted text: {text_length} characters")
        if text_length:
            logger.info(f"First 100 chars: {text[:100]}")
        else:
            logger.warning("[RUNPOD] DeepSeek returned empty text")

        lines = self._normalise_lines(payload.get("lines"))
        confidence = self._average_confidence(lines)

        processing_time = payload.get("processing_time")
        if isinstance(processing_time, (int, float)):
            elapsed = float(processing_time)
        else:
            elapsed = time.time() - start_time

        return OCRResult(
            text=text,
            lines=lines,
            confidence=confidence,
            processing_time=elapsed,
            source_file=source,
            processor=OCRProcessor.DEEPSEEK.value,
            backend="remote",
        )

    def _normalise_lines(
        self, raw_lines: Any
    ) -> Optional[List[Dict[str, Any]]]:  # pragma: no cover - exercised via integration
        if not isinstance(raw_lines, list):
            return None

        normalised: List[Dict[str, Any]] = []
        for index, entry in enumerate(raw_lines):
            if not isinstance(entry, dict):
                continue

            text = str(entry.get("text", "")).strip()
            if not text:
                continue

            polygon = self._coerce_polygon(entry.get("poly") or entry.get("polygon"))
            bbox = self._coerce_bbox(entry.get("bbox_px") or entry.get("bbox"))
            if not polygon and bbox:
                polygon = self._bbox_to_polygon(bbox)
            conf = entry.get("conf") or entry.get("confidence")

            try:
                conf_value = float(conf)
            except (TypeError, ValueError):
                conf_value = 0.0

            normalised.append(
                {
                    "line_id": str(entry.get("line_id") or f"ln_{index:04d}"),
                    "text": text,
                    "poly": polygon,
                    "bbox_px": bbox if bbox else self._polygon_to_bbox(polygon),
                    "conf": conf_value,
                }
            )

        return normalised or None

    def _coerce_polygon(self, polygon: Any) -> List[List[int]]:
        if not isinstance(polygon, list):
            return []

        points: List[List[int]] = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = int(float(point[0]))
                y = int(float(point[1]))
            except (TypeError, ValueError):
                continue
            points.append([x, y])

        return points[:4] if len(points) >= 4 else points

    def _coerce_bbox(self, bbox: Any) -> List[int]:
        if isinstance(bbox, dict):
            keys = ("x", "left", "y", "top", "w", "width", "h", "height")
            if not any(key in bbox for key in keys):
                return []
            x = bbox.get("x") or bbox.get("left") or 0
            y = bbox.get("y") or bbox.get("top") or 0
            w = bbox.get("w") or bbox.get("width") or 0
            h = bbox.get("h") or bbox.get("height") or 0
            try:
                return [int(float(x)), int(float(y)), int(float(w)), int(float(h))]
            except (TypeError, ValueError):
                return []

        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                return [int(float(value)) for value in bbox[:4]]
            except (TypeError, ValueError):
                return []

        return []

    def _bbox_to_polygon(self, bbox: List[int]) -> List[List[int]]:
        if len(bbox) < 4:
            return []
        x, y, w, h = bbox[:4]
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    def _polygon_to_bbox(self, polygon: List[List[int]]) -> List[int]:
        if len(polygon) < 4:
            return []
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return [x1, y1, x2 - x1, y2 - y1]

    def _average_confidence(
        self, lines: Optional[List[Dict[str, Any]]]
    ) -> Optional[float]:  # pragma: no cover - exercised via integration
        if not lines:
            return None

        scores: List[float] = []
        for entry in lines:
            value = entry.get("conf")
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if 0.0 <= score <= 1.0:
                scores.append(score)
                continue
            if 1.0 < score <= 100.0:
                scores.append(score / 100.0)

        if not scores:
            return None

        return sum(scores) / len(scores)

    def is_available(self) -> bool:
        try:
            response = self._session.get(f"{self.endpoint}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_info(self) -> Dict[str, Any]:
        return {
            "backend": "remote",
            "endpoint": self.endpoint,
            "available": self.is_available(),
        }


class GPUBackendManager:
    """Unified interface for GPU backends."""

    def __init__(self, config: GPUBackendConfig) -> None:
        self.config = config
        self.backend = self._create_backend(config)

    def _create_backend(self, config: GPUBackendConfig) -> GPUBackend:
        if config.use_remote:
            if not config.runpod_api_key or not config.runpod_endpoint:
                raise ValueError(
                    "Remote GPU credentials required when USE_REMOTE_GPU=True",
                )
            logger.info("Using remote GPU backend: %s", config.runpod_endpoint)
            return RemoteGPUBackend(
                api_key=config.runpod_api_key,
                endpoint=config.runpod_endpoint,
                timeout=config.runpod_timeout,
            )

        logger.info("Using local GPU backend: %s", config.local_device)
        return LocalGPUBackend(device=config.local_device)

    def run_deepseek_ocr(
        self,
        image: Union[str, Path, Image.Image],
        **kwargs: Any,
    ) -> OCRResult:
        return self.backend.run_deepseek_ocr(image, **kwargs)

    def is_available(self) -> bool:
        return self.backend.is_available()

    def get_info(self) -> Dict[str, Any]:
        info = self.backend.get_info()
        info["use_remote"] = self.config.use_remote
        return info
