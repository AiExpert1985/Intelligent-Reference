"""DeepSeek OCR processor implementation."""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from config import settings
from core.interfaces import IPdfToImageConverter
from infrastructure.document_processors import BaseOCRProcessor, LineRec
from infrastructure.gpu_backends import GPUBackendConfig, GPUBackendManager

logger = logging.getLogger(settings.LOGGER_NAME)


class DeepSeekProcessor(BaseOCRProcessor):
    """DeepSeek OCR processor backed by the GPU backend manager."""

    def __init__(
        self,
        pdf_converter: Optional[IPdfToImageConverter] = None,
        *,
        backend_manager: Optional[GPUBackendManager] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 60,
    ) -> None:
        super().__init__(
            pdf_converter=pdf_converter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if backend_manager is None:
            config = GPUBackendConfig.from_settings(settings)
            backend_manager = GPUBackendManager(config)

        self.backend = backend_manager
        logger.info("DeepSeek processor initialised: %s", self.backend.get_info())

    async def _extract_lines(self, image: Image.Image) -> List[LineRec]:
        result = await asyncio.to_thread(self.backend.run_deepseek_ocr, image)
        lines = self._normalise_lines(result.lines, image.size)
        if not lines and result.text:
            lines = self._create_fallback_line(result.text, image.size)
        return lines

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        logger.info("Using DeepSeekProcessor")
        result = await asyncio.to_thread(self.backend.run_deepseek_ocr, image)
        return result.text or ""

    def _normalise_lines(
        self,
        raw_lines: Optional[List[Dict[str, Any]]],
        image_size: Tuple[int, int],
    ) -> List[LineRec]:
        if not raw_lines:
            return []

        width, height = image_size
        normalised: List[LineRec] = []
        for entry in raw_lines:
            if not isinstance(entry, dict):
                continue

            text = str(entry.get("text", "")).strip()
            if not text:
                continue

            poly = entry.get("poly")
            bbox_px = entry.get("bbox_px")
            if not poly or not bbox_px:
                poly = [[0, 0], [width, 0], [width, height], [0, height]]
                bbox_px = (0, 0, width, height)

            line_id = entry.get("line_id") or f"ln_{uuid.uuid4().hex[:8]}"
            conf = float(entry.get("conf", 0.0))

            normalised.append(
                {
                    "line_id": line_id,
                    "poly": poly,
                    "bbox_px": bbox_px,
                    "text": text,
                    "conf": conf,
                }
            )
        return normalised

    def _create_fallback_line(
        self,
        text: str,
        image_size: Tuple[int, int],
    ) -> List[LineRec]:
        clean_text = text.strip()
        if not clean_text:
            return []

        width, height = image_size
        poly = [[0, 0], [width, 0], [width, height], [0, height]]

        return [
            {
                "line_id": f"ln_{uuid.uuid4().hex[:8]}",
                "poly": poly,
                "bbox_px": (0, 0, width, height),
                "text": clean_text,
                "conf": 0.0,
            }
        ]
