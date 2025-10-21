# services/document_processor_factory.py
"""Improved factory with proper separation of concerns"""
from typing import Dict, Type
from core.interfaces import IDocumentProcessor, IPdfToImageConverter
from infrastructure.document_processors import (
    EasyOCRProcessor, 
    PaddleOCRProcessor,
    TesseractProcessor
)
from config import settings
from infrastructure.pdf_converters import PyMuPDFConverter

import logging

def _get_logger():
    from config import settings
    return logging.getLogger(settings.LOGGER_NAME)

logger = _get_logger()

class DocumentProcessorFactory:
    """
    Factory for creating OCR processors based on file type and OCR_ENGINE config.
    Swap OCR engines (EasyOCR/Tesseract/PaddleOCR) via config without code changes.
    """
    
    def __init__(self, pdf_converter_class=None):
        self._ocr_strategies: Dict[str, Type[IDocumentProcessor]] = {
            "easyocr": EasyOCRProcessor,
            "paddleocr": PaddleOCRProcessor,
            "tesseract": TesseractProcessor
        }
        self.pdf_converter_class = pdf_converter_class or PyMuPDFConverter

    def _get_pdf_processor(self) -> IDocumentProcessor:
        """Gets the document processor configured for PDF handling."""
        if settings.PDF_PROCESSING_METHOD == "ocr":
            return self._create_ocr_processor(needs_pdf_converter=True)
        # Future: elif settings.PDF_PROCESSING_METHOD == "text_extraction":
        #     return TextExtractionProcessor()
        else:
            raise ValueError(f"Unknown PDF processing method: {settings.PDF_PROCESSING_METHOD}")

    def _create_ocr_processor(self, needs_pdf_converter: bool = False) -> IDocumentProcessor:
        """
        Create OCR processor using configured OCR_ENGINE.
        PDF converter included only if needs_pdf_converter=True.
        """
        engine = settings.OCR_ENGINE.lower()
        processor_class = self._ocr_strategies.get(engine)
        
        if not processor_class:
            available = ", ".join(self._ocr_strategies.keys())
            raise ValueError(f"Unknown OCR engine: '{engine}'. Available: {available}")
        
        pdf_converter = self.pdf_converter_class() if needs_pdf_converter else None
        return processor_class(
            pdf_converter=pdf_converter, # type: ignore
            chunk_size=settings.CHUNK_SIZE, # type: ignore
            chunk_overlap=settings.CHUNK_OVERLAP # type: ignore
        )
    
    def get_processor(self, file_type: str) -> IDocumentProcessor:
        """
        Get OCR processor for file type ("pdf", "jpg", "png").
        Uses configured OCR_ENGINE (easyocr/tesseract/paddleocr).
        """
        if file_type == "pdf":
            return self._get_pdf_processor()
        elif file_type in settings.IMAGE_EXTENSIONS:
            return self._create_ocr_processor(needs_pdf_converter=False)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        

    def _build_processor(self, engine: str, file_type: str) -> IDocumentProcessor:
        """
        Build OCR processor for specific engine (bypasses config).
        
        Args:
            engine: OCR engine name (easyocr, tesseract, paddleocr)
            file_type: File type (pdf, jpg, png)
        
        Returns:
            Configured processor instance
        
        Raises:
            ValueError: If engine is unknown
        """
        processor_class = self._ocr_strategies.get(engine)
        if not processor_class:
            available = ", ".join(self._ocr_strategies.keys())
            raise ValueError(f"Unknown OCR engine: '{engine}'. Available: {available}")
        
        # Add PDF converter only if processing PDFs
        pdf_converter = None
        if file_type == "pdf":
            pdf_converter = self.pdf_converter_class()
        
        return processor_class(
            pdf_converter=pdf_converter, # type: ignore
            chunk_size=settings.CHUNK_SIZE, # type: ignore
            chunk_overlap=settings.CHUNK_OVERLAP # type: ignore
        )

    def get_fallback_processor(self, file_type: str) -> IDocumentProcessor:
        """
        Get fallback OCR processor when primary engine fails.
        
        Tries engines in order: easyocr → tesseract → paddleocr
        (excluding currently configured engine)
        
        Args:
            file_type: File type being processed
        
        Returns:
            Working processor instance
        
        Raises:
            RuntimeError: If no OCR engines are available
        """
        # Get current engine from config
        current = settings.OCR_ENGINE.lower()
        
        # Try fallbacks in priority order (excluding current)
        fallbacks = [e for e in ["easyocr", "tesseract", "paddleocr"] if e != current]
        
        last_err = None
        for engine in fallbacks:
            try:
                logger.info(f"[OCR] Attempting fallback engine: {engine}")
                processor = self._build_processor(engine=engine, file_type=file_type)
                logger.info(f"[OCR] Successfully loaded fallback: {engine}")
                return processor
            except Exception as e:
                logger.warning(f"[OCR] Fallback {engine} failed: {e}")
                last_err = e
        
        # All engines failed
        raise RuntimeError(
            f"No OCR engines available. Tried: {', '.join(fallbacks)}. "
            f"Last error: {last_err}"
        )