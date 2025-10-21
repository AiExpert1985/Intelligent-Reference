# core/domain.py
"""Shared enumerations used across the application."""
from enum import Enum

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# ============= Enums =============

class ErrorCode(str, Enum):
    """Error codes for user-facing error messages."""
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FORMAT = "INVALID_FORMAT"
    DUPLICATE_FILE = "DUPLICATE_FILE"
    NO_TEXT_FOUND = "NO_TEXT_FOUND"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    OCR_TIMEOUT = "OCR_TIMEOUT"


class ProcessingStatus(str, Enum):
    """Document processing pipeline stages."""
    PENDING = "pending"
    VALIDATING = "validating"
    EXTRACTING_TEXT = "extracting_text"
    GENERATING_EMBEDDINGS = "generating_embeddings"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    
    @staticmethod
    def from_string(status: str) -> 'ProcessingStatus':
        """Convert string to ProcessingStatus enum."""
        try:
            return ProcessingStatus(status)
        except ValueError:
            return ProcessingStatus.FAILED
        
class DocumentResponseStatus(str, Enum):
    """Status for document upload response."""
    PROCESSING = "processing"  # Upload accepted, processing in background
    ERROR = "error"             # Upload or validation failed



# ============= Domain Models =============

@dataclass
class ProcessedDocument:
    """Domain model for documents"""
    id: str
    filename: str
    file_hash: str
    metadata: Dict[str, Any]

@dataclass  
class DocumentChunk:
    """Domain model for document chunks"""
    id: str
    content: str
    document_id: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None # Vector of float numbers

@dataclass
class ChunkSearchResult:
    """Domain model for search results"""
    chunk: DocumentChunk
    score: float
    highlights: Optional[List[str]] = None


@dataclass
class PageSearchResult:
    """Page-level search result with highlights and images."""
    document_id: str
    document_name: str
    page_number: int
    score: float
    chunk_count: int
    image_url: str
    thumbnail_url: str
    highlights: List[str]
    download_url: Optional[str] = None
    highlight_token: Optional[str] = None
