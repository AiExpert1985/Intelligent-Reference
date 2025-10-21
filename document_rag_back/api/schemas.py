# api/schemas.py
from pydantic import BaseModel, HttpUrl
from typing import Optional, List

from core.domain import DocumentResponseStatus, ErrorCode, ProcessingStatus

class Message(BaseModel):
    sender: str
    content: str

class DocumentsListItem(BaseModel):
    id: str
    filename: str
    download_url: HttpUrl

class ChatRequest(BaseModel):
    question: str

class SearchResult(BaseModel):
    document_name: str
    page_number: int
    content_snippet: str
    document_id: str
    download_url: HttpUrl

class SearchResponse(BaseModel):
    status: str
    query: str
    results: List[SearchResult]
    total_results: int

class DocumentProcessingError(Exception):
    """Raised when document processing fails with a specific error code"""
    
    def __init__(self, message: str, error_code: 'ErrorCode'):
        self.message = message
        self.error_code = error_code
        super().__init__(message)
    
    def __str__(self):
        # Format used for logging and progress store
        return f"[{self.error_code.value}] {self.message}"


class ProcessDocumentResponse(BaseModel):
    status: DocumentResponseStatus
    filename: str
    document_id: str
    chunks: int
    pages: int
    message: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[ErrorCode] = None

class StatusResponse(BaseModel):
    document_loaded: Optional[str] = None
    chunks_available: int = 0
    ready_for_queries: bool = False

class DocumentsListResponse(BaseModel):
    documents: List[DocumentsListItem]

class DeleteResponse(BaseModel):
    status: str
    message: str

# NEW: Progress tracking response
class ProcessingProgress(BaseModel):
    document_id: str
    filename: str
    status: ProcessingStatus
    progress_percent: int  # 0-100
    current_step: str
    error: Optional[str] = None
    error_code: Optional[ErrorCode] = None


class PageSearchResultItem(BaseModel):
    document_id: str
    document_name: str
    page_number: int
    score: float
    chunk_count: int
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    highlights: List[str]
    download_url: str            # ✅ Can be empty string
    highlight_token: Optional[str] = None


class PageSearchResponse(BaseModel):
    status: str
    query: str
    results: List[PageSearchResultItem]
    total_results: int