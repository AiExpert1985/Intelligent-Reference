# config.py
"""Enhanced configuration with new settings"""
from typing import List
from pydantic_settings import BaseSettings
from utils.common import get_log_file_path, get_project_root

class Settings(BaseSettings):
    """Application configuration"""
    
    # Logger configuration
    LOG_FILE_PATH: str = get_log_file_path()
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./chat.db"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE: int = 3600  # Recycle connections after 1 hour
    
    # Vector store
    VECTOR_DB_PATH: str = "./vector_db"
    VECTOR_STORE_TYPE: str = "chromadb"
    
    # Embedding model
    EMBEDDING_MODEL_NAME: str = "paraphrase-multilingual-mpnet-base-v2"
    
    # Document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE: int = 50 * 1024 * 1024
    UPLOADS_DIR: str = f"{get_project_root()}/uploads"
    
    # PDF processing method selection
    PDF_PROCESSING_METHOD: str = "ocr"
    OCR_ENGINE: str = "easyocr" # Options: tesseract, easyocr, paddleocr
    OCR_DPI: int = 300
    OCR_LANGUAGES: List[str] = ["ar", "en"]
    
    # File type categorization
    IMAGE_EXTENSIONS: List[str] = ["jpg", "jpeg", "png"]
    DOCUMENT_EXTENSIONS: List[str] = ["pdf"]
    
    @property
    def ALLOWED_FILE_EXTENSIONS(self) -> List[str]:
        return self.IMAGE_EXTENSIONS + self.DOCUMENT_EXTENSIONS
    
    # API/RAG Configuration
    OCR_TIMEOUT_SECONDS: float = 300.0
    SNIPPET_LENGTH: int = 300
    DEFAULT_SEARCH_RESULTS: int = 5
    
    # ============= END: Search Quality Controls =============
    
    # API settings
    REQUEST_TIMEOUT: int = 60
    MAX_SEARCH_RESULTS: int = 10  # Legacy setting
    
    # App metadata
    APP_TITLE: str = "Document RAG System"
    APP_VERSION: str = "2.0.0"
    
    # Performance
    USE_CONNECTION_POOLING: bool = True
    MAX_CONNECTIONS: int = 10

    # Document processing
    ARABIC_BIDI_WRAP_FOR_DISPLAY: bool = True  # Add RTL markers for display


    # Search quality controls
    SEARCH_CANDIDATE_K: int = 40
    SEARCH_SCORE_THRESHOLD: float = 0.72
    
    LEXICAL_GATE_ENABLED: bool = True
    LEXICAL_MIN_KEYWORDS: int = 1
    
    # Reranking configuration
    RERANK_ENABLED: bool = True
    RERANK_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_SCORE_THRESHOLD: float = 0.50
    RERANK_TOP_K: int = 15

    # NEW: Reranker gating (skip reranking for obvious hits/misses)
    RERANK_GATE_LOW: float = 0.55   # Don't rerank if best score below this
    RERANK_GATE_HIGH: float = 0.75  # Don't rerank if best score above this
    RERANK_CANDIDATE_CAP: int = 15  # Max items to send to cross-encoder

    # Search defaults
    TOP_K: int = 5
    DEFAULT_SEARCH_RESULTS: int = 5
    SNIPPET_LENGTH: int = 300

    # Uploads base directory
    UPLOADS_DIR: str = "uploads"

    # Security flags (Phase 0 MVP)
    REQUIRE_AUTHENTICATION: bool = False
    ENABLE_DOCUMENT_ACL: bool = False

    # Logging
    LOGGER_NAME: str = "alfahras"
    DEBUG_SEARCH_DUMPS: bool = False
    DEBUG_SEARCH_JSON: bool = False
    DEBUG_SEARCH_MAX_ITEMS: int = 10

    # Highlight preview feature flags / params
    ENABLE_HIGHLIGHT_PREVIEW: bool = True
    HIGHLIGHT_STYLE_ID: str = "default"
    HIGHLIGHT_MAX_REGIONS: int = 10
    HIGHLIGHT_TIMEOUT_SEC: int = 10

    # Token secret (SET IN .env FOR PROD)
    HIGHLIGHT_TOKEN_SECRET: str = "CHANGE_ME_IN_PROD"

    # ---------- Highlight / Search knobs ----------
    HIGHLIGHT_SCORE_THRESHOLD: float = 0.65
    HIGHLIGHT_MERGE_POLICY: str = "none"

    # Sentence segmentation
    SENT_MIN_CHARS: int = 30
    SENT_MAX_PER_PAGE: int = 80

    # Hybrid retrieval caps
    HYBRID_TOP_SENTENCES: int = 400  # Replaces HYBRID_TOP_LINES
    HYBRID_TOP_CHUNKS: int = 100
    HYBRID_PER_PAGE_SENTENCES: int = 6  # Replaces HYBRID_PER_PAGE_LINES

    # Backward-compatible aliases during migration
    HYBRID_TOP_LINES: int = 400
    HYBRID_PER_PAGE_LINES: int = 6

    # Perf guards
    HYBRID_RETRIEVAL_TIMEOUT_SEC: float = 5.0
    HYBRID_MAX_TOTAL_HITS: int = 600  # cap before normalization

    # Saturating aggregation (lines)
    HYBRID_ALPHA: float = 0.7
    HYBRID_LAMBDA: float = 0.9

    # Fusion weight
    HYBRID_BETA: float = 0.6

    # Line indexing filters
    LINE_MIN_CHARS: int = 20
    LINE_EXCLUDE_HEADER_FOOTER_BAND: float = 0.06

    # Telemetry
    ENABLE_HYBRID_TELEMETRY: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
