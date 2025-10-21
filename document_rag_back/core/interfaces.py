# core/interfaces.py
"""Core interfaces for the RAG system"""
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from fastapi import UploadFile, Request

from api.schemas import DocumentsListItem, ProcessDocumentResponse
from core.domain import ChunkSearchResult, DocumentChunk, PageSearchResult, ProcessedDocument 

# ============= Vector Store Interface =============
class IVectorStore(ABC):
    """Interface for vector storage operations"""
    
    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks with embeddings"""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[ChunkSearchResult]:
        """Search for similar chunks"""
        pass
    
    @abstractmethod
    async def delete_by_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all stored vectors"""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Get total number of chunks"""
        pass

# ============= Document Processor Interface =============
class IDocumentProcessor(ABC):
    """ Document text extraction and chunking (OCR-based)."""
    
    @abstractmethod
    async def process(
        self, 
        file_path: str, 
        file_type: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[DocumentChunk], Dict[int, Dict[str, Any]]]:
        """
        Extract text via OCR and split into chunks.

        Returns:
            Tuple:
              - List[DocumentChunk]
              - Dict[int, Dict[str, Any]]  # geometry_by_page
        """
        pass

    @abstractmethod
    async def load_images(self, file_path: str, file_type: str) -> List[Any]:
        """Load images from file (PDF or image file)."""
        pass

# ============= Embedding Service Interface =============
class IEmbeddingService(ABC):
    """Interface for embedding generation"""
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        pass
    
    @abstractmethod
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        pass

# ============= Repository Interfaces =============
class IDocumentRepository(ABC):
    """
    Interface for document metadata persistence (filenames, hashes, storage paths).
    
    Does NOT handle: physical files (see IFileStorage) or vectors (see IVectorStore).
    Implementations: SQLDocumentRepository. Swap for MongoDB, DynamoDB, etc.
    """

    @abstractmethod
    async def update_metadata(self, document_id: str, metadata: Dict[str, Any]) -> bool:
        """Persist document.metadata JSON/JSONB field."""
        raise NotImplementedError
    
    @abstractmethod
    async def create(self, document_id: str, filename: str, file_hash: str, stored_filename: str) -> ProcessedDocument:
        """
        Create DB record for document metadata. Physical file saved separately.
        stored_filename is UUID-based name on disk for later retrieval/deletion.
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, document_id: str) -> Optional[ProcessedDocument]:
        """Get document by ID"""
        pass
    
    @abstractmethod
    async def get_by_hash(self, file_hash: str) -> Optional[ProcessedDocument]:
        """
        Find document by SHA256 hash for duplicate detection.
        Returns None if not found (safe to upload).

        """
        pass
    
    @abstractmethod
    async def list_all(self) -> List[ProcessedDocument]:
        """List all documents"""
        pass
    
    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete document record"""
        pass
    
    @abstractmethod
    async def delete_all(self) -> bool:
        """Delete all documents"""
        pass

    @abstractmethod
    async def exists_bulk(self, document_ids: List[str]) -> Set[str]:
        """
        Return set of existing document IDs from provided list.
        Used for efficient existence checking without loading all documents.
        """
        pass

class IMessageRepository(ABC):
    """Interface for message/search history"""
    
    @abstractmethod
    async def save_search(self, query: str, results_count: int) -> None:
        """Save search query"""
        pass
    
    @abstractmethod
    async def get_search_history(self, limit: int = 10) -> List[Dict]:
        """Get search history"""
        pass
    
    @abstractmethod
    async def clear_history(self) -> bool:
        """Clear search history"""
        pass

    @abstractmethod
    async def save_search_results(self, query: str, results: List[ChunkSearchResult]) -> None:
        """Save search results in structured format"""
        pass

# ============= File Storage Interface =============
class IFileStorage(ABC):
    """Interface for physical file storage operations"""

    @abstractmethod
    async def save(self, file: UploadFile, filename: str) -> str:
        """
        Saves an uploaded file to the configured storage directory.
        
        Writes the file content to disk using the provided filename. Creates the
        upload directory if it doesn't exist. The filename should already be
        sanitized and unique (typically UUID-based) by the caller.
        
        Args:
            file: The uploaded file object containing content to save
            filename: Secure filename to use for storage (e.g., "uuid123.pdf")
            
        Returns:
            str: Full absolute path to the saved file
            
        Raises:
            Exception: If file cannot be written to disk (permissions, disk full, etc.)
            
        Example:
            file_path = await storage.save(uploaded_file, "abc-123.pdf")

            Returns: "/app/uploads/abc-123.pdf"
        """
        pass

    @abstractmethod
    async def get_path(self, filename: str) -> Optional[str]:
        """Get the full path to a stored file."""
        pass

    @abstractmethod
    async def delete(self, filename: str) -> bool:
        """Delete a stored file."""
        pass

    @abstractmethod
    async def save_page_image(
        self, 
        image: Any, 
        document_id: str, 
        page_number: int
    ) -> Tuple[str, str]:
        """Save page image and thumbnail. Returns (original_path, thumb_path)."""
        pass

# ============= Service Layer Interfaces =============
class IRAGService(ABC):
    """High-level RAG operations interface"""

    document_repo: IDocumentRepository
    vector_store: IVectorStore
    reranker: Optional['IReranker']  # ✅ String delays type checking
    

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        pass
    
    @abstractmethod
    async def process_document(self, file: UploadFile) -> ProcessDocumentResponse:
        """
        Validate file and start background processing (returns immediately).
        
        Fast path: validation + file save (< 1s)
        Background: duplicate check → OCR → embeddings → storage (30s-5min)
        Client polls /processing-status/{doc_id} for progress updates.
        """
        pass

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[ChunkSearchResult]:
        """Search across all documents"""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks"""
        pass
        
    @abstractmethod
    async def clear_all(self) -> bool:
        """Clear all documents and vectors"""
        pass
        
    @abstractmethod
    async def list_documents(self, request: Request) -> List[DocumentsListItem]:
        """List all documents, including download URLs."""
        pass
        
    @abstractmethod
    async def get_document_with_path(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document's details and its physical file path."""
        pass

    @abstractmethod
    async def search_chunks(self, query: str, top_k: int = 5) -> List[ChunkSearchResult]:
        """Return chunk-level results (for LLM/compat)."""
        pass

    @abstractmethod
    async def search_pages(self, query: str, top_k: int = 5) -> List[PageSearchResult]:
        """Return page-level results with images (for UI)."""
        pass

# ============= PDF to Image Converter Interface =============
class IPdfToImageConverter(ABC):
    """Interface for PDF to image conversion"""

    @abstractmethod
    def convert(self, file_path: str, dpi: int = 300) -> List[Any]:
        """Converts a PDF file to a list of images."""
        pass


# ============= Semantic Reranking =============

class IReranker(ABC):
    """
    Interface for semantic reranking of search results.
    Allows swapping reranker implementations (cross-encoder, APIs, custom models).
    """
    
    @abstractmethod
    async def rerank(
        self, 
        query: str, 
        results: List['ChunkSearchResult'], 
        top_k: int = 5
    ) -> List['ChunkSearchResult']:
        """
        Rerank search results by semantic relevance.
        
        Args:
            query: Search query
            results: Initial results from vector search
            top_k: Maximum results to return
            
        Returns:
            Reranked results (sorted by relevance)
        """
        pass