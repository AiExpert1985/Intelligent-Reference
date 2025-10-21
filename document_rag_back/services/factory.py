# services/factory.py
from typing import Optional
import chromadb
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from database.session import get_db
from config import settings
from core.interfaces import (
    IRAGService, IVectorStore, IEmbeddingService, IReranker,
    IDocumentRepository, IMessageRepository, IFileStorage
)
from infrastructure.faiss_store import FAISSVectorStore
from infrastructure.vector_stores import ChromaDBVectorStore
from infrastructure.embedding_services import SentenceTransformerEmbedding
from infrastructure.repositories import SQLDocumentRepository, SQLMessageRepository
from infrastructure.file_storage import LocalFileStorage
from services.rag_service import RAGService
from services.document_processor_factory import DocumentProcessorFactory

from infrastructure.reranker import CrossEncoderReranker

# Provider functions for each component
def get_vector_store() -> IVectorStore:
    """Create vector store based on configuration."""
    if settings.VECTOR_STORE_TYPE == "chromadb":
        client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
        return ChromaDBVectorStore(client)
    elif settings.VECTOR_STORE_TYPE.lower() == "faiss":
        # Pass the configured path to the FAISS store
        return FAISSVectorStore(index_path=settings.VECTOR_DB_PATH)
    # Future: elif settings.VECTOR_STORE_TYPE.lower() == "pinecone":
    #     return PineconeVectorStore(...)
    else:
        raise ValueError(f"Unknown vector store type: {settings.VECTOR_STORE_TYPE}")
    
def get_line_vector_store() -> IVectorStore:
    """Create line vector store (separate collection/index for line-level retrieval)."""
    if settings.VECTOR_STORE_TYPE == "chromadb":
        client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
        return ChromaDBVectorStore(client, collection_name="lines")
    elif settings.VECTOR_STORE_TYPE.lower() == "faiss":
        from pathlib import Path
        line_index_path = str(Path(settings.VECTOR_DB_PATH) / "lines")
        return FAISSVectorStore(index_path=line_index_path)
    else:
        raise ValueError(f"Unknown vector store type: {settings.VECTOR_STORE_TYPE}")

def get_sentence_vector_store() -> IVectorStore:
    """Create sentence vector store (separate collection/index)."""
    if settings.VECTOR_STORE_TYPE == "chromadb":
        client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
        return ChromaDBVectorStore(client, collection_name="sentences")
    elif settings.VECTOR_STORE_TYPE.lower() == "faiss":
        from pathlib import Path
        sentence_index_path = str(Path(settings.VECTOR_DB_PATH) / "sentences")
        return FAISSVectorStore(index_path=sentence_index_path)
    else:
        raise ValueError(f"Unknown vector store type: {settings.VECTOR_STORE_TYPE}")

def get_embedding_service() -> IEmbeddingService:
    """Create embedding service based on configuration."""
    return SentenceTransformerEmbedding(settings.EMBEDDING_MODEL_NAME)
    # Future: if settings.EMBEDDING_PROVIDER == "openai":
    #     return OpenAIEmbeddingService(...)

def get_file_storage() -> IFileStorage:
    """Create file storage based on configuration."""
    return LocalFileStorage(base_path=settings.UPLOADS_DIR)
    # Future: if settings.STORAGE_TYPE == "s3":
    #     return S3FileStorage(...)

def get_document_processor_factory() -> DocumentProcessorFactory:
    """Create document processor factory."""
    return DocumentProcessorFactory()

def get_document_repository(session: AsyncSession = Depends(get_db)) -> IDocumentRepository:
    """Create document repository with injected session."""
    return SQLDocumentRepository(session)

def get_message_repository(session: AsyncSession = Depends(get_db)) -> IMessageRepository:
    """Create message repository with injected session."""
    return SQLMessageRepository(session)


def get_reranker() -> Optional[IReranker]:
    """Create reranker if enabled in config."""
    if not settings.RERANK_ENABLED:
        return None
    return CrossEncoderReranker(settings.RERANK_MODEL_NAME)

def get_rag_service(
    vector_store: IVectorStore = Depends(get_vector_store),
    line_vector_store: IVectorStore = Depends(get_line_vector_store),  # << ADD THIS LINE
    sentence_vector_store: IVectorStore = Depends(get_sentence_vector_store),
    embedding_service: IEmbeddingService = Depends(get_embedding_service),
    doc_processor_factory: DocumentProcessorFactory = Depends(get_document_processor_factory),
    file_storage: IFileStorage = Depends(get_file_storage),
    document_repo: IDocumentRepository = Depends(get_document_repository),
    message_repo: IMessageRepository = Depends(get_message_repository),
    reranker: IReranker = Depends(get_reranker)
) -> IRAGService:
    return RAGService(
        vector_store=vector_store,
        line_vector_store=line_vector_store,  # << ADD THIS LINE
        sentence_vector_store=sentence_vector_store,
        doc_processor_factory=doc_processor_factory,
        embedding_service=embedding_service,
        file_storage=file_storage,
        document_repo=document_repo,
        message_repo=message_repo,
        reranker=reranker
    )