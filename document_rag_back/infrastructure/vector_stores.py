# infrastructure/vector_stores.py
"""Concrete implementations of vector stores"""
import asyncio
import logging
from typing import Any, List

import chromadb

from core.interfaces import IVectorStore, DocumentChunk
from core.domain import ChunkSearchResult

from config import settings

logger = logging.getLogger(settings.LOGGER_NAME)


class ChromaDBVectorStore(IVectorStore):
    """ChromaDB implementation with normalized cosine similarity scoring (0-1 scale)"""
    
    def __init__(self, client: Any, collection_name: str = "rag_chunks"):
        self._client = client
        self._collection_name = collection_name
        self._collection: Any = None
        
    async def _ensure_collection(self):
        """Lazy initialization of collection"""
        if not self._collection:
            self._collection = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=self._collection_name
            )
    
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add chunks to ChromaDB"""
        try:
            await self._ensure_collection()
            
            if not chunks:
                return True
                
            texts = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata or {} for chunk in chunks]
            ids = [chunk.id for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
            
            await asyncio.to_thread(
                self._collection.add,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings if embeddings else None
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}")
            return False
    
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[ChunkSearchResult]:
        """
        Search with unified cosine similarity scores (0-1 scale).
        
        CHANGED: Score calculation now returns normalized similarity:
        - Old: score = 1 - distance → assumes distance in [0,1]
        - New: score = 1 - (distance/2) → handles cosine distance in [0,2]
        
        ChromaDB behavior (for L2-normalized vectors):
        - Returns: cosine distance = 1 - cos(θ) 
        - Range: [0, 2] where 0=identical, 2=opposite
        - Conversion: similarity = 1 - (distance/2) maps to [0,1]
        
        This makes 0.70 threshold mean the same thing in ChromaDB and FAISS.
        """
        try:
            await self._ensure_collection()
            
            results = await asyncio.to_thread(
                self._collection.query,
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['metadatas', 'documents', 'distances']
            )
            
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    # ============= NEW: Unified Similarity Scoring =============
                    # Convert ChromaDB cosine distance to similarity [0,1]
                    cosine_distance = results['distances'][0][i]
                    
                    # Cosine distance ∈ [0,2] for normalized vectors
                    # Map to similarity ∈ [0,1]: 1 - (distance/2)
                    similarity = 1.0 - (cosine_distance / 2.0)
                    
                    # Clamp to [0,1] for numerical stability
                    similarity = max(0.0, min(1.0, similarity))
                    # ============= END: Unified Similarity Scoring =============
                    
                    metadata = results['metadatas'][0][i] or {}
                    chunk = DocumentChunk(
                        id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        document_id=str(metadata.get('document_id') or ""),
                        metadata=metadata
                    )
                    search_results.append(ChunkSearchResult(chunk=chunk, score=similarity))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed in ChromaDB: {e}")
            return []
    
    async def delete_by_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            await self._ensure_collection()
            await asyncio.to_thread(
                self._collection.delete,
                where={"document_id": document_id}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all vectors"""
        try:
            await asyncio.to_thread(
                self._client.delete_collection,
                name=self._collection_name
            )
            self._collection = None
            return True
        except chromadb.errors.NotFoundError:
            self._collection = None
            return True
        except ValueError as e:
            if "does not exist" in str(e).lower():
                self._collection = None
                return True
            logger.error(f"Failed to clear collection: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    async def count(self) -> int:
        """Get chunk count"""
        try:
            await self._ensure_collection()
            return await asyncio.to_thread(self._collection.count)
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return 0

