# infrastructure/faiss_store.py
import asyncio
import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

import faiss
import numpy as np

from core.interfaces import IVectorStore, DocumentChunk
from core.domain import ChunkSearchResult
from config import settings

logger = logging.getLogger(settings.LOGGER_NAME)

class FAISSVectorStore(IVectorStore):
    """
    FAISS implementation with thread-safe mutations and stable index ordering.
    
    Key improvements:
    - Single asyncio.Lock() prevents race conditions
    - _row_ids list maintains stable FAISS row → chunk_id mapping
    - Deletion rebuilds using stable order (no dict key order dependency)
    - Load/save persists both metadata and row order
    """
    
    def __init__(self, index_path: str = settings.VECTOR_DB_PATH):
        self._index: Optional[faiss.IndexFlatL2] = None
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._row_ids: List[str] = []  # FAISS row → chunk_id (stable order)
        self._lock = asyncio.Lock()  # Protects all mutations
        
        self._index_path = Path(index_path) / "faiss.index"
        self._metadata_path = Path(index_path) / "faiss_metadata.json"
        
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_index()

    def _load_index(self):
        """Loads index, metadata, and row order from disk"""
        # Load FAISS index
        if self._index_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                logger.info(f"[FAISS] Loaded index from {self._index_path}")
            except Exception as e:
                logger.warning(f"[FAISS] Failed to load index: {e}. Starting fresh.")
                self._index = None
        
        # Load metadata + row_ids
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._metadata = data.get("metadata", {})
                    self._row_ids = data.get("row_ids", [])
                logger.info(f"[FAISS] Loaded {len(self._metadata)} chunks with stable ordering.")
            except Exception as e:
                logger.warning(f"[FAISS] Failed to load metadata: {e}. Starting fresh.")
                self._metadata = {}
                self._row_ids = []

    async def _save_index_locked(self):
        """
        Saves index and metadata to disk (must be called under self._lock).
        Persists both chunk metadata and row order for stable reconstruction.
        """
        # Save FAISS index
        if self._index:
            await asyncio.to_thread(faiss.write_index, self._index, str(self._index_path))
        
        # Save metadata + row_ids together
        data = {
            "metadata": self._metadata,
            "row_ids": self._row_ids
        }
        with open(self._metadata_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[FAISS] Saved index and metadata.")

    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add chunks with thread-safe mutations"""
        if not chunks:
            return True

        # Prepare data
        embeddings = np.array([c.embedding for c in chunks], dtype='float32')
        metas = [
            {
                "chunk_id": c.id,
                "content": c.content,
                "document_id": c.document_id,
                "metadata": c.metadata or {}
            }
            for c in chunks
        ]

        # Initialize index if needed
        if self._index is None:
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatL2(dim)
            logger.info(f"[FAISS] Initialized new index with dimension {dim}")

        try:
            async with self._lock:
                # Add to FAISS index
                await asyncio.to_thread(self._index.add, embeddings) # type: ignore
                
                # Update metadata and row order
                for meta in metas:
                    chunk_id = meta["chunk_id"]
                    self._row_ids.append(chunk_id)
                    self._metadata[chunk_id] = meta
                
                # Persist atomically
                await self._save_index_locked()
            
            return True
        except Exception as e:
            logger.error(f"[FAISS] Failed to add chunks: {e}")
            return False

    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[ChunkSearchResult]:
        """
        Search with thread-safe reads using stable row ordering.
        Returns results with normalized cosine similarity scores [0,1].
        """
        if not self._index:
            logger.warning("[FAISS] Search called but index is empty.")
            return []

        try:
            query_vector = np.array([query_embedding], dtype='float32')
            distances, indices = await asyncio.to_thread(
                self._index.search, query_vector, top_k
            ) # type: ignore
            
            results = []
            
            async with self._lock:  # Read lock for consistency
                idxs = indices[0]
                dists = distances[0]
                
                for pos, row in enumerate(idxs):
                    # Skip invalid indices
                    if row == -1 or row >= len(self._row_ids):
                        continue
                    
                    # Get chunk via stable row → chunk_id mapping
                    chunk_id = self._row_ids[row]
                    metadata = self._metadata.get(chunk_id)
                    
                    if not metadata:
                        continue
                    
                    # Convert L2 distance (on normalized vectors) to cosine similarity
                    # For unit vectors: d² = 2(1-cos) → cos = 1 - d²/2
                    # Map to [0,1]: similarity = 1 - d²/4
                    l2_distance_squared = float(dists[pos])
                    similarity = 1.0 - (l2_distance_squared / 4.0)
                    similarity = max(0.0, min(1.0, similarity))  # Clamp
                    
                    # Build result
                    chunk = DocumentChunk(
                        id=chunk_id,
                        content=metadata['content'],
                        document_id=metadata['document_id'],
                        metadata=metadata['metadata']
                    )
                    results.append(ChunkSearchResult(chunk=chunk, score=similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"[FAISS] Search failed: {e}")
            return []

    async def delete_by_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document and rebuild index with stable ordering.
        Uses _row_ids to ensure correct vector reconstruction.
        """
        if not self._index or not self._metadata:
            return True

        try:
            async with self._lock:
                keep_ids: List[str] = []
                keep_vecs: List[np.ndarray] = []
                keep_meta: Dict[str, Dict[str, Any]] = {}
                
                # Iterate using stable row order
                for row, chunk_id in enumerate(self._row_ids):
                    meta = self._metadata.get(chunk_id)
                    if not meta:
                        continue
                    
                    # Keep chunks from other documents
                    if meta.get('document_id') != document_id:
                        keep_ids.append(chunk_id)
                        keep_meta[chunk_id] = meta
                        # Reconstruct vector using row index
                        vec = self._index.reconstruct(row) # type: ignore
                        keep_vecs.append(vec)
                
                # Rebuild index
                dim = self._index.d
                new_index = faiss.IndexFlatL2(dim)
                
                if keep_vecs:
                    embeddings = np.vstack(keep_vecs).astype('float32', copy=False)
                    await asyncio.to_thread(new_index.add, embeddings)  # type: ignore
                
                # Update state
                self._index = new_index
                self._row_ids = keep_ids
                self._metadata = keep_meta
                
                # Persist
                await self._save_index_locked()
                
                logger.info(f"[FAISS] Deleted document {document_id}. "
                           f"Remaining chunks: {len(keep_meta)}")
                return True
                
        except Exception as e:
            logger.error(f"[FAISS] Deletion failed: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all vectors, metadata, and row order"""
        try:
            async with self._lock:
                # Reset to empty index
                if self._index:
                    dim = self._index.d
                    self._index = faiss.IndexFlatL2(dim)
                else:
                    self._index = None
                
                self._metadata.clear()
                self._row_ids.clear()
                
                # Remove files
                if self._index_path.exists():
                    await asyncio.to_thread(self._index_path.unlink)
                if self._metadata_path.exists():
                    await asyncio.to_thread(self._metadata_path.unlink)
                
                return True
        except Exception as e:
            logger.error(f"[FAISS] Clear failed: {e}")
            return False

    async def count(self) -> int:
        """Get total number of chunks"""
        return self._index.ntotal if self._index else 0