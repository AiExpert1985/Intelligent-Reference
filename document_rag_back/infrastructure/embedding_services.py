# infrastructure/embedding_services.py
"""Embedding generation with L2 normalization for consistent similarity scoring"""
import asyncio
import logging
import numpy as np  # NEW: Required for normalization
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from core.interfaces import IEmbeddingService
from config import settings

logger = logging.getLogger(settings.LOGGER_NAME)

class SentenceTransformerEmbedding(IEmbeddingService):
    """
    Sentence transformer with L2 normalization (unit vectors).
    
    Why normalization matters:
    - FAISS uses L2 distance: d² = ||a-b||²
    - ChromaDB uses cosine distance: d = 1 - cos(a,b)
    - With normalized vectors: L2² = 2(1-cos), making both metrics equivalent
    - Result: One threshold (0.70) behaves consistently across both stores
    
    Mathematical foundation:
    - For unit vectors (||v|| = 1): cos(a,b) = a·b
    - L2 distance²: ||a-b||² = 2 - 2(a·b) = 2(1 - cos(a,b))
    - This relationship enables unified similarity scoring
    """
    
    _model: Optional[SentenceTransformer] = None  # Singleton cache
    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        """Initializes the service, loading the heavy model only once."""
        
        if SentenceTransformerEmbedding._model is None:
            try:
                logger.info(f"Attempting to load model {model_name} from local cache...")
                SentenceTransformerEmbedding._model = SentenceTransformer(
                    model_name,
                    local_files_only=True 
                )
                logger.info(f"Successfully loaded {model_name} from local cache.")

            except Exception as e:
                logger.warning(
                    f"Model {model_name} not found in cache. Attempting online download. "
                    f"This may take a few minutes. Error: {e}"
                )
                SentenceTransformerEmbedding._model = SentenceTransformer(model_name)
                logger.info(f"Successfully downloaded and loaded {model_name}.")
            
        self.model = SentenceTransformerEmbedding._model
    
    def _l2_normalize(self, arr: np.ndarray) -> np.ndarray:
        """
        L2 normalize vectors to unit length (||v|| = 1).
        
        This is the foundation for consistent similarity scoring:
        - Converts arbitrary vectors to unit sphere
        - Makes cosine similarity = dot product
        - Enables FAISS L2 distance to represent cosine distance
        
        Args:
            arr: (N, D) array of N vectors with D dimensions
            
        Returns:
            (N, D) array of unit-normalized vectors
        """
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12  # Avoid division by zero
        return arr / norms
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate L2-normalized embeddings for multiple texts.
        
        CHANGED: Added normalization step for consistent similarity scoring.
        All stored vectors are now unit vectors, enabling geometric consistency.
        """
        raw = await asyncio.to_thread(
            self.model.encode,
            texts,
            convert_to_tensor=False
        )
        # NEW: Normalize to unit vectors
        normalized = self._l2_normalize(np.array(raw, dtype="float32"))
        return normalized.tolist()
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate L2-normalized embedding for query.
        
        CHANGED: Added normalization step to match stored vectors.
        Query and document vectors now live in the same geometric space.
        """
        raw = await asyncio.to_thread(
            self.model.encode,
            query,
            convert_to_tensor=False
        )
        # NEW: Normalize to unit vector
        normalized = self._l2_normalize(
            np.array(raw, dtype="float32").reshape(1, -1)
        )
        return normalized[0].tolist()