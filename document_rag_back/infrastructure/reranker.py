# infrastructure/rerankder.py

"""Cross-encoder reranker implementation."""
import asyncio
import logging
from typing import List

from sentence_transformers import CrossEncoder
from core.interfaces import IReranker
from core.domain import ChunkSearchResult
from config import settings

def _get_logger():
    from config import settings
    return logging.getLogger(settings.LOGGER_NAME)

logger = _get_logger()

class CrossEncoderReranker(IReranker):
    """Multilingual cross-encoder for semantic precision."""
    
    _model = None
    _model_name = None
    
    def __init__(self, model_name: str | None = None):  # CHANGE: Add | None
        # CHANGE: Provide default if None
        self.model_name = model_name or getattr(
            settings, 
            'RERANK_MODEL_NAME', 
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        
        if (CrossEncoderReranker._model is None or 
            CrossEncoderReranker._model_name != self.model_name):
            self._load_model()
        
        self.model = CrossEncoderReranker._model
    
    def _load_model(self) -> None:
        """Load cross-encoder with fallback to download."""
        try:
            logger.info(f"[RERANK] Loading {self.model_name}...")
            model = CrossEncoder(self.model_name)
            CrossEncoderReranker._model = model
            CrossEncoderReranker._model_name = self.model_name
            logger.info(f"[RERANK] Loaded successfully")
        except Exception as e:
            logger.error(f"[RERANK] Failed: {e}")
            raise RuntimeError(f"Could not load reranker: {e}")
    
    async def rerank(
        self, 
        query: str, 
        results: List[ChunkSearchResult], 
        top_k: int = 5
    ) -> List[ChunkSearchResult]:
        """Rerank results by semantic relevance."""
        if not results:
            return []
        
        # ADD THIS CHECK:
        if self.model is None:
            logger.warning("[RERANK] Model not loaded, returning original results")
            return results[:top_k]
        
        try:
            pairs = [(query, r.chunk.content) for r in results]
            scores = await asyncio.to_thread(self.model.predict, pairs)  # Line 59
            
            for result, score in zip(results, scores):
                result.score = float(score)
            
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"[RERANK] Scoring failed: {e}")
            return results[:top_k]