"""Retrieval helpers for hybrid search pipelines."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from core.domain import ChunkSearchResult
from core.interfaces import (
    IDocumentRepository,
    IEmbeddingService,
    IMessageRepository,
    IReranker,
    IVectorStore,
)
from utils.arabic_text import has_keyword_match

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    retrieval_timeout_sec: float
    search_candidate_k: int
    search_score_threshold: float
    lexical_gate_enabled: bool
    lexical_min_keywords: int
    rerank_enabled: bool
    rerank_threshold: float
    rerank_gate_low: float
    rerank_gate_high: float
    rerank_candidate_cap: int
    line_min_chars: int


class Retriever:
    """Coordinate vector store access with safety guards."""

    def __init__(
        self,
        embedding_service: IEmbeddingService,
        chunk_store: IVectorStore,
        document_repo: IDocumentRepository,
        message_repo: IMessageRepository,
        config: RetrievalConfig,
        sentence_store: Optional[IVectorStore] = None,
        line_store: Optional[IVectorStore] = None,
        reranker: Optional[IReranker] = None,
    ) -> None:
        self._embedding = embedding_service
        self._chunk_store = chunk_store
        self._sentence_store = sentence_store
        self._line_store = line_store
        self._document_repo = document_repo
        self._message_repo = message_repo
        self._config = config
        self._reranker = reranker

    async def retrieve_sentences(
        self, query: str, top_k: int
    ) -> List[ChunkSearchResult]:
        if self._sentence_store is None:
            logger.debug("Sentence vector store not configured")
            return []

        embedding = await self._embedding.generate_query_embedding(query)
        try:
            results = await asyncio.wait_for(
                self._sentence_store.search(embedding, top_k=top_k),
                timeout=self._config.retrieval_timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Sentence retrieval timed out after %ss",
                self._config.retrieval_timeout_sec,
            )
            return []

        hits = [res for res in results if res and hasattr(res, "chunk")]
        return await self._filter_missing_documents(hits)

    async def retrieve_chunks(
        self, query: str, top_k: int
    ) -> List[ChunkSearchResult]:
        embedding = await self._embedding.generate_query_embedding(query)
        try:
            results = await asyncio.wait_for(
                self._chunk_store.search(embedding, top_k=top_k),
                timeout=self._config.retrieval_timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Chunk retrieval timed out after %ss",
                self._config.retrieval_timeout_sec,
            )
            return []

        hits = [res for res in results if res and hasattr(res, "chunk")]
        return await self._filter_missing_documents(hits)

    async def retrieve_lines(
        self, query: str, top_k: int
    ) -> List[ChunkSearchResult]:
        if self._line_store is None:
            return []

        embedding = await self._embedding.generate_query_embedding(query)
        try:
            hits = await asyncio.wait_for(
                self._line_store.search(embedding, top_k=top_k),
                timeout=self._config.retrieval_timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Line retrieval timed out after %ss",
                self._config.retrieval_timeout_sec,
            )
            return []

        min_chars = max(0, self._config.line_min_chars)
        return [hit for hit in hits if len(getattr(hit, "text", "") or "") >= min_chars]

    async def search_chunks(self, query: str, top_k: int) -> List[ChunkSearchResult]:
        try:
            query_embedding = await self._embedding.generate_query_embedding(query)
            candidate_k = max(top_k * 2, self._config.search_candidate_k)
            raw_results = await self._chunk_store.search(query_embedding, candidate_k)

            existing_ids: Set[str] = set()
            if raw_results:
                doc_ids: Set[str] = set()
                for res in raw_results:
                    chunk = getattr(res, "chunk", None)
                    if chunk is None:
                        continue
                    metadata_obj = getattr(chunk, "metadata", None)
                    metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
                    doc_id_value = getattr(chunk, "document_id", None) or metadata.get("document_id")
                    if isinstance(doc_id_value, str):
                        doc_ids.add(doc_id_value)
                if doc_ids:
                    existing_ids = await self._document_repo.exists_bulk(list(doc_ids))

            threshold = self._config.search_score_threshold
            filtered = []
            for res in raw_results:
                if not res or not getattr(res, "chunk", None):
                    continue
                metadata_obj = getattr(res.chunk, "metadata", None)
                metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
                doc_id = getattr(res.chunk, "document_id", None) or metadata.get("document_id")
                if doc_id in existing_ids and res.score >= threshold:
                    filtered.append(res)

            if not filtered:
                await self._message_repo.save_search_results(query, [])
                return []

            candidates = filtered
            if self._config.lexical_gate_enabled:
                with_keywords = [
                    res
                    for res in filtered
                    if has_keyword_match(query, res.chunk.content, self._config.lexical_min_keywords)
                ]
                if with_keywords:
                    candidates = with_keywords

            final_results = candidates
            if self._config.rerank_enabled and self._reranker:
                reranked = await self._maybe_rerank(query, candidates)
                rerank_threshold = self._config.rerank_threshold
                final_results = [res for res in reranked if res.score >= rerank_threshold][:top_k]
            else:
                candidates.sort(key=lambda res: res.score, reverse=True)
                final_results = candidates[:top_k]

            if final_results:
                scores = [res.score for res in final_results]
                logger.info(
                    "[SEARCH] raw=%s filtered=%s lexical=%s final=%s | scores=%.3f-%.3f",
                    len(raw_results),
                    len(filtered),
                    len(candidates),
                    len(final_results),
                    min(scores),
                    max(scores),
                )

            await self._message_repo.save_search_results(query, final_results)
            return final_results
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("[SEARCH] Failed: %s", exc, exc_info=True)
            return []

    async def _filter_missing_documents(
        self, hits: Sequence[ChunkSearchResult]
    ) -> List[ChunkSearchResult]:
        if not hits:
            return []

        doc_ids: Set[str] = set()
        for hit in hits:
            chunk = getattr(hit, "chunk", None)
            if chunk is None:
                continue
            metadata_obj = getattr(chunk, "metadata", None)
            metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
            doc_id = getattr(chunk, "document_id", None) or metadata.get("document_id")
            if isinstance(doc_id, str):
                doc_ids.add(doc_id)

        if not doc_ids:
            return list(hits)

        existing = await self._document_repo.exists_bulk(list(doc_ids))
        filtered: List[ChunkSearchResult] = []
        for hit in hits:
            chunk = getattr(hit, "chunk", None)
            if chunk is None:
                continue
            metadata_obj = getattr(chunk, "metadata", None)
            metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
            doc_id = getattr(chunk, "document_id", None) or metadata.get("document_id")
            if isinstance(doc_id, str) and doc_id in existing:
                filtered.append(hit)

        dropped = len(doc_ids) - len(existing)
        if dropped > 0:
            logger.info("[RETRIEVE] Dropped %s hits for deleted documents", dropped)

        return filtered

    async def _maybe_rerank(
        self, query: str, candidates: Sequence[ChunkSearchResult]
    ) -> List[ChunkSearchResult]:
        if not self._reranker:
            return list(candidates)

        scores = [candidate.score for candidate in candidates]
        if not self._should_rerank(scores):
            if scores:
                logger.info("[RERANK] Skipping (top score: %.3f)", max(scores))
            return list(candidates)

        cap = min(len(candidates), self._config.rerank_candidate_cap)
        capped = list(candidates)[:cap]
        reranked = await self._reranker.rerank(query, capped, top_k=cap)

        capped_ids = {c.chunk.id for c in capped if hasattr(c.chunk, "id")}
        remainder = [c for c in candidates if getattr(c.chunk, "id", None) not in capped_ids]

        logger.info("[RERANK] Reranked %s candidates", len(capped))
        return list(reranked) + remainder

    def _should_rerank(self, scores: Sequence[float]) -> bool:
        if not scores:
            return False
        top_score = max(scores)
        return self._config.rerank_gate_low <= top_score <= self._config.rerank_gate_high
