"""Thin orchestrator wiring together the decomposed RAG modules."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from fastapi import Request, UploadFile

from api.schemas import DocumentsListItem, ProcessDocumentResponse
from core.domain import ChunkSearchResult, PageSearchResult
from core.interfaces import IRAGService
from services.debug_dump import SearchDebugDump
from services.highlighter import Highlighter
from services.ingestion import DocumentIngestion
from services.housekeeping import Housekeeping
from services.retriever import Retriever
from services.scorer import Scorer
from utils.highlight_token import sign

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    sentence_top_k: int
    sentence_fallback_top_k: int
    chunk_top_k: int
    highlight_preview_enabled: bool
    highlight_style_id: str
    highlight_telemetry_enabled: bool
    debug_text_enabled: bool
    debug_json_enabled: bool
    debug_max_items: int


class RAGService(IRAGService):
    """Implementation of IRAGService delegating to dedicated modules."""

    def __init__(
        self,
        retriever: Retriever,
        scorer: Scorer,
        highlighter: Highlighter,
        ingestion: DocumentIngestion,
        housekeeping: Housekeeping,
        search_debug_dump: Optional[SearchDebugDump],
        config: HybridSearchConfig,
    ) -> None:
        self._retriever = retriever
        self._scorer = scorer
        self._highlighter = highlighter
        self._ingestion = ingestion
        self._housekeeping = housekeeping
        self._debug_dump = search_debug_dump
        self._config = config

    async def process_document(self, file: UploadFile) -> ProcessDocumentResponse:
        return await self._ingestion.process_document(file)

    async def delete_document(self, document_id: str) -> bool:
        return await self._housekeeping.delete_document(document_id)

    async def clear_all(self) -> bool:
        return await self._housekeeping.clear_all()

    async def list_documents(self, request: Request) -> List[DocumentsListItem]:
        return await self._housekeeping.list_documents(request)

    async def get_document_with_path(self, document_id: str) -> Optional[Dict[str, str]]:
        return await self._housekeeping.get_document_with_path(document_id)

    async def get_status(self) -> Dict[str, Any]:
        return await self._housekeeping.get_status()

    async def search_chunks(self, query: str, top_k: int = 5) -> List[ChunkSearchResult]:
        return await self._retriever.search_chunks(query, top_k)

    async def search_pages(self, query: str, top_k: int = 5) -> List[PageSearchResult]:
        sentence_top_k = self._config.sentence_top_k or self._config.sentence_fallback_top_k
        chunk_top_k = self._config.chunk_top_k

        sentence_hits, chunk_hits = await asyncio.gather(
            self._retriever.retrieve_sentences(query, sentence_top_k),
            self._retriever.retrieve_chunks(query, chunk_top_k),
        )

        if not sentence_hits and not chunk_hits:
            logger.info("[HYBRID-SEARCH] No hits for query: %s", query[:60])
            return []

        ranked_pages = self._scorer.score(query, sentence_hits, chunk_hits)
        if not ranked_pages:
            return []

        results: List[PageSearchResult] = []
        for page_data in ranked_pages[:top_k]:
            doc_id = page_data["document_id"]
            page_idx = page_data["page_index"]
            if page_idx == 0:
                logger.warning("[SEARCH-PAGES] Normalized page_idx from 0 to 1 for doc=%s", doc_id)
                page_idx = 1

            document = await self._housekeeping.get_document(doc_id)
            if not document:
                continue

            metadata = document.metadata or {}
            page_images = metadata.get("page_image_paths", {}) or {}
            page_thumbnails = metadata.get("page_thumbnail_paths", {}) or {}
            image_key: Any = page_idx
            if str(page_idx) in page_images:
                image_key = str(page_idx)
            elif page_idx in page_images:
                image_key = page_idx
            if image_key in page_images:
                image_url = f"/page-image/{doc_id}/{page_idx}"
            else:
                image_url = ""

            thumb_key: Any = page_idx
            if str(page_idx) in page_thumbnails:
                thumb_key = str(page_idx)
            elif page_idx in page_thumbnails:
                thumb_key = page_idx
            if thumb_key in page_thumbnails:
                thumbnail_url = f"/page-image/{doc_id}/{page_idx}?size=thumbnail"
            else:
                thumbnail_url = ""

            chunk_evidence = page_data["evidence"]["chunks"]
            sentence_evidence = page_data["evidence"]["sentences"]
            highlights = []
            for hit in chunk_evidence[:3]:
                text = getattr(hit.chunk, "content", "") if getattr(hit, "chunk", None) else ""
                snippet = text[:150] + ("..." if len(text) > 150 else "")
                highlights.append(snippet)

            line_ids = self._highlighter.select_line_ids(sentence_evidence, chunk_evidence)
            page_data["highlights"] = line_ids

            if self._config.highlight_telemetry_enabled:
                source = "sentences" if sentence_evidence and line_ids else "chunk_lines" if line_ids else "none"
                logger.info(
                    "[HIGHLIGHT] doc=%s page=%s source=%s sentence_ev=%s chunk_ev=%s line_ids=%s",
                    doc_id,
                    page_idx,
                    source,
                    len(sentence_evidence),
                    len(chunk_evidence),
                    len(line_ids),
                )

            highlight_token = None
            if self._config.highlight_preview_enabled and line_ids:
                payload = {
                    "doc_id": doc_id,
                    "page_index": page_idx,
                    "style_id": self._config.highlight_style_id,
                    "line_ids": line_ids,
                }
                highlight_token = sign(payload, exp_seconds=600)

            results.append(
                PageSearchResult(
                    document_id=doc_id,
                    document_name=document.filename,
                    page_number=page_idx,
                    score=round(page_data["score"], 3),
                    chunk_count=len(chunk_evidence),
                    image_url=image_url,
                    thumbnail_url=thumbnail_url,
                    highlights=highlights,
                    download_url=f"/download/{doc_id}",
                    highlight_token=highlight_token,
                )
            )

        if self._debug_dump and (self._config.debug_text_enabled or self._config.debug_json_enabled):
            try:
                doc_cache = await self._build_doc_cache(ranked_pages)
                if self._config.debug_text_enabled:
                    await asyncio.to_thread(
                        self._debug_dump.write_search_text,
                        query,
                        sentence_hits,
                        chunk_hits,
                        ranked_pages,
                        doc_cache,
                        self._config.debug_max_items,
                    )
                if self._config.debug_json_enabled:
                    await asyncio.to_thread(
                        self._debug_dump.write_search_json,
                        query,
                        sentence_hits,
                        chunk_hits,
                        ranked_pages,
                        doc_cache,
                        self._config.debug_max_items,
                    )
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("[DEBUG-SEARCH] Dump failed: %s", exc)

        return results

    async def search(self, query: str, top_k: int = 5) -> List[PageSearchResult]:
        return await self.search_pages(query, top_k)

    async def _build_doc_cache(self, ranked_pages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        doc_ids = sorted({page["document_id"] for page in ranked_pages if page.get("document_id")})
        tasks = [self._housekeeping.get_document(doc_id) for doc_id in doc_ids]
        docs = await asyncio.gather(*tasks, return_exceptions=True)
        cache = {}
        for doc_id, doc in zip(doc_ids, docs):
            if isinstance(doc, Exception) or doc is None:
                continue
            cache[doc_id] = doc
        return cache
