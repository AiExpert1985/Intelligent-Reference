"""Utility writers for debug text and JSON dumps."""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.domain import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class DebugDumpConfig:
    ocr_dir: str = "debug_ocr"
    search_dir: str = "debug_search"
    search_max_items: int = 10


class SearchDebugDump:
    """Encapsulate filesystem writes for troubleshooting artifacts."""

    def __init__(self, config: DebugDumpConfig) -> None:
        self._config = config

    # ----- OCR helpers -----
    def save_chunks(self, chunks: Sequence[DocumentChunk], filename: str) -> None:
        if not chunks:
            return
        directory = Path(self._config.ocr_dir)
        directory.mkdir(exist_ok=True)
        out = directory / f"{filename}.txt"
        with out.open("w", encoding="utf-8") as fh:
            fh.write(f"Total chunks: {len(chunks)}\n")
            fh.write("=" * 80 + "\n\n")
            for index, chunk in enumerate(chunks, 1):
                fh.write(f"CHUNK {index}\n")
                fh.write(f"Page: {chunk.metadata.get('page', 'N/A')}\n")
                fh.write("-" * 80 + "\n")
                fh.write(chunk.content)
                fh.write("\n\n" + "=" * 80 + "\n\n")
        logger.info("Debug: Saved %d chunks to %s", len(chunks), out)

    def save_sentences(self, sentences: Sequence[DocumentChunk], filename: str) -> None:
        if not sentences:
            return
        directory = Path(self._config.ocr_dir)
        directory.mkdir(exist_ok=True)
        out = directory / f"{filename}_sentences.txt"
        with out.open("w", encoding="utf-8") as fh:
            fh.write(f"Total sentences: {len(sentences)}\n")
            fh.write("=" * 80 + "\n\n")
            for index, sentence in enumerate(sentences, 1):
                metadata = sentence.metadata or {}
                fh.write(f"SENTENCE {index}\n")
                fh.write(f"Page: {metadata.get('page_index', 'N/A')}\n")
                fh.write(f"Sentence ID: {metadata.get('sentence_id', 'N/A')}\n")
                fh.write(f"Line IDs: {metadata.get('line_ids', [])}\n")
                fh.write("-" * 80 + "\n")
                fh.write(sentence.content.replace("\n", " ").strip())
                fh.write("\n\n" + "=" * 80 + "\n\n")
        logger.info("Debug: Saved %d sentences to %s", len(sentences), out)

    def save_lines(self, lines_by_page: Dict[str, List[Dict[str, Any]]], filename: str) -> None:
        if not lines_by_page:
            return
        total_lines = sum(len(lines or []) for lines in lines_by_page.values())
        if total_lines == 0:
            return
        directory = Path(self._config.ocr_dir)
        directory.mkdir(exist_ok=True)
        out = directory / f"{filename}_lines.txt"

        def _page_sort(page_key: str) -> Any:
            try:
                return int(page_key), page_key
            except (TypeError, ValueError):
                return 0, page_key

        with out.open("w", encoding="utf-8") as fh:
            fh.write(f"Total lines: {total_lines}\n")
            fh.write("=" * 80 + "\n\n")
            for page_key in sorted(lines_by_page.keys(), key=_page_sort):
                page_lines = lines_by_page.get(page_key) or []
                if not page_lines:
                    continue
                fh.write(f"PAGE {page_key}\n")
                fh.write("-" * 80 + "\n")
                for index, line in enumerate(page_lines, 1):
                    line_id = line.get("line_id", "N/A")
                    bbox = line.get("bbox", [])
                    text = (line.get("text", "") or "").replace("\n", " ").strip()
                    fh.write(f"Line {index} (ID: {line_id})\n")
                    fh.write(f"BBox: {bbox}\n")
                    fh.write(text + "\n\n")
                fh.write("=" * 80 + "\n\n")
        logger.info("Debug: Saved %d lines to %s", total_lines, out)

    # ----- Search dumps -----
    def write_search_text(
        self,
        query: str,
        sentence_hits: Sequence[Any],
        chunk_hits: Sequence[Any],
        ranked_pages: Sequence[Dict[str, Any]],
        doc_cache: Dict[str, Any],
        max_items: Optional[int] = None,
    ) -> None:
        directory = Path(self._config.search_dir)
        directory.mkdir(exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = directory / f"search_{ts}.txt"
        max_list = max_items or self._config.search_max_items

        def _snip(text: str, limit: int = 200) -> str:
            text = text.replace("\n", " ").strip()
            return text if len(text) <= limit else text[: limit - 1] + "…"

        with out.open("w", encoding="utf-8") as fh:
            fh.write("SEARCH DEBUG DUMP\n")
            fh.write(f"Timestamp: {ts}\n")
            fh.write(f"Query: {query}\n")
            fh.write("=" * 80 + "\n\n")

            fh.write(f"SENTENCE HITS (total={len(sentence_hits)}, showing top {max_list}):\n")
            fh.write("-" * 80 + "\n\n")
            for index, hit in enumerate(sentence_hits[:max_list], 1):
                chunk = getattr(hit, "chunk", None)
                metadata = getattr(chunk, "metadata", {}) if chunk else {}
                doc_id = getattr(chunk, "document_id", None) or metadata.get("document_id")
                page = metadata.get("page_index") or metadata.get("page")
                fh.write(f"[{index}] Sentence Hit\n")
                fh.write(f"    Document: {doc_id}\n")
                fh.write(f"    Page: {page}\n")
                fh.write(f"    Score (raw): {getattr(hit, 'score', 0.0):.4f}\n")
                fh.write(f"    Score (bounded): {getattr(hit, 'score_bounded', 0.0)}\n")
                fh.write(f"    Line IDs: {metadata.get('line_ids', [])}\n")
                fh.write(f"    Text: {_snip(getattr(chunk, 'content', ''))}\n\n")

            fh.write(f"CHUNK HITS (total={len(chunk_hits)}, showing top {max_list}):\n")
            fh.write("-" * 80 + "\n\n")
            for index, hit in enumerate(chunk_hits[:max_list], 1):
                chunk = getattr(hit, "chunk", None)
                metadata = getattr(chunk, "metadata", {}) if chunk else {}
                doc_id = getattr(chunk, "document_id", None) or metadata.get("document_id")
                page = metadata.get("page")
                fh.write(f"[{index}] Chunk Hit\n")
                fh.write(f"    Document: {doc_id}\n")
                fh.write(f"    Page: {page}\n")
                fh.write(f"    Score (raw): {getattr(hit, 'score', 0.0):.4f}\n")
                fh.write(f"    Score (bounded): {getattr(hit, 'score_bounded', 0.0)}\n")
                fh.write(f"    Line IDs: {metadata.get('line_ids', [])}\n")
                fh.write(f"    Text: {_snip(getattr(chunk, 'content', ''))}\n\n")

            fh.write(f"RANKED PAGES (total={len(ranked_pages)}, showing top {max_list}):\n")
            fh.write("-" * 80 + "\n\n")
            for index, page in enumerate(ranked_pages[:max_list], 1):
                doc_id = page.get("document_id")
                page_index = page.get("page_index")
                components = page.get("components", {})
                fh.write(f"[{index}] Page Rank\n")
                fh.write(f"    Document: {doc_id}\n")
                fh.write(f"    Page: {page_index}\n")
                fh.write(f"    Score (total): {page.get('score', 0.0):.4f}\n")
                fh.write(f"    Score (sentences): {components.get('S_sentences', 0.0):.4f}\n")
                fh.write(f"    Score (chunks): {components.get('S_chunks', 0.0):.4f}\n")
                fh.write(f"    Source: {page.get('source')}\n")

                sentence_ev = page.get("evidence", {}).get("sentences", [])
                chunk_ev = page.get("evidence", {}).get("chunks", [])

                fh.write(f"    Sentence evidence: {len(sentence_ev)} hits\n")
                fh.write(f"    Chunk evidence: {len(chunk_ev)} hits\n")

                doc = doc_cache.get(doc_id)
                if doc and getattr(doc, "metadata", None):
                    page_lines = (doc.metadata.get("lines") or {}).get(str(page_index), [])
                    by_id = {ln.get("line_id"): ln.get("text", "") for ln in page_lines}
                    selected = page.get("highlights", []) or []
                    if selected:
                        fh.write("    Highlighted Line Texts:\n")
                        for line_id in selected:
                            fh.write(f"        {line_id}: {_snip(by_id.get(line_id, ''))}\n")

                fh.write("\n")

        logger.info("[DEBUG-SEARCH] Wrote text dump → %s", out)

    def write_search_json(
        self,
        query: str,
        sentence_hits: Sequence[Any],
        chunk_hits: Sequence[Any],
        ranked_pages: Sequence[Dict[str, Any]],
        doc_cache: Dict[str, Any],
        max_items: Optional[int] = None,
    ) -> None:
        directory = Path(self._config.search_dir)
        directory.mkdir(exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = re.sub(r"[^\w\-\u0600-\u06FF]+", "_", query).strip("_")[:80] or "query"
        out = directory / f"search_{safe_query}_{ts}.json"
        max_list = max_items or self._config.search_max_items

        def _serialize_hits(hits: Sequence[Any]) -> List[Dict[str, Any]]:
            serialized: List[Dict[str, Any]] = []
            for hit in hits[:max_list]:
                chunk = getattr(hit, "chunk", None)
                metadata = getattr(chunk, "metadata", {}) if chunk else {}
                serialized.append(
                    {
                        "score": getattr(hit, "score", 0.0),
                        "score_bounded": getattr(hit, "score_bounded", 0.0),
                        "document_id": getattr(chunk, "document_id", None) or metadata.get("document_id"),
                        "page": metadata.get("page") or metadata.get("page_index"),
                        "line_ids": metadata.get("line_ids", []),
                        "text": getattr(chunk, "content", ""),
                    }
                )
            return serialized

        payload = {
            "query": query,
            "sentence_hits": _serialize_hits(sentence_hits),
            "chunk_hits": _serialize_hits(chunk_hits),
            "ranked_pages": [],
        }

        for page in ranked_pages[:max_list]:
            doc_id = page.get("document_id")
            page_index = page.get("page_index")
            doc = doc_cache.get(doc_id)
            lines = []
            if doc and getattr(doc, "metadata", None):
                page_lines = (doc.metadata.get("lines") or {}).get(str(page_index), [])
                lines = [
                    {
                        "line_id": ln.get("line_id"),
                        "text": ln.get("text", ""),
                    }
                    for ln in page_lines
                ]

            payload["ranked_pages"].append(
                {
                    "document_id": doc_id,
                    "page_index": page_index,
                    "score": page.get("score"),
                    "components": page.get("components", {}),
                    "source": page.get("source"),
                    "highlights": page.get("highlights", []),
                    "lines": lines,
                }
            )

        with out.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        logger.info("[DEBUG-SEARCH] Wrote JSON dump → %s", out)
