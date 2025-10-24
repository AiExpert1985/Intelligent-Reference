"""Page scoring helpers for hybrid retrieval."""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Sequence, TypedDict

from core.domain import ChunkSearchResult

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    alpha: float
    lam: float
    beta: float
    per_page_sentence_cap: int
    max_total_hits: int
    telemetry_enabled: bool = False


class PageScoreComponents(TypedDict):
    S_sentences: float
    S_chunks: float


class PageScoreEvidence(TypedDict):
    sentences: List[ChunkSearchResult]
    chunks: List[ChunkSearchResult]


class RankedPageBase(TypedDict):
    document_id: str
    page_index: int
    score: float
    source: str
    components: PageScoreComponents
    evidence: PageScoreEvidence


class RankedPage(RankedPageBase, total=False):
    highlights: List[str]


class Scorer:
    """Fuse sentence and chunk evidence into ranked page results."""

    def __init__(self, config: ScoringConfig) -> None:
        self._config = config

    def score(
        self,
        query: str,
        sentence_hits: Sequence[ChunkSearchResult],
        chunk_hits: Sequence[ChunkSearchResult],
    ) -> List[RankedPage]:
        _ = query  # reserved for future telemetry hooks
        cap = max(1, self._config.max_total_hits)
        sentence_hits = list(sentence_hits[:cap])
        chunk_hits = list(chunk_hits[:cap])

        self._zscore_normalize(sentence_hits, "score")
        self._zscore_normalize(chunk_hits, "score")

        for hit in sentence_hits:
            bounded = self._sigmoid(getattr(hit, "score_z", 0.0))
            setattr(hit, "score_bounded", bounded)
        for hit in chunk_hits:
            bounded = self._sigmoid(getattr(hit, "score_z", 0.0))
            setattr(hit, "score_bounded", bounded)

        sentences_by_page = defaultdict(list)
        for hit in sentence_hits:
            doc_id = getattr(hit, "document_id", None)
            page_idx = getattr(hit, "page_index", None)

            chunk = getattr(hit, "chunk", None)
            if chunk is not None:
                metadata = chunk.metadata or {}
                doc_id = doc_id or getattr(chunk, "document_id", None) or metadata.get("document_id")
                if page_idx is None:
                    page_idx = (
                        metadata.get("page_index")
                        or metadata.get("page")
                        or metadata.get("page_number")
                    )

            if not isinstance(doc_id, str) or page_idx is None:
                continue

            try:
                page_idx_int = int(page_idx)
            except (TypeError, ValueError):
                continue

            sentences_by_page[(doc_id, page_idx_int)].append(hit)

        chunks_by_page = defaultdict(list)
        for hit in chunk_hits:
            chunk = getattr(hit, "chunk", None)
            if chunk is None:
                continue
            metadata = chunk.metadata or {}
            doc_id = getattr(chunk, "document_id", None) or metadata.get("document_id")
            if not isinstance(doc_id, str):
                continue
            page_idx = metadata.get("page")
            try:
                page_idx_int = int(page_idx)
            except (TypeError, ValueError):
                continue
            chunks_by_page[(doc_id, page_idx_int)].append(hit)

        pages = set(sentences_by_page.keys()) | set(chunks_by_page.keys())

        ranked: List[RankedPage] = []
        for doc_id, page_idx in pages:
            sentence_scores = sorted(
                (getattr(hit, "score_bounded", 0.0) for hit in sentences_by_page[(doc_id, page_idx)]),
                reverse=True,
            )
            chunk_scores = sorted(
                (getattr(hit, "score_bounded", 0.0) for hit in chunks_by_page[(doc_id, page_idx)]),
                reverse=True,
            )

            if sentence_scores:
                S_sentences = self._saturating_sum(
                    sentence_scores,
                    self._config.alpha,
                    self._config.lam,
                    max(1, self._config.per_page_sentence_cap),
                )
            else:
                S_sentences = 0.0

            S_chunks = chunk_scores[0] if chunk_scores else 0.0

            if S_sentences == 0.0 and S_chunks == 0.0:
                continue
            if S_sentences == 0.0:
                total_score = S_chunks
                source = "chunks_only"
            elif S_chunks == 0.0:
                total_score = S_sentences
                source = "sentences_only"
            else:
                total_score = self._config.beta * S_sentences + (1.0 - self._config.beta) * S_chunks
                source = "hybrid"

            ranked.append(
                {
                    "document_id": doc_id,
                    "page_index": page_idx,
                    "score": total_score,
                    "source": source,
                    "components": {
                        "S_sentences": S_sentences,
                        "S_chunks": S_chunks,
                    },
                    "evidence": {
                        "sentences": sentences_by_page[(doc_id, page_idx)],
                        "chunks": chunks_by_page[(doc_id, page_idx)],
                    },
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)

        if self._config.telemetry_enabled and ranked:
            top = ranked[0]
            logger.info(
                "[HYBRID] doc=%s page=%s score=%.3f source=%s sentences=%s chunks=%s",
                top["document_id"],
                top["page_index"],
                top["score"],
                top["source"],
                len(top["evidence"]["sentences"]),
                len(top["evidence"]["chunks"]),
            )

        return ranked

    @staticmethod
    def _zscore_normalize(items: Iterable[object], score_attr: str) -> None:
        items = list(items)
        if not items:
            return
        def _extract_value(obj: object) -> float:
            raw = getattr(obj, score_attr, 0.0)
            if isinstance(raw, (int, float)):
                return float(raw)
            return 0.0

        values = [_extract_value(item) for item in items]
        if len(values) < 2:
            for item, value in zip(items, values):
                setattr(item, "score_z", value)
            return
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(max(variance, 1e-12))
        for item, value in zip(items, values):
            setattr(item, "score_z", (value - mean) / std)

    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    @staticmethod
    def _saturating_sum(scores_desc: Sequence[float], alpha: float, lam: float, cap: int) -> float:
        if not scores_desc:
            return 0.0
        cap = max(1, cap)
        s1 = scores_desc[0]
        tail = scores_desc[1:cap]
        sat = sum(score * (alpha ** idx) for idx, score in enumerate(tail, start=1))
        return s1 + lam * sat
