"""Deterministic highlight selection for hybrid search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, cast

from core.domain import ChunkSearchResult
from utils.metadata import normalize_metadata_list


@dataclass
class HighlighterConfig:
    score_threshold: float
    max_regions: int


class Highlighter:
    """Select the most relevant line identifiers for highlighting."""

    def __init__(self, config: HighlighterConfig) -> None:
        self._config = config

    def select_line_ids(
        self,
        sentence_evidence: Sequence[ChunkSearchResult],
        chunk_evidence: Sequence[ChunkSearchResult],
    ) -> List[str]:
        line_ids = self._extract_from_sentences(sentence_evidence)
        if line_ids:
            return line_ids
        return self._extract_from_chunks(chunk_evidence)

    def _extract_from_sentences(
        self, sentence_hits: Sequence[ChunkSearchResult]
    ) -> List[str]:
        if not sentence_hits:
            return []

        threshold = self._config.score_threshold
        line_scores: Dict[str, float] = {}

        for hit in sentence_hits:
            score = float(getattr(hit, "score", 0.0) or 0.0)
            if score < threshold:
                continue

            line_ids: List[str] = []
            chunk = getattr(hit, "chunk", None)
            if chunk is not None and getattr(chunk, "metadata", None):
                line_ids = normalize_metadata_list(chunk.metadata.get("line_ids"))
            else:
                extra_hit = cast(Any, hit)
                metadata = getattr(extra_hit, "metadata", None)
                if isinstance(metadata, dict):
                    line_ids = normalize_metadata_list(metadata.get("line_ids"))
                elif hasattr(extra_hit, "line_ids"):
                    line_ids = normalize_metadata_list(getattr(extra_hit, "line_ids"))

            for line_id in line_ids:
                best = line_scores.get(line_id)
                if best is None or score > best:
                    line_scores[line_id] = score

        if not line_scores:
            return []

        ordered = sorted(line_scores.items(), key=lambda item: (-item[1], item[0]))
        return [line_id for line_id, _ in ordered[: self._config.max_regions]]

    def _extract_from_chunks(
        self, chunk_hits: Sequence[ChunkSearchResult]
    ) -> List[str]:
        if not chunk_hits:
            return []

        threshold = self._config.score_threshold
        best_by_line: Dict[str, float] = {}

        for hit in chunk_hits:
            score = float(getattr(hit, "score", 0.0) or 0.0)
            if score < threshold:
                continue

            chunk = getattr(hit, "chunk", None)
            metadata: Dict[str, Any] = {}
            if chunk is not None:
                metadata_obj = getattr(chunk, "metadata", None)
                if isinstance(metadata_obj, dict):
                    metadata = metadata_obj
            for line_id in normalize_metadata_list(metadata.get("line_ids")):
                existing = best_by_line.get(line_id)
                if existing is None or score > existing:
                    best_by_line[line_id] = score

        if not best_by_line:
            return []

        ordered = sorted(best_by_line.items(), key=lambda item: (-item[1], item[0]))
        return [line_id for line_id, _ in ordered[: self._config.max_regions]]
