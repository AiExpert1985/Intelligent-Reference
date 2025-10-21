"""Helpers for sentence segmentation and mapping to line geometry."""

import re
from typing import Any, Dict, List, Sequence


_SENTENCE_BOUNDARY_PATTERN = re.compile(r"([.!?؟])\s+")


def split_into_sentences(text: str, min_chars: int = 30) -> List[Dict[str, Any]]:
    """Split ``text`` into sentence dictionaries preserving character offsets.

    Args:
        text: Full text to split.
        min_chars: Minimum length for a sentence to be kept.

    Returns:
        List of dictionaries with ``text``, ``start`` and ``end`` keys.
    """
    if not text or len(text) < min_chars:
        return []

    parts = _SENTENCE_BOUNDARY_PATTERN.split(text)
    sentences: List[Dict[str, Any]] = []
    cursor = 0
    idx = 0

    while idx < len(parts):
        raw = parts[idx]
        punct = ""

        if idx + 1 < len(parts) and parts[idx + 1] in ".!?؟":
            punct = parts[idx + 1]
            idx += 2
        else:
            idx += 1

        full_slice = raw + punct
        start = cursor
        end = start + len(full_slice)
        cursor = end

        cleaned = full_slice.strip()
        if len(cleaned) >= min_chars:
            sentences.append({
                "text": cleaned,
                "start": start,
                "end": end,
            })

    return sentences


def map_sentences_to_line_ids(
    sentences: Sequence[Dict[str, Any]],
    segment_lines: Sequence[Dict[str, Any]],
    separator: str = "\n",
) -> List[Dict[str, Any]]:
    """Attach ``line_ids`` to each sentence based on character overlap.

    Args:
        sentences: Sequence returned by :func:`split_into_sentences`.
        segment_lines: Lines belonging to the segment with ``line_id`` and ``text``.
        separator: Separator inserted between line texts when creating the segment text.

    Returns:
        The ``sentences`` list with ``line_ids`` populated.
    """
    if not sentences or not segment_lines:
        return list(sentences)

    offsets: List[Dict[str, Any]] = []
    cursor = 0
    last_index = len(segment_lines) - 1

    for i, line in enumerate(segment_lines):
        text = (line.get("text") or "")
        start = cursor
        end = start + len(text)
        offsets.append({"line_id": line.get("line_id"), "start": start, "end": end})
        cursor = end
        if i < last_index:
            cursor += len(separator)

    enriched: List[Dict[str, Any]] = []
    for sent in sentences:
        line_ids: List[str] = []
        sent_start = int(sent.get("start", 0) or 0)
        sent_end = int(sent.get("end", 0) or 0)

        for entry in offsets:
            lid = entry.get("line_id")
            if not lid:
                continue

            line_start = entry["start"]
            line_end = entry["end"]
            if sent_end <= line_start or sent_start >= line_end:
                continue
            line_ids.append(lid)

        enriched_sent = dict(sent)
        enriched_sent["line_ids"] = line_ids
        enriched.append(enriched_sent)

    return enriched

