"""Utility helpers for storing list metadata as scalar strings."""
from __future__ import annotations

from typing import Iterable, List, Optional, Union

METADATA_LIST_DELIMITER = "|"


def deserialize_metadata_list(value: Optional[str]) -> List[str]:
    """Split a pipe-delimited metadata string back into a list of strings."""
    if not value:
        return []

    return [part for part in value.split(METADATA_LIST_DELIMITER) if part]


def normalize_metadata_list(
    value: Optional[Union[str, Iterable[Union[str, int]]]]
) -> List[str]:
    """Return a list of strings regardless of how the metadata was stored."""
    if value is None:
        return []

    if isinstance(value, str):
        return deserialize_metadata_list(value)

    cleaned: List[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def serialize_metadata_list(values: Optional[Iterable[Union[str, int]]]) -> str:
    """Join iterable metadata into a pipe-delimited string acceptable to vector stores."""
    normalized = normalize_metadata_list(values)
    if not normalized:
        return ""
    return METADATA_LIST_DELIMITER.join(normalized)
