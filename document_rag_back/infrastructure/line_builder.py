# infrastructure/line_builder.py
"""
Rebuild text from OCR bounding boxes with proper line ordering.

Handles RTL (Arabic) and LTR (English) text by:
1. Clustering tokens into lines based on vertical position
2. Ordering tokens within each line (RTL: right-to-left, LTR: left-to-right)
3. Preserving blank lines between paragraphs
"""
from typing import List, Tuple
import re

BBox = List[List[int]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

# Bounding box utilities
def _cx(b: BBox) -> float:
    """Calculate center x-coordinate of bbox."""
    return sum(p[0] for p in b) / 4.0

def _cy(b: BBox) -> float:
    """Calculate center y-coordinate of bbox."""
    return sum(p[1] for p in b) / 4.0

def _h(b: BBox) -> float:
    """Calculate height of bbox."""
    ys = [p[1] for p in b]
    return max(ys) - min(ys)

def _is_rtl_text(tokens: List[str]) -> bool:
    """
    Determine if tokens represent RTL (Arabic) text.
    
    Uses character-level counting: if Arabic characters dominate,
    treat as RTL. This handles mixed content better than token-level
    heuristics.
    """
    s = "".join(tokens)
    ar_count = sum(1 for ch in s if '\u0600' <= ch <= '\u06FF')
    latin_count = sum(1 for ch in s if ('A' <= ch <= 'Z') or ('a' <= ch <= 'z'))
    
    # RTL if Arabic >= Latin and has at least some Arabic
    return ar_count >= max(latin_count, 1) and ar_count > 0

def rebuild_text_from_boxes(
    items: List[Tuple[BBox, str, float]],
    *,
    alpha_line: float = 0.7,   # Line clustering tolerance (× median height)
    beta_blank: float = 1.6    # Blank line gap threshold (× median height)
) -> str:
    """
    Rebuild text from OCR bounding boxes with proper line ordering.
    
    Args:
        items: List of (bbox, text, confidence) tuples from OCR
        alpha_line: Vertical tolerance for same-line clustering
        beta_blank: Vertical gap threshold for blank lines
    
    Returns:
        Reconstructed text with proper line breaks and ordering
    
    Process:
        1. Sort tokens by vertical position (y-coordinate)
        2. Cluster into lines based on vertical proximity
        3. Within each line, order tokens by direction (RTL/LTR)
        4. Detect and preserve blank lines between paragraphs
    """
    if not items:
        return ""

    # Sort by y, then x for stable processing
    items_sorted = sorted(items, key=lambda it: (_cy(it[0]), _cx(it[0])))
    
    # Calculate median height for thresholding
    heights = [_h(b) for b, _, _ in items_sorted if _h(b) > 0]
    median_height = sorted(heights)[len(heights) // 2] if heights else 16.0
    if median_height <= 0:
        median_height = 16.0

    lines: List[List[Tuple[BBox, str, float]]] = []
    current_line: List[Tuple[BBox, str, float]] = []
    prev_y = None
    prev_bottom = None

    def flush_line():
        """Flush current line with proper token ordering."""
        nonlocal current_line
        if not current_line:
            return
        
        # Determine direction from tokens
        tokens = [text for _, text, _ in current_line if text]
        is_rtl = _is_rtl_text(tokens)
        
        # Sort tokens: RTL → descending x, LTR → ascending x
        current_line.sort(key=lambda it: _cx(it[0]), reverse=is_rtl)
        lines.append(current_line.copy())
        current_line.clear()

    # Cluster tokens into lines
    for bbox, text, conf in items_sorted:
        if not text:
            continue
        
        cy = _cy(bbox)
        top = min(p[1] for p in bbox)
        bottom = max(p[1] for p in bbox)

        # Check if token belongs to current line
        if prev_y is None or abs(cy - prev_y) <= alpha_line * median_height:
            current_line.append((bbox, text, conf))
        else:
            # New line detected
            gap = top - (prev_bottom if prev_bottom is not None else top)
            flush_line()
            
            # Insert blank line marker if gap is large
            if gap > beta_blank * median_height:
                lines.append([])  # Blank line marker
            
            current_line.append((bbox, text, conf))
        
        prev_y = cy
        prev_bottom = bottom

    flush_line()

    # Build final text
    out_lines: List[str] = []
    for line in lines:
        if not line:
            out_lines.append("")  # Blank line
        else:
            out_lines.append(" ".join(token for _, token, _ in line))
    
    return "\n".join(out_lines).strip()