# utils/arabic_text.py

"""Robust Arabic text normalization for lexical filtering."""
import re
from typing import Set

ARABIC_STOPWORDS: Set[str] = {
    "في", "من", "إلى", "على", "عن", "مع", "عند",
    "و", "أو", "ثم", "أن", "إن", "لا", "ما",
    "هذا", "هذه", "ذلك", "تلك", "هو", "هي",
}

def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for keyword matching.
    Handles diacritics, character variants, and morphological variations.
    """
    if not text:
        return ""
    
    # Remove diacritics (harakat)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # Normalize Alif variants: أ/إ/آ → ا
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    
    # Normalize other variants
    text = text.replace('ى', 'ي')  # Alef Maksura
    text = text.replace('ة', 'ه')  # Ta Marbuta
    text = text.replace('ئ', 'ي')  # Hamza on Ya
    text = text.replace('ؤ', 'و')  # Hamza on Waw
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()

def extract_keywords(text: str) -> Set[str]:
    """Extract significant keywords (minus stopwords)."""
    normalized = normalize_arabic(text)
    tokens = normalized.split()
    return {
        token for token in tokens
        if len(token) >= 2 and token not in ARABIC_STOPWORDS
    }

def has_keyword_match(query: str, text: str, min_matches: int = 1) -> bool:
    """
    Check if text contains minimum query keywords.
    Fast pre-filter before neural reranking.
    """
    query_keywords = extract_keywords(query)
    text_keywords = extract_keywords(text)
    
    if not query_keywords:
        return True
    
    overlap = query_keywords.intersection(text_keywords)
    return len(overlap) >= min_matches