"""String distance and similarity metrics."""

from difflib import SequenceMatcher
from typing import Set


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Calculate the Levenshtein-like similarity ratio between two strings [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1, s2).ratio()


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Calculate character n-gram Jaccard similarity [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0

    ngrams1: Set[str] = {s1[i : i + n] for i in range(len(s1) - n + 1)}
    ngrams2: Set[str] = {s2[i : i + n] for i in range(len(s2) - n + 1)}

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return float(intersection) / union if union > 0 else 0.0


def length_ratio(s1: str, s2: str) -> float:
    """Ratio of lengths of two strings min(len)/max(len)."""
    l1, l2 = len(s1), len(s2)
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return float(min(l1, l2)) / max(l1, l2)
