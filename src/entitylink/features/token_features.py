"""Token-level comparison features."""

import re
from typing import List, Set, Tuple


def token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute Jaccard similarity between two token lists."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection) / union if union > 0 else 0.0


def token_containment(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute maximum directional token containment |set1 & set2| / min(|set1|, |set2|)."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    min_size = min(len(set1), len(set2))
    return float(intersection) / min_size if min_size > 0 else 0.0


def extract_numbers(text: str) -> Set[str]:
    """Extract numeric sequences from text (useful for street numbers and postal codes)."""
    return set(re.findall(r"\b\d+\b", text))


def number_overlap(text1: str, text2: str) -> Tuple[float, int]:
    """Calculate the overlap ratio of numeric tokens and count of shared numbers."""
    nums1, nums2 = extract_numbers(text1), extract_numbers(text2)
    if not nums1 and not nums2:
        return 1.0, 0
    if not nums1 or not nums2:
        return 0.0, 0
    shared = len(nums1 & nums2)
    total = len(nums1 | nums2)
    return float(shared) / total, shared
