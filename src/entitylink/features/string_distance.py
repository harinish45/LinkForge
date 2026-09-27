"""String distance and similarity metrics."""

from typing import Set


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate exact Levenshtein edit distance between two strings using O(min(m, n)) space."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    current_row = list(range(len1 + 1))
    for j, c2 in enumerate(s2, 1):
        previous_row = current_row
        current_row = [j] + [0] * len1
        for i, c1 in enumerate(s1, 1):
            add = previous_row[i] + 1
            delete = current_row[i - 1] + 1
            change = previous_row[i - 1] + (0 if c1 == c2 else 1)
            current_row[i] = min(add, delete, change)

    return current_row[len1]


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Calculate exact normalized Levenshtein similarity ratio [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    dist = levenshtein_distance(s1, s2)
    return 1.0 - (dist / max_len)


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """Calculate Jaro-Winkler similarity score [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    sim_j = (
        (matches / len1) +
        (matches / len2) +
        ((matches - transpositions / 2.0) / matches)
    ) / 3.0

    # Prefix match up to max_l characters
    prefix = 0
    for i in range(min(len1, len2, max_l)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return sim_j + prefix * p * (1.0 - sim_j)


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

