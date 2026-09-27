"""Vectorized match selection replicating DecisionCalibrator semantics."""

from typing import Tuple

import numpy as np


def topk_per_group_mask(group: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask keeping the top-k scoring rows (by score desc) per group."""
    n = group.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((-score, group))
    g = group[order]
    starts_mask = np.empty(g.shape[0], dtype=bool)
    starts_mask[0] = True
    np.not_equal(g[1:], g[:-1], out=starts_mask[1:])
    starts = np.flatnonzero(starts_mask)
    counts = np.diff(np.append(starts, g.shape[0]))
    rank = np.arange(g.shape[0]) - np.repeat(starts, counts)
    keep = rank < k
    mask = np.zeros(n, dtype=bool)
    mask[order[keep]] = True
    return mask


def select_matches_mask(
    group: np.ndarray,
    score: np.ndarray,
    threshold: float,
    confidence_gap: float = 0.15,
    max_matches: int = 10,
) -> np.ndarray:
    """Threshold + confidence-gap + top-N selection, fully vectorized.

    Mirrors DecisionCalibrator.select_matches: an entity matches nothing unless its
    best candidate passes ``threshold``; accepted extras must score within
    ``confidence_gap`` of that best score, and at most ``max_matches`` are kept.
    """
    n = group.shape[0]
    mask = np.zeros(n, dtype=bool)
    keep = score >= threshold
    if n == 0 or not keep.any():
        return mask

    kept_pos = np.flatnonzero(keep)
    g = group[kept_pos]
    s = score[kept_pos]

    order = np.lexsort((-s, g))
    g_sorted = g[order]
    s_sorted = s[order]

    starts_mask = np.empty(g_sorted.shape[0], dtype=bool)
    starts_mask[0] = True
    np.not_equal(g_sorted[1:], g_sorted[:-1], out=starts_mask[1:])
    starts = np.flatnonzero(starts_mask)
    counts = np.diff(np.append(starts, g_sorted.shape[0]))

    rank = np.arange(g_sorted.shape[0]) - np.repeat(starts, counts)
    group_top = np.repeat(s_sorted[starts], counts)
    floor = np.maximum(threshold, group_top - confidence_gap)

    ok = (rank < max_matches) & (s_sorted >= floor)
    mask[kept_pos[np.asarray(order)[ok]]] = True
    return mask


def group_ids_from_index(owner: np.ndarray) -> Tuple[np.ndarray, int]:
    """Map arbitrary group labels to dense 0..G-1 codes (owner must be sorted-ish)."""
    uniq, codes = np.unique(owner, return_inverse=True)
    return codes, int(uniq.shape[0])
