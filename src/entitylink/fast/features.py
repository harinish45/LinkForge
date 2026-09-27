"""Vectorized pair features computed from encoded payloads (no Python loops)."""

from typing import Dict, List

import numpy as np

FAST_FEATURE_NAMES: List[str] = [
    "name_exact",
    "name_tok_jaccard",
    "name_tok_containment",
    "name_char3_jaccard",
    "name_len_ratio",
    "addr_exact",
    "addr_tok_jaccard",
    "addr_num_jaccard",
    "addr_num_shared",
    "addr_len_ratio",
    "postal_match",
    "country_match",
    "combined_geom_sim",
    "strong_agreement",
    "strong_disagreement",
]


def _pc(v: np.ndarray) -> np.ndarray:
    """Population count over uint64 lanes -> float64."""
    return np.bitwise_count(v).sum(axis=1).astype(np.float64)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)


def gather_pair_payload(store, idx: np.ndarray) -> Dict[str, np.ndarray]:
    """Gather scalar + bitset payload for row indices into flat arrays."""
    out: Dict[str, np.ndarray] = {}
    for name in ("name_hash", "addr_hash", "postal", "country", "name_len", "addr_len",
                 "n_name_tok", "n_addr_tok", "n_addr_num"):
        out[name] = store.scalars[name][idx]
    for name in ("name_tok_bits", "name_char_bits", "addr_tok_bits", "addr_num_bits"):
        out[name] = store.bits[name][idx]
    return out


def cheap_score(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> np.ndarray:
    """Cheap ranking score used to cap candidates per entity before full scoring."""
    name_exact = (a["name_hash"] == b["name_hash"]) & (a["name_hash"] != 0)
    addr_exact = (a["addr_hash"] == b["addr_hash"]) & (a["addr_hash"] != 0)
    tok_and = _pc(a["name_tok_bits"] & b["name_tok_bits"])
    tok_or = _pc(a["name_tok_bits"] | b["name_tok_bits"])
    char_and = _pc(a["name_char_bits"] & b["name_char_bits"])
    char_or = _pc(a["name_char_bits"] | b["name_char_bits"])
    atok_and = _pc(a["addr_tok_bits"] & b["addr_tok_bits"])
    atok_or = _pc(a["addr_tok_bits"] | b["addr_tok_bits"])
    postal = (a["postal"] != 0) & (a["postal"] == b["postal"])
    country = (a["country"] != 0) & (a["country"] == b["country"])
    return (name_exact.astype(np.float64) + _ratio(tok_and, tok_or)
            + _ratio(char_and, char_or) + addr_exact.astype(np.float64)
            + 0.5 * _ratio(atok_and, atok_or) + 0.3 * postal + 0.2 * country)


def extract_fast_features(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute the fast feature matrix (n_pairs, len(FAST_FEATURE_NAMES))."""
    n = a["name_hash"].shape[0]
    X = np.zeros((n, len(FAST_FEATURE_NAMES)), dtype=np.float64)

    name_exact = ((a["name_hash"] == b["name_hash"]) & (a["name_hash"] != 0)).astype(np.float64)
    addr_exact = ((a["addr_hash"] == b["addr_hash"]) & (a["addr_hash"] != 0)).astype(np.float64)

    nt_and = _pc(a["name_tok_bits"] & b["name_tok_bits"])
    nt_or = _pc(a["name_tok_bits"] | b["name_tok_bits"])
    name_tok_jac = _ratio(nt_and, nt_or)
    min_tok = np.minimum(a["n_name_tok"], b["n_name_tok"]).astype(np.float64)
    name_tok_con = _ratio(nt_and, min_tok)

    nc_and = _pc(a["name_char_bits"] & b["name_char_bits"])
    nc_or = _pc(a["name_char_bits"] | b["name_char_bits"])
    name_char_jac = _ratio(nc_and, nc_or)

    name_ratio = _ratio(np.minimum(a["name_len"], b["name_len"]).astype(np.float64),
                        np.maximum(a["name_len"], b["name_len"]).astype(np.float64))

    at_and = _pc(a["addr_tok_bits"] & b["addr_tok_bits"])
    at_or = _pc(a["addr_tok_bits"] | b["addr_tok_bits"])
    addr_tok_jac = _ratio(at_and, at_or)

    an_and = _pc(a["addr_num_bits"] & b["addr_num_bits"])
    an_or = _pc(a["addr_num_bits"] | b["addr_num_bits"])
    addr_num_jac = _ratio(an_and, an_or)

    addr_ratio = _ratio(np.minimum(a["addr_len"], b["addr_len"]).astype(np.float64),
                        np.maximum(a["addr_len"], b["addr_len"]).astype(np.float64))

    postal_match = ((a["postal"] != 0) & (a["postal"] == b["postal"])).astype(np.float64)
    country_match = ((a["country"] != 0) & (a["country"] == b["country"])).astype(np.float64)

    name_sim = (name_tok_jac + name_char_jac) / 2.0
    addr_sim = (addr_tok_jac + addr_num_jac) / 2.0
    combined = np.sqrt(np.maximum(name_sim * addr_sim, 0.0))
    agreement = ((name_sim > 0.85) & (addr_sim > 0.70) & (country_match == 1.0)).astype(np.float64)
    disagreement = ((name_sim < 0.25) | (country_match == 0.0)).astype(np.float64)

    cols = [
        name_exact, name_tok_jac, name_tok_con, name_char_jac, name_ratio,
        addr_exact, addr_tok_jac, addr_num_jac, an_and, addr_ratio,
        postal_match, country_match, combined, agreement, disagreement,
    ]
    for i, col in enumerate(cols):
        X[:, i] = col
    return X
