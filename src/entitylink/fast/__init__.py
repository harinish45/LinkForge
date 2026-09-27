"""High-throughput vectorized scoring path for full-scale inference."""

from entitylink.fast.fastnorm import encode_record, crc32
from entitylink.fast.payload import PayloadStore, SCALAR_FIELDS, BIT_FIELDS
from entitylink.fast.features import FAST_FEATURE_NAMES, extract_fast_features
from entitylink.fast.selection import select_matches_mask

__all__ = [
    "encode_record",
    "crc32",
    "PayloadStore",
    "SCALAR_FIELDS",
    "BIT_FIELDS",
    "FAST_FEATURE_NAMES",
    "extract_fast_features",
    "select_matches_mask",
]
