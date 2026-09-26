"""Candidate generation and blocking interface.

Designed to be modular so Adi's specialized blocking algorithms integrate seamlessly.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from entitylink.data.schema import EntityRecord
from entitylink.preprocessing.normalizer import (
    normalize_business_name,
    normalize_address,
    normalize_country,
)


def extract_blocking_keys(record: EntityRecord) -> List[str]:
    """Generate multiple blocking keys for an entity."""
    keys: List[str] = []
    norm_name = normalize_business_name(record.business_name)
    norm_country = normalize_country(record.country)
    norm_addr = normalize_address(record.business_address)

    tokens = norm_name.split()
    if tokens:
        # Key 1: country + first significant name token
        keys.append(f"{norm_country}#name_first#{tokens[0]}")
        # Key 2: country + first 4 chars of name
        if len(norm_name) >= 4:
            keys.append(f"{norm_country}#name_prefix#{norm_name[:4]}")
        # Key 3: country + second token if available
        if len(tokens) > 1 and len(tokens[1]) > 2:
            keys.append(f"{norm_country}#name_sec#{tokens[1]}")

    # Key 4: country + numeric tokens from address (e.g. zip/pin or building number)
    addr_tokens = norm_addr.split()
    addr_nums = [t for t in addr_tokens if t.isdigit() and len(t) >= 3]
    for num in addr_nums[:2]:
        keys.append(f"{norm_country}#num#{num}")

    return keys


def generate_candidates(
    source1: Dict[str, EntityRecord],
    source2: Dict[str, EntityRecord],
    source3: Dict[str, EntityRecord],
    config: Optional[Dict] = None,
) -> Dict[str, Set[str]]:
    """Primary candidate generation interface.
    
    Returns:
        Mapping from source1_entity_id -> Set of candidate entity IDs (from S2 / S3).
    """
    config = config or {}
    max_cands = config.get("max_candidates_per_entity", 100)

    # Build inverted index for S2 and S3
    index: Dict[str, List[str]] = defaultdict(list)

    for s2_id, record in source2.items():
        keys = extract_blocking_keys(record)
        for k in keys:
            index[k].append(s2_id)

    for s3_id, record in source3.items():
        keys = extract_blocking_keys(record)
        for k in keys:
            index[k].append(s3_id)

    # Query index for each S1 record
    candidates: Dict[str, Set[str]] = {}

    for s1_id, s1_rec in source1.items():
        matched_cands: Set[str] = set()
        s1_keys = extract_blocking_keys(s1_rec)
        
        for k in s1_keys:
            if k in index:
                matched_cands.update(index[k])
                if len(matched_cands) >= max_cands:
                    break

        candidates[s1_id] = matched_cands

    return candidates
