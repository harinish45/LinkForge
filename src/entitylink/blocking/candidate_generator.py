"""Candidate generation and blocking interface.

Designed to be modular so Adi's specialized blocking algorithms integrate seamlessly.
"""

from collections import defaultdict
import re
from typing import Dict, List, Optional, Set, Tuple
from entitylink.data.schema import EntityRecord
from entitylink.preprocessing.normalizer import (
    normalize_business_name,
    normalize_address,
    normalize_country,
)


def extract_blocking_keys(record: EntityRecord) -> List[str]:
    """Generate high-precision composite blocking keys for an entity.
    
    Composite blocking combines country with name tokens, prefixes, and address numbers
    to keep candidate buckets small while maintaining high true-positive recall.
    """
    keys: List[str] = []
    norm_name = normalize_business_name(record.business_name)
    norm_country = normalize_country(record.country)
    norm_addr = normalize_address(record.business_address)

    tokens = norm_name.split()
    if tokens:
        # Key 1: country + first significant name token (>= 3 chars)
        if len(tokens[0]) >= 3:
            keys.append(f"{norm_country}#name_first#{tokens[0]}")
        # Key 2: country + first 4 chars of normalized name
        if len(norm_name) >= 4:
            keys.append(f"{norm_country}#name_prefix#{norm_name[:4]}")
        # Key 3: country + second token if available
        if len(tokens) > 1 and len(tokens[1]) >= 3:
            keys.append(f"{norm_country}#name_sec#{tokens[1]}")

    # Key 4: country + street/building number + 3-char name prefix
    addr_nums = re.findall(r"\b\d+\b", norm_addr)
    if addr_nums and tokens:
        keys.append(f"{norm_country}#num_name#{addr_nums[0]}#{norm_name[:3]}")

    # Key 5: country + 5 or 6 digit postal code + first token prefix
    for num in addr_nums:
        if len(num) in (5, 6) and tokens:
            keys.append(f"{norm_country}#zip_name#{num}#{tokens[0][:3]}")
            break

    # Key 6: country + address numbers pair (handles multilingual name translations with same physical address)
    if len(addr_nums) >= 2:
        keys.append(f"{norm_country}#addr_nums#{addr_nums[0]}#{addr_nums[1]}")
    elif len(addr_nums) == 1 and len(addr_nums[0]) >= 3:
        common_words = {"street", "drive", "road", "avenue", "lane", "court", "floor", "shop", "suite", "unit"}
        addr_words = [w for w in re.findall(r"[a-z]+", norm_addr) if len(w) >= 4 and w not in common_words]
        if addr_words:
            keys.append(f"{norm_country}#addr_num_word#{addr_nums[0]}#{addr_words[0]}")

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

