"""Multi-Strategy Blocking Engine & Candidate Generator.

Implements multiple blocking strategies (exact, compact, informative token,
character n-gram, address-numeric anchor, country-assisted, plus high-precision
composite country+token/numeric keys) with token-frequency explosion safeguards,
candidate provenance tracking, and deterministic candidate union.
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from entitylink.data.schema import EntityRecord
from entitylink.preprocessing.normalizer import (
    NormalizedRecord,
    build_normalized_record,
    normalize_address,
    normalize_business_name,
    normalize_country,
)


@dataclass
class CandidatePair:
    """Represents a candidate pair with tracking of blocking provenance."""
    source1_id: str
    candidate_id: str
    block_reasons: Set[str] = field(default_factory=set)


def generate_char_ngrams(text: str, n: int = 3) -> List[str]:
    """Extract character n-grams from normalized text."""
    if not text or len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


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
        if len(tokens[0]) >= 3:
            keys.append(f"{norm_country}#name_first#{tokens[0]}")
        if len(norm_name) >= 4:
            keys.append(f"{norm_country}#name_prefix#{norm_name[:4]}")
        if len(tokens) > 1 and len(tokens[1]) >= 3:
            keys.append(f"{norm_country}#name_sec#{tokens[1]}")

    addr_nums = re.findall(r"\b\d+\b", norm_addr)
    if addr_nums and tokens:
        keys.append(f"{norm_country}#num_name#{addr_nums[0]}#{norm_name[:3]}")

    for num in addr_nums:
        if len(num) in (5, 6) and tokens:
            keys.append(f"{norm_country}#zip_name#{num}#{tokens[0][:3]}")
            break

    if len(addr_nums) >= 2:
        keys.append(f"{norm_country}#addr_nums#{addr_nums[0]}#{addr_nums[1]}")
    elif len(addr_nums) == 1 and len(addr_nums[0]) >= 3:
        common_words = {"street", "drive", "road", "avenue", "lane", "court", "floor", "shop", "suite", "unit"}
        addr_words = [w for w in re.findall(r"[a-z]+", norm_addr) if len(w) >= 4 and w not in common_words]
        if addr_words:
            keys.append(f"{norm_country}#addr_num_word#{addr_nums[0]}#{addr_words[0]}")

    return keys


class BaseBlocker:
    """Abstract interface for a blocking strategy."""
    
    name: str = "base_blocker"

    def extract_keys(self, record: NormalizedRecord) -> List[str]:
        raise NotImplementedError


class ExactNameBlocker(BaseBlocker):
    """Blocks records sharing the exact normalized business name."""
    
    name: str = "name_exact"

    def extract_keys(self, record: NormalizedRecord) -> List[str]:
        if record.business_name_normalized:
            return [f"exact_name#{record.business_name_normalized}"]
        return []


class CompactNameBlocker(BaseBlocker):
    """Blocks records sharing the exact compact business name (punctuation/spaces removed)."""
    
    name: str = "name_compact"

    def extract_keys(self, record: NormalizedRecord) -> List[str]:
        if record.business_name_compact and len(record.business_name_compact) >= 3:
            return [f"compact_name#{record.business_name_compact}"]
        return []


class InformativeTokenBlocker(BaseBlocker):
    """Blocks records sharing informative, non-generic business name tokens."""
    
    name: str = "name_token"

    def extract_keys(self, record: NormalizedRecord) -> List[str]:
        keys = []
        for token in record.business_name_tokens:
            if len(token) >= 3:
                keys.append(f"token#{token}")
        return keys


class CharacterNGramBlocker(BaseBlocker):
    """Blocks records sharing character n-grams to catch typos and minor spelling variations."""
    
    name: str = "name_ngram"

    def __init__(self, ngram_size: int = 3):
        self.ngram_size = ngram_size

    def extract_keys(self, record: NormalizedRecord) -> List[str]:
        ngrams = generate_char_ngrams(record.business_name_compact, n=self.ngram_size)
        return [f"ngram3#{ng}" for ng in ngrams if len(ng) == self.ngram_size]


class AddressAnchorBlocker(BaseBlocker):
    """Blocks records sharing numeric/building anchors and significant address tokens."""
    
    name: str = "address_anchor"

    def extract_keys(self, record: NormalizedRecord) -> List[str]:
        keys = []
        # Numeric tokens (building number, street number, suite, postal code)
        nums = [t for t in record.business_address_tokens if t.isdigit() and len(t) >= 2]
        for num in nums[:3]:
            keys.append(f"addr_num#{num}")
        
        # Significant address tokens
        for token in record.business_address_tokens:
            if not token.isdigit() and len(token) >= 4 and token not in ("street", "road", "avenue", "drive"):
                keys.append(f"addr_token#{token}")

        return keys


class CountryAssistedBlocker(BaseBlocker):
    """Blocks records sharing country code combined with first name prefix/token."""
    
    name: str = "country_assisted"

    def extract_keys(self, record: NormalizedRecord) -> List[str]:
        keys = []
        c = record.country_normalized
        if c and c != "UNKNOWN" and record.business_name_tokens:
            first_token = record.business_name_tokens[0]
            if len(first_token) >= 3:
                keys.append(f"country_token#{c}#{first_token}")
        return keys


class CompositeKeyBlocker(BaseBlocker):
    """Blocks records using high-precision composite country + token/numeric keys."""

    name: str = "composite_key"

    def extract_keys(self, record: NormalizedRecord) -> List[str]:
        # Reuse composite keys on normalized content (normalization is idempotent).
        proxy_rec = EntityRecord(
            entity_id=record.entity_id,
            business_name=record.business_name_normalized,
            business_address=record.business_address_normalized,
            country=record.country_normalized,
        )
        return extract_blocking_keys(proxy_rec)


class MultiStrategyBlockingEngine:
    """Engine that orchestrates multi-strategy candidate generation and deduplication."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_candidates_per_entity = self.config.get("max_candidates_per_entity", 150)
        self.max_key_postings = self.config.get("max_key_postings", 500)
        
        self.blockers: List[BaseBlocker] = [
            ExactNameBlocker(),
            CompactNameBlocker(),
            InformativeTokenBlocker(),
            CharacterNGramBlocker(ngram_size=3),
            AddressAnchorBlocker(),
            CountryAssistedBlocker(),
            CompositeKeyBlocker(),
        ]

    def build_indexes(
        self,
        target_records: Dict[str, NormalizedRecord]
    ) -> Dict[str, Set[Tuple[str, str]]]:
        """Build inverted index: key -> Set of (target_entity_id, blocker_name)."""
        inverted_index: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

        for entity_id, rec in target_records.items():
            for blocker in self.blockers:
                keys = blocker.extract_keys(rec)
                for k in keys:
                    inverted_index[k].add((entity_id, blocker.name))

        # Candidate explosion control: purge keys with excessive postings
        pruned_index: Dict[str, Set[Tuple[str, str]]] = {}
        for key, postings in inverted_index.items():
            if len(postings) <= self.max_key_postings:
                pruned_index[key] = postings

        return pruned_index

    def generate_candidates_with_provenance(
        self,
        source1_records: Dict[str, NormalizedRecord],
        target_records: Dict[str, NormalizedRecord],
    ) -> Dict[str, Dict[str, CandidatePair]]:
        """Generate candidate set mapping source1_id -> candidate_id -> CandidatePair."""
        inverted_index = self.build_indexes(target_records)

        results: Dict[str, Dict[str, CandidatePair]] = {}

        for s1_id, s1_rec in source1_records.items():
            s1_candidates: Dict[str, CandidatePair] = {}
            
            for blocker in self.blockers:
                s1_keys = blocker.extract_keys(s1_rec)
                for k in s1_keys:
                    if k in inverted_index:
                        for cand_id, blocker_name in inverted_index[k]:
                            if cand_id not in s1_candidates:
                                s1_candidates[cand_id] = CandidatePair(
                                    source1_id=s1_id,
                                    candidate_id=cand_id,
                                    block_reasons=set()
                                )
                            s1_candidates[cand_id].block_reasons.add(blocker_name)

            # Cap candidates if exceeding configured max
            if len(s1_candidates) > self.max_candidates_per_entity:
                # Rank candidates by number of matching blocking strategies
                sorted_cands = sorted(
                    s1_candidates.values(),
                    key=lambda cp: len(cp.block_reasons),
                    reverse=True
                )[:self.max_candidates_per_entity]
                s1_candidates = {cp.candidate_id: cp for cp in sorted_cands}

            results[s1_id] = s1_candidates

        return results


def generate_candidates(
    source1: Dict[str, EntityRecord],
    source2: Dict[str, EntityRecord],
    source3: Dict[str, EntityRecord],
    config: Optional[Dict] = None,
) -> Dict[str, Set[str]]:
    """Primary candidate generation function contract with Harinish's matcher.
    
    Returns:
        Mapping from source1_entity_id -> Set of candidate entity IDs (from S2 / S3).
    """
    # 1. Normalize all records
    norm_s1 = {eid: build_normalized_record(eid, r.business_name, r.business_address, r.country) for eid, r in source1.items()}
    
    targets: Dict[str, NormalizedRecord] = {}
    for eid, r in source2.items():
        targets[eid] = build_normalized_record(eid, r.business_name, r.business_address, r.country)
    for eid, r in source3.items():
        targets[eid] = build_normalized_record(eid, r.business_name, r.business_address, r.country)

    # 2. Run multi-strategy engine
    engine = MultiStrategyBlockingEngine(config=config)
    candidate_map_with_prov = engine.generate_candidates_with_provenance(norm_s1, targets)

    # 3. Export simple candidate mapping source1_id -> set(candidate_ids)
    candidate_sets: Dict[str, Set[str]] = {}
    for s1_id in source1.keys():
        cand_dict = candidate_map_with_prov.get(s1_id, {})
        candidate_sets[s1_id] = set(cand_dict.keys())

    return candidate_sets
