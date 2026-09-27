"""Data profiling utility for dataset statistical analysis."""

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from entitylink.data.loader import stream_tsv_records, load_ground_truth
from entitylink.preprocessing.normalizer import normalize_country, normalize_text


@dataclass
class DatasetProfileReport:
    """Summary profile of a source table."""
    file_path: str
    total_records: int
    null_name_count: int
    null_address_count: int
    null_country_count: int
    unique_names_count: int
    unique_addresses_count: int
    country_distribution: Dict[str, int]
    avg_name_token_count: float
    avg_address_token_count: float


@dataclass
class GroundTruthProfileReport:
    """Summary profile of ground truth dataset."""
    total_source1_entities: int
    total_matched_pairs: int
    singleton_count: int
    singleton_ratio: float
    match_count_distribution: Dict[int, int]


def profile_source_file(file_path: str, max_rows: Optional[int] = None) -> DatasetProfileReport:
    """Profile a single TSV source file."""
    total = 0
    null_names = 0
    null_addrs = 0
    null_countries = 0
    
    unique_names: Set[str] = set()
    unique_addrs: Set[str] = set()
    countries: Counter = Counter()
    
    total_name_tokens = 0
    total_addr_tokens = 0

    for rec in stream_tsv_records(file_path, max_rows=max_rows):
        total += 1
        
        if not rec.business_name:
            null_names += 1
        else:
            unique_names.add(rec.business_name.strip().lower())
            total_name_tokens += len(normalize_text(rec.business_name).split())
            
        if not rec.business_address:
            null_addrs += 1
        else:
            unique_addrs.add(rec.business_address.strip().lower())
            total_addr_tokens += len(normalize_text(rec.business_address).split())

        c_norm = normalize_country(rec.country)
        countries[c_norm] += 1

    return DatasetProfileReport(
        file_path=file_path,
        total_records=total,
        null_name_count=null_names,
        null_address_count=null_addrs,
        null_country_count=null_countries,
        unique_names_count=len(unique_names),
        unique_addresses_count=len(unique_addrs),
        country_distribution=dict(countries),
        avg_name_token_count=total_name_tokens / total if total > 0 else 0.0,
        avg_address_token_count=total_addr_tokens / total if total > 0 else 0.0,
    )


def profile_ground_truth(file_path: str, max_rows: Optional[int] = None) -> GroundTruthProfileReport:
    """Profile ground truth mapping file."""
    gt = load_ground_truth(file_path, max_rows=max_rows)
    total_s1 = len(gt)
    total_pairs = sum(len(matches) for matches in gt.values())
    singletons = sum(1 for matches in gt.values() if len(matches) == 0)
    
    match_counts: Counter = Counter()
    for matches in gt.values():
        match_counts[len(matches)] += 1

    return GroundTruthProfileReport(
        total_source1_entities=total_s1,
        total_matched_pairs=total_pairs,
        singleton_count=singletons,
        singleton_ratio=singletons / total_s1 if total_s1 > 0 else 0.0,
        match_count_distribution=dict(match_counts),
    )
