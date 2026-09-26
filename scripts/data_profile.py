"""Dataset profiling and exploratory analysis script.

Analyzes dataset shapes, country distributions, ID uniqueness, and ground-truth match multiplicity.
"""

import argparse
import os
import sys
from collections import Counter
from typing import Dict

# Ensure src/ is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from entitylink.data.loader import stream_tsv_records, load_ground_truth
from entitylink.data.schema import validate_entity_id


def profile_source_file(file_path: str, max_rows: int = 10000) -> Dict:
    """Analyze a single source TSV file."""
    total = 0
    countries = Counter()
    missing_names = 0
    missing_addrs = 0
    id_prefixes = Counter()

    print(f"\n[*] Profiling: {os.path.basename(file_path)} (sample up to {max_rows:,} rows)")
    for rec in stream_tsv_records(file_path, max_rows=max_rows):
        total += 1
        prefix = rec.entity_id.split("-")[0] if "-" in rec.entity_id else "NONE"
        id_prefixes[prefix] += 1
        countries[rec.country] += 1
        if not rec.business_name:
            missing_names += 1
        if not rec.business_address:
            missing_addrs += 1

    print(f"    Total scanned: {total}")
    print(f"    ID prefixes: {dict(id_prefixes)}")
    print(f"    Countries: {dict(countries)}")
    print(f"    Missing names: {missing_names}, Missing addresses: {missing_addrs}")
    return {"total": total, "countries": countries}


def profile_ground_truth(gt_path: str, max_rows: int = 10000) -> None:
    """Analyze ground-truth match multiplicity and singleton rates."""
    print(f"\n[*] Profiling Ground Truth: {os.path.basename(gt_path)} (sample up to {max_rows:,} rows)")
    gt = load_ground_truth(gt_path, max_rows=max_rows)
    match_counts = Counter(len(matches) for matches in gt.values())
    total_entities = len(gt)
    singletons = match_counts[0]
    multi_matches = sum(count for n, count in match_counts.items() if n > 1)

    print(f"    Total S1 entities scanned: {total_entities}")
    print(f"    Singleton (0 matches): {singletons} ({singletons / total_entities * 100:.2f}%)")
    print(f"    Single match (1 match): {match_counts[1]} ({match_counts[1] / total_entities * 100:.2f}%)")
    print(f"    Multi-match (>1 matches): {multi_matches} ({multi_matches / total_entities * 100:.2f}%)")
    print(f"    Distribution (match count -> entity count):")
    for n in sorted(match_counts.keys())[:10]:
        print(f"      {n} matches: {match_counts[n]}")


def main():
    parser = argparse.ArgumentParser(description="Profile dataset files.")
    parser.add_argument("--data-dir", default="data", help="Root data directory containing train/ and test/")
    parser.add_argument("--sample-size", type=int, default=10000, help="Row sample limit per file for fast profiling")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    if os.path.exists(train_dir):
        profile_source_file(os.path.join(train_dir, "train_source1.tsv"), max_rows=args.sample_size)
        profile_source_file(os.path.join(train_dir, "train_source2.tsv"), max_rows=args.sample_size)
        profile_source_file(os.path.join(train_dir, "train_source3.tsv"), max_rows=args.sample_size)
        gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
        if os.path.exists(gt_path):
            profile_ground_truth(gt_path, max_rows=args.sample_size)

    test_dir = os.path.join(args.data_dir, "test")
    if os.path.exists(test_dir):
        profile_source_file(os.path.join(test_dir, "test_source1.tsv"), max_rows=args.sample_size)


if __name__ == "__main__":
    main()
