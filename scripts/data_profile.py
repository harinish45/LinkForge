"""Dataset profiling and exploratory analysis script.

Analyzes dataset shapes, country distributions, ID uniqueness, and ground-truth match multiplicity.
"""

import argparse
import os
import sys
from typing import Dict

# Ensure src/ is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from entitylink.data.profiler import profile_source_file, profile_ground_truth


def main():
    parser = argparse.ArgumentParser(description="Profile dataset files.")
    parser.add_argument("--data-dir", default="data", help="Root data directory containing train/ and test/")
    parser.add_argument("--sample-size", type=int, default=None, help="Row sample limit per file for fast profiling")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    if os.path.exists(train_dir):
        for fname in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
            fpath = os.path.join(train_dir, fname)
            if os.path.exists(fpath):
                report = profile_source_file(fpath, max_rows=args.sample_size)
                print(f"\n[*] Profile for {fname}:")
                print(f"    Total records: {report.total_records:,}")
                print(f"    Unique names: {report.unique_names_count:,}")
                print(f"    Unique addresses: {report.unique_addresses_count:,}")
                print(f"    Avg tokens per name: {report.avg_name_token_count:.2f}")
                print(f"    Avg tokens per address: {report.avg_address_token_count:.2f}")
                print(f"    Top countries: {dict(list(report.country_distribution.items())[:5])}")

        gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
        if os.path.exists(gt_path):
            gt_report = profile_ground_truth(gt_path, max_rows=args.sample_size)
            print(f"\n[*] Ground Truth Profile:")
            print(f"    Total S1 entities: {gt_report.total_source1_entities:,}")
            print(f"    Total matched pairs: {gt_report.total_matched_pairs:,}")
            print(f"    Singletons (0 matches): {gt_report.singleton_count:,} ({gt_report.singleton_ratio * 100:.2f}%)")
            print(f"    Match multiplicity distribution: {gt_report.match_count_distribution}")

    test_dir = os.path.join(args.data_dir, "test")
    if os.path.exists(test_dir):
        for fname in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
            fpath = os.path.join(test_dir, fname)
            if os.path.exists(fpath):
                report = profile_source_file(fpath, max_rows=args.sample_size)
                print(f"\n[*] Profile for {fname}:")
                print(f"    Total records: {report.total_records:,}")
                print(f"    Unique names: {report.unique_names_count:,}")
                print(f"    Unique addresses: {report.unique_addresses_count:,}")


if __name__ == "__main__":
    main()
