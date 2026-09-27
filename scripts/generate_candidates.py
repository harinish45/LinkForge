"""Candidate Generation CLI script.

Executes multi-strategy blocking to generate candidate_pairs.tsv for downstream matching models.
Optionally evaluates candidate recall and distribution statistics if ground truth is available.
"""

import argparse
import os
import sys
from typing import Dict, Set

# Ensure src/ is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from entitylink.data.loader import load_records_dict, load_ground_truth
from entitylink.blocking.candidate_generator import generate_candidates
from entitylink.pipeline.writer import write_candidate_pairs_tsv
from entitylink.evaluation.metrics import evaluate_blocking_performance


def main():
    parser = argparse.ArgumentParser(description="Generate candidate pairs using multi-strategy blocking.")
    parser.add_argument("--test-dir", default="data/test", help="Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv (or train_*)")
    parser.add_argument("--output-dir", default="output", help="Directory to save candidate_pairs.tsv")
    parser.add_argument("--max-candidates", type=int, default=150, help="Maximum candidates per S1 entity")
    parser.add_argument("--max-key-postings", type=int, default=500, help="Maximum postings per blocking key before pruning")
    args = parser.parse_args()

    s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    s3_path = os.path.join(args.test_dir, "test_source3.tsv")

    # Fallback to train prefix if test prefix files do not exist
    if not os.path.exists(s1_path):
        s1_path = os.path.join(args.test_dir, "train_source1.tsv")
        s2_path = os.path.join(args.test_dir, "train_source2.tsv")
        s3_path = os.path.join(args.test_dir, "train_source3.tsv")

    if not os.path.exists(s1_path):
        print(f"ERROR: Source 1 TSV not found in: {args.test_dir}")
        sys.exit(1)

    print(f"[*] Loading data sources from: {args.test_dir}")
    s1_records = load_records_dict(s1_path)
    s2_records = load_records_dict(s2_path) if os.path.exists(s2_path) else {}
    s3_records = load_records_dict(s3_path) if os.path.exists(s3_path) else {}

    print(f"    Loaded S1: {len(s1_records):,} records | S2: {len(s2_records):,} | S3: {len(s3_records):,}")

    config = {
        "max_candidates_per_entity": args.max_candidates,
        "max_key_postings": args.max_key_postings,
    }

    print("[*] Generating candidates via multi-strategy blocking engine...")
    candidate_map: Dict[str, Set[str]] = generate_candidates(
        s1_records, s2_records, s3_records, config=config
    )

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    out_candidate_file = os.path.join(args.output_dir, "candidate_pairs.tsv")

    write_candidate_pairs_tsv(out_candidate_file, candidate_map, required_s1_ids=sorted(s1_records.keys()))
    print(f"[+] Exported candidates to: {out_candidate_file}")

    # If ground truth exists, compute recall & distribution stats
    gt_path = os.path.join(args.test_dir, "train_ground_truth.tsv")
    if os.path.exists(gt_path):
        gt = load_ground_truth(gt_path)
        total_targets = len(s2_records) + len(s3_records)
        report = evaluate_blocking_performance(gt, candidate_map, total_target_records=total_targets)
        print("\n" + "=" * 50)
        print(" CANDIDATE GENERATION RECALL REPORT")
        print("=" * 50)
        print(f"  Candidate Recall:              {report.candidate_recall * 100:.2f}% ({report.covered_true_matches}/{report.total_true_matches} true matches covered)")
        print(f"  Total Candidates Generated:   {report.total_candidates:,}")
        print(f"  Avg Candidates per Entity:     {report.avg_candidates_per_entity:.2f}")
        print(f"  Median Candidates per Entity:  {report.median_candidates_per_entity:.1f}")
        print(f"  P95 Candidates per Entity:     {report.p95_candidates_per_entity:.1f}")
        print(f"  Max Candidates per Entity:     {report.max_candidates_per_entity}")
        print(f"  Reduction Ratio:               {report.reduction_ratio * 100:.4f}%")
        print("=" * 50)


if __name__ == "__main__":
    main()
