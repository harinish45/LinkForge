"""Evaluation script computing official F0.5 score, candidate recall, and error breakdown on labeled validation datasets."""

import argparse
import os
import sys

# Ensure src/ is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from entitylink.data.loader import load_ground_truth
from entitylink.evaluation.metrics import (
    evaluate_matching_predictions,
    evaluate_blocking_performance,
    analyze_blocking_errors,
)


def parse_id_list_file(file_path: str):
    """Parse output matching or candidate TSV into mapping."""
    mapping = {}
    with open(file_path, encoding="utf-8") as f:
        next(f, None)  # skip header
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t", 1)
            s1_id = parts[0].strip()
            rest = parts[1].strip() if len(parts) > 1 else ""
            ids = {i.strip() for i in rest.split(",") if i.strip()}
            mapping[s1_id] = ids
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Evaluate EntityLink matching outputs against ground truth.")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth TSV")
    parser.add_argument("--matching", default=None, help="Path to predicted matching_results.tsv")
    parser.add_argument("--candidate", default=None, help="Path to candidate_pairs.tsv for candidate recall evaluation")
    args = parser.parse_args()

    if not args.matching and not args.candidate:
        print("ERROR: Must specify at least one of --matching or --candidate.")
        sys.exit(1)

    gt = load_ground_truth(args.ground_truth)

    if args.matching and os.path.exists(args.matching):
        preds = parse_id_list_file(args.matching)
        common_keys = set(gt.keys()) & set(preds.keys())
        sub_gt = {k: gt[k] for k in common_keys}
        sub_preds = {k: preds[k] for k in common_keys}
        report = evaluate_matching_predictions(sub_gt, sub_preds)

        print("\n" + "=" * 60)
        print("   ML Challenge 2026 — Official Metric Evaluation")
        print("=" * 60)
        print(f"Evaluated Entities:           {len(common_keys):,}")
        print(f"Ground Truth Match Pairs:     {report.total_ground_truth_matches:,}")
        print(f"Predicted Match Pairs:        {report.total_predicted_matches:,}")
        print(f"True Positives (TP):          {report.true_positives:,}")
        print(f"False Positives (FP):         {report.false_positives:,}")
        print(f"False Negatives (FN):         {report.false_negatives:,}")
        print("-" * 60)
        print(f"PRECISION:                    {report.precision:.4f}")
        print(f"RECALL:                       {report.recall:.4f}")
        print(f"F0.5 SCORE (Official):        {report.f05:.4f}")
        print(f"F1 SCORE:                     {report.f1:.4f}")
        print(f"Singleton Accuracy:           {report.singleton_accuracy:.4f}")
        print("=" * 60)

    if args.candidate and os.path.exists(args.candidate):
        cands = parse_id_list_file(args.candidate)
        common_keys = set(gt.keys()) & set(cands.keys())
        sub_gt = {k: gt[k] for k in common_keys}
        sub_cands = {k: cands[k] for k in common_keys}
        
        cand_report = evaluate_blocking_performance(sub_gt, sub_cands)
        missed = analyze_blocking_errors(sub_gt, sub_cands)

        print("\n" + "=" * 60)
        print("   Candidate Generation (Blocking) Evaluation")
        print("=" * 60)
        print(f"Candidate Recall:             {cand_report.candidate_recall * 100:.2f}% ({cand_report.covered_true_matches}/{cand_report.total_true_matches})")
        print(f"Total Candidate Pairs:        {cand_report.total_candidates:,}")
        print(f"Avg Candidates / Entity:      {cand_report.avg_candidates_per_entity:.2f}")
        print(f"Median Candidates / Entity:   {cand_report.median_candidates_per_entity:.1f}")
        print(f"P95 Candidates / Entity:      {cand_report.p95_candidates_per_entity:.1f}")
        print(f"Max Candidates / Entity:      {cand_report.max_candidates_per_entity}")
        print(f"Missed True Pairs:            {len(missed):,}")
        if missed:
            print(f"Sample Missed Candidates:     {missed[:3]}")
        print("=" * 60)


if __name__ == "__main__":
    main()
