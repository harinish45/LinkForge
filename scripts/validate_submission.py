"""Submission validation CLI wrapping official challenge rules."""

import argparse
import os
import sys

# Ensure student_resource/utils is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "student_resource")))

from utils.validate_submission import validate


def main():
    parser = argparse.ArgumentParser(description="Validate challenge submission TSVs against official rules.")
    parser.add_argument("--matching", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="data/test", help="Path to test directory")
    parser.add_argument("--check-ids", action="store_true", help="Enable memory-heavy full ID existence cross-check")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("   ML Challenge 2026 — Submission Validation")
    print("=" * 60)

    errors, warnings = validate(
        matching_path=args.matching,
        candidate_path=args.candidate,
        test_dir=args.test_dir,
        check_ids=args.check_ids,
    )

    if warnings:
        print("\n[!] Warnings:")
        for w in warnings:
            print(f"  * {w}")

    if errors:
        print("\n[X] VALIDATION FAILED — Fix the following errors before submitting:")
        for e in errors:
            print(f"  * {e}")
        sys.exit(1)
    else:
        print("\n[V] VALIDATION PASSED! Submission files are formatted correctly and safe to submit.")
        sys.exit(0)


if __name__ == "__main__":
    main()
