"""Inference prediction CLI generating matching_results.tsv and candidate_pairs.tsv."""

import argparse
import os
import sys

# Ensure src/ is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from entitylink.pipeline.runner import PipelineRunner
from entitylink.models.classifier import MLPairMatcher
from entitylink.models.rule_matcher import RuleBasedMatcher
from entitylink.models.calibrator import DecisionCalibrator


def main():
    parser = argparse.ArgumentParser(description="Generate entity resolution predictions on test data.")
    parser.add_argument("--test-dir", default="data/test", help="Path to test set directory")
    parser.add_argument("--output-dir", default="output", help="Directory where TSV files will be saved")
    parser.add_argument("--model-path", default=None, help="Path to pre-trained ML model artifact")
    parser.add_argument("--threshold", type=float, default=0.62, help="Decision threshold for match calibration")
    parser.add_argument("--confidence-gap", type=float, default=0.15, help="Confidence gap for multi-match acceptance")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for test/debugging")

    args = parser.parse_args()

    if args.model_path and os.path.exists(args.model_path):
        print(f"[*] Loading model from {args.model_path}...")
        matcher = MLPairMatcher.load(args.model_path)
    else:
        print("[*] Using rule-based matcher...")
        matcher = RuleBasedMatcher()

    calibrator = DecisionCalibrator(
        threshold=args.threshold,
        confidence_gap=args.confidence_gap,
    )

    runner = PipelineRunner(
        matcher=matcher,
        calibrator=calibrator,
    )

    runner.run_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
