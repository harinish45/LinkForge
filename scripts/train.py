"""Train supervised pair matching model and calibrate decision thresholds."""

import argparse
import os
import sys
import random
from typing import Dict, List, Set, Tuple

from entitylink.data.loader import load_records_dict, load_ground_truth
from entitylink.data.schema import EntityRecord
from entitylink.blocking.candidate_generator import generate_candidates
from entitylink.models.classifier import MLPairMatcher
from entitylink.models.calibrator import DecisionCalibrator


def build_training_pairs(
    source1: Dict[str, EntityRecord],
    targets: Dict[str, EntityRecord],
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    neg_to_pos_ratio: int = 3,
) -> Tuple[List[Tuple[EntityRecord, EntityRecord]], List[int]]:
    """Build dataset of positive and negative candidate pairs."""
    pairs: List[Tuple[EntityRecord, EntityRecord]] = []
    labels: List[int] = []

    for s1_id, cand_ids in candidates.items():
        s1_rec = source1.get(s1_id)
        if not s1_rec:
            continue
        
        true_matches = ground_truth.get(s1_id, set())

        # Positives
        pos_ids = [cid for cid in cand_ids if cid in true_matches and cid in targets]
        for pid in pos_ids:
            pairs.append((s1_rec, targets[pid]))
            labels.append(1)

        # Negatives (sampled from candidates that are not true matches)
        neg_ids = [cid for cid in cand_ids if cid not in true_matches and cid in targets]
        sample_neg_count = min(len(neg_ids), max(2, len(pos_ids) * neg_to_pos_ratio))
        for nid in random.sample(neg_ids, sample_neg_count):
            pairs.append((s1_rec, targets[nid]))
            labels.append(0)

    # In case blocker missed true matches, inject known ground truth pairs present in targets
    for s1_id, true_matches in ground_truth.items():
        if s1_id in source1:
            s1_rec = source1[s1_id]
            for tid in true_matches:
                if tid in targets:
                    pairs.append((s1_rec, targets[tid]))
                    labels.append(1)

    return pairs, labels


def main():
    parser = argparse.ArgumentParser(description="Train EntityLink AI matching model.")
    parser.add_argument("--train-dir", default="data/train", help="Path to training data directory")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of records to sample for training")
    parser.add_argument("--model-type", default="gradient_boosting", choices=["gradient_boosting", "logistic_regression", "random_forest"])
    parser.add_argument("--output-model", default="models/pair_matcher.joblib", help="Output path for trained model")
    args = parser.parse_args()

    random.seed(42)
    print(f"[*] Starting model training pipeline (sample size: {args.sample_size:,})...")

    s1_path = os.path.join(args.train_dir, "train_source1.tsv")
    s2_path = os.path.join(args.train_dir, "train_source2.tsv")
    s3_path = os.path.join(args.train_dir, "train_source3.tsv")
    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")

    # 1. Load source data slices
    print("  [1/4] Loading training records...")
    s1_records = load_records_dict(s1_path, max_rows=args.sample_size)
    s2_records = load_records_dict(s2_path, max_rows=args.sample_size * 2)
    s3_records = load_records_dict(s3_path, max_rows=args.sample_size * 2)
    gt = load_ground_truth(gt_path, max_rows=args.sample_size)

    targets: Dict[str, EntityRecord] = {}
    targets.update(s2_records)
    targets.update(s3_records)
    print(f"        Loaded S1: {len(s1_records)}, S2: {len(s2_records)}, S3: {len(s3_records)}")

    # 2. Generate candidate pairs
    print("  [2/4] Generating candidates & pair features...")
    candidates = generate_candidates(s1_records, s2_records, s3_records)

    # 3. Form training pairs
    pairs, labels = build_training_pairs(s1_records, targets, gt, candidates)
    
    # If dataset has no positives in this slice, synthesize high-fidelity self/near-duplicate pairs
    pos_count = sum(labels)
    if pos_count == 0:
        print("        Synthesizing high-confidence training pairs for initial calibration...")
        for s1_rec in list(s1_records.values())[:100]:
            # Positive pair (near duplicate with legal suffix alteration)
            pseudo_pos = EntityRecord(
                entity_id="S2-pseudo",
                business_name=f"{s1_rec.business_name} Corp",
                business_address=s1_rec.business_address,
                country=s1_rec.country,
            )
            pairs.append((s1_rec, pseudo_pos))
            labels.append(1)

    print(f"        Total training pairs: {len(pairs)} (Positives: {sum(labels)}, Negatives: {len(labels) - sum(labels)})")

    # 4. Train model
    print(f"  [4/4] Fitting {args.model_type} classifier...")
    matcher = MLPairMatcher(model_type=args.model_type, random_state=42)
    matcher.fit(pairs, labels)
    matcher.save(args.output_model)
    print(f"[V] Successfully saved trained model to: {args.output_model}")


if __name__ == "__main__":
    main()
