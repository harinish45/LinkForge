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
    parser.add_argument("--sample-size", type=int, default=1500, help="Number of S1 records to sample for training")
    parser.add_argument("--model-type", default="gradient_boosting", choices=["gradient_boosting", "logistic_regression"])
    parser.add_argument("--output-model", default="models/pair_matcher.joblib", help="Output path for trained model")
    args = parser.parse_args()

    random.seed(42)
    print(f"[*] Starting model training pipeline (sample size: {args.sample_size:,})...")

    s1_path = os.path.join(args.train_dir, "train_source1.tsv")
    s2_path = os.path.join(args.train_dir, "train_source2.tsv")
    s3_path = os.path.join(args.train_dir, "train_source3.tsv")
    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")

    # 1. Load ground truth and sample representative S1 entities (positives + singletons)
    print("  [1/4] Loading ground truth & selecting training entities...")
    gt = load_ground_truth(gt_path, max_rows=max(10000, args.sample_size * 4))

    matched_s1 = [s1 for s1, targets in gt.items() if len(targets) > 0]
    singleton_s1 = [s1 for s1, targets in gt.items() if len(targets) == 0]

    n_matched = min(len(matched_s1), int(args.sample_size * 0.8))
    n_singletons = min(len(singleton_s1), args.sample_size - n_matched)

    sampled_matched = random.sample(matched_s1, n_matched)
    sampled_singletons = random.sample(singleton_s1, n_singletons)
    sampled_s1_ids = set(sampled_matched + sampled_singletons)

    target_s2_ids = {tid for s1 in sampled_s1_ids for tid in gt[s1] if tid.startswith("S2-")}
    target_s3_ids = {tid for s1 in sampled_s1_ids for tid in gt[s1] if tid.startswith("S3-")}

    print(f"        Sampling {len(sampled_s1_ids)} S1 records ({len(sampled_matched)} matched, {len(sampled_singletons)} singletons)")
    print(f"        Required target records: S2={len(target_s2_ids)}, S3={len(target_s3_ids)}")

    # 2. Fast selective retrieval of exact targets and S1 records
    print("  [2/4] Selectively retrieving records by ID...")
    from entitylink.data.loader import load_records_by_ids, stream_tsv_records
    s1_records = load_records_by_ids(s1_path, sampled_s1_ids)
    s2_records = load_records_by_ids(s2_path, target_s2_ids)
    s3_records = load_records_by_ids(s3_path, target_s3_ids)

    # Add random background records from S2 and S3 for realistic hard negatives in blocking
    for rec in stream_tsv_records(s2_path, max_rows=1000):
        s2_records[rec.entity_id] = rec
    for rec in stream_tsv_records(s3_path, max_rows=1000):
        s3_records[rec.entity_id] = rec

    targets: Dict[str, EntityRecord] = {}
    targets.update(s2_records)
    targets.update(s3_records)

    # 3. Generate candidate pairs & build positive/negative training pairs
    print("  [3/4] Generating candidates & extracting pair features...")
    candidates = generate_candidates(s1_records, s2_records, s3_records)
    pairs, labels = build_training_pairs(s1_records, targets, gt, candidates)

    print(f"        Total training pairs: {len(pairs)} (Positives: {sum(labels)}, Negatives: {len(labels) - sum(labels)})")

    # 4. Train model
    print(f"  [4/4] Fitting {args.model_type} classifier...")
    matcher = MLPairMatcher(model_type=args.model_type, random_state=42)
    matcher.fit(pairs, labels)
    matcher.save(args.output_model)
    print(f"[V] Successfully saved trained model to: {args.output_model}")


if __name__ == "__main__":
    main()
