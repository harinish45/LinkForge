"""Train the fast (vectorized-feature) pair matcher on the training split.

The trainer mirrors deployment: candidates come from the same hashed blocking
index used at inference time, features come from the same vectorized extractor,
and the label of a candidate pair is ground-truth membership.
"""

import argparse
import csv
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import joblib
import numpy as np

from entitylink.data.loader import load_records_by_ids, stream_tsv_records
from entitylink.data.schema import EntityRecord
from entitylink.fast.features import FAST_FEATURE_NAMES
from entitylink.fast.pipeline import FastMatcher, build_bundle
from entitylink.models.classifier import FastGradientBoostingClassifier, MLPairMatcher


def parse_ids(raw: str) -> Set[str]:
    raw = (raw or "").strip()
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def sample_gt(gt_path: str, n_sample: int, exclude_s1: Optional[Set[str]] = None,
              seed: int = 42, stride: int = 1) -> Dict[str, Set[str]]:
    """Reservoir-sample (s1 -> true target ids) rows from the ground truth file."""
    exclude_s1 = exclude_s1 or set()
    rng = random.Random(seed)
    sample: List[Tuple[str, Set[str]]] = []
    seen = 0
    with open(gt_path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for i, row in enumerate(reader):
            if stride > 1 and i % stride:
                continue
            s1 = row[0].strip() if row else ""
            if not s1 or s1 in exclude_s1:
                continue
            ids = parse_ids(row[1] if len(row) > 1 else "")
            seen += 1
            if len(sample) < n_sample:
                sample.append((s1, ids))
            else:
                j = rng.randrange(seen)
                if j < n_sample:
                    sample[j] = (s1, ids)
    return dict(sample)


def sample_background(paths: Sequence[str], n_bg: int, exclude_ids: Optional[Set[str]] = None,
                      seed: int = 42) -> List[EntityRecord]:
    """Reservoir-sample target records (distractors) from the source files."""
    exclude_ids = exclude_ids or set()
    rng = random.Random(seed)
    sample: List[EntityRecord] = []
    seen = 0
    for path in paths:
        for rec in stream_tsv_records(path):
            if rec.entity_id in exclude_ids:
                continue
            seen += 1
            if len(sample) < n_bg:
                sample.append(rec)
            else:
                j = rng.randrange(seen)
                if j < n_bg:
                    sample[j] = rec
    return sample


def train_fast_model(
    train_dir: str,
    n_sample_s1: int = 100_000,
    n_bg_per_source: int = 100_000,
    exclude_s1: Optional[Set[str]] = None,
    exclude_targets: Optional[Set[str]] = None,
    seed: int = 42,
    stride: int = 1,
    chunk: int = 5000,
    max_train_rows: int = 2_000_000,
    verbose: bool = True,
    gt_sample: Optional[Dict[str, Set[str]]] = None,
    bg_override: Optional[List[EntityRecord]] = None,
):
    """Fit a FastGradientBoostingClassifier on fast-path candidate pairs."""
    exclude_s1 = set(exclude_s1 or ())
    exclude_targets = set(exclude_targets or ())
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    t0 = time.time()
    if gt_sample is None:
        gt = sample_gt(gt_path, n_sample_s1, exclude_s1, seed=seed, stride=stride)
    else:
        gt = gt_sample
    n_pos_s1 = sum(1 for v in gt.values() if v)
    if verbose:
        print(f"    gt sample: {len(gt):,} S1 ({n_pos_s1:,} matched, "
              f"{len(gt) - n_pos_s1:,} singletons) [{time.time()-t0:.0f}s]")

    pos_ids = {t for ids in gt.values() for t in ids} - exclude_targets
    s1_recs = load_records_by_ids(s1_path, set(gt.keys()))
    tgt_recs = load_records_by_ids(s2_path, {t for t in pos_ids if t.startswith("S2-")})
    tgt_recs.update(load_records_by_ids(s3_path, {t for t in pos_ids if t.startswith("S3-")}))
    if verbose:
        print(f"    loaded S1={len(s1_recs):,} true-targets={len(tgt_recs):,} [{time.time()-t0:.0f}s]")

    if bg_override is not None:
        bg = bg_override
    else:
        bg = sample_background([s2_path, s3_path], n_bg_per_source,
                               exclude_ids=set(tgt_recs) | exclude_targets, seed=seed)
    if verbose:
        print(f"    background distractors: {len(bg):,} [{time.time()-t0:.0f}s]")

    records = list(tgt_recs.values()) + bg
    bundle = build_bundle(iter(records), capacity=len(records))
    id_row = bundle.id_to_row()
    if verbose:
        print(f"    bundle: {bundle.count:,} targets, index keys={bundle.index.size:,} "
              f"[{time.time()-t0:.0f}s]")

    matcher = FastMatcher(bundle, model=MLPairMatcher())  # model unused for candidates/features
    s1_list = list(s1_recs.values())
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    total_pairs = 0
    covered_pos = 0
    total_pos = 0
    for start in range(0, len(s1_list), chunk):
        part = s1_list[start:start + chunk]
        s1_store, owners, tids = matcher.candidates_for(part)
        if owners.shape[0] == 0:
            continue
        X = matcher.features_for(s1_store, bundle.store, owners, tids)
        pos_packed: List[int] = []
        for local, rec in enumerate(part):
            true_ids = gt.get(rec.entity_id, ())
            total_pos += len(true_ids)
            for tid in true_ids:
                row = id_row.get(tid)
                if row is not None:
                    pos_packed.append((local << 32) | row)
        if pos_packed:
            pos_arr = np.array(sorted(set(pos_packed)), dtype=np.uint64)
            pair_packed = (owners.astype(np.uint64) << np.uint64(32)) | tids.astype(np.uint64)
            y = np.isin(pair_packed, pos_arr).astype(np.int32)
            covered_pos += int(y.sum())
        else:
            y = np.zeros(owners.shape[0], dtype=np.int32)
        X_parts.append(X)
        y_parts.append(y)
        total_pairs += owners.shape[0]

    X_all = np.vstack(X_parts) if X_parts else np.zeros((0, len(FAST_FEATURE_NAMES)))
    y_all = np.concatenate(y_parts) if y_parts else np.zeros(0, dtype=np.int32)
    if verbose:
        print(f"    candidate pairs: {total_pairs:,} | positives in candidates: {covered_pos:,}"
              f" (blocking recall {covered_pos / max(total_pos, 1):.4f}) [{time.time()-t0:.0f}s]")

    if X_all.shape[0] > max_train_rows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X_all.shape[0], max_train_rows, replace=False)
        X_all, y_all = X_all[idx], y_all[idx]

    model = FastGradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=seed)
    model.fit(X_all, y_all)
    if verbose:
        print(f"    fitted {len(model.trees)} stumps on {X_all.shape[0]:,} rows "
              f"(pos={int(y_all.sum()):,}) [{time.time()-t0:.0f}s]")
    stats = {
        "train_rows": int(X_all.shape[0]),
        "positives": int(y_all.sum()),
        "candidate_pairs": total_pairs,
        "positives_covered": covered_pos,
        "positives_total": total_pos,
        "init_val": float(model.init_val),
    }
    return model, stats, bundle


def main():
    ap = argparse.ArgumentParser(description="Train the fast vectorized pair matcher.")
    ap.add_argument("--train-dir", default="data/train")
    ap.add_argument("--sample-s1", type=int, default=100_000)
    ap.add_argument("--bg-per-source", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--output-model", default="models/fast_matcher.joblib")
    ap.add_argument("--bg-cache-dir", default=None,
                    help="Directory with a cached bg_<n>_<seed>.tsv distractor sample")
    args = ap.parse_args()

    print("[*] Training fast matcher...")
    bg_override = None
    if args.bg_cache_dir:
        cache = os.path.join(args.bg_cache_dir, f"bg_{args.bg_per_source}_{args.seed}.tsv")
        if os.path.isfile(cache):
            bg_override = list(stream_tsv_records(cache))
            print(f"    background from cache: {len(bg_override):,} ({cache})")
    model, stats, _ = train_fast_model(
        args.train_dir, n_sample_s1=args.sample_s1, n_bg_per_source=args.bg_per_source,
        seed=args.seed, stride=args.stride, bg_override=bg_override,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_model)), exist_ok=True)
    joblib.dump({"model": model, "type": "gradient_boosting_fast",
                 "features": FAST_FEATURE_NAMES, "fitted": True, "stats": stats},
                args.output_model)
    print(f"[V] Saved fast matcher to {args.output_model}")
    print(f"    stats: {stats}")


if __name__ == "__main__":
    main()
