"""Honest dev evaluation for the fast vectorized path (zero target-side leakage).

Protocol (TRAIN files only; test set never touched):
  1. Deterministic S1 split: DEV = every Nth GT row (sub-sampled), TRAIN = rest.
  2. Model is trained ONLY on TRAIN S1s; dev S1 ids and dev true-target ids are
     excluded from training data entirely.
  3. DEV universe = dev true targets + reservoir-sampled distractors from the
     full train sources (an UPPER BOUND vs the ~10M-target test universe).
  4. Score = official macro-averaged per-S1 F0.5 over ALL dev entities.
"""

import argparse
import csv
import os
import random
import sys
import time
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np

from entitylink.data.loader import load_records_by_ids, stream_tsv_records
from entitylink.evaluation.metrics import evaluate_matching_predictions
from entitylink.fast.pipeline import FastMatcher, build_bundle
from entitylink.fast.selection import select_matches_mask

from train_fast import parse_ids, sample_background, train_fast_model


def cached_background(paths, n_bg, exclude_ids, seed, cache_dir=None):
    """Reservoir-sample background targets, caching the sample on disk if asked.

    Exclusions are applied after load/caching so a cache stays reusable.
    """
    exclude_ids = exclude_ids or set()
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"bg_{n_bg}_{seed}.tsv")
        if os.path.isfile(cache):
            recs = list(stream_tsv_records(cache))
            kept = [r for r in recs if r.entity_id not in exclude_ids]
            print(f"      background from cache: {len(kept):,} kept of {len(recs):,} ({cache})")
            return kept
    recs = sample_background(paths, n_bg, exclude_ids=None, seed=seed)
    if cache:
        with open(cache, "w", encoding="utf-8", newline="\n") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for r in recs:
                f.write(f"{r.entity_id}\t{r.business_name}\t{r.business_address}\t{r.country}\n")
        print(f"      background cached: {len(recs):,} ({cache})")
    return [r for r in recs if r.entity_id not in exclude_ids]


def split_dev(gt_path: str, n_dev: int, dev_mod: int = 10, stride: int = 1):
    """Return (dev_s1_ids, gt_dev, train_pool)."""
    dev_s1: List[str] = []
    gt_dev: Dict[str, Set[str]] = {}
    pool: List[Tuple[str, Set[str]]] = []
    with open(gt_path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for i, row in enumerate(reader):
            s1 = row[0].strip() if row else ""
            if not s1:
                continue
            ids = parse_ids(row[1] if len(row) > 1 else "")
            if i % dev_mod == 0:
                dev_s1.append(s1)
                gt_dev[s1] = ids
            else:
                pool.append((s1, ids))
    if stride > 1:
        dev_s1 = dev_s1[::stride][:n_dev]
    else:
        dev_s1 = dev_s1[:n_dev]
    gt_dev = {s: gt_dev[s] for s in dev_s1}
    assert not (set(dev_s1) & {s for s, _ in pool})
    return dev_s1, gt_dev, pool


def main():
    ap = argparse.ArgumentParser(description="Honest dev eval for the fast path.")
    ap.add_argument("--train-dir", default="data/train")
    ap.add_argument("--n-dev", type=int, default=20000)
    ap.add_argument("--dev-mod", type=int, default=10)
    ap.add_argument("--bg-dev", type=int, default=200000)
    ap.add_argument("--train-s1", type=int, default=80000)
    ap.add_argument("--bg-train", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--thresholds", default="0.45,0.5,0.55,0.6,0.65,0.7,0.75")
    ap.add_argument("--max-candidates", type=int, default=25)
    ap.add_argument("--gaps", default="0.15")
    ap.add_argument("--save-model", default=None)
    ap.add_argument("--load-model", default=None)
    ap.add_argument("--cache-dir", default="output/eval_cache")
    args = ap.parse_args()

    t0 = time.time()
    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(args.train_dir, "train_source1.tsv")
    s2_path = os.path.join(args.train_dir, "train_source2.tsv")
    s3_path = os.path.join(args.train_dir, "train_source3.tsv")

    print("[1/6] Split (dev = every %d-th GT row, stride-sampled)..." % args.dev_mod)
    dev_s1, gt_dev, pool = split_dev(gt_path, args.n_dev, args.dev_mod, 1)
    dev_targets = {t for ids in gt_dev.values() for t in ids}
    n_sing = sum(1 for s in dev_s1 if not gt_dev[s])
    print(f"      dev S1={len(dev_s1):,} (singletons={n_sing:,}) | dev true targets="
          f"{len(dev_targets):,} | train pool={len(pool):,}  [{time.time()-t0:.0f}s]")

    print("[2/6] Train model on TRAIN-only pool (dev ids excluded)...")
    cache = args.cache_dir or None
    if args.load_model and os.path.isfile(args.load_model):
        import joblib
        model = joblib.load(args.load_model)["model"]
        stats = {"loaded_from": args.load_model}
        print(f"      loaded model from {args.load_model}")
    else:
        rng = random.Random(args.seed)
        train_sample = pool[:]
        rng.shuffle(train_sample)
        gt_train = dict(train_sample[: args.train_s1])
        bg_train = cached_background([s2_path, s3_path], args.bg_train, None, args.seed,
                                     os.path.join(cache, "train") if cache else None)
        model, stats, _ = train_fast_model(
            args.train_dir, n_sample_s1=args.train_s1, n_bg_per_source=args.bg_train,
            exclude_s1=set(dev_s1), exclude_targets=dev_targets, seed=args.seed,
            chunk=args.chunk, gt_sample=gt_train, bg_override=bg_train,
        )
        if args.save_model:
            import joblib
            from entitylink.fast.features import FAST_FEATURE_NAMES
            joblib.dump({"model": model, "type": "gradient_boosting_fast",
                         "features": FAST_FEATURE_NAMES, "fitted": True, "stats": stats},
                        args.save_model)
            print(f"      saved model -> {args.save_model}")
        print(f"      trained: {stats}  [{time.time()-t0:.0f}s]")

    print("[3/6] Building DEV universe (true targets + distractors)...")
    dev_s2 = {t for t in dev_targets if t.startswith("S2-")}
    dev_s3 = {t for t in dev_targets if t.startswith("S3-")}
    s1_dev = load_records_by_ids(s1_path, set(dev_s1))
    t2 = load_records_by_ids(s2_path, dev_s2)
    t3 = load_records_by_ids(s3_path, dev_s3)
    assert len(t2) + len(t3) == len(dev_targets), (len(t2) + len(t3), len(dev_targets))
    bg = cached_background([s2_path, s3_path], args.bg_dev, set(t2) | set(t3), args.seed,
                           os.path.join(cache, f"dev_{args.n_dev}_{args.dev_mod}")
                           if cache else None)
    records = list(t2.values()) + list(t3.values()) + bg
    bundle = build_bundle(iter(records), capacity=len(records))
    print(f"      universe={bundle.count:,} (true={len(dev_targets):,} bg={len(bg):,}) "
          f"index keys={bundle.index.size:,}  [{time.time()-t0:.0f}s]")

    print("[4/6] Blocking + scoring dev entities...")
    matcher = FastMatcher(bundle, model=model, threshold=0.5, max_candidates=args.max_candidates)
    s1_list = list(s1_dev.values())
    missing = [s for s in dev_s1 if s not in s1_dev]
    if missing:
        print(f"      WARNING: {len(missing)} dev S1 ids not found in source1 file")
    chunks = []
    for start in range(0, len(s1_list), args.chunk):
        part = s1_list[start:start + args.chunk]
        res = matcher.score_chunk(part)
        chunks.append(([r.entity_id for r in part], res.owners, res.tids, res.scores))
    print(f"      raw pairs={matcher.stats['raw_pairs']:,} | candidate-capped pairs="
          f"{matcher.stats['scored_pairs']:,} | avg candidates/entity="
          f"{matcher.stats['scored_pairs']/max(len(s1_list),1):.2f}  [{time.time()-t0:.0f}s]")

    print("[5/6] Blocking recall over dev true matches...")
    row_of = bundle.id_to_row()
    covered = 0
    for s1_ids, owners, tids, _ in chunks:
        for i, sid in enumerate(s1_ids):
            true_rows = {row_of[t] for t in gt_dev.get(sid, ()) if t in row_of}
            if not true_rows:
                continue
            cand_rows = set(tids[owners == i].tolist())
            covered += len(true_rows & cand_rows)
    total_true = sum(len(v) for v in gt_dev.values())
    print(f"      covered={covered:,}/{total_true:,} = {covered/max(total_true,1):.4f}")

    print("[6/6] Threshold/gap sweep (macro-F0.5 over ALL dev)...")
    results = []
    thresholds = [float(x) for x in args.thresholds.split(",")]
    gaps = [float(x) for x in args.gaps.split(",")]
    for gap in gaps:
        for th in thresholds:
            preds: Dict[str, Set[str]] = {}
            for s1_ids, owners, tids, scores in chunks:
                mask = select_matches_mask(owners, scores, th, gap, 10)
                for pos in np.flatnonzero(mask):
                    preds.setdefault(s1_ids[int(owners[pos])], set()).add(
                        bundle.ids[int(tids[pos])])
            rep = evaluate_matching_predictions(gt_dev, preds)
            results.append((th, gap, rep))
            print(f"      thr={th:.2f} gap={gap:.2f}  P={rep.precision:.4f}  R={rep.recall:.4f}  "
                  f"F0.5={rep.f05:.4f}  pred_matches={rep.total_predicted_matches:,}  "
                  f"singleton_acc={rep.singleton_accuracy:.4f}")

    best = max(results, key=lambda x: x[2].f05)
    print("\n" + "=" * 64)
    print(f" BEST dev macro-F0.5 = {best[2].f05:.4f} @ threshold {best[0]:.2f}, gap {best[1]:.2f} "
          f"(P={best[2].precision:.4f}, R={best[2].recall:.4f})")
    print(f" (UPPER BOUND: dev universe has {bundle.count:,} targets vs ~10M on test)")
    print("=" * 64)


if __name__ == "__main__":
    main()
