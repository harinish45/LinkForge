"""Honest eval part 1: helpers (split, background, parsing).

Protocol (TRAIN files only; test never touched):
 1. Deterministic S1 split: DEV = every 10th GT row, TRAIN = rest.
 2. Train ONLY on TRAIN S1 targets (+ tail background). Assert zero overlap.
 3. DEV universe = dev true targets + tail distractors (UPPER BOUND caveat).
 4. Macro-F0.5 over ALL dev entities (missing preds count as empty).
"""

import argparse
import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from entitylink.data.loader import load_records_by_ids, stream_tsv_records
from entitylink.blocking.candidate_generator import generate_candidates
from entitylink.models.classifier import MLPairMatcher
from entitylink.models.calibrator import DecisionCalibrator
from entitylink.evaluation.metrics import evaluate_matching_predictions
from entitylink.evaluation.metrics import evaluate_blocking_performance

TAIL_SKIP = 200000


def parse_ids(raw):
    raw = (raw or "").strip()
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def tail_bg(path, store, need, skip=TAIL_SKIP):
    got = 0
    for i, rec in enumerate(stream_tsv_records(path)):
        if i < skip:
            continue
        if got >= need:
            break
        if rec.entity_id not in store:
            store[rec.entity_id] = rec
            got += 1
    return got


def split_s1(gt_path, n_dev):
    dev_s1, gt_dev, pool = [], {}, []
    with open(gt_path, encoding="utf-8", errors="replace") as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)
        for i, row in enumerate(r):
            s1 = row[0].strip()
            ids = parse_ids(row[1] if len(row) > 1 else "")
            if i % 10 == 0:
                if len(dev_s1) < n_dev:
                    dev_s1.append(s1)
                    gt_dev[s1] = ids
            else:
                pool.append((s1, ids))
    assert not (set(dev_s1) & {s for s, _ in pool})
    return dev_s1, gt_dev, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="data/train")
    ap.add_argument("--n-dev", type=int, default=500)
    ap.add_argument("--bg-dev", type=int, default=5000)
    ap.add_argument("--thresholds", default="0.5,0.62,0.7,0.8")
    args = ap.parse_args()
    t0 = time.time()
    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(args.train_dir, "train_source1.tsv")
    s2_path = os.path.join(args.train_dir, "train_source2.tsv")
    s3_path = os.path.join(args.train_dir, "train_source3.tsv")

    print("[1/7] S1 split...")
    dev_s1, gt_dev, pool = split_s1(gt_path, args.n_dev)
    nds = sum(1 for s in dev_s1 if not gt_dev[s])
    print(f"      DEV={len(dev_s1)} ({nds} singletons), pool={len(pool)}")

    print("[2/7] TRAIN-only load...")
    random.seed(42)
    sample = random.sample(pool, min(1500, len(pool)))
    tr_gt = {s: ids for s, ids in sample}
    tr_s1 = {s for s, _ in sample}
    tr_s2 = {t for ids in tr_gt.values() for t in ids if t.startswith("S2-")}
    tr_s3 = {t for ids in tr_gt.values() for t in ids if t.startswith("S3-")}
    s1_tr = load_records_by_ids(s1_path, tr_s1)
    s2_tr = load_records_by_ids(s2_path, tr_s2)
    s3_tr = load_records_by_ids(s3_path, tr_s3)
    b2 = tail_bg(s2_path, s2_tr, 1000)
    b3 = tail_bg(s3_path, s3_tr, 1000)
    leak = (tr_s2 | tr_s3) & {t for ids in gt_dev.values() for t in ids}
    print(f"      S1={len(s1_tr)} S2t={len(tr_s2)} S3t={len(tr_s3)} bg=({b2},{b3})")
    print(f"      leak(dev-targets-as-train-pos)={len(leak)} MUST be 0")
    assert len(leak) == 0

    from train import build_training_pairs
    targets_tr = dict(s2_tr)
    targets_tr.update(s3_tr)
    cands_tr = generate_candidates(s1_tr, s2_tr, s3_tr)
    pairs, labels = build_training_pairs(s1_tr, targets_tr, tr_gt, cands_tr)
    print(f"[3/7] pairs={len(pairs)} pos={sum(labels)} neg={len(labels)-sum(labels)}")
    matcher = MLPairMatcher(model_type="gradient_boosting", random_state=42)
    matcher.fit(pairs, labels)

    print("[4/7] DEV universe...")
    dev_s2 = {t for ids in gt_dev.values() for t in ids if t.startswith("S2-")}
    dev_s3 = {t for ids in gt_dev.values() for t in ids if t.startswith("S3-")}
    s1_dev = load_records_by_ids(s1_path, set(dev_s1))
    s2_dev = load_records_by_ids(s2_path, dev_s2)
    s3_dev = load_records_by_ids(s3_path, dev_s3)
    g2 = tail_bg(s2_path, s2_dev, args.bg_dev)
    g3 = tail_bg(s3_path, s3_dev, args.bg_dev)
    print(f"      true S2={len(dev_s2)} S3={len(dev_s3)} S1found={len(s1_dev)}/{len(dev_s1)}")
    print(f"      distractors=({g2},{g3}) universe=({len(s2_dev)},{len(s3_dev)})")

    print("[5/7] Blocking...")
    cands = generate_candidates(s1_dev, s2_dev, s3_dev)
    blk = evaluate_blocking_performance(gt_dev, cands,
                                        total_target_records=len(s2_dev) + len(s3_dev))
    print(f"      recall={blk.candidate_recall:.4f} covered={blk.covered_true_matches}/{blk.total_true_matches} "
          f"avg={blk.avg_candidates_per_entity:.2f} max={blk.max_candidates_per_entity}")

    print("[6/7] Sweep (macro-F0.5 over ALL dev)...")
    targets = dict(s2_dev)
    targets.update(s3_dev)
    scores = {}
    for s1, cids in cands.items():
        rec = s1_dev.get(s1)
        if rec is None:
            continue
        out = []
        for cid in cids:
            t = targets.get(cid)
            if t is not None:
                out.append((cid, matcher.score_pair(rec, t)))
        scores[s1] = out
    best = (-1.0, None, None)
    for th in [float(x) for x in args.thresholds.split(",")]:
        cal = DecisionCalibrator(threshold=th)
        preds = {s: set(cal.select_matches(scores.get(s, []))) for s in dev_s1}
        rep = evaluate_matching_predictions(gt_dev, preds)
        print(f"      th={th:.2f} P={rep.precision:.4f} R={rep.recall:.4f} F0.5={rep.f05:.4f} "
              f"F1={rep.f1:.4f} sing={rep.singleton_accuracy:.4f} "
              f"TP={rep.true_positives} FP={rep.false_positives} FN={rep.false_negatives}")
        if rep.f05 > best[0]:
            best = (rep.f05, th, rep)
    f05, th, rep = best
    print(f"[7/7] BEST th={th} F0.5={f05:.4f} P={rep.precision:.4f} R={rep.recall:.4f} in {time.time()-t0:.1f}s")
    print("CAVEAT: sampled distractors => UPPER BOUND vs full 10M deployment.")
    miss = len(dev_s1) - len(s1_dev)
    if miss:
        print(f"NOTE: {miss} DEV S1 missing in train_source1.tsv; scored as empty.")


if __name__ == "__main__":
    main()

