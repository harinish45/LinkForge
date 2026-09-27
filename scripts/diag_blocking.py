"""Diagnose which blocking-key classes cover dev true matches (and which miss).

For every dev ground-truth pair it inspects the keys both sides generate and
reports the per-class coverage plus the "no shared key" share, so key design can
be improved with evidence instead of guesswork.
"""

import argparse
import csv
import os
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from entitylink.data.loader import load_records_by_ids
from entitylink.fast.fastnorm import KEY_CLASSES, encode_record

CLASS_NAME = {v: k for k, v in KEY_CLASSES.items()}


def keys_by_class(name: str, addr: str, country: str) -> Dict[int, Set[int]]:
    _, keys = encode_record(name, addr, country)
    out: Dict[int, Set[int]] = {}
    for k in keys:
        out.setdefault(k >> 28, set()).add(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="data/train")
    ap.add_argument("--dev-mod", type=int, default=10)
    ap.add_argument("--n-dev", type=int, default=20000)
    ap.add_argument("--max-pairs", type=int, default=200000)
    args = ap.parse_args()

    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(args.train_dir, "train_source1.tsv")
    s2_path = os.path.join(args.train_dir, "train_source2.tsv")
    s3_path = os.path.join(args.train_dir, "train_source3.tsv")

    dev = []
    with open(gt_path, encoding="utf-8", errors="replace") as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)
        for i, row in enumerate(r):
            if i % args.dev_mod:
                continue
            s1 = row[0].strip()
            ids = {x.strip() for x in (row[1] if len(row) > 1 else "").split(",") if x.strip()}
            if ids:
                dev.append((s1, ids))
            if len(dev) >= args.n_dev:
                break

    need_s1 = {s for s, _ in dev}
    need_t = {t for _, ids in dev for t in ids}
    s1_recs = load_records_by_ids(s1_path, need_s1)
    t_recs = load_records_by_ids(s2_path, {t for t in need_t if t.startswith("S2-")})
    t_recs.update(load_records_by_ids(s3_path, {t for t in need_t if t.startswith("S3-")}))
    print(f"dev S1={len(s1_recs):,} dev targets={len(t_recs):,}")

    per_class = Counter()
    pair_count = 0
    covered_by_any = 0
    missing_examples: List[Tuple[str, str]] = []
    for s1, ids in dev:
        rec1 = s1_recs.get(s1)
        if rec1 is None:
            continue
        k1 = keys_by_class(rec1.business_name, rec1.business_address, rec1.country)
        k1_flat = set().union(*k1.values()) if k1 else set()
        for tid in ids:
            rec2 = t_recs.get(tid)
            if rec2 is None:
                continue
            pair_count += 1
            k2 = keys_by_class(rec2.business_name, rec2.business_address, rec2.country)
            k2_flat = set().union(*k2.values()) if k2 else set()
            shared_classes = {c for c in k1 if c in k2 and (k1[c] & k2[c])}
            if shared_classes:
                covered_by_any += 1
                for c in shared_classes:
                    per_class[CLASS_NAME.get(c, str(c))] += 1
            elif len(missing_examples) < 8:
                missing_examples.append((f"{s1}: {rec1.business_name!r} / {rec1.business_address!r}",
                                         f"{tid}: {rec2.business_name!r} / {rec2.business_address!r}"))
            if pair_count >= args.max_pairs:
                break
        if pair_count >= args.max_pairs:
            break

    print(f"\ntrue pairs inspected: {pair_count:,}")
    print(f"covered by >=1 shared key class: {covered_by_any:,} "
          f"({covered_by_any/max(pair_count,1):.4f})")
    print("\nper-class coverage (share of true pairs where that class matches):")
    for cls, n in per_class.most_common():
        print(f"  {cls:<18} {n:>8,}  {n/max(pair_count,1):.4f}")
    print("\nexample uncovered pairs:")
    for a, b in missing_examples:
        print(f"  - {a}\n      vs {b}")


if __name__ == "__main__":
    main()
