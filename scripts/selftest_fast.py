"""Self-test for the fast vectorized path (payload/index/features/selection)."""

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np

from entitylink.data.loader import stream_tsv_records
from entitylink.fast.fastnorm import encode_record, normalize_name, normalize_addr
from entitylink.fast.features import FAST_FEATURE_NAMES
from entitylink.fast.pipeline import FastMatcher, build_bundle
from entitylink.fast.selection import select_matches_mask
from entitylink.models.calibrator import DecisionCalibrator
from entitylink.models.classifier import MLPairMatcher

OK = True


def check(label, cond):
    global OK
    OK = OK and bool(cond)
    print(f"  [{'ok' if cond else 'FAIL'}] {label}")


def main():
    s2 = list(itertools.islice(stream_tsv_records("data/train/train_source2.tsv"), 3000))
    s3 = list(itertools.islice(stream_tsv_records("data/train/train_source3.tsv"), 3000))
    s1 = list(itertools.islice(stream_tsv_records("data/train/train_source1.tsv"), 400))

    print("[1] normalization parity")
    n1, t1 = normalize_name("Orelee's Barbershop Inc.")
    check(f"name normalize -> {n1!r}", n1 == "orelee s barbershop")
    a1, at, an = normalize_addr("17560 Ellis Road, Tahlequah, OK")
    check(f"addr normalize -> {a1!r}", a1 == "17560 ellis rd tahlequah ok")
    check("addr nums", an == ["17560"])
    _, _, an2 = normalize_addr("2621 Cotten Road, Tyler, TX 75701")
    check("postal extraction candidate", "75701" in an2)

    print("[2] payload store roundtrip")
    bundle = build_bundle(iter(s2 + s3), capacity=len(s2) + len(s3))
    check(f"bundle rows={bundle.count} keys={bundle.index.size}", bundle.count == 6000)
    check("bit fields shape", bundle.store.bits["name_tok_bits"].shape == (6000, 4))

    print("[3] exact-match features")
    matcher = FastMatcher(bundle, model=MLPairMatcher())
    s1_store, owners, tids = matcher.candidates_for(s1)
    print(f"  S1={len(s1)} candidate pairs={owners.shape[0]} "
          f"avg={(owners.shape[0]/max(len(s1),1)):.2f}")
    if owners.shape[0] > 0:
        X = matcher.features_for(s1_store, bundle.store, owners, tids)
        check(f"feature matrix {X.shape}", X.shape == (owners.shape[0], len(FAST_FEATURE_NAMES)))
        check("features finite", bool(np.isfinite(X).all()))
        idx = {name: i for i, name in enumerate(FAST_FEATURE_NAMES)}
        check("jaccard in [0,1]",
              bool((X[:, idx["name_tok_jaccard"]] >= 0).all()
                   and (X[:, idx["name_tok_jaccard"]] <= 1).all()))
        # Manual check on synthetic identical vs unrelated records
        p_a, _ = encode_record("Zephay Labs Inc", "2621 Cotten Road, Tyler, TX", "US")
        p_b, _ = encode_record("Zephay Labs", "2621 Cotten Road, Tyler, TX", "US")
        p_c, _ = encode_record("Totally Different Name", "9 Nowhere Ave, Fargo, ND", "US")
        from entitylink.fast.payload import PayloadStore
        store = PayloadStore.create(3)
        store.add(p_a); store.add(p_b); store.add(p_c); store.finalize()
        owned = np.array([0, 0], dtype=np.int64)
        targ = np.array([1, 2], dtype=np.int64)
        Xm = FastMatcher.features_for(store, store, owned, targ)
        check("identical-ish name exact=1", Xm[0, idx["name_exact"]] == 1.0)
        check("identical-ish addr exact=1", Xm[0, idx["addr_exact"]] == 1.0)
        check("unrelated name exact=0", Xm[1, idx["name_exact"]] == 0.0)
        check("unrelated low sim", Xm[1, idx["combined_geom_sim"]] < 0.2)

    print("[4] selection equivalence vs DecisionCalibrator")
    rng = random.Random(7)
    cal = DecisionCalibrator(threshold=0.7, confidence_gap=0.15, max_matches=10)
    mismatches = 0
    for _ in range(400):
        k = rng.randint(0, 12)
        scores = np.array([round(rng.uniform(0.0, 1.0), 3) for _ in range(k)], dtype=np.float64)
        cands = [f"C{i}" for i in range(k)]
        expected = set(cal.select_matches(list(zip(cands, scores.tolist()))))
        mask = select_matches_mask(np.zeros(k, dtype=np.int64), scores, 0.7, 0.15, 10)
        got = {cands[i] for i in range(k) if mask[i]}
        if got != expected:
            mismatches += 1
            if mismatches < 3:
                print(f"   mismatch: scores={scores.tolist()} expected={expected} got={got}")
    check(f"400 random cases match ({mismatches} mismatches)", mismatches == 0)

    print("[5] multi-group selection")
    group = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    score = np.array([0.9, 0.85, 0.50, 0.60, 0.59], dtype=np.float64)
    mask = select_matches_mask(group, score, 0.7, 0.15, 10)
    check("group0 keeps top+gap", mask.tolist() == [True, True, False, False, False])
    mask2 = select_matches_mask(group, score, 0.55, 0.15, 10)
    check("group1 accepts both above floor", mask2[3] and mask2[4])

    print("\nRESULT:", "ALL PASS" if OK else "FAILURES PRESENT")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
