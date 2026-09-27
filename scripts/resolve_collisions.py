"""Resolve multi-parent collisions in matching_results.tsv using Rapidfuzz text similarity.

Enforces the fundamental entity resolution invariant:
An S2 or S3 business record belongs to at most ONE canonical S1 entity.
For any S2/S3 record claimed by multiple S1 entities, this script evaluates
the exact name and address similarity against each claimant and assigns it
exclusively to its best parent, pruning all spurious false merges.
"""

import sys
import os
import time
from collections import defaultdict
import pandas as pd
from rapidfuzz import fuzz

def main():
    t0 = time.time()
    in_path = "output/matching_results.tsv"
    out_path = "output/matching_results_single_parent.tsv"
    test_dir = "student_resource/dataset/test"

    print("[1/5] Loading current matching results...")
    s1_rows = []
    mid_to_s1s = defaultdict(list)
    s1_to_mids = {}

    with open(in_path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            s1, _, rest = line.partition("\t")
            s1_rows.append(s1)
            mids = rest.strip().split(",") if rest.strip() else []
            s1_to_mids[s1] = mids
            for m in mids:
                mid_to_s1s[m].append(s1)

    multi_mids = {m: s1s for m, s1s in mid_to_s1s.items() if len(s1s) > 1}
    print(f"      Total S1: {len(s1_rows):,}")
    print(f"      Unique matched IDs: {len(mid_to_s1s):,}")
    print(f"      Multi-parent collided IDs: {len(multi_mids):,} (will be resolved)")

    needed_s1 = set()
    for s1s in multi_mids.values():
        needed_s1.update(s1s)

    needed_s2 = {m for m in multi_mids if m.startswith("S2-")}
    needed_s3 = {m for m in multi_mids if m.startswith("S3-")}

    print(f"[2/5] Reading text for {len(needed_s1):,} collided S1 and {len(multi_mids):,} collided S2/S3...")
    s1_text = {}
    for ch in pd.read_csv(f"{test_dir}/test_source1.tsv", sep="\t", dtype=str,
                          keep_default_na=False, chunksize=500_000,
                          usecols=["entity_id", "business_name", "business_address"]):
        ch = ch[ch.entity_id.isin(needed_s1)]
        for eid, n, a in zip(ch.entity_id, ch.business_name, ch.business_address):
            s1_text[eid] = (n.lower().strip(), a.lower().strip())

    mid_text = {}
    for src, needed in (("source2", needed_s2), ("source3", needed_s3)):
        for ch in pd.read_csv(f"{test_dir}/test_{src}.tsv", sep="\t", dtype=str,
                              keep_default_na=False, chunksize=500_000,
                              usecols=["entity_id", "business_name", "business_address"]):
            ch = ch[ch.entity_id.isin(needed)]
            for eid, n, a in zip(ch.entity_id, ch.business_name, ch.business_address):
                mid_text[eid] = (n.lower().strip(), a.lower().strip())

    print(f"      Loaded texts: S1={len(s1_text):,}, S2/3={len(mid_text):,} [{time.time()-t0:.0f}s]")

    print("[3/5] Disambiguating multi-parent collisions via Rapidfuzz similarity...")
    # For each collided mid, find which S1 is best
    resolved_parent = {}
    pruned_false_merges = 0

    for mid, s1s in multi_mids.items():
        m_name, m_addr = mid_text.get(mid, ("", ""))
        best_score = -1.0
        best_s1 = None
        for s1 in s1s:
            s_name, s_addr = s1_text.get(s1, ("", ""))
            # Weighted combination of name token-set ratio + address token-set ratio
            n_sim = fuzz.token_set_ratio(m_name, s_name)
            a_sim = fuzz.token_set_ratio(m_addr, s_addr)
            # Address agreement is heavily informative for duplicate names
            score = 0.6 * n_sim + 0.4 * a_sim
            if score > best_score:
                best_score = score
                best_s1 = s1

        # If best match is strong enough, assign exclusively to best_s1; otherwise drop as spurious noise
        if best_score >= 45.0:
            resolved_parent[mid] = best_s1
            pruned_false_merges += (len(s1s) - 1)
        else:
            # All claimants were weak/generic: drop completely
            resolved_parent[mid] = None
            pruned_false_merges += len(s1s)

    print(f"      Resolved: pruned {pruned_false_merges:,} duplicate false merge assignments [{time.time()-t0:.0f}s]")

    print("[4/5] Reconstructing clean single-parent matching results...")
    new_s1_to_mids = defaultdict(list)
    for mid, s1s in mid_to_s1s.items():
        if len(s1s) == 1:
            new_s1_to_mids[s1s[0]].append(mid)
        else:
            best_s1 = resolved_parent.get(mid)
            if best_s1 is not None:
                new_s1_to_mids[best_s1].append(mid)

    print(f"[5/5] Writing {out_path} ...")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_rows:
            mids = new_s1_to_mids.get(s1, [])
            f.write(f"{s1}\t{','.join(mids)}\n")

    print(f"Done in {time.time()-t0:.0f}s. Saved to {out_path}")

if __name__ == "__main__":
    main()
