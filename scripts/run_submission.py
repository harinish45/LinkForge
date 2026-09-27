"""Produce the final test-set submission with the fast vectorized pipeline.

Outputs (competition format, TAB-separated):
  * matching_results.tsv   - source1_entity_id, matched_entity_ids   (scored on the leaderboard)
  * candidate_pairs.tsv    - source1_entity_id, candidate_entity_ids (set scored by the model)

Runs are resumable: per-chunk state (rows done + output byte offsets) is written
to the work directory, so an interrupted run continues where it stopped.
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np

from entitylink.data.loader import stream_tsv_records
from entitylink.data.schema import EntityRecord
from entitylink.fast.index import BlockIndex
from entitylink.fast.payload import PayloadStore
from entitylink.fast.pipeline import FastMatcher, TargetBundle, build_target_bundle, chunk_output_rows
from entitylink.models.classifier import MLPairMatcher

MATCH_HEADER = "source1_entity_id\tmatched_entity_ids\n"
CAND_HEADER = "source1_entity_id\tcandidate_entity_ids\n"
STATE_FILE = "run_state.json"


def bundle_paths(work_dir: str):
    return {
        "keys": os.path.join(work_dir, "index_keys.npy"),
        "ids": os.path.join(work_dir, "index_ids.npy"),
        "targets": os.path.join(work_dir, "target_ids.txt"),
        "meta": os.path.join(work_dir, "bundle_meta.json"),
    }


def save_bundle(bundle: TargetBundle, work_dir: str) -> None:
    p = bundle_paths(work_dir)
    np.save(p["keys"], bundle.index.keys)
    np.save(p["ids"], bundle.index.ids)
    with open(p["targets"], "w", encoding="utf-8", newline="\n") as f:
        for start in range(0, len(bundle.ids), 1_000_000):
            f.write("\n".join(bundle.ids[start:start + 1_000_000]))
            f.write("\n")
    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump({"count": int(bundle.count), "keys": int(bundle.index.size),
                   "store_prefix": "t_"}, f)


def load_bundle(work_dir: str) -> TargetBundle:
    p = bundle_paths(work_dir)
    with open(p["meta"], encoding="utf-8") as f:
        meta = json.load(f)
    index = BlockIndex(np.load(p["keys"], mmap_mode="r"), np.load(p["ids"], mmap_mode="r"))
    store = PayloadStore.open(work_dir, meta["count"], prefix=meta.get("store_prefix", "t_"))
    with open(p["targets"], encoding="utf-8") as f:
        target_ids = [line.rstrip("\n") for line in f if line.strip()]
    assert len(target_ids) == meta["count"], (len(target_ids), meta["count"])
    return TargetBundle(store=store, index=index, ids=target_ids, count=meta["count"])


def read_state(work_dir: str) -> Optional[dict]:
    path = os.path.join(work_dir, STATE_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_state(work_dir: str, state: dict) -> None:
    with open(os.path.join(work_dir, STATE_FILE), "w", encoding="utf-8") as f:
        json.dump(state, f)


def main():
    ap = argparse.ArgumentParser(description="Run full test inference -> submission TSVs.")
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--work-dir", default="output/fast_work")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--model", default="models/fast_matcher.joblib")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--confidence-gap", type=float, default=0.15)
    ap.add_argument("--max-matches", type=int, default=10)
    ap.add_argument("--max-candidates", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N S1 rows (smoke test)")
    ap.add_argument("--rebuild-bundle", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    s3_path = os.path.join(args.test_dir, "test_source3.tsv")
    match_path = os.path.join(args.out_dir, "matching_results.tsv")
    cand_path = os.path.join(args.out_dir, "candidate_pairs.tsv")

    t0 = time.time()
    meta_path = bundle_paths(args.work_dir)["meta"]
    if args.rebuild_bundle or not os.path.isfile(meta_path):
        print(f"[1/3] Building target bundle (S2+S3) into {args.work_dir} ...", flush=True)
        bundle = build_target_bundle([s2_path, s3_path], work_dir=args.work_dir)
        save_bundle(bundle, args.work_dir)
        print(f"      bundle built: {bundle.count:,} targets, {bundle.index.size:,} index keys "
              f"[{time.time()-t0:.0f}s]", flush=True)
    else:
        print(f"[1/3] Loading cached target bundle from {args.work_dir} ...", flush=True)
        bundle = load_bundle(args.work_dir)
        print(f"      bundle loaded: {bundle.count:,} targets, {bundle.index.size:,} index keys "
              f"[{time.time()-t0:.0f}s]", flush=True)

    matcher = FastMatcher(bundle, model_path=args.model, threshold=args.threshold,
                          confidence_gap=args.confidence_gap, max_matches=args.max_matches,
                          max_candidates=args.max_candidates)

    state = read_state(args.work_dir) if args.resume else None
    if state:
        rows_done = int(state.get("rows_done", 0))
        for path, key in ((match_path, "match_bytes"), (cand_path, "cand_bytes")):
            if os.path.isfile(path) and state.get(key):
                os.truncate(path, int(state[key]))
        print(f"      resuming after {rows_done:,} S1 rows", flush=True)
    else:
        rows_done = 0
        for path, header in ((match_path, MATCH_HEADER), (cand_path, CAND_HEADER)):
            if os.path.isfile(path):
                os.remove(path)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(header)
        write_state(args.work_dir, {"rows_done": 0, "match_bytes": os.path.getsize(match_path),
                                    "cand_bytes": os.path.getsize(cand_path)})

    limit = (rows_done + args.limit) if args.limit else None
    print(f"[2/3] Scoring S1 rows (threshold={args.threshold}, max_candidates="
          f"{args.max_candidates}) -> {match_path}", flush=True)
    fm = open(match_path, "a", encoding="utf-8", newline="\n")
    fc = open(cand_path, "a", encoding="utf-8", newline="\n")
    chunk: List[EntityRecord] = []
    rows_written = rows_done
    matches_found = 0
    skipped = 0
    last_log = time.time()
    try:
        for rec in stream_tsv_records(s1_path):
            if skipped < rows_done:
                skipped += 1
                continue
            chunk.append(rec)
            if len(chunk) >= args.chunk:
                rows_written, matches_found = process_chunk(
                    matcher, chunk, fm, fc, args.work_dir, rows_written, matches_found)
                chunk = []
                if limit is not None and rows_written >= limit:
                    break
                if time.time() - last_log > 20:
                    elapsed = max(time.time() - t0, 1e-9)
                    print(f"      {rows_written:,} rows | {rows_written/elapsed:,.0f} rows/s | "
                          f"raw_pairs={matcher.stats['raw_pairs']:,} | "
                          f"scored={matcher.stats['scored_pairs']:,} | matches={matches_found:,}",
                          flush=True)
                    last_log = time.time()
        if chunk and (limit is None or rows_written < limit):
            rows_written, matches_found = process_chunk(
                matcher, chunk, fm, fc, args.work_dir, rows_written, matches_found)
    finally:
        fm.close()
        fc.close()

    print(f"[3/3] Done. S1 rows written: {rows_written:,} | matched pairs: {matches_found:,} | "
          f"elapsed {time.time()-t0:.0f}s", flush=True)
    print(f"      matching:  {match_path} ({os.path.getsize(match_path)/1e6:.1f} MB)")
    print(f"      candidates:{cand_path} ({os.path.getsize(cand_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()


