"""Full-scale matching pipeline: hashed index build + chunked vectorized scoring.

This is the scale-out path used for the real 1.73M-entity test submission. It
mirrors the semantics of the reference pipeline (blocking -> pair features ->
calibrated decisions) but computes everything with numpy so that ~10M targets
can be indexed and scored within laptop memory/time budgets.
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from entitylink.data.loader import stream_tsv_records
from entitylink.data.schema import EntityRecord
from entitylink.fast.fastnorm import encode_record
from entitylink.fast.features import cheap_score, extract_fast_features, gather_pair_payload
from entitylink.fast.index import CAPS_BY_CLASS, BlockIndex, BlockIndexBuilder
from entitylink.fast.payload import PayloadStore
from entitylink.fast.selection import select_matches_mask, topk_per_group_mask
from entitylink.models.classifier import MLPairMatcher

KEYS_PER_ROW_ESTIMATE = 8
ADD_BATCH = 200_000


@dataclass
class TargetBundle:
    """Payload store + blocking index + entity ids for one target universe."""

    store: PayloadStore
    index: BlockIndex
    ids: List[str]
    count: int

    def id_to_row(self) -> Dict[str, int]:
        return {eid: i for i, eid in enumerate(self.ids)}


def count_rows(path: str) -> int:
    """Count data rows (excluding header) fast."""
    with open(path, "rb") as f:
        f.readline()
        return sum(1 for _ in f)


def build_bundle(record_iter: Iterable[EntityRecord], capacity: int,
                 work_dir: Optional[str] = None, prefix: str = "t_") -> TargetBundle:
    """Encode records into a payload store + hashed blocking index."""
    store = PayloadStore.create(capacity, base_path=work_dir, prefix=prefix)
    builder = BlockIndexBuilder(capacity * KEYS_PER_ROW_ESTIMATE)
    ids: List[str] = []
    batch_keys: List[List[int]] = []
    count = 0
    for rec in record_iter:
        payload, keys = encode_record(rec.business_name, rec.business_address, rec.country)
        store.add(payload)
        ids.append(rec.entity_id)
        batch_keys.append(keys)
        count += 1
        if len(batch_keys) >= ADD_BATCH:
            builder.add_batch(batch_keys, count - len(batch_keys))
            batch_keys = []
    store.finalize()
    if batch_keys:
        builder.add_batch(batch_keys, count - len(batch_keys))
    index = builder.finalize()
    return TargetBundle(store=store, index=index, ids=ids, count=count)


def build_target_bundle(paths: Sequence[str], work_dir: Optional[str] = None,
                        prefix: str = "t_") -> TargetBundle:
    """Build the target bundle by streaming one or more source TSV files."""
    total = sum(count_rows(p) for p in paths)
    return build_bundle(_stream_all(paths), capacity=total, work_dir=work_dir, prefix=prefix)


def _stream_all(paths: Sequence[str]) -> Iterator[EntityRecord]:
    for p in paths:
        for rec in stream_tsv_records(p):
            yield rec


@dataclass
class ChunkScores:
    """Scored candidate pairs for one chunk of S1 entities (candidate-capped)."""

    n_s1: int
    owners: np.ndarray   # row index of the S1 entity inside the chunk
    tids: np.ndarray     # target row index in the bundle
    scores: np.ndarray   # model probability
    selected: np.ndarray  # bool mask: accepted final match

    @property
    def n_pairs(self) -> int:
        return int(self.owners.shape[0])


class FastMatcher:
    """Blocking + vectorized scoring + calibrator-equivalent decision selection."""

    def __init__(self, bundle: TargetBundle, model_path: Optional[str] = None,
                 model=None, threshold: float = 0.70, confidence_gap: float = 0.15,
                 max_matches: int = 10, max_candidates: int = 25):
        self.bundle = bundle
        if model is None:
            if not model_path:
                raise ValueError("Either model_path or model must be provided.")
            model = MLPairMatcher.load(model_path).model
        self.model = model
        self.threshold = threshold
        self.confidence_gap = confidence_gap
        self.max_matches = max_matches
        self.max_candidates = max_candidates
        self.stats = {"raw_pairs": 0, "scored_pairs": 0}

    # -- S1 side encoding --------------------------------------------------
    @staticmethod
    def encode_s1(records: Sequence[EntityRecord]) -> Tuple[PayloadStore, np.ndarray]:
        store = PayloadStore.create(len(records))
        key_lists: List[List[int]] = []
        for rec in records:
            payload, keys = encode_record(rec.business_name, rec.business_address, rec.country)
            store.add(payload)
            key_lists.append(keys)
        store.finalize()
        width = max((len(k) for k in key_lists), default=1)
        mat = np.zeros((len(records), width), dtype=np.int64)
        for i, ks in enumerate(key_lists):
            if ks:
                mat[i, :len(ks)] = ks
        return store, mat

    # -- blocking ----------------------------------------------------------
    def gather_pairs(self, s1_store: PayloadStore, key_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n, width = key_mat.shape
        valid = key_mat > 0
        empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        if not valid.any():
            return empty
        q = key_mat[valid]
        owner = np.repeat(np.arange(n, dtype=np.int64), width)[valid.ravel()]
        lo, hi = self.bundle.index.ranges(q)
        lengths = hi - lo
        caps = CAPS_BY_CLASS[(q >> 28) & 15]
        accept = (lengths > 0) & (lengths <= caps)
        if not accept.any():
            return empty
        owners, tids = self.bundle.index.expand(owner[accept], lo[accept], hi[accept])
        self.stats["raw_pairs"] += int(owners.shape[0])
        packed = (owners.astype(np.uint64) << np.uint64(32)) | tids.astype(np.uint64)
        packed = np.unique(packed)
        return (packed >> np.uint64(32)).astype(np.int64), (packed & np.uint64(0xFFFFFFFF)).astype(np.int64)

    # -- scoring + selection ----------------------------------------------
    def candidates_for(self, records: Sequence[EntityRecord]):
        """Encode S1 records, block against the target index, cap per entity."""
        s1_store, key_mat = self.encode_s1(records)
        owners, tids = self.gather_pairs(s1_store, key_mat)
        if owners.shape[0] > 0:
            a = gather_pair_payload(s1_store, owners)
            b = gather_pair_payload(self.bundle.store, tids)
            keep = topk_per_group_mask(owners, cheap_score(a, b), self.max_candidates)
            owners, tids = owners[keep], tids[keep]
        return s1_store, owners, tids

    @staticmethod
    def features_for(s1_store: PayloadStore, bundle_store: PayloadStore,
                     owners: np.ndarray, tids: np.ndarray) -> np.ndarray:
        """Fast feature matrix for (S1 row, target row) pairs."""
        if owners.shape[0] == 0:
            from entitylink.fast.features import FAST_FEATURE_NAMES
            return np.zeros((0, len(FAST_FEATURE_NAMES)), dtype=np.float64)
        a = gather_pair_payload(s1_store, owners)
        b = gather_pair_payload(bundle_store, tids)
        return extract_fast_features(a, b)

    def score_chunk(self, records: Sequence[EntityRecord]) -> ChunkScores:
        s1_store, owners, tids = self.candidates_for(records)
        if owners.shape[0] > 0:
            X = self.features_for(s1_store, self.bundle.store, owners, tids)
            scores = self.model.predict_proba(X)[:, 1]
            selected = select_matches_mask(owners, scores, self.threshold,
                                           self.confidence_gap, self.max_matches)
        else:
            scores = np.zeros(0, dtype=np.float64)
            selected = np.zeros(0, dtype=bool)
        self.stats["scored_pairs"] += int(owners.shape[0])
        return ChunkScores(n_s1=len(records), owners=owners, tids=tids,
                           scores=scores, selected=selected)


def chunk_output_rows(s1_ids: Sequence[str], res: ChunkScores,
                      target_ids: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Format one chunk's candidates/matches as ready-to-write TSV lines."""
    order = np.argsort(res.owners, kind="stable")
    cand_map: Dict[int, List[str]] = {}
    match_map: Dict[int, List[str]] = {}
    for pos in order:
        o = int(res.owners[pos])
        tid = target_ids[int(res.tids[pos])]
        cand_map.setdefault(o, []).append(tid)
        if res.selected[pos]:
            match_map.setdefault(o, []).append(tid)
    cand_rows: List[str] = []
    match_rows: List[str] = []
    for i, sid in enumerate(s1_ids):
        cand_rows.append(f"{sid}\t{','.join(sorted(set(cand_map.get(i, ()))))}")
        match_rows.append(f"{sid}\t{','.join(sorted(set(match_map.get(i, ()))))}")
    return cand_rows, match_rows

