"""Throughput benchmark to size the full-test submission run."""
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
from entitylink.data.loader import stream_tsv_records
from entitylink.features.pair_features import PairFeatureExtractor, FEATURE_NAMES
from entitylink.preprocessing.normalizer import build_normalized_record
from entitylink.blocking.candidate_generator import extract_blocking_keys
from entitylink.models.classifier import MLPairMatcher

N = 20000
t0 = time.time()
recs = list(itertools.islice(stream_tsv_records("data/train/train_source2.tsv"), N))
dt = time.time() - t0
print(f"parse {N}: {dt:.2f}s -> {N/dt:.0f} lines/s -> 10M rows = {10e6/(N/dt)/60:.1f} min")

t0 = time.time()
for r in recs:
    build_normalized_record(r.entity_id, r.business_name, r.business_address, r.country)
dt = time.time() - t0
print(f"normalize {N}: {dt:.2f}s -> {N/dt:.0f} rec/s -> 10M = {10e6/(N/dt)/60:.1f} min")

t0 = time.time()
for r in recs:
    extract_blocking_keys(r)
dt = time.time() - t0
print(f"blocking keys {N}: {dt:.2f}s -> {N/dt:.0f} rec/s -> 10M = {10e6/(N/dt)/60:.1f} min")

ex = PairFeatureExtractor(FEATURE_NAMES)
pairs = [(recs[i], recs[i + 1]) for i in range(2000)]
t0 = time.time()
for a, b in pairs:
    ex.extract_vector(a, b)
dt = time.time() - t0
per = dt / len(pairs) * 1000
print(f"features {len(pairs)}: {dt:.2f}s -> {per:.3f} ms/pair -> 20M pairs = {20e6*per/1000/3600:.1f} h")

m = MLPairMatcher.load("models/pair_matcher.joblib")
print("model type:", m.model_type, "fitted:", m.is_fitted, "n_feats:", len(FEATURE_NAMES))
X = np.array([ex.extract_vector(a, b) for a, b in pairs])
t0 = time.time()
for _ in range(20):
    p = m.model.predict_proba(X)
dt = time.time() - t0
print(f"predict {len(X)*20}: {dt:.2f}s -> {len(X)*20/dt:.0f} pairs/s")
print("pred range:", float(p[:, 1].min()), float(p[:, 1].max()))
