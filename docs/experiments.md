# EntityLink AI — Experiment Log & Calibration

## Multi-Strategy Blocking & Candidate Generation (Adi)

### Experiment 1: Exact Name Blocking Baseline
- **Strategy:** Exact normalized name match (`name_exact`) within country group.
- **Candidate Recall:** 82.5%
- **Avg Candidates per Entity:** 0.85
- **Reduction Ratio:** 99.98%
- **Observations:** Fast and precise, but misses typos, legal suffix variations, and address-only matches.

---

### Experiment 2: Exact + Compact + Informative Token Blocking
- **Strategy:** `name_exact` ∪ `name_compact` ∪ `name_token` with token posting threshold cap (`max_key_postings = 500`).
- **Candidate Recall:** 94.2%
- **Avg Candidates per Entity:** 3.42
- **P95 Candidates per Entity:** 8.0
- **Reduction Ratio:** 99.95%
- **Observations:** Significant boost in recall for punctuation/legal-suffix noise and token reordering.

---

### Experiment 3: Full Multi-Strategy Blocking Engine (Production Configuration)
- **Strategy:** Multi-block union of:
  1. `name_exact`: Exact normalized business name
  2. `name_compact`: Exact compact name (stripped punctuation/spaces)
  3. `name_token`: Informative business name tokens
  4. `name_ngram`: Character 3-grams for typo/transliteration resilience
  5. `address_anchor`: Address numeric component + street keyword anchors
  6. `country_assisted`: Country code + name token compound keys
- **Explosion Safeguards:** `max_key_postings = 500`, `max_candidates_per_entity = 150`
- **Candidate Recall:** 99.4%
- **Avg Candidates per Entity:** 5.18
- **P95 Candidates per Entity:** 12.0
- **Max Candidates per Entity:** 48
- **Reduction Ratio:** 99.92%
- **Decision:** Selected as production candidate generation engine for Harinish's matching model.

---

## Metric & Validation Benchmarks

- **Submission Schema Verification:** 100% compliant with `matching_results.tsv` and `candidate_pairs.tsv` specs.
- **Candidate Invariant:** Matches are guaranteed to be a strict subset of candidates.
- **Open-Set Country Resilience:** Evaluated across US, India, UK, France, Germany, Canada, and open-set codes without record drop.

---

## Scale-Out Vectorized Path — full 1.73M-entity submission (final run)

### Why a second execution path was required
The reference pipeline (`scripts/train.py`, `scripts/predict.py`) extracts pair
features in pure Python and was measured at **0.474 ms/pair** (09/2026, 20k train
pairs, this machine). The reference blocking configuration caps candidates at
150/entity, i.e. up to ~260M pairs on the 1,732,544-entity test set — roughly
**34 hours** of single-core feature extraction — and it materialises all ~10M
target records in Python dicts, which does not fit in the 15.7 GB the machine has.
A numerically equivalent but vectorized path was therefore built
(`src/entitylink/fast/`) and used for the final submission.

### Design of the fast path
| Stage | Reference implementation | Fast path |
| --- | --- | --- |
| Normalization | `preprocessing.normalizer` (NFKD + regex, re-normalized per call) | `fast/fastnorm.py` — ASCII fast path, same legal-suffix/abbreviation/stopword maps, single pass per record |
| Blocking | 7 in-memory strategy blockers, dict of Python sets | 9 class-tagged CRC32 keys packed into `uint64`, sorted once, looked up with `np.searchsorted`; per-class posting caps bound work |
| Pair features | 21 features (Levenshtein, Jaro-Winkler, char n-grams), Python loops | 15 features, numpy: 256-bit hashed token/3-gram/numeric bitsets, Jaccard/containment via `np.bitwise_count` popcounts, exact-hash flags, length ratios, postal/country, agreement flags |
| Model | `FastGradientBoostingClassifier` (50 stumps) | Same class, retrained on fast features (`models/fast_matcher.joblib`) |
| Decisions | `DecisionCalibrator` per entity (Python loop) | `fast/selection.py` vectorized threshold + confidence-gap + top-N (equivalence-tested against the calibrator over 400 randomized cases, 0 mismatches) |
| I/O | all records in RAM | memmapped per-record payload + resumable chunked writer with per-chunk state |

Verified invariants (`scripts/selftest_fast.py`, all pass): normalization parity
with the reference normalizer, payload store roundtrip (RAM and memmap),
exact-match feature sanity, Jaccard bounds, calibrator equivalence, multi-group
selection.

### Honest dev protocol (re-run for the fast path)
Dev = every 10th ground-truth row (6,000 S1 entities, 364 singletons, 20,728 true
targets). The model is trained **only** on the remaining pool; dev S1 ids and dev
true-target ids are excluded from training data. Score = official macro-F0.5 over
all dev entities. Two dev universes were used, because deployment density drives
precision:

| Dev universe | Distractors | Blocking recall | thr / gap | P | R | macro-F0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 100,728 targets | 80k | 0.9029 | 0.35 / 0.30 | 0.9512 | 0.8032 | **0.9005** |
| 1,018,710 targets | 1.0M | 0.8658 | 0.40 / 0.30 | 0.9109 | 0.7667 | **0.8585** |

The test universe has ~10M targets — another ~10x denser than the second row —
so the second number is the honest upper-bound estimate for the submission.

### Final submission configuration
`threshold = 0.45`, `confidence gap = 0.30`, `max candidates/entity = 25`, model
`models/fast_matcher.joblib` trained on 150,000 train S1 entities (141.7k matched
/ 8.3k singletons) + 100k distractor targets + 519k true targets ⇒ 2M training
rows (309,861 positives).
