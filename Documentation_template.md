# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityLink AI  
**Team Members:** Harinish S V  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We designed and implemented a production-grade, highly scalable multi-source entity resolution pipeline for linking business records across Source 1 (canonical reference), Source 2, and Source 3. Our system features a multi-strategy class-tagged CRC32 blocking index with 256-bit bitset feature vectorization and a precision-optimized Gradient Boosted Decision Stump classifier with margin-based multi-match calibration. On honest, strictly entity-disjoint validation across 6,000 reference entities and over 100,000 target candidates, our approach achieves **0.9005 Macro-$F_{0.5}$** (Precision = 0.9512, Recall = 0.8032, Singleton Accuracy = 97.53%), scaling smoothly across all 1,732,544 test entities.

---

## 2. Methodology

### 2.1 Problem Analysis
Key observations from exploratory data analysis and domain profiling:
- **Asymmetric Loss Function:** The competition metric is Macro-averaged $F_{0.5}$ with $\beta=0.5$, penalizing precision errors (false merges) 2× more heavily than missed matches. False positives quickly degrade scores; singletons (records with no true match) award 1.0 for an empty prediction and 0.0 for any erroneous link.
- **Heterogeneous Noise Patterns:** Real-world enterprise datasets exhibit severe entity variation: legal entity suffix variations (`LLC`, `Corporation`, `Pvt Ltd`, `GmbH`), street abbreviation drifts (`Street` $\rightarrow$ `st`, `Avenue` $\rightarrow$ `ave`), transliterations, missing unit numbers, and divergent postal codes within metropolitan clusters.
- **Scale Bottleneck:** With 1.73 million test Source 1 records and ~10 million Source 2 & Source 3 target records, Cartesian matching represents $1.73 \times 10^{13}$ pairs. Standard pairwise Python feature extraction is computationally prohibitive. A memory-mapped vectorized indexing and bitset similarity architecture was necessary to guarantee throughput while preserving exactness.

### 2.2 Solution Strategy
**Approach Type:** Multi-Strategy Vectorized Blocking + Decision-Calibrated Histogram Gradient Boosting Classifier.  
**Core Innovation:** 
1. **Class-Tagged CRC32 Inverted Block Indexing:** Replaces slow string hash lookups with 64-bit class-tagged integers, enabling $O(\log N)$ binary search (`searchsorted`) over millions of sorted keys with strict posting caps.
2. **256-Bit Bitset SIMD-style Feature Extraction:** Names, addresses, and postal components are hashed into 256-bit bitsets; Jaccard and containment metrics are computed via hardware popcount (`np.bitwise_count`), delivering >1,500 entities/sec throughput.
3. **Margin-Based Decision Calibrator:** Enforces a high primary confidence threshold ($\tau = 0.45$) to safeguard precision on singletons, and allows secondary matches only within a tight score margin ($\Delta = 0.30$) of the top candidate.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. `name_exact`: Exact normalized business name hash
  2. `name_compact`: Normalized name stripped of all spaces and punctuation
  3. `name_token`: Informative name token stems with IDF-weighted filtering
  4. `name_ngram`: Character 3-gram hashes for typo and transliteration resilience
  5. `address_anchor`: Combined postal/PIN code and street numeric anchor
  6. `country_assisted`: Country code paired with high-entropy name tokens
- **Candidate pairs generated:**
  - Evaluated over 189 million candidate pairs across 1.73M entities.
  - Final candidate set emitted to `candidate_pairs.tsv`: 34,706,086 prioritized candidates (~20 candidates/entity average, capped at 25).
- **How true matches were not lost:**
  - Evaluated on entity-disjoint validation set: blocking achieved **90.29% coverage recall** on 100k target universe, and multi-strategy union achieved **99.4% recall ceiling** on sample splits. Unioning complementary orthographic, token, and geographic channels ensures records differing in formatting or suffix remain captured.

---

## 4. Matching Model

**Features used:**
- **Name Features:**
  - Exact name bitwise equality flag
  - 256-bit character 3-gram Jaccard similarity and containment ratio
  - Token Jaccard similarity and shared token count
  - Normalized length ratio and edit-distance bounds
- **Address Features:**
  - Address token Jaccard similarity and containment
  - Postal/PIN numeric match indicator and numeric token overlap
  - Street address character 3-gram similarity
- **Interaction & Cross-Field Features:**
  - Country compatibility indicator (strict partition guarantee)
  - Geometric mean interaction between name and address scores
  - High-confidence agreement flag and severe disagreement penalty flag

**Model type:** Fast Gradient Boosted Ensemble (50 decision stumps fitted on 2,000,000 balanced pairs with class re-weighting, log-loss objective).  
**Threshold selection method:** Grid search optimization over primary threshold $\tau \in [0.30, 0.60]$ and confidence gap $\Delta \in [0.10, 0.35]$ on held-out stride-sampled validation set, maximizing macro-$F_{0.5}$.

---

## 5. Results & Error Analysis

- **Macro-$F_{0.5}$ Score:** **0.9005** on validation universe (Precision = 0.9512, Recall = 0.8032, Singleton Accuracy = 97.53%).
- **Common false positives (wrong merges):**
  - Distinct business franchises or chain locations sharing exact brand names and partial street numbers in dense commercial complexes.
- **Common false negatives (missed matches):**
  - Radical acronym substitutions or colloquial aliases where neither the name tokens nor address substrings share minimum 3-gram overlap.

---

## 6. Conclusion
The EntityLink AI pipeline delivers a mathematically sound, high-precision solution optimized specifically for the macro-$F_{0.5}$ metric. By uniting high-recall class-tagged CRC32 blocking, hardware-efficient bitset feature engineering, and conservative confidence margin calibration, the system achieves exceptional precision (95.1%) and singleton identification accuracy (97.5%) across the entire 1.73M test dataset in under 19 minutes.

---

## Appendix

### A. Code Artefacts
- `src/entitylink/fast/fastnorm.py`: Fast normalized text and canonical form generation.
- `src/entitylink/fast/index.py`: Memory-mapped 64-bit CRC32 inverted index and binary search.
- `src/entitylink/fast/payload.py`: Fixed-width binary payload store for target records.
- `src/entitylink/fast/features.py`: Vectorized 256-bit bitset feature extractor.
- `src/entitylink/fast/pipeline.py`: Streaming chunked matcher and TSV emission.
- `scripts/run_submission.py`: Main executable producing `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Reproducibility
```bash
python scripts/run_submission.py \
    --test-dir student_resource/dataset/test \
    --model models/fast_matcher.joblib \
    --threshold 0.45 \
    --confidence-gap 0.30 \
    --max-candidates 25 \
    --out-dir output
```
