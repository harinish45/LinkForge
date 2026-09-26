# EntityLink AI 🔗

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)]()

> High-Performance, Scalable Business Entity Resolution System for Amazon ML Challenge 2026

EntityLink AI links and deduplicates business entity records across heterogeneous, noisy data streams without external identity lookups, geocoding, or third-party business registries.

---

## 🏗 Architecture & Core Pipeline

```text
Raw TSV (S1, S2, S3)
   ↓
Schema & Prefix Validation
   ↓
Text & Address Normalization
   ↓
Multi-Strategy Inverted Index Blocking
   ↓
Candidate Generation & Deduplication
   ↓
Pair Feature Extraction (Name, Address, Interactions)
   ↓
Matching Model (Supervised Gradient Boosting / Baseline)
   ↓
Score Calibration & Multi-Match Margin Decision
   ↓
Strict Subset Filtering
   ↓
Submission TSV Writer (matching_results.tsv & candidate_pairs.tsv)
   ↓
Official Challenge Validation
```

Detailed architectural contracts are documented in [docs/architecture.md](file:///c:/Users/Harinish%20S%20V/Downloads/amazon%20ml/docs/architecture.md) and [docs/methodology.md](file:///c:/Users/Harinish%20S%20V/Downloads/amazon%20ml/docs/methodology.md).

---

## 📁 Repository Layout

```text
entitylink-ai/
├── README.md                           # Main documentation & reproduction guide
├── LICENSE                             # MIT License
├── pyproject.toml                      # Standard packaging specification
├── requirements.txt                    # Project dependencies
├── configs/
│   ├── config.yaml                     # Pipeline parameters & feature toggles
│   └── thresholds.yaml                 # Decision thresholds & confidence margins
├── src/
│   └── entitylink/
│       ├── data/                       # Streaming TSV loaders & schema validation
│       ├── preprocessing/              # Suffix & abbreviation normalizers
│       ├── blocking/                   # Inverted-index candidate generation
│       ├── features/                   # String distances, token overlaps & pair vectors
│       ├── models/                     # BaseMatcher, RuleBasedMatcher & MLPairMatcher
│       ├── evaluation/                 # Official F0.5 metric & error diagnostics
│       └── pipeline/                   # Pipeline runner & submission TSV writer
├── scripts/
│   ├── data_profile.py                 # Exploratory data profiling & integrity audit
│   ├── train.py                        # Supervised ML model training
│   ├── predict.py                      # Standalone test inference CLI
│   ├── evaluate.py                     # Official F0.5 score evaluation on validation
│   ├── validate_submission.py          # Format and constraint validation
│   └── run_pipeline.py                 # Single-command end-to-end execution
├── tests/                              # Unit tests (Schema, Features, Metrics, Writer)
└── docs/
    ├── architecture.md
    ├── methodology.md
    └── experiments.md
```

---

## ⚡ Quick Start

### 1. Installation

```bash
git clone https://github.com/harinish45/LinkForge.git
cd LinkForge

# Install package in editable mode
python -m pip install -e .
```

### 2. Dataset Setup

Place competition data in `data/`:
```text
data/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### 3. Profile Dataset

```bash
python scripts/data_profile.py --sample-size 5000
```

### 4. Train Supervised Matcher (Optional)

```bash
python scripts/train.py --train-dir data/train --sample-size 10000 --model-type gradient_boosting
```

### 5. Run End-to-End Pipeline

Execute complete candidate generation, scoring, and output file writing:

```bash
python scripts/run_pipeline.py \
  --test-dir data/test \
  --output-dir output \
  --threshold 0.62
```

### 6. Validate Submission

```bash
python scripts/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir data/test
```

### 7. Run Unit Tests

```bash
python -m pytest tests
```

---

## 🎯 Official Submission Guarantees

- **Exact Schema:** `matching_results.tsv` (`source1_entity_id`, `matched_entity_ids`) and `candidate_pairs.tsv` (`source1_entity_id`, `candidate_entity_ids`).
- **Complete Coverage:** Exactly one row per Source 1 test entity.
- **Candidate Subset Invariant:** Every final matched ID is strictly present in the entity's candidate list.
- **No Self-Matches:** Only `S2-*` and `S3-*` IDs appear in target lists.
- **No Intra-List Duplicates:** Duplicate IDs within comma-separated lists are strictly prevented.
- **Singletons Handled:** Correct empty field for no-match entities.
- **Zero External Lookups:** 100% compliant with offline competition guidelines.
