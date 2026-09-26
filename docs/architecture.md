# EntityLink AI — System Architecture

## Overview

EntityLink AI is a modular, scalable machine learning system for cross-source business entity resolution, designed for the Amazon ML Challenge 2026. The system links heterogeneous, noisy business entities across three disjoint sources:
- **Source 1 ($S_1$):** Deduplicated reference entities.
- **Source 2 ($S_2$):** Noisy secondary records.
- **Source 3 ($S_3$):** Noisy tertiary records.

```mermaid
flowchart TD
    subgraph Input Data
        S1["Source 1 TSV (Reference)"]
        S2["Source 2 TSV (Noisy)"]
        S3["Source 3 TSV (Noisy)"]
    end

    subgraph Data Layer
        Schema["Schema Validation & Streaming Loader"]
        Norm["Text & Entity Normalization"]
    end

    subgraph Blocking Layer
        Keys["Multi-Key Generation"]
        Index["Inverted Multi-Index"]
        Cands["Candidate Deduplication"]
    end

    subgraph Feature Layer
        NameFeats["Name Similarities (Jaccard, Levenshtein, 3-grams)"]
        AddrFeats["Address Similarities & Numeric Token Overlap"]
        CrossFeats["Country Compatibility & Interaction Signals"]
    end

    subgraph Decision Layer
        Matcher["Matching Model (HistGradientBoosting / Rule-Based)"]
        Calibrator["Decision Calibrator & Gap Filter"]
    end

    subgraph Output Validation
        Writer["Deterministic TSV Writer"]
        Validator["Submission Validator"]
    end

    S1 & S2 & S3 --> Schema --> Norm
    Norm --> Keys --> Index --> Cands
    Cands --> NameFeats & AddrFeats & CrossFeats
    NameFeats & AddrFeats & CrossFeats --> Matcher --> Calibrator
    Calibrator --> Writer --> Validator
```

---

## Architectural Principles

1. **Clear Modular Boundaries:** Preprocessing, candidate generation, feature extraction, scoring, and output generation are decoupled. Any module can be updated independently without breaking downstream consumers.
2. **Open-Set Country Support:** Countries are treated dynamically as an open set of string labels rather than hard-coded categories, ensuring robust generalization to test countries (such as France) that do not appear in the training split.
3. **Strict Candidate Subset Invariant:** Every final match in `matching_results.tsv` is mathematically guaranteed to be a subset of `candidate_pairs.tsv`.
4. **Memory Efficiency & Streaming:** All source readers support streaming iteration to handle multi-gigabyte datasets without out-of-memory errors on commodity hardware.
5. **Deterministic Calibration:** Decision thresholds and confidence gaps are tuned specifically for the precision-heavy official metric ($F_{0.5}$).
