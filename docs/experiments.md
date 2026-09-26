# EntityLink AI — Experiment Log & Calibration

## Experiment 1: Baseline Architecture & Metric Verification
- **Model:** Deterministic Heuristic Matcher (`RuleBasedMatcher`)
- **Features:** Exact match + Token Jaccard + Levenshtein + Number Overlap
- **Threshold:** Primary $\tau = 0.62$, Gap $\Delta = 0.15$
- **Candidate Strategy:** Multi-key inverted index (country + name prefix + address numeric tokens)

### Results on Validation Split (Sample 5,000 S1 records)
- **Singleton Rate in Ground Truth:** ~5.46%
- **Multi-match Rate in Ground Truth:** ~89.44%
- **Evaluation Metric:** $F_{0.5}$ (Precision-heavy)

## Key Observations
1. **Multi-Match Multiplicity:** High proportion of multi-matches requires multi-candidate admission logic rather than 1-to-1 matching.
2. **Open-Set Countries:** France entities in test must not be rejected by hard-coded filters.
3. **Subset Consistency:** Every predicted match must be in `candidate_pairs.tsv` to ensure zero validation warnings.
