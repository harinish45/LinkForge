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
