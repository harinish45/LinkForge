# EntityLink AI — Technical Methodology

## 1. Problem Formulation & Metric

The objective is multi-source entity resolution where each reference entity $e \in S_1$ must be mapped to zero, one, or multiple records $\{c_1, \dots, c_k\} \subseteq S_2 \cup S_3$.

The competition optimizes **$F_{0.5}$ score**:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

Because $F_{0.5}$ weights Precision twice as heavily as Recall ($\beta = 0.5$), false positives (incorrect merges) are penalized severely. The system is designed with conservative decision boundaries and score margins.

---

## 2. Pipeline Stages

### 2.1 Schema & Ingestion
- Reads raw tab-separated records: `entity_id`, `business_name`, `business_address`, `country`.
- Validates entity prefix integrity (`S1-*`, `S2-*`, `S3-*`).
- Handles malformed, null, or empty fields safely.

### 2.2 Preprocessing & Normalization
- Canonicalizes legal entity suffixes (`Corporation` $\rightarrow$ `corp`, `Private Limited` $\rightarrow$ `pvt ltd`, `LLC` $\rightarrow$ `llc`).
- Standardizes street and unit abbreviations (`Street` $\rightarrow$ `st`, `Avenue` $\rightarrow$ `ave`, `Road` $\rightarrow$ `rd`, `Apartment` $\rightarrow$ `apt`).
- Normalizes country codes open-set (`USA` $\rightarrow$ `US`, `Bharat` $\rightarrow$ `India`).

### 2.3 Blocking (Candidate Generation)
To reduce the $O(|S_1| \times (|S_2| + |S_3|))$ search space:
- Partitions entities by country (never match cross-country).
- Builds inverted index over token stems, 4-gram prefixes, and address numeric tokens (PIN/postal codes, building numbers).
- Produces candidate sets capped per entity to maintain bounded runtime.

### 2.4 Feature Engineering
Each candidate pair $(e_1, e_2)$ is transformed into a dense vector:
- **Name Features:**
  - Raw and normalized exact match indicators
  - Token Jaccard similarity and containment ratio
  - Levenshtein edit distance ratio
  - Character 3-gram Jaccard similarity
  - Relative length ratio and shared token count
- **Address Features:**
  - Normalized token Jaccard similarity
  - Address edit distance and character 3-gram similarity
  - Numeric token overlap (postal codes, unit numbers)
- **Interaction & Cross-Field Features:**
  - Country compatibility indicator
  - Geometric mean of name and address similarities
  - Strong agreement / strong disagreement flags

### 2.5 Modeling & Score Calibration
1. **Rule-Based Baseline:** Heuristic weighted combination of name, address, and exact match signals.
2. **Supervised Tree Classifier:** `HistGradientBoostingClassifier` trained on positive and negative candidate pairs with class-weight balancing.
3. **Decision Calibrator:**
   - Evaluates top candidate against primary threshold $\tau$. If $\text{score} < \tau$, predicted as a singleton.
   - For multi-matches, admits secondary candidates only if within confidence margin $\Delta = 0.15$ of the top score.
