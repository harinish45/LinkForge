import sys
from pathlib import Path
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

"""Per-entity match selection by expected F0.5 (replaces the global threshold).

The metric is F0.5 per Source 1 entity, so the right number of matches to keep
depends on the entity's own candidates: with probabilities p_1 >= p_2 >= ...
of the records assigned to it, keeping the top k gives expected F0.5 about
1.25 * sum(p_1..p_k) / (0.25 * sum(all p) + k), and keeping none scores
P(no true match) = prod(1 - p_i).  We keep the k with the highest value.

usage: python postprocess.py        (after predict.py; rewrites matching_results.tsv)
"""
import numpy as np
import pandas as pd

from block import load
from config import WORK_DIR, OUT_DIR
from predict import id_lists, write_tsv
from train import assign

FLOOR = 0.05      # records below this probability are never considered


def ef_select(best):
    """best: one row per S2/S3 record (its most probable S1) with columns s1_row, p.
    Returns the rows to keep as matches.

    ef's denominator (0.25 * sum of all candidate probabilities for this S1) and p_none
    (probability that none of them is a true match) must be computed over EVERY candidate
    of the entity, not just the ones above FLOOR -- otherwise an entity with many weak
    sub-floor candidates (a generic/high-collision "hub" record) has its recall mass and
    p_none silently computed as if those candidates did not exist, which inflates p_none
    and biases the decision toward wrongly predicting no match at all. FLOOR only decides
    which individual candidates are eligible to be *selected* as the top-k match.
    """
    totals = best.groupby('s1_row').p.sum()
    p_none = np.exp(np.log1p(-best.p.clip(upper=1 - 1e-6)).groupby(best.s1_row).sum())

    b = best[best.p >= FLOOR].sort_values(['s1_row', 'p'], ascending=[True, False]).copy()
    g = b.groupby('s1_row')
    b['k'] = g.cumcount() + 1
    total_p = b.s1_row.map(totals)
    b['ef'] = 1.25 * g.p.cumsum() / (0.25 * total_p + b.k)
    p_none_b = b.s1_row.map(p_none)
    best_ef = b.groupby('s1_row').ef.transform('max')
    k_best = b.s1_row.map(b[b.ef == best_ef].groupby('s1_row').k.min())
    return b[(b.k <= k_best) & (best_ef > p_none_b)]


def main():
    df = pd.read_parquet(WORK_DIR / 'test_prob.parquet')
    best = assign(df, df.p2.values, 0.0)
    matches = ef_select(best)
    s1_ids = load('test', 'source1', ['entity_id']).entity_id.values
    q_ids = {i: load('test', f'source{i}', ['entity_id']).entity_id.values for i in (2, 3)}
    write_tsv(OUT_DIR / 'matching_results.tsv', 'matched_entity_ids', s1_ids,
              id_lists(s1_ids, matches, q_ids))
    n_match = matches.groupby('s1_row').size().reindex(np.arange(len(s1_ids)), fill_value=0)
    print(f'expected-F selection: {len(matches)} matches, {(n_match == 0).mean():.3%} of S1 left empty')


if __name__ == '__main__':
    main()
