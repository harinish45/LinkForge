"""Evaluation metrics including official per-S1 macro F0.5 score, candidate recall, and precision."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class MatchEvaluationReport:
    """Detailed evaluation report on entity resolution matches."""
    precision: float
    recall: float
    f05: float
    f1: float
    total_ground_truth_matches: int
    total_predicted_matches: int
    true_positives: int
    false_positives: int
    false_negatives: int
    singleton_ground_truth_count: int
    singleton_predicted_count: int
    singleton_accuracy: float


@dataclass
class CandidateEvaluationReport:
    """Detailed evaluation report on candidate generation / blocking performance."""
    candidate_recall: float
    covered_true_matches: int
    total_true_matches: int
    total_candidates: int
    avg_candidates_per_entity: float
    median_candidates_per_entity: float
    p95_candidates_per_entity: float
    max_candidates_per_entity: int
    reduction_ratio: float


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F-beta score.
    
    For beta=0.5:
        F0.5 = (1 + 0.25) * P * R / (0.25 * P + R)
    """
    if precision + recall == 0.0:
        return 0.0
    beta_sq = beta ** 2
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    return numerator / denominator if denominator > 0.0 else 0.0


def compute_entity_f05(gt_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    """Compute precision, recall, and F0.5 for a single Source 1 entity.
    
    Official rules:
    - If both ground truth and predictions are empty (singleton correctly identified):
      Precision = 1.0, Recall = 1.0, F0.5 = 1.0
    - If ground truth is empty but prediction is non-empty (false merge on singleton):
      Precision = 0.0, Recall = 0.0, F0.5 = 0.0
    - If ground truth is non-empty but prediction is empty (missed matches):
      Precision = 0.0, Recall = 0.0, F0.5 = 0.0
    - If both non-empty:
      Precision = |TP| / |Pred|, Recall = |TP| / |GT|, F0.5 computed with beta=0.5
    """
    if not gt_ids and not pred_ids:
        return 1.0, 1.0, 1.0

    if not gt_ids and pred_ids:
        return 0.0, 0.0, 0.0

    if gt_ids and not pred_ids:
        return 0.0, 0.0, 0.0

    tp = len(gt_ids & pred_ids)
    p = tp / len(pred_ids) if pred_ids else 0.0
    r = tp / len(gt_ids) if gt_ids else 0.0
    f05 = compute_f_beta(p, r, beta=0.5)
    return p, r, f05


def evaluate_matching_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> MatchEvaluationReport:
    """Compute official macro-averaged per-S1 precision, recall, and F0.5 score."""
    tp_total = 0
    fp_total = 0
    fn_total = 0

    gt_singletons = 0
    pred_singletons = 0
    correct_singletons = 0

    all_s1 = set(ground_truth.keys()) | set(predictions.keys())
    if not all_s1:
        return MatchEvaluationReport(
            precision=1.0, recall=1.0, f05=1.0, f1=1.0,
            total_ground_truth_matches=0, total_predicted_matches=0,
            true_positives=0, false_positives=0, false_negatives=0,
            singleton_ground_truth_count=0, singleton_predicted_count=0,
            singleton_accuracy=1.0
        )

    sum_f05 = 0.0
    sum_prec = 0.0
    sum_rec = 0.0

    for s1 in all_s1:
        gt_set = ground_truth.get(s1, set())
        pred_set = predictions.get(s1, set())

        is_gt_sing = len(gt_set) == 0
        is_pred_sing = len(pred_set) == 0

        if is_gt_sing:
            gt_singletons += 1
        if is_pred_sing:
            pred_singletons += 1
        if is_gt_sing and is_pred_sing:
            correct_singletons += 1

        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        tp_total += tp
        fp_total += fp
        fn_total += fn

        p_ent, r_ent, f05_ent = compute_entity_f05(gt_set, pred_set)
        sum_prec += p_ent
        sum_rec += r_ent
        sum_f05 += f05_ent

    n_entities = len(all_s1)
    macro_precision = sum_prec / n_entities
    macro_recall = sum_rec / n_entities
    macro_f05 = sum_f05 / n_entities
    macro_f1 = compute_f_beta(macro_precision, macro_recall, beta=1.0)
    singleton_acc = float(correct_singletons) / gt_singletons if gt_singletons > 0 else 1.0

    return MatchEvaluationReport(
        precision=macro_precision,
        recall=macro_recall,
        f05=macro_f05,
        f1=macro_f1,
        total_ground_truth_matches=tp_total + fn_total,
        total_predicted_matches=tp_total + fp_total,
        true_positives=tp_total,
        false_positives=fp_total,
        false_negatives=fn_total,
        singleton_ground_truth_count=gt_singletons,
        singleton_predicted_count=pred_singletons,
        singleton_accuracy=singleton_acc,
    )


def evaluate_blocking_performance(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    total_target_records: Optional[int] = None
) -> CandidateEvaluationReport:
    """Compute comprehensive candidate recall, distribution stats, and reduction ratio."""
    total_true_pairs = 0
    covered_pairs = 0
    cand_counts: List[int] = []

    for s1, true_matches in ground_truth.items():
        total_true_pairs += len(true_matches)
        cands = candidates.get(s1, set())
        cand_counts.append(len(cands))
        covered_pairs += len(true_matches & cands)

    num_s1 = len(ground_truth)
    total_candidates = sum(cand_counts)
    recall = float(covered_pairs) / total_true_pairs if total_true_pairs > 0 else 1.0
    avg_cands = float(total_candidates) / num_s1 if num_s1 > 0 else 0.0

    if cand_counts:
        sorted_counts = sorted(cand_counts)
        median_cands = float(sorted_counts[len(sorted_counts) // 2])
        p95_idx = int(math.ceil(0.95 * len(sorted_counts))) - 1
        p95_idx = max(0, min(p95_idx, len(sorted_counts) - 1))
        p95_cands = float(sorted_counts[p95_idx])
        max_cands = sorted_counts[-1]
    else:
        median_cands = 0.0
        p95_cands = 0.0
        max_cands = 0

    if total_target_records and total_target_records > 0 and num_s1 > 0:
        total_search_space = num_s1 * total_target_records
        reduction_ratio = 1.0 - (float(total_candidates) / total_search_space)
    else:
        reduction_ratio = 0.0

    return CandidateEvaluationReport(
        candidate_recall=recall,
        covered_true_matches=covered_pairs,
        total_true_matches=total_true_pairs,
        total_candidates=total_candidates,
        avg_candidates_per_entity=avg_cands,
        median_candidates_per_entity=median_cands,
        p95_candidates_per_entity=p95_cands,
        max_candidates_per_entity=max_cands,
        reduction_ratio=reduction_ratio,
    )


def evaluate_blocking_recall(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Backward compatible wrapper for blocking recall calculation."""
    report = evaluate_blocking_performance(ground_truth, candidates)
    return {
        "candidate_recall": report.candidate_recall,
        "covered_true_matches": report.covered_true_matches,
        "total_true_matches": report.total_true_matches,
        "average_candidates_per_entity": report.avg_candidates_per_entity,
    }

def analyze_blocking_errors(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
) -> List[Tuple[str, str]]:
    """Return list of (source1_id, missed_target_id) false negative candidate pairs."""
    missed_pairs: List[Tuple[str, str]] = []
    for s1_id, true_matches in ground_truth.items():
        cands = candidates.get(s1_id, set())
        for missed in (true_matches - cands):
            missed_pairs.append((s1_id, missed))
    return missed_pairs
