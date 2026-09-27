"""Evaluation metrics including official F0.5 score, candidate recall, distribution stats, and error analysis."""

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


def evaluate_matching_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> MatchEvaluationReport:
    """Compute micro-averaged pair-level precision, recall, and F0.5."""
    tp = 0
    fp = 0
    fn = 0

    gt_singletons = 0
    pred_singletons = 0
    correct_singletons = 0

    all_s1 = set(ground_truth.keys()) | set(predictions.keys())

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

        tp += len(pred_set & gt_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)

    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = compute_f_beta(precision, recall, beta=0.5)
    f1 = compute_f_beta(precision, recall, beta=1.0)
    singleton_acc = float(correct_singletons) / gt_singletons if gt_singletons > 0 else 1.0

    return MatchEvaluationReport(
        precision=precision,
        recall=recall,
        f05=f05,
        f1=f1,
        total_ground_truth_matches=tp + fn,
        total_predicted_matches=tp + fp,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
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
