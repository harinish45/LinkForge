"""Evaluation metrics including official F0.5 score, candidate recall, and precision."""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


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


def evaluate_blocking_recall(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Compute recall of the blocking / candidate generation stage."""
    total_true_pairs = 0
    covered_pairs = 0
    total_candidates = 0

    for s1, true_matches in ground_truth.items():
        total_true_pairs += len(true_matches)
        cands = candidates.get(s1, set())
        total_candidates += len(cands)
        covered_pairs += len(true_matches & cands)

    recall = float(covered_pairs) / total_true_pairs if total_true_pairs > 0 else 1.0
    avg_cands = float(total_candidates) / len(ground_truth) if ground_truth else 0.0

    return {
        "candidate_recall": recall,
        "covered_true_matches": covered_pairs,
        "total_true_matches": total_true_pairs,
        "average_candidates_per_entity": avg_cands,
    }
