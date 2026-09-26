"""Unit tests for metric computations (F0.5, Precision, Recall)."""

import pytest
from entitylink.evaluation.metrics import compute_f_beta, evaluate_matching_predictions


def test_compute_f_beta_precision_heavy():
    # If precision is high (1.0) and recall is 0.5:
    # F0.5 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333
    f05 = compute_f_beta(precision=1.0, recall=0.5, beta=0.5)
    assert abs(f05 - (0.625 / 0.75)) < 1e-4

    # If precision is 0.5 and recall is 1.0:
    # F0.5 = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555
    f05_low_p = compute_f_beta(precision=0.5, recall=1.0, beta=0.5)
    assert abs(f05_low_p - (0.625 / 1.125)) < 1e-4

    # F0.5 penalizes lower precision much more heavily than lower recall
    assert f05 > f05_low_p


def test_evaluate_matching_predictions():
    ground_truth = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": set(),  # singleton
        "S1-3": {"S2-30"},
    }
    predictions = {
        "S1-1": {"S2-10", "S3-20"},  # 2 TP
        "S1-2": set(),               # 1 correct singleton
        "S1-3": set(),               # 1 FN
    }
    report = evaluate_matching_predictions(ground_truth, predictions)
    assert report.true_positives == 2
    assert report.false_positives == 0
    assert report.false_negatives == 1
    assert report.precision == 1.0
    assert report.recall == 2.0 / 3.0
    assert report.singleton_accuracy == 1.0
