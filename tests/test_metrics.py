"""Unit tests for metric computations (F0.5, Precision, Recall)."""

import pytest
from entitylink.evaluation.metrics import (
    compute_f_beta,
    compute_entity_f05,
    evaluate_matching_predictions,
)


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


def test_compute_entity_f05():
    # Empty ground truth and empty prediction -> 1.0, 1.0, 1.0
    p, r, f = compute_entity_f05(set(), set())
    assert (p, r, f) == (1.0, 1.0, 1.0)

    # Empty ground truth and non-empty prediction -> 0.0, 0.0, 0.0
    p, r, f = compute_entity_f05(set(), {"S2-1"})
    assert (p, r, f) == (0.0, 0.0, 0.0)

    # Non-empty ground truth and empty prediction -> 0.0, 0.0, 0.0
    p, r, f = compute_entity_f05({"S2-1"}, set())
    assert (p, r, f) == (0.0, 0.0, 0.0)

    # Example from competition README:
    # GT = [S2-00047, S3-00812], Pred = [S2-00047, S2-00193, S3-00812]
    # Precision = 2/3, Recall = 1.0, F0.5 = 0.7142857
    p, r, f = compute_entity_f05({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"})
    assert abs(p - (2.0 / 3.0)) < 1e-4
    assert abs(r - 1.0) < 1e-4
    assert abs(f - 0.7142857) < 1e-4


def test_evaluate_matching_predictions_macro():
    ground_truth = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": set(),  # singleton
        "S1-3": {"S2-30"},
    }
    predictions = {
        "S1-1": {"S2-10", "S3-20"},  # Perfect match: P=1.0, R=1.0, F0.5=1.0
        "S1-2": set(),               # Correct singleton: P=1.0, R=1.0, F0.5=1.0
        "S1-3": set(),               # Missed match: P=0.0, R=0.0, F0.5=0.0
    }
    report = evaluate_matching_predictions(ground_truth, predictions)
    assert report.true_positives == 2
    assert report.false_positives == 0
    assert report.false_negatives == 1

    # Macro averages across 3 entities:
    # S1-1: P=1, R=1, F0.5=1
    # S1-2: P=1, R=1, F0.5=1
    # S1-3: P=0, R=0, F0.5=0
    # Mean P = 2/3, Mean R = 2/3, Mean F0.5 = 2/3
    assert abs(report.precision - (2.0 / 3.0)) < 1e-4
    assert abs(report.recall - (2.0 / 3.0)) < 1e-4
    assert abs(report.f05 - (2.0 / 3.0)) < 1e-4
    assert report.singleton_accuracy == 1.0

