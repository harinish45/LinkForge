"""Unit tests for ML Malicious URL Classifier."""

import os
import tempfile
import pytest
from src.models.classifier import MaliciousUrlClassifier, ThreatAssessment, ThreatLevel


@pytest.fixture
def classifier() -> MaliciousUrlClassifier:
    """Fixture providing initialized MaliciousUrlClassifier instance."""
    return MaliciousUrlClassifier()


def test_benign_url_scoring(classifier: MaliciousUrlClassifier) -> None:
    """Test benign legitimate URLs score low threat level."""
    benign_url = "https://www.google.com/search?q=open+source"
    assessment: ThreatAssessment = classifier.predict(benign_url)

    assert assessment.threat_score < 0.50
    assert assessment.is_malicious is False
    assert assessment.risk_level in {ThreatLevel.LOW, ThreatLevel.MEDIUM}


def test_malicious_ip_with_login(classifier: MaliciousUrlClassifier) -> None:
    """Test blatant phishing IP with login keywords scores high or critical."""
    phishing_url = "http://192.168.1.1/paypal/login/verify-account.php"
    assessment: ThreatAssessment = classifier.predict(phishing_url)

    assert assessment.threat_score >= 0.50
    assert assessment.is_malicious is True
    assert assessment.risk_level in {ThreatLevel.HIGH, ThreatLevel.CRITICAL}
    assert len(assessment.detected_threats) > 0


def test_at_symbol_credential_trick(classifier: MaliciousUrlClassifier) -> None:
    """Test detection of '@' credential stuffing trick in URL."""
    trick_url = "http://google.com@evil-attacker-site.com/auth"
    assessment: ThreatAssessment = classifier.predict(trick_url)

    assert assessment.threat_score >= 0.50
    assert any("@" in t for t in assessment.detected_threats)


def test_model_serialization(classifier: MaliciousUrlClassifier) -> None:
    """Test saving and loading trained pipeline from disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.joblib")
        classifier.save(model_path)
        assert os.path.exists(model_path)

        loaded_classifier = MaliciousUrlClassifier(model_path=model_path)
        assert loaded_classifier.pipeline is not None

        test_url = "https://github.com"
        res1 = classifier.predict(test_url)
        res2 = loaded_classifier.predict(test_url)
        assert res1.threat_score == res2.threat_score
