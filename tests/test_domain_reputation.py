"""Unit tests for domain reputation and category classification."""

import pytest
from src.models.reputation import (
    DomainCategory,
    DomainReputationClassifier,
    DomainReputationResult,
    ReputationTier,
)


@pytest.fixture
def rep_classifier() -> DomainReputationClassifier:
    """Fixture providing DomainReputationClassifier instance."""
    return DomainReputationClassifier()


def test_known_tech_domain(rep_classifier: DomainReputationClassifier) -> None:
    """Test recognized Tech domain like github.com."""
    result: DomainReputationResult = rep_classifier.evaluate("https://github.com/torvalds")
    assert result.category == DomainCategory.TECH
    assert result.reputation_score >= 90
    assert result.tier == ReputationTier.TRUSTED


def test_known_social_domain(rep_classifier: DomainReputationClassifier) -> None:
    """Test recognized Social domain like twitter.com."""
    result: DomainReputationResult = rep_classifier.evaluate("https://twitter.com/elonmusk")
    assert result.category == DomainCategory.SOCIAL
    assert result.reputation_score >= 90
    assert result.tier == ReputationTier.TRUSTED


def test_known_commerce_domain(rep_classifier: DomainReputationClassifier) -> None:
    """Test recognized Commerce domain like amazon.com."""
    result: DomainReputationResult = rep_classifier.evaluate("https://www.amazon.com/dp/12345")
    assert result.category == DomainCategory.COMMERCE
    assert result.reputation_score >= 90


def test_government_and_education_tld(rep_classifier: DomainReputationClassifier) -> None:
    """Test classification of .gov and .edu domains."""
    gov_result = rep_classifier.evaluate("https://nasa.gov/mission")
    assert gov_result.category == DomainCategory.GOVERNMENT_EDUCATION

    edu_result = rep_classifier.evaluate("https://stanford.edu/courses")
    assert edu_result.category == DomainCategory.GOVERNMENT_EDUCATION


def test_ip_address_reputation_penalty(rep_classifier: DomainReputationClassifier) -> None:
    """Test direct IP penalty applied to reputation score."""
    result: DomainReputationResult = rep_classifier.evaluate("http://192.168.1.1/admin")
    assert result.reputation_score < 50
    assert result.tier in {ReputationTier.SUSPICIOUS, ReputationTier.MALICIOUS}
    assert any("Direct IP" in f for f in result.factors)


def test_suspicious_tld_penalty(rep_classifier: DomainReputationClassifier) -> None:
    """Test high-abuse TLD like .xyz receives penalty."""
    result: DomainReputationResult = rep_classifier.evaluate("http://random-portal-123.xyz/login")
    assert result.reputation_score < 60
    assert any("High-abuse TLD" in f for f in result.factors)
