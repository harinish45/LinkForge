"""Unit tests for URL feature extraction module."""

import pytest
from src.processors.features import UrlFeatureExtractor, UrlFeatures


@pytest.fixture
def extractor() -> UrlFeatureExtractor:
    """Fixture providing a UrlFeatureExtractor instance."""
    return UrlFeatureExtractor()


def test_calculate_entropy(extractor: UrlFeatureExtractor) -> None:
    """Test Shannon entropy calculation on uniform vs varied strings."""
    assert extractor.calculate_entropy("") == 0.0
    assert extractor.calculate_entropy("aaaa") == 0.0
    varied_entropy = extractor.calculate_entropy("abcdefgh")
    assert varied_entropy > 2.5


def test_vowel_consonant_ratio(extractor: UrlFeatureExtractor) -> None:
    """Test vowel-to-consonant ratio computation."""
    assert extractor.calculate_vowel_consonant_ratio("aeiou") == 5.0
    assert extractor.calculate_vowel_consonant_ratio("bcdfgh") == 0.0
    assert extractor.calculate_vowel_consonant_ratio("apple") == 0.6667


def test_feature_extraction_standard(extractor: UrlFeatureExtractor) -> None:
    """Test feature extraction on a clean HTTPS URL."""
    url = "https://www.google.com/search?q=security"
    features: UrlFeatures = extractor.extract(url)

    assert features.has_https is True
    assert features.is_ip is False
    assert features.is_shortened is False
    assert features.url_length > 20
    assert features.domain_depth >= 1
    assert isinstance(features.to_vector(), list)
    assert len(features.to_vector()) == 24


def test_feature_extraction_suspicious_phishing(extractor: UrlFeatureExtractor) -> None:
    """Test extraction of suspicious keywords, IP format, and suspicious TLD."""
    url = "http://192.168.0.1/paypal/login/verify-account.xyz"
    features: UrlFeatures = extractor.extract(url)

    assert features.is_ip is True
    assert features.has_https is False
    assert features.suspicious_token_count >= 2
    assert "login" in features.suspicious_tokens
    assert "paypal" in features.suspicious_tokens


def test_feature_extraction_shortened_url(extractor: UrlFeatureExtractor) -> None:
    """Test detection of shortening service domain."""
    url = "https://bit.ly/3xYz123"
    features: UrlFeatures = extractor.extract(url)

    assert features.is_shortened is True


def test_feature_vector_conversion(extractor: UrlFeatureExtractor) -> None:
    """Test vector serialization format for ML training."""
    url = "https://github.com/harinish45/LinkForge"
    features = extractor.extract(url)
    vector = features.to_vector()

    assert all(isinstance(v, float) for v in vector)
    assert len(vector) == 24
