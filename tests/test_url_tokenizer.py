"""Unit tests for URL Tokenizer module."""

import pytest
from src.processors.tokenizer import TokenizedUrl, UrlTokenizer


@pytest.fixture
def tokenizer() -> UrlTokenizer:
    """Fixture providing a UrlTokenizer instance."""
    return UrlTokenizer()


def test_tokenize_standard_https_url(tokenizer: UrlTokenizer) -> None:
    """Test tokenization of standard HTTPS URL."""
    url = "https://www.google.com/search?q=machine+learning&lang=en"
    result: TokenizedUrl = tokenizer.tokenize(url)

    assert result.scheme == "https"
    assert result.domain == "google"
    assert result.tld == "com"
    assert "www" in result.subdomains
    assert result.port is None
    assert not result.is_ip
    assert "search" in result.path_tokens
    assert "machine" in result.query_tokens
    assert "learning" in result.query_tokens


def test_tokenize_multi_part_tld(tokenizer: UrlTokenizer) -> None:
    """Test domain parsing with multi-part TLD like co.uk."""
    url = "http://sub.portal.example.co.uk/account/settings"
    result: TokenizedUrl = tokenizer.tokenize(url)

    assert result.domain == "example"
    assert result.tld == "co.uk"
    assert result.subdomains == ["sub", "portal"]
    assert "account" in result.path_tokens
    assert "settings" in result.path_tokens


def test_tokenize_raw_ipv4(tokenizer: UrlTokenizer) -> None:
    """Test tokenization of direct IPv4 address."""
    url = "http://192.168.1.1:8080/admin/login.php"
    result: TokenizedUrl = tokenizer.tokenize(url)

    assert result.is_ip is True
    assert result.port == 8080
    assert "192.168.1.1" in result.netloc
    assert "admin" in result.path_tokens
    assert "login" in result.path_tokens


def test_tokenize_raw_ipv6(tokenizer: UrlTokenizer) -> None:
    """Test tokenization of direct IPv6 address."""
    url = "http://[2001:db8::1]:80/index.html"
    result: TokenizedUrl = tokenizer.tokenize(url)

    assert result.is_ip is True
    assert result.port == 80
    assert "index" in result.path_tokens


def test_tokenize_missing_scheme(tokenizer: UrlTokenizer) -> None:
    """Test URL without explicit scheme gets normalized with http."""
    url = "example.org/path/test"
    result: TokenizedUrl = tokenizer.tokenize(url)

    assert result.scheme == "http"
    assert result.domain == "example"
    assert result.tld == "org"
    assert "path" in result.path_tokens


def test_tokenize_empty_and_whitespace(tokenizer: UrlTokenizer) -> None:
    """Test handling of empty or whitespace-only inputs."""
    result: TokenizedUrl = tokenizer.tokenize("   ")
    assert result.normalized_url == ""
    assert result.domain == ""
    assert len(result.all_tokens) == 0


def test_tokenize_non_standard_scheme(tokenizer: UrlTokenizer) -> None:
    """Test handling of non-standard schemes."""
    url = "javascript:alert('xss')"
    result: TokenizedUrl = tokenizer.tokenize(url)
    assert result.scheme == "javascript"
    assert "alert" in result.path_tokens or "xss" in result.path_tokens
