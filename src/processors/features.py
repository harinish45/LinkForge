"""Lexical and statistical URL feature extraction module."""

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Set

from src.processors.tokenizer import TokenizedUrl, UrlTokenizer


@dataclass(frozen=True)
class UrlFeatures:
    """Statistical and lexical feature vector extracted from a URL."""

    # Length features
    url_length: int
    domain_length: int
    path_length: int
    query_length: int

    # Entropy features
    url_entropy: float
    domain_entropy: float

    # Structural / Host features
    domain_depth: int
    subdomain_count: int
    is_ip: bool
    is_shortened: bool
    has_https: bool
    has_port: bool

    # Character distributions & ratios
    vowel_to_consonant_ratio: float
    digit_ratio: float
    special_char_ratio: float

    # Character counts
    digit_count: int
    dot_count: int
    hyphen_count: int
    underscore_count: int
    slash_count: int
    question_count: int
    equal_count: int
    at_count: int
    percent_count: int

    # Semantic & Token features
    suspicious_token_count: int
    suspicious_tokens: List[str]
    has_suspicious_tld: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert feature dataclass to dictionary."""
        return asdict(self)

    def to_vector(self) -> List[float]:
        """Convert primary numerical features to vector for ML classifier."""
        return [
            float(self.url_length),
            float(self.domain_length),
            float(self.path_length),
            float(self.query_length),
            self.url_entropy,
            self.domain_entropy,
            float(self.domain_depth),
            float(self.subdomain_count),
            1.0 if self.is_ip else 0.0,
            1.0 if self.is_shortened else 0.0,
            1.0 if self.has_https else 0.0,
            1.0 if self.has_port else 0.0,
            self.vowel_to_consonant_ratio,
            self.digit_ratio,
            self.special_char_ratio,
            float(self.digit_count),
            float(self.dot_count),
            float(self.hyphen_count),
            float(self.underscore_count),
            float(self.slash_count),
            float(self.at_count),
            float(self.percent_count),
            float(self.suspicious_token_count),
            1.0 if self.has_suspicious_tld else 0.0,
        ]


class UrlFeatureExtractor:
    """Extracts lexical, structural, and statistical features from URLs."""

    SUSPICIOUS_TOKENS: Set[str] = {
        "login", "verify", "verification", "account", "update", "security",
        "banking", "secure", "confirm", "wallet", "password", "signin",
        "credential", "claim", "free", "gift", "bonus", "support",
        "invoice", "billing", "authenticate", "validation", "paypal",
        "appleid", "recovery", "alert", "session",
    }

    SHORTENING_DOMAINS: Set[str] = {
        "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
        "buff.ly", "adf.ly", "bit.do", "cutt.ly", "shorturl.at",
        "rb.gy", "rebrand.ly", "tiny.cc", "lnkd.in",
    }

    SUSPICIOUS_TLDS: Set[str] = {
        "xyz", "top", "tk", "ml", "ga", "cf", "gq", "buzz", "fit",
        "icu", "loan", "work", "click", "rest", "surf", "date", "racing",
    }

    VOWELS: Set[str] = set("aeiou")
    CONSONANTS: Set[str] = set("bcdfghjklmnpqrstvwxyz")

    def __init__(self, tokenizer: UrlTokenizer | None = None) -> None:
        self.tokenizer: UrlTokenizer = tokenizer or UrlTokenizer()

    def calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy for character distribution in text."""
        if not text:
            return 0.0
        length: float = float(len(text))
        counts: Counter[str] = Counter(text)
        entropy: float = 0.0
        for count in counts.values():
            probability: float = count / length
            entropy -= probability * math.log2(probability)
        return round(entropy, 4)

    def calculate_vowel_consonant_ratio(self, text: str) -> float:
        """Calculate vowel to consonant ratio in alphabetic text."""
        lower_chars: str = text.lower()
        vowels: int = sum(1 for c in lower_chars if c in self.VOWELS)
        consonants: int = sum(1 for c in lower_chars if c in self.CONSONANTS)
        if consonants == 0:
            return float(vowels)
        return round(vowels / consonants, 4)

    def _is_shortener(self, host: str) -> bool:
        """Detect whether host belongs to a known URL shortening provider."""
        clean_host: str = host.split(":")[0].lower()
        return clean_host in self.SHORTENING_DOMAINS

    def _find_suspicious_tokens(self, tokens: List[str]) -> List[str]:
        """Find matches against known suspicious token vocabulary."""
        matched: List[str] = []
        for token in tokens:
            cleaned: str = token.lower()
            if cleaned in self.SUSPICIOUS_TOKENS:
                matched.append(cleaned)
        return sorted(list(set(matched)))

    def _count_special_chars(self, text: str) -> int:
        """Count non-alphanumeric characters in URL string."""
        return sum(1 for c in text if not c.isalnum())

    def extract(self, url: str) -> UrlFeatures:
        """Extract all features from URL string."""
        tokenized: TokenizedUrl = self.tokenizer.tokenize(url)
        normalized: str = tokenized.normalized_url
        host: str = tokenized.netloc

        url_len: int = len(normalized)
        domain_str: str = f"{tokenized.domain}.{tokenized.tld}" if tokenized.tld else tokenized.domain

        # Calculate character statistics
        special_count: int = self._count_special_chars(normalized)
        digit_count: int = sum(1 for c in normalized if c.isdigit())
        digit_ratio: float = round(digit_count / url_len, 4) if url_len > 0 else 0.0
        special_ratio: float = round(special_count / url_len, 4) if url_len > 0 else 0.0

        # Suspicious elements detection
        found_suspicious: List[str] = self._find_suspicious_tokens(tokenized.all_tokens)
        suspicious_tld: bool = tokenized.tld.lower() in self.SUSPICIOUS_TLDS

        return UrlFeatures(
            url_length=url_len,
            domain_length=len(domain_str),
            path_length=len(tokenized.path),
            query_length=len(tokenized.query),
            url_entropy=self.calculate_entropy(normalized),
            domain_entropy=self.calculate_entropy(domain_str),
            domain_depth=len(tokenized.subdomains) + 1 if tokenized.domain else 0,
            subdomain_count=len(tokenized.subdomains),
            is_ip=tokenized.is_ip,
            is_shortened=self._is_shortener(host),
            has_https=tokenized.scheme == "https",
            has_port=tokenized.port is not None,
            vowel_to_consonant_ratio=self.calculate_vowel_consonant_ratio(domain_str or normalized),
            digit_ratio=digit_ratio,
            special_char_ratio=special_ratio,
            digit_count=digit_count,
            dot_count=normalized.count("."),
            hyphen_count=normalized.count("-"),
            underscore_count=normalized.count("_"),
            slash_count=normalized.count("/"),
            question_count=normalized.count("?"),
            equal_count=normalized.count("="),
            at_count=normalized.count("@"),
            percent_count=normalized.count("%"),
            suspicious_token_count=len(found_suspicious),
            suspicious_tokens=found_suspicious,
            has_suspicious_tld=suspicious_tld,
        )
