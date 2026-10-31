"""Domain reputation and category classification module."""

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from src.processors.tokenizer import TokenizedUrl, UrlTokenizer


class DomainCategory(str, Enum):
    """Categorical classification for domains and URLs."""

    SOCIAL = "Social"
    COMMERCE = "Commerce"
    TECH = "Tech"
    FINANCE = "Finance"
    NEWS = "News"
    MALICIOUS = "Malicious/Phishing"
    GOVERNMENT_EDUCATION = "Government/Education"
    ENTERTAINMENT = "Entertainment"
    OTHER = "Other"


class ReputationTier(str, Enum):
    """Reputation classification tiers."""

    TRUSTED = "TRUSTED"
    NEUTRAL = "NEUTRAL"
    SUSPICIOUS = "SUSPICIOUS"
    MALICIOUS = "MALICIOUS"


@dataclass(frozen=True)
class DomainReputationResult:
    """Domain reputation assessment and categorization."""

    domain: str
    reputation_score: int
    tier: ReputationTier
    category: DomainCategory
    category_confidence: float
    factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


class DomainReputationClassifier:
    """Evaluates domain reputation score and classifies into business categories."""

    KNOWN_DOMAINS: Dict[str, Tuple[DomainCategory, int]] = {
        # Social
        "twitter.com": (DomainCategory.SOCIAL, 95),
        "x.com": (DomainCategory.SOCIAL, 95),
        "facebook.com": (DomainCategory.SOCIAL, 92),
        "instagram.com": (DomainCategory.SOCIAL, 93),
        "linkedin.com": (DomainCategory.SOCIAL, 96),
        "reddit.com": (DomainCategory.SOCIAL, 90),
        "tiktok.com": (DomainCategory.SOCIAL, 88),
        "pinterest.com": (DomainCategory.SOCIAL, 90),
        "discord.com": (DomainCategory.SOCIAL, 89),
        "t.me": (DomainCategory.SOCIAL, 80),
        "telegram.org": (DomainCategory.SOCIAL, 90),
        "whatsapp.com": (DomainCategory.SOCIAL, 95),
        # Commerce
        "amazon.com": (DomainCategory.COMMERCE, 98),
        "ebay.com": (DomainCategory.COMMERCE, 94),
        "walmart.com": (DomainCategory.COMMERCE, 95),
        "target.com": (DomainCategory.COMMERCE, 94),
        "shopify.com": (DomainCategory.COMMERCE, 94),
        "aliexpress.com": (DomainCategory.COMMERCE, 85),
        "etsy.com": (DomainCategory.COMMERCE, 92),
        "bestbuy.com": (DomainCategory.COMMERCE, 93),
        # Tech
        "github.com": (DomainCategory.TECH, 98),
        "gitlab.com": (DomainCategory.TECH, 95),
        "google.com": (DomainCategory.TECH, 99),
        "microsoft.com": (DomainCategory.TECH, 99),
        "apple.com": (DomainCategory.TECH, 99),
        "cloudflare.com": (DomainCategory.TECH, 97),
        "stackoverflow.com": (DomainCategory.TECH, 96),
        "python.org": (DomainCategory.TECH, 98),
        "aws.amazon.com": (DomainCategory.TECH, 98),
        "azure.com": (DomainCategory.TECH, 98),
        "docker.com": (DomainCategory.TECH, 95),
        # Finance
        "paypal.com": (DomainCategory.FINANCE, 97),
        "stripe.com": (DomainCategory.FINANCE, 97),
        "chase.com": (DomainCategory.FINANCE, 98),
        "bankofamerica.com": (DomainCategory.FINANCE, 98),
        "wellsfargo.com": (DomainCategory.FINANCE, 97),
        "coinbase.com": (DomainCategory.FINANCE, 92),
        "binance.com": (DomainCategory.FINANCE, 88),
        "visa.com": (DomainCategory.FINANCE, 98),
        "mastercard.com": (DomainCategory.FINANCE, 98),
        # News
        "nytimes.com": (DomainCategory.NEWS, 95),
        "bbc.com": (DomainCategory.NEWS, 97),
        "reuters.com": (DomainCategory.NEWS, 98),
        "bloomberg.com": (DomainCategory.NEWS, 96),
        "theguardian.com": (DomainCategory.NEWS, 94),
        "cnn.com": (DomainCategory.NEWS, 92),
        "forbes.com": (DomainCategory.NEWS, 91),
        "wsj.com": (DomainCategory.NEWS, 96),
        # Entertainment
        "youtube.com": (DomainCategory.ENTERTAINMENT, 98),
        "netflix.com": (DomainCategory.ENTERTAINMENT, 96),
        "spotify.com": (DomainCategory.ENTERTAINMENT, 95),
        "twitch.tv": (DomainCategory.ENTERTAINMENT, 92),
    }

    CATEGORY_KEYWORDS: Dict[DomainCategory, Set[str]] = {
        DomainCategory.COMMERCE: {"shop", "store", "market", "cart", "buy", "pay", "order", "sale", "deals"},
        DomainCategory.TECH: {"tech", "dev", "code", "cloud", "api", "software", "data", "ai", "stack", "system"},
        DomainCategory.FINANCE: {"bank", "invest", "crypto", "credit", "loan", "trade", "wealth", "forex", "fund"},
        DomainCategory.NEWS: {"news", "daily", "press", "times", "post", "herald", "gazette", "tribune", "journal"},
        DomainCategory.SOCIAL: {"social", "chat", "forum", "community", "meet", "share", "connect"},
        DomainCategory.ENTERTAINMENT: {"video", "stream", "music", "game", "play", "cinema", "movie", "tv"},
    }

    SUSPICIOUS_TLDS: Set[str] = {
        "xyz", "top", "tk", "ml", "ga", "cf", "gq", "buzz", "fit", "icu", "loan", "work", "click",
    }

    def __init__(self, tokenizer: Optional[UrlTokenizer] = None) -> None:
        self.tokenizer: UrlTokenizer = tokenizer or UrlTokenizer()

    def _determine_tier(self, score: int) -> ReputationTier:
        """Map numerical score to reputation tier."""
        if score >= 80:
            return ReputationTier.TRUSTED
        if score >= 50:
            return ReputationTier.NEUTRAL
        if score >= 25:
            return ReputationTier.SUSPICIOUS
        return ReputationTier.MALICIOUS

    def _classify_category(self, full_domain: str, tokens: List[str]) -> Tuple[DomainCategory, float]:
        """Classify domain into primary category with confidence."""
        if full_domain in self.KNOWN_DOMAINS:
            cat, _ = self.KNOWN_DOMAINS[full_domain]
            return cat, 0.99

        if full_domain.endswith(".gov") or full_domain.endswith(".edu") or full_domain.endswith(".ac.uk"):
            return DomainCategory.GOVERNMENT_EDUCATION, 0.95

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in full_domain or kw in tokens:
                    return category, 0.75

        return DomainCategory.OTHER, 0.50

    def evaluate(self, url: str) -> DomainReputationResult:
        """Assess domain reputation score and category classification."""
        tokenized: TokenizedUrl = self.tokenizer.tokenize(url)
        full_domain: str = f"{tokenized.domain}.{tokenized.tld}" if tokenized.tld else tokenized.domain
        factors: List[str] = []

        # Check known domain registry
        if full_domain in self.KNOWN_DOMAINS:
            category, score = self.KNOWN_DOMAINS[full_domain]
            factors.append("Verified high-reputation domain")
            return DomainReputationResult(
                domain=full_domain,
                reputation_score=score,
                tier=self._determine_tier(score),
                category=category,
                category_confidence=0.99,
                factors=factors,
            )

        # Baseline score for unknown domains
        score = 65
        factors.append("Standard unknown domain baseline")

        # IP address penalty
        if tokenized.is_ip:
            score -= 40
            factors.append("Direct IP address used (-40)")

        # Suspicious TLD penalty
        if tokenized.tld.lower() in self.SUSPICIOUS_TLDS:
            score -= 25
            factors.append(f"High-abuse TLD .{tokenized.tld} (-25)")

        # Subdomain depth penalty
        if len(tokenized.subdomains) > 2:
            penalty = min(20, len(tokenized.subdomains) * 7)
            score -= penalty
            factors.append(f"Excessive subdomains: {len(tokenized.subdomains)} (-{penalty})")

        # Hyphens in domain penalty
        hyphen_count = full_domain.count("-")
        if hyphen_count >= 2:
            penalty = min(15, hyphen_count * 5)
            score -= penalty
            factors.append(f"Multiple hyphens in domain: {hyphen_count} (-{penalty})")

        # HTTPS bonus
        if tokenized.scheme == "https":
            score += 5
            factors.append("Valid HTTPS scheme (+5)")

        # Bound score between 0 and 100
        score = max(0, min(100, score))
        category, confidence = self._classify_category(full_domain, tokenized.all_tokens)

        # If score is critically low, flag as Malicious/Phishing category
        if score < 25 and category == DomainCategory.OTHER:
            category = DomainCategory.MALICIOUS
            confidence = 0.85

        return DomainReputationResult(
            domain=full_domain,
            reputation_score=score,
            tier=self._determine_tier(score),
            category=category,
            category_confidence=confidence,
            factors=factors,
        )
