"""Malicious and phishing URL classifier combining heuristic rules and scikit-learn ML pipeline."""

import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.processors.features import UrlFeatureExtractor, UrlFeatures
from src.processors.tokenizer import TokenizedUrl, UrlTokenizer

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat severity levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class ThreatAssessment:
    """Detailed threat assessment report for an analyzed URL."""

    url: str
    is_malicious: bool
    threat_score: float
    risk_level: ThreatLevel
    confidence: float
    detected_threats: List[str]
    model_score: float
    heuristic_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment dataclass to dictionary."""
        return asdict(self)


class MaliciousUrlClassifier:
    """Hybrid malicious URL classifier powered by heuristics and an ML pipeline."""

    BRAND_KEYWORDS: set[str] = {
        "paypal", "apple", "google", "microsoft", "amazon", "chase",
        "wellsfargo", "bankofamerica", "netflix", "facebook", "instagram",
        "binance", "coinbase", "metamask", "whatsapp",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        extractor: Optional[UrlFeatureExtractor] = None,
        threshold: float = 0.50,
    ) -> None:
        self.extractor: UrlFeatureExtractor = extractor or UrlFeatureExtractor()
        self.threshold: float = threshold
        self.pipeline: Optional[Pipeline] = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self._initialize_default_model()

    def _generate_synthetic_training_data(self) -> Tuple[List[str], List[int]]:
        """Generate representative URLs and binary labels for bootstrapping."""
        benign_urls: List[str] = [
            "https://www.google.com/search?q=machine+learning",
            "https://github.com/torvalds/linux/commit/12345",
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://amazon.com/dp/B08N5WRWNW",
            "https://news.ycombinator.com/item?id=30000000",
            "https://stackoverflow.com/questions/1234567/python-type-hints",
            "https://nytimes.com/2026/10/30/technology/ai-updates.html",
            "https://docs.python.org/3/library/urllib.parse.html",
            "https://linkedin.com/in/harinish-developer",
            "https://apple.com/macbook-pro/specs/",
            "https://netflix.com/browse/genre/83",
            "https://spotify.com/album/4aawyAB9vmqN3uQFRjZTkK",
            "https://reddit.com/r/MachineLearning/top/",
            "https://bbc.com/news/world-us-canada-12345",
            "https://nature.com/articles/d41586-026-00001-w",
            "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API",
            "https://pypi.org/project/scikit-learn/",
            "https://fastapi.tiangolo.com/tutorial/first-steps/",
            "https://cloudflare.com/learning/security/what-is-ddos/",
            "https://mit.edu/research/computing.html",
        ]

        malicious_urls: List[str] = [
            "http://192.168.1.100/paypal/login/verify-account.php",
            "http://104.244.42.1/update-billing-session/confirm.html",
            "http://secure-login.paypal-update.account-verification.xyz/auth",
            "http://appleid-support-recover-account.top/password/reset",
            "http://netflix-billing-failed-update.buzz/login?session=98723",
            "http://chase-bank-verify-identity.click/login.htm@user:pass",
            "http://binance-wallet-airdrop-claim.loan/free-tokens",
            "http://chasebank-account-suspended.icu/update/credentials",
            "http://xzqjklmnpqrstvw.top/payload.exe",
            "http://facebook-security-alert-checkpoint.cf/login.php",
            "http://google-drive-shared-doc-view.ga/auth/google",
            "http://metamask-seed-phrase-verify.ml/wallet/connect",
            "http://wellsfargo-customer-notification.fit/ebanking",
            "http://microsoft-365-password-expiry.work/update",
            "http://amazon-order-refund-support.date/confirm?id=82371",
            "http://198.51.100.22/admin/login?dest=auth&action=save",
            "http://bankofamerica-secure-auth.racing/sign-in",
            "http://update-instagram-copyright-infringement.buzz/appeal",
            "http://coinbase-fraud-protection-desk.xyz/verify",
            "http://steam-trade-gift-card-free.rest/claim",
        ]

        urls: List[str] = benign_urls + malicious_urls
        labels: List[int] = [0] * len(benign_urls) + [1] * len(malicious_urls)
        return urls, labels

    def _initialize_default_model(self) -> None:
        """Initialize and fit a baseline lightweight scikit-learn classifier."""
        logger.info("Initializing baseline ML classifier pipeline")
        urls, labels = self._generate_synthetic_training_data()
        feature_vectors = [self.extractor.extract(u).to_vector() for u in urls]

        x_train = np.array(feature_vectors)
        y_train = np.array(labels)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)),
        ])
        self.pipeline.fit(x_train, y_train)

    def save(self, filepath: str) -> None:
        """Serialize trained pipeline to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load trained pipeline from disk."""
        self.pipeline = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")

    def _evaluate_heuristics(self, url: str, features: UrlFeatures) -> Tuple[float, List[str]]:
        """Evaluate heuristic rules and compute heuristic threat score."""
        threats: List[str] = []
        score: float = 0.0

        if features.is_ip:
            score += 0.40
            threats.append("Direct IP address host used in URL")
            if features.suspicious_token_count > 0:
                score += 0.35
                threats.append("IP address paired with credential/authentication keywords")

        if features.at_count > 0:
            score += 0.50
            threats.append("Suspicious '@' symbol detected (credential embedding/obfuscation)")

        if features.has_suspicious_tld:
            score += 0.30
            threats.append("Domain uses high-abuse/suspicious top-level domain")

        if features.subdomain_count >= 3:
            score += 0.25
            threats.append(f"Excessive subdomain depth ({features.subdomain_count} subdomains)")

        if features.suspicious_token_count >= 2:
            score += min(0.35, features.suspicious_token_count * 0.15)
            threats.append(f"Multiple suspicious keywords: {', '.join(features.suspicious_tokens)}")

        # Check for brand impersonation in subdomain/path
        url_lower: str = url.lower()
        for brand in self.BRAND_KEYWORDS:
            if brand in url_lower and not (f".{brand}.com" in url_lower or f"//{brand}.com" in url_lower):
                score += 0.35
                threats.append(f"Potential brand impersonation detected for '{brand}'")
                break

        # Check DGA pattern: high entropy with low vowel ratio
        if features.domain_entropy > 3.6 and features.vowel_to_consonant_ratio < 0.20:
            score += 0.30
            threats.append("High entropy and unusual character distribution (possible DGA)")

        return min(1.0, round(score, 4)), threats

    def _compute_risk_level(self, score: float) -> ThreatLevel:
        """Map threat score to risk level tier."""
        if score >= 0.80:
            return ThreatLevel.CRITICAL
        if score >= 0.60:
            return ThreatLevel.HIGH
        if score >= 0.35:
            return ThreatLevel.MEDIUM
        return ThreatLevel.LOW

    def predict(self, url: str) -> ThreatAssessment:
        """Predict whether a URL is malicious, returning a detailed ThreatAssessment."""
        features: UrlFeatures = self.extractor.extract(url)
        vector = np.array([features.to_vector()])

        ml_score: float = 0.0
        if self.pipeline is not None:
            proba = self.pipeline.predict_proba(vector)[0]
            ml_score = round(float(proba[1]), 4)

        heuristic_score, detected_threats = self._evaluate_heuristics(url, features)

        # Ensemble: take the higher signal or weighted blend
        final_score: float = round(max(ml_score, heuristic_score), 4)
        is_malicious: bool = final_score >= self.threshold
        risk_level: ThreatLevel = self._compute_risk_level(final_score)
        confidence: float = round(abs(final_score - 0.5) * 2, 4)

        return ThreatAssessment(
            url=url,
            is_malicious=is_malicious,
            threat_score=final_score,
            risk_level=risk_level,
            confidence=max(0.50, confidence),
            detected_threats=detected_threats,
            model_score=ml_score,
            heuristic_score=heuristic_score,
        )
