"""Pydantic schemas for LinkForge API requests and responses."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request payload for single URL analysis."""

    url: str = Field(..., min_length=1, max_length=4096, description="URL string to analyze")
    fetch_metadata: bool = Field(default=False, description="Whether to fetch remote preview metadata")
    timeout: float = Field(default=5.0, ge=0.5, le=30.0, description="HTTP metadata request timeout in seconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://www.google.com/search?q=cybersecurity",
                "fetch_metadata": False,
                "timeout": 5.0,
            }
        }
    }


class BatchAnalyzeRequest(BaseModel):
    """Request payload for batch URL analysis."""

    urls: List[str] = Field(..., min_length=1, max_length=100, description="List of URL strings to analyze")
    fetch_metadata: bool = Field(default=False, description="Whether to fetch remote preview metadata for each URL")
    max_concurrency: int = Field(default=10, ge=1, le=25, description="Maximum concurrent worker tasks")

    model_config = {
        "json_schema_extra": {
            "example": {
                "urls": [
                    "https://www.github.com",
                    "http://192.168.1.1/paypal/login.php",
                ],
                "fetch_metadata": False,
                "max_concurrency": 5,
            }
        }
    }


class FeaturesResponse(BaseModel):
    """Extracted URL lexical and statistical features."""

    url_length: int
    domain_length: int
    path_length: int
    query_length: int
    url_entropy: float
    domain_entropy: float
    domain_depth: int
    subdomain_count: int
    is_ip: bool
    is_shortened: bool
    has_https: bool
    has_port: bool
    vowel_to_consonant_ratio: float
    digit_ratio: float
    special_char_ratio: float
    suspicious_token_count: int
    suspicious_tokens: List[str]
    has_suspicious_tld: bool


class ThreatResponse(BaseModel):
    """Malicious and phishing classification outcome."""

    is_malicious: bool
    threat_score: float
    risk_level: str
    confidence: float
    detected_threats: List[str]
    model_score: float
    heuristic_score: float


class DomainReputationResponse(BaseModel):
    """Domain reputation score and categorization."""

    domain: str
    reputation_score: int
    tier: str
    category: str
    category_confidence: float
    factors: List[str]


class MetadataResponse(BaseModel):
    """Live HTTP preview metadata response."""

    url: str
    final_url: str
    status_code: Optional[int] = None
    title: Optional[str] = None
    description: Optional[str] = None
    favicon_url: Optional[str] = None
    content_type: Optional[str] = None
    response_time_ms: float
    redirect_chain: List[str] = []
    error: Optional[str] = None


class AnalyzeResponse(BaseModel):
    """Full analysis result for an individual URL."""

    url: str
    is_valid: bool
    features: Optional[FeaturesResponse] = None
    classification: Optional[ThreatResponse] = None
    reputation: Optional[DomainReputationResponse] = None
    metadata: Optional[MetadataResponse] = None
    error: Optional[str] = None
    processed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BatchAnalyzeResponse(BaseModel):
    """Response payload containing batch analysis results."""

    total_requested: int
    total_successful: int
    total_failed: int
    duration_ms: float
    results: List[AnalyzeResponse]


class HealthResponse(BaseModel):
    """Service health check status."""

    status: str
    version: str
    model_loaded: bool
    uptime_seconds: float
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
