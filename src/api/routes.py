"""FastAPI router endpoints for LinkForge link analysis API."""

import asyncio
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    DomainReputationResponse,
    FeaturesResponse,
    HealthResponse,
    MetadataResponse,
    ThreatResponse,
)
from src.models.classifier import MaliciousUrlClassifier, ThreatAssessment
from src.models.reputation import DomainReputationClassifier, DomainReputationResult
from src.processors.features import UrlFeatureExtractor, UrlFeatures
from src.processors.metadata import LinkMetadataEnricher, UrlMetadata

logger = logging.getLogger(__name__)

router = APIRouter()
START_TIME: float = time.time()

# Shared service instances
feature_extractor = UrlFeatureExtractor()
classifier = MaliciousUrlClassifier(extractor=feature_extractor)
reputation_classifier = DomainReputationClassifier()


def _is_url_valid(url: str) -> bool:
    """Validate if URL has non-empty host or domain structure."""
    trimmed = url.strip()
    if not trimmed:
        return False
    # Filter out empty or whitespace-only inputs
    return len(trimmed) >= 3


async def analyze_single_url(
    url: str,
    fetch_metadata: bool = False,
    timeout: float = 5.0,
) -> AnalyzeResponse:
    """Process a single URL through feature extractor, classifier, and reputation engine."""
    if not _is_url_valid(url):
        return AnalyzeResponse(
            url=url,
            is_valid=False,
            error="Malformed or empty URL input",
        )

    try:
        features: UrlFeatures = feature_extractor.extract(url)
        threat: ThreatAssessment = classifier.predict(url)
        reputation: DomainReputationResult = reputation_classifier.evaluate(url)

        metadata_resp: Optional[MetadataResponse] = None
        if fetch_metadata:
            enricher = LinkMetadataEnricher(timeout=timeout)
            meta: UrlMetadata = await enricher.fetch_async(url)
            metadata_resp = MetadataResponse(
                url=meta.url,
                final_url=meta.final_url,
                status_code=meta.status_code,
                title=meta.title,
                description=meta.description,
                favicon_url=meta.favicon_url,
                content_type=meta.content_type,
                response_time_ms=meta.response_time_ms,
                redirect_chain=meta.redirect_chain,
                error=meta.error,
            )

        features_resp = FeaturesResponse(
            url_length=features.url_length,
            domain_length=features.domain_length,
            path_length=features.path_length,
            query_length=features.query_length,
            url_entropy=features.url_entropy,
            domain_entropy=features.domain_entropy,
            domain_depth=features.domain_depth,
            subdomain_count=features.subdomain_count,
            is_ip=features.is_ip,
            is_shortened=features.is_shortened,
            has_https=features.has_https,
            has_port=features.has_port,
            vowel_to_consonant_ratio=features.vowel_to_consonant_ratio,
            digit_ratio=features.digit_ratio,
            special_char_ratio=features.special_char_ratio,
            suspicious_token_count=features.suspicious_token_count,
            suspicious_tokens=features.suspicious_tokens,
            has_suspicious_tld=features.has_suspicious_tld,
        )

        threat_resp = ThreatResponse(
            is_malicious=threat.is_malicious,
            threat_score=threat.threat_score,
            risk_level=threat.risk_level.value,
            confidence=threat.confidence,
            detected_threats=threat.detected_threats,
            model_score=threat.model_score,
            heuristic_score=threat.heuristic_score,
        )

        rep_resp = DomainReputationResponse(
            domain=reputation.domain,
            reputation_score=reputation.reputation_score,
            tier=reputation.tier.value,
            category=reputation.category.value,
            category_confidence=reputation.category_confidence,
            factors=reputation.factors,
        )

        return AnalyzeResponse(
            url=url,
            is_valid=True,
            features=features_resp,
            classification=threat_resp,
            reputation=rep_resp,
            metadata=metadata_resp,
        )
    except Exception as exc:
        logger.error(f"Error analyzing URL {url}: {exc}")
        return AnalyzeResponse(
            url=url,
            is_valid=False,
            error=f"Processing exception: {str(exc)}",
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return API health status and model availability."""
    uptime: float = round(time.time() - START_TIME, 2)
    model_ready: bool = classifier.pipeline is not None
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=model_ready,
        uptime_seconds=uptime,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_url(payload: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze a single URL for phishing threats, reputation, and features."""
    return await analyze_single_url(
        url=payload.url,
        fetch_metadata=payload.fetch_metadata,
        timeout=payload.timeout,
    )


@router.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(payload: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    """Analyze a batch of URLs with bounded concurrency."""
    start_time: float = time.perf_counter()
    semaphore = asyncio.Semaphore(payload.max_concurrency)

    async def _worker(url: str) -> AnalyzeResponse:
        async with semaphore:
            return await analyze_single_url(
                url=url,
                fetch_metadata=payload.fetch_metadata,
            )

    tasks = [_worker(u) for u in payload.urls]
    results: List[AnalyzeResponse] = await asyncio.gather(*tasks)

    elapsed_ms: float = round((time.perf_counter() - start_time) * 1000, 2)
    successful = sum(1 for r in results if r.is_valid and r.error is None)
    failed = len(results) - successful

    return BatchAnalyzeResponse(
        total_requested=len(payload.urls),
        total_successful=successful,
        total_failed=failed,
        duration_ms=elapsed_ms,
        results=results,
    )
