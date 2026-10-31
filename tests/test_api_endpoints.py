"""Unit tests for LinkForge FastAPI REST endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint() -> None:
    """Test root discovery endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "LinkForge API"
    assert "version" in data


def test_health_check_endpoint() -> None:
    """Test health check endpoints return healthy status."""
    for path in ["/health", "/api/v1/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["uptime_seconds"] >= 0


def test_analyze_single_benign_url() -> None:
    """Test single analyze endpoint on benign URL."""
    payload = {
        "url": "https://www.google.com/search?q=fastapi",
        "fetch_metadata": False,
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_valid"] is True
    assert data["url"] == payload["url"]
    assert data["classification"]["is_malicious"] is False
    assert data["classification"]["threat_score"] < 0.50
    assert data["reputation"]["category"] in ["Tech", "Other"]
    assert data["features"]["has_https"] is True
    assert data["metadata"] is None


def test_analyze_single_phishing_url() -> None:
    """Test single analyze endpoint on suspicious phishing URL."""
    payload = {
        "url": "http://192.168.1.1/paypal/login/verify-account.php",
        "fetch_metadata": False,
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_valid"] is True
    assert data["classification"]["is_malicious"] is True
    assert data["classification"]["threat_score"] >= 0.50
    assert data["features"]["is_ip"] is True
    assert len(data["classification"]["detected_threats"]) > 0


def test_analyze_localhost_url() -> None:
    """Test handling of localhost URLs."""
    payload = {
        "url": "http://localhost:8080/dashboard",
        "fetch_metadata": False,
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_valid"] is True
    assert data["features"]["has_port"] is True


def test_analyze_non_standard_scheme() -> None:
    """Test handling of non-standard javascript/data schemes."""
    payload = {
        "url": "javascript:alert(document.cookie)",
        "fetch_metadata": False,
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True


def test_analyze_empty_or_malformed_url() -> None:
    """Test rejection or graceful handling of malformed input."""
    payload = {"url": "   "}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is False
    assert "error" in data


def test_batch_analyze_urls() -> None:
    """Test batch URL analysis endpoint with multiple entries."""
    payload = {
        "urls": [
            "https://github.com",
            "https://twitter.com",
            "http://192.168.0.1/auth/login",
            "https://amazon.com",
        ],
        "fetch_metadata": False,
        "max_concurrency": 4,
    }
    response = client.post("/api/v1/analyze/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_requested"] == 4
    assert len(data["results"]) == 4
    assert data["total_successful"] == 4
    assert data["total_failed"] == 0
    assert data["duration_ms"] > 0
