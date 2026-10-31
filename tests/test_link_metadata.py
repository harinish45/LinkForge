"""Unit tests for Link Metadata Enrichment processor."""

import pytest
import httpx
from bs4 import BeautifulSoup
from src.processors.metadata import LinkMetadataEnricher, UrlMetadata


@pytest.fixture
def enricher() -> LinkMetadataEnricher:
    """Fixture providing LinkMetadataEnricher instance with short timeout."""
    return LinkMetadataEnricher(timeout=1.0)


def test_html_parsing_extracts_title_and_description(enricher: LinkMetadataEnricher) -> None:
    """Test extracting OpenGraph and standard meta tags from HTML."""
    html_doc = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Original Title</title>
        <meta property="og:title" content="Social Media Title" />
        <meta property="og:description" content="This is an OG description" />
        <link rel="icon" href="/static/favicon.ico" />
    </head>
    <body><h1>Welcome</h1></body>
    </html>
    """
    title, desc, favicon = enricher._parse_html_body(html_doc, "https://example.com/page")

    assert title == "Social Media Title"
    assert desc == "This is an OG description"
    assert favicon == "https://example.com/static/favicon.ico"


def test_html_parsing_fallback_to_standard_title(enricher: LinkMetadataEnricher) -> None:
    """Test fallback when open graph tags are missing."""
    html_doc = """
    <html>
    <head>
        <title>Standard Page Title</title>
        <meta name="description" content="Standard description content" />
    </head>
    <body></body>
    </html>
    """
    title, desc, favicon = enricher._parse_html_body(html_doc, "https://sample.org")

    assert title == "Standard Page Title"
    assert desc == "Standard description content"
    assert favicon == "https://sample.org/favicon.ico"


def test_disallowed_schemes_rejected_safely(enricher: LinkMetadataEnricher) -> None:
    """Test non-HTTP schemes are rejected without network calls."""
    js_result: UrlMetadata = enricher.fetch("javascript:void(0)")
    assert js_result.status_code is None
    assert "Unsupported scheme" in (js_result.error or "")

    file_result: UrlMetadata = enricher.fetch("file:///etc/passwd")
    assert file_result.status_code is None
    assert "Unsupported scheme" in (file_result.error or "")


@pytest.mark.asyncio
async def test_async_disallowed_scheme(enricher: LinkMetadataEnricher) -> None:
    """Test async rejection of invalid schemes."""
    res = await enricher.fetch_async("data:text/plain;base64,SGVsbG8=")
    assert res.status_code is None
    assert "Unsupported scheme" in (res.error or "")
