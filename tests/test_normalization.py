"""Unit tests for text normalization module."""

import pytest
from entitylink.preprocessing.normalizer import (
    normalize_text,
    normalize_business_name,
    get_business_name_compact,
    get_business_name_tokens,
    normalize_address,
    get_address_compact,
    get_address_tokens,
    normalize_country,
    build_normalized_record,
)


def test_normalize_text_unicode_and_case():
    assert normalize_text("Café René & Co.") == "cafe rene co"
    assert normalize_text("ACME   Corporation!!!") == "acme corporation"


def test_normalize_business_name_legal_suffixes():
    assert normalize_business_name("Acme Corporation") == "acme corp"
    assert normalize_business_name("Apex Logistics Private Limited") == "apex logistics pvt ltd"
    assert normalize_business_name("Global Tech Solutions Ltd") == "global tech sol ltd"


def test_compact_name_and_tokens():
    raw = "Royal Hotel & Suites, LLC"
    assert get_business_name_compact(raw) == "royalhotelsuitesllc"
    tokens = get_business_name_tokens(raw)
    assert "royal" in tokens
    assert "hotel" in tokens
    assert "suites" in tokens


def test_normalize_address_keywords():
    addr = "123 Main Street Suite 100"
    norm = normalize_address(addr)
    assert "st" in norm
    assert "ste" in norm
    assert get_address_compact(addr) == "123mainstste100"


def test_normalize_country_open_set():
    assert normalize_country("US") == "US"
    assert normalize_country("United States of America") == "US"
    assert normalize_country("U.S.A.") == "US"
    assert normalize_country("IND") == "INDIA"
    assert normalize_country("Bharat") == "INDIA"
    assert normalize_country("Brazil") == "BRAZIL"  # Open-set preserved
    assert normalize_country("XYZ Unknown Country") == "XYZ UNKNOWN COUNTRY"
    assert normalize_country("") == "UNKNOWN"


def test_build_normalized_record():
    rec = build_normalized_record(
        entity_id="S1-001",
        business_name="Acme Corporation",
        business_address="123 Main Street",
        country="USA"
    )
    assert rec.entity_id == "S1-001"
    assert rec.business_name_normalized == "acme corp"
    assert rec.business_name_compact == "acmecorp"
    assert rec.country_normalized == "US"
    assert "st" in rec.business_address_normalized
