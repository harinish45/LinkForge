"""Unit tests for data schema and entity validation."""

import pytest
from entitylink.data.schema import (
    EntityRecord,
    validate_entity_id,
    validate_records,
)


def test_entity_record_source():
    rec1 = EntityRecord("S1-1234", "Acme Inc", "123 Main St", "US")
    rec2 = EntityRecord("S2-5678", "Acme Corp", "123 Main Street", "US")
    rec3 = EntityRecord("S3-9999", "Acme Ltd", "Main Rd", "India")
    rec_bad = EntityRecord("X-0000", "Unknown", "Somewhere", "France")

    assert rec1.source == "S1"
    assert rec2.source == "S2"
    assert rec3.source == "S3"
    assert rec_bad.source == "UNKNOWN"


def test_validate_entity_id():
    assert validate_entity_id("S1-12345") is True
    assert validate_entity_id("S2-98765") is True
    assert validate_entity_id("S3-44444") is True
    assert validate_entity_id("S1-12345", expected_prefix="S1-") is True
    assert validate_entity_id("S2-12345", expected_prefix="S1-") is False
    assert validate_entity_id("INVALID") is False
    assert validate_entity_id("") is False


def test_validate_records_integrity():
    records = [
        EntityRecord("S1-100", "Biz A", "100 Road", "US"),
        EntityRecord("S1-101", "Biz B", "200 Ave", "India"),
    ]
    report = validate_records(records, expected_prefix="S1-")
    assert report.is_valid is True
    assert report.total_records == 2
    assert len(report.duplicate_ids) == 0
    assert len(report.malformed_ids) == 0
