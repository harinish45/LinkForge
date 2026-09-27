"""Unit tests for multi-strategy blocking engine."""

import pytest
from entitylink.data.schema import EntityRecord
from entitylink.blocking.candidate_generator import (
    generate_candidates,
    MultiStrategyBlockingEngine,
    ExactNameBlocker,
    CompactNameBlocker,
    InformativeTokenBlocker,
    CharacterNGramBlocker,
    AddressAnchorBlocker,
)
from entitylink.preprocessing.normalizer import build_normalized_record


def test_blocker_key_extraction():
    rec = build_normalized_record(
        entity_id="S1-001",
        business_name="Acme Corporation",
        business_address="123 Main Street Suite 100",
        country="USA"
    )
    exact_keys = ExactNameBlocker().extract_keys(rec)
    compact_keys = CompactNameBlocker().extract_keys(rec)
    token_keys = InformativeTokenBlocker().extract_keys(rec)
    ngram_keys = CharacterNGramBlocker(ngram_size=3).extract_keys(rec)
    addr_keys = AddressAnchorBlocker().extract_keys(rec)

    assert "exact_name#acme corp" in exact_keys
    assert "compact_name#acmecorp" in compact_keys
    assert "token#acme" in token_keys
    assert "addr_num#123" in addr_keys or "addr_num#100" in addr_keys


def test_generate_candidates_integration():
    source1 = {
        "S1-001": EntityRecord("S1-001", "Acme Corporation", "123 Main St", "US"),
        "S1-002": EntityRecord("S1-002", "Apex Logistics Ltd", "45 Industrial Rd", "India"),
    }
    source2 = {
        "S2-001": EntityRecord("S2-001", "ACME Corp", "123 Main Street", "USA"),
        "S2-002": EntityRecord("S2-002", "Other Company", "99 Random Rd", "US"),
    }
    source3 = {
        "S3-001": EntityRecord("S3-001", "Apex Logistics Pvt Ltd", "45 Ind. Rd", "India"),
    }

    candidates = generate_candidates(source1, source2, source3)

    assert "S2-001" in candidates["S1-001"]
    assert "S3-001" in candidates["S1-002"]
    assert "S2-002" not in candidates["S1-001"]


def test_candidate_provenance_and_limits():
    s1_norm = {"S1-001": build_normalized_record("S1-001", "Acme Corp", "123 Main", "US")}
    s2_norm = {"S2-001": build_normalized_record("S2-001", "Acme Corp", "123 Main", "US")}

    engine = MultiStrategyBlockingEngine(config={"max_candidates_per_entity": 10})
    cand_prov = engine.generate_candidates_with_provenance(s1_norm, s2_norm)

    assert "S2-001" in cand_prov["S1-001"]
    reasons = cand_prov["S1-001"]["S2-001"].block_reasons
    assert "name_exact" in reasons or "name_compact" in reasons
