"""Unit tests for string distance and feature engineering."""

import pytest
from entitylink.data.schema import EntityRecord
from entitylink.features.string_distance import levenshtein_ratio, char_ngram_jaccard, length_ratio
from entitylink.features.token_features import token_jaccard, token_containment, number_overlap
from entitylink.features.pair_features import PairFeatureExtractor


def test_string_distances():
    assert levenshtein_ratio("acme corp", "acme corp") == 1.0
    assert levenshtein_ratio("acme corp", "acme corporation") > 0.70
    assert char_ngram_jaccard("google inc", "google llc", n=3) > 0.40
    assert length_ratio("short", "longer_string") == 5.0 / 13.0


def test_token_features():
    toks1 = ["acme", "logistics", "pvt", "ltd"]
    toks2 = ["acme", "logistics", "co"]
    assert token_jaccard(toks1, toks2) == 2.0 / 5.0
    assert token_containment(toks1, toks2) == 2.0 / 3.0

    ratio, shared = number_overlap("100 Main St Apt 4B", "Suite 4B 100 Main Street")
    assert shared >= 1


def test_pair_feature_extractor():
    extractor = PairFeatureExtractor()
    r1 = EntityRecord("S1-1", "Prime Money", "17560 Ellis Road, Tahlequah, OK", "US")
    r2 = EntityRecord("S2-2", "Prime Money LLC", "17560 Ellis Rd, Tahlequah", "US")
    feats = extractor.extract_features(r1, r2)

    assert feats["country_match"] == 1.0
    assert feats["name_norm_exact"] == 1.0  # LLC normalized
    assert feats["name_token_jaccard"] > 0.8
    assert feats["addr_levenshtein"] > 0.7
    assert feats["combined_geom_sim"] > 0.7
    assert feats["strong_disagreement"] == 0.0
