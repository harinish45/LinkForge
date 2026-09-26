"""Interpretable deterministic rule-based baseline matcher."""

from typing import Dict, List, Optional, Set, Tuple
from entitylink.data.schema import EntityRecord
from entitylink.models.base import BaseMatcher
from entitylink.features.pair_features import PairFeatureExtractor


class RuleBasedMatcher(BaseMatcher):
    """Deterministic rule-based baseline combining name, address, and country heuristics."""

    def __init__(
        self,
        name_weight: float = 0.55,
        addr_weight: float = 0.35,
        exact_boost: float = 0.10,
    ):
        self.extractor = PairFeatureExtractor()
        self.name_weight = name_weight
        self.addr_weight = addr_weight
        self.exact_boost = exact_boost

    def fit(
        self,
        pairs: List[Tuple[EntityRecord, EntityRecord]],
        labels: List[int],
    ) -> "RuleBasedMatcher":
        """Rule-based matcher requires no parameter fitting."""
        return self

    def score_pair(
        self,
        record1: EntityRecord,
        record2: EntityRecord,
    ) -> float:
        """Calculate weighted heuristic similarity score in [0.0, 1.0]."""
        feats = self.extractor.extract_features(record1, record2)

        # Country must match if known
        if feats["country_match"] == 0.0:
            return 0.0

        # Exact match shortcut
        if feats["name_norm_exact"] == 1.0 and feats["addr_norm_exact"] == 1.0:
            return 1.0

        # Name score (mix of jaccard, levenshtein, and character 3-gram)
        name_sim = (
            feats["name_token_jaccard"] * 0.4
            + feats["name_levenshtein"] * 0.3
            + feats["name_char_3gram"] * 0.3
        )

        # Address score
        addr_sim = (
            feats["addr_token_jaccard"] * 0.4
            + feats["addr_levenshtein"] * 0.3
            + feats["addr_num_overlap"] * 0.3
        )

        score = (self.name_weight * name_sim) + (self.addr_weight * addr_sim)

        # Boost for strong agreement
        if feats["strong_agreement"] == 1.0:
            score += self.exact_boost

        return float(min(1.0, max(0.0, score)))
