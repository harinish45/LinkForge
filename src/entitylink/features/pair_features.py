"""Pair-level feature extraction for candidate entity pairs."""

from typing import Dict, List, Tuple
from entitylink.data.schema import EntityRecord
from entitylink.features.string_distance import levenshtein_ratio, char_ngram_jaccard, length_ratio
from entitylink.features.token_features import token_jaccard, token_containment, number_overlap
from entitylink.preprocessing.normalizer import (
    normalize_business_name,
    normalize_address,
    normalize_country,
)

FEATURE_NAMES: List[str] = [
    # Name features
    "name_raw_exact",
    "name_norm_exact",
    "name_token_jaccard",
    "name_token_containment",
    "name_levenshtein",
    "name_char_3gram",
    "name_length_ratio",
    "name_shared_tokens",
    # Address features
    "addr_norm_exact",
    "addr_token_jaccard",
    "addr_levenshtein",
    "addr_char_3gram",
    "addr_num_overlap",
    "addr_shared_nums",
    "addr_length_ratio",
    # Cross-field & interaction features
    "country_match",
    "combined_geom_sim",
    "strong_agreement",
    "strong_disagreement",
]


class PairFeatureExtractor:
    """Extracts a dense vector of numerical similarity features for an entity pair."""

    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names or FEATURE_NAMES

    def extract_features(
        self,
        record1: EntityRecord,
        record2: EntityRecord,
    ) -> Dict[str, float]:
        """Extract a dictionary of named features for two records."""
        # 1. Names
        raw_name1 = record1.business_name
        raw_name2 = record2.business_name
        name_raw_exact = 1.0 if raw_name1 == raw_name2 and raw_name1 else 0.0

        norm_name1 = normalize_business_name(raw_name1, strip_legal_suffixes=True)
        norm_name2 = normalize_business_name(raw_name2, strip_legal_suffixes=True)
        name_norm_exact = 1.0 if norm_name1 == norm_name2 and norm_name1 else 0.0

        toks_name1 = norm_name1.split()
        toks_name2 = norm_name2.split()
        name_tok_jaccard = token_jaccard(toks_name1, toks_name2)
        name_tok_containment = token_containment(toks_name1, toks_name2)
        name_lev = levenshtein_ratio(norm_name1, norm_name2)
        name_ngram = char_ngram_jaccard(norm_name1, norm_name2, n=3)
        name_len_rat = length_ratio(norm_name1, norm_name2)
        name_shared = float(len(set(toks_name1) & set(toks_name2)))

        # 2. Addresses
        raw_addr1 = record1.business_address
        raw_addr2 = record2.business_address
        norm_addr1 = normalize_address(raw_addr1)
        norm_addr2 = normalize_address(raw_addr2)
        addr_norm_exact = 1.0 if norm_addr1 == norm_addr2 and norm_addr1 else 0.0

        toks_addr1 = norm_addr1.split()
        toks_addr2 = norm_addr2.split()
        addr_tok_jaccard = token_jaccard(toks_addr1, toks_addr2)
        addr_lev = levenshtein_ratio(norm_addr1, norm_addr2)
        addr_ngram = char_ngram_jaccard(norm_addr1, norm_addr2, n=3)
        num_ratio, shared_nums = number_overlap(norm_addr1, norm_addr2)
        addr_len_rat = length_ratio(norm_addr1, norm_addr2)

        # 3. Country & Interactions
        country1 = normalize_country(record1.country)
        country2 = normalize_country(record2.country)
        country_match = 1.0 if country1 == country2 and country1 != "UNKNOWN" else 0.0

        # Geometric mean of name and address similarities
        sim_name = (name_tok_jaccard + name_lev + name_ngram) / 3.0
        sim_addr = (addr_tok_jaccard + addr_lev + addr_ngram) / 3.0
        combined_geom = (sim_name * sim_addr) ** 0.5

        strong_agreement = 1.0 if (sim_name > 0.85 and sim_addr > 0.70 and country_match == 1.0) else 0.0
        strong_disagreement = 1.0 if (sim_name < 0.25 or country_match == 0.0) else 0.0

        feature_map: Dict[str, float] = {
            "name_raw_exact": name_raw_exact,
            "name_norm_exact": name_norm_exact,
            "name_token_jaccard": name_tok_jaccard,
            "name_token_containment": name_tok_containment,
            "name_levenshtein": name_lev,
            "name_char_3gram": name_ngram,
            "name_length_ratio": name_len_rat,
            "name_shared_tokens": name_shared,
            "addr_norm_exact": addr_norm_exact,
            "addr_token_jaccard": addr_tok_jaccard,
            "addr_levenshtein": addr_lev,
            "addr_char_3gram": addr_ngram,
            "addr_num_overlap": num_ratio,
            "addr_shared_nums": float(shared_nums),
            "addr_length_ratio": addr_len_rat,
            "country_match": country_match,
            "combined_geom_sim": combined_geom,
            "strong_agreement": strong_agreement,
            "strong_disagreement": strong_disagreement,
        }
        return feature_map

    def extract_vector(
        self,
        record1: EntityRecord,
        record2: EntityRecord,
    ) -> List[float]:
        """Extract ordered feature list according to self.feature_names."""
        f_map = self.extract_features(record1, record2)
        return [f_map[fn] for fn in self.feature_names]
