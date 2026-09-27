"""Robust text normalization module for business names, addresses, and countries.

Preserves original raw fields while generating normalized, compact, and tokenized representations.
Handles open-set country codes, legal suffix canonicalization, and address keyword standardization.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


# Standard legal suffix canonicalization mapping
LEGAL_SUFFIXES: Dict[str, str] = {
    "corporation": "corp",
    "corp": "corp",
    "incorporated": "inc",
    "inc": "inc",
    "limited": "ltd",
    "ltd": "ltd",
    "private": "pvt",
    "pvt": "pvt",
    "company": "co",
    "co": "co",
    "llc": "llc",
    "plc": "plc",
    "gmbh": "gmbh",
    "sarl": "sarl",
    "sa": "sa",
}

LEGAL_SUFFIX_SET: Set[str] = set(LEGAL_SUFFIXES.values()) | set(LEGAL_SUFFIXES.keys())

# Standard address keyword canonicalization mapping
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    "street": "st",
    "st": "st",
    "road": "rd",
    "rd": "rd",
    "avenue": "ave",
    "ave": "ave",
    "boulevard": "blvd",
    "blvd": "blvd",
    "drive": "dr",
    "dr": "dr",
    "lane": "ln",
    "ln": "ln",
    "court": "ct",
    "ct": "ct",
    "apartment": "apt",
    "apt": "apt",
    "suite": "ste",
    "ste": "ste",
    "building": "bldg",
    "bldg": "bldg",
    "floor": "fl",
    "fl": "fl",
    "highway": "hwy",
    "hwy": "hwy",
    "parkway": "pkwy",
    "pkwy": "pkwy",
    "square": "sq",
    "sq": "sq",
    "industrial": "ind",
    "ind": "ind",
    "estate": "est",
    "est": "est",
}

# General non-destructive word canonicalizations (abbreviations commonly found in business names)
BUSINESS_ABBREVIATIONS: Dict[str, str] = {
    "international": "intl",
    "intl": "intl",
    "solutions": "sol",
    "sol": "sol",
    "technology": "tech",
    "tech": "tech",
    "technologies": "tech",
    "services": "svc",
    "svc": "svc",
    "center": "ctr",
    "centre": "ctr",
    "ctr": "ctr",
    "management": "mgmt",
    "mgmt": "mgmt",
    "development": "dev",
    "dev": "dev",
}

# High-frequency generic stopwords (never strip domain words like bank, royal, national, hospital)
GENERIC_STOPWORDS: Set[str] = {
    "and", "&", "the", "of", "in", "for", "on", "at", "to", "a", "an"
}


@dataclass
class NormalizedRecord:
    """Encapsulates raw and normalized representations of an entity record."""
    entity_id: str
    business_name_raw: str
    business_name_normalized: str
    business_name_compact: str
    business_name_tokens: List[str]
    
    business_address_raw: str
    business_address_normalized: str
    business_address_compact: str
    business_address_tokens: List[str]
    
    country_raw: str
    country_normalized: str


def unicode_normalize(text: str) -> str:
    """Decompose Unicode characters to ASCII equivalent where possible."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def normalize_text(text: str) -> str:
    """Clean, lowercase, strip punctuation, and collapse whitespace."""
    if not text:
        return ""
    t = unicode_normalize(text).lower()
    # Replace punctuation and special characters with spaces
    t = re.sub(r"[^\w\s]", " ", t)
    # Collapse multiple spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_business_name(name: str, strip_legal_suffixes: bool = False) -> str:
    """Normalize a business name by standardizing terms and optional suffix cleanup."""
    clean = normalize_text(name)
    tokens = clean.split()
    if not tokens:
        return ""

    canonicalized: List[str] = []
    for token in tokens:
        if token in GENERIC_STOPWORDS:
            continue
        canon = LEGAL_SUFFIXES.get(token, BUSINESS_ABBREVIATIONS.get(token, token))
        canonicalized.append(canon)

    if strip_legal_suffixes:
        while len(canonicalized) > 1 and canonicalized[-1] in LEGAL_SUFFIX_SET:
            canonicalized.pop()

    return " ".join(canonicalized)


def get_business_name_compact(name: str) -> str:
    """Return compact (no spaces or punctuation) version of business name."""
    norm = normalize_business_name(name, strip_legal_suffixes=False)
    return norm.replace(" ", "")


def get_business_name_tokens(name: str) -> List[str]:
    """Extract informative non-stopword tokens from business name."""
    norm = normalize_business_name(name, strip_legal_suffixes=False)
    tokens = norm.split()
    return [t for t in tokens if t not in GENERIC_STOPWORDS and len(t) >= 2]


def normalize_address(address: str) -> str:
    """Normalize a business address with standardized keywords."""
    clean = normalize_text(address)
    tokens = clean.split()
    if not tokens:
        return ""

    canonicalized: List[str] = []
    for token in tokens:
        if token in GENERIC_STOPWORDS:
            continue
        canon = ADDRESS_ABBREVIATIONS.get(token, token)
        canonicalized.append(canon)

    return " ".join(canonicalized)


def get_address_compact(address: str) -> str:
    """Return compact version of normalized address."""
    norm = normalize_address(address)
    return norm.replace(" ", "")


def get_address_tokens(address: str) -> List[str]:
    """Extract tokens from normalized address."""
    norm = normalize_address(address)
    return norm.split()


def normalize_country(country: str) -> str:
    """Normalize country string in an open-set, unconstrained manner."""
    if not country:
        return "UNKNOWN"
    
    clean = unicode_normalize(country).strip().upper()
    clean = re.sub(r"[^\w\s]", "", clean).strip()
    
    if not clean:
        return "UNKNOWN"

    # Known standard aliases mapped to consistent codes
    alias_map = {
        "UNITED STATES": "US",
        "UNITED STATES OF AMERICA": "US",
        "USA": "US",
        "US": "US",
        "U S A": "US",
        "U S": "US",
        "INDIA": "INDIA",
        "IND": "INDIA",
        "BHARAT": "INDIA",
        "UNITED KINGDOM": "UK",
        "GREAT BRITAIN": "UK",
        "UK": "UK",
        "GB": "UK",
        "FRANCE": "FRANCE",
        "FR": "FRANCE",
        "GERMANY": "GERMANY",
        "DE": "GERMANY",
        "DEUTSCHLAND": "GERMANY",
        "CANADA": "CANADA",
        "CA": "CANADA",
    }

    return alias_map.get(clean, clean)


def build_normalized_record(
    entity_id: str,
    business_name: str,
    business_address: str,
    country: str
) -> NormalizedRecord:
    """Factory function to build a NormalizedRecord from raw fields."""
    norm_name = normalize_business_name(business_name)
    compact_name = get_business_name_compact(business_name)
    tokens_name = get_business_name_tokens(business_name)
    
    norm_addr = normalize_address(business_address)
    compact_addr = get_address_compact(business_address)
    tokens_addr = get_address_tokens(business_address)
    
    norm_country = normalize_country(country)

    return NormalizedRecord(
        entity_id=entity_id,
        business_name_raw=business_name or "",
        business_name_normalized=norm_name,
        business_name_compact=compact_name,
        business_name_tokens=tokens_name,
        business_address_raw=business_address or "",
        business_address_normalized=norm_addr,
        business_address_compact=compact_addr,
        business_address_tokens=tokens_addr,
        country_raw=country or "",
        country_normalized=norm_country,
    )
