"""Text normalization utilities for business names and addresses."""

import re
from typing import Dict, List, Set

# Standard legal suffix canonicalization mapping
LEGAL_SUFFIXES: Dict[str, str] = {
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
    "llc": "llc",
    "plc": "plc",
    "gmbh": "gmbh",
    "sarl": "sarl",
    "sa": "sa",
}

# Standard address abbreviation canonicalization mapping
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "apartment": "apt",
    "suite": "ste",
    "building": "bldg",
    "floor": "fl",
    "highway": "hwy",
}


def normalize_text(text: str) -> str:
    """Basic lowercasing, punctuation stripping, and whitespace collapse."""
    if not text:
        return ""
    # Lowercase
    t = text.lower()
    # Replace non-alphanumeric (except whitespace) with space
    t = re.sub(r"[^\w\s]", " ", t)
    # Collapse multiple whitespaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


LEGAL_SUFFIX_SET: Set[str] = {
    "corp", "corporation", "inc", "incorporated", "ltd", "limited",
    "pvt", "private", "llc", "co", "company", "plc", "gmbh", "sarl", "sa"
}


def normalize_business_name(name: str, strip_legal_suffixes: bool = True) -> str:
    """Normalize a business name by standardizing and stripping legal suffixes."""
    tokens = normalize_text(name).split()
    if not tokens:
        return ""
    
    cleaned_tokens: List[str] = []
    for token in tokens:
        # Canonicalize legal suffixes
        canon = LEGAL_SUFFIXES.get(token, token)
        cleaned_tokens.append(canon)
        
    if strip_legal_suffixes:
        while len(cleaned_tokens) > 1 and cleaned_tokens[-1] in LEGAL_SUFFIX_SET:
            cleaned_tokens.pop()

    return " ".join(cleaned_tokens)


def normalize_address(address: str) -> str:
    """Normalize a business address by standardizing street/unit keywords."""
    tokens = normalize_text(address).split()
    if not tokens:
        return ""
    
    cleaned_tokens: List[str] = []
    for token in tokens:
        canon = ADDRESS_ABBREVIATIONS.get(token, token)
        cleaned_tokens.append(canon)
        
    return " ".join(cleaned_tokens)


def normalize_country(country: str) -> str:
    """Normalize country code or name in an open-set manner."""
    if not country:
        return "UNKNOWN"
    c = country.strip().upper()
    # Common variations
    if c in ("UNITED STATES", "USA", "U.S.A.", "U.S."):
        return "US"
    if c in ("IND", "BHARAT"):
        return "INDIA"
    if c in ("FR", "REPUBLIQUE FRANCAISE"):
        return "FRANCE"
    return c
