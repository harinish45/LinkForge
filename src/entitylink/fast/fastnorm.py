"""Lean normalization + hashing primitives for the high-throughput path.

Semantics mirror ``entitylink.preprocessing.normalizer`` (same legal-suffix,
abbreviation and stopword handling) but are written for speed: ASCII fast path,
no repeated re-normalization, deterministic CRC32 hashing (stable across runs).
"""

import re
import string
import unicodedata
import zlib
from typing import List, Optional, Tuple

LEGAL_SUFFIXES = {
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt",
    "company": "co", "co": "co", "llc": "llc", "plc": "plc",
    "gmbh": "gmbh", "sarl": "sarl", "sa": "sa",
}
LEGAL_SUFFIX_SET = set(LEGAL_SUFFIXES) | set(LEGAL_SUFFIXES.values())

BUSINESS_ABBREVIATIONS = {
    "international": "intl", "intl": "intl", "solutions": "sol", "sol": "sol",
    "technology": "tech", "tech": "tech", "technologies": "tech",
    "services": "svc", "svc": "svc", "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "management": "mgmt", "mgmt": "mgmt", "development": "dev", "dev": "dev",
}

ADDRESS_ABBREVIATIONS = {
    "street": "st", "st": "st", "road": "rd", "rd": "rd", "avenue": "ave", "ave": "ave",
    "boulevard": "blvd", "blvd": "blvd", "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "apartment": "apt", "apt": "apt", "suite": "ste", "ste": "ste",
    "building": "bldg", "bldg": "bldg", "floor": "fl", "fl": "fl", "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy", "square": "sq", "sq": "sq", "industrial": "ind", "ind": "ind",
    "estate": "est", "est": "est",
}

GENERIC_STOPWORDS = {"and", "&", "the", "of", "in", "for", "on", "at", "to", "a", "an"}

_TRANS = str.maketrans({c: " " for c in string.punctuation})
_NONWORD = re.compile(r"[^\w\s]")


def crc32(text: str) -> int:
    """Deterministic 32-bit hash of a string (stable across processes/runs)."""
    return zlib.crc32(text.encode("utf-8", "replace")) & 0xFFFFFFFF


def norm_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (ASCII fast path)."""
    if not text:
        return ""
    if text.isascii():
        return " ".join(text.lower().translate(_TRANS).split())
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    return " ".join(_NONWORD.sub(" ", t).split())


def normalize_name(name: str) -> Tuple[str, List[str]]:
    """Return (canonical suffix-stripped name, informative tokens)."""
    tokens = norm_text(name).split()
    if not tokens:
        return "", []
    canonical = []
    informative = []
    for tok in tokens:
        if tok in GENERIC_STOPWORDS:
            continue
        canon = LEGAL_SUFFIXES.get(tok, BUSINESS_ABBREVIATIONS.get(tok, tok))
        canonical.append(canon)
        if len(canon) >= 2:
            informative.append(canon)
    while len(canonical) > 1 and canonical[-1] in LEGAL_SUFFIX_SET:
        canonical.pop()
    return " ".join(canonical), informative


def normalize_addr(address: str) -> Tuple[str, List[str], List[str]]:
    """Return (canonical address, tokens, numeric tokens)."""
    toks = norm_text(address).split()
    if not toks:
        return "", [], []
    canon = [ADDRESS_ABBREVIATIONS.get(t, t) for t in toks if t not in GENERIC_STOPWORDS]
    nums = [t for t in canon if t.isdigit()]
    return " ".join(canon), canon, nums


def normalize_country(country: str) -> str:
    """Open-set country normalization matching the main pipeline's aliases."""
    if not country:
        return ""
    clean = norm_text(country).upper()
    if not clean:
        return ""
    aliases = {
        "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US", "USA": "US",
        "U S A": "US", "U S": "US", "INDIA": "INDIA", "IND": "INDIA", "BHARAT": "INDIA",
        "UNITED KINGDOM": "UK", "GREAT BRITAIN": "UK", "GB": "UK",
        "FRANCE": "FRANCE", "FR": "FRANCE", "GERMANY": "GERMANY", "DE": "GERMANY",
        "DEUTSCHLAND": "GERMANY", "CANADA": "CANADA", "CA": "CANADA",
    }


def _bitset_256(hashes) -> Tuple[int, int, int, int]:
    """Fold 8-bit token/gram hashes into a 256-bit set (4 x uint64 lanes)."""
    b0 = b1 = b2 = b3 = 0
    for h in hashes:
        bit = h & 255
        word = bit >> 6
        mask = 1 << (bit & 63)
        if word == 0:
            b0 |= mask
        elif word == 1:
            b1 |= mask
        elif word == 2:
            b2 |= mask
        else:
            b3 |= mask
    return b0, b1, b2, b3


KEY_CLASSES = {
    "name_exact": 0, "name_tokbag": 1, "postal": 2, "postal_namepfx": 3,
    "addr_exact": 4, "addrnum_namepfx": 5, "country_addrnum": 6, "longest_tok": 7,
    "street_tok": 8,
}
CLASS_SHIFT = 28          # class tag occupies the top 4 bits of the 32-bit key
CLASS_MASK = 0xF
_HASH_MASK = (1 << CLASS_SHIFT) - 1


def _mk(key_class: int, text: str) -> int:
    """Class-tagged 28-bit hash key (class in the top 4 bits)."""
    return (key_class << CLASS_SHIFT) | (crc32(text) & _HASH_MASK)


def blocking_keys(
    norm_name: str,
    name_toks: List[str],
    norm_addr: str,
    addr_nums: List[str],
    postal: int,
    norm_country: str,
    addr_toks: Optional[List[str]] = None,
) -> List[int]:
    """High-precision class-tagged blocking keys (symmetric for S1 and targets)."""
    keys: List[int] = []
    if norm_name:
        keys.append(_mk(KEY_CLASSES["name_exact"], "n#" + norm_name))
    if name_toks:
        keys.append(_mk(KEY_CLASSES["name_tokbag"], "t#" + " ".join(sorted(set(name_toks)))))
    if postal:
        keys.append(_mk(KEY_CLASSES["postal"], "z#" + str(postal)))
        if name_toks:
            keys.append(_mk(KEY_CLASSES["postal_namepfx"], f"zn#{postal}#{name_toks[0][:4]}"))
    if norm_addr:
        keys.append(_mk(KEY_CLASSES["addr_exact"], "a#" + norm_addr))
    if addr_nums:
        if name_toks:
            keys.append(_mk(KEY_CLASSES["addrnum_namepfx"], f"an#{addr_nums[0]}#{name_toks[0][:3]}"))
        if norm_country:
            keys.append(_mk(KEY_CLASSES["country_addrnum"], f"ac#{norm_country}#{addr_nums[0]}"))
    if name_toks:
        longest_two = sorted(set(name_toks), key=len, reverse=True)[:2]
        for tok in longest_two:
            if len(tok) >= 4:
                keys.append(_mk(KEY_CLASSES["longest_tok"], "w#" + tok))
    if addr_toks:
        for tok in addr_toks:
            if len(tok) >= 5 and not tok.isdigit():
                keys.append(_mk(KEY_CLASSES["street_tok"], "s#" + tok))
                break
    return keys



def encode_record(name: str, address: str, country: str):
    """Encode one record into (payload_tuple, blocking_keys).

    payload_tuple order matches ``payload.PAYLOAD_FIELDS``:
      name_hash, addr_hash, postal, country, name_len, addr_len,
      n_name_tok, n_addr_tok, n_addr_num,
      name_tok_bits[4], name_char_bits[4], addr_tok_bits[4], addr_num_bits[4]
    """
    norm_name, name_toks = normalize_name(name)
    norm_addr, addr_toks, addr_nums = normalize_addr(address)
    norm_country = normalize_country(country)

    name_hash = crc32(norm_name) if norm_name else 0
    addr_hash = crc32(norm_addr) if norm_addr else 0
    postal = 0
    for tok in addr_toks:
        if len(tok) in (5, 6) and tok.isdigit():
            postal = int(tok)
            break
    country_hash = crc32(norm_country) if norm_country else 0

    compact = norm_name.replace(" ", "")
    if len(compact) >= 3:
        ngrams = {compact[i:i + 3] for i in range(len(compact) - 2)}
    else:
        ngrams = {compact} if compact else set()

    ntb = _bitset_256(crc32(t) & 255 for t in set(name_toks))
    ncb = _bitset_256(crc32(g) & 255 for g in ngrams)
    atb = _bitset_256(crc32(t) & 255 for t in set(addr_toks))
    anb = _bitset_256(crc32(t) & 255 for t in set(addr_nums))

    payload = (
        name_hash, addr_hash, postal, country_hash,
        len(norm_name), len(norm_addr),
        len(set(name_toks)), len(set(addr_toks)), len(set(addr_nums)),
        ntb[0], ntb[1], ntb[2], ntb[3],
        ncb[0], ncb[1], ncb[2], ncb[3],
        atb[0], atb[1], atb[2], atb[3],
        anb[0], anb[1], anb[2], anb[3],
    )
    keys = blocking_keys(norm_name, name_toks, norm_addr, addr_nums, postal,
                         norm_country, addr_toks)
    return payload, keys

