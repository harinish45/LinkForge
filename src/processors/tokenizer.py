"""URL tokenization and decomposition module."""

import ipaddress
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class TokenizedUrl:
    """Immutable representation of tokenized URL components."""

    raw_url: str
    normalized_url: str
    scheme: str
    netloc: str
    domain: str
    subdomains: List[str]
    tld: str
    port: Optional[int]
    path: str
    path_tokens: List[str]
    query: str
    query_tokens: List[str]
    fragment: str
    is_ip: bool
    all_tokens: List[str] = field(default_factory=list)


class UrlTokenizer:
    """Robust URL tokenizer for lexical and structural decomposition."""

    DELIMITER_PATTERN: re.Pattern[str] = re.compile(r"[\s/\\.,;:\-_?=&%+#@!~*|`^$()]+|(?=[A-Z])")
    KNOWN_MULTI_TLDS: set[str] = {
        "co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "net.au",
        "org.au", "co.nz", "co.jp", "ne.jp", "co.in", "net.in",
        "org.in", "gov.in", "com.br", "org.br", "co.za", "com.sg",
    }
    SCHEME_PREFIX_RE: re.Pattern[str] = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-.]*:")

    def __init__(self) -> None:
        pass

    def normalize(self, raw_url: str) -> str:
        """Strip whitespace and enforce basic scheme if absent."""
        trimmed: str = raw_url.strip()
        if not trimmed:
            return ""
        if not self.SCHEME_PREFIX_RE.match(trimmed):
            return f"http://{trimmed}"
        return trimmed

    def is_ip_address(self, host: str) -> bool:
        """Check whether host is an IPv4 or IPv6 address."""
        if not host:
            return False
        clean_host: str = host.strip("[]")
        # Strip port only if single colon (IPv4:port) and not IPv6
        if ":" in clean_host and clean_host.count(":") == 1:
            clean_host = clean_host.split(":")[0]
        try:
            ipaddress.ip_address(clean_host)
            return True
        except ValueError:
            return False

    def _split_domain_parts(self, host: str) -> tuple[str, List[str], str]:
        """Split host into registered domain, subdomains list, and TLD."""
        clean_host: str = host.strip("[]")
        if ":" in clean_host and clean_host.count(":") == 1:
            clean_host = clean_host.split(":")[0].lower()
        else:
            clean_host = clean_host.lower()

        if not clean_host or self.is_ip_address(clean_host):
            return clean_host, [], ""

        labels: List[str] = clean_host.split(".")
        if len(labels) == 1:
            return labels[0], [], ""

        for multi_tld in self.KNOWN_MULTI_TLDS:
            tld_parts: List[str] = multi_tld.split(".")
            if len(labels) > len(tld_parts) and labels[-len(tld_parts):] == tld_parts:
                tld: str = multi_tld
                domain: str = labels[-len(tld_parts) - 1]
                subdomains: List[str] = labels[:-len(tld_parts) - 1]
                return domain, subdomains, tld

        tld = labels[-1]
        domain = labels[-2]
        subdomains = labels[:-2]
        return domain, subdomains, tld

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize arbitrary string segment by common delimiters."""
        if not text:
            return []
        raw_tokens: List[str] = self.DELIMITER_PATTERN.split(text)
        return [t.lower() for t in raw_tokens if t and len(t) > 1]

    def _safe_get_port(self, parsed: urllib.parse.SplitResult) -> Optional[int]:
        """Safely extract port without raising on malformed host info."""
        try:
            return parsed.port
        except (ValueError, TypeError):
            return None

    def tokenize(self, url: str) -> TokenizedUrl:
        """Parse and tokenize a given URL string."""
        normalized: str = self.normalize(url)
        parsed = urllib.parse.urlsplit(normalized)

        raw_host: str = parsed.hostname or parsed.netloc or ""
        is_ip: bool = self.is_ip_address(raw_host)
        domain, subdomains, tld = self._split_domain_parts(raw_host)

        path_tokens: List[str] = self._tokenize_text(parsed.path)
        query_tokens: List[str] = self._tokenize_text(parsed.query)

        host_tokens: List[str] = [domain] + subdomains if domain else []
        combined_tokens: List[str] = host_tokens + path_tokens + query_tokens

        return TokenizedUrl(
            raw_url=url,
            normalized_url=normalized,
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc,
            domain=domain,
            subdomains=subdomains,
            tld=tld,
            port=self._safe_get_port(parsed),
            path=parsed.path,
            path_tokens=path_tokens,
            query=parsed.query,
            query_tokens=query_tokens,
            fragment=parsed.fragment,
            is_ip=is_ip,
            all_tokens=combined_tokens,
        )
