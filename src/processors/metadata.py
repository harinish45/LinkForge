"""Link metadata enrichment processor for preview fetching."""

import logging
import time
import urllib.parse
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UrlMetadata:
    """Preview metadata and HTTP delivery details for a URL."""

    url: str
    final_url: str
    status_code: Optional[int]
    title: Optional[str]
    description: Optional[str]
    favicon_url: Optional[str]
    content_type: Optional[str]
    response_time_ms: float
    redirect_chain: List[str]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)


class LinkMetadataEnricher:
    """Enriches URLs with live HTTP preview metadata."""

    DEFAULT_USER_AGENT: str = "LinkForge-Analyzer/1.0 (+https://github.com/harinish45/LinkForge)"
    DISALLOWED_SCHEMES: set[str] = {"javascript", "data", "file", "ftp", "mailto"}

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout: float = timeout

    def _resolve_url(self, base_url: str, relative_path: str) -> str:
        """Resolve a relative URL against base URL."""
        if not relative_path:
            return ""
        return urllib.parse.urljoin(base_url, relative_path)

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract title from open graph tag or title element."""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return str(og_title.get("content")).strip()

        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        return None

    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract description from standard or og meta tags."""
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            return str(og_desc.get("content")).strip()

        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return str(meta_desc.get("content")).strip()
        return None

    def _extract_favicon(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """Extract favicon link or construct default favicon url."""
        icon_tag = soup.find("link", rel=lambda r: bool(r and "icon" in r.lower()))
        if icon_tag and icon_tag.get("href"):
            return self._resolve_url(base_url, str(icon_tag.get("href")))

        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}/favicon.ico"
        return None

    def _parse_html_body(self, html_text: str, current_url: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse HTML string and extract metadata elements."""
        soup = BeautifulSoup(html_text, "html.parser")
        title: Optional[str] = self._extract_title(soup)
        desc: Optional[str] = self._extract_description(soup)
        favicon: Optional[str] = self._extract_favicon(soup, current_url)
        return title, desc, favicon

    def _validate_scheme(self, url: str) -> Optional[str]:
        """Validate URL scheme and return error message if invalid."""
        parsed = urllib.parse.urlsplit(url)
        scheme: str = parsed.scheme.lower()
        if not scheme:
            return None
        if scheme in self.DISALLOWED_SCHEMES:
            return f"Unsupported scheme: {scheme}"
        if scheme not in {"http", "https"}:
            return f"Non-HTTP scheme: {scheme}"
        return None

    async def fetch_async(self, url: str, client: Optional[httpx.AsyncClient] = None) -> UrlMetadata:
        """Fetch metadata asynchronously via httpx."""
        scheme_err: Optional[str] = self._validate_scheme(url)
        if scheme_err:
            return UrlMetadata(
                url=url,
                final_url=url,
                status_code=None,
                title=None,
                description=None,
                favicon_url=None,
                content_type=None,
                response_time_ms=0.0,
                redirect_chain=[],
                error=scheme_err,
            )

        headers = {"User-Agent": self.DEFAULT_USER_AGENT}
        start_time: float = time.perf_counter()

        should_close_client = client is None
        async_client = client or httpx.AsyncClient(timeout=self.timeout, follow_redirects=True, verify=False)

        try:
            target_url = url if "://" in url else f"http://{url}"
            response = await async_client.get(target_url, headers=headers)
            elapsed_ms: float = round((time.perf_counter() - start_time) * 1000, 2)

            redirects: List[str] = [str(r.url) for r in response.history]
            content_type: str = response.headers.get("content-type", "")
            title, desc, favicon = (None, None, None)

            if "text/html" in content_type:
                title, desc, favicon = self._parse_html_body(response.text, str(response.url))

            return UrlMetadata(
                url=url,
                final_url=str(response.url),
                status_code=response.status_code,
                title=title,
                description=desc,
                favicon_url=favicon,
                content_type=content_type,
                response_time_ms=elapsed_ms,
                redirect_chain=redirects,
                error=None,
            )
        except httpx.TimeoutException as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(f"Timeout fetching metadata for {url}: {exc}")
            return UrlMetadata(
                url=url,
                final_url=url,
                status_code=None,
                title=None,
                description=None,
                favicon_url=None,
                content_type=None,
                response_time_ms=elapsed_ms,
                redirect_chain=[],
                error=f"Request timeout after {self.timeout}s",
            )
        except httpx.RequestError as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(f"Network error fetching metadata for {url}: {exc}")
            return UrlMetadata(
                url=url,
                final_url=url,
                status_code=None,
                title=None,
                description=None,
                favicon_url=None,
                content_type=None,
                response_time_ms=elapsed_ms,
                redirect_chain=[],
                error=f"Connection error: {type(exc).__name__}",
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"Unexpected error fetching metadata for {url}: {exc}")
            return UrlMetadata(
                url=url,
                final_url=url,
                status_code=None,
                title=None,
                description=None,
                favicon_url=None,
                content_type=None,
                response_time_ms=elapsed_ms,
                redirect_chain=[],
                error=str(exc),
            )
        finally:
            if should_close_client:
                await async_client.aclose()

    def fetch(self, url: str) -> UrlMetadata:
        """Synchronously fetch preview metadata for a URL."""
        scheme_err: Optional[str] = self._validate_scheme(url)
        if scheme_err:
            return UrlMetadata(
                url=url,
                final_url=url,
                status_code=None,
                title=None,
                description=None,
                favicon_url=None,
                content_type=None,
                response_time_ms=0.0,
                redirect_chain=[],
                error=scheme_err,
            )

        headers = {"User-Agent": self.DEFAULT_USER_AGENT}
        start_time: float = time.perf_counter()
        target_url = url if "://" in url else f"http://{url}"

        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True, verify=False) as client:
                response = client.get(target_url, headers=headers)
                elapsed_ms: float = round((time.perf_counter() - start_time) * 1000, 2)
                redirects: List[str] = [str(r.url) for r in response.history]
                content_type: str = response.headers.get("content-type", "")
                title, desc, favicon = (None, None, None)

                if "text/html" in content_type:
                    title, desc, favicon = self._parse_html_body(response.text, str(response.url))

                return UrlMetadata(
                    url=url,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    title=title,
                    description=desc,
                    favicon_url=favicon,
                    content_type=content_type,
                    response_time_ms=elapsed_ms,
                    redirect_chain=redirects,
                    error=None,
                )
        except httpx.TimeoutException:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return UrlMetadata(
                url=url,
                final_url=url,
                status_code=None,
                title=None,
                description=None,
                favicon_url=None,
                content_type=None,
                response_time_ms=elapsed_ms,
                redirect_chain=[],
                error=f"Request timeout after {self.timeout}s",
            )
        except httpx.RequestError as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return UrlMetadata(
                url=url,
                final_url=url,
                status_code=None,
                title=None,
                description=None,
                favicon_url=None,
                content_type=None,
                response_time_ms=elapsed_ms,
                redirect_chain=[],
                error=f"Connection error: {type(exc).__name__}",
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return UrlMetadata(
                url=url,
                final_url=url,
                status_code=None,
                title=None,
                description=None,
                favicon_url=None,
                content_type=None,
                response_time_ms=elapsed_ms,
                redirect_chain=[],
                error=str(exc),
            )
