"""
Web scraper for wiki pages.

This module provides async web scraping functionality:
- Async HTTP client using aiohttp
- HTML parsing with BeautifulSoup
- Rate limiting to be respectful to servers
- Retry logic with exponential backoff
- Link extraction for crawling

Usage:
    scraper = WikiScraper(base_url="https://eu5.paradoxwikis.com")
    async with scraper:
        pages = await scraper.crawl(start_url="...", max_pages=100)
"""

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup, NavigableString
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ScrapedPage:
    """Represents a successfully scraped wiki page."""

    url: str
    title: str
    content: str  # Cleaned text content
    content_hash: str  # SHA-256 hash for change detection
    word_count: int
    scraped_at: datetime = field(default_factory=datetime.utcnow)
    links: list[str] = field(default_factory=list)  # Internal links found


@dataclass
class ScrapeError:
    """Represents a failed scrape attempt."""

    url: str
    error: str
    status_code: int | None = None


# ============================================================================
# WIKI SCRAPER
# ============================================================================


class WikiScraper:
    """
    Async wiki page scraper with rate limiting and retry logic.

    Why async?
    - Can fetch multiple pages concurrently (faster)
    - Non-blocking I/O (better resource usage)
    - Works well with FastAPI's async nature

    Why rate limiting?
    - Be respectful to wiki servers
    - Avoid getting IP banned
    - Follow robots.txt guidelines
    """

    def __init__(
        self,
        base_url: str,
        delay_seconds: float | None = None,
        max_concurrent: int = 5,
        timeout_seconds: int = 30,
    ):
        """
        Initialize the scraper.

        Args:
            base_url: Base URL of the wiki (e.g., "https://eu5.paradoxwikis.com")
            delay_seconds: Delay between requests (default from config)
            max_concurrent: Max concurrent requests
            timeout_seconds: Request timeout
        """
        self.base_url = base_url.rstrip("/")
        self.delay_seconds = delay_seconds or settings.scraper_delay_seconds
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds

        # Parse base URL for domain checking
        parsed = urlparse(base_url)
        self.domain = parsed.netloc

        # Session will be created in __aenter__
        self._session: aiohttp.ClientSession | None = None

        # Rate limiting semaphore
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Track visited URLs to avoid duplicates
        self._visited: set[str] = set()

    async def __aenter__(self) -> "WikiScraper":
        """Create aiohttp session when entering context."""
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                "User-Agent": settings.scraper_user_agent,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close aiohttp session when exiting context."""
        if self._session:
            await self._session.close()
            self._session = None

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def fetch_page(self, url: str) -> str:
        """
        Fetch a single page's HTML content.

        Uses tenacity for retry with exponential backoff:
        - Attempt 1: immediate
        - Attempt 2: wait 2 seconds
        - Attempt 3: wait 4 seconds

        Raises:
            aiohttp.ClientError: On network errors
            asyncio.TimeoutError: On timeout
        """
        if not self._session:
            raise RuntimeError("Scraper must be used as async context manager")

        async with self._semaphore:  # Limit concurrent requests
            async with self._session.get(url) as response:
                response.raise_for_status()
                return await response.text()

    async def scrape_page(self, url: str) -> ScrapedPage | ScrapeError:
        """
        Scrape a single wiki page.

        Returns:
            ScrapedPage on success, ScrapeError on failure
        """
        try:
            logger.debug("scraping_page", url=url)

            # Fetch HTML
            html = await self.fetch_page(url)

            # Parse and extract content
            soup = BeautifulSoup(html, "html.parser")

            # Extract title
            title = self._extract_title(soup)

            # Extract main content (wiki-specific)
            content = self._extract_content(soup)

            # Extract internal links for crawling
            links = self._extract_links(soup, url)

            # Calculate hash for change detection
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Count words
            word_count = len(content.split())

            logger.info(
                "page_scraped",
                url=url,
                title=title[:50],
                word_count=word_count,
                links_found=len(links),
            )

            return ScrapedPage(
                url=url,
                title=title,
                content=content,
                content_hash=content_hash,
                word_count=word_count,
                links=links,
            )

        except aiohttp.ClientResponseError as e:
            logger.warning("scrape_http_error", url=url, status=e.status)
            return ScrapeError(url=url, error=str(e), status_code=e.status)

        except Exception as e:
            logger.error("scrape_error", url=url, error=str(e))
            return ScrapeError(url=url, error=str(e))

    async def crawl(
        self,
        start_url: str,
        max_pages: int | None = None,
    ) -> AsyncGenerator[ScrapedPage | ScrapeError, None]:
        """
        Crawl wiki starting from a URL, following internal links.

        This is a breadth-first crawl that:
        1. Starts from start_url
        2. Extracts internal links from each page
        3. Adds new links to the queue
        4. Stops when max_pages reached or no more links

        Args:
            start_url: URL to start crawling from
            max_pages: Maximum pages to scrape (default from config)

        Yields:
            ScrapedPage or ScrapeError for each page
        """
        max_pages = max_pages or settings.scraper_max_pages

        # Initialize queue with start URL
        queue: list[str] = [start_url]
        pages_scraped = 0

        logger.info(
            "crawl_started",
            start_url=start_url,
            max_pages=max_pages,
        )

        while queue and pages_scraped < max_pages:
            # Get next URL
            url = queue.pop(0)

            # Skip if already visited
            if url in self._visited:
                continue

            # Mark as visited
            self._visited.add(url)

            # Scrape the page
            result = await self.scrape_page(url)
            pages_scraped += 1

            # Yield result
            yield result

            # If successful, add new links to queue
            if isinstance(result, ScrapedPage):
                for link in result.links:
                    if link not in self._visited and link not in queue:
                        queue.append(link)

            # Rate limiting delay
            if self.delay_seconds > 0:
                await asyncio.sleep(self.delay_seconds)

        logger.info(
            "crawl_completed",
            pages_scraped=pages_scraped,
            urls_in_queue=len(queue),
        )

    # ========================================================================
    # HTML PARSING HELPERS
    # ========================================================================

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from HTML."""
        # Try <h1> first (most wiki pages)
        h1 = soup.find("h1", {"id": "firstHeading"})
        if h1:
            return h1.get_text(strip=True)

        # Fall back to <title> tag
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Remove common suffixes like " - Wiki Name"
            if " - " in title:
                title = title.split(" - ")[0]
            return title

        return "Untitled"

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content from wiki page.

        This method is tailored for MediaWiki-style pages (like Paradox wikis).
        It removes navigation, sidebars, and other non-content elements.
        """
        # Find main content container
        # MediaWiki uses different IDs, try common ones
        content_div = (
            soup.find("div", {"id": "mw-content-text"})
            or soup.find("div", {"id": "bodyContent"})
            or soup.find("div", {"class": "mw-parser-output"})
            or soup.find("main")
            or soup.find("article")
        )

        if not content_div:
            # Fall back to body
            content_div = soup.find("body")

        if not content_div:
            return ""

        # Remove unwanted elements
        for selector in [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            ".navbox",  # Navigation boxes
            ".infobox",  # Info boxes (keep? maybe include later)
            ".toc",  # Table of contents
            ".mw-editsection",  # Edit links
            ".reference",  # Reference numbers
            ".noprint",  # Non-printable elements
            "#siteSub",  # Site subtitle
            "#contentSub",  # Content subtitle
            ".mw-indicators",  # Page indicators
            ".catlinks",  # Category links
        ]:
            for element in content_div.select(selector):
                element.decompose()

        # Extract text with structure
        text_parts = []

        for element in content_div.descendants:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    text_parts.append(text)
            elif element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                # Add headers with markers for chunking later
                header_text = element.get_text(strip=True)
                if header_text:
                    text_parts.append(f"\n\n## {header_text}\n")
            elif element.name == "p":
                # Paragraphs get newlines
                text_parts.append("\n")
            elif element.name in ["li"]:
                # List items
                text_parts.append("\n• ")
            elif element.name == "br":
                text_parts.append("\n")

        # Join and clean up
        text = " ".join(text_parts)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 newlines
        text = re.sub(r" {2,}", " ", text)  # Max 1 space
        text = re.sub(r"^\s+", "", text, flags=re.MULTILINE)  # Leading whitespace

        # Remove common junk patterns
        junk_patterns = [
            r"OptanonConsentNoticeStart.*?OptanonConsentNoticeEnd\s*",
            r"OneTrustConsentNotice.*?OneTrustConsentNoticeEnd\s*",
        ]
        for pattern in junk_patterns:
            text = re.sub(pattern, "", text, flags=re.DOTALL)

        return text.strip()

    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> list[str]:
        """
        Extract internal wiki links from page.

        Only returns links that:
        - Are on the same domain
        - Are wiki content pages (not special pages, talk pages, etc.)
        - Haven't been visited yet
        """
        links = []

        # Find main content area for links
        content = (
            soup.find("div", {"id": "mw-content-text"})
            or soup.find("div", {"id": "bodyContent"})
            or soup
        )

        for a_tag in content.find_all("a", href=True):
            href = a_tag["href"]

            # Skip anchors
            if href.startswith("#"):
                continue

            # Convert to absolute URL
            full_url = urljoin(current_url, href)
            parsed = urlparse(full_url)

            # Only same domain
            if parsed.netloc != self.domain:
                continue

            # Skip non-content pages (MediaWiki specific)
            path = parsed.path.lower()
            skip_patterns = [
                "/special:",
                "/talk:",
                "/user:",
                "/user_talk:",
                "/file:",
                "/mediawiki:",
                "/template:",
                "/help:",
                "/category:",
                "action=",
                "oldid=",
                "/index.php",
            ]
            if any(pattern in path or pattern in parsed.query.lower() for pattern in skip_patterns):
                continue

            # Remove query string and fragment
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            # Add if not already in list
            if clean_url not in links and clean_url != current_url:
                links.append(clean_url)

        return links

    def reset(self) -> None:
        """Reset visited URLs for a fresh crawl."""
        self._visited.clear()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def scrape_single_page(url: str) -> ScrapedPage | ScrapeError:
    """
    Convenience function to scrape a single page.

    Usage:
        result = await scrape_single_page("https://eu5.paradoxwikis.com/Trade")
    """
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    async with WikiScraper(base_url=base_url) as scraper:
        return await scraper.scrape_page(url)
