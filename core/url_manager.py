from typing import List, Dict, Set, Optional, Tuple
from urllib.parse import urlparse, urljoin
import re
from datetime import datetime
from collections import defaultdict

from core.data_structures import URLInfo, SearchResult
from core.config import get_config
from core.logging_config import get_logger


class URLManager:
    """Manages URL discovery, ranking, filtering, and diversity"""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.url_history: Dict[str, URLInfo] = {}
        self.domain_counts: Dict[str, int] = defaultdict(int)

    def normalize_url(self, url: str) -> str:
        """Normalize URL for consistency"""
        # Remove fragment
        url = url.split("#")[0]
        # Remove trailing slash
        url = url.rstrip("/")
        # Convert to lowercase for domain
        parsed = urlparse(url)
        normalized = (
            f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path}{parsed.params}"
        )
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized

    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        return parsed.netloc.lower()

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible"""
        try:
            parsed = urlparse(url)
            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False
            # Must be http or https
            if parsed.scheme not in ["http", "https"]:
                return False
            # Check against blocked domains
            domain = self.extract_domain(url)
            if domain in self.config.blocked_domains:
                return False
            # Check allowed domains if specified
            if (
                self.config.allowed_domains
                and domain not in self.config.allowed_domains
            ):
                return False
            # Skip common non-content URLs
            skip_patterns = [
                r"/signin",
                r"/login",
                r"/register",
                r"/logout",
                r"\.pdf$",
                r"\.doc[x]?$",
                r"\.ppt[x]?$",
                r"/search\?",
                r"/tag/",
                r"/category/",
            ]
            for pattern in skip_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return False
            return True
        except Exception as e:
            self.logger.debug("URL_VALIDATION", f"Invalid URL: {url}", error=str(e))
            return False

    def discover_urls_from_search(
        self, search_results: List[SearchResult]
    ) -> List[URLInfo]:
        """Extract and process URLs from search results"""
        discovered_urls = []

        for result in search_results:
            url = self.normalize_url(result.url)

            if not self.is_valid_url(url):
                continue

            # Create URLInfo object
            url_info = URLInfo(
                url=url,
                title=result.title,
                relevance_score=result.relevance_score,
                schema_fields_coverage=[],  # Will be populated later
                metadata={
                    "source": "search",
                    "snippet": result.snippet,
                    "discovered_at": datetime.now(),
                },
            )

            # Update history
            if url not in self.url_history:
                self.url_history[url] = url_info
                discovered_urls.append(url_info)
                self.domain_counts[self.extract_domain(url)] += 1

        self.logger.info(
            "URL_DISCOVERY",
            f"Discovered {len(discovered_urls)} new URLs from search",
            total_results=len(search_results),
        )

        return discovered_urls

    def discover_urls_from_content(self, base_url: str, content: str) -> List[URLInfo]:
        """Extract URLs from webpage content"""
        discovered_urls = []

        # Find all URLs in content using regex
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|href=["\']([^"\']+)["\']'
        matches = re.findall(url_pattern, content)

        for match in matches:
            # Handle href matches
            if match.startswith("href="):
                url = match.split("=", 1)[1].strip("\"'")
            else:
                url = match

            # Convert relative URLs to absolute
            if not url.startswith(("http://", "https://")):
                url = urljoin(base_url, url)

            url = self.normalize_url(url)

            if not self.is_valid_url(url):
                continue

            if url not in self.url_history:
                url_info = URLInfo(
                    url=url,
                    title="",  # Will be filled when visited
                    relevance_score=0.0,  # Will be calculated
                    schema_fields_coverage=[],
                    metadata={
                        "source": "content_extraction",
                        "found_on": base_url,
                        "discovered_at": datetime.now(),
                    },
                )
                self.url_history[url] = url_info
                discovered_urls.append(url_info)
                self.domain_counts[self.extract_domain(url)] += 1

        return discovered_urls

    def rank_urls(
        self,
        urls: List[URLInfo],
        query: str,
        empty_fields: Set[str],
        visited_urls: Set[str],
    ) -> List[URLInfo]:
        """Rank URLs based on relevance and potential value"""
        # Filter out already visited URLs
        unvisited_urls = [u for u in urls if u.url not in visited_urls]

        # Calculate scores for each URL
        scored_urls = []
        for url_info in unvisited_urls:
            score = self._calculate_url_score(url_info, query, empty_fields)
            scored_urls.append((score, url_info))

        # Sort by score (descending)
        scored_urls.sort(key=lambda x: x[0], reverse=True)

        # Apply diversity filtering
        diverse_urls = self._ensure_diversity([u[1] for u in scored_urls])

        self.logger.debug(
            "URL_RANKING",
            f"Ranked {len(diverse_urls)} URLs",
            top_scores=[u[0] for u in scored_urls[:5]],
        )

        return diverse_urls

    def _calculate_url_score(
        self, url_info: URLInfo, query: str, empty_fields: Set[str]
    ) -> float:
        """Calculate relevance score for a URL"""
        score = 0.0

        # Base relevance score from search
        score += url_info.relevance_score * 0.3

        # Title relevance
        if url_info.title:
            query_terms = query.lower().split()
            title_lower = url_info.title.lower()
            matching_terms = sum(1 for term in query_terms if term in title_lower)
            score += (matching_terms / len(query_terms)) * 0.2

        # Schema field coverage potential
        if url_info.schema_fields_coverage:
            coverage_score = len(set(url_info.schema_fields_coverage) & empty_fields)
            score += (coverage_score / len(empty_fields)) * 0.3 if empty_fields else 0

        # Source credibility
        domain = self.extract_domain(url_info.url)
        credible_domains = [
            "wikipedia.org",
            "edu",
            "gov",
            "ieee.org",
            "acm.org",
            "nature.com",
            "science.org",
            "nih.gov",
            "arxiv.org",
        ]
        if any(cred in domain for cred in credible_domains):
            score += 0.1

        # Freshness (if available)
        if "discovered_at" in url_info.metadata:
            score += 0.1  # Bonus for recently discovered URLs

        # Penalty for over-represented domains
        if self.domain_counts[domain] > 3:
            score *= 0.7  # Reduce score for domains we've seen too much

        return min(score, 1.0)  # Cap at 1.0

    def _ensure_diversity(
        self, ranked_urls: List[URLInfo], max_per_domain: int = 2
    ) -> List[URLInfo]:
        """Ensure URL diversity by limiting URLs per domain"""
        domain_counts = defaultdict(int)
        diverse_urls = []

        for url_info in ranked_urls:
            domain = self.extract_domain(url_info.url)
            if domain_counts[domain] < max_per_domain:
                diverse_urls.append(url_info)
                domain_counts[domain] += 1

        return diverse_urls

    def filter_urls(
        self, urls: List[URLInfo], min_relevance: Optional[float] = None
    ) -> List[URLInfo]:
        """Filter URLs based on various criteria"""
        if min_relevance is None:
            min_relevance = self.config.min_relevance_threshold

        filtered = []
        for url_info in urls:
            # Check relevance threshold
            if url_info.relevance_score < min_relevance:
                continue

            # Additional filtering logic can be added here
            filtered.append(url_info)

        return filtered

    def get_url_batch(
        self, ranked_urls: List[URLInfo], batch_size: Optional[int] = None
    ) -> List[URLInfo]:
        """Get a batch of URLs for processing"""
        if batch_size is None:
            batch_size = self.config.max_urls_per_step

        return ranked_urls[:batch_size]

    def update_url_info(self, url: str, **kwargs):
        """Update information about a URL"""
        if url in self.url_history:
            url_info = self.url_history[url]
            for key, value in kwargs.items():
                if hasattr(url_info, key):
                    setattr(url_info, key, value)
                else:
                    url_info.metadata[key] = value

    def get_statistics(self) -> Dict[str, any]:
        """Get URL management statistics"""
        return {
            "total_discovered": len(self.url_history),
            "domains": dict(self.domain_counts),
            "visited": sum(1 for u in self.url_history.values() if u.visit_count > 0),
            "successful": sum(
                1 for u in self.url_history.values() if u.extraction_success
            ),
        }
