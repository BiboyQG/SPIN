from typing import List, Dict, Optional, Any
from datetime import datetime
import requests
import time
import json
import os

from core.logging_config import get_logger, SearchError
from core.data_structures import SearchResult
from core.config import get_config


class SearchEngine:
    """Handles search operations with various providers"""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.last_search_time = 0
        self.search_count = 0

    def search(
        self, query: str, num_results: Optional[int] = None
    ) -> List[SearchResult]:
        """Execute a search query and return results"""
        if not query or not query.strip():
            raise SearchError("Empty search query")

        # Rate limiting
        self._enforce_rate_limit()

        # Get search results based on provider
        provider = self.config.search_config.provider

        try:
            if provider == "brave":
                results = self._brave_search(query, num_results)
            elif provider == "serpapi":
                results = self._serpapi_search(query, num_results)
            else:
                raise SearchError(f"Unsupported search provider: {provider}")

            self.search_count += 1
            self.logger.info(
                "SEARCH_EXECUTED",
                f"Searched for: {query}",
                provider=provider,
                results_count=len(results),
            )

            return results

        except Exception as e:
            self.logger.error(
                "SEARCH_FAILED", f"Search failed for query: {query}", error=str(e)
            )
            raise SearchError(f"Search failed: {str(e)}", "search", {"query": query})

    def _enforce_rate_limit(self):
        """Enforce rate limiting between searches"""
        time_since_last = time.time() - self.last_search_time
        if time_since_last < 3.0:  # Minimum 3 seconds between searches
            time.sleep(3.0 - time_since_last)
        self.last_search_time = time.time()

    def _brave_search(
        self, query: str, num_results: Optional[int] = None
    ) -> List[SearchResult]:
        """Execute search using Brave Search API"""
        api_key = self.config.search_config.api_key
        if not api_key:
            raise SearchError("Brave Search API key not configured")

        base_url = os.getenv(
            "BRAVE_SEARCH_URL", "https://api.search.brave.com/res/v1/web/search"
        )

        headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}

        params = {
            "q": query,
            "count": num_results or self.config.search_config.max_results_per_query,
        }

        if self.config.search_config.safe_search:
            params["safesearch"] = "strict"

        try:
            response = requests.get(
                base_url,
                headers=headers,
                params=params,
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()

            data = response.json()
            results = []

            # Parse web results
            web_results = data.get("web", {}).get("results", [])

            for idx, result in enumerate(web_results):
                search_result = SearchResult(
                    url=result.get("url", ""),
                    title=result.get("title", ""),
                    snippet=result.get("description", ""),
                    timestamp=datetime.now(),
                    metadata={
                        "age": result.get("age"),
                        "language": result.get("language", "en"),
                        "family_friendly": result.get("family_friendly", True),
                    },
                )
                results.append(search_result)

            return results

        except requests.RequestException as e:
            raise SearchError(f"Brave Search API request failed: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise SearchError(f"Failed to parse Brave Search response: {str(e)}")

    def _serpapi_search(
        self, query: str, num_results: Optional[int] = None
    ) -> List[SearchResult]:
        """Execute search using SerpApi (Google Search)"""
        api_key = self.config.search_config.api_key
        if not api_key:
            raise SearchError("SerpApi API key not configured")

        base_url = os.getenv("SERPAPI_URL", "https://serpapi.com/search")

        params = {
            "api_key": api_key,
            "q": query,
            "num": num_results or self.config.search_config.max_results_per_query,
            "engine": "google",
        }

        if self.config.search_config.safe_search:
            params["safe"] = "active"

        try:
            response = requests.get(
                base_url,
                params=params,
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()

            data = response.json()
            results = []

            # Check for errors in SerpApi response
            if "error" in data:
                raise SearchError(f"SerpApi error: {data['error']}")

            # Parse organic results
            organic_results = data.get("organic_results", [])

            for idx, result in enumerate(organic_results):
                search_result = SearchResult(
                    url=result.get("link", ""),
                    title=result.get("title", ""),
                    snippet=result.get("snippet", ""),
                    timestamp=datetime.now(),
                    metadata={
                        "position": result.get("position"),
                        "displayed_link": result.get("displayed_link"),
                        "cached_page_link": result.get("cached_page_link"),
                        "related_pages_link": result.get("related_pages_link"),
                        "search_engine": "google",
                    },
                )
                results.append(search_result)

            return results

        except requests.RequestException as e:
            raise SearchError(f"SerpApi request failed: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise SearchError(f"Failed to parse SerpApi response: {str(e)}")
