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

    def generate_queries(
        self, original_query: str, context: Dict[str, Any]
    ) -> List[str]:
        """Generate multiple search queries based on context"""
        queries = [original_query]

        # Extract entity name if available
        entity_name = self._extract_entity_name(original_query)

        # Add queries for empty fields
        empty_fields = context.get("empty_fields", [])
        for field in empty_fields:  # TODO: Limit to top 3 empty fields
            if entity_name:
                queries.append(f"{entity_name} {field}")
            else:
                queries.append(f"{original_query} {field}")

        # Add alternative phrasings
        queries.extend(self._generate_alternative_queries(original_query))

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_normalized = q.lower().strip()
            if q_normalized not in seen:
                seen.add(q_normalized)
                unique_queries.append(q)

        return unique_queries[: self.config.max_search_queries]

    def _extract_entity_name(self, query: str) -> Optional[str]:
        """Try to extract entity name from query"""
        # Simple heuristic: look for patterns like "John Doe professor"
        parts = query.split()

        # Check if first words are capitalized (likely a name)
        name_parts = []
        for part in parts:
            if part[0].isupper():
                name_parts.append(part)
            else:
                break

        if name_parts:
            return " ".join(name_parts)

        return None

    def _generate_alternative_queries(self, query: str) -> List[str]:
        """Generate alternative phrasings of a query"""
        alternatives = []

        # Add quotes for exact match
        if '"' not in query:
            alternatives.append(f'"{query}"')

        # Add common suffixes/prefixes
        entity_keywords = [
            "biography",
            "profile",
            "information",
            "details",
            "background",
        ]

        for keyword in entity_keywords:  # TODO: Use LLMs to generate queries
            if keyword not in query.lower():
                alternatives.append(f"{query} {keyword}")

        return alternatives

    def rewrite_query_for_field(
        self, base_query: str, field_name: str, field_description: Optional[str] = None
    ) -> str:
        """Rewrite a query to target a specific schema field"""
        # Clean field name (convert from snake_case to readable)
        field_readable = field_name.replace("_", " ")

        # Extract entity name if possible
        entity_name = self._extract_entity_name(base_query)

        if entity_name:
            # More focused query with entity name
            rewritten = f"{entity_name} {field_readable}"

            # Add context from field description if available
            if field_description:
                # Extract key terms from description
                key_terms = self._extract_key_terms(field_description)
                if key_terms:
                    rewritten += f" {' '.join(key_terms[:2])}"
        else:
            # Fallback to appending field to original query
            rewritten = f"{base_query} {field_readable}"

        return rewritten

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Simple extraction of important words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
        }

        words = text.lower().split()
        key_terms = [w for w in words if w not in stop_words and len(w) > 3]

        return key_terms
