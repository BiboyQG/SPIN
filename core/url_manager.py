from typing import List, Dict, Set, Optional, Tuple
from urllib.parse import urlparse, urljoin
import re
from datetime import datetime
from collections import defaultdict
from openai import OpenAI
import json

from core.data_structures import URLInfo, SearchResult
from core.config import get_config
from core.logging_config import get_logger
from core.response_model import ResponseOfCredibility, ResponseOfSelection


class URLManager:
    """Manages URL discovery, filtering, and LLM-based credibility assessment"""

    def __init__(self, llm_client: OpenAI):
        self.config = get_config()
        self.logger = get_logger()
        self.llm_client = llm_client

        # Enhanced URL storage with text labels
        self.url_registry: Dict[str, Dict[str, any]] = {}
        # url_registry structure: {
        #     "url": {
        #         "info": URLInfo object,
        #         "link_text": "text that was clickable",
        #         "credibility": "high|medium|low",
        #         "credibility_reason": "why this rating",
        #         "domain": "example.com"
        #     }
        # }

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
                schema_fields_coverage=[],
                metadata={
                    "source": "search",
                    "snippet": result.snippet,
                    "discovered_at": datetime.now(),
                },
            )

            # Store in registry with additional metadata
            if url not in self.url_registry:
                self.url_registry[url] = {
                    "info": url_info,
                    "link_text": result.title,  # For search results, use title as link text
                    "domain": self.extract_domain(url),
                    "credibility": None,  # Will be assessed later
                    "credibility_reason": None,
                }
                discovered_urls.append(url_info)
                self.domain_counts[self.extract_domain(url)] += 1

        self.logger.info(
            "URL_DISCOVERY",
            f"Discovered {len(discovered_urls)} new URLs from search",
            total_results=len(search_results),
        )

        return discovered_urls

    def discover_urls_from_content(self, base_url: str, content: str) -> List[URLInfo]:
        """Extract URLs from webpage content with their link text"""
        discovered_urls = []

        # Enhanced regex to capture link text
        # Match [link text](url) patterns from Markdown
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(link_pattern, content, re.IGNORECASE)

        for link_text, url in matches:
            # Clean up link text
            link_text = re.sub(r"\s+", " ", link_text.strip())

            # Convert relative URLs to absolute
            if not url.startswith(("http://", "https://")):
                url = urljoin(base_url, url)

            url = self.normalize_url(url)

            if not self.is_valid_url(url):
                continue

            url = url.rstrip(")")

            if url not in self.url_registry:
                url_info = URLInfo(
                    url=url,
                    title=link_text,  # Use link text as title
                    relevance_score=0.0,
                    schema_fields_coverage=[],
                    metadata={
                        "source": "content_extraction",
                        "found_on": base_url,
                        "discovered_at": datetime.now(),
                    },
                )

                self.url_registry[url] = {
                    "info": url_info,
                    "link_text": link_text,
                    "domain": self.extract_domain(url),
                    "credibility": None,
                    "credibility_reason": None,
                }
                discovered_urls.append(url_info)
                self.domain_counts[self.extract_domain(url)] += 1

        # Also find plain URLs without link text
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        plain_urls = re.findall(url_pattern, content)

        for plain_url in plain_urls:
            url = self.normalize_url(plain_url)

            if not self.is_valid_url(url) or url in self.url_registry:
                continue

            url_info = URLInfo(
                url=url,
                title="",
                relevance_score=0.0,
                schema_fields_coverage=[],
                metadata={
                    "source": "content_extraction",
                    "found_on": base_url,
                    "discovered_at": datetime.now(),
                },
            )

            self.url_registry[url] = {
                "info": url_info,
                "link_text": "",  # No link text for plain URLs
                "domain": self.extract_domain(url),
                "credibility": None,
                "credibility_reason": None,
            }
            discovered_urls.append(url_info)
            self.domain_counts[self.extract_domain(url)] += 1

        return discovered_urls

    def assess_url_credibility(
        self, urls: List[str], research_context: str
    ) -> Dict[str, Tuple[str, str]]:
        """Use LLM to assess credibility of URLs"""
        if not urls:
            return {}

        # Prepare URL information for assessment
        url_details = []
        for url in urls[:20]:  # Limit to 20 URLs per assessment
            if url in self.url_registry:
                entry = self.url_registry[url]
                url_details.append(
                    {
                        "url": url,
                        "domain": entry["domain"],
                        "link_text": entry["link_text"],
                        "title": entry["info"].title,
                    }
                )

        prompt = f"""Assess the credibility of these URLs for research on: {research_context}

URLs to assess:
{json.dumps(url_details, indent=2)}

For each URL, rate its credibility as "high", "medium", or "low" based on:
- Domain reputation (edu, gov, established organizations or entity's website vs unknown sites)
- URL structure (official pages vs user-generated content)
- Link text relevance to the research topic
- Likelihood of containing authoritative information

Respond in JSON format:
{{
    "assessments": [
        {{
            "url": "url here",
            "credibility": "high|medium|low",
            "reason": "brief explanation"
        }}
    ]
}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at assessing source credibility.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm_config.temperature,
                max_tokens=self.config.llm_config.max_tokens,
                extra_body={"guided_json": ResponseOfCredibility.model_json_schema()},
            )

            result = json.loads(response.choices[0].message.content)

            # Update registry with assessments
            credibility_results = {}
            for assessment in result.get("assessments", []):
                url = assessment["url"]
                credibility = assessment["credibility"]
                reason = assessment["reason"]

                if url in self.url_registry:
                    self.url_registry[url]["credibility"] = credibility
                    self.url_registry[url]["credibility_reason"] = reason
                    credibility_results[url] = (credibility, reason)

            return credibility_results

        except Exception as e:
            self.logger.error(
                "CREDIBILITY_ASSESSMENT_ERROR", f"Failed to assess URLs: {str(e)}"
            )
            return {}

    def select_urls_for_visit(
        self,
        query: str,
        empty_fields: Set[str],
        visited_urls: Set[str],
        failed_urls: Set[str],
        max_urls: Optional[int] = None,
    ) -> List[URLInfo]:
        """Use LLM to select best URLs to visit next"""
        if max_urls is None:
            max_urls = self.config.max_urls_per_step

        # Get unvisited URLs
        unvisited_urls = []
        for url, entry in self.url_registry.items():
            if url not in visited_urls and url not in failed_urls:
                unvisited_urls.append(
                    {
                        "url": url,
                        "link_text": entry["link_text"],
                        "title": entry["info"].title,
                        "domain": entry["domain"],
                        "credibility": entry.get("credibility", "unknown"),
                        "snippet": entry["info"].metadata.get("snippet", ""),
                    }
                )

        if not unvisited_urls:
            return []

        # Prepare prompt for URL selection
        prompt = f"""Select the best URLs to visit for researching: {query}

Empty fields to fill: {list(empty_fields)}

Available URLs (showing first 30):
{json.dumps(unvisited_urls[:30], indent=2)}

Select up to {max_urls} URLs that are most likely to contain information for the empty fields.
Prioritize:
1. High credibility sources
2. URLs with relevant titles/link text
3. Diverse domains (don't select too many from same site)
4. Official or primary sources over secondary

Respond in JSON format:
{{
    "selected_urls": ["url1", "url2", ...],
    "reasoning": "brief explanation of selection"
}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at selecting relevant research sources.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm_config.temperature,
                max_tokens=self.config.llm_config.max_tokens,
                extra_body={"guided_json": ResponseOfSelection.model_json_schema()},
            )

            result = json.loads(response.choices[0].message.content)
            selected_urls = result.get("selected_urls", [])

            # Convert to URLInfo objects
            selected_url_infos = []
            for url in selected_urls:
                if url in self.url_registry:
                    selected_url_infos.append(self.url_registry[url]["info"])

            self.logger.info(
                "URL_SELECTION",
                f"Selected {len(selected_url_infos)} URLs to visit",
                reasoning=result.get("reasoning", ""),
            )

            return selected_url_infos

        except Exception as e:
            self.logger.error("URL_SELECTION_ERROR", f"Failed to select URLs: {str(e)}")
            # Fallback to simple selection
            fallback_urls = []
            for url, entry in list(self.url_registry.items())[:max_urls]:
                if url not in visited_urls and url not in failed_urls:
                    fallback_urls.append(entry["info"])
            return fallback_urls

    def update_url_info(self, url: str, **kwargs):
        """Update information about a URL"""
        if url in self.url_registry:
            url_info = self.url_registry[url]["info"]
            for key, value in kwargs.items():
                if hasattr(url_info, key):
                    setattr(url_info, key, value)
                elif key in ["credibility", "credibility_reason"]:
                    self.url_registry[url][key] = value
                else:
                    url_info.metadata[key] = value

    def get_url_with_context(self, url: str) -> Dict[str, any]:
        """Get URL information including link text and credibility"""
        if url in self.url_registry:
            entry = self.url_registry[url]
            return {
                "url": url,
                "link_text": entry["link_text"],
                "title": entry["info"].title,
                "credibility": entry.get("credibility", "unknown"),
                "credibility_reason": entry.get("credibility_reason", ""),
                "domain": entry["domain"],
            }
        return None

    def get_statistics(self) -> Dict[str, any]:
        """Get URL management statistics"""
        credibility_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
        for entry in self.url_registry.values():
            cred = entry.get("credibility", "unknown")
            if cred in credibility_counts:
                credibility_counts[cred] += 1
            else:
                credibility_counts["unknown"] += 1

        return {
            "total_discovered": len(self.url_registry),
            "domains": dict(self.domain_counts),
            "visited": sum(
                1
                for entry in self.url_registry.values()
                if entry["info"].visit_count > 0
            ),
            "successful": sum(
                1
                for entry in self.url_registry.values()
                if entry["info"].extraction_success
            ),
            "credibility_distribution": credibility_counts,
            "urls_with_link_text": sum(
                1 for entry in self.url_registry.values() if entry["link_text"]
            ),
        }
