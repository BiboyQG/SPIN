from typing import List, Dict, Set, Optional, Any
from collections import defaultdict
from urllib.parse import urlparse
from datetime import datetime
from openai import OpenAI
import json
import re

from core.response_model import ResponseOfSelection
from core.data_structures import URLInfo, SearchResult
from core.logging_config import get_logger
from core.config import get_config


class URLManager:
    """Manages URL discovery, filtering, and LLM-based URL selection"""

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
        #         "domain": "example.com"
        #     }
        # }

        self.domain_counts: Dict[str, int] = defaultdict(int)

    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        return parsed.netloc.lower()

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible"""
        try:
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
            url = result.url

            if not self.is_valid_url(url):
                continue

            # Create URLInfo object
            url_info = URLInfo(
                url=url,
                title=result.title,
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
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        matches = re.findall(link_pattern, content, re.IGNORECASE)

        for link_text, url in matches:
            # Clean up link text
            link_text = re.sub(r"\s+", " ", link_text.strip())

            if not self.is_valid_url(url):
                continue

            if url not in self.url_registry:
                url_info = URLInfo(
                    url=url,
                    title=link_text,  # Use link text as title
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
                }
                discovered_urls.append(url_info)
                self.domain_counts[self.extract_domain(url)] += 1

        # Also find plain URLs without link text
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        plain_urls = re.findall(url_pattern, content)

        for plain_url in plain_urls:
            url = plain_url
            if not self.is_valid_url(url) or url in self.url_registry:
                continue

            url_info = URLInfo(
                url=url,
                title="",
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
            }
            discovered_urls.append(url_info)
            self.domain_counts[self.extract_domain(url)] += 1

        return discovered_urls

    def select_urls_for_visit(
        self,
        query: str,
        empty_fields: Set[str],
        visited_urls: Set[str],
        failed_urls: Set[str],
        max_urls: Optional[int] = None,
        current_extraction: Optional[Dict[str, Any]] = None,
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
                        "snippet": entry["info"].metadata.get("snippet", ""),
                    }
                )

        if not unvisited_urls:
            return []

        # Prepare prompt for URL selection
        prompt = f"""Select the best URLs to visit for researching: {query}

Current information that we have for the entity: {current_extraction}

Empty fields to fill: {list(empty_fields)}

Available URLs (showing first 30):
{json.dumps(unvisited_urls[:30], indent=2)}

Select up to {max_urls} URLs that are most likely to contain information for the empty fields.
Prioritize: URLs with **relevant URL address or titles or link text or snippet**.

If there are no URLs that are likely to contain information for the empty fields, return an empty list and explain why.

Respond in JSON format:
{{
    "selected_urls": ["url1", "url2", ...],
    "reasoning": "brief explanation of selection"
}}"""

        if self.config.llm_config.enable_reasoning:
            prompt += "\n\nPlease reason and think about the given context and instructions before answering the question in JSON format."

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

            try:
                result = json.loads(response.choices[0].message.content)
            except Exception as e:
                try:
                    result = json.loads(response.choices[0].message.reasoning_content)
                except Exception as e:
                    self.logger.error(
                        "LLM_URL_SELECTION_ERROR", f"Failed to select URLs: {str(e)}"
                    )
                    raise ValueError("URLs could not be selected")

            selected_urls = result.get("selected_urls", [])

            if not selected_urls:
                self.logger.info(
                    "URL_SELECTION",
                    "No URLs selected for visit",
                    reasoning=result.get("reasoning", ""),
                )
                return []

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
                else:
                    url_info.metadata[key] = value
