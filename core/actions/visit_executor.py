from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from core.actions.base import ActionExecutor
from core.data_structures import (
    ResearchContext,
    ResearchAction,
    KnowledgeItem,
    KnowledgeType,
)
from core.url_manager import URLManager
from core.knowledge_accumulator import KnowledgeAccumulator
from core.logging_config import ExtractionError
from scraper import WebScraper


class VisitExecutor(ActionExecutor):
    """Executes VISIT actions to extract content from URLs"""

    def __init__(
        self,
        url_manager: URLManager,
        knowledge_accumulator: KnowledgeAccumulator,
        web_scraper: WebScraper,
    ):
        super().__init__()
        self.url_manager = url_manager
        self.knowledge_accumulator = knowledge_accumulator
        self.web_scraper = web_scraper

    def execute(
        self, action: ResearchAction, context: ResearchContext
    ) -> Dict[str, Any]:
        """Execute a visit action"""
        self.pre_execute(action, context)

        try:
            urls = action.parameters.get("urls", [])
            if not urls:
                # Get URLs from URL manager
                all_urls = list(context.discovered_urls.values())
                ranked_urls = self.url_manager.rank_urls(
                    all_urls,
                    context.original_query,
                    context.empty_fields,
                    context.visited_urls,
                )
                urls = [u.url for u in self.url_manager.get_url_batch(ranked_urls)]

            # Visit each URL
            successful_visits = 0
            extracted_knowledge = []

            for url in urls:
                if url in context.visited_urls:
                    continue

                try:
                    # Scrape the URL
                    scrape_result = self.web_scraper.scrape_url(url)
                    content = scrape_result.get("markdown")

                    if not content:
                        self.logger.warning(
                            "VISIT_EMPTY_CONTENT", f"No content extracted from {url}"
                        )
                        context.failed_urls.add(url)
                        continue

                    # Mark as visited
                    context.visited_urls.add(url)

                    # Update URL info
                    self.url_manager.update_url_info(
                        url,
                        content=content,
                        last_visited=datetime.now(),
                        visit_count=1,
                        extraction_success=True,
                    )

                    # Extract knowledge from content
                    knowledge_items = self._extract_knowledge_from_content(
                        url, content, context
                    )

                    for item in knowledge_items:
                        self.knowledge_accumulator.add_knowledge(item)
                        extracted_knowledge.append(item)

                    # Discover new URLs from content
                    new_urls = self.url_manager.discover_urls_from_content(url, content)

                    successful_visits += 1

                    # Rate limiting
                    time.sleep(self.config.step_delay)

                except Exception as e:
                    self.logger.error(
                        "VISIT_URL_FAILED", f"Failed to visit {url}", error=str(e)
                    )
                    context.failed_urls.add(url)

            result = {
                "success": successful_visits > 0,
                "urls_visited": successful_visits,
                "urls_failed": len(urls) - successful_visits,
                "knowledge_extracted": len(extracted_knowledge),
                "items_processed": successful_visits,
            }

            self.post_execute(action, context, result)
            return result

        except Exception as e:
            self.handle_error(action, context, e)
            return {"success": False, "error": str(e), "items_processed": 0}

    def _extract_knowledge_from_content(
        self, url: str, content: str, context: ResearchContext
    ) -> List[KnowledgeItem]:
        """Extract knowledge items from webpage content"""
        knowledge_items = []

        # Split content into chunks for analysis
        chunks = self._split_content_into_chunks(content)

        for chunk in chunks:
            # Identify which fields this chunk might relate to
            related_fields = self._identify_content_fields(chunk, context)

            if related_fields:
                # Create knowledge item
                knowledge_item = KnowledgeItem(
                    question=f"What information about {context.original_query} is found on {url}?",
                    answer=chunk,
                    source_urls=[url],
                    confidence=0.7,  # Base confidence for extracted content
                    timestamp=datetime.now(),
                    item_type=KnowledgeType.EXTRACTION,
                    schema_fields=related_fields,
                    metadata={
                        "extraction_method": "content_chunking",
                        "chunk_size": len(chunk),
                    },
                )
                knowledge_items.append(knowledge_item)

        return knowledge_items

    def _split_content_into_chunks(
        self, content: str, max_chunk_size: int = 1000
    ) -> List[str]:
        """Split content into manageable chunks"""
        # Split by paragraphs first
        paragraphs = content.split("\n\n")

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _identify_content_fields(
        self, content: str, context: ResearchContext
    ) -> List[str]:
        """Identify which schema fields content relates to"""
        related_fields = []
        content_lower = content.lower()

        # Check for entity name in content
        entity_name_parts = context.original_query.lower().split()
        if not any(part in content_lower for part in entity_name_parts):
            # Content doesn't mention the entity
            return []

        # Check each schema field
        for field in context.empty_fields:
            field_keywords = self._get_field_keywords(field)

            # Check if any keywords appear in content
            if any(keyword in content_lower for keyword in field_keywords):
                related_fields.append(field)

        return related_fields

    def _get_field_keywords(self, field_name: str) -> List[str]:
        """Get keywords associated with a field"""
        # Convert field name to keywords
        keywords = [field_name.replace("_", " ").lower()]

        # Add common variations
        field_mappings = {
            "email": ["email", "e-mail", "contact", "@"],
            "phone": ["phone", "telephone", "tel", "mobile", "call"],
            "address": ["address", "location", "office", "building"],
            "education": [
                "education",
                "degree",
                "university",
                "college",
                "phd",
                "masters",
            ],
            "experience": ["experience", "work", "position", "job", "career"],
            "research": ["research", "publication", "paper", "study", "project"],
            "bio": ["biography", "bio", "about", "background"],
            "department": ["department", "dept", "division", "school"],
            "title": ["title", "position", "role", "professor", "associate"],
        }

        # Add mapped keywords if available
        for key, values in field_mappings.items():
            if key in field_name.lower():
                keywords.extend(values)

        return list(set(keywords))
