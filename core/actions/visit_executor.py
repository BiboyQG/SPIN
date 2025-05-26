from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from core.actions.base import ActionExecutor
from core.data_structures import (
    ResearchContext,
    ResearchAction,
    KnowledgeItem,
    KnowledgeType,
    URLInfo,
)
from core.url_manager import URLManager
from core.knowledge_accumulator import KnowledgeAccumulator
from core.logging_config import ExtractionError
from scraper import WebScraper
from core.response_model import ResponseOfWorthVisiting
from openai import OpenAI
from tqdm import tqdm


class VisitExecutor(ActionExecutor):
    """Executes VISIT actions to extract content from URLs"""

    def __init__(
        self,
        url_manager: URLManager,
        knowledge_accumulator: KnowledgeAccumulator,
        web_scraper: WebScraper,
        llm_client: OpenAI,
    ):
        super().__init__()
        self.url_manager = url_manager
        self.knowledge_accumulator = knowledge_accumulator
        self.web_scraper = web_scraper
        self.llm_client = llm_client
        self.max_considered_urls = 100
        self.num_considered_urls = 0

    def execute(
        self, action: ResearchAction, context: ResearchContext
    ) -> Dict[str, Any]:
        """Execute a visit action"""
        self.pre_execute(action, context)

        try:
            urls = action.parameters.get("urls", [])
            if not urls:
                # Use the new LLM-based URL selection method
                selected_url_infos = self.url_manager.select_urls_for_visit(
                    query=context.original_query,
                    empty_fields=context.empty_fields,
                    visited_urls=context.visited_urls,
                    failed_urls=context.failed_urls,
                    max_urls=context.max_urls_per_step,
                )
                urls = [url_info.url for url_info in selected_url_infos]

            if not urls:
                self.logger.warning("VISIT_NO_URLS", "No URLs available to visit")
                return {
                    "success": False,
                    "urls_visited": 0,
                    "urls_failed": 0,
                    "knowledge_extracted": 0,
                    "items_processed": 0,
                }

            # Visit each URL
            successful_visits = 0
            extracted_knowledge = []

            self.num_considered_urls += len(urls)

            for url in tqdm(urls, desc="Visiting URLs"):
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
                        context.discovered_urls.pop(url)
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

                    # Extract knowledge from content using LLM-based approach
                    knowledge_items = self._extract_knowledge_from_content(
                        url, content, context
                    )

                    for item in knowledge_items:
                        self.knowledge_accumulator.add_knowledge(item, context)
                        extracted_knowledge.append(item)

                    # Discover new URLs from content
                    new_urls = self.url_manager.discover_urls_from_content(url, content)

                    self.num_considered_urls += len(new_urls)

                    # Filter new URLs using LLM evaluation and add to context
                    for url_info in tqdm(new_urls, desc="Evaluating New URLs"):
                        if self.num_considered_urls > self.max_considered_urls:
                            self.logger.debug(
                                "MAX_CONSIDERED_URLS_REACHED",
                                f"Max considered URLs reached: {self.max_considered_urls}",
                            )
                            break

                        # Skip if URL is already visited or failed
                        if (
                            url_info.url in context.visited_urls
                            or url_info.url in context.failed_urls
                        ):
                            continue

                        # Use LLM to evaluate if URL is worth visiting
                        if self._is_worth_visiting(url_info, context):
                            context.discovered_urls[url_info.url] = url_info
                            self.logger.debug(
                                "URL_ADDED_TO_QUEUE",
                                f"Added URL to discovery queue: {url_info.url}",
                            )
                        else:
                            self.logger.debug(
                                "URL_FILTERED_OUT", f"Filtered out URL: {url_info.url}"
                            )

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
        """Extract knowledge items from webpage content using improved field-specific extraction"""
        knowledge_items = []

        # Get empty fields to focus extraction
        empty_fields = list(context.empty_fields)
        if not empty_fields:
            return knowledge_items

        # Split content into meaningful sections
        sections = self._split_content_into_sections(content)

        for section in sections:
            # Check if section is relevant to the entity
            if not self._is_section_relevant(section, context.original_query):
                continue

            # For each empty field, try to extract relevant information
            for field in empty_fields:
                field_info = self._extract_field_specific_info(section, field, context)

                if field_info:
                    knowledge_item = KnowledgeItem(
                        question=f"What is the {field} of {context.original_query}?",
                        answer=field_info,
                        source_urls=[url],
                        confidence=0.8,  # Higher confidence for targeted extraction
                        timestamp=datetime.now(),
                        item_type=KnowledgeType.EXTRACTION,
                        schema_fields=[field],
                        metadata={
                            "extraction_method": "field_specific",
                            "section_length": len(section),
                            "url": url,
                        },
                    )
                    knowledge_items.append(knowledge_item)

        return knowledge_items

    def _split_content_into_sections(self, content: str) -> List[str]:
        """Split content into meaningful sections"""
        # Split by headers and major breaks
        sections = []

        # Split by markdown headers
        lines = content.split("\n")
        current_section = []

        for line in lines:
            # Check for section breaks (headers, horizontal rules, etc.)
            if (
                line.startswith("#")
                or line.startswith("---")
                or line.startswith("===")
                or (
                    len(current_section) > 0
                    and len(line.strip()) == 0
                    and len(current_section) > 10
                )
            ):
                if current_section:
                    section_text = "\n".join(current_section).strip()
                    if len(section_text) > 50:  # Only keep substantial sections
                        sections.append(section_text)
                    current_section = []

            current_section.append(line)

        # Add the last section
        if current_section:
            section_text = "\n".join(current_section).strip()
            if len(section_text) > 50:
                sections.append(section_text)

        # If no clear sections found, split by paragraphs
        if not sections:
            paragraphs = content.split("\n\n")
            sections = [p.strip() for p in paragraphs if len(p.strip()) > 100]

        return sections

    def _is_section_relevant(self, section: str, entity_query: str) -> bool:
        """Check if a section is relevant to the entity being researched"""
        section_lower = section.lower()

        # Check if entity name appears in section
        entity_parts = entity_query.lower().split()
        entity_mentions = sum(1 for part in entity_parts if part in section_lower)

        # Require at least half of entity name parts to be mentioned
        return entity_mentions >= len(entity_parts) / 2

    def _extract_field_specific_info(
        self, section: str, field: str, context: ResearchContext
    ) -> Optional[str]:
        """Extract information specific to a field from a section"""
        section_lower = section.lower()
        field_lower = field.lower()

        # Get field-specific patterns and keywords
        field_patterns = self._get_field_extraction_patterns(field)

        # Check if any field keywords appear in the section
        field_mentioned = any(
            keyword in section_lower for keyword in field_patterns["keywords"]
        )

        if not field_mentioned:
            return None

        # Extract relevant sentences/phrases
        sentences = section.split(".")
        relevant_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(keyword in sentence_lower for keyword in field_patterns["keywords"]):
                # Clean up the sentence
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 10:  # Avoid very short fragments
                    relevant_sentences.append(clean_sentence)

        if relevant_sentences:
            # Join relevant sentences, limiting length
            result = ". ".join(relevant_sentences[:3])  # Max 3 sentences
            return result[:2000] if len(result) > 2000 else result  # Limit length

        return None

    def _is_worth_visiting(self, url_info: URLInfo, context: ResearchContext) -> bool:
        """Use LLM to check if a URL is worth visiting"""
        try:
            # Create prompt for LLM to evaluate URL worthiness
            prompt = f"""You are an expert research assistant evaluating whether a URL is worth visiting for research purposes.

Research Query: {context.original_query}
Empty Fields: {", ".join(context.empty_fields) if context.empty_fields else "None"}
Already Visited URLs: {len(context.visited_urls)} URLs
Failed URLs: {len(context.failed_urls)} URLs

URL to Evaluate:
- URL: {url_info.url}
- Title: {url_info.title or "Unknown"}

Determine if this URL is worth visiting based on:
1. Relevance to the research query
2. Potential to fill empty fields
3. URL quality and credibility
4. Whether it's likely to contain useful information

Return your decision as a JSON object with 'worth_visiting' (boolean) and 'reason' (string explaining your decision)."""

            response = self.llm_client.chat.completions.create(
                model=self.config.llm_config.planning_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research assistant that evaluates URL worthiness for research tasks.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm_config.temperature,
                max_tokens=self.config.llm_config.max_tokens,
                extra_body={"guided_json": ResponseOfWorthVisiting.model_json_schema()},
            )

            if self.config.llm_config.enable_reasoning:
                reasoning_content = response.choices[0].message.reasoning_content
                self.logger.debug("URL_EVALUATION_REASONING", reasoning_content)

            try:
                result = ResponseOfWorthVisiting.model_validate_json(
                    response.choices[0].message.content
                )
            except Exception as e:
                # Fallback to reasoning content if main content fails
                if self.config.llm_config.enable_reasoning:
                    try:
                        result = ResponseOfWorthVisiting.model_validate_json(
                            response.choices[0].message.reasoning_content
                        )
                    except Exception as e:
                        # Default to visiting if parsing fails
                        self.logger.warning(
                            "URL_EVALUATION_PARSE_ERROR_FALLBACK",
                            f"Failed to parse LLM response for URL {url_info.url}: {e}",
                        )
                        return True
                else:
                    return True

            self.logger.debug(
                "URL_EVALUATION_RESULT",
                f"URL {url_info.url} evaluation: {result.worth_visiting} - {result.reason}",
            )

            return result.worth_visiting

        except Exception as e:
            self.logger.error(
                "URL_EVALUATION_ERROR", f"Error evaluating URL {url_info.url}: {e}"
            )
            # Default to visiting if evaluation fails
            return False

    def _get_field_extraction_patterns(self, field_name: str) -> Dict[str, List[str]]:
        """Get extraction patterns for specific fields"""
        patterns = {
            "keywords": [field_name.replace("_", " ").lower()],
            "indicators": [],
        }

        # Field-specific keyword mappings
        field_mappings = {
            "name": ["name", "called", "known as", "title"],
            "email": ["email", "e-mail", "contact", "@", "reach"],
            "phone": ["phone", "telephone", "tel", "mobile", "call", "contact"],
            "address": ["address", "location", "office", "building", "street", "city"],
            "education": [
                "education",
                "degree",
                "university",
                "college",
                "phd",
                "masters",
                "bachelor",
                "graduated",
            ],
            "experience": [
                "experience",
                "work",
                "position",
                "job",
                "career",
                "employed",
                "worked",
            ],
            "research": [
                "research",
                "publication",
                "paper",
                "study",
                "project",
                "published",
            ],
            "bio": ["biography", "bio", "about", "background", "profile"],
            "biography": ["biography", "bio", "about", "background", "profile", "life"],
            "department": ["department", "dept", "division", "school", "faculty"],
            "title": [
                "title",
                "position",
                "role",
                "professor",
                "associate",
                "director",
            ],
            "founded": ["founded", "established", "started", "began", "inception"],
            "headquarters": ["headquarters", "hq", "based", "located", "office"],
            "products": ["products", "services", "offers", "develops", "creates"],
            "founders": ["founder", "founded by", "established by", "created by"],
        }

        # Add mapped keywords
        for key, values in field_mappings.items():
            if key in field_name.lower():
                patterns["keywords"].extend(values)

        # Remove duplicates
        patterns["keywords"] = list(set(patterns["keywords"]))

        return patterns
