from typing import Dict, Any, List
from datetime import datetime
import time

from core.response_model import ResponseOfWorthVisiting, ResponseOfKnowledgeExtraction
from core.knowledge_accumulator import KnowledgeAccumulator
from core.actions.base import ActionExecutor
from core.url_manager import URLManager
from core.data_structures import (
    ResearchContext,
    ResearchAction,
    KnowledgeItem,
    KnowledgeType,
    URLInfo,
)
from scraper import WebScraper
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

                        self.num_considered_urls += 1

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
        """Extract knowledge items from webpage content using LLM-based field-specific extraction"""
        knowledge_items = []

        # Get empty fields to focus extraction
        empty_fields = list(context.empty_fields)
        if not empty_fields:
            return knowledge_items

        # Prepare prompt for LLM
        prompt = f"""You are an expert research assistant extracting information from a webpage.

Research Query: {context.original_query}

Entity Type: {context.entity_type}

Webpage Content (Markdown):
{content}

Current information that we have for the entity: {context.current_extraction}

Empty fields to fill: {empty_fields}

Extract relevant information for each empty field from the provided content.
Return your findings as a JSON object following this schema:
{ResponseOfKnowledgeExtraction.model_json_schema()}

Instructions:
- For each field, if you find relevant information, provide the field name and the extracted value.
- If no relevant information is found for a field, do not include it in the output.
- Ensure the extracted value is detailed and directly answers the implicit question for that field.
"""

        if self.config.llm_config.enable_reasoning:
            prompt += "\n\nPlease reason and think about the given context and instructions before answering the question in JSON format."

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_config.extraction_model,  # Assuming you have an extraction_model in your config
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research assistant that extracts information from text.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm_config.temperature,
                max_tokens=self.config.llm_config.max_tokens_extraction,  # Assuming a specific max_tokens for extraction
                extra_body={
                    "guided_json": ResponseOfKnowledgeExtraction.model_json_schema()
                },
            )

            try:
                llm_response = ResponseOfKnowledgeExtraction.model_validate_json(
                    response.choices[0].message.content
                )
            except Exception as e:
                try:
                    llm_response = ResponseOfKnowledgeExtraction.model_validate_json(
                        response.choices[0].message.reasoning_content
                    )
                except Exception as e:
                    self.logger.warning(
                        "KNOWLEDGE_EXTRACTION_PARSE_ERROR",
                        f"Failed to parse LLM response for knowledge extraction from {url}: {e}",
                    )
                    return knowledge_items  # Return empty if parsing fails

            for item in llm_response.extracted_items:
                knowledge_item = KnowledgeItem(
                    question=f"What is the {item.field_name} of {context.original_query}?",
                    answer=item.extracted_value,
                    source_urls=[url],
                    timestamp=datetime.now(),
                    item_type=KnowledgeType.EXTRACTION,
                    schema_fields=[item.field_name],
                    metadata={
                        "extraction_method": "llm_field_specific",
                        "url": url,
                        "reasoning": item.reasoning,
                    },
                )
                knowledge_items.append(knowledge_item)

        except Exception as e:
            self.logger.error(
                "KNOWLEDGE_EXTRACTION_ERROR",
                f"Error during LLM-based knowledge extraction from {url}: {e}",
            )

        return knowledge_items

    def _is_worth_visiting(self, url_info: URLInfo, context: ResearchContext) -> bool:
        """Use LLM to check if a URL is worth visiting"""
        try:
            # Create prompt for LLM to evaluate URL worthiness
            prompt = f"""You are an expert research assistant evaluating whether a URL is worth visiting for research purposes.

Research Query: {context.original_query}

Current information that we have for the entity: {context.current_extraction}

Empty fields to fill: {list(context.empty_fields)}

URL to Evaluate:
- URL: {url_info.url}
- Title: {url_info.title or "Unknown"}
- Link Text: {url_info.link_text or "Unknown"}
- Domain: {url_info.domain or "Unknown"}
- Snippet: {url_info.snippet or "Unknown"}

Determine if this URL is worth visiting based on:
1. URL or Title or Link Text or Snippet relevance to the entity fields in the entity schema
2. Potential to fill empty fields in the entity

Return your decision as a JSON object with 'worth_visiting' (boolean) and 'reason' (string explaining your decision)."""

            if self.config.llm_config.enable_reasoning:
                prompt += "\n\nPlease reason and think about the given context and instructions before answering the question in JSON format."

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

            try:
                result = ResponseOfWorthVisiting.model_validate_json(
                    response.choices[0].message.content
                )
            except Exception as e:
                # Fallback to reasoning content if main content fails
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
                    raise ValueError("URL evaluation could not be determined")

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
